from __future__ import annotations

from itertools import combinations
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import torch

from ..backends import (
    DEFAULT_BICOP_BACKEND,
    build_marginal_estimator,
    normalize_bicop_backend,
    normalize_marginal_backend,
)
from ..bicop import BiCop
from ..util import ENUM_FUNC_BIDEP, kendall_tau, kendall_tau_matrix
from .artifact import VineBuildArtifact

__all__ = ["VineBuilder", "VineCopBuilderMixin"]

if TYPE_CHECKING:
    from .model import VineCop


class VineBuilder:
    def __init__(self, model: VineCop) -> None:
        self.model = model

    def build(self, obs: torch.Tensor, **kwargs) -> VineBuildArtifact:
        return self.model._fit_impl(obs=obs, **kwargs)


class VineCopBuilderMixin:
    def _normalize_fit_request(
        self,
        *,
        marginal_backend: str,
        marginal_kwargs: dict | None,
        bicop_backend: str | None,
        bicop_kwargs: dict | None,
        num_iter_max: int,
        bandwidth: str | float | torch.Tensor,
    ) -> tuple[str, dict, str, dict]:
        marginal_backend = normalize_marginal_backend(marginal_backend)
        normalized_marginal_kwargs = dict(marginal_kwargs or {})
        if marginal_backend == "grid":
            normalized_marginal_kwargs.setdefault("bandwidth", bandwidth)
            normalized_marginal_kwargs.setdefault("num_step_grid", self.num_step_grid)

        bicop_backend = normalize_bicop_backend(bicop_backend or DEFAULT_BICOP_BACKEND)
        aligned_bicop_backends = {"ttcv", "ttpi", "tll1", "tll2", "tll1nn", "tll2nn", "beta"}
        normalized_bicop_kwargs = dict(bicop_kwargs or {})
        if bicop_backend == "tll_ref":
            normalized_bicop_kwargs.setdefault("nonparametric_method", "constant")
        else:
            default_bicop_bw = (
                "auto"
                if bicop_backend in aligned_bicop_backends
                and isinstance(bandwidth, str)
                and bandwidth in {"silverman", "isj"}
                else bandwidth
            )
            if "bandwidth" not in normalized_bicop_kwargs and default_bicop_bw != "isj":
                normalized_bicop_kwargs["bandwidth"] = default_bicop_bw
            normalized_bicop_kwargs.setdefault("num_iter_max", num_iter_max)
            if bicop_backend in {"ttcv", "ttpi"}:
                bw_val = normalized_bicop_kwargs.get("bandwidth", "auto")
                if isinstance(bw_val, str):
                    if bw_val in {"silverman", "isj"}:
                        normalized_bicop_kwargs["bandwidth"] = "auto"
                elif torch.as_tensor(bw_val).numel() != 4:
                    normalized_bicop_kwargs["bandwidth"] = "auto"
            elif bicop_backend in {"tll1", "tll2", "beta"}:
                bw_val = normalized_bicop_kwargs.get("bandwidth", "auto")
                if isinstance(bw_val, str) and bw_val in {"silverman", "isj"}:
                    normalized_bicop_kwargs["bandwidth"] = "auto"
            elif bicop_backend in {"tll1nn", "tll2nn"}:
                bw_val = normalized_bicop_kwargs.get("bandwidth", "auto")
                if isinstance(bw_val, str) and bw_val in {"silverman", "isj"}:
                    normalized_bicop_kwargs["bandwidth"] = "auto"
        return marginal_backend, normalized_marginal_kwargs, bicop_backend, normalized_bicop_kwargs

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        is_dissmann: bool = True,
        matrix: torch.Tensor = None,
        first_tree_vertex: tuple = tuple(),
        mtd_vine: str = "rvine",
        mtd_bidep: str = "chatterjee_xi",
        thresh_trunc: None | float = 0.01,
        num_iter_max: int = 5,
        is_tau_est: bool = False,
        marginal_backend: str = "grid",
        marginal_kwargs: dict | None = None,
        bicop_backend: str | None = None,
        bicop_kwargs: dict | None = None,
        bandwidth: str | float | torch.Tensor = "isj",
        generator: torch.Generator | None = None,
        bidep_backend: Literal["auto", "scipy", "torch"] = "auto",
    ) -> None:
        artifact = VineBuilder(self).build(
            obs=obs,
            is_dissmann=is_dissmann,
            matrix=matrix,
            first_tree_vertex=first_tree_vertex,
            mtd_vine=mtd_vine,
            mtd_bidep=mtd_bidep,
            thresh_trunc=thresh_trunc,
            num_iter_max=num_iter_max,
            is_tau_est=is_tau_est,
            marginal_backend=marginal_backend,
            marginal_kwargs=marginal_kwargs,
            bicop_backend=bicop_backend,
            bicop_kwargs=bicop_kwargs,
            bandwidth=bandwidth,
            generator=generator,
            bidep_backend=bidep_backend,
        )
        self._artifact = artifact
        self._refresh_engine(artifact=artifact)

    @torch.no_grad()
    def _fit_impl(
        self,
        obs: torch.Tensor,
        is_dissmann: bool = True,
        matrix: torch.Tensor = None,
        first_tree_vertex: tuple = tuple(),
        mtd_vine: str = "rvine",
        mtd_bidep: str = "chatterjee_xi",
        thresh_trunc: None | float = 0.01,
        num_iter_max: int = 5,
        is_tau_est: bool = False,
        marginal_backend: str = "grid",
        marginal_kwargs: dict | None = None,
        bicop_backend: str | None = None,
        bicop_kwargs: dict | None = None,
        bandwidth: str | float | torch.Tensor = "isj",
        generator: torch.Generator | None = None,
        bidep_backend: Literal["auto", "scipy", "torch"] = "auto",
    ) -> VineBuildArtifact:
        if bidep_backend not in {"auto", "scipy", "torch"}:
            raise ValueError("bidep_backend must be one of 'auto', 'scipy', 'torch'.")
        self.reset()
        device, dtype = self.device, self.dtype
        (
            marginal_backend,
            normalized_marginal_kwargs,
            bicop_backend,
            normalized_bicop_kwargs,
        ) = self._normalize_fit_request(
            marginal_backend=marginal_backend,
            marginal_kwargs=marginal_kwargs,
            bicop_backend=bicop_backend,
            bicop_kwargs=bicop_kwargs,
            num_iter_max=num_iter_max,
            bandwidth=bandwidth,
        )
        self.marginal_backend = marginal_backend
        self.marginal_backend_config = dict(normalized_marginal_kwargs)
        self.bicop_backend = bicop_backend
        self.bicop_backend_config = dict(normalized_bicop_kwargs)
        if self.is_cop_scale:
            obs_mvcp = obs.to(device=device, dtype=dtype)
        else:
            for v in range(self.num_dim):
                self.marginals[v] = build_marginal_estimator(
                    backend_name=marginal_backend,
                    x=obs[:, v],
                    backend_kwargs=normalized_marginal_kwargs,
                ).to(device=device, dtype=dtype)
            obs_mvcp = torch.hstack(
                [self.marginals[v].cdf(obs[:, [v]]) for v in range(self.num_dim)]
            ).to(device=device, dtype=dtype)

        self.num_obs.copy_(obs_mvcp.shape[0])
        self.mtd_bidep = mtd_bidep
        is_kendall_tau = mtd_bidep == "kendall_tau"
        f_bidep = ENUM_FUNC_BIDEP[mtd_bidep].value
        self.first_tree_vertex = first_tree_vertex
        # * lv_0 obs, empty cond_ing
        dct_obs = [dict() for _ in range(self.num_dim)]
        dct_obs[0] = {
            # ! v_s: obs
            (idx,): obs_mvcp[:, [idx]]
            for idx in range(self.num_dim)
        }

        def _visit_hfunc(lv: int, v_s: tuple) -> None:
            """
            Lazy hfunc for pseudo-obs at this v_s.
            """
            lv_up = lv - 1
            cond_ed = self.struct_obs[lv][v_s]
            v_l, v_r = map(int, cond_ed.split(","))
            # * v_down (cond_ed) and s_down (cond_ing) of child pseudo obs
            # * s_up (cond_ing) of parent bicop
            s_up = self.struct_bcp[cond_ed]["cond_ing"]
            bcp = self.bicops[cond_ed]

            if bcp.is_indep:
                dct_obs[lv][v_s] = dct_obs[lv_up][v_s[0], *s_up]
            else:
                dct_obs[lv][v_s] = (bcp.hfunc_r if v_s[0] == v_l else bcp.hfunc_l)(
                    obs=torch.hstack(
                        [
                            dct_obs[lv_up][v_l, *s_up],
                            dct_obs[lv_up][v_r, *s_up],
                        ]
                    )
                )

        for lv in range(self.num_dim - 1):
            curr_v_s_parent = self.struct_obs[lv]
            if is_dissmann:
                # * obs2edge, list possible edges that connect two pseudo obs, calc f_bidep
                # ! proximity condition: only obs sharing 'cond_ing' can have edges
                # * group vertices by their conditioning set
                grp_s = defaultdict(list)
                for v_s in curr_v_s_parent:
                    v, *s = v_s
                    s = tuple(s)
                    grp_s[s].append(v)
                tmp_edge_weight = {}
                tmp_edge_pvalue = {}
                for s, lst_v in grp_s.items():
                    if len(lst_v) < 2:
                        continue
                    lst_v.sort()
                    # * ensure all pseudo-obs are ready
                    for v in lst_v:
                        if dct_obs[lv][(v, *s)] is None:
                            _visit_hfunc(lv=lv, v_s=(v, *s))
                    if is_kendall_tau:
                        obs_group = torch.hstack([dct_obs[lv][(v, *s)] for v in lst_v])
                        tau_mat, pvalue_mat = kendall_tau_matrix(
                            obs_group,
                            backend=bidep_backend,
                        )
                        for idx_l, idx_r in combinations(range(len(lst_v)), 2):
                            v_l, v_r = lst_v[idx_l], lst_v[idx_r]
                            tmp_edge_weight[(v_l, v_r, *s)] = tau_mat[idx_l, idx_r]
                            tmp_edge_pvalue[(v_l, v_r, *s)] = pvalue_mat[idx_l, idx_r]
                    else:
                        for v_l, v_r in combinations(lst_v, 2):
                            # ! already sorted !
                            w = f_bidep(
                                x=dct_obs[lv][v_l, *s],
                                y=dct_obs[lv][v_r, *s],
                            )
                            tmp_edge_weight[(v_l, v_r, *s)] = w
                if mtd_vine == "dvine":
                    # * edge2tree, dvine
                    if lv == 0:
                        tree = self._mst_from_edge_dvine(edge_weight_lv=tmp_edge_weight)
                    else:
                        tree = list(tmp_edge_weight)
                elif mtd_vine == "cvine":
                    # * edge2tree, cvine
                    tree = self._mst_from_edge_cvine(edge_weight_lv=tmp_edge_weight)
                elif mtd_vine == "rvine":
                    # * edge2tree, rvine
                    tree = self._mst_from_edge_rvine(edge_weight_lv=tmp_edge_weight)
                else:
                    raise ValueError("mtd_vine must be one of 'cvine', 'dvine', 'rvine'")
                # * store tree and weights
                self.tree_bidep[lv] = {lr_s: tmp_edge_weight[lr_s] for lr_s in tree}
            else:
                # * tree structure is inferred from matrix, if not Dissmann
                tree = []
                tmp_edge_pvalue = {}
                for idx in range(self.num_dim - lv - 1):
                    # ! sorted !
                    v_l, v_r = sorted(
                        (
                            int(matrix[idx][idx]),
                            int(matrix[idx][self.num_dim - lv - 1]),
                        )
                    )
                    s = sorted((int(_) for _ in matrix[idx][(self.num_dim - lv) :]))
                    tree.append((v_l, v_r, *s))
                    # * ensure all pseudo-obs are ready
                    for v in (v_l, v_r):
                        if dct_obs[lv][(v, *s)] is None:
                            _visit_hfunc(lv=lv, v_s=(v, *s))
                    if is_kendall_tau:
                        stat = kendall_tau(
                            dct_obs[lv][v_l, *s],
                            dct_obs[lv][v_r, *s],
                            backend=bidep_backend,
                        )
                        self.tree_bidep[lv][v_l, v_r, *s] = stat[0]
                        tmp_edge_pvalue[(v_l, v_r, *s)] = stat[1]
                    else:
                        w = f_bidep(
                            x=dct_obs[lv][v_l, *s],
                            y=dct_obs[lv][v_r, *s],
                        )
                        self.tree_bidep[lv][v_l, v_r, *s] = w

            # * tree2bicop, fit bicop & record key of potential pseudo obs for next lv (lazy hfunc later)
            for v_l, v_r, *s in tree:
                is_fitting = True
                s = tuple(s)
                cond_ed: str = f"{v_l},{v_r}"
                bcp: BiCop = self.bicops[cond_ed]
                # * fit/truncate bicop
                obs_bcp = torch.hstack(
                    [
                        dct_obs[lv][(v_l, *s)],
                        dct_obs[lv][(v_r, *s)],
                    ]
                )
                if thresh_trunc is not None:
                    pvalue = tmp_edge_pvalue.get((v_l, v_r, *s))
                    if pvalue is None:
                        pvalue = kendall_tau(
                            obs_bcp[:, [0]],
                            obs_bcp[:, [1]],
                            backend=bidep_backend,
                        )[1]
                    is_fitting = pvalue <= thresh_trunc
                if is_fitting:
                    bcp.fit(
                        obs=obs_bcp,
                        is_tau_est=is_tau_est,
                        bicop_backend=bicop_backend,
                        bicop_kwargs=normalized_bicop_kwargs,
                        generator=generator,
                    )
                    self.struct_bcp[cond_ed]["is_indep"] = False
                else:
                    bcp.num_obs.copy_(self.num_obs)
                lv_next = lv + 1
                next_v_s_parent = self.struct_obs[lv_next]
                tmp = (v_l, *sorted((*s, v_r)))
                next_v_s_parent[tmp] = cond_ed
                dct_obs[lv_next][tmp] = None
                tmp = (v_r, *sorted((*s, v_l)))
                next_v_s_parent[tmp] = cond_ed
                dct_obs[lv_next][tmp] = None
                # * update structure
                self.struct_bcp[cond_ed].update(
                    {
                        "cond_ing": s,
                        "left": curr_v_s_parent[v_l, *s],
                        "right": curr_v_s_parent[v_r, *s],
                    }
                )
            # ! garbage collection
            if lv > 0:
                dct_obs[lv - 1].clear()
        # * update sample order
        self._sample_order()
        artifact = self._export_artifact()
        self._artifact = artifact
        return artifact
