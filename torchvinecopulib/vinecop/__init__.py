import copy
import heapq
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from pprint import pformat
from textwrap import indent

import torch

from ..bicop import BiCop
from ..util import ENUM_FUNC_BIDEP, kdeCDFPPF1D

__all__ = [
    "VineCop",
]


class VineCop(torch.nn.Module):
    # ! hinv
    _EPS: float = 1e-7

    def __init__(
        self,
        num_dim: int,
        is_cop_scale: bool = False,
        num_step_grid: int = 128,
    ) -> None:
        super().__init__()
        self.num_dim = num_dim
        self.is_cop_scale = is_cop_scale
        self.marginals = torch.nn.ModuleList([None] * num_dim)
        self.bicops = torch.nn.ModuleDict()
        self.struct_bcp = {}
        self.struct_obs = [{} for _ in range(num_dim)]
        for i in range(num_dim):
            self.struct_obs[0][(i,)] = ""
        for i, j in combinations(range(num_dim), 2):
            # * num_bicop = num_dim * (num_dim - 1) // 2
            # NOTE i < j by itertools.combinations;
            # ! ModuleDict key must be str
            cond_ed = f"{i},{j}"
            self.bicops[cond_ed] = BiCop(num_step_grid=num_step_grid)
            self.struct_bcp[cond_ed] = dict(
                # * cond_ed, cond_ing (now empty) of a bicop
                cond_ed=(i, j),
                cond_ing=tuple(),
                is_indep=True,
                # ! left parent cond_ed str
                left=None,
                # ! right parent cond_ed str
                right=None,
            )
        self.mtd_bidep = None
        self.tree_bidep = [{} for _ in range(num_dim - 1)]
        self.num_step_grid = num_step_grid
        self.sample_order = tuple(_ for _ in range(num_dim))
        self.register_buffer("num_obs", torch.empty((), dtype=torch.int))
        # ! device agnostic
        self.register_buffer("_dd", torch.tensor([], dtype=torch.float64))

    @property
    def device(self):
        return self._dd.device

    @property
    def dtype(self):
        return self._dd.dtype

    @property
    def matrix(self) -> torch.Tensor:
        with torch.no_grad():
            mat: list[list[int]] = []
            seen: set[int] = set()
            # * iterate levels reversely; the diag‐variable is sample_order[idx]
            for idx, lv in enumerate(range(self.num_dim - 2, -1, -1)):
                v_diag = int(self.sample_order[idx])
                row: list[int] = [-1] * idx + [v_diag]
                # * look up tree-edges from level lv upwards
                for up in range(lv, -1, -1):
                    for v_l, v_r, *_ in self.tree_bidep[up]:
                        if (
                            v_diag in (v_l, v_r)
                            and (v_l not in seen)
                            and (v_r not in seen)
                        ):
                            # * append the "other" variable
                            row.append(v_l if (v_diag == v_r) else v_r)
                seen.add(v_diag)
                mat.append(row)
            #
            mat.append([-1] * (self.num_dim - 1) + [int(self.sample_order[-1])])
            return torch.tensor(mat, dtype=torch.int, device=self.device)

    @torch.no_grad()
    def reset(self) -> None:
        self.num_obs.zero_()
        for i in range(self.num_dim):
            self.struct_obs[0][(i,)] = ""
            if i > 0:
                self.struct_obs[i].clear()
        self.tree_bidep = [{} for _ in range(self.num_dim - 1)]
        for _, dd in self.struct_bcp.items():
            dd["cond_ing"] = tuple()
            dd["is_indep"] = True
            dd["left"] = None
            dd["right"] = None
        for bicop in self.bicops.values():
            bicop.reset()

    @staticmethod
    @torch.no_grad()
    def ref_count_hfunc(num_dim: int, struct_obs: list, sample_order: tuple):
        missing = set(range(num_dim)) - set(sample_order)
        lst_source = []
        for idx, v in enumerate(sample_order):
            s = set(sample_order[idx + 1 :]) | missing
            lst_source.append((v, *sorted(s)))
        for v in missing:
            lst_source.append((v,))
        lst_source.reverse()
        ref_cnt = Counter()
        num_hfunc = 0

        def _visit(v_s: tuple, is_hinv: bool = False):
            nonlocal num_hfunc
            if len(v_s) == 1:
                ref_cnt[v_s] += 1
                return
            v_down, *s_down = v_s
            # * locate parent bicop
            v_l, v_r = map(int, struct_obs[len(s_down)][v_s].split(","))
            s_up = tuple(sorted(set(s_down) - {v_l, v_r}))
            # ! oppo parent if is_hinv else both parents
            if is_hinv:
                frontier = [(v_l if v_down == v_r else v_r, *s_up)]
            else:
                frontier = [(v_l, *s_up), (v_r, *s_up)]
            for v_s in frontier:
                # * recurse on any missing parents
                if ref_cnt[v_s] == 0:
                    _visit(v_s=v_s, is_hinv=False)
                    num_hfunc += 1
            # * increment reference counts for all three vertices
            frontier = [(v_l, *s_up), (v_r, *s_up), (v_down, *s_down)]
            for v_s in frontier:
                ref_cnt[v_s] += 1
            # * next if climbing up
            if is_hinv:
                return (v_down, *s_up)

        # * climb each source until cond-ing set empty
        for v_s in lst_source:
            while len(v_s) > 1:
                v_s = _visit(v_s=v_s, is_hinv=True)
        return dict(ref_cnt), lst_source, num_hfunc

    @torch.no_grad()
    def _sample_order(self) -> tuple[int, ...]:
        last_tree_vertex = set(range(self.num_dim)) - set(self.first_tree_vertex)
        sample_order = []
        for v_s_parent in self.struct_obs[::-1]:
            cost_best = float("inf")
            cand_v = set()
            for v_s, cond_ed in v_s_parent.items():
                if cond_ed:
                    # * not yet top lv
                    v_l, v_r = map(int, cond_ed.split(","))
                    if v_l not in sample_order and v_r not in sample_order:
                        cand_v.add(v_l)
                        cand_v.add(v_r)
                elif v_s[0] not in sample_order:
                    # * top lv, only one choice
                    cand_v.add(v_s[0])
            # ! prioritize those not in first_tree_vertex
            cand_v_last = cand_v & last_tree_vertex
            if cand_v_last:
                cand_v = cand_v_last
            for v in sorted(cand_v):
                _, _, cost = self.ref_count_hfunc(
                    num_dim=self.num_dim,
                    struct_obs=self.struct_obs,
                    sample_order=sample_order + [v],
                )
                if cost < cost_best:
                    cost_best = cost
                    v_best = v
            sample_order.append(v_best)
        self.sample_order = tuple(sample_order)

    @torch.no_grad()
    def _mst_from_edge_dvine(self, edge_weight_lv: dict) -> None:
        # * edge2tree, dvine (MST, restricted to dvine)
        # * TSP with precedence constraints (clustered TSP), only called at lv-0
        # ! all s have to be empty for level‑0 D‑vine
        edge_list = [
            ((v_l, v_r), -abs(w)) for (v_l, v_r, *s), w in edge_weight_lv.items()
        ]
        edge_list.sort(key=lambda x: x[1])
        parent = list(range(self.num_dim))
        degree = [0] * self.num_dim

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
                return True
            return False

        def grow(edge_list, target):
            for (v_l, v_r), _ in edge_list:
                if len(path) >= target:
                    break
                if degree[v_l] < 2 and degree[v_r] < 2 and union(v_l, v_r):
                    path.append((v_l, v_r))
                    degree[v_l] += 1
                    degree[v_r] += 1

        path: list[tuple[int, int]] = []
        cand_v = list(self.first_tree_vertex)
        if len(cand_v) > 1:
            cand_e = [e for e in edge_list if e[0][0] in cand_v and e[0][1] in cand_v]
            grow(edge_list=cand_e, target=len(cand_v) - 1)
        grow(edge_list=edge_list, target=self.num_dim - 1)
        # * canonicalize ordering
        return [(min(v_l, v_r), max(v_l, v_r)) for v_l, v_r in path]

    @torch.no_grad()
    def _mst_from_edge_cvine(self, edge_weight_lv: dict) -> None:
        # * edge2tree, cvine (MST, restricted to cvine)
        # * accumulate |weight| for every pseudo obs vertex (v,s)
        score = defaultdict(float)
        for (v_l, v_r, *s), w in edge_weight_lv.items():
            s = tuple(s)
            w = abs(w)
            score[v_l, s] += w
            score[v_r, s] += w
        # * restrict candidate set if precedence given
        # * fallback if impossible
        cand_v = None
        if self.first_tree_vertex:
            cand_v = {v_s for v_s in score if v_s[0] in self.first_tree_vertex}
        if not cand_v:
            cand_v = set(score)
        v_c, s_c = max(cand_v, key=lambda x: score[x])
        mst = [
            (v_l, v_r, tuple(s))
            for (v_l, v_r, *s), _ in edge_weight_lv.items()
            if tuple(s) == s_c and (v_l == v_c or v_r == v_c)
        ]
        # * canonicalize ordering
        return [(min(v_l, v_r), max(v_l, v_r), *s) for v_l, v_r, s in mst]

    @torch.no_grad()
    def _mst_from_edge_rvine(self, edge_weight_lv: dict) -> None:
        # * edge2tree, rvine (Kruskal's MST, disjoint set/ union find)
        # * lr_s[2:] is cond_ing set
        edge_weight_lv = copy.deepcopy(edge_weight_lv)
        lv = len(next(iter(edge_weight_lv))[2:])
        # ! pseudo obs vertices point to parent bicop vertices (constraint set)
        parent = {}
        for lr_s, cond_ed in self.struct_obs[lv].items():
            parent[lr_s] = (
                frozenset(lr_s + tuple(map(int, cond_ed.split(","))))
                if lv > 0
                else frozenset(lr_s)
            )
        # * bicop vertices point to themselves
        parent.update({r: r for r in parent.values()})
        rank = {r: 0 for r in parent}
        mst = []

        def find(v):
            """path compression"""
            if parent[v] != v:
                parent[v] = find(parent[v])
            return parent[v]

        def union(x, y):
            """union by rank"""
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            rank[rx] += rank[rx] == rank[ry]
            return True

        def kruskal(cand_e_w: dict, num_mst: int) -> None:
            if cand_e_w:
                # ! min heap, by -ABS(bidep) in ASCENDING order
                heap_bidep_abs = [
                    (-abs(bidep), lr_s) for lr_s, bidep in cand_e_w.items()
                ]
                heapq.heapify(heap_bidep_abs)
                while len(mst) < num_mst:
                    _, lr_s = heapq.heappop(heap_bidep_abs)
                    v_l, v_r, *s = lr_s
                    if union(find((v_l, *s)), find((v_r, *s))):
                        mst.append(lr_s)

        # * gradually grow the vine, filter for edges
        cand_v = set()
        for step_v in (set(self.first_tree_vertex), set(range(self.num_dim))):
            cand_v |= step_v
            cand_e_w = {
                # * pop update edge_weight_lv
                e: edge_weight_lv.pop(e)
                for e in list(edge_weight_lv)
                # * edges with both vertices in cand_v
                if e[0] in cand_v and e[1] in cand_v
            }
            kruskal(
                cand_e_w=cand_e_w,
                num_mst=max(
                    0, len(cand_v) - lv - 1
                ),  # * number of edges in the MST, at this stage
            )
        return mst

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        is_dissmann: bool = True,
        matrix: torch.Tensor = None,
        first_tree_vertex: tuple = tuple(),
        mtd_vine: str = "rvine",
        mtd_bidep: str = "chatterjee_xi",
        thresh_trunc: None | float = None,
        num_obs_max: int = None,
        seed: int = 42,
        num_iter_max: int = 17,
        is_tau_est: bool = False,
        num_step_grid_kde1d: int = None,
        **kde_kwargs,
    ) -> None:
        self.reset()
        # ! device agnostic
        device, dtype = self.device, self.dtype
        if self.is_cop_scale:
            obs_mvcp = obs.to(device=device, dtype=dtype)
        else:
            for v in range(self.num_dim):
                self.marginals[v] = kdeCDFPPF1D(
                    x=obs[:, v],
                    num_step_grid=num_step_grid_kde1d,
                    **kde_kwargs,
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
            """lazy hfunc for pseudo-obs at this v_s"""
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
                for s, lst_v in grp_s.items():
                    if len(lst_v) < 2:
                        continue
                    lst_v.sort()
                    # * ensure all pseudo-obs are ready
                    for v in lst_v:
                        if dct_obs[lv][(v, *s)] is None:
                            _visit_hfunc(lv=lv, v_s=(v, *s))
                    for v_l, v_r in combinations(lst_v, 2):
                        # ! already sorted !
                        w = f_bidep(
                            x=dct_obs[lv][v_l, *s],
                            y=dct_obs[lv][v_r, *s],
                        )
                        tmp_edge_weight[(v_l, v_r, *s)] = w[0] if is_kendall_tau else w
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
                    raise ValueError(
                        "mtd_vine must be one of 'cvine', 'dvine', 'rvine'"
                    )
                # * store tree and weights
                self.tree_bidep[lv] = {lr_s: tmp_edge_weight[lr_s] for lr_s in tree}
            else:
                # * tree structure is inferred from matrix, if not Dissmann
                tree = []
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
                    w = f_bidep(
                        x=dct_obs[lv][v_l, *s],
                        y=dct_obs[lv][v_r, *s],
                    )
                    self.tree_bidep[lv][v_l, v_r, *s] = w[0] if is_kendall_tau else w

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
                    is_fitting = (
                        ENUM_FUNC_BIDEP.kendall_tau(obs_bcp[:, [0]], obs_bcp[:, [1]])[1]
                        <= thresh_trunc
                    )
                if is_fitting:
                    bcp.fit(
                        obs=obs_bcp,
                        num_obs_max=num_obs_max,
                        seed=seed,
                        num_iter_max=num_iter_max,
                        is_tau_est=is_tau_est,
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

    def log_pdf(self, obs: torch.Tensor) -> torch.Tensor:
        # ! device agnostic
        device, dtype = self.device, self.dtype
        if self.is_cop_scale:
            obs_mvcp = obs.to(device=device, dtype=dtype)
        else:
            obs_mvcp = torch.hstack(
                [self.marginals[v].cdf(obs[:, [v]]) for v in range(self.num_dim)]
            ).to(device=device, dtype=dtype)
        num_obs, num_dim = obs_mvcp.shape
        lpdf = torch.zeros((num_obs, 1), dtype=dtype, device=device)
        dct_obs = [dict() for _ in range(num_dim)]
        # * lv-0 marginals
        for idx in range(num_dim):
            dct_obs[0][(idx,)] = obs_mvcp[:, [idx]]

        def _visit_hfunc(lv: int, v_s: tuple) -> None:
            """lazy hfunc for pseudo-obs at this v_s"""
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
                            dct_obs[lv_up][(v_l, *s_up)],
                            dct_obs[lv_up][(v_r, *s_up)],
                        ]
                    ),
                )

        for lv in range(num_dim - 1):
            for (v_l, v_r, *s), _ in self.tree_bidep[lv].items():
                s = tuple(s)
                for v in (v_l, v_r):
                    if (v, *s) not in dct_obs[lv]:
                        _visit_hfunc(lv=lv, v_s=(v, *s))
                # ! MUST be a differentiable call!
                lpdf = lpdf + self.bicops[f"{v_l},{v_r}"].log_pdf(
                    torch.hstack(
                        [
                            dct_obs[lv][v_l, *s],
                            dct_obs[lv][v_r, *s],
                        ]
                    )
                )
            if lv > 0:
                # * free prev lv dict
                with torch.no_grad():
                    dct_obs[lv - 1].clear()
        if not self.is_cop_scale:
            # * add-on marginal pdfs
            for v in range(num_dim):
                lpdf = lpdf + self.marginals[v].log_pdf(x=obs[:, [v]])
        return lpdf

    def forward(self, x) -> torch.Tensor:
        return -self.log_pdf(x).mean()

    def rosenblatt(
        self, obs: torch.Tensor, sample_order: tuple | None = None
    ) -> torch.Tensor:
        # TODO, needs grad as residuals
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        num_sample: int = 1000,
        seed: int = 42,
        is_sobol: bool = False,
        sample_order: tuple[int, ...] | None = None,
        dct_v_s_obs: dict[tuple[int, ...], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        def _ref_count_decrement(v_s) -> None:
            ref_count[v_s] -= 1
            if ref_count[v_s] < 1 and (len(v_s) > 1):
                del dct_obs[v_s]

        def _visit(lv: int, v_s: tuple, is_hinv: bool):
            v_down, *s_down = v_s
            s_down = tuple(s_down)
            # * locate the bicop at upper level that connects the three vertices
            cond_ed = self.struct_obs[lv][v_s]
            bcp: BiCop = self.bicops[cond_ed]
            v_l, v_r = self.struct_bcp[cond_ed]["cond_ed"]
            s_up = self.struct_bcp[cond_ed]["cond_ing"]
            is_down_right = v_down == v_r
            # * check missing parents, call hfunc from even upper to visit
            if is_hinv:
                frontier = [(v_l if is_down_right else v_r, *s_up)]
            else:
                frontier = [(v_l, *s_up), (v_r, *s_up)]
            for v_s in frontier:
                if v_s not in dct_obs:
                    _visit(lv=lv - 1, v_s=v_s, is_hinv=False)
            # * update pseudo-obs
            v_s_next = (v_down, *s_up)
            if is_hinv:
                if bcp.is_indep:
                    dct_obs[v_s_next] = dct_obs[v_down, *s_down]
                elif is_down_right:
                    dct_obs[v_s_next] = bcp.hinv_l(
                        torch.hstack(
                            [
                                dct_obs[v_l, *s_up],
                                dct_obs[v_r, *s_down],
                            ]
                        )
                    )
                else:
                    dct_obs[v_s_next] = bcp.hinv_r(
                        torch.hstack(
                            [
                                dct_obs[v_l, *s_down],
                                dct_obs[v_r, *s_up],
                            ]
                        )
                    )
            else:
                if bcp.is_indep:
                    dct_obs[v_down, *s_down] = dct_obs[v_down, *s_up]
                else:
                    dct_obs[v_down, *s_down] = (
                        bcp.hfunc_l if is_down_right else bcp.hfunc_r
                    )(
                        torch.hstack(
                            [
                                dct_obs[v_l, *s_up],
                                dct_obs[v_r, *s_up],
                            ]
                        )
                    )
            # * garbage collection check
            for v_s in [(v_l, *s_up), (v_r, *s_up), (v_down, *s_down)]:
                _ref_count_decrement(v_s=v_s)
            if is_hinv:
                return v_s_next

        # ! device agnostic
        device, dtype = self.device, self.dtype
        torch.manual_seed(seed=seed)
        # ! start with any user‐provided pseudo obs
        dct_obs = dict()
        if dct_v_s_obs:
            for v_s, vec in dct_v_s_obs.items():
                v, *_ = v_s
                dct_obs[v_s] = self.marginals[v].cdf(vec).to(device=device, dtype=dtype)
        # * source vertices in each path; reference counting for whole DAG
        ref_count, lst_source, _ = self.ref_count_hfunc(
            num_dim=self.num_dim,
            struct_obs=self.struct_obs,
            sample_order=sample_order if sample_order else self.sample_order,
        )
        # * draw indep uniform for default source vertices
        dim_sim = sum(1 for v_s in lst_source if v_s not in dct_obs)
        if dim_sim > 0:
            if is_sobol:
                obs_mvcp_indep = (
                    torch.quasirandom.SobolEngine(
                        dimension=dim_sim, scramble=True, seed=seed
                    )
                    .draw(n=num_sample, dtype=dtype)
                    .to(device=device)
                )
            else:
                obs_mvcp_indep = torch.rand(
                    size=(num_sample, dim_sim), device=device, dtype=dtype
                )
        # * initialize source vertices
        idx = 0
        for v_s in lst_source:
            if v_s not in dct_obs:
                dct_obs[v_s] = obs_mvcp_indep[:, [idx]]
                idx += 1
        for v_s in dct_obs:
            dct_obs[v_s].clamp_(min=self._EPS, max=1.0 - self._EPS)
        del obs_mvcp_indep, idx
        # * source to target (empty cond_ing), from the shallowest to the deepest
        for v_s in lst_source:
            lv = len(v_s) - 1
            while lv > 0:
                v_s = _visit(lv=lv, v_s=v_s, is_hinv=True)
                lv -= 1
        # ! gather pseudo-obs by v
        obs_mvcp = torch.hstack([dct_obs[(v,)] for v in range(self.num_dim)])
        if self.is_cop_scale:
            return obs_mvcp
        else:
            # * transform to original scale
            return torch.hstack(
                [self.marginals[v].ppf(obs_mvcp[:, [v]]) for v in range(self.num_dim)]
            )

    @torch.no_grad()
    def cdf(
        self, obs: torch.Tensor, num_sample: int = 10007, seed: int = 0
    ) -> torch.Tensor:
        # ! device agnostic
        device, dtype = self.device, self.dtype
        if self.is_cop_scale:
            obs_mvcp = obs.to(device=device, dtype=dtype)
        else:
            obs_mvcp = torch.hstack(
                [self.marginals[v].cdf(obs[:, [v]]) for v in range(self.num_dim)]
            ).to(device=device, dtype=dtype)
        # * broadcast
        return (
            (
                self.sample(num_sample=num_sample, seed=seed, is_sobol=True).unsqueeze(
                    dim=1
                )  # * (num_sample,1,num_dim)
                <= obs_mvcp  # * (num_sample, num_obs, num_dim)
            )
            .all(
                dim=2,
                keepdim=True,
                # * (num_sample, num_obs, 1)
            )
            .sum(
                axis=0,
                keepdim=False,
                # * (num_obs, 1)
            )
            / num_sample
            # * bool -> float32 -> dtype
        ).to(device=device, dtype=dtype)

    def __str__(self) -> str:
        header = self.__class__.__name__
        params = {
            "num_dim": int(self.num_dim),
            "num_obs": int(self.num_obs),
            "is_cop_scale": self.is_cop_scale,
            "mtd_bidep": self.mtd_bidep,
            "negloglik": float(
                sum(bcp.negloglik for bcp in self.bicops.values()).round(decimals=4)
            ),
            "num_step_grid": int(self.num_step_grid),
            "dtype": self.dtype,
            "device": self.device,
            "sample_order": self.sample_order,
        }
        params_str = pformat(params, sort_dicts=False, underscore_numbers=True)
        matrix_str = indent(str(self.matrix), " " * 4)
        return f"{header}\n{params_str[1:-1]},\n 'matrix':\n{matrix_str}\n\n"

    @torch.no_grad()
    def draw_lv(
        self,
        lv: int = 0,
        is_bcp: bool = True,
        title: str | None = None,
        num_digit: int = 2,
        font_size_vertex: int = 8,
        font_size_edge: int = 7,
        f_path: Path = None,
        fig_size: tuple = None,
    ):
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "Please install matplotlib and networkx to draw the vine copula."
            ) from e
        tree = self.tree_bidep[lv]
        edge_weight = []
        if lv == 0:
            # * level‐0: plain variable indices
            for (u, v, *_), w in tree.items():
                edge_weight.append((u, v, round(w.item(), num_digit)))
        elif is_bcp:
            # ! nodes are the parent‐bicop "l,r;s"
            for (u, v, *s), w in tree.items():
                # * parent bicop cond_ed str
                label_u = self.struct_obs[lv][(u, *s)]
                label_v = self.struct_obs[lv][(v, *s)]
                # * append the cond_ing set
                sep = "" if lv == 1 else "\n"
                label_u = f"{label_u};{sep}{
                    ','.join(
                        str(x) for x in sorted(self.struct_bcp[label_u]['cond_ing'])
                    )
                }"
                label_v = f"{label_v};{sep}{
                    ','.join(
                        str(x) for x in sorted(self.struct_bcp[label_v]['cond_ing'])
                    )
                }"
                edge_weight.append((label_u, label_v, round(w.item(), num_digit)))
        else:
            # ! nodes are pseudo‐obs "v|s"
            for (u, v, *s), w in tree.items():
                s_str = ",".join(str(x) for x in sorted(s))
                label_u = f"{u}|{s_str}"
                label_v = f"{v}|{s_str}"
                edge_weight.append((label_u, label_v, round(w.item(), num_digit)))

        # * weighted undirected graph
        G = nx.Graph()
        G.add_weighted_edges_from(edge_weight)
        fig, ax = plt.subplots(figsize=fig_size)
        if title is None:
            title = f"Vine level {lv}"
        ax.set_title(title, fontsize=font_size_vertex + 1)
        pos = nx.planar_layout(G)
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color="white",
            node_shape="s" if (is_bcp and lv > 0) else "o",
            linewidths=0.5,
            edgecolors="gray",
            alpha=0.8,
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax, font_size=font_size_vertex, font_color="black"
        )
        # * scale line‐width by weight
        widths = [
            math.log1p(0.5 + 100 * abs(data["weight"]))
            for _, _, data in G.edges(data=True)
        ]
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, style="--", alpha=0.9)
        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=nx.get_edge_attributes(G, "weight"),
            font_size=font_size_edge,
        )
        ax.set_axis_off()
        fig.tight_layout()
        plt.draw_if_interactive()

        if f_path:
            fig.savefig(f_path, bbox_inches="tight")
            return fig, ax, G, f_path
        return fig, ax, G

    @torch.no_grad()
    def draw_dag(
        self,
        sample_order: tuple[int, ...] = None,
        title: str = "Vine comp graph",
        font_size_vertex: int = 8,
        f_path: Path = None,
        fig_size: tuple = None,
    ):
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "Please install matplotlib and networkx to draw the vine copula."
            ) from e

        G = nx.DiGraph()
        labels: dict = {}
        pos_obs: dict = {}
        pos_bcp: dict = {}

        def add_level(lv: int):
            edges = []
            bicops = []
            downstream = []
            # * lv-0 marginals
            if lv == 0:
                xs = np.linspace(-self.num_dim / 2, self.num_dim / 2, self.num_dim)
                for v, x in enumerate(xs):
                    node = (v, frozenset())
                    labels[node] = str(v)
                    pos_obs[node] = (float(x), 1.0)
            # * traverse around the bcp
            for v_l, v_r, *cond in self.tree_bidep[lv]:
                cond_set = frozenset(cond)
                bcp = (v_l, v_r, cond_set)
                bicops.append(bcp)
                up_l = (v_l, cond_set)
                up_r = (v_r, cond_set)
                down_l = (v_l, cond_set | {v_r})
                down_r = (v_r, cond_set | {v_l})
                downstream.extend([down_l, down_r])
                # * edges: upstream → bicop → downstream
                edges += [
                    (up_l, bcp),
                    (up_r, bcp),
                    (bcp, down_l),
                    (bcp, down_r),
                ]
                # * labels
                labels[down_l] = f"{down_l[0]}|{','.join(map(str, sorted(down_l[1])))}"
                labels[down_r] = f"{down_r[0]}|{','.join(map(str, sorted(down_r[1])))}"
                br = "\n" if lv > 0 else ""
                labels[bcp] = f"{v_l},{v_r};{br}{','.join(map(str, sorted(cond_set)))}"
            # * layout downstream at y = –lv
            if downstream:
                xs = np.linspace(
                    -len(downstream) / 2, len(downstream) / 2, len(downstream)
                )
                for i, node in enumerate(downstream):
                    pos_obs[node] = (float(xs[i]), float(-lv))
            # * layout bicops at y = –lv + 0.5
            if bicops:
                xs = np.linspace(-len(bicops) / 2, len(bicops) / 2, len(bicops))
                for i, node in enumerate(bicops):
                    pos_bcp[node] = (float(xs[i]), float(-lv + 0.5))
            return edges

        # * accumulate over all levels
        all_edges = []
        for lv in range(len(self.tree_bidep)):
            all_edges.extend(add_level(lv))
        G.add_edges_from(all_edges)
        # * layout dictionary
        pos = {**pos_obs, **pos_bcp}
        # * pseudo-obs to highlight
        _, node_source, _ = self.ref_count_hfunc(
            num_dim=self.num_dim,
            struct_obs=self.struct_obs,
            sample_order=sample_order
            if sample_order is not None
            else self.sample_order,
        )
        node_source = [(v_s[0], frozenset(v_s[1:])) for v_s in node_source]
        # * draw
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        ax.set_title(label=title, fontsize=font_size_vertex + 1)
        # * pseudo-obs (white)
        node_obs = [_ for _ in G.nodes if len(_) == 2 and _ not in node_source]
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            nodelist=node_obs,
            ax=ax,
            node_shape="o",
            node_color="white",
            edgecolors="gray",
            linewidths=0.5,
            alpha=0.8,
        )
        # * pseudo-obs (yellow)
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            nodelist=node_source,
            ax=ax,
            node_shape="o",
            node_color="yellow",
            edgecolors="gray",
            linewidths=0.5,
            alpha=0.9,
        )
        # * bcp nodes
        node_bcp = [_ for _ in G.nodes if len(_) == 3]
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            nodelist=node_bcp,
            ax=ax,
            node_shape="s",
            node_color="white",
            edgecolors="gray",
            linewidths=0.5,
            alpha=0.8,
        )
        nx.draw_networkx_labels(
            G=G,
            pos=pos,
            labels=labels,
            ax=ax,
            font_size=font_size_vertex,
            font_color="black",
        )
        nx.draw_networkx_edges(
            G=G,
            pos=pos,
            ax=ax,
            edge_color="gray",
            style="--",
            width=0.5,
            alpha=0.8,
        )
        ax.set_axis_off()
        fig.tight_layout()
        plt.draw_if_interactive()
        if f_path:
            fig.savefig(fname=f_path, bbox_inches="tight")
            return fig, ax, G, f_path
        else:
            return fig, ax, G
