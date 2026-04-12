from __future__ import annotations

import torch

from .artifact import VineBuildArtifact, _device_from_artifact, _dtype_from_artifact
from .plan import (
    _FORWARD_COPY,
    _FORWARD_HFUNC_L,
    _SAMPLE_COPY,
    _SAMPLE_HFUNC_L,
    _SAMPLE_HFUNC_R,
    _SAMPLE_HINV_L,
    _build_sample_plan,
    _canonical_vertex,
    _tensor_int,
)
from .structure import _ref_count_hfunc_impl

__all__ = ["VineCopEngine"]


class VineCopEngine(torch.nn.Module):
    def __init__(self, artifact: VineBuildArtifact | None = None) -> None:
        super().__init__()
        object.__setattr__(self, "artifact", artifact)
        if artifact is None:
            self.marginals = torch.nn.ModuleList()
            self.bicops = torch.nn.ModuleList()
            self.vertex_order = tuple()
            self.slot_by_vertex = {}
            self.sample_order = tuple()
            self.source_vertices = tuple()
            self.bicop_order = tuple()
            self.register_buffer("_dd", torch.tensor([], dtype=torch.float64))
            self.register_buffer("num_obs", torch.zeros((), dtype=torch.int))
            return
        device = _device_from_artifact(artifact)
        dtype = _dtype_from_artifact(artifact)
        self.is_cop_scale = artifact.is_cop_scale
        self.boundary_policy = artifact.boundary_policy
        self.num_dim = artifact.num_dim
        self.num_step_grid = artifact.num_step_grid
        self.sample_order = tuple(artifact.sample_order)
        self.source_vertices = tuple(artifact.source_vertices)
        self.vertex_order = tuple(artifact.vertex_order)
        self.slot_by_vertex = dict(artifact.slot_by_vertex)
        self.bicop_order = tuple(artifact.bicop_order)
        self.marginals = artifact.marginals
        self.bicops = torch.nn.ModuleList([artifact.bicops[name] for name in artifact.bicop_order])
        self.register_buffer("_dd", torch.tensor([], device=device, dtype=dtype))
        self.register_buffer(
            "num_obs", torch.tensor(artifact.num_obs, device=device, dtype=torch.int)
        )
        for name in (
            "base_slots",
            "source_slots",
            "forward_output_slots",
            "forward_left_slots",
            "forward_right_slots",
            "forward_source_slots",
            "forward_bicop_ids",
            "forward_modes",
            "logpdf_left_slots",
            "logpdf_right_slots",
            "logpdf_bicop_ids",
            "sample_output_slots",
            "sample_left_slots",
            "sample_right_slots",
            "sample_bicop_ids",
            "sample_modes",
        ):
            self.register_buffer(name, getattr(artifact, name))
        self._base_slots_py = tuple(int(x) for x in artifact.base_slots.tolist())
        self._source_slots_py = tuple(int(x) for x in artifact.source_slots.tolist())
        self._forward_output_slots_py = tuple(
            int(x) for x in artifact.forward_output_slots.tolist()
        )
        self._forward_left_slots_py = tuple(int(x) for x in artifact.forward_left_slots.tolist())
        self._forward_right_slots_py = tuple(int(x) for x in artifact.forward_right_slots.tolist())
        self._forward_source_slots_py = tuple(
            int(x) for x in artifact.forward_source_slots.tolist()
        )
        self._forward_bicop_ids_py = tuple(int(x) for x in artifact.forward_bicop_ids.tolist())
        self._forward_modes_py = tuple(int(x) for x in artifact.forward_modes.tolist())
        self._logpdf_left_slots_py = tuple(int(x) for x in artifact.logpdf_left_slots.tolist())
        self._logpdf_right_slots_py = tuple(int(x) for x in artifact.logpdf_right_slots.tolist())
        self._logpdf_bicop_ids_py = tuple(int(x) for x in artifact.logpdf_bicop_ids.tolist())
        self._sample_output_slots_py = tuple(int(x) for x in artifact.sample_output_slots.tolist())
        self._sample_left_slots_py = tuple(int(x) for x in artifact.sample_left_slots.tolist())
        self._sample_right_slots_py = tuple(int(x) for x in artifact.sample_right_slots.tolist())
        self._sample_bicop_ids_py = tuple(int(x) for x in artifact.sample_bicop_ids.tolist())
        self._sample_modes_py = tuple(int(x) for x in artifact.sample_modes.tolist())

    @property
    def device(self) -> torch.device:
        return self._dd.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dd.dtype

    def _as_copula_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.is_cop_scale:
            return obs.to(device=self.device, dtype=self.dtype)
        return torch.hstack([self.marginals[v].cdf(obs[:, [v]]) for v in range(self.num_dim)]).to(
            device=self.device,
            dtype=self.dtype,
        )

    def _alloc_pool(self, batch_size: int) -> torch.Tensor:
        return torch.empty(
            (len(self.vertex_order), batch_size), device=self.device, dtype=self.dtype
        )

    def _run_forward_plan(self, obs_mvcp: torch.Tensor) -> torch.Tensor:
        num_obs = obs_mvcp.shape[0]
        pool = self._alloc_pool(num_obs)
        pool[self.base_slots] = obs_mvcp.T
        if not self._forward_modes_py:
            return pool
        for idx, mode in enumerate(self._forward_modes_py):
            out_slot = self._forward_output_slots_py[idx]
            if mode == _FORWARD_COPY:
                pool[out_slot] = pool[self._forward_source_slots_py[idx]]
                continue
            pair = torch.stack(
                [
                    pool[self._forward_left_slots_py[idx]],
                    pool[self._forward_right_slots_py[idx]],
                ],
                dim=1,
            )
            bicop = self.bicops[self._forward_bicop_ids_py[idx]]
            out = bicop.hfunc_l(pair) if mode == _FORWARD_HFUNC_L else bicop.hfunc_r(pair)
            pool[out_slot] = out.squeeze(1)
        return pool

    def _resolve_source_slots(
        self,
        sample_order: tuple[int, ...] | None = None,
    ) -> tuple[tuple[tuple[int, ...], ...], torch.Tensor]:
        if sample_order is None or tuple(sample_order) == self.sample_order:
            return self.source_vertices, self.source_slots
        try:
            _, source_vertices_raw, _ = _ref_count_hfunc_impl(
                num_dim=self.num_dim,
                struct_obs=self.artifact.struct_obs,
                sample_order=tuple(sample_order),
            )
        except KeyError as exc:
            raise ValueError(
                "sample_order is incompatible with the fitted vine structure."
            ) from exc
        source_vertices = tuple(_canonical_vertex(v_s) for v_s in source_vertices_raw)
        source_slots = _tensor_int(
            [self.slot_by_vertex[v_s] for v_s in source_vertices], self.device
        )
        return source_vertices, source_slots

    def log_pdf(self, obs: torch.Tensor) -> torch.Tensor:
        obs_mvcp = self._as_copula_obs(obs)
        pool = self._run_forward_plan(obs_mvcp)
        lpdf = torch.zeros((obs_mvcp.shape[0], 1), dtype=self.dtype, device=self.device)
        if self._logpdf_bicop_ids_py:
            for idx, bicop_id in enumerate(self._logpdf_bicop_ids_py):
                pair = torch.stack(
                    [
                        pool[self._logpdf_left_slots_py[idx]],
                        pool[self._logpdf_right_slots_py[idx]],
                    ],
                    dim=1,
                )
                lpdf = lpdf + self.bicops[bicop_id].log_pdf(pair)
        if not self.is_cop_scale:
            for v in range(self.num_dim):
                lpdf = lpdf + self.marginals[v].log_pdf(x=obs[:, [v]])
        return lpdf

    def rosenblatt(
        self, obs: torch.Tensor, sample_order: tuple[int, ...] | None = None
    ) -> torch.Tensor:
        obs_mvcp = self._as_copula_obs(obs)
        pool = self._run_forward_plan(obs_mvcp)
        _, source_slots = self._resolve_source_slots(sample_order=sample_order)
        return pool[source_slots].T

    @torch.no_grad()
    def _execute_sample_plan(
        self,
        *,
        num_sample: int,
        source_vertices: tuple[tuple[int, ...], ...],
        source_slots: torch.Tensor,
        sample_output_slots: torch.Tensor,
        sample_left_slots: torch.Tensor,
        sample_right_slots: torch.Tensor,
        sample_bicop_ids: torch.Tensor,
        sample_modes: torch.Tensor,
        seed: int | None = 42,
        is_sobol: bool = False,
        dct_v_s_obs: dict[tuple[int, ...], torch.Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        pool = self._alloc_pool(num_sample)
        filled = torch.zeros((len(self.vertex_order),), dtype=torch.bool, device=self.device)
        if dct_v_s_obs:
            for v_s, vec in dct_v_s_obs.items():
                v_s = _canonical_vertex(v_s)
                slot = self.slot_by_vertex[v_s]
                if len(v_s) == 1 and not self.is_cop_scale:
                    pool[slot] = (
                        self.marginals[v_s[0]]
                        .cdf(vec)
                        .view(-1)
                        .to(device=self.device, dtype=self.dtype)
                    )
                else:
                    pool[slot] = vec.view(-1).to(device=self.device, dtype=self.dtype)
                filled[slot] = True
        missing = [slot for slot in source_slots.tolist() if not bool(filled[slot])]
        dim_sim = len(missing)
        if dim_sim > 0:
            if is_sobol:
                obs_mvcp_indep = (
                    torch.quasirandom.SobolEngine(dimension=dim_sim, scramble=True, seed=seed)
                    .draw(n=num_sample, dtype=self.dtype)
                    .to(device=self.device)
                )
            else:
                if generator is None:
                    generator = torch.Generator(device=self.device)
                    if seed is not None:
                        generator.manual_seed(seed)
                obs_mvcp_indep = torch.rand(
                    size=(num_sample, dim_sim),
                    device=self.device,
                    dtype=self.dtype,
                    generator=generator,
                )
            for idx, slot in enumerate(missing):
                pool[slot] = obs_mvcp_indep[:, idx]
                filled[slot] = True
        if sample_modes.numel() > 0:
            pair = torch.empty((num_sample, 2), device=self.device, dtype=self.dtype)
            sample_output_slots_py = tuple(int(x) for x in sample_output_slots.tolist())
            sample_left_slots_py = tuple(int(x) for x in sample_left_slots.tolist())
            sample_right_slots_py = tuple(int(x) for x in sample_right_slots.tolist())
            sample_bicop_ids_py = tuple(int(x) for x in sample_bicop_ids.tolist())
            sample_modes_py = tuple(int(x) for x in sample_modes.tolist())
            for idx, mode in enumerate(sample_modes_py):
                out_slot = sample_output_slots_py[idx]
                if filled[out_slot]:
                    continue
                left_slot = sample_left_slots_py[idx]
                if mode == _SAMPLE_COPY:
                    pool[out_slot] = pool[left_slot]
                else:
                    pair[:, 0] = pool[left_slot]
                    pair[:, 1] = pool[sample_right_slots_py[idx]]
                    bicop = self.bicops[sample_bicop_ids_py[idx]]
                    if mode == _SAMPLE_HFUNC_L:
                        pool[out_slot] = bicop.hfunc_l(pair).squeeze(1)
                    elif mode == _SAMPLE_HFUNC_R:
                        pool[out_slot] = bicop.hfunc_r(pair).squeeze(1)
                    elif mode == _SAMPLE_HINV_L:
                        pool[out_slot] = bicop.hinv_l(pair).squeeze(1)
                    else:
                        pool[out_slot] = bicop.hinv_r(pair).squeeze(1)
                filled[out_slot] = True
        return pool[self.base_slots].T

    @torch.no_grad()
    def _sample_u(
        self,
        num_sample: int = 1000,
        seed: int | None = 42,
        is_sobol: bool = False,
        sample_order: tuple[int, ...] | None = None,
        dct_v_s_obs: dict[tuple[int, ...], torch.Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if sample_order is None or tuple(sample_order) == self.sample_order:
            return self._execute_sample_plan(
                num_sample=num_sample,
                source_vertices=self.source_vertices,
                source_slots=self.source_slots,
                sample_output_slots=self.sample_output_slots,
                sample_left_slots=self.sample_left_slots,
                sample_right_slots=self.sample_right_slots,
                sample_bicop_ids=self.sample_bicop_ids,
                sample_modes=self.sample_modes,
                seed=seed,
                is_sobol=is_sobol,
                dct_v_s_obs=dct_v_s_obs,
                generator=generator,
            )
        (
            source_vertices,
            source_slots,
            sample_output_slots,
            sample_left_slots,
            sample_right_slots,
            sample_bicop_ids,
            sample_modes,
        ) = _build_sample_plan(
            num_dim=self.num_dim,
            struct_obs=self.artifact.struct_obs,
            struct_bcp=self.artifact.struct_bcp,
            bicops=self.artifact.bicops,
            slot_by_vertex=self.slot_by_vertex,
            bicop_id_by_cond_ed={name: idx for idx, name in enumerate(self.bicop_order)},
            sample_order=tuple(sample_order),
            device=self.device,
        )
        return self._execute_sample_plan(
            num_sample=num_sample,
            source_vertices=source_vertices,
            source_slots=source_slots,
            sample_output_slots=sample_output_slots,
            sample_left_slots=sample_left_slots,
            sample_right_slots=sample_right_slots,
            sample_bicop_ids=sample_bicop_ids,
            sample_modes=sample_modes,
            seed=seed,
            is_sobol=is_sobol,
            dct_v_s_obs=dct_v_s_obs,
            generator=generator,
        )

    @torch.no_grad()
    def inverse_rosenblatt(
        self,
        obs: torch.Tensor,
        sample_order: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        source_vertices, _ = self._resolve_source_slots(sample_order=sample_order)
        dct_v_s_obs = {
            v_s: obs[:, [idx]].to(device=self.device, dtype=self.dtype)
            for idx, v_s in enumerate(source_vertices)
        }
        out = self._sample_u(
            num_sample=obs.shape[0],
            sample_order=sample_order,
            dct_v_s_obs=dct_v_s_obs,
        )
        if self.is_cop_scale:
            return out
        return torch.hstack([self.marginals[v].ppf(out[:, [v]]) for v in range(self.num_dim)])

    @torch.no_grad()
    def sample(
        self,
        num_sample: int = 1000,
        seed: int | None = 42,
        is_sobol: bool = False,
        sample_order: tuple[int, ...] | None = None,
        dct_v_s_obs: dict[tuple[int, ...], torch.Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        obs_mvcp = self._sample_u(
            num_sample=num_sample,
            seed=seed,
            is_sobol=is_sobol,
            sample_order=sample_order,
            dct_v_s_obs=dct_v_s_obs,
            generator=generator,
        )
        if self.is_cop_scale:
            return obs_mvcp
        return torch.hstack([self.marginals[v].ppf(obs_mvcp[:, [v]]) for v in range(self.num_dim)])

    @torch.no_grad()
    def cdf(self, obs: torch.Tensor, num_sample: int = 10007, seed: int = 42) -> torch.Tensor:
        obs_mvcp = self._as_copula_obs(obs)
        return (
            (
                self._sample_u(num_sample=num_sample, seed=seed, is_sobol=True).unsqueeze(dim=1)
                <= obs_mvcp
            )
            .all(dim=2, keepdim=True)
            .sum(axis=0, keepdim=False)
            / num_sample
        ).to(device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_pdf(x).mean()
