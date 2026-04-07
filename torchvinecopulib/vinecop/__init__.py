"""Multivariate vine copula fitting and query execution.

This module provides the public `VineCop` façade, the explicit `VineBuilder`, and the
plan-backed `VineCopEngine` used for multivariate fitting, likelihood evaluation, Rosenblatt
transforms, sampling, and diagnostics.
"""

import copy
from dataclasses import dataclass
import heapq
import math
import warnings
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from pathlib import Path
from pprint import pformat
from textwrap import indent
from typing import Any, Literal

import torch

from ..backends import (
    API_VERSION,
    build_marginal_estimator,
    build_marginal_shell,
    normalize_bicop_backend,
    normalize_marginal_backend,
)
from ..bicop import BiCop
from ..util import ENUM_FUNC_BIDEP, kendall_tau, kendall_tau_matrix

__all__ = [
    "VineCop",
    "VineBuildArtifact",
    "VineExecutionPlan",
    "VineDiagnostics",
    "VineBuilder",
    "VineCopEngine",
]

_FORWARD_COPY = 0
_FORWARD_HFUNC_L = 1
_FORWARD_HFUNC_R = 2

_SAMPLE_COPY = 0
_SAMPLE_HFUNC_L = 1
_SAMPLE_HFUNC_R = 2
_SAMPLE_HINV_L = 3
_SAMPLE_HINV_R = 4


@dataclass(frozen=True)
class VineDiagnostics:
    num_edges: int
    itp_failures: int
    bisect_refinements: int
    fallback_to_indep: int
    max_abs_hfunc_error: float


@dataclass
class VineBuildArtifact:
    num_dim: int
    is_cop_scale: bool
    num_step_grid: int
    boundary_policy: Literal["hard", "st"]
    marginals: torch.nn.ModuleList
    bicops: torch.nn.ModuleDict
    struct_bcp: dict[str, dict[str, Any]]
    struct_obs: list[dict[tuple[int, ...], str]]
    tree_bidep: list[dict[tuple[int, ...], torch.Tensor | float]]
    sample_order: tuple[int, ...]
    first_tree_vertex: tuple[int, ...]
    marginal_backend: str
    marginal_backend_config: dict[str, Any]
    bicop_backend: str
    bicop_backend_config: dict[str, Any]
    mtd_bidep: str | None
    num_obs: int
    level_edge_tensors: tuple[torch.Tensor, ...]
    vertex_order: tuple[tuple[int, ...], ...]
    slot_by_vertex: dict[tuple[int, ...], int]
    base_slots: torch.Tensor
    source_vertices: tuple[tuple[int, ...], ...]
    source_slots: torch.Tensor
    bicop_order: tuple[str, ...]
    forward_output_slots: torch.Tensor
    forward_left_slots: torch.Tensor
    forward_right_slots: torch.Tensor
    forward_source_slots: torch.Tensor
    forward_bicop_ids: torch.Tensor
    forward_modes: torch.Tensor
    logpdf_left_slots: torch.Tensor
    logpdf_right_slots: torch.Tensor
    logpdf_bicop_ids: torch.Tensor
    sample_output_slots: torch.Tensor
    sample_left_slots: torch.Tensor
    sample_right_slots: torch.Tensor
    sample_bicop_ids: torch.Tensor
    sample_modes: torch.Tensor


VineExecutionPlan = VineBuildArtifact


def _canonical_vertex(v_s: tuple[int, ...]) -> tuple[int, ...]:
    v, *s = v_s
    return (int(v), *sorted(int(x) for x in s))


@torch.no_grad()
def _ref_count_hfunc_impl(
    num_dim: int,
    struct_obs: list[dict[tuple[int, ...], str]],
    sample_order: tuple[int, ...],
) -> tuple[dict[tuple[int, ...], int], list[tuple[int, ...]], int]:
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

    def _visit(v_s: tuple[int, ...], is_hinv: bool = False):
        nonlocal num_hfunc
        if len(v_s) == 1:
            ref_cnt[v_s] += 1
            return
        v_down, *s_down = v_s
        v_l, v_r = map(int, struct_obs[len(s_down)][v_s].split(","))
        s_up = tuple(sorted(set(s_down) - {v_l, v_r}))
        frontier = [(v_l if v_down == v_r else v_r, *s_up)] if is_hinv else [(v_l, *s_up), (v_r, *s_up)]
        for v_s_parent in frontier:
            if ref_cnt[v_s_parent] == 0:
                _visit(v_s=v_s_parent, is_hinv=False)
                num_hfunc += 1
        frontier = [(v_l, *s_up), (v_r, *s_up), (v_down, *s_down)]
        for v_s_parent in frontier:
            ref_cnt[v_s_parent] += 1
        if is_hinv:
            return (v_down, *s_up)

    for v_s in lst_source:
        if len(v_s) == 1:
            ref_cnt[v_s] += 1
        while len(v_s) > 1:
            v_s = _visit(v_s=v_s, is_hinv=True)
    return dict(ref_cnt), lst_source, num_hfunc


def _device_from_artifact(artifact: VineBuildArtifact) -> torch.device:
    for marginal in artifact.marginals:
        if marginal is not None:
            return marginal.device
    for bicop_name in artifact.bicop_order:
        return artifact.bicops[bicop_name].device
    return artifact.base_slots.device


def _dtype_from_artifact(artifact: VineBuildArtifact) -> torch.dtype:
    for marginal in artifact.marginals:
        if marginal is not None:
            return marginal.dtype
    for bicop_name in artifact.bicop_order:
        return artifact.bicops[bicop_name].dtype
    return torch.float64


def _tensor_int(values: list[int], device: torch.device) -> torch.Tensor:
    if values:
        return torch.tensor(values, dtype=torch.long, device=device)
    return torch.empty((0,), dtype=torch.long, device=device)


def _build_vertex_slots(
    *,
    num_dim: int,
    struct_obs: list[dict[tuple[int, ...], str]],
    device: torch.device,
) -> tuple[tuple[tuple[int, ...], ...], dict[tuple[int, ...], int], torch.Tensor]:
    vertex_order: list[tuple[int, ...]] = [(idx,) for idx in range(num_dim)]
    for lv in range(1, num_dim):
        vertex_order.extend(sorted(_canonical_vertex(v_s) for v_s in struct_obs[lv]))
    slot_by_vertex = {v_s: idx for idx, v_s in enumerate(vertex_order)}
    base_slots = torch.arange(num_dim, dtype=torch.long, device=device)
    return tuple(vertex_order), slot_by_vertex, base_slots


def _build_forward_plan(
    *,
    struct_obs: list[dict[tuple[int, ...], str]],
    struct_bcp: dict[str, dict[str, Any]],
    bicops: torch.nn.ModuleDict,
    slot_by_vertex: dict[tuple[int, ...], int],
    bicop_id_by_cond_ed: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output_slots: list[int] = []
    left_slots: list[int] = []
    right_slots: list[int] = []
    source_slots: list[int] = []
    bicop_ids: list[int] = []
    modes: list[int] = []
    for lv in range(1, len(struct_obs)):
        for v_s in sorted(struct_obs[lv]):
            v_s = _canonical_vertex(v_s)
            cond_ed = struct_obs[lv][v_s]
            info = struct_bcp[cond_ed]
            v_l, v_r = info["cond_ed"]
            s_up = tuple(int(x) for x in info["cond_ing"])
            output_slots.append(slot_by_vertex[v_s])
            left_slots.append(slot_by_vertex[(v_l, *s_up)])
            right_slots.append(slot_by_vertex[(v_r, *s_up)])
            source_slots.append(slot_by_vertex[(v_s[0], *s_up)])
            bicop_ids.append(bicop_id_by_cond_ed[cond_ed])
            if bicops[cond_ed].is_indep:
                modes.append(_FORWARD_COPY)
            else:
                modes.append(_FORWARD_HFUNC_R if v_s[0] == v_l else _FORWARD_HFUNC_L)
    return (
        _tensor_int(output_slots, device),
        _tensor_int(left_slots, device),
        _tensor_int(right_slots, device),
        _tensor_int(source_slots, device),
        _tensor_int(bicop_ids, device),
        _tensor_int(modes, device),
    )


def _build_logpdf_plan(
    *,
    tree_bidep: list[dict[tuple[int, ...], torch.Tensor | float]],
    slot_by_vertex: dict[tuple[int, ...], int],
    bicop_id_by_cond_ed: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left_slots: list[int] = []
    right_slots: list[int] = []
    bicop_ids: list[int] = []
    for level in tree_bidep:
        for v_l, v_r, *s in level:
            s_up = tuple(sorted(int(x) for x in s))
            left_slots.append(slot_by_vertex[(int(v_l), *s_up)])
            right_slots.append(slot_by_vertex[(int(v_r), *s_up)])
            bicop_ids.append(bicop_id_by_cond_ed[f"{int(v_l)},{int(v_r)}"])
    return (
        _tensor_int(left_slots, device),
        _tensor_int(right_slots, device),
        _tensor_int(bicop_ids, device),
    )


def _build_sample_plan(
    *,
    num_dim: int,
    struct_obs: list[dict[tuple[int, ...], str]],
    struct_bcp: dict[str, dict[str, Any]],
    bicops: torch.nn.ModuleDict,
    slot_by_vertex: dict[tuple[int, ...], int],
    bicop_id_by_cond_ed: dict[str, int],
    sample_order: tuple[int, ...],
    device: torch.device,
) -> tuple[
    tuple[tuple[int, ...], ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _, source_vertices_raw, _ = _ref_count_hfunc_impl(
        num_dim=num_dim,
        struct_obs=struct_obs,
        sample_order=sample_order,
    )
    source_vertices = tuple(_canonical_vertex(v_s) for v_s in source_vertices_raw)
    source_slots = _tensor_int([slot_by_vertex[v_s] for v_s in source_vertices], device)
    output_slots: list[int] = []
    left_slots: list[int] = []
    right_slots: list[int] = []
    bicop_ids: list[int] = []
    modes: list[int] = []
    available = set(source_vertices)

    def _record_visit(lv: int, v_s: tuple[int, ...], is_hinv: bool) -> tuple[int, ...] | None:
        v_s = _canonical_vertex(v_s)
        v_down, *s_down = v_s
        s_down = tuple(s_down)
        cond_ed = struct_obs[lv][v_s]
        info = struct_bcp[cond_ed]
        v_l, v_r = map(int, info["cond_ed"])
        s_up = tuple(int(x) for x in info["cond_ing"])
        is_down_right = v_down == v_r
        frontier = (
            [(v_l if is_down_right else v_r, *s_up)]
            if is_hinv
            else [(v_l, *s_up), (v_r, *s_up)]
        )
        for parent_v in frontier:
            parent_v = _canonical_vertex(parent_v)
            if parent_v not in available:
                _record_visit(lv=lv - 1, v_s=parent_v, is_hinv=False)
        v_s_next = (v_down, *s_up)
        if is_hinv:
            if bicops[cond_ed].is_indep:
                mode = _SAMPLE_COPY
                left_v = (v_down, *s_down)
                right_v = ()
            elif is_down_right:
                mode = _SAMPLE_HINV_L
                left_v = (v_l, *s_up)
                right_v = (v_r, *s_down)
            else:
                mode = _SAMPLE_HINV_R
                left_v = (v_l, *s_down)
                right_v = (v_r, *s_up)
            output_v = v_s_next
        else:
            if bicops[cond_ed].is_indep:
                mode = _SAMPLE_COPY
                left_v = (v_down, *s_up)
                right_v = ()
            elif is_down_right:
                mode = _SAMPLE_HFUNC_L
                left_v = (v_l, *s_up)
                right_v = (v_r, *s_up)
            else:
                mode = _SAMPLE_HFUNC_R
                left_v = (v_l, *s_up)
                right_v = (v_r, *s_up)
            output_v = (v_down, *s_down)
        output_slots.append(slot_by_vertex[_canonical_vertex(output_v)])
        left_slots.append(slot_by_vertex[_canonical_vertex(left_v)])
        right_slots.append(slot_by_vertex[_canonical_vertex(right_v)] if right_v else -1)
        bicop_ids.append(bicop_id_by_cond_ed[cond_ed])
        modes.append(mode)
        available.add(_canonical_vertex(output_v))
        if is_hinv:
            return _canonical_vertex(v_s_next)
        return None

    for v_s in source_vertices:
        curr = v_s
        lv = len(curr) - 1
        while lv > 0:
            next_v = _record_visit(lv=lv, v_s=curr, is_hinv=True)
            curr = next_v if next_v is not None else curr
            lv -= 1

    return (
        source_vertices,
        source_slots,
        _tensor_int(output_slots, device),
        _tensor_int(left_slots, device),
        _tensor_int(right_slots, device),
        _tensor_int(bicop_ids, device),
        _tensor_int(modes, device),
    )


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
        self.register_buffer("num_obs", torch.tensor(artifact.num_obs, device=device, dtype=torch.int))
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
        self._forward_output_slots_py = tuple(int(x) for x in artifact.forward_output_slots.tolist())
        self._forward_left_slots_py = tuple(int(x) for x in artifact.forward_left_slots.tolist())
        self._forward_right_slots_py = tuple(int(x) for x in artifact.forward_right_slots.tolist())
        self._forward_source_slots_py = tuple(int(x) for x in artifact.forward_source_slots.tolist())
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
        return torch.empty((len(self.vertex_order), batch_size), device=self.device, dtype=self.dtype)

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
            raise ValueError("sample_order is incompatible with the fitted vine structure.") from exc
        source_vertices = tuple(_canonical_vertex(v_s) for v_s in source_vertices_raw)
        source_slots = _tensor_int([self.slot_by_vertex[v_s] for v_s in source_vertices], self.device)
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

    def rosenblatt(self, obs: torch.Tensor, sample_order: tuple[int, ...] | None = None) -> torch.Tensor:
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
                    pool[slot] = self.marginals[v_s[0]].cdf(vec).view(-1).to(device=self.device, dtype=self.dtype)
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
        source_vertices, source_slots, sample_output_slots, sample_left_slots, sample_right_slots, sample_bicop_ids, sample_modes = _build_sample_plan(
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


class VineBuilder:
    def __init__(self, model: "VineCop") -> None:
        self.model = model

    def build(self, obs: torch.Tensor, **kwargs) -> VineBuildArtifact:
        return self.model._fit_impl(obs=obs, **kwargs)


class VineCop(torch.nn.Module):
    def __init__(
        self,
        num_dim: int,
        is_cop_scale: bool = False,
        num_step_grid: int = 128,
        boundary_policy: Literal["hard", "st"] = "hard",
    ) -> None:
        """Initialize a VineCop object.

        Args:
            num_dim (int): number of dimensions.
            is_cop_scale (bool, optional): if True, the marginals are assumed in copula scale [0,1]. Otherwise, the marginals are fitted via KDE. Defaults to False.
            num_step_grid (int, optional): Grid resolution (power of 2) passed to each BiCop. Defaults to 128.
            boundary_policy (Literal["hard", "st"], optional): Query-time boundary handling policy
                propagated to all underlying pair-copula modules. Defaults to "hard".

        Attributes:
            num_dim (int):
            is_cop_scale (bool):
            num_step_grid (int):
            marginals (torch.nn.ModuleList): list of marginals for each dimension.
            bicops (torch.nn.ModuleDict): dictionary of BiCop
            struct_bcp (dict): BiCop structures on condition-ed/ing sets, independence, parents.
            struct_obs (list): Pseudo-obs structure per vine level.
            tree_bidep (list): Learned edge weights per vine level.
            sample_order (tuple): Sampling order for inverse Rosenblatt transform.
            num_obs (torch.Tensor): number of observations.
        """
        super().__init__()
        self.num_dim = num_dim
        self.is_cop_scale = is_cop_scale
        self.boundary_policy: Literal["hard", "st"] = boundary_policy
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
            self.bicops[cond_ed] = BiCop(
                num_step_grid=num_step_grid,
                boundary_policy=boundary_policy,
            )
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
        self.api_version = API_VERSION
        self.marginal_backend = "grid"
        self.marginal_backend_config = {}
        self.bicop_backend = "grid_reflect"
        self.bicop_backend_config = {}
        self.tree_bidep = [{} for _ in range(num_dim - 1)]
        self.num_step_grid = num_step_grid
        self.sample_order = tuple(_ for _ in range(num_dim))
        self.first_tree_vertex = tuple()
        self.register_buffer("num_obs", torch.zeros((), dtype=torch.int))
        # ! device agnostic
        self.register_buffer("_dd", torch.tensor([], dtype=torch.float64))
        self._artifact: VineBuildArtifact | None = None
        object.__setattr__(self, "engine", VineCopEngine(artifact=None))

    @property
    def device(self):
        """
        Device of internal buffers.
        """
        return self._dd.device

    @property
    def dtype(self):
        """
        Data type of internal buffers.
        """
        return self._dd.dtype

    @property
    def matrix(self) -> torch.Tensor:
        """Matrix representation of the vine.

        Diagonal elements form the sampling order. Read in row-wise: a row of `(0, 1, 3, 4, 2)`
        indicates a source vertex `(0|1,2,3,4)` and bicops `(0,2;)`, `(0,4;2)`, `(0,3;2,4)`, and
        `(0,1;2,3,4)`.

        Returns:
            torch.Tensor: Matrix representation of the vine.
        """
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
                        if v_diag in (v_l, v_r) and (v_l not in seen) and (v_r not in seen):
                            # * append the "other" variable
                            row.append(v_l if (v_diag == v_r) else v_r)
                seen.add(v_diag)
                mat.append(row)
            #
            mat.append([-1] * (self.num_dim - 1) + [int(self.sample_order[-1])])
            return torch.tensor(mat, dtype=torch.int, device=self.device)

    @torch.no_grad()
    def reset(self) -> None:
        """
        Reset the VineCop object to its initial state.
        """
        self.num_obs.zero_()
        self.marginal_backend = "grid"
        self.marginal_backend_config = {}
        self.bicop_backend = "grid_reflect"
        self.bicop_backend_config = {}
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
            bicop.boundary_policy = self.boundary_policy
            bicop.reset()
        self._artifact = None
        self._refresh_engine()

    def get_extra_state(self) -> dict:
        return {
            "api_version": self.api_version,
            "boundary_policy": self.boundary_policy,
            "marginal_backend": self.marginal_backend,
            "marginal_backend_config": dict(self.marginal_backend_config),
            "bicop_backend": self.bicop_backend,
            "bicop_backend_config": dict(self.bicop_backend_config),
            "sample_order": list(self.sample_order),
            "first_tree_vertex": list(self.first_tree_vertex),
            "tree_bidep": [
                [
                    {
                        "edge": list(edge),
                        "weight": float(torch.as_tensor(weight).reshape(-1)[0].item()),
                    }
                    for edge, weight in sorted(tree.items())
                ]
                for tree in self.tree_bidep
            ],
            "struct_obs": [
                [
                    {
                        "vertex_set": list(v_s),
                        "cond_ed": cond_ed,
                    }
                    for v_s, cond_ed in sorted(level.items())
                ]
                for level in self.struct_obs
            ],
            "struct_bcp": {
                cond_ed: {
                    "cond_ing": list(dd["cond_ing"]),
                    "is_indep": bool(dd["is_indep"]),
                    "left": dd["left"],
                    "right": dd["right"],
                }
                for cond_ed, dd in self.struct_bcp.items()
            },
        }

    def set_extra_state(self, state: dict) -> None:
        if not state:
            return
        self.api_version = state.get("api_version", self.api_version)
        self.boundary_policy = state.get("boundary_policy", self.boundary_policy)
        self.marginal_backend = state.get("marginal_backend", self.marginal_backend)
        self.marginal_backend_config = dict(state.get("marginal_backend_config", {}))
        self.bicop_backend = state.get("bicop_backend", self.bicop_backend)
        self.bicop_backend_config = dict(state.get("bicop_backend_config", {}))
        for bicop in self.bicops.values():
            bicop.boundary_policy = self.boundary_policy
        if "sample_order" in state:
            self.sample_order = tuple(int(v) for v in state["sample_order"])
        if "first_tree_vertex" in state:
            self.first_tree_vertex = tuple(int(v) for v in state["first_tree_vertex"])
        tree_bidep = state.get("tree_bidep")
        if tree_bidep is not None:
            restored_tree_bidep = []
            for level in tree_bidep:
                restored_level = {}
                for item in level:
                    restored_level[tuple(int(v) for v in item["edge"])] = float(item["weight"])
                restored_tree_bidep.append(restored_level)
            if len(restored_tree_bidep) == self.num_dim - 1:
                self.tree_bidep = restored_tree_bidep
        struct_obs = state.get("struct_obs")
        if struct_obs is not None:
            restored_struct_obs = []
            for level in struct_obs:
                restored_level = {}
                for item in level:
                    restored_level[tuple(int(v) for v in item["vertex_set"])] = item["cond_ed"]
                restored_struct_obs.append(restored_level)
            if len(restored_struct_obs) == self.num_dim:
                self.struct_obs = restored_struct_obs
        struct_bcp = state.get("struct_bcp")
        if struct_bcp is not None:
            for cond_ed, payload in struct_bcp.items():
                if cond_ed not in self.struct_bcp:
                    continue
                self.struct_bcp[cond_ed]["cond_ing"] = tuple(int(v) for v in payload.get("cond_ing", ()))
                self.struct_bcp[cond_ed]["is_indep"] = bool(payload.get("is_indep", True))
                self.struct_bcp[cond_ed]["left"] = payload.get("left")
                self.struct_bcp[cond_ed]["right"] = payload.get("right")
        self._refresh_engine()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        needs_top_level_state = "_extra_state" not in state_dict
        missing_bicop_state = [
            f"bicops.{cond_ed}._extra_state"
            for cond_ed in self.bicops
            if f"bicops.{cond_ed}._extra_state" not in state_dict
        ]
        marginal_backend = "grid"
        top_level_state = state_dict.get("_extra_state")
        if isinstance(top_level_state, dict):
            marginal_backend = str(top_level_state.get("marginal_backend", marginal_backend))
        for idx in range(self.num_dim):
            if self.marginals[idx] is None and f"marginals.{idx}.grid_x" in state_dict:
                self.marginals[idx] = build_marginal_shell(backend_name=marginal_backend)
        if needs_top_level_state or missing_bicop_state:
            patched_state = OrderedDict(state_dict)
            if needs_top_level_state:
                patched_state["_extra_state"] = {
                    "api_version": self.api_version,
                    "marginal_backend": "grid",
                    "marginal_backend_config": {},
                    "bicop_backend": "grid_reflect",
                    "bicop_backend_config": {},
                }
            for key in missing_bicop_state:
                patched_state[key] = {
                    "api_version": self.api_version,
                    "backend_name": "grid_reflect",
                    "normalized_backend_config": {},
                }
            state_dict = patched_state
        out = super().load_state_dict(state_dict, strict=strict, assign=assign)
        try:
            self._artifact = self._export_artifact()
        except Exception:
            self._artifact = None
        self._refresh_engine()
        return out

    def _refresh_engine(self, artifact: VineBuildArtifact | None = None) -> None:
        object.__setattr__(self, "engine", VineCopEngine(artifact=artifact or self._artifact))

    def _build_level_edge_tensors(self) -> tuple[torch.Tensor, ...]:
        edge_tensors = []
        for level in self.tree_bidep:
            edges = list(level)
            if edges:
                edge_tensors.append(torch.tensor(edges, dtype=torch.int, device=self.device))
            else:
                edge_tensors.append(torch.empty((0, 0), dtype=torch.int, device=self.device))
        return tuple(edge_tensors)

    def _bicop_order(self) -> tuple[str, ...]:
        return tuple(sorted(self.bicops.keys(), key=lambda key: tuple(int(part) for part in key.split(","))))

    def _export_artifact(self) -> VineBuildArtifact:
        vertex_order, slot_by_vertex, base_slots = _build_vertex_slots(
            num_dim=self.num_dim,
            struct_obs=self.struct_obs,
            device=self.device,
        )
        bicop_order = self._bicop_order()
        bicop_id_by_cond_ed = {cond_ed: idx for idx, cond_ed in enumerate(bicop_order)}
        (
            forward_output_slots,
            forward_left_slots,
            forward_right_slots,
            forward_source_slots,
            forward_bicop_ids,
            forward_modes,
        ) = _build_forward_plan(
            struct_obs=self.struct_obs,
            struct_bcp=self.struct_bcp,
            bicops=self.bicops,
            slot_by_vertex=slot_by_vertex,
            bicop_id_by_cond_ed=bicop_id_by_cond_ed,
            device=self.device,
        )
        logpdf_left_slots, logpdf_right_slots, logpdf_bicop_ids = _build_logpdf_plan(
            tree_bidep=self.tree_bidep,
            slot_by_vertex=slot_by_vertex,
            bicop_id_by_cond_ed=bicop_id_by_cond_ed,
            device=self.device,
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
            struct_obs=self.struct_obs,
            struct_bcp=self.struct_bcp,
            bicops=self.bicops,
            slot_by_vertex=slot_by_vertex,
            bicop_id_by_cond_ed=bicop_id_by_cond_ed,
            sample_order=self.sample_order,
            device=self.device,
        )
        return VineBuildArtifact(
            num_dim=self.num_dim,
            is_cop_scale=self.is_cop_scale,
            num_step_grid=self.num_step_grid,
            boundary_policy=self.boundary_policy,
            marginals=self.marginals,
            bicops=self.bicops,
            struct_bcp=copy.deepcopy(self.struct_bcp),
            struct_obs=copy.deepcopy(self.struct_obs),
            tree_bidep=copy.deepcopy(self.tree_bidep),
            sample_order=tuple(self.sample_order),
            first_tree_vertex=tuple(self.first_tree_vertex),
            marginal_backend=self.marginal_backend,
            marginal_backend_config=dict(self.marginal_backend_config),
            bicop_backend=self.bicop_backend,
            bicop_backend_config=dict(self.bicop_backend_config),
            mtd_bidep=self.mtd_bidep,
            num_obs=int(self.num_obs),
            level_edge_tensors=self._build_level_edge_tensors(),
            vertex_order=vertex_order,
            slot_by_vertex=slot_by_vertex,
            base_slots=base_slots,
            source_vertices=tuple(source_vertices),
            source_slots=source_slots,
            bicop_order=bicop_order,
            forward_output_slots=forward_output_slots,
            forward_left_slots=forward_left_slots,
            forward_right_slots=forward_right_slots,
            forward_source_slots=forward_source_slots,
            forward_bicop_ids=forward_bicop_ids,
            forward_modes=forward_modes,
            logpdf_left_slots=logpdf_left_slots,
            logpdf_right_slots=logpdf_right_slots,
            logpdf_bicop_ids=logpdf_bicop_ids,
            sample_output_slots=sample_output_slots,
            sample_left_slots=sample_left_slots,
            sample_right_slots=sample_right_slots,
            sample_bicop_ids=sample_bicop_ids,
            sample_modes=sample_modes,
        )

    def _install_artifact(self, artifact: VineBuildArtifact) -> None:
        self.num_dim = artifact.num_dim
        self.is_cop_scale = artifact.is_cop_scale
        self.num_step_grid = artifact.num_step_grid
        self.boundary_policy = artifact.boundary_policy
        self.marginals = artifact.marginals
        self.bicops = artifact.bicops
        for bicop in self.bicops.values():
            bicop.boundary_policy = artifact.boundary_policy
        self.struct_bcp = copy.deepcopy(artifact.struct_bcp)
        self.struct_obs = copy.deepcopy(artifact.struct_obs)
        self.tree_bidep = copy.deepcopy(artifact.tree_bidep)
        self.sample_order = tuple(artifact.sample_order)
        self.first_tree_vertex = tuple(artifact.first_tree_vertex)
        self.marginal_backend = artifact.marginal_backend
        self.marginal_backend_config = dict(artifact.marginal_backend_config)
        self.bicop_backend = artifact.bicop_backend
        self.bicop_backend_config = dict(artifact.bicop_backend_config)
        self.mtd_bidep = artifact.mtd_bidep
        self.num_obs.copy_(torch.tensor(artifact.num_obs, device=self.device, dtype=torch.int))
        self._artifact = artifact
        self._refresh_engine(artifact=artifact)

    @classmethod
    def from_artifact(cls, artifact: VineBuildArtifact) -> "VineCop":
        model = cls(
            num_dim=artifact.num_dim,
            is_cop_scale=artifact.is_cop_scale,
            num_step_grid=artifact.num_step_grid,
            boundary_policy=artifact.boundary_policy,
        )
        model._install_artifact(artifact)
        return model

    @torch.no_grad()
    def export_inference_plan(self, dtype: torch.dtype | None = None) -> VineBuildArtifact:
        artifact = copy.deepcopy(self._artifact if self._artifact is not None else self._export_artifact())
        if dtype is None:
            return artifact
        for marginal in artifact.marginals:
            if marginal is not None:
                marginal.to(device=self.device, dtype=dtype)
        for bicop in artifact.bicops.values():
            bicop.to(device=self.device, dtype=dtype)
        return artifact

    @torch.no_grad()
    def diagnostics(self) -> VineDiagnostics:
        max_err = 0.0
        itp_failures = 0
        bisect_refinements = 0
        fallback_to_indep = 0
        for bicop in self.bicops.values():
            diag = bicop.diagnostics()
            itp_failures += diag.itp_failures_l + diag.itp_failures_r
            bisect_refinements += diag.bisect_refinements_l + diag.bisect_refinements_r
            fallback_to_indep += diag.fallback_to_indep_l + diag.fallback_to_indep_r
            max_err = max(max_err, diag.max_abs_hfunc_error_l, diag.max_abs_hfunc_error_r)
        return VineDiagnostics(
            num_edges=len(self.bicops),
            itp_failures=itp_failures,
            bisect_refinements=bisect_refinements,
            fallback_to_indep=fallback_to_indep,
            max_abs_hfunc_error=max_err,
        )

    def _apply(self, fn):
        out = super()._apply(fn)
        if self._artifact is not None:
            try:
                self._artifact = self._export_artifact()
            except Exception:
                self._artifact = None
        self._refresh_engine(self._artifact)
        return out

    def _normalize_fit_request(
        self,
        *,
        marginal_backend: str,
        marginal_kwargs: dict | None,
        bicop_backend: str | None,
        bicop_kwargs: dict | None,
        mtd_kde: str | None,
        mtd_tll: str,
        num_iter_max: int,
        num_step_grid_kde1d: int | None,
        bandwidth: str | float | torch.Tensor,
        bandwidth_scale: float,
        kde_kwargs: dict,
    ) -> tuple[str, dict, str, dict]:
        marginal_backend = normalize_marginal_backend(marginal_backend)
        normalized_marginal_kwargs = dict(marginal_kwargs or {})
        if kde_kwargs:
            warnings.warn(
                "Legacy keyword args passed through '**kde_kwargs' are deprecated; use 'marginal_kwargs' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            normalized_marginal_kwargs = {**kde_kwargs, **normalized_marginal_kwargs}
        if marginal_backend == "grid":
            if "bandwidth" not in normalized_marginal_kwargs:
                normalized_marginal_kwargs["bandwidth"] = bandwidth
            elif bandwidth != "isj":
                warnings.warn(
                    "Both 'marginal_kwargs[\"bandwidth\"]' and legacy top-level 'bandwidth' were provided; "
                    "using marginal_kwargs.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if "bandwidth_scale" not in normalized_marginal_kwargs:
                normalized_marginal_kwargs["bandwidth_scale"] = bandwidth_scale
            elif bandwidth_scale != 1.0:
                warnings.warn(
                    "Both 'marginal_kwargs[\"bandwidth_scale\"]' and legacy top-level "
                    "'bandwidth_scale' were provided; using marginal_kwargs.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if "num_step_grid" not in normalized_marginal_kwargs:
                normalized_marginal_kwargs["num_step_grid"] = num_step_grid_kde1d
            elif num_step_grid_kde1d is not None:
                warnings.warn(
                    "Both 'marginal_kwargs[\"num_step_grid\"]' and legacy 'num_step_grid_kde1d' were "
                    "provided; using marginal_kwargs.",
                    DeprecationWarning,
                    stacklevel=3,
                )

        if bicop_backend is None:
            if mtd_kde is None:
                bicop_backend = "grid_reflect"
            else:
                warnings.warn(
                    "'mtd_kde' is deprecated; use 'bicop_backend' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                bicop_backend = {
                    "fastKDE": "grid_reflect",
                    "torch_grid": "grid_reflect",
                    "tll": "tll_ref",
                }.get(mtd_kde, mtd_kde)
        bicop_backend = normalize_bicop_backend(bicop_backend)
        normalized_bicop_kwargs = dict(bicop_kwargs or {})
        if bicop_backend == "tll_ref":
            if "nonparametric_method" not in normalized_bicop_kwargs:
                normalized_bicop_kwargs["nonparametric_method"] = mtd_tll
            elif mtd_tll != "constant":
                warnings.warn(
                    "Both 'bicop_kwargs[\"nonparametric_method\"]' and legacy 'mtd_tll' were provided; "
                    "using bicop_kwargs.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        else:
            if "bandwidth" not in normalized_bicop_kwargs and bandwidth != "isj":
                normalized_bicop_kwargs["bandwidth"] = bandwidth
            if "bandwidth_scale" not in normalized_bicop_kwargs:
                normalized_bicop_kwargs["bandwidth_scale"] = bandwidth_scale
            elif bandwidth_scale != 1.0:
                warnings.warn(
                    "Both 'bicop_kwargs[\"bandwidth_scale\"]' and legacy top-level "
                    "'bandwidth_scale' were provided; using bicop_kwargs.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if "num_iter_max" not in normalized_bicop_kwargs:
                normalized_bicop_kwargs["num_iter_max"] = num_iter_max
            elif num_iter_max != 5:
                warnings.warn(
                    "Both 'bicop_kwargs[\"num_iter_max\"]' and legacy top-level 'num_iter_max' were "
                    "provided; using bicop_kwargs.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        return marginal_backend, normalized_marginal_kwargs, bicop_backend, normalized_bicop_kwargs

    @staticmethod
    @torch.no_grad()
    def ref_count_hfunc(
        num_dim: int, struct_obs: list, sample_order: tuple
    ) -> tuple[dict, list, int]:
        """Count references of pseudo-obs, identify source vertices, and count number of hfuncs.

        Args:
            num_dim (int): number of dimensions in the vine.
            struct_obs (list): structure of pseudo observations. (parents)
            sample_order (tuple): sampling order.

        Returns:
            tuple[dict, list, int]: reference counts of pseudo-obs, list of source vertices, and number of hfuncs.
        """
        return _ref_count_hfunc_impl(num_dim=num_dim, struct_obs=struct_obs, sample_order=sample_order)

    @torch.no_grad()
    def _sample_order(self) -> tuple[int, ...]:
        """Schedule an optimized sampling order to minimize h‐function calls.

        Returns:
            tuple[int, ...]: New ``sample_order`` of variable indices for inverse Rosenblatt sampling.
        """
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
        edge_list = [((v_l, v_r), -abs(w)) for (v_l, v_r, *s), w in edge_weight_lv.items()]
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
            """
            Path compression.
            """
            if parent[v] != v:
                parent[v] = find(parent[v])
            return parent[v]

        def union(x, y):
            """
            Union by rank.
            """
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
                heap_bidep_abs = [(-abs(bidep), lr_s) for lr_s, bidep in cand_e_w.items()]
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
        thresh_trunc: None | float = 0.01,
        mtd_kde: str | None = None,
        mtd_tll: str = "constant",
        num_iter_max: int = 5,
        is_tau_est: bool = False,
        num_step_grid_kde1d: int = None,
        marginal_backend: str = "grid",
        marginal_kwargs: dict | None = None,
        bicop_backend: str | None = None,
        bicop_kwargs: dict | None = None,
        bandwidth: str | float | torch.Tensor = "isj",
        bandwidth_scale: float = 1.0,
        generator: torch.Generator | None = None,
        bidep_backend: Literal["auto", "scipy", "torch"] = "auto",
        **kde_kwargs,
    ) -> None:
        artifact = VineBuilder(self).build(
            obs=obs,
            is_dissmann=is_dissmann,
            matrix=matrix,
            first_tree_vertex=first_tree_vertex,
            mtd_vine=mtd_vine,
            mtd_bidep=mtd_bidep,
            thresh_trunc=thresh_trunc,
            mtd_kde=mtd_kde,
            mtd_tll=mtd_tll,
            num_iter_max=num_iter_max,
            is_tau_est=is_tau_est,
            num_step_grid_kde1d=num_step_grid_kde1d,
            marginal_backend=marginal_backend,
            marginal_kwargs=marginal_kwargs,
            bicop_backend=bicop_backend,
            bicop_kwargs=bicop_kwargs,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            generator=generator,
            bidep_backend=bidep_backend,
            **kde_kwargs,
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
        mtd_kde: str | None = None,
        mtd_tll: str = "constant",
        num_iter_max: int = 5,
        is_tau_est: bool = False,
        num_step_grid_kde1d: int = None,
        marginal_backend: str = "grid",
        marginal_kwargs: dict | None = None,
        bicop_backend: str | None = None,
        bicop_kwargs: dict | None = None,
        bandwidth: str | float | torch.Tensor = "isj",
        bandwidth_scale: float = 1.0,
        generator: torch.Generator | None = None,
        bidep_backend: Literal["auto", "scipy", "torch"] = "auto",
        **kde_kwargs,
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
            mtd_kde=mtd_kde,
            mtd_tll=mtd_tll,
            num_iter_max=num_iter_max,
            num_step_grid_kde1d=num_step_grid_kde1d,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            kde_kwargs=kde_kwargs,
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
                        mtd_kde=None,
                        mtd_tll=mtd_tll,
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

    def log_pdf(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the log-density function of the vine copula at the given observations.

        The input may include marginal-scale observations when the model was fitted with
        `is_cop_scale=False`.

        Args:
            obs (torch.Tensor): Points at which to evaluate the log-density function. Shape (num_obs, num_dim).

        Returns:
            torch.Tensor: Log-density function values at the given observations. Shape (num_obs, 1).
        """
        if self.engine.artifact is None:
            raise RuntimeError("log_pdf() requires a fitted model or installed execution plan.")
        return self.engine.log_pdf(obs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neg average log-likelihood function for the given observations.

        Args:
            x (torch.Tensor): Points at which to evaluate. Shape (num_obs, num_dim).

        Returns:
            torch.Tensor: scalar loss value.
        """
        return -self.log_pdf(x).mean()

    def rosenblatt(self, obs: torch.Tensor, sample_order: tuple | None = None) -> torch.Tensor:
        """Compute the Rosenblatt transform of observations.

        Maps input `obs` into uniform pseudo‐observations via successive conditional CDFs (Rosenblatt).
        """
        if self.engine.artifact is not None:
            return self.engine.rosenblatt(obs=obs, sample_order=sample_order)
        raise RuntimeError("rosenblatt() requires a fitted model or installed execution plan.")

    @torch.no_grad()
    def inverse_rosenblatt(
        self,
        obs: torch.Tensor,
        sample_order: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if self.engine.artifact is not None:
            return self.engine.inverse_rosenblatt(obs=obs, sample_order=sample_order)
        raise RuntimeError("inverse_rosenblatt() requires a fitted model or installed execution plan.")

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
        if self.engine.artifact is None:
            raise RuntimeError("_sample_u() requires a fitted model or installed execution plan.")
        return self.engine._sample_u(
            num_sample=num_sample,
            seed=seed,
            is_sobol=is_sobol,
            sample_order=sample_order,
            dct_v_s_obs=dct_v_s_obs,
            generator=generator,
        )

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
        if self.engine.artifact is None:
            raise RuntimeError("sample() requires a fitted model or installed execution plan.")
        return self.engine.sample(
            num_sample=num_sample,
            seed=seed,
            is_sobol=is_sobol,
            sample_order=sample_order,
            dct_v_s_obs=dct_v_s_obs,
            generator=generator,
        )

    @torch.no_grad()
    def cdf(self, obs: torch.Tensor, num_sample: int = 10007, seed: int = 42) -> torch.Tensor:
        if self.engine.artifact is None:
            raise RuntimeError("cdf() requires a fitted model or installed execution plan.")
        return self.engine.cdf(obs=obs, num_sample=num_sample, seed=seed)

    def __str__(self) -> str:
        """String representation of the ``VineCop`` object.

        Returns:
            str: String representation of the ``VineCop`` object.
        """
        header = self.__class__.__name__
        params = {
            "num_dim": int(self.num_dim),
            "num_obs": int(self.num_obs),
            "is_cop_scale": self.is_cop_scale,
            "mtd_bidep": self.mtd_bidep,
            "marginal_backend": self.marginal_backend,
            "bicop_backend": self.bicop_backend,
            "boundary_policy": self.boundary_policy,
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
    ) -> tuple:
        """Draw the weighted undirected graph at a single level of the vine copula.

        This constructs a NetworkX graph of bivariate-copula edges at level `lv`, where nodes
        represent either raw variables (`lv=0`), parent-copula modules, or pseudo-observations.
        Edge widths encode dependence strength.

        Args:
            lv (int, optional): Level to draw. Defaults to 0.
            is_bcp (bool, optional): If True, nodes are parent‐bicop "l,r;s". Otherwise, nodes are pseudo‐obs "v|s". Defaults to True.
            title (str | None, optional): Title of the plot. Defaults to ``f"Vine level {lv}"``.
            num_digit (int, optional): Number of decimal digits for edge weights. Defaults to 2.
            font_size_vertex (int, optional): Font size for vertex labels. Defaults to 8.
            font_size_edge (int, optional): Font size for edge labels. Defaults to 7.
            f_path (Path, optional): Path to save the figure. Defaults to None.
            fig_size (tuple, optional): Figure size. Defaults to None.

        Raises:
            ImportError: If matplotlib or networkx is not installed.

        Returns:
            tuple: Figure, axis, graph object, and file path (if saved).
        """
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
                label_u = f"""{label_u};{sep}{
                    ",".join(str(x) for x in sorted(self.struct_bcp[label_u]["cond_ing"]))
                }"""
                label_v = f"""{label_v};{sep}{
                    ",".join(str(x) for x in sorted(self.struct_bcp[label_v]["cond_ing"]))
                }"""
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
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size_vertex, font_color="black")
        # * scale line‐width by weight
        widths = [math.log1p(0.5 + 100 * abs(data["weight"])) for _, _, data in G.edges(data=True)]
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
    ) -> tuple:
        """Draw the computational graph (DAG) of the vine copula.

        This creates a directed graph where edges flow from upstream pseudo-observations and
        pair-copula modules to downstream pseudo-observations, laid out by vine level.

        Args:
            sample_order (tuple[int, ...], optional): Variable sampling order. Defaults to `self.sample_order`.
            title (str, optional): Title of the plot. Defaults to "Vine comp graph".
            font_size_vertex (int, optional): Font size for vertex labels. Defaults to 8.
            f_path (Path, optional): Path to save the figure. If provided, the figure will be saved. Defaults to None.
            fig_size (tuple, optional): Figure size. Defaults to None.

        Raises:
            ImportError: If matplotlib or networkx is not installed.

        Returns:
            tuple: Figure, axis, graph object, and file path (if saved).
        """
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
            # * traverse around the bcp; sorted!
            for v_l, v_r, *cond in sorted(self.tree_bidep[lv]):
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
                xs = np.linspace(-len(downstream) / 2, len(downstream) / 2, len(downstream))
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
            sample_order=sample_order if sample_order is not None else self.sample_order,
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
