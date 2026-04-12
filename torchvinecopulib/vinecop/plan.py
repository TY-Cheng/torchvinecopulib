from __future__ import annotations

from typing import Any

import torch

from .structure import _ref_count_hfunc_impl

_FORWARD_COPY = 0
_FORWARD_HFUNC_L = 1
_FORWARD_HFUNC_R = 2

_SAMPLE_COPY = 0
_SAMPLE_HFUNC_L = 1
_SAMPLE_HFUNC_R = 2
_SAMPLE_HINV_L = 3
_SAMPLE_HINV_R = 4

__all__ = [
    "_FORWARD_COPY",
    "_FORWARD_HFUNC_L",
    "_FORWARD_HFUNC_R",
    "_SAMPLE_COPY",
    "_SAMPLE_HFUNC_L",
    "_SAMPLE_HFUNC_R",
    "_SAMPLE_HINV_L",
    "_SAMPLE_HINV_R",
    "_canonical_vertex",
    "_tensor_int",
    "_build_vertex_slots",
    "_build_forward_plan",
    "_build_logpdf_plan",
    "_build_sample_plan",
]


def _canonical_vertex(v_s: tuple[int, ...]) -> tuple[int, ...]:
    v, *s = v_s
    return (int(v), *sorted(int(x) for x in s))


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
            [(v_l if is_down_right else v_r, *s_up)] if is_hinv else [(v_l, *s_up), (v_r, *s_up)]
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
