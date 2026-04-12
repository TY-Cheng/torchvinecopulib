from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

__all__ = [
    "VineDiagnostics",
    "VineBuildArtifact",
    "_device_from_artifact",
    "_dtype_from_artifact",
]


@dataclass
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
