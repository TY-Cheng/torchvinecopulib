from __future__ import annotations

import copy
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch

from ..backends import DEFAULT_BICOP_BACKEND, build_marginal_shell
from .artifact import VineBuildArtifact, VineDiagnostics
from .engine import VineCopEngine
from .plan import _build_forward_plan, _build_logpdf_plan, _build_sample_plan, _build_vertex_slots

__all__ = ["VineCopStateMixin"]

if TYPE_CHECKING:
    from .model import VineCop


class VineCopStateMixin:
    @torch.no_grad()
    def reset(self) -> None:
        """
        Reset the VineCop object to its initial state.
        """
        self.num_obs.zero_()
        self.marginal_backend = "grid"
        self.marginal_backend_config = {}
        self.bicop_backend = DEFAULT_BICOP_BACKEND
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
                self.struct_bcp[cond_ed]["cond_ing"] = tuple(
                    int(v) for v in payload.get("cond_ing", ())
                )
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
        return tuple(
            sorted(self.bicops.keys(), key=lambda key: tuple(int(part) for part in key.split(",")))
        )

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
    def from_artifact(cls, artifact: VineBuildArtifact) -> VineCop:
        model = cls(
            num_dim=artifact.num_dim,
            is_cop_scale=artifact.is_cop_scale,
            num_step_grid=artifact.num_step_grid,
            boundary_policy=artifact.boundary_policy,
        )
        model._install_artifact(artifact)
        return model

    def export_inference_plan(self, dtype: torch.dtype | None = None) -> VineBuildArtifact:
        artifact = copy.deepcopy(
            self._artifact if self._artifact is not None else self._export_artifact()
        )
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
