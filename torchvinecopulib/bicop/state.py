from __future__ import annotations

from typing import Any

import torch

from ..backends import DEFAULT_BICOP_BACKEND

__all__ = ["BiCopStateMixin"]


class BiCopStateMixin:
    def get_extra_state(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "backend_name": self.bicop_backend,
            "normalized_backend_config": dict(self.backend_config),
            "boundary_policy": self.boundary_policy,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self.api_version = state.get("api_version", self.api_version)
        backend_name = state.get("backend_name")
        if backend_name:
            self.bicop_backend = backend_name
        self.backend_config = dict(state.get("normalized_backend_config", {}))
        self.boundary_policy = state.get("boundary_policy", self.boundary_policy)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in ("_pdf_grid", "_cdf_grid", "_hfunc_l_grid", "_hfunc_r_grid"):
            key = prefix + name
            if key in state_dict:
                tensor = state_dict[key]
                if self._buffers[name].shape != tensor.shape:
                    self._buffers[name] = torch.empty_like(tensor)
        key = prefix + "bandwidth"
        if key in state_dict and self.bandwidth.shape != state_dict[key].shape:
            self._buffers["bandwidth"] = torch.empty_like(state_dict[key])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._ensure_geometry()
        self.is_indep = self._pdf_grid.numel() == 0

    def reset(self) -> None:
        self.is_indep = True
        self.bicop_backend = DEFAULT_BICOP_BACKEND
        self.backend_config = {}
        self._ensure_geometry()
        self.tau.zero_()
        self.num_obs.zero_()
        self.negloglik.zero_()
        self.hinv_fallback_l.zero_()
        self.hinv_fallback_r.zero_()
        self.hinv_bisect_l.zero_()
        self.hinv_bisect_r.zero_()
        self.fallback_to_indep_l.zero_()
        self.fallback_to_indep_r.zero_()
        self.max_abs_hfunc_error_l.zero_()
        self.max_abs_hfunc_error_r.zero_()
        self.bandwidth.zero_()
        self._pdf_grid = self._empty_grid()
        self._cdf_grid = self._empty_grid()
        self._hfunc_l_grid = self._empty_grid()
        self._hfunc_r_grid = self._empty_grid()
        self.last_hinv_status_l = None
        self.last_hinv_status_r = None
