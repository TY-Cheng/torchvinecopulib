from __future__ import annotations

from pprint import pformat
from typing import Any, Literal

import torch

from ..backends import (
    API_VERSION,
    DEFAULT_BICOP_BACKEND,
    GridReflectBicopEstimator,
    build_bicop_estimator,
)
from ..backends.common import _EPS
from ..util import kendall_tau
from .fit_request import BiCopFitRequestMixin
from .plot import BiCopPlotMixin
from .query import BiCopQueryMixin
from .state import BiCopStateMixin

__all__ = ["BiCop", "GridReflectBicopEstimator"]


class BiCop(
    BiCopFitRequestMixin,
    BiCopStateMixin,
    BiCopQueryMixin,
    BiCopPlotMixin,
    torch.nn.Module,
):
    """Continuous bivariate copula model backed by regular-grid buffers.

    The fitted object stores a copula density grid together with cumulative buffers for `cdf()`,
    conditional CDF evaluation, and stabilized inverse-conditional queries. Build-time estimation is
    executed under `torch.no_grad()`, whereas forward query methods such as `pdf()`, `log_pdf()`,
    `cdf()`, and `hfunc_*()` remain torch-native runtime operators.

    Args:
        num_step_grid: Number of grid points per axis used by the stored copula surface.
        boundary_policy: Query-time boundary handling. `"hard"` applies a standard clamp;
            `"st"` uses a straight-through clamp so the forward pass stays in-domain while the
            backward pass preserves the interior gradient signal.
    """

    _EPS: float = _EPS

    def __init__(
        self,
        num_step_grid: int = 128,
        boundary_policy: Literal["hard", "st"] = "hard",
    ):
        super().__init__()
        self.api_version = API_VERSION
        self.is_indep = True
        self.bicop_backend = DEFAULT_BICOP_BACKEND
        self.backend_config: dict[str, Any] = {}
        self.boundary_policy: Literal["hard", "st"] = boundary_policy
        self.num_step_grid = int(num_step_grid)
        self._target = float(self.num_step_grid - 1)
        self.step_grid = 1.0 / max(self._target, 1.0)
        self.register_buffer("tau", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("num_obs", torch.zeros((), dtype=torch.int))
        self.register_buffer("negloglik", torch.zeros((), dtype=torch.float64))
        self.register_buffer("hinv_fallback_l", torch.zeros((), dtype=torch.int64))
        self.register_buffer("hinv_fallback_r", torch.zeros((), dtype=torch.int64))
        self.register_buffer("hinv_bisect_l", torch.zeros((), dtype=torch.int64))
        self.register_buffer("hinv_bisect_r", torch.zeros((), dtype=torch.int64))
        self.register_buffer("fallback_to_indep_l", torch.zeros((), dtype=torch.int64))
        self.register_buffer("fallback_to_indep_r", torch.zeros((), dtype=torch.int64))
        self.register_buffer("max_abs_hfunc_error_l", torch.zeros((), dtype=torch.float64))
        self.register_buffer("max_abs_hfunc_error_r", torch.zeros((), dtype=torch.float64))
        self.register_buffer("bandwidth", torch.empty(2, dtype=torch.float64))
        self.register_buffer("_pdf_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_cdf_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_hfunc_l_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_hfunc_r_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_dd", torch.tensor([], dtype=torch.float64))
        self.last_hinv_status_l: dict[str, torch.Tensor] | None = None
        self.last_hinv_status_r: dict[str, torch.Tensor] | None = None

    @property
    def device(self) -> torch.device:
        return self._dd.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dd.dtype

    def _empty_grid(self) -> torch.Tensor:
        return torch.empty(0, 0, device=self.device, dtype=self.dtype)

    def _ensure_geometry(self) -> None:
        if self._pdf_grid.numel() > 0:
            self.num_step_grid = int(self._pdf_grid.shape[0])
        self._target = float(self.num_step_grid - 1)
        self.step_grid = 1.0 / max(self._target, 1.0)

    def _assign_fit_bundle(self, estimator: torch.nn.Module) -> None:
        self._pdf_grid = estimator._pdf_grid.to(device=self.device, dtype=self.dtype)
        self._cdf_grid = estimator._cdf_grid.to(device=self.device, dtype=self.dtype)
        self._hfunc_l_grid = estimator._hfunc_l_grid.to(device=self.device, dtype=self.dtype)
        self._hfunc_r_grid = estimator._hfunc_r_grid.to(device=self.device, dtype=self.dtype)
        self.bandwidth = estimator.bandwidth.to(device=self.device, dtype=self.dtype)
        self.bicop_backend = estimator.backend_name
        self.backend_config = dict(estimator.backend_config)
        self._ensure_geometry()

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        num_iter_max: int = 2000,
        is_tau_est: bool = False,
        *,
        bicop_backend: str | None = None,
        bicop_kwargs: dict[str, Any] | None = None,
        bandwidth: str | float | torch.Tensor = "silverman",
        generator: torch.Generator | None = None,
    ) -> None:
        """Fit a nonparametric bivariate copula from observations on the unit square.

        Args:
            obs: Pseudo-observations with shape `(num_obs, 2)` and values in `[0, 1]`.
            num_iter_max: Maximum number of copula-margin renormalization iterations forwarded to
                the selected backend.
            is_tau_est: Whether to estimate and store Kendall's tau from `obs`.
            bicop_backend: Backend name. If `None`, the current package default is used.
            bicop_kwargs: Backend-specific keyword arguments after canonical normalization.
            bandwidth: Legacy convenience argument forwarded through the fit-request normalizer.
            generator: Reserved for API compatibility. It is currently unused by the builder path.
        """
        del generator
        device, dtype = self.device, self.dtype
        obs = obs.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
        normalized_backend, normalized_kwargs = self._normalize_fit_request(
            bicop_backend=bicop_backend,
            bicop_kwargs=bicop_kwargs,
            num_iter_max=num_iter_max,
            bandwidth=bandwidth,
        )
        self.reset()
        self.is_indep = False
        self.num_obs.copy_(torch.tensor(obs.shape[0], device=device, dtype=torch.int))
        if is_tau_est:
            self.tau.copy_(kendall_tau(obs[:, [0]], obs[:, [1]]).to(device=device, dtype=dtype))
        estimator = build_bicop_estimator(
            backend_name=normalized_backend,
            obs=obs.to(dtype=torch.float64),
            num_step_grid=self.num_step_grid,
            backend_kwargs=normalized_kwargs,
        )
        self._assign_fit_bundle(estimator)
        self.negloglik.copy_(-self.log_pdf(obs=obs).sum())

    def __str__(self) -> str:
        return f"""{self.__class__.__name__}\n{
            pformat(
                object={
                    "is_indep": self.is_indep,
                    "num_obs": int(self.num_obs),
                    "negloglik": self.negloglik.round(decimals=4),
                    "num_step_grid": self.num_step_grid,
                    "tau": self.tau.round(decimals=4),
                    "bicop_backend": self.bicop_backend,
                    "backend_config": self.backend_config,
                    "boundary_policy": self.boundary_policy,
                    "bandwidth": self.bandwidth.round(decimals=6),
                    "hinv_fallback_l": int(self.hinv_fallback_l),
                    "hinv_fallback_r": int(self.hinv_fallback_r),
                    "hinv_bisect_l": int(self.hinv_bisect_l),
                    "hinv_bisect_r": int(self.hinv_bisect_r),
                    "fallback_to_indep_l": int(self.fallback_to_indep_l),
                    "fallback_to_indep_r": int(self.fallback_to_indep_r),
                    "max_abs_hfunc_error_l": float(self.max_abs_hfunc_error_l),
                    "max_abs_hfunc_error_r": float(self.max_abs_hfunc_error_r),
                    "dtype": self._dd.dtype,
                    "device": self._dd.device,
                },
                compact=True,
                sort_dicts=False,
                underscore_numbers=True,
            )
        }"""
