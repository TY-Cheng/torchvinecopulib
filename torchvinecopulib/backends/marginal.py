from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pprint import pformat
from typing import Any

import torch

from .common import (
    _EPS,
    API_VERSION,
    MarginalFitResult,
    _build_pdf_buffers_1d,
    _isj_bandwidth_1d,
    _linear_bin_1d,
    _next_power_of_two_plus_one,
    _normalize_backend_kwargs,
    _smooth_1d,
    _silverman_bandwidth_1d,
)

__all__ = [
    "BaseMarginal1D",
    "GridKDE1D",
    "LPRefMarginal1D",
    "TorchKDE1D",
    "build_marginal_estimator",
    "build_marginal_shell",
    "normalize_marginal_backend",
]


def _lazy_reference_kde():
    try:
        import pyvinecopulib as pv
    except ImportError:
        pv = None
    if pv is not None and hasattr(pv, "Kde1d"):
        return "pyvinecopulib", pv.Kde1d
    try:
        from scipy.stats import gaussian_kde
    except ImportError as exc:
        raise ImportError("lp_ref requires scipy or pyvinecopulib with Kde1d support.") from exc
    return "scipy", gaussian_kde


def normalize_marginal_backend(name: str) -> str:
    aliases = {
        "torch_grid": "grid",
        "grid": "grid",
        "lp_ref": "lp_ref",
    }
    if name not in aliases:
        raise ValueError("marginal_backend must be one of 'grid' or 'lp_ref'.")
    return aliases[name]


class BaseMarginal1D(torch.nn.Module, ABC):
    api_version = API_VERSION

    def __init__(self) -> None:
        super().__init__()
        self.backend_name = "unknown"
        self.backend_config: dict[str, Any] = {}
        self.bandwidth_method = "unknown"
        self.num_obs = 0
        self.x_min = 0.0
        self.x_max = 1.0
        self.register_buffer("grid_x", torch.empty(0, dtype=torch.float64))
        self.register_buffer("grid_pdf", torch.empty(0, dtype=torch.float64))
        self.register_buffer("grid_cdf", torch.empty(0, dtype=torch.float64))
        self.register_buffer("slope_pdf", torch.empty(0, dtype=torch.float64))
        self.register_buffer("slope_fwd", torch.empty(0, dtype=torch.float64))
        self.register_buffer("slope_inv", torch.empty(0, dtype=torch.float64))
        self.register_buffer("bandwidth", torch.empty((), dtype=torch.float64))
        self.register_buffer("negloglik", torch.zeros((), dtype=torch.float64))
        self.register_buffer("_dd", torch.tensor([], dtype=torch.float64))

    @property
    def device(self) -> torch.device:
        return self._dd.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dd.dtype

    def _assign_result(self, result: MarginalFitResult) -> None:
        self.grid_x = result.grid_x.to(device=self.device, dtype=self.dtype)
        self.grid_pdf = result.grid_pdf.to(device=self.device, dtype=self.dtype)
        self.grid_cdf = result.grid_cdf.to(device=self.device, dtype=self.dtype)
        self.slope_pdf = result.slope_pdf.to(device=self.device, dtype=self.dtype)
        self.slope_fwd = result.slope_fwd.to(device=self.device, dtype=self.dtype)
        self.slope_inv = result.slope_inv.to(device=self.device, dtype=self.dtype)
        self.bandwidth = result.bandwidth.to(device=self.device, dtype=self.dtype)
        self.backend_name = result.backend_name
        self.backend_config = result.backend_config
        self.bandwidth_method = str(result.backend_config.get("bandwidth", result.backend_name))
        self.x_min = float(result.x_min)
        self.x_max = float(result.x_max)
        self.num_obs = int(result.num_obs)

    @abstractmethod
    def fit(self, x: torch.Tensor, **kwargs: Any) -> None:
        raise NotImplementedError

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x_clamped = x.clamp(self.x_min, self.x_max)
        idx = torch.searchsorted(self.grid_x, x_clamped, right=False).clamp(1, self.grid_x.numel() - 1)
        y = self.grid_cdf[idx - 1] + self.slope_fwd[idx - 1] * (x_clamped - self.grid_x[idx - 1])
        y = torch.where(x < self.x_min, torch.zeros_like(y), y)
        y = torch.where(x > self.x_max, torch.ones_like(y), y)
        return y.clamp(0.0, 1.0)

    def ppf(self, q: torch.Tensor) -> torch.Tensor:
        q = q.to(device=self.device, dtype=self.dtype)
        q_clamped = q.clamp(0.0, 1.0)
        idx = torch.searchsorted(self.grid_cdf, q_clamped, right=False).clamp(1, self.grid_cdf.numel() - 1)
        x = self.grid_x[idx - 1] + self.slope_inv[idx - 1] * (q_clamped - self.grid_cdf[idx - 1])
        x = torch.where(q < 0.0, torch.full_like(x, self.x_min), x)
        x = torch.where(q > 1.0, torch.full_like(x, self.x_max), x)
        return x.clamp(self.x_min, self.x_max)

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x_clamped = x.clamp(self.x_min, self.x_max)
        idx = torch.searchsorted(self.grid_x, x_clamped, right=False).clamp(1, self.grid_pdf.numel() - 1)
        pdf = self.grid_pdf[idx - 1] + self.slope_pdf[idx - 1] * (x_clamped - self.grid_x[idx - 1])
        pdf = torch.where((x < self.x_min) | (x > self.x_max), torch.zeros_like(pdf), pdf)
        return pdf.clamp_min(0.0)

    def log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.pdf(x).clamp_min(_EPS).log()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_pdf(x).mean()

    def extra_state(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "backend_name": self.backend_name,
            "backend_config": dict(self.backend_config),
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "num_obs": int(self.num_obs),
        }

    def get_extra_state(self) -> dict[str, Any]:
        return self.extra_state()

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self.backend_name = state.get("backend_name", self.backend_name)
        self.backend_config = dict(state.get("backend_config", {}))
        self.bandwidth_method = str(self.backend_config.get("bandwidth", self.bandwidth_method))
        self.x_min = float(state.get("x_min", self.x_min))
        self.x_max = float(state.get("x_max", self.x_max))
        self.num_obs = int(state.get("num_obs", self.num_obs))

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
        for name in ("grid_x", "grid_pdf", "grid_cdf", "slope_pdf", "slope_fwd", "slope_inv"):
            key = prefix + name
            if key in state_dict and self._buffers[name].shape != state_dict[key].shape:
                self._buffers[name] = torch.empty_like(state_dict[key])
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
        if self.grid_x.numel():
            self.x_min = float(self.grid_x[0].item())
            self.x_max = float(self.grid_x[-1].item())

    def __str__(self) -> str:
        params = {
            "num_obs": int(self.num_obs),
            "negloglik": float(self.negloglik.round(decimals=4)),
            "x_min": float(round(self.x_min, 4)),
            "x_max": float(round(self.x_max, 4)),
            "num_step_grid": int(self.grid_x.numel()),
            "bandwidth": float(self.bandwidth.reshape(-1)[0].round(decimals=6)) if self.bandwidth.numel() else 0.0,
            "backend_name": self.backend_name,
            "backend_config": self.backend_config,
            "dtype": self.dtype,
            "device": self.device,
        }
        return f"{self.__class__.__name__}\n{pformat(params, sort_dicts=False, underscore_numbers=True)[1:-1]}\n"


class GridKDE1D(BaseMarginal1D):
    def __init__(
        self,
        x: torch.Tensor,
        *,
        num_step_grid: int | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        pad: float = 0.1,
        bandwidth: str | float | torch.Tensor = "isj",
        bandwidth_scale: float = 1.0,
        smoother: str = "auto",
    ) -> None:
        super().__init__()
        x = x.view(-1, 1).to(dtype=torch.float64)
        if num_step_grid is None:
            num_step_grid = _next_power_of_two_plus_one(x.shape[0])
        self.num_step_grid = int(num_step_grid)
        self.x_min = float(x_min if x_min is not None else x.min().item() - pad)
        self.x_max = float(x_max if x_max is not None else x.max().item() + pad)
        self.fit(
            x=x,
            num_step_grid=self.num_step_grid,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            smoother=smoother,
        )

    @classmethod
    def from_result(cls, result: MarginalFitResult) -> "GridKDE1D":
        self = cls.__new__(cls)
        BaseMarginal1D.__init__(self)
        self.num_step_grid = int(result.grid_x.numel())
        self.x_min = float(result.x_min)
        self.x_max = float(result.x_max)
        self._assign_result(result)
        return self

    @classmethod
    def empty(cls) -> "GridKDE1D":
        self = cls.__new__(cls)
        BaseMarginal1D.__init__(self)
        self.num_step_grid = 0
        return self

    @staticmethod
    def fit_reference(
        x: torch.Tensor,
        *,
        num_step_grid: int,
        x_min: float,
        x_max: float,
        bandwidth: str | float | torch.Tensor,
        bandwidth_scale: float,
        smoother: str,
    ) -> MarginalFitResult:
        x = x.view(-1).to(dtype=torch.float64)
        grid_x = torch.linspace(x_min, x_max, num_step_grid, device=x.device, dtype=x.dtype)
        step = float(grid_x[1] - grid_x[0])
        if isinstance(bandwidth, str):
            if bandwidth == "isj":
                try:
                    h = _isj_bandwidth_1d(x, x_min=x_min, x_max=x_max, num_step_grid=num_step_grid)
                    method = "isj"
                except ValueError:
                    h = _silverman_bandwidth_1d(x)
                    method = "silverman"
            elif bandwidth == "silverman":
                h = _silverman_bandwidth_1d(x)
                method = "silverman"
            else:
                raise ValueError("Unsupported bandwidth for grid marginal backend.")
        else:
            h = torch.as_tensor(bandwidth, device=x.device, dtype=x.dtype)
            method = "custom"
        h = (h * float(bandwidth_scale)).clamp_min(step)
        sigma_bins = h / step
        counts = _linear_bin_1d(x, num_step_grid, x_min, x_max).to(device=x.device, dtype=x.dtype)
        pdf = _smooth_1d(counts, sigma_bins, smoother=smoother)
        pdf = (pdf / (x.numel() * step)).clamp_min(_EPS)
        pdf /= pdf.sum().clamp_min(_EPS) * step
        cdf, slope_fwd, slope_inv, slope_pdf = _build_pdf_buffers_1d(pdf=pdf, grid_x=grid_x)
        return MarginalFitResult(
            grid_x=grid_x,
            grid_pdf=pdf,
            grid_cdf=cdf,
            slope_pdf=slope_pdf,
            slope_fwd=slope_fwd,
            slope_inv=slope_inv,
            bandwidth=h.reshape(()),
            x_min=x_min,
            x_max=x_max,
            num_obs=int(x.numel()),
            backend_name="grid",
            backend_config={
                "bandwidth": method if isinstance(bandwidth, str) else "custom",
                "bandwidth_scale": float(bandwidth_scale),
                "num_step_grid": int(num_step_grid),
                "smoother": smoother,
            },
        )

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        *,
        num_step_grid: int | None = None,
        bandwidth: str | float | torch.Tensor = "isj",
        bandwidth_scale: float = 1.0,
        smoother: str = "auto",
    ) -> None:
        if num_step_grid is not None:
            self.num_step_grid = int(num_step_grid)
        result = self.fit_reference(
            x=x,
            num_step_grid=self.num_step_grid,
            x_min=self.x_min,
            x_max=self.x_max,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            smoother=smoother,
        )
        self._assign_result(result)
        self.negloglik.copy_(-self.log_pdf(x.view(-1, 1).to(device=self.device, dtype=self.dtype)).mean())


class LPRefMarginal1D(BaseMarginal1D):
    def __init__(
        self,
        x: torch.Tensor,
        *,
        num_step_grid: int | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        pad: float = 0.1,
        reference_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        x = x.view(-1, 1).to(dtype=torch.float64)
        if num_step_grid is None:
            num_step_grid = _next_power_of_two_plus_one(x.shape[0])
        self.num_step_grid = int(num_step_grid)
        self.x_min = float(x_min if x_min is not None else x.min().item() - pad)
        self.x_max = float(x_max if x_max is not None else x.max().item() + pad)
        self.fit(
            x=x,
            num_step_grid=self.num_step_grid,
            reference_kwargs=reference_kwargs,
        )

    @classmethod
    def from_result(cls, result: MarginalFitResult) -> "LPRefMarginal1D":
        self = cls.__new__(cls)
        BaseMarginal1D.__init__(self)
        self.num_step_grid = int(result.grid_x.numel())
        self.x_min = float(result.x_min)
        self.x_max = float(result.x_max)
        self._assign_result(result)
        return self

    @classmethod
    def empty(cls) -> "LPRefMarginal1D":
        self = cls.__new__(cls)
        BaseMarginal1D.__init__(self)
        self.num_step_grid = 0
        return self

    @staticmethod
    def fit_reference(
        x: torch.Tensor,
        *,
        num_step_grid: int,
        x_min: float,
        x_max: float,
        reference_kwargs: dict[str, Any] | None,
    ) -> MarginalFitResult:
        source, kde_cls = _lazy_reference_kde()
        x = x.view(-1).cpu().to(dtype=torch.float64)
        grid_x = torch.linspace(x_min, x_max, num_step_grid, dtype=torch.float64)
        if source == "pyvinecopulib":
            kde = kde_cls(x.numpy(), **(reference_kwargs or {}))
            grid_pdf = torch.as_tensor(kde.pdf(grid_x.numpy()), dtype=torch.float64)
            grid_cdf_raw = torch.as_tensor(kde.cdf(grid_x.numpy()), dtype=torch.float64)
            bandwidth = torch.full((), float("nan"), dtype=torch.float64)
        else:
            if reference_kwargs:
                allowed = {"bw_method", "weights"}
                unknown = set(reference_kwargs) - allowed
                if unknown:
                    raise ValueError(
                        f"Unknown reference_kwargs for lp_ref: {sorted(unknown)}. "
                        f"Allowed keys: {sorted(allowed)}"
                    )
            kde = kde_cls(x.numpy(), **(reference_kwargs or {}))
            grid_pdf = torch.as_tensor(kde(grid_x.numpy()), dtype=torch.float64)
            bandwidth = torch.as_tensor(float(math.sqrt(float(kde.covariance[0, 0]))), dtype=torch.float64)
            grid_cdf_raw = (grid_pdf * float(grid_x[1] - grid_x[0])).cumsum(dim=0)
        step = float(grid_x[1] - grid_x[0])
        grid_pdf = grid_pdf.clamp_min(_EPS)
        grid_pdf /= grid_pdf.sum().clamp_min(_EPS) * step
        if source == "pyvinecopulib":
            grid_cdf = grid_cdf_raw.clamp(0.0, 1.0)
            grid_cdf /= grid_cdf[-1].clamp_min(_EPS)
            delta_x = torch.full((grid_x.numel() - 1,), step, dtype=grid_pdf.dtype)
            delta_cdf = (grid_cdf[1:] - grid_cdf[:-1]).clamp_min(_EPS)
            slope_fwd = delta_cdf / delta_x
            slope_inv = delta_x / delta_cdf
            slope_pdf = (grid_pdf[1:] - grid_pdf[:-1]) / delta_x
        else:
            grid_cdf, slope_fwd, slope_inv, slope_pdf = _build_pdf_buffers_1d(pdf=grid_pdf, grid_x=grid_x)
        return MarginalFitResult(
            grid_x=grid_x,
            grid_pdf=grid_pdf,
            grid_cdf=grid_cdf,
            slope_pdf=slope_pdf,
            slope_fwd=slope_fwd,
            slope_inv=slope_inv,
            bandwidth=bandwidth.reshape(()),
            x_min=x_min,
            x_max=x_max,
            num_obs=int(x.numel()),
            backend_name="lp_ref",
            backend_config={
                "reference_impl": source,
                "reference_kwargs": dict(reference_kwargs or {}),
                "num_step_grid": int(num_step_grid),
            },
        )

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        *,
        num_step_grid: int | None = None,
        reference_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if num_step_grid is not None:
            self.num_step_grid = int(num_step_grid)
        result = self.fit_reference(
            x=x,
            num_step_grid=self.num_step_grid,
            x_min=self.x_min,
            x_max=self.x_max,
            reference_kwargs=reference_kwargs,
        )
        self._assign_result(result)
        self.negloglik.copy_(-self.log_pdf(x.view(-1, 1).to(device=self.device, dtype=self.dtype)).mean())


MARGINAL_BACKENDS = {
    "grid": GridKDE1D,
    "lp_ref": LPRefMarginal1D,
}


def build_marginal_estimator(
    *,
    backend_name: str,
    x: torch.Tensor,
    backend_kwargs: dict[str, Any] | None = None,
) -> BaseMarginal1D:
    backend_name = normalize_marginal_backend(backend_name)
    defaults = (
        {
            "bandwidth": "isj",
            "bandwidth_scale": 1.0,
            "num_step_grid": None,
            "smoother": "auto",
        }
        if backend_name == "grid"
        else {"reference_kwargs": None, "num_step_grid": None}
    )
    allowed = set(defaults)
    normalized = _normalize_backend_kwargs(
        backend_name=backend_name,
        kwargs=backend_kwargs,
        defaults=defaults,
        allowed=allowed,
    )
    estimator_cls = MARGINAL_BACKENDS[backend_name]
    return estimator_cls(x=x, **normalized)


def build_marginal_shell(*, backend_name: str) -> BaseMarginal1D:
    backend_name = normalize_marginal_backend(backend_name)
    estimator_cls = MARGINAL_BACKENDS[backend_name]
    return estimator_cls.empty()


TorchKDE1D = GridKDE1D
