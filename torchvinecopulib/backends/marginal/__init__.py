"""Marginal backend registry and estimator families."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pprint import pformat
from typing import Any

import torch

from ..common import (
    _EPS,
    API_VERSION,
    MarginalFitResult,
    _build_pdf_buffers_1d,
    _isj_bandwidth_1d,
    _linear_bin_1d,
    _next_power_of_two_plus_one,
    _normalize_backend_kwargs,
    _normal_cdf,
    _normal_pdf_drv,
    _normal_ppf,
    _robust_scale,
    _smooth_1d,
    _silverman_bandwidth_1d,
)

_LP_GRID_SIZE = 401
_LP_BOUNDARY_EPS = 1e-5

__all__ = [
    "BaseMarginal1D",
    "GridKDE1D",
    "LocalPolynomialKDE1D",
    "build_marginal_estimator",
    "build_marginal_shell",
    "normalize_marginal_backend",
]


def _canonical_marginal_backend(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def normalize_marginal_backend(name: str) -> str:
    aliases = {
        "torch_grid": "grid",
        "grid": "grid",
        "lp": "lp",
        "lp_torch": "lp",
        "local_polynomial": "lp",
        "kde1d_torch": "lp",
    }
    normalized = _canonical_marginal_backend(name)
    if normalized not in aliases:
        raise ValueError("marginal_backend must be one of 'grid' or 'lp'.")
    return aliases[normalized]


def _trapz_integral_1d(pdf: torch.Tensor, grid_x: torch.Tensor) -> torch.Tensor:
    delta_x = (grid_x[1:] - grid_x[:-1]).to(device=pdf.device, dtype=pdf.dtype).clamp_min(_EPS)
    cell_mass = 0.5 * (pdf[:-1] + pdf[1:]) * delta_x
    return cell_mass.sum().clamp_min(_EPS)


def _normalize_pdf_1d(pdf: torch.Tensor, grid_x: torch.Tensor) -> torch.Tensor:
    pdf = pdf.to(device=grid_x.device, dtype=grid_x.dtype).clamp_min(0.0)
    mass = _trapz_integral_1d(pdf, grid_x)
    return (pdf / mass).clamp_min(_EPS)


def _support_bounds(
    x: torch.Tensor,
    *,
    x_min: float | None,
    x_max: float | None,
    pad: float,
) -> tuple[float, float]:
    x = x.view(-1).to(dtype=torch.float64)
    left = float(x_min if x_min is not None else x.min().item() - pad)
    right = float(x_max if x_max is not None else x.max().item() + pad)
    if not math.isfinite(left) or not math.isfinite(right) or right <= left:
        raise ValueError("Marginal support bounds must define a positive interval.")
    return left, right


def _resolve_grid_bandwidth(
    x: torch.Tensor,
    *,
    x_min: float,
    x_max: float,
    num_step_grid: int,
    bandwidth: str | float | torch.Tensor,
    bandwidth_scale: float,
) -> tuple[torch.Tensor, str]:
    step = max((x_max - x_min) / max(num_step_grid - 1, 1), _EPS)
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
    return h.reshape(()), method


def _lp_boundary_kind(x_min: float | None, x_max: float | None) -> str:
    if x_min is not None and x_max is not None:
        return "two"
    if x_min is not None:
        return "left"
    if x_max is not None:
        return "right"
    return "none"


def _lp_boundary_transform(
    x: torch.Tensor,
    *,
    x_min: float | None,
    x_max: float | None,
    inverse: bool = False,
) -> torch.Tensor:
    x = x.to(dtype=torch.float64)
    kind = _lp_boundary_kind(x_min, x_max)
    if kind == "none":
        return x
    if kind == "two":
        assert x_min is not None and x_max is not None
        rng = max(x_max - x_min, _EPS)
        if inverse:
            return _normal_cdf(x) * (1.0001 * rng) + x_min - 5e-5 * rng
        u = ((x - x_min + 5e-5 * rng) / (1.0001 * rng)).clamp(_EPS, 1.0 - _EPS)
        return _normal_ppf(u)
    if kind == "left":
        assert x_min is not None
        if inverse:
            return torch.exp(x) + x_min - _LP_BOUNDARY_EPS
        return torch.log((x - x_min + _LP_BOUNDARY_EPS).clamp_min(_EPS))
    assert x_max is not None
    if inverse:
        return x_max + _LP_BOUNDARY_EPS - torch.exp(x)
    return torch.log((x_max - x + _LP_BOUNDARY_EPS).clamp_min(_EPS))


def _lp_boundary_correction(
    x: torch.Tensor,
    *,
    x_min: float | None,
    x_max: float | None,
) -> torch.Tensor:
    x = x.to(dtype=torch.float64)
    kind = _lp_boundary_kind(x_min, x_max)
    if kind == "none":
        return torch.ones_like(x)
    if kind == "two":
        assert x_min is not None and x_max is not None
        rng = max(x_max - x_min, _EPS)
        u = ((x - x_min + 5e-5 * rng) / (x_max - x_min + 1e-4 * rng)).clamp(_EPS, 1.0 - _EPS)
        corr = _normal_pdf_drv(_normal_ppf(u), 0) / (x_max - x_min + 1e-4 * rng)
        return corr.clamp_min(1e-6).reciprocal()
    if kind == "left":
        assert x_min is not None
        return (x - x_min + _LP_BOUNDARY_EPS).clamp_min(1e-6).reciprocal()
    assert x_max is not None
    return (x_max - x + _LP_BOUNDARY_EPS).clamp_min(1e-6).reciprocal()


def _kdefft_bin_counts(
    x: torch.Tensor,
    *,
    lower: float,
    upper: float,
    num_bins: int,
) -> torch.Tensor:
    return _linear_bin_1d(x, num_bins + 1, lower, upper)


def _kdefft_derivative_from_counts(
    *,
    bin_counts: torch.Tensor,
    bandwidth: torch.Tensor,
    lower: float,
    upper: float,
    drv: int,
) -> torch.Tensor:
    num_bins = bin_counts.numel() - 1
    delta = max((upper - lower) / max(num_bins, 1), _EPS)
    h = float(bandwidth.reshape(()).item())
    L = min(int(math.floor((4.0 + drv) * h / delta)), num_bins + 1)
    L = max(L, 1)
    arg = torch.linspace(
        0.0,
        float(L) * delta / max(h, _EPS),
        steps=L + 1,
        device=bin_counts.device,
        dtype=bin_counts.dtype,
    )
    kernel = _normal_pdf_drv(arg, drv) / (
        (max(h, _EPS) ** (drv + 1)) * bin_counts.sum().clamp_min(_EPS)
    )
    P = 1 << int(math.ceil(math.log2(max(num_bins + L + 2, 2))))
    kernel_pad = torch.zeros(P, device=bin_counts.device, dtype=bin_counts.dtype)
    kernel_pad[: L + 1] = kernel
    if L > 0:
        sign = -1.0 if drv % 2 else 1.0
        kernel_pad[-L:] = kernel[1:].flip(0) * sign
    counts_pad = torch.zeros(P, device=bin_counts.device, dtype=bin_counts.dtype)
    counts_pad[: num_bins + 1] = bin_counts
    conv = torch.fft.irfft(
        torch.fft.rfft(kernel_pad) * torch.fft.rfft(counts_pad),
        n=P,
    )
    return conv[: num_bins + 1]


def _select_lp_bandwidth(
    x: torch.Tensor,
    *,
    degree: int,
    bandwidth: str | float | torch.Tensor,
    bandwidth_scale: float,
) -> tuple[torch.Tensor, str]:
    x = x.view(-1).to(dtype=torch.float64)
    if not isinstance(bandwidth, str):
        h = torch.as_tensor(bandwidth, device=x.device, dtype=x.dtype).reshape(())
        return (h * float(bandwidth_scale)).clamp_min(_EPS), "custom"
    if bandwidth == "silverman":
        h = _silverman_bandwidth_1d(x)
        return (h * float(bandwidth_scale)).clamp_min(_EPS), "silverman"
    if bandwidth not in {"plugin", "isj"}:
        raise ValueError(
            "LP marginal backend only supports 'plugin', 'isj', 'silverman', or a scalar."
        )
    scale = _robust_scale(x)
    n_eff = max(int(x.numel()), 2)
    bw_power = 4 if degree < 2 else 8
    fallback = torch.as_tensor(
        4.0 * 1.06 * float(scale.item()) * n_eff ** (-1.0 / (bw_power + 1)),
        device=x.device,
        dtype=x.dtype,
    ).clamp_min(_EPS)
    if bandwidth == "isj":
        lower = float(x.min().item())
        upper = float(x.max().item())
        if upper <= lower:
            return (fallback * float(bandwidth_scale)).clamp_min(_EPS), "silverman"
        try:
            h = _isj_bandwidth_1d(x, x_min=lower, x_max=upper, num_step_grid=_LP_GRID_SIZE)
            return (h * float(bandwidth_scale)).clamp_min(_EPS), "isj"
        except ValueError:
            return (fallback * float(bandwidth_scale)).clamp_min(_EPS), "silverman"

    lower = float(x.min().item())
    upper = float(x.max().item())
    if upper <= lower or x.numel() < 16:
        return (fallback * float(bandwidth_scale)).clamp_min(_EPS), "plugin_fallback"
    bin_counts = _kdefft_bin_counts(x, lower=lower, upper=upper, num_bins=400)

    def dnorm_drv0(order: int) -> torch.Tensor:
        return _normal_pdf_drv(
            torch.zeros(1, device=x.device, dtype=x.dtype),
            order,
        )[0]

    def get_bandwidth_for_bkfe(drv: int) -> torch.Tensor:
        if drv % 2:
            raise ValueError("BKFE helper only supports even derivatives.")
        r = drv + 4
        psi = 1.0 if ((r // 2) % 2 == 0) else -1.0
        psi *= math.gamma(r + 1)
        psi /= (2.0 * float(scale.item())) ** (r + 1) * math.gamma(r / 2 + 1) * math.sqrt(math.pi)
        Kr = float(dnorm_drv0(r - 2).item())
        initial = (-2.0 * Kr / (psi * n_eff)) ** (1.0 / (r + 1))
        initial_t = torch.as_tensor(initial, device=x.device, dtype=x.dtype).clamp_min(_EPS)
        psi_hat = (
            bin_counts
            * _kdefft_derivative_from_counts(
                bin_counts=bin_counts,
                bandwidth=initial_t,
                lower=lower,
                upper=upper,
                drv=drv + 2,
            )
        ).sum() / bin_counts.sum().clamp_min(_EPS)
        r -= 2
        Kr = float(dnorm_drv0(r - 2).item())
        value = -2.0 * Kr / (float(psi_hat.item()) * n_eff)
        return (
            torch.as_tensor(value, device=x.device, dtype=x.dtype)
            .clamp_min(_EPS)
            .pow(1.0 / (r + 1))
        )

    try:
        if degree == 0:
            pilot = get_bandwidth_for_bkfe(4)
            f4 = _kdefft_derivative_from_counts(
                bin_counts=bin_counts,
                bandwidth=pilot,
                lower=lower,
                upper=upper,
                drv=4,
            )
            arg = 0.25 * f4
            ibias2 = (bin_counts * arg.square()).sum() / bin_counts.sum().clamp_min(_EPS)
        elif degree == 1:
            pilot = get_bandwidth_for_bkfe(4)
            f0 = _kdefft_derivative_from_counts(
                bin_counts=bin_counts,
                bandwidth=pilot,
                lower=lower,
                upper=upper,
                drv=0,
            ).clamp_min(_EPS)
            f1 = _kdefft_derivative_from_counts(
                bin_counts=bin_counts,
                bandwidth=pilot,
                lower=lower,
                upper=upper,
                drv=1,
            )
            f2 = _kdefft_derivative_from_counts(
                bin_counts=bin_counts,
                bandwidth=pilot,
                lower=lower,
                upper=upper,
                drv=2,
            )
            arg = (0.5 * f2 + f1.square() / f0).square() / f0
            ibias2 = (bin_counts * arg).sum() / bin_counts.sum().clamp_min(_EPS)
        elif degree == 2:
            pilot = get_bandwidth_for_bkfe(8)
            f0 = _kdefft_derivative_from_counts(
                bin_counts=bin_counts,
                bandwidth=pilot,
                lower=lower,
                upper=upper,
                drv=0,
            ).clamp_min(_EPS)
            f1 = _kdefft_derivative_from_counts(
                bin_counts=bin_counts,
                bandwidth=pilot,
                lower=lower,
                upper=upper,
                drv=1,
            )
            f2 = _kdefft_derivative_from_counts(
                bin_counts=bin_counts,
                bandwidth=pilot,
                lower=lower,
                upper=upper,
                drv=2,
            )
            f4 = _kdefft_derivative_from_counts(
                bin_counts=bin_counts,
                bandwidth=pilot,
                lower=lower,
                upper=upper,
                drv=4,
            )
            arg = f4 - 3.0 * f2.square() / f0 + 2.0 * f1.pow(4) / f0.pow(3)
            arg = (0.125 * arg).square() / f0
            ibias2 = (bin_counts * arg).sum() / bin_counts.sum().clamp_min(_EPS)
        else:
            raise ValueError("degree must be one of {0, 1, 2}.")
        ivar = (1.0 if degree < 2 else 27.0 / 16.0) * 0.5 / math.sqrt(math.pi)
        h = (ivar / (bw_power * n_eff * float(ibias2.clamp_min(_EPS).item()))) ** (
            1.0 / (bw_power + 1)
        )
        h_t = torch.as_tensor(h, device=x.device, dtype=x.dtype)
        method = "plugin"
    except Exception:
        h_t = fallback
        method = "plugin_fallback"
    return (h_t * float(bandwidth_scale)).clamp_min(_EPS), method


def _fit_local_polynomial_density(
    x: torch.Tensor,
    *,
    z_grid: torch.Tensor,
    bandwidth: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    lower = float(z_grid[0].item())
    upper = float(z_grid[-1].item())
    num_bins = z_grid.numel() - 1
    bin_counts = _kdefft_bin_counts(x, lower=lower, upper=upper, num_bins=num_bins)
    f0 = _kdefft_derivative_from_counts(
        bin_counts=bin_counts,
        bandwidth=bandwidth,
        lower=lower,
        upper=upper,
        drv=0,
    ).clamp_min(_EPS)
    if degree == 0:
        return f0
    f1 = _kdefft_derivative_from_counts(
        bin_counts=bin_counts,
        bandwidth=bandwidth,
        lower=lower,
        upper=upper,
        drv=1,
    )
    S = torch.full_like(f0, float(bandwidth.item()))
    b = f1 / f0
    out = f0
    if degree == 2:
        h = float(bandwidth.item())
        f2 = _kdefft_derivative_from_counts(
            bin_counts=bin_counts,
            bandwidth=bandwidth,
            lower=lower,
            upper=upper,
            drv=2,
        )
        D = f2 / f0 - b.square()
        R = (1.0 + h * h * D).clamp_min(_EPS).rsqrt()
        S = (R / max(h, _EPS)).square()
        b = b * (h * h)
        out = h * torch.sqrt(S) * out
    return (out * torch.exp(-0.5 * b.square() * S)).clamp_min(0.0)


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
        idx = torch.searchsorted(self.grid_x, x_clamped, right=False).clamp(
            1, self.grid_x.numel() - 1
        )
        y = self.grid_cdf[idx - 1] + self.slope_fwd[idx - 1] * (x_clamped - self.grid_x[idx - 1])
        y = torch.where(x < self.x_min, torch.zeros_like(y), y)
        y = torch.where(x > self.x_max, torch.ones_like(y), y)
        return y.clamp(0.0, 1.0)

    def ppf(self, q: torch.Tensor) -> torch.Tensor:
        q = q.to(device=self.device, dtype=self.dtype)
        q_clamped = q.clamp(0.0, 1.0)
        idx = torch.searchsorted(self.grid_cdf, q_clamped, right=False).clamp(
            1, self.grid_cdf.numel() - 1
        )
        x = self.grid_x[idx - 1] + self.slope_inv[idx - 1] * (q_clamped - self.grid_cdf[idx - 1])
        x = torch.where(q < 0.0, torch.full_like(x, self.x_min), x)
        x = torch.where(q > 1.0, torch.full_like(x, self.x_max), x)
        return x.clamp(self.x_min, self.x_max)

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x_clamped = x.clamp(self.x_min, self.x_max)
        idx = torch.searchsorted(self.grid_x, x_clamped, right=False).clamp(
            1, self.grid_pdf.numel() - 1
        )
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
            "bandwidth": float(self.bandwidth.reshape(-1)[0].round(decimals=6))
            if self.bandwidth.numel()
            else 0.0,
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
        self.x_min, self.x_max = _support_bounds(x, x_min=x_min, x_max=x_max, pad=pad)
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
        h, method = _resolve_grid_bandwidth(
            x,
            x_min=x_min,
            x_max=x_max,
            num_step_grid=num_step_grid,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
        )
        sigma_bins = h / step
        counts = _linear_bin_1d(x, num_step_grid, x_min, x_max).to(device=x.device, dtype=x.dtype)
        pdf = _smooth_1d(counts, sigma_bins, smoother=smoother)
        pdf = (pdf / (x.numel() * step)).clamp_min(_EPS)
        pdf = _normalize_pdf_1d(pdf, grid_x)
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
                "bandwidth": method,
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
        self.negloglik.copy_(
            -self.log_pdf(x.view(-1, 1).to(device=self.device, dtype=self.dtype)).mean()
        )


class LocalPolynomialKDE1D(BaseMarginal1D):
    """Torch-native continuous subset of the `kde1d` local-polynomial KDE.

    This backend mirrors the continuous-path design of `kde1d`:

    - `degree=0`: log-constant fitting,
    - `degree=1`: log-linear fitting,
    - `degree=2`: log-quadratic fitting.

    When support boundaries are supplied, the implementation applies the same
    transformation strategy used by `kde1d`:

    - no boundary: work directly on the original scale,
    - one boundary: log transform,
    - two boundaries: probit transform.

    Density derivatives and the plug-in bandwidth rule are evaluated on a binned
    FFT grid in transformed coordinates, then mapped back to the original domain
    by the inverse Jacobian. Discrete and zero-inflated extensions from the full
    `kde1d` package are intentionally out of scope here because
    `torchvinecopulib` currently targets continuous data only.
    """

    def __init__(
        self,
        x: torch.Tensor,
        *,
        num_step_grid: int | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        degree: int = 2,
        bandwidth: str | float | torch.Tensor = "plugin",
        bandwidth_scale: float = 1.0,
    ) -> None:
        super().__init__()
        x = x.view(-1, 1).to(dtype=torch.float64)
        if num_step_grid is None:
            num_step_grid = _LP_GRID_SIZE
        self.num_step_grid = int(num_step_grid)
        self.support_min = x_min
        self.support_max = x_max
        self.fit(
            x=x,
            num_step_grid=self.num_step_grid,
            x_min=x_min,
            x_max=x_max,
            degree=degree,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
        )

    @classmethod
    def from_result(cls, result: MarginalFitResult) -> "LocalPolynomialKDE1D":
        self = cls.__new__(cls)
        BaseMarginal1D.__init__(self)
        self.num_step_grid = int(result.grid_x.numel())
        self.support_min = result.backend_config.get("support_min")
        self.support_max = result.backend_config.get("support_max")
        self._assign_result(result)
        return self

    @classmethod
    def empty(cls) -> "LocalPolynomialKDE1D":
        self = cls.__new__(cls)
        BaseMarginal1D.__init__(self)
        self.num_step_grid = 0
        self.support_min = None
        self.support_max = None
        return self

    @staticmethod
    def fit_reference(
        x: torch.Tensor,
        *,
        num_step_grid: int,
        x_min: float | None,
        x_max: float | None,
        degree: int,
        bandwidth: str | float | torch.Tensor,
        bandwidth_scale: float,
    ) -> MarginalFitResult:
        if degree not in {0, 1, 2}:
            raise ValueError("degree must be one of {0, 1, 2}.")
        x = x.view(-1).to(dtype=torch.float64)
        z_obs = _lp_boundary_transform(x, x_min=x_min, x_max=x_max)
        h, method = _select_lp_bandwidth(
            z_obs,
            degree=degree,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
        )
        z_lo = float(z_obs.min().item())
        z_hi = float(z_obs.max().item())
        if _lp_boundary_kind(x_min, x_max) == "none":
            z_lo -= 4.0 * float(h.item())
            z_hi += 4.0 * float(h.item())
        z_grid = torch.linspace(z_lo, z_hi, steps=num_step_grid, device=x.device, dtype=x.dtype)
        grid_x = _lp_boundary_transform(z_grid, x_min=x_min, x_max=x_max, inverse=True)
        pdf_z = _fit_local_polynomial_density(z_obs, z_grid=z_grid, bandwidth=h, degree=degree)
        pdf_x = pdf_z * _lp_boundary_correction(grid_x, x_min=x_min, x_max=x_max)
        if _lp_boundary_kind(x_min, x_max) == "right":
            grid_x = grid_x.flip(0)
            pdf_x = pdf_x.flip(0)
        if x_min is not None:
            grid_x[0] = float(x_min)
        if x_max is not None:
            grid_x[-1] = float(x_max)
        pdf_x = _normalize_pdf_1d(pdf_x, grid_x)
        cdf, slope_fwd, slope_inv, slope_pdf = _build_pdf_buffers_1d(pdf=pdf_x, grid_x=grid_x)
        return MarginalFitResult(
            grid_x=grid_x,
            grid_pdf=pdf_x,
            grid_cdf=cdf,
            slope_pdf=slope_pdf,
            slope_fwd=slope_fwd,
            slope_inv=slope_inv,
            bandwidth=h.reshape(()),
            x_min=float(grid_x[0].item()),
            x_max=float(grid_x[-1].item()),
            num_obs=int(x.numel()),
            backend_name="lp",
            backend_config={
                "bandwidth": method,
                "bandwidth_scale": float(bandwidth_scale),
                "degree": int(degree),
                "num_step_grid": int(num_step_grid),
                "support_min": x_min,
                "support_max": x_max,
            },
        )

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        *,
        num_step_grid: int | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        degree: int = 2,
        bandwidth: str | float | torch.Tensor = "plugin",
        bandwidth_scale: float = 1.0,
    ) -> None:
        if num_step_grid is not None:
            self.num_step_grid = int(num_step_grid)
        result = self.fit_reference(
            x=x,
            num_step_grid=self.num_step_grid,
            x_min=x_min if x_min is not None else self.support_min,
            x_max=x_max if x_max is not None else self.support_max,
            degree=degree,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
        )
        self._assign_result(result)
        self.support_min = result.backend_config.get("support_min")
        self.support_max = result.backend_config.get("support_max")
        self.negloglik.copy_(
            -self.log_pdf(x.view(-1, 1).to(device=self.device, dtype=self.dtype)).mean()
        )


MARGINAL_BACKENDS = {
    "grid": GridKDE1D,
    "lp": LocalPolynomialKDE1D,
}


def build_marginal_estimator(
    *,
    backend_name: str,
    x: torch.Tensor,
    backend_kwargs: dict[str, Any] | None = None,
) -> BaseMarginal1D:
    backend_name = normalize_marginal_backend(backend_name)
    if backend_name == "grid":
        defaults = {
            "bandwidth": "isj",
            "bandwidth_scale": 1.0,
            "num_step_grid": None,
            "smoother": "auto",
            "x_min": None,
            "x_max": None,
            "pad": 0.1,
        }
    else:
        defaults = {
            "num_step_grid": _LP_GRID_SIZE,
            "x_min": None,
            "x_max": None,
            "degree": 2,
            "bandwidth": "plugin",
            "bandwidth_scale": 1.0,
        }
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
