from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

_EPS = 1e-10
_SQRT_2PI = math.sqrt(2.0 * math.pi)
API_VERSION = "1.3.0"


@dataclass
class MarginalFitResult:
    grid_x: torch.Tensor
    grid_pdf: torch.Tensor
    grid_cdf: torch.Tensor
    slope_pdf: torch.Tensor
    slope_fwd: torch.Tensor
    slope_inv: torch.Tensor
    bandwidth: torch.Tensor
    x_min: float
    x_max: float
    num_obs: int
    backend_name: str
    backend_config: dict[str, Any]


@dataclass
class BicopFitResult:
    pdf_grid: torch.Tensor
    cdf_grid: torch.Tensor
    hfunc_l_grid: torch.Tensor
    hfunc_r_grid: torch.Tensor
    bandwidth: torch.Tensor
    backend_name: str
    backend_config: dict[str, Any]


def _normalize_backend_kwargs(
    *,
    backend_name: str,
    kwargs: dict[str, Any] | None,
    defaults: dict[str, Any],
    allowed: set[str],
) -> dict[str, Any]:
    merged = dict(defaults)
    if kwargs:
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"Unknown kwargs for backend '{backend_name}': {sorted(unknown)}. "
                f"Allowed keys: {sorted(allowed)}"
            )
        merged.update(kwargs)
    return merged


def _next_power_of_two_plus_one(num: int) -> int:
    base = max(32, int(num))
    return 1 + (1 << int(math.ceil(math.log2(base))))


def _robust_scale(x: torch.Tensor) -> torch.Tensor:
    x = x.view(-1).to(dtype=torch.float64)
    if x.numel() < 2:
        return torch.ones((), device=x.device, dtype=x.dtype)
    std = x.std(unbiased=False).clamp_min(_EPS)
    q25 = torch.quantile(x, 0.25)
    q75 = torch.quantile(x, 0.75)
    iqr = ((q75 - q25) / 1.349).clamp_min(_EPS)
    return torch.minimum(std, iqr)


def _silverman_bandwidth_1d(x: torch.Tensor) -> torch.Tensor:
    scale = _robust_scale(x)
    return (0.9 * scale * x.numel() ** (-0.2)).clamp_min(_EPS)


def _silverman_bandwidth_2d(obs: torch.Tensor) -> torch.Tensor:
    n = max(int(obs.shape[0]), 2)
    scales = torch.stack([_robust_scale(obs[:, 0]), _robust_scale(obs[:, 1])])
    h = 1.06 * scales * n ** (-1.0 / 6.0)
    return h.clamp_min(_EPS)


def _dct_ii(x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=torch.float64)
    n = x.numel()
    even = x[::2]
    odd = x[1::2].flip(0)
    v = torch.cat([even, odd], dim=0)
    V = torch.fft.fft(v)
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    phase = torch.exp(-0.5j * math.pi * k / n)
    return 2.0 * torch.real(V[:n] * phase)


def _isj_fixed_point(
    t: torch.Tensor,
    num_obs: int,
    i_sq: torch.Tensor,
    a_sq: torch.Tensor,
) -> torch.Tensor:
    num_obs_t = torch.as_tensor(float(num_obs), device=t.device, dtype=t.dtype)
    f = 2.0 * math.pi**14 * torch.sum(i_sq**7 * a_sq * torch.exp(-(math.pi**2) * i_sq * t))
    f = f.clamp_min(_EPS)
    for s in range(6, 1, -1):
        k0 = math.prod(range(1, 2 * s, 2)) / math.sqrt(2.0 * math.pi)
        const = (1.0 + 0.5 ** (s + 0.5)) / 3.0
        time = (2.0 * const * k0 / (num_obs_t * f)) ** (2.0 / (3.0 + 2.0 * s))
        f = (
            2.0
            * math.pi ** (2 * s)
            * torch.sum(i_sq**s * a_sq * torch.exp(-(math.pi**2) * i_sq * time))
        ).clamp_min(_EPS)
    return t - (2.0 * num_obs_t * math.sqrt(math.pi) * f) ** (-0.4)


def _isj_bandwidth_1d(
    x: torch.Tensor,
    *,
    x_min: float,
    x_max: float,
    num_step_grid: int,
) -> torch.Tensor:
    x = x.view(-1).to(dtype=torch.float64)
    if x.numel() < 32:
        raise ValueError("ISJ requires at least 32 samples.")
    span = float(x_max - x_min)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("ISJ requires a positive support span.")
    probs = _linear_bin_1d(x, num_step_grid, x_min, x_max)
    probs = probs / probs.sum().clamp_min(_EPS)
    a = _dct_ii(probs)
    i_sq = torch.arange(1, probs.numel(), device=x.device, dtype=x.dtype).square()
    a_sq = (a[1:] / 2.0).square()
    lo = torch.tensor(1e-7, device=x.device, dtype=x.dtype)
    hi = torch.tensor(0.1, device=x.device, dtype=x.dtype)
    f_lo = _isj_fixed_point(lo, x.numel(), i_sq, a_sq)
    f_hi = _isj_fixed_point(hi, x.numel(), i_sq, a_sq)
    if torch.sign(f_lo) == torch.sign(f_hi):
        raise ValueError("ISJ fixed point did not bracket a root.")
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        f_mid = _isj_fixed_point(mid, x.numel(), i_sq, a_sq)
        if f_mid.abs() < 1e-12 or (hi - lo) < 1e-12:
            root = mid
            break
        if torch.sign(f_mid) == torch.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    else:
        root = 0.5 * (lo + hi)
    bandwidth = torch.sqrt(root.clamp_min(_EPS)) * span
    return bandwidth.clamp_min(span / max(num_step_grid - 1, 1))


def _normal_pdf(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z.square()) / _SQRT_2PI


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _normal_ppf(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(_EPS, 1.0 - _EPS)
    return math.sqrt(2.0) * torch.special.erfinv(2.0 * u - 1.0)


def _gaussian_kernel1d(sigma_bins: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    sigma_bins = sigma_bins.to(device=device, dtype=dtype).clamp_min(0.5)
    radius = int(torch.ceil(4.0 * sigma_bins).item())
    grid = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (grid / sigma_bins) ** 2)
    return kernel / kernel.sum().clamp_min(_EPS)


def _fft_conv_same_1d(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    n_full = signal.numel() + kernel.numel() - 1
    n_fft = 1 << int(math.ceil(math.log2(max(n_full, 2))))
    signal_fft = torch.fft.rfft(F.pad(signal, (0, n_fft - signal.numel())))
    kernel_fft = torch.fft.rfft(F.pad(kernel, (0, n_fft - kernel.numel())))
    full = torch.fft.irfft(signal_fft * kernel_fft, n=n_fft)[:n_full]
    start = (kernel.numel() - 1) // 2
    end = start + signal.numel()
    return full[start:end]


def _conv_same_1d(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if signal.numel() >= 512 or kernel.numel() >= 63:
        return _fft_conv_same_1d(signal, kernel)
    pad = kernel.numel() // 2
    return F.conv1d(signal.view(1, 1, -1), kernel.view(1, 1, -1), padding=pad).view(-1)


def _fft_conv_same_2d(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    full_h = signal.shape[0] + kernel.shape[0] - 1
    full_w = signal.shape[1] + kernel.shape[1] - 1
    fft_h = 1 << int(math.ceil(math.log2(max(full_h, 2))))
    fft_w = 1 << int(math.ceil(math.log2(max(full_w, 2))))
    s_pad = F.pad(signal, (0, fft_w - signal.shape[1], 0, fft_h - signal.shape[0]))
    k_pad = F.pad(kernel, (0, fft_w - kernel.shape[1], 0, fft_h - kernel.shape[0]))
    full = torch.fft.irfft2(torch.fft.rfft2(s_pad) * torch.fft.rfft2(k_pad), s=(fft_h, fft_w))
    full = full[:full_h, :full_w]
    start_h = (kernel.shape[0] - 1) // 2
    start_w = (kernel.shape[1] - 1) // 2
    return full[start_h : start_h + signal.shape[0], start_w : start_w + signal.shape[1]]


def _recursive_gaussian_line(signal: torch.Tensor, sigma_bins: float, num_passes: int = 4) -> torch.Tensor:
    if sigma_bins <= 0.5 or signal.numel() < 2:
        return signal
    alpha = math.exp(-math.sqrt(2.0) / max(float(sigma_bins), 1e-6))
    alpha_t = torch.as_tensor(alpha, device=signal.device, dtype=signal.dtype)
    one_minus = 1.0 - alpha_t
    out = signal.clone()
    for _ in range(num_passes):
        forward = torch.empty_like(out)
        forward[0] = out[0]
        for idx in range(1, out.numel()):
            forward[idx] = alpha_t * forward[idx - 1] + one_minus * out[idx]
        backward = torch.empty_like(out)
        backward[-1] = forward[-1]
        for idx in range(out.numel() - 2, -1, -1):
            backward[idx] = alpha_t * backward[idx + 1] + one_minus * forward[idx]
        out = backward
    return out


def _recursive_gaussian_smooth_1d(signal: torch.Tensor, sigma_bins: torch.Tensor) -> torch.Tensor:
    return _recursive_gaussian_line(signal, float(torch.as_tensor(sigma_bins).item()))


def _smooth_1d(signal: torch.Tensor, sigma_bins: torch.Tensor, smoother: str = "auto") -> torch.Tensor:
    smoother = smoother.lower()
    if smoother == "auto":
        smoother = "fft" if signal.numel() >= 512 else "conv"
    if smoother == "recursive":
        return _recursive_gaussian_smooth_1d(signal, sigma_bins)
    kernel = _gaussian_kernel1d(sigma_bins, dtype=signal.dtype, device=signal.device)
    if smoother == "fft":
        return _fft_conv_same_1d(signal, kernel)
    if smoother == "conv":
        return _conv_same_1d(signal, kernel)
    raise ValueError(f"Unknown smoother '{smoother}'.")


def _recursive_gaussian_smooth_2d(signal: torch.Tensor, sigma_bins: torch.Tensor) -> torch.Tensor:
    out = signal.clone()
    sigma_x = float(sigma_bins[0].item())
    sigma_y = float(sigma_bins[1].item())
    for idx in range(out.shape[0]):
        out[idx, :] = _recursive_gaussian_line(out[idx, :], sigma_y)
    tmp = out.clone()
    for idx in range(tmp.shape[1]):
        tmp[:, idx] = _recursive_gaussian_line(tmp[:, idx], sigma_x)
    return tmp


def _smooth_2d(signal: torch.Tensor, sigma_bins: torch.Tensor, smoother: str = "auto") -> torch.Tensor:
    smoother = smoother.lower()
    if smoother == "auto":
        smoother = (
            "fft"
            if signal.shape[0] >= 256 or signal.shape[1] >= 256 or float(sigma_bins.max()) >= 16.0
            else "conv"
        )
    if smoother == "recursive":
        return _recursive_gaussian_smooth_2d(signal, sigma_bins)
    kernel_x = _gaussian_kernel1d(sigma_bins[0], dtype=signal.dtype, device=signal.device)
    kernel_y = _gaussian_kernel1d(sigma_bins[1], dtype=signal.dtype, device=signal.device)
    if smoother == "fft":
        return _fft_conv_same_2d(signal, torch.outer(kernel_x, kernel_y))
    if smoother != "conv":
        raise ValueError(f"Unknown smoother '{smoother}'.")
    pad_x = kernel_x.numel() // 2
    pad_y = kernel_y.numel() // 2
    out = F.conv2d(
        signal.view(1, 1, *signal.shape),
        kernel_x.view(1, 1, -1, 1),
        padding=(pad_x, 0),
    )
    out = F.conv2d(
        out,
        kernel_y.view(1, 1, 1, -1),
        padding=(0, pad_y),
    )
    return out.view_as(signal)


def _linear_bin_1d(x: torch.Tensor, num_step_grid: int, x_min: float, x_max: float) -> torch.Tensor:
    x = x.view(-1).to(dtype=torch.float64)
    step = max((x_max - x_min) / max(num_step_grid - 1, 1), _EPS)
    idx = ((x - x_min) / step).clamp(0.0, num_step_grid - 1.0)
    i0 = idx.floor().to(dtype=torch.long)
    w1 = (idx - i0.to(dtype=x.dtype)).clamp(0.0, 1.0)
    i1 = torch.clamp(i0 + 1, max=num_step_grid - 1)
    w0 = 1.0 - w1
    counts = torch.zeros(num_step_grid, device=x.device, dtype=x.dtype)
    counts.scatter_add_(0, i0, w0)
    counts.scatter_add_(0, i1, w1)
    return counts


def _bilinear_bin_2d_unit(obs: torch.Tensor, num_step_grid: int) -> torch.Tensor:
    obs = obs.to(dtype=torch.float64)
    step = 1.0 / max(num_step_grid - 1, 1)
    idx = (obs / step).clamp(0.0, num_step_grid - 1.0)
    i0 = idx.floor().to(dtype=torch.long)
    frac = idx - i0.to(dtype=obs.dtype)
    i1 = torch.clamp(i0 + 1, max=num_step_grid - 1)
    wx0 = 1.0 - frac[:, 0]
    wy0 = 1.0 - frac[:, 1]
    wx1 = frac[:, 0]
    wy1 = frac[:, 1]
    flat = torch.zeros(num_step_grid * num_step_grid, device=obs.device, dtype=obs.dtype)
    stride = num_step_grid
    flat.scatter_add_(0, i0[:, 0] * stride + i0[:, 1], wx0 * wy0)
    flat.scatter_add_(0, i1[:, 0] * stride + i0[:, 1], wx1 * wy0)
    flat.scatter_add_(0, i0[:, 0] * stride + i1[:, 1], wx0 * wy1)
    flat.scatter_add_(0, i1[:, 0] * stride + i1[:, 1], wx1 * wy1)
    return flat.view(num_step_grid, num_step_grid)


def _bilinear_bin_2d_rect(
    obs: torch.Tensor,
    *,
    num_step_grid: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> torch.Tensor:
    obs = obs.to(dtype=torch.float64)
    step_x = max((x_max - x_min) / max(num_step_grid - 1, 1), _EPS)
    step_y = max((y_max - y_min) / max(num_step_grid - 1, 1), _EPS)
    idx_x = ((obs[:, 0] - x_min) / step_x).clamp(0.0, num_step_grid - 1.0)
    idx_y = ((obs[:, 1] - y_min) / step_y).clamp(0.0, num_step_grid - 1.0)
    i0_x = idx_x.floor().to(dtype=torch.long)
    i0_y = idx_y.floor().to(dtype=torch.long)
    i1_x = torch.clamp(i0_x + 1, max=num_step_grid - 1)
    i1_y = torch.clamp(i0_y + 1, max=num_step_grid - 1)
    wx1 = idx_x - i0_x.to(dtype=obs.dtype)
    wy1 = idx_y - i0_y.to(dtype=obs.dtype)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    flat = torch.zeros(num_step_grid * num_step_grid, device=obs.device, dtype=obs.dtype)
    stride = num_step_grid
    flat.scatter_add_(0, i0_x * stride + i0_y, wx0 * wy0)
    flat.scatter_add_(0, i1_x * stride + i0_y, wx1 * wy0)
    flat.scatter_add_(0, i0_x * stride + i1_y, wx0 * wy1)
    flat.scatter_add_(0, i1_x * stride + i1_y, wx1 * wy1)
    return flat.view(num_step_grid, num_step_grid)


def _mirror_grid_2d(grid: torch.Tensor) -> torch.Tensor:
    flip0 = grid.flip(0)
    flip1 = grid.flip(1)
    flip01 = grid.flip(0, 1)
    return torch.cat(
        [
            torch.cat([flip01, flip0, flip01], dim=1),
            torch.cat([flip1, grid, flip1], dim=1),
            torch.cat([flip01, flip0, flip01], dim=1),
        ],
        dim=0,
    )


def _build_pdf_buffers_1d(
    *,
    pdf: torch.Tensor,
    grid_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    step = float(grid_x[1] - grid_x[0])
    cdf = (pdf * step).cumsum(dim=0)
    cdf /= cdf[-1].clamp_min(_EPS)
    delta_x = torch.full((grid_x.numel() - 1,), step, dtype=pdf.dtype, device=pdf.device)
    delta_cdf = (cdf[1:] - cdf[:-1]).clamp_min(_EPS)
    slope_fwd = delta_cdf / delta_x
    slope_inv = delta_x / delta_cdf
    slope_pdf = (pdf[1:] - pdf[:-1]) / delta_x
    return cdf, slope_fwd, slope_inv, slope_pdf


def _trapezoid_weights(
    num_step_grid: int,
    *,
    step: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.full((num_step_grid,), step, dtype=dtype, device=device)
    if num_step_grid > 0:
        weights[0] = 0.5 * step
        weights[-1] = 0.5 * step
    return weights


def _normalize_copula_pdf_grid(
    pdf_grid: torch.Tensor,
    *,
    step: float,
    marginal_tol: float,
    num_iter_max: int,
) -> torch.Tensor:
    pdf_grid = pdf_grid.clamp_min(_EPS)
    weights = _trapezoid_weights(
        pdf_grid.shape[0],
        step=step,
        dtype=pdf_grid.dtype,
        device=pdf_grid.device,
    )
    pdf_grid *= weights.sum() / (pdf_grid * weights.view(1, -1)).sum(dim=1, keepdim=True).clamp_min(_EPS)
    pdf_grid *= weights.sum() / (pdf_grid * weights.view(-1, 1)).sum(dim=0, keepdim=True).clamp_min(_EPS)
    pdf_grid /= (
        (pdf_grid * torch.outer(weights, weights)).sum().clamp_min(_EPS)
    )
    err = torch.maximum(
        ((pdf_grid * weights.view(-1, 1)).sum(dim=0) - 1.0).abs().max(),
        ((pdf_grid * weights.view(1, -1)).sum(dim=1) - 1.0).abs().max(),
    )
    if float(err.item()) > marginal_tol:
        for _ in range(num_iter_max):
            pdf_grid *= (
                weights.sum()
                / (pdf_grid * weights.view(-1, 1)).sum(dim=0, keepdim=True).clamp_min(_EPS)
            )
            pdf_grid *= (
                weights.sum()
                / (pdf_grid * weights.view(1, -1)).sum(dim=1, keepdim=True).clamp_min(_EPS)
            )
            pdf_grid /= (
                (pdf_grid * torch.outer(weights, weights)).sum().clamp_min(_EPS)
            )
            err = torch.maximum(
                ((pdf_grid * weights.view(-1, 1)).sum(dim=0) - 1.0).abs().max(),
                ((pdf_grid * weights.view(1, -1)).sum(dim=1) - 1.0).abs().max(),
            )
            if float(err.item()) <= marginal_tol:
                break
    if not torch.isfinite(pdf_grid).all():
        raise RuntimeError("Non-finite copula density grid after normalization.")
    return pdf_grid


def _build_pdf_buffers_2d(
    *,
    pdf_grid: torch.Tensor,
    step: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_step_grid = pdf_grid.shape[0]
    axis = torch.linspace(0.0, 1.0, steps=num_step_grid, dtype=pdf_grid.dtype, device=pdf_grid.device)
    cdf_grid = torch.zeros_like(pdf_grid)
    cell_mass = (
        0.25
        * step**2
        * (
            pdf_grid[:-1, :-1]
            + pdf_grid[1:, :-1]
            + pdf_grid[:-1, 1:]
            + pdf_grid[1:, 1:]
        )
    )
    if cell_mass.numel():
        cdf_grid[1:, 1:] = cell_mass.cumsum(dim=0).cumsum(dim=1)
    cdf_grid[0, :] = 0.0
    cdf_grid[:, 0] = 0.0
    cdf_grid[:, -1] = axis
    cdf_grid[-1, :] = axis
    cdf_grid[-1, -1] = 1.0
    cdf_grid.clamp_(0.0, 1.0)

    hfunc_l_grid = torch.zeros_like(pdf_grid)
    hfunc_r_grid = torch.zeros_like(pdf_grid)
    if num_step_grid > 1:
        hfunc_l_grid[:, 1:] = (0.5 * step * (pdf_grid[:, :-1] + pdf_grid[:, 1:])).cumsum(dim=1)
        hfunc_r_grid[1:, :] = (0.5 * step * (pdf_grid[:-1, :] + pdf_grid[1:, :])).cumsum(dim=0)
    hfunc_l_grid[:, 0] = 0.0
    hfunc_l_grid[:, -1] = 1.0
    hfunc_r_grid[0, :] = 0.0
    hfunc_r_grid[-1, :] = 1.0
    hfunc_l_grid.clamp_(0.0, 1.0)
    hfunc_r_grid.clamp_(0.0, 1.0)
    return cdf_grid, hfunc_l_grid, hfunc_r_grid


def fit_grid_reflect_bicop(
    obs: torch.Tensor,
    *,
    num_step_grid: int,
    bandwidth: str | float | torch.Tensor,
    bandwidth_scale: float,
    smoother: str,
    marginal_tol: float,
    num_iter_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS)
    step = 1.0 / max(num_step_grid - 1, 1)
    if isinstance(bandwidth, str):
        if bandwidth in {"silverman", "isj"}:
            h = _silverman_bandwidth_2d(obs)
        else:
            raise ValueError("Unsupported copula bandwidth.")
    else:
        h = torch.as_tensor(bandwidth, device=obs.device, dtype=obs.dtype).view(-1)
        if h.numel() == 1:
            h = h.repeat(2)
    h = (h * float(bandwidth_scale)).clamp_min(step)
    sigma_bins = (h / step).clamp_min(0.5)
    hist = _bilinear_bin_2d_unit(obs, num_step_grid)
    mirrored = _mirror_grid_2d(hist)
    smoothed = _smooth_2d(mirrored, sigma_bins=sigma_bins, smoother=smoother)
    start = num_step_grid
    stop = 2 * num_step_grid
    pdf_grid = smoothed[start:stop, start:stop]
    pdf_grid = _normalize_copula_pdf_grid(
        pdf_grid,
        step=step,
        marginal_tol=marginal_tol,
        num_iter_max=num_iter_max,
    )
    cdf_grid, hfunc_l_grid, hfunc_r_grid = _build_pdf_buffers_2d(pdf_grid=pdf_grid, step=step)
    return pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, h


def _interp_rect_2d(
    grid: torch.Tensor,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> torch.Tensor:
    step_x = max((x_max - x_min) / max(grid.shape[0] - 1, 1), _EPS)
    step_y = max((y_max - y_min) / max(grid.shape[1] - 1, 1), _EPS)
    idx_x = ((x - x_min) / step_x).clamp(0.0, grid.shape[0] - 1.0)
    idx_y = ((y - y_min) / step_y).clamp(0.0, grid.shape[1] - 1.0)
    i0_x = idx_x.floor().to(dtype=torch.long)
    i0_y = idx_y.floor().to(dtype=torch.long)
    i1_x = torch.clamp(i0_x + 1, max=grid.shape[0] - 1)
    i1_y = torch.clamp(i0_y + 1, max=grid.shape[1] - 1)
    dx = idx_x - i0_x.to(dtype=x.dtype)
    dy = idx_y - i0_y.to(dtype=y.dtype)
    g00 = grid[i0_x, i0_y]
    g10 = grid[i1_x, i0_y]
    g01 = grid[i0_x, i1_y]
    g11 = grid[i1_x, i1_y]
    return g00 + (g10 - g00) * dx + (g01 - g00) * dy + (g11 - g01 - g10 + g00) * dx * dy


def fit_grid_probit_bicop(
    obs: torch.Tensor,
    *,
    num_step_grid: int,
    bandwidth: str | float | torch.Tensor,
    bandwidth_scale: float,
    smoother: str,
    marginal_tol: float,
    num_iter_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS)
    z = _normal_ppf(obs)
    if isinstance(bandwidth, str):
        if bandwidth in {"silverman", "isj"}:
            h = _silverman_bandwidth_2d(z)
        else:
            raise ValueError("Unsupported copula bandwidth.")
    else:
        h = torch.as_tensor(bandwidth, device=z.device, dtype=z.dtype).view(-1)
        if h.numel() == 1:
            h = h.repeat(2)
    h = (h * float(bandwidth_scale)).clamp_min(_EPS)
    q_lo = torch.quantile(z, 0.001, dim=0)
    q_hi = torch.quantile(z, 0.999, dim=0)
    z_lim_x = max(4.0, abs(float(q_lo[0].item())), abs(float(q_hi[0].item())))
    z_lim_y = max(4.0, abs(float(q_lo[1].item())), abs(float(q_hi[1].item())))
    hist = _bilinear_bin_2d_rect(
        z,
        num_step_grid=num_step_grid,
        x_min=-z_lim_x,
        x_max=z_lim_x,
        y_min=-z_lim_y,
        y_max=z_lim_y,
    )
    step_x = (2.0 * z_lim_x) / max(num_step_grid - 1, 1)
    step_y = (2.0 * z_lim_y) / max(num_step_grid - 1, 1)
    sigma_bins = torch.tensor(
        [float(h[0].item()) / max(step_x, _EPS), float(h[1].item()) / max(step_y, _EPS)],
        device=z.device,
        dtype=z.dtype,
    ).clamp_min(0.5)
    z_pdf_grid = _smooth_2d(hist, sigma_bins=sigma_bins, smoother=smoother)
    z_pdf_grid /= z_pdf_grid.sum().clamp_min(_EPS) * step_x * step_y

    u_axis = torch.linspace(_EPS, 1.0 - _EPS, num_step_grid, device=z.device, dtype=z.dtype)
    z_axis = _normal_ppf(u_axis)
    zz_l, zz_r = torch.meshgrid(z_axis, z_axis, indexing="ij")
    canonical = _interp_rect_2d(
        z_pdf_grid,
        x=zz_l.reshape(-1),
        y=zz_r.reshape(-1),
        x_min=-z_lim_x,
        x_max=z_lim_x,
        y_min=-z_lim_y,
        y_max=z_lim_y,
    ).view(num_step_grid, num_step_grid)
    jac = torch.outer(_normal_pdf(z_axis), _normal_pdf(z_axis)).clamp_min(_EPS)
    pdf_grid = canonical / jac
    step = 1.0 / max(num_step_grid - 1, 1)
    pdf_grid = _normalize_copula_pdf_grid(
        pdf_grid,
        step=step,
        marginal_tol=marginal_tol,
        num_iter_max=num_iter_max,
    )
    cdf_grid, hfunc_l_grid, hfunc_r_grid = _build_pdf_buffers_2d(pdf_grid=pdf_grid, step=step)
    return pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, h
