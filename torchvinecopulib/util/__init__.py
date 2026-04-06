"""
torchvinecopulib.util
----------------------
Utility routines for dependence measures, empirical pseudo-observations, and root-finding.
Torch-native marginal and bicop KDE implementations live under ``torchvinecopulib.backends`` and
are re-exported here for backward compatibility.
"""

from __future__ import annotations

import enum
from typing import Callable

import torch

from ..backends.bicop import TorchCopulaKDE2D
from ..backends.common import _EPS, fit_grid_reflect_bicop
from ..backends.marginal import TorchKDE1D

__all__ = [
    "ENUM_FUNC_BIDEP",
    "TorchKDE1D",
    "TorchCopulaKDE2D",
    "chatterjee_xi",
    "empirical_pobs",
    "ferreira_tail_dep_coeff",
    "kendall_tau",
    "mutual_info",
    "solve_ITP",
    "torch_copula_kde_grid",
]


def _lazy_kendalltau():
    try:
        from scipy.stats import kendalltau as scipy_kendalltau
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "kendall_tau requires scipy. Install torchvinecopulib with scipy available."
        ) from exc
    return scipy_kendalltau


def empirical_pobs(x: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, 1)
    ranks = x.argsort(dim=0).argsort(dim=0).to(dtype=torch.float64) + 1.0
    out_dtype = x.dtype if x.is_floating_point() else torch.float64
    return (ranks / (x.shape[0] + 1.0)).to(device=x.device, dtype=out_dtype)


@torch.no_grad()
def kendall_tau(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    scipy_kendalltau = _lazy_kendalltau()
    return torch.as_tensor(
        scipy_kendalltau(x.view(-1).cpu(), y.view(-1).cpu()),
        dtype=x.dtype,
        device=x.device,
    )


@torch.no_grad()
def ferreira_tail_dep_coeff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        3.0
        - (
            1.0
            - torch.stack([torch.maximum(x, y), torch.maximum(1.0 - x, y)], dim=1)
            .mean(dim=0)
            .clamp(0.5, 0.6666666666666666)
            .min()
        ).reciprocal()
    )


@torch.no_grad()
def chatterjee_xi(x: torch.Tensor, y: torch.Tensor, M: int = 1) -> torch.Tensor:
    xrank, yrank = (
        x.argsort(dim=0).argsort(dim=0) + 1,
        y.argsort(dim=0).argsort(dim=0) + 1,
    )
    xrank, yrank = xrank[yrank.argsort(dim=0)], yrank[xrank.argsort(dim=0)]

    def xy_sum(m: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            (torch.minimum(xrank[:-m], xrank[m:])).sum() + xrank[-m:].sum(),
            (torch.minimum(yrank[:-m], yrank[m:])).sum() + yrank[-m:].sum(),
        )

    n = x.shape[0]
    return -2.0 + 24.0 * (
        torch.as_tensor([xy_sum(m) for m in range(1, M + 1)], device=x.device, dtype=x.dtype)
        .sum(dim=0)
        .max()
    ) / (M * (1.0 + n) * (1.0 + M + 4.0 * n))


@torch.no_grad()
def torch_copula_kde_grid(
    obs: torch.Tensor,
    *,
    num_step_grid: int = 128,
    bandwidth: str | float | torch.Tensor = "silverman",
    bandwidth_scale: float = 1.0,
    smoother: str = "auto",
    marginal_tol: float = 1e-3,
    num_iter_max: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return fit_grid_reflect_bicop(
        obs=obs,
        num_step_grid=num_step_grid,
        bandwidth=bandwidth,
        bandwidth_scale=bandwidth_scale,
        smoother=smoother,
        marginal_tol=marginal_tol,
        num_iter_max=num_iter_max,
    )


@torch.no_grad()
def mutual_info(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_step_grid: int = 128,
    bandwidth_scale: float = 1.0,
) -> torch.Tensor:
    obs = torch.hstack([empirical_pobs(x), empirical_pobs(y)]).to(device=x.device, dtype=x.dtype)
    pdf_grid, *_ = torch_copula_kde_grid(
        obs=obs,
        num_step_grid=num_step_grid,
        bandwidth="silverman",
        bandwidth_scale=bandwidth_scale,
    )
    step = 1.0 / max(num_step_grid - 1, 1)
    pdf_grid = pdf_grid.to(device=x.device, dtype=x.dtype)
    return (pdf_grid * pdf_grid.clamp_min(_EPS).log()).sum() * step**2


class ENUM_FUNC_BIDEP(enum.Enum):
    chatterjee_xi = enum.member(chatterjee_xi)
    ferreira_tail_dep_coeff = enum.member(ferreira_tail_dep_coeff)
    kendall_tau = enum.member(kendall_tau)
    mutual_info = enum.member(mutual_info)

    def __call__(self, x: torch.Tensor, y: torch.Tensor, **kw):
        return self.value(x, y, **kw)


@torch.no_grad()
def solve_ITP(
    fun: Callable[[torch.Tensor], torch.Tensor],
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    eps_2: float = 1e-10,
    k_1: float = 0.1,
    k_2: float = 2.0,
    n_0: int = 1,
) -> torch.Tensor:
    x_a, x_b = x_a.clone(), x_b.clone()
    y_a, y_b = fun(x_a), fun(x_b)
    x_wid = x_b - x_a
    n_max = torch.ceil(torch.log2(x_wid / (2.0 * eps_2))).to(dtype=torch.int)
    n_max_half = n_max + n_0
    for jdx in range(int(n_max.max().item())):
        idx = (x_wid > 2.0 * eps_2) & (~((y_a == 0.0) | (y_b == 0.0)))
        if not idx.any():
            break
        x_half = (x_a + x_b) / 2.0
        r = eps_2 * (2.0 ** (n_max_half - jdx)) - x_wid / 2.0
        x_f = (y_b * x_a - y_a * x_b) / (y_b - y_a)
        sigma = torch.sign(x_half - x_f)
        delta = k_1 * x_wid.pow(k_2)
        x_t = x_f + sigma * delta
        idx_r = (x_t - x_half).abs() > r
        x_itp = x_t
        x_itp[idx_r] = x_half[idx_r] - sigma[idx_r] * r[idx_r]
        y_itp = fun(x_itp)
        idx_a = idx & (y_itp * y_a > 0.0)
        idx_b = idx & (y_itp * y_b > 0.0)
        x_a[idx_a], y_a[idx_a] = x_itp[idx_a], y_itp[idx_a]
        x_b[idx_b], y_b[idx_b] = x_itp[idx_b], y_itp[idx_b]
        x_wid[idx] = x_b[idx] - x_a[idx]
    return (x_a + x_b) / 2.0
