"""Public helper routines for dependence measures and root finding.

The standalone torch-native KDE classes belong to :mod:`torchvinecopulib.backends`.
"""

from __future__ import annotations

import enum
import math
from typing import Callable, Literal

import torch

from ..backends.common import _EPS, fit_grid_reflect_bicop

__all__ = [
    "ENUM_FUNC_BIDEP",
    "chatterjee_xi",
    "empirical_pobs",
    "ferreira_tail_dep_coeff",
    "kendall_tau",
    "kendall_tau_matrix",
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


def _resolve_kendall_backend(
    *,
    x: torch.Tensor,
    backend: Literal["auto", "scipy", "torch"],
) -> Literal["scipy", "torch"]:
    if backend == "auto":
        return "torch" if x.device.type == "cuda" and x.shape[0] <= 4096 else "scipy"
    return backend


def _kendall_tau_pvalue_approx(tau: torch.Tensor, num_obs: int) -> torch.Tensor:
    if num_obs < 2:
        return torch.ones_like(tau)
    denom = max(num_obs * (num_obs - 1), 1)
    var = 2.0 * (2.0 * num_obs + 5.0) / (9.0 * denom)
    z = tau.abs() / math.sqrt(max(var, _EPS))
    return torch.special.erfc(z / math.sqrt(2.0)).clamp(0.0, 1.0)


def kendall_tau_matrix(
    x: torch.Tensor,
    *,
    backend: Literal["auto", "scipy", "torch"] = "auto",
    max_scratch_bytes: int = 256 * 1024 * 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x if x.ndim == 2 else x.view(x.shape[0], -1)
    resolved_backend = _resolve_kendall_backend(x=x, backend=backend)
    num_obs, num_dim = x.shape
    if num_dim == 0:
        empty = torch.empty(0, 0, device=x.device, dtype=x.dtype)
        return empty, empty
    if resolved_backend == "scipy":
        tau = torch.empty(num_dim, num_dim, dtype=x.dtype, device=x.device)
        pvalue = torch.empty_like(tau)
        for idx in range(num_dim):
            tau[idx, idx] = 1.0
            pvalue[idx, idx] = 0.0
            for jdx in range(idx + 1, num_dim):
                stat = kendall_tau(x[:, [idx]], x[:, [jdx]], backend="scipy")
                tau[idx, jdx] = tau[jdx, idx] = stat[0]
                pvalue[idx, jdx] = pvalue[jdx, idx] = stat[1]
        return tau, pvalue

    x_work = x.to(dtype=torch.float64)
    concordance = torch.zeros((num_dim, num_dim), dtype=torch.float64, device=x.device)
    ordered_ties = torch.zeros((num_dim,), dtype=torch.float64, device=x.device)
    # `diff` and the sign reduction both stay in float64 so concordance counts remain exact.
    bytes_per_scalar = 2 * x_work.element_size()
    block_rows = max(
        1, min(num_obs, max_scratch_bytes // max(num_obs * num_dim * bytes_per_scalar, 1))
    )
    for idx in range(0, num_obs, block_rows):
        block = x_work[idx : idx + block_rows]
        diff = block[:, None, :] - x_work[None, :, :]
        signs = torch.sign(diff)
        concordance += torch.einsum("bnk,bnl->kl", signs, signs)
        ordered_ties += diff.eq(0.0).sum(dim=(0, 1)).to(dtype=torch.float64)
    ordered_ties -= float(num_obs)
    denom_vec = (num_obs * (num_obs - 1) - ordered_ties).clamp_min(1.0)
    denom = torch.sqrt(denom_vec[:, None] * denom_vec[None, :])
    tau = (concordance / denom).clamp(-1.0, 1.0).to(dtype=x.dtype)
    tau.fill_diagonal_(1.0)
    pvalue = _kendall_tau_pvalue_approx(tau=tau, num_obs=num_obs)
    pvalue.fill_diagonal_(0.0)
    return tau, pvalue


@torch.no_grad()
def kendall_tau(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    backend: Literal["auto", "scipy", "torch"] = "auto",
    max_scratch_bytes: int = 256 * 1024 * 1024,
) -> torch.Tensor:
    resolved_backend = _resolve_kendall_backend(x=x, backend=backend)
    if resolved_backend == "scipy":
        scipy_kendalltau = _lazy_kendalltau()
        return torch.as_tensor(
            scipy_kendalltau(x.view(-1).cpu(), y.view(-1).cpu()),
            dtype=x.dtype,
            device=x.device,
        )
    tau, pvalue = kendall_tau_matrix(
        torch.hstack([x.view(-1, 1), y.view(-1, 1)]),
        backend="torch",
        max_scratch_bytes=max_scratch_bytes,
    )
    return torch.stack([tau[0, 1], pvalue[0, 1]])


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
    max_iter: int = 64,
    failure_policy: Literal["raise", "linear", "midpoint"] = "raise",
    return_status: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x_a, x_b = x_a.clone(), x_b.clone()
    x_lo = torch.minimum(x_a, x_b)
    x_hi = torch.maximum(x_a, x_b)
    x_a, x_b = x_lo.clone(), x_hi.clone()
    y_a, y_b = fun(x_a), fun(x_b)
    x_wid = x_b - x_a
    bracketed = (y_a * y_b <= 0.0) | (y_a == 0.0) | (y_b == 0.0)
    safe_wid = x_wid.clamp_min(2.0 * eps_2)
    n_max = torch.ceil(torch.log2(safe_wid / (2.0 * eps_2))).to(dtype=torch.int)
    n_max_half = n_max + n_0
    converged = ~bracketed
    used_fallback = ~bracketed
    max_steps = max_iter
    iter_count = torch.zeros_like(x_a, dtype=torch.int)
    for jdx in range(max_steps):
        idx = bracketed & (x_wid > 2.0 * eps_2) & (~((y_a == 0.0) | (y_b == 0.0)))
        if not idx.any():
            break
        iter_count[idx] += 1
        x_half = (x_a + x_b) / 2.0
        r = eps_2 * (2.0 ** (n_max_half - jdx)) - x_wid / 2.0
        denom = torch.where((y_b - y_a).abs() <= _EPS, torch.ones_like(y_a), y_b - y_a)
        x_f = (y_b * x_a - y_a * x_b) / denom
        sigma = torch.sign(x_half - x_f)
        delta = k_1 * x_wid.pow(k_2)
        x_t = x_f + sigma * delta
        idx_r = (x_t - x_half).abs() > r
        x_itp = x_t
        x_itp[idx_r] = x_half[idx_r] - sigma[idx_r] * r[idx_r]
        x_itp = x_itp.clamp(min=x_lo, max=x_hi)
        y_itp = fun(x_itp)
        idx_root = idx & (y_itp == 0.0)
        x_a[idx_root], y_a[idx_root] = x_itp[idx_root], y_itp[idx_root]
        x_b[idx_root], y_b[idx_root] = x_itp[idx_root], y_itp[idx_root]
        idx_a = idx & (y_itp * y_a > 0.0)
        idx_b = idx & (y_itp * y_b > 0.0)
        x_a[idx_a], y_a[idx_a] = x_itp[idx_a], y_itp[idx_a]
        x_b[idx_b], y_b[idx_b] = x_itp[idx_b], y_itp[idx_b]
        x_wid[idx] = x_b[idx] - x_a[idx]
    root = ((x_a + x_b) / 2.0).clamp(min=x_lo, max=x_hi)
    converged = bracketed & ((x_wid <= 2.0 * eps_2) | (y_a == 0.0) | (y_b == 0.0))
    failed_unbracketed = ~bracketed
    failed_convergence = bracketed & ~converged
    failed = failed_unbracketed | failed_convergence
    if failed.any():
        used_fallback = used_fallback | failed
        if failure_policy == "raise":
            raise RuntimeError(
                "ITP root finding failed to bracket or converge for one or more samples."
            )
        fallback_denom = y_b - y_a
        fallback_denom = torch.where(
            fallback_denom.abs() <= _EPS,
            torch.where(
                fallback_denom < 0.0,
                -torch.full_like(fallback_denom, _EPS),
                torch.full_like(fallback_denom, _EPS),
            ),
            fallback_denom,
        )
        fallback_linear = (
            x_lo + (0.0 - y_a) * (x_b - x_a) / fallback_denom
            if failure_policy == "linear"
            else 0.5 * (x_lo + x_hi)
        )
        fallback_mid = 0.5 * (x_a + x_b)
        fallback = torch.where(
            failed_unbracketed,
            fallback_linear.clamp(min=x_lo, max=x_hi),
            fallback_mid.clamp(min=x_lo, max=x_hi),
        )
        root = torch.where(failed, fallback, root)
    status = {
        "bracketed": bracketed,
        "converged": converged,
        "used_fallback": used_fallback,
        "iterations": iter_count,
    }
    return (root, status) if return_status else root
