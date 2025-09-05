"""
bandwidth.py

Pure-torch bandwidth selectors:
- isj_bandwidth(x):         Improved Sheather–Jones plug-in
- icv_bandwidth(x, folds):  Indirect cross-validation
"""

import torch
import math
from .constants import _DEFAULT_CV_FOLDS, _ISJ_GRID_SIZE, _ISJ_MAX_ITER, _ISJ_TOL, _EPS, _KFOLD_LSCV_GRID_SIZE


# Improved Sheather–Jones (ISJ)

def isj_bandwidth(x: torch.Tensor, grid_size: int = _ISJ_GRID_SIZE, max_iter: int = _ISJ_MAX_ITER, tol: float = _ISJ_TOL) -> torch.Tensor:
    """
    ISJ plug-in selector using torch:
    1) Estimate variance, fourth derivative functional via small-bandwidth torch FFT.
    2) Solve the SJ fixed-point equation for h with torch root finding.
    Returns h (scalar)
    """
    n = x.numel()
    # 1. Initial pilot h using normal-reference rule   
    # Silverman pilot
    std = x.std(unbiased=True).clamp_min(1e-12)
    silverman = 1.06 * std * (n ** (-1.0/5.0))
    h = silverman.clone()

    # Precompute grid
    x_min, x_max = torch.min(x), torch.max(x)
    L = (x_max - x_min).clamp_min(1e-12) 
    grid = torch.linspace(x_min - 0.5 * std, x_max + 0.5 * std, grid_size, device=x.device)
    dx = grid[1] - grid[0]

    # Precompute k^2 frequencies for DCT/FFT
    k = torch.arange(grid_size, device=x.device)
    k_sq = (k * torch.pi / L) ** 2

    R_K = 1.0 / (2.0 * torch.sqrt(torch.pi))  # for Gaussian
    mu2_K = 1.0

    for _ in range(max_iter):
        # 2. Bin the data onto the grid
        bins = torch.bucketize(x, grid)
        counts = torch.zeros(grid_size, device=x.device).scatter_add_(0, bins, torch.ones_like(x, device=x.device))
        relfreq = counts / n

        # 3. DCT of the relative frequency
        a_k = torch.fft.dct(relfreq, type=2, norm='ortho')  # requires PyTorch >= 1.8

        # 4. Estimate R(f'') via the DCT coefficients
        t = h**2 / 2.0
        R2 = 0.5 * torch.pi ** 4 * torch.sum(k_sq ** 2 * a_k ** 2 * torch.exp(-k_sq * t))  # Equation from ArviZ (https://python.arviz.org/en/stable/_modules/arviz/stats/density_utils.html)

        # if R2 becomes tiny/NaN, bail out to Silverman
        if not torch.isfinite(R2) or R2 <= 0:
            h = silverman
            break

        # 5. Fixed-point update
        h_new = (R_K / (mu2_K ** 2 * R2 * n)) ** 0.2
        if torch.abs(h_new - h) < tol:
            h = h_new
            break
        h = h_new

        # keep h positive and not ridiculously smaller than grid spacing
        h = h.clamp_min(dx * 1e-3)

    return h


# K-fold cross-validation

def _make_kfold_indices(n: int, folds: int = _DEFAULT_CV_FOLDS, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    parts = torch.chunk(perm, folds)
    for i in range(folds):
        val_idx = parts[i]
        train_idx = torch.cat([parts[j] for j in range(folds) if j != i], dim=0)
        yield train_idx, val_idx

@torch.no_grad()
def kfold_lcv_bandwidth(
    x: torch.Tensor,
    h_grid: torch.Tensor | None = None,
    folds: int = _DEFAULT_CV_FOLDS,
) -> torch.Tensor:
    """
    Likelihood CV for 1D Gaussian KDE:
      minimize -(1/|val|) * sum_{val} log f_{train,h}(x)
    Torch-only; numerically stable via log-sum-exp.
    """
    x = x.view(-1).to(dtype=torch.float64)
    n = x.numel()

    # Default grid around Silverman pilot if not provided
    if h_grid is None:
        std = x.std(unbiased=True).clamp_min(1e-12)
        h0 = 1.06 * std * (n ** (-1.0 / 5.0))
        h_grid = torch.logspace(
            math.log10(0.5 * h0), math.log10(2.0 * h0),
            steps=_KFOLD_LSCV_GRID_SIZE, dtype=torch.float64, device=x.device
        )
    else:
        h_grid = h_grid.to(dtype=torch.float64, device=x.device)

    TWO_PI = 2.0 * math.pi
    best_loss = torch.tensor(float("inf"), dtype=torch.float64, device=x.device)
    best_h = h_grid[0]

    for h in h_grid:
        fold_losses = []
        for tr_idx, va_idx in _make_kfold_indices(n, folds=folds, seed=0):
            x_tr = x[tr_idx]  # [n_tr]
            x_va = x[va_idx]  # [n_va]

            # log f_h(x_va) = log( (1/n_tr) * sum_j phi((x_va - x_tr[j])/h) / h )
            # = logsumexp_j( -0.5*((x_va - x_tr[j])/h)^2 - log(h*sqrt(2π)) ) - log(n_tr)
            dif = (x_va.unsqueeze(1) - x_tr.unsqueeze(0)) / h  # [n_va, n_tr]
            log_k = -0.5 * dif.pow(2) - 0.5 * math.log(TWO_PI) - math.log(h)
            log_f = torch.logsumexp(log_k, dim=1) - math.log(x_tr.numel())
            fold_losses.append(-(log_f.mean()))
        loss = torch.stack(fold_losses).mean()
        if loss < best_loss:
            best_loss = loss
            best_h = h

    return best_h

def optimal_bandwidth(x: torch.Tensor, method: str = 'isj', **kwargs) -> torch.Tensor:
    """
    Choose bandwidth via:
      'isj'    -> isj_bandwidth (if available), else Silverman
      'kfold'  -> kfold_lcv_bandwidth
      'auto'   -> ISJ/Silverman -> refine ±50% via LCV
    """
    x = x.view(-1)
    n = x.numel()
    std = x.std(unbiased=True).clamp_min(1e-12)
    silverman = 1.06 * std * (n ** (-1.0 / 5.0))

    if method == 'isj':
        try:
            return isj_bandwidth(x)
        except Exception:
            return silverman

    if method == 'kfold':
        return kfold_lcv_bandwidth(x, kwargs.get('h_grid', None), kwargs.get('folds', _DEFAULT_CV_FOLDS))

    if method == 'auto':
        try:
            h0 = isj_bandwidth(x)
        except Exception:
            h0 = silverman
        # refine ±50% around h0
        h_grid = torch.logspace(
            math.log10(0.5 * h0), math.log10(1.5 * h0),
            steps=64, dtype=torch.float64, device=x.device
        )
        return kfold_lcv_bandwidth(x, h_grid=h_grid, folds=kwargs.get('folds', _DEFAULT_CV_FOLDS))

    raise ValueError(f"Unknown method {method!r}")

