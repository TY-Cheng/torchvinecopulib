"""
torchvinecopulib.bicop
-----------------------
Provides ``BiCop`` (``torch.nn.Module``) for estimating, evaluating, and sampling
from bivariate copulas via ``tll`` or ``fastKDE`` approaches.

Decorators
-----------
* torch.compile() for bilinear interpolation
* torch.no_grad() for fit(), hinv_l(), hinv_r(), sample(), and imshow()

Key Features
-------------
- Fits a copula density on a uniform [0,1]² grid and caches PDF/CDF/h‐functions
- Device‐agnostic: all buffers live on the same device/dtype you fit on
- Fast bilinear interpolation compiled with ``torch.compile``
- Convenient ``.cdf()``, ``.pdf()``, ``.hfunc_*()``, ``.hinv_*()``, and ``.sample()`` APIs
- Plotting helpers: ``.imshow()`` and ``.plot(contour|surface)``

Usage
------
>>> from torchvinecopulib.bicop import BiCop
>>> cop = BiCop(num_step_grid=256)
>>> cop.fit(obs)  # obs: Tensor of shape (n,2) in [0,1]²
>>> u = torch.rand(10, 2)
>>> cdf_vals = cop.cdf(u)
>>> samples = cop.sample(1000, is_sobol=True)

References
-----------
- Nagler, T., Schellhase, C., & Czado, C. (2017). Nonparametric estimation of simplified vine copula models: comparison of methods. Dependence Modeling, 5(1), 99-120.
- O’Brien, T. A., Kashinath, K., Cavanaugh, N. R., Collins, W. D. & O’Brien, J. P. A fast and objective multidimensional kernel density estimation method: fastKDE. Comput. Stat. Data Anal. 101, 148–160 (2016). http://dx.doi.org/10.1016/j.csda.2016.02.014
- O’Brien, T. A., Collins, W. D., Rauscher, S. A. & Ringler, T. D. Reducing the computational cost of the ECF using a nuFFT: A fast and objective probability density estimation method. Comput. Stat. Data Anal. 79, 222–234 (2014). http://dx.doi.org/10.1016/j.csda.2014.06.002
"""

from pprint import pformat
from typing import Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pyvinecopulib as pv
import torch
import math
# from fastkde import pdf as fkpdf  # can remove fastKDE dependency
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats import kendalltau, norm
import torch.nn.functional as F
from ..util.bandwidth import *

from ..util import _EPS, solve_ITP

__all__ = [
    "BiCop",
]


class BiCop(torch.nn.Module):
    # ! hinv
    _EPS: float = _EPS

    def __init__(
        self,
        num_step_grid: int = 128,
    ):
        """Initializes the bivariate copula (BiCop) class. By default an independent bicop.

        Args:
            num_step_grid (int, optional): number of steps per dimension for the precomputed grids (must be a power of 2). Defaults to 128.
        """
        super().__init__()
        # * by default an independent bicop, otherwise cache grids from KDE
        self.is_indep = True
        self.mtd_kde = "tll"  # default method for estimating the copula density
        self.num_step_grid = num_step_grid
        self.register_buffer("tau", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("num_obs", torch.empty((), dtype=torch.int))
        self.register_buffer("negloglik", torch.zeros((), dtype=torch.float64))
        self.register_buffer(
            "_pdf_grid",
            torch.ones(num_step_grid, num_step_grid, dtype=torch.float64),
        )
        self.register_buffer(
            "_cdf_grid",
            torch.empty(num_step_grid, num_step_grid, dtype=torch.float64),
        )
        self.register_buffer(
            "_hfunc_l_grid",
            torch.empty(num_step_grid, num_step_grid, dtype=torch.float64),
        )
        self.register_buffer(
            "_hfunc_r_grid",
            torch.empty(num_step_grid, num_step_grid, dtype=torch.float64),
        )
        # ! device agnostic
        self.register_buffer("_dd", torch.tensor([], dtype=torch.float64))
        self._LOG_EPS = 1e-6   # set this so that log(1e-6) = -13.815510557964274

    @property
    def device(self) -> torch.device:
        """Get the device of the bicop model (all internal buffers).

        Returns:
            torch.device: The device on which the registered buffers reside.
        """
        return self._dd.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the bicop model (all internal buffers). Should be torch.float64.

        Returns:
            torch.dtype: The data type of the registered buffers.
        """
        return self._dd.dtype

    # 2 helper functions to clean data
    def _sanitize_train_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Keep only fully finite rows and clamp to [0,1]^2 for fitting.
        No change on self.num_obs (tests expect original count).
        """
        obs = obs.to(device=self.device, dtype=self.dtype)
        m = torch.isfinite(obs).all(dim=1)
        if not m.any():
            return obs[:0]  # empty [0,2]
        return obs[m].clamp_(0.0, 1.0)

    def _eval_on_finite(self, obs: torch.Tensor, eval_fn, clamp_inputs: bool = True) -> torch.Tensor:
        """
        Numerically-safe evaluation wrapper (tll/fastKDE/torchKDE).

        - Builds an (N,1) output tensor initialized to NaN.
        - Runs `eval_fn` ONLY on rows where (u,v) are fully finite.
        - Optionally clamps those finite rows to [EPS, 1-EPS] to prevent OOB indexing.
        - Fills results back into the corresponding rows; non-finite rows stay NaN.

        Args:
            obs (torch.Tensor): (N,2) query points.
            eval_fn (callable): function mapping (k,2) -> (k,1).
            clamp_inputs (bool): clamp finite (u,v) to [EPS, 1-EPS] before eval, default True.

        Returns:
            torch.Tensor: (N,1) tensor with evaluated results (finite rows) and NaN elsewhere.
        """
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        # ensure a 2D shape first
        if obs.ndim == 1:
            obs = obs.view(-1, 2)
        N = obs.shape[0]
        out = torch.full((N, 1), float("nan"), dtype=self.dtype, device=self.device)

        mask = torch.isfinite(obs).all(dim=1)
        if mask.any():
            z = obs[mask]
            if clamp_inputs:
                z = z.clamp(self._EPS, 1.0 - self._EPS)
            val = eval_fn(z)
            out[mask, 0] = val.view(-1)

        return out

    @torch.no_grad()
    def reset(self) -> None:
        """Reinitialize state and zero all statistics and precomputed grids.

        Sets the bicop back to independent bicop and clears accumulated
        metrics (``tau``, ``num_obs``, ``negloglik``) as well as all grid buffers
        (``_pdf_grid``, ``_cdf_grid``, ``_hfunc_l_grid``, ``_hfunc_r_grid``).
        """
        self.is_indep = True
        self.tau.zero_()
        self.num_obs.zero_()
        self.negloglik.zero_()
        self._pdf_grid.zero_()
        self._cdf_grid.zero_()
        self._hfunc_l_grid.zero_()
        self._hfunc_r_grid.zero_()

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        mtd_kde: str = "tll",
        mtd_tll: str = "constant",
        num_iter_max: int = 17,
        is_tau_est: bool = False,
    ) -> None:
        """Estimate and cache PDF/CDF/h-function grids from bivariate copula observations.

        This method computes KDE-based bicopula densities on a uniform [0,1]² grid and populates internal buffers
        (``_pdf_grid``, ``_cdf_grid``, ``_hfunc_l_grid``, ``_hfunc_r_grid``, ``negloglik``).

        - Nagler, T., Schellhase, C., & Czado, C. (2017). Nonparametric estimation of simplified vine copula models: comparison of methods. Dependence Modeling, 5(1), 99-120.
        - O’Brien, T. A., Kashinath, K., Cavanaugh, N. R., Collins, W. D. & O’Brien, J. P. A fast and objective multidimensional kernel density estimation method: fastKDE. Comput. Stat. Data Anal. 101, 148–160 (2016). http://dx.doi.org/10.1016/j.csda.2016.02.014
        - O’Brien, T. A., Collins, W. D., Rauscher, S. A. & Ringler, T. D. Reducing the computational cost of the ECF using a nuFFT: A fast and objective probability density estimation method. Comput. Stat. Data Anal. 79, 222–234 (2014). http://dx.doi.org/10.1016/j.csda.2014.06.002


        Args:
            obs (torch.Tensor): shape (n, 2) bicop obs in [0, 1]².
            mtd_kde (str, optional): Method for estimating the copula density. One of ("tll", "torchkde", "fastKDE"). Defaults to "tll".
            mtd_tll (str, optional): fit method for the transformation local-likelihood (TLL) nonparametric family, used only when ``mtd_kde="tll"``, one of ("constant", "linear", or "quadratic"). Defaults to "constant".
            num_iter_max (int, optional): num of Sinkhorn/IPF iters for grid normalization, used only when ``mtd_kde="fastKDE"``. Defaults to 17.
            is_tau_est (bool, optional): If True, compute and store Kendall’s τ. Defaults to ``False``.
        """
        # ! device agnostic
        device, dtype = self.device, self.dtype
        self.is_indep = False
        self.mtd_kde = mtd_kde
        self.num_obs.copy_(obs.shape[0])

        # clean training points once
        obs_clean = self._sanitize_train_obs(obs)
        if is_tau_est:
            if obs_clean.numel() == 0:
                self.tau.zero_()
            else:
                self.tau.copy_(torch.as_tensor(
                    kendalltau(obs_clean[:, 0].cpu(), obs_clean[:, 1].cpu()),
                    device=self.device, dtype=self.dtype
                ))
        # grid geometry
        self._target = self.num_step_grid - 1.0
        self.step_grid = 1.0 / self._target

        # ! pdf
        if mtd_kde == "tll":
            controls = pv.FitControlsBicop(
                family_set=[pv.tll],
                num_threads=torch.get_num_threads(),
                nonparametric_method=mtd_tll,
            )
            cop = pv.Bicop.from_data(data=obs_clean.cpu().numpy(), controls=controls)
            axis = torch.linspace(
                _EPS,
                1.0 - _EPS,
                steps=self.num_step_grid,
                device="cpu",
                dtype=torch.float64,
            )
            pdf_grid = (
                torch.from_numpy(cop.pdf(torch.cartesian_prod(axis, axis).view(-1, 2).numpy()))
                .view(self.num_step_grid, self.num_step_grid)
                .to(device=device, dtype=dtype)
            )
        elif mtd_kde == "fastKDE":
            if obs_clean.numel() == 0:
                pdf_grid = torch.ones(self.num_step_grid, self.num_step_grid, dtype=dtype, device=device)
            else:
                pdf_grid = torch.from_numpy(
                    fkpdf(obs_clean[:, 0].cpu(), obs_clean[:, 1].cpu(),
                        num_points=self.num_step_grid * 2 + 1).values
                ).to(device=device, dtype=dtype)
            # * padding/trimming after fastkde.pdf
            H, W = pdf_grid.shape
            if H < self.num_step_grid:
                pdf_grid = torch.cat(
                    [
                        pdf_grid,
                        torch.zeros(self.num_step_grid - H, W, dtype=dtype, device=device),
                    ],
                    dim=0,
                )
            H, W = pdf_grid.shape
            if W < self.num_step_grid:
                pdf_grid = torch.cat(
                    [
                        pdf_grid,
                        torch.zeros(H, self.num_step_grid - W, dtype=dtype, device=device),
                    ],
                    dim=1,
                )
            pdf_grid = pdf_grid[: self.num_step_grid, : self.num_step_grid].clamp_min(0.0)
            pdf_grid = pdf_grid.view(self.num_step_grid, self.num_step_grid).T
            # * normalization: Sinkhorn / iterative proportional fitting (IPF)
            for _ in range(num_iter_max):
                pdf_grid *= self._target / pdf_grid.sum(dim=0, keepdim=True)
                pdf_grid *= self._target / pdf_grid.sum(dim=1, keepdim=True)
            pdf_grid /= pdf_grid.sum() * self.step_grid**2

        elif mtd_kde == "torchKDE":
            if obs_clean.numel() == 0:
                # Independent fallback (finite, valid grids)
                pdf_grid = torch.ones(self.num_step_grid, self.num_step_grid, dtype=dtype, device=device)
            else:
                U = obs_clean[:, 0].to(torch.float64)
                V = obs_clean[:, 1].to(torch.float64)
                n = U.numel()
                G = int(self.num_step_grid)
                tgt = float(self._target)
                dx  = float(self.step_grid)

                # per-axis bandwidths (fallback to Scott)
                try:
                    hx = float(optimal_bandwidth(U, method="auto"))
                    hy = float(optimal_bandwidth(V, method="auto"))
                except Exception:
                    sf = n ** (-1.0 / 6.0)
                    hx = float(sf * U.std(unbiased=True).clamp_min(_EPS))
                    hy = float(sf * V.std(unbiased=True).clamp_min(_EPS))
                
                # avoid kernels narrower than grid spacing
                hx = max(hx, 0.5 * dx)
                hy = max(hy, 0.5 * dx)

                # 2D histogram on the unit square grid
                ix = (U / dx).floor().clamp_(0, G - 1).to(torch.int64)
                iy = (V / dx).floor().clamp_(0, G - 1).to(torch.int64)
                lin = ix * G + iy
                counts = torch.zeros(G * G, dtype=torch.float64, device=device)
                counts.scatter_add_(0, lin, torch.ones_like(lin, dtype=torch.float64, device=device))
                counts = counts.view(G, G)

                # separable Gaussian kernels (truncate at 4 * sigma)
                rx = max(1, int(math.ceil(4.0 * (hx / dx))))
                ry = max(1, int(math.ceil(4.0 * (hy / dx))))
                ox = torch.arange(-rx, rx + 1, dtype=torch.float64, device=device) * dx
                oy = torch.arange(-ry, ry + 1, dtype=torch.float64, device=device) * dx
                kx = torch.exp(-0.5 * (ox / hx) ** 2) / (hx * math.sqrt(2.0 * math.pi))
                ky = torch.exp(-0.5 * (oy / hy) ** 2) / (hy * math.sqrt(2.0 * math.pi))

                # normalize discrete kernels so their (Riemann) sums are 1
                kx = kx / (kx.sum().clamp_min(_EPS) * dx) 
                ky = ky / (ky.sum().clamp_min(_EPS) * dx)

                s = counts.unsqueeze(0).unsqueeze(0)
                s = F.pad(s, (ry, ry, rx, rx), mode="reflect")  # (left,right,top,bottom)
                s = F.conv2d(s, kx.view(1, 1, -1, 1))
                s = F.conv2d(s, ky.view(1, 1, 1, -1))
                pdf_grid = (s.squeeze(0).squeeze(0) / max(n, 1)).clamp_min(0.0)

                # small positive floor before IPF to avoid stuck-zero rows/cols
                pdf_grid = pdf_grid.clamp_min(self._EPS)

                # IPF to enforce discrete uniform marginals on the copula grid
                for _ in range(int(num_iter_max)):
                    pdf_grid *= tgt / pdf_grid.sum(dim=0, keepdim=True).clamp_min(_EPS)
                    pdf_grid *= tgt / pdf_grid.sum(dim=1, keepdim=True).clamp_min(_EPS)

                # normalize to integrate to 1 on [0,1]^2 (discrete dx^2)
                pdf_grid /= (pdf_grid.sum().clamp_min(_EPS) * (dx * dx))
                # keep strictly positive so log_pdf on finite rows is finite later
                pdf_grid = pdf_grid.clamp_min(self._EPS)
                pdf_grid /= (pdf_grid.sum().clamp_min(_EPS) * (dx * dx))

            # guarantee grid is finite
            pdf_grid = pdf_grid.nan_to_num(0.0, posinf=0.0, neginf=0.0).to(device=device, dtype=dtype)

        else:
            raise NotImplementedError
        self._pdf_grid = pdf_grid
        # * negloglik
        self.negloglik = -self.log_pdf(obs=obs).nan_to_num(posinf=0.0, neginf=0.0).sum()
        # ! cdf
        self._cdf_grid = ((self._pdf_grid * self.step_grid**2).cumsum(dim=0).cumsum(dim=1)).clamp_(
            0.0, 1.0
        )
        # ! h functions
        self._hfunc_l_grid = (self._pdf_grid * self.step_grid).cumsum(dim=1).clamp_(0.0, 1.0)
        self._hfunc_r_grid = (self._pdf_grid * self.step_grid).cumsum(dim=0).clamp_(0.0, 1.0)

    # @torch.compile
    def _interp(self, grid: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Bilinearly interpolate values on a 2D grid at given sample points.

        Args:
            grid (torch.Tensor): Precomputed grid of values (e.g., PDF/CDF/h‐function), shape (m,m).
            obs (torch.Tensor): Points in [0,1]² where to interpolate (rows are (u₁,u₂)), shape (n,2).

        Returns:
            torch.Tensor: Interpolated grid values at each observation, clamped ≥0, shape (n,1).
        """
        idx = obs.clamp(self._EPS, 1 - self._EPS) / self.step_grid
        i0 = idx.floor().long()
        di = idx - i0
        i1 = torch.minimum(
            i0 + 1,
            torch.full_like(input=i0, fill_value=self._target, device=idx.device),
        )
        g00 = grid[i0[:, 0], i0[:, 1]]
        g10 = grid[i1[:, 0], i0[:, 1]]
        g01 = grid[i0[:, 0], i1[:, 1]]
        g11 = grid[i1[:, 0], i1[:, 1]]
        return (
            g00
            + (g10 - g00) * di[:, 0]
            + (g01 - g00) * di[:, 1]
            + (g11 - g01 - g10 + g00) * di[:, 0] * di[:, 1]
        ).clamp_min(0.0)

    def cdf(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the copula CDF at given points. For independent copula, returns u₁·u₂.

        Args:
            obs (torch.Tensor): Points in [0,1]² where to evaluate the CDF (rows are (u₁,u₂)), shape (n,2).

        Returns:
            torch.Tensor: CDF values at each observation, shape (n,1).
        """
        # Independent copula: C(u,v) = u*v; mask so NaN rows remain NaN
        if self.is_indep:
            return self._eval_on_finite(obs, lambda z: (z[:, [0]] * z[:, [1]]), clamp_inputs=False).clamp_(0.0, 1.0)

        # Grid-based evaluation for all KDE modes
        val = self._eval_on_finite(obs, lambda z: self._interp(self._cdf_grid, z).unsqueeze(1))
        return val.clamp_(0.0, 1.0)

    def hfunc_l(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the left h-function at given points. Computes H(u₂ | u₁):= ∂/∂u₁ C(u₁,u₂) for
        the fitted copula. For independent copula, returns u₂.

        Args:
            obs (torch.Tensor): Points in [0,1]² where to evaluate the left h-function (rows are (u₁,u₂)), shape (n,2).

        Returns:
            torch.Tensor: Left h-function values at each observation, shape (n,1).
        """
        # Independent copula: h_l(v|u) = v
        if self.is_indep:
            return self._eval_on_finite(obs, lambda z: z[:, [1]], clamp_inputs=False)

        val = self._eval_on_finite(obs, lambda z: self._interp(self._hfunc_l_grid, z).unsqueeze(1))
        return val.clamp_(0.0, 1.0)

    def hfunc_r(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the right h-function at given points. Computes H(u₁ | u₂):= ∂/∂u₂ C(u₁,u₂) for
        the fitted copula. For independent copula, returns u₁.

        Args:
            obs (torch.Tensor): Points in [0,1]² where to evaluate the right h-function (rows are (u₁,u₂)), shape (n,2).

        Returns:
            torch.Tensor: Right h-function values at each observation, shape (n,1).
        """
        # Independent copula: h_r(u|v) = u
        if self.is_indep:
            return self._eval_on_finite(obs, lambda z: z[:, [0]], clamp_inputs=False)

        val = self._eval_on_finite(obs, lambda z: self._interp(self._hfunc_r_grid, z).unsqueeze(1))
        return val.clamp_(0.0, 1.0)

    @torch.no_grad()
    def hinv_l(self, obs: torch.Tensor) -> torch.Tensor:
        """Invert the left h-function: solve u2 s.t. H_l(u2 | u1) = p.

        Args:
            obs (torch.Tensor): Rows are (u1, p), nominally in [0,1]^2, shape (n,2).
        Returns:
            torch.Tensor: u2 ∈ [0,1], shape (n,1). Non-finite input rows yield NaN.
        """
        # normalize device/dtype once
        obs = obs.to(device=self.device, dtype=self.dtype)
        # Ensure a 2D shape
        if obs.ndim == 1:
            obs = obs.view(-1, 2)

        # independent copula: H_l(v|u) = v ⇒ u2 = p
        if self.is_indep:
            return obs[:, [1]]

        n = obs.shape[0]
        out = torch.full((n, 1), float("nan"), dtype=self.dtype, device=self.device)

        # keep only fully finite rows; others remain NaN
        mask = torch.isfinite(obs).all(dim=1)
        if not mask.any():
            return out

        z = obs[mask]
        # known u1 in [ε, 1−ε] to stay on-grid; target prob p ∈ [0,1]
        u1 = z[:, [0]].clamp(self._EPS, 1.0 - self._EPS)
        p  = z[:, [1]].clamp(0.0, 1.0)

        # fast paths for edge probabilities
        u2 = torch.empty_like(p)
        # make 1-D masks
        edge0 = (p <= self._EPS).squeeze(1)       # (K,)
        edge1 = (p >= 1.0 - self._EPS).squeeze(1) # (K,)
        u2[edge0, 0] = 0.0
        u2[edge1, 0] = 1.0

        mid = ~(edge0 | edge1)                    # (K,)
        if mid.any():
            u1m = u1[mid, :]                      # (Km,1)
            pm  = p[mid, :]                       # (Km,1)
            sol = solve_ITP(
                fun=lambda u2m: self.hfunc_l(obs=torch.hstack([u1m, u2m])) - pm,
                x_a=torch.zeros_like(pm),
                x_b=torch.ones_like(pm),
            )
            u2[mid, 0] = sol.view(-1)             # assign as flat column

        out[mask] = u2.clamp_(0.0, 1.0)
        return out


    @torch.no_grad()
    def hinv_r(self, obs: torch.Tensor) -> torch.Tensor:
        """Invert the right h-function: solve u1 s.t. H_r(u1 | u2) = p.

        Args:
            obs (torch.Tensor): Rows are (p, u2) or (u1?, u2?)? (Your current API uses (p,u2):
                            in this implementation we follow your code: obs[:,0]=p, obs[:,1]=u2.)
                            Shape (n,2), values nominally in [0,1].
        Returns:
            torch.Tensor: u1 ∈ [0,1], shape (n,1). Non-finite input rows yield NaN.
        """
        # normalize device/dtype once
        obs = obs.to(device=self.device, dtype=self.dtype)
        # Ensure a 2D shape
        if obs.ndim == 1:
            obs = obs.view(-1, 2)

        # independent copula: H_r(u|v) = u ⇒ u1 = p
        if self.is_indep:
            return obs[:, [0]]

        n = obs.shape[0]
        out = torch.full((n, 1), float("nan"), dtype=self.dtype, device=self.device)

        # keep only fully finite rows; others remain NaN
        mask = torch.isfinite(obs).all(dim=1)
        if not mask.any():
            return out

        z = obs[mask]
        # known u2 in [ε, 1−ε] to stay on-grid; target prob p ∈ [0,1]
        p  = z[:, [0]].clamp(0.0, 1.0)
        u2 = z[:, [1]].clamp(self._EPS, 1.0 - self._EPS)

        # fast paths for edge probabilities
        u1 = torch.empty_like(p)
        # make 1-D masks
        edge0 = (p <= self._EPS).squeeze(1)        # (K,)
        edge1 = (p >= 1.0 - self._EPS).squeeze(1)  # (K,)
        u1[edge0, 0] = 0.0
        u1[edge1, 0] = 1.0

        mid = ~(edge0 | edge1)                     # (K,)
        if mid.any():
            pm  = p[mid, :]                        # (Km,1)
            u2m = u2[mid, :]                       # (Km,1)
            sol = solve_ITP(
                fun=lambda u1m: self.hfunc_r(obs=torch.hstack([u1m, u2m])) - pm,
                x_a=torch.zeros_like(pm),
                x_b=torch.ones_like(pm),
            )
            u1[mid, 0] = sol.view(-1)

        out[mask] = u1.clamp_(0.0, 1.0)
        return out


    def pdf(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the copula PDF at given points. For independent copula, returns 1 on finite rows,
        and leaves non-finite rows as NaN (via _eval_on_finite).
        """
        if self.is_indep:
            # finite rows -> 1; non-finite rows stay NaN
            return self._eval_on_finite(
                obs,
                lambda z: torch.ones((z.shape[0], 1), dtype=self.dtype, device=self.device),
                clamp_inputs=False,   # indep copula doesn't need clamping
            )

        # nonparametric modes: interpolate and keep strictly positive on finite rows
        return self._eval_on_finite(
            obs,
            # clamp_min only applies to finite rows because it's inside eval_fn
            lambda z: self._interp(self._pdf_grid, z).unsqueeze(1).clamp_min(self._EPS)
        )


    def log_pdf(self, obs: torch.Tensor) -> torch.Tensor:
        """
        log-PDF: do not sanitize outputs; NaN rows must remain NaN per tests.
        Finite rows are finite because pdf() >= EPS for them.
        """
        p = self.pdf(obs)            # shape (N,1), NaN on bad rows via _eval_on_finite
        logp = torch.log(p)          # NaN stays NaN, zeros -> -inf
        # Only replace -inf (coming from zeros) with ln(LOG_EPS), leave everything else as-is
        neg_inf = torch.isinf(logp) & (logp < 0)
        if neg_inf.any():
            logp = logp.clone()
            logp[neg_inf] = math.log(self._LOG_EPS)
        return logp


    @torch.no_grad()
    def sample(
        self, num_sample: int = 100, seed: int = 42, is_sobol: bool = False
    ) -> torch.Tensor:
        """Sample from the copula by inverse Rosenblatt transform. Uses Sobol sequence if
        ``is_sobol=True``, otherwise uniform RNG. For independent copula, returns uniform samples in
        [0,1]².

        Args:
            num_sample (int, optional): number of samples to generate. Defaults to 100.
            seed (int, optional): random seed for reproducibility. Defaults to 42.
            is_sobol (bool, optional): If True, use Sobol sampling. Defaults to False.
        Returns:
            torch.Tensor: Generated samples, shape (num_sample, 2).
        """
        # ! device agnostic
        device, dtype = self.device, self.dtype
        if is_sobol:
            obs = (
                torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
                .draw(n=num_sample, dtype=dtype)
                .to(device=device)
            )
        else:
            torch.manual_seed(seed=seed)
            obs = torch.rand(size=(num_sample, 2), dtype=dtype, device=device)
        if not self.is_indep:
            obs[:, [1]] = self.hinv_l(obs=obs)
        return obs

    def __str__(self) -> str:
        """String representation of the BiCop class.

        Returns:
            str: String representation of the BiCop class.
        """
        return f"""{self.__class__.__name__}\n{
            pformat(
                object={
                    "is_indep": self.is_indep,
                    "num_obs": self.num_obs,
                    "negloglik": self.negloglik.round(decimals=4),
                    "num_step_grid": self.num_step_grid,
                    "tau": self.tau.round(decimals=4),
                    "mtd_kde": self.mtd_kde,
                    "dtype": self._dd.dtype,
                    "device": self._dd.device,
                },
                compact=True,
                sort_dicts=False,
                underscore_numbers=True,
            )
        }"""

    @torch.no_grad()
    def imshow(
        self,
        is_log_pdf: bool = False,
        ax: plt.Axes | None = None,
        cmap: str = "inferno",
        xlabel: str = r"$u_{left}$",
        ylabel: str = r"$u_{right}$",
        title: str = "Estimated bivariate copula density",
        colorbartitle: str = "Density",
        **imshow_kwargs: dict,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Display the (log-)PDF grid as a heatmap.

        Args:
            is_log_pdf (bool, optional): If True, plot log-PDF. Defaults to False.
            ax (plt.Axes, optional): Matplotlib Axes object to plot on. If None, a new figure and axes are created. Defaults to None.
            cmap (str, optional): Colormap for the plot. Defaults to "inferno".
            xlabel (str, optional): X-axis label. Defaults to r"$u_{left}$".
            ylabel (str, optional): Y-axis label. Defaults to r"$u_{right}$".
            title (str, optional): Plot title. Defaults to "Estimated bivariate copula density".
            colorbartitle (str, optional): Colorbar title. Defaults to "Density".
            **imshow_kwargs: Additional keyword arguments for imshow.
        Returns:
            tuple[plt.Figure, plt.Axes]: The figure and axes objects.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        im = ax.imshow(
            X=self._pdf_grid.log().nan_to_num(posinf=0.0, neginf=-13.815510557964274).cpu()
            if is_log_pdf
            else self._pdf_grid.cpu(),
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap=cmap,
            **imshow_kwargs,
        )
        ax.set_xlabel(xlabel=xlabel)
        ax.set_ylabel(ylabel=ylabel)
        ax.set_title(label=title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.colorbar(im, ax=ax, label=colorbartitle)
        return fig, ax

    @torch.no_grad()
    def plot(
        self,
        plot_type: str = "surface",
        margin_type: str = "unif",
        xylim: Optional[tuple[float, float]] = None,
        grid_size: Optional[int] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the bivariate copula density.

        Args:
            plot_type (str, optional): Type of plot, either "contour" or "surface". Defaults to "surface".
            margin_type (str, optional): Type of margin, either "unif" or "norm". Defaults to "unif".
            xylim (tuple[float, float], optional): Limits for x and y axes. Defaults to None.
            grid_size (int, optional): Size of the grid for the plot. Defaults to None.
        Returns:
            tuple[plt.Figure, plt.Axes]: The figure and axes objects.
        """
        # * validate inputs
        if plot_type not in ["contour", "surface"]:
            raise ValueError("Unknown type")
        elif plot_type == "contour" and grid_size is None:
            grid_size = 100
        elif plot_type == "surface" and grid_size is None:
            grid_size = 40
        # * margin type and grid points
        if margin_type not in ["unif", "norm"]:
            raise ValueError("Unknown margin type")
        elif margin_type == "unif":
            if xylim is None:
                xylim = (1e-2, 1 - 1e-2)
            if plot_type == "contour":
                points = np.linspace(1e-5, 1 - 1e-5, grid_size)
            else:
                points = np.linspace(1, grid_size, grid_size) / (grid_size + 1)
            g = np.meshgrid(points, points)
            points = g[0][0]
            adj = 1
            levels = [0.2, 0.6, 1, 1.5, 2, 3, 5, 10, 20]
            xlabel, ylabel = "u1", "u2"
        elif margin_type == "norm":
            if xylim is None:
                xylim = (-3, 3)
            points = norm.cdf(np.linspace(xylim[0], xylim[1], grid_size))
            g = np.meshgrid(points, points)
            points = norm.ppf(g[0][0])
            adj = np.outer(norm.pdf(points), norm.pdf(points))
            levels = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
            xlabel, ylabel = "z1", "z2"

        # * evaluate on grid
        g_tensor = torch.from_numpy(np.stack(g, axis=-1).reshape(-1, 2)).to(
            device=self.device, dtype=self.dtype
        )
        vals = self.pdf(g_tensor).cpu().numpy()
        cop = np.reshape(vals, (grid_size, grid_size))

        # * adjust for margins
        dens = cop * adj
        if len(np.unique(dens)) == 1:
            dens[0] = 1.000001 * dens[0]
        if margin_type == "unif":
            zlim = (0, max(3, 1.1 * max(dens.ravel())))
        elif margin_type == "norm":
            zlim = (0, max(0.4, 1.1 * max(dens.ravel())))

        # * create a colormap
        jet_colors = LinearSegmentedColormap.from_list(
            name="jet_colors",
            colors=[
                "#00007F",
                "blue",
                "#007FFF",
                "cyan",
                "#7FFF7F",
                "yellow",
                "#FF7F00",
                "red",
                "#7F0000",
            ],
            N=100,
        )

        # * plot
        if plot_type == "contour":
            fig, ax = plt.subplots()
            contour = ax.contour(points, points, dens, levels=levels, cmap="gray")
            ax.clabel(contour, inline=True, fontsize=8, fmt="%1.2f")
            ax.set_aspect("equal")
            ax.grid(True)
        elif plot_type == "surface":
            fig = plt.figure()
            ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
            ax.view_init(elev=30, azim=-110)
            X, Y = np.meshgrid(points, points)
            ax.plot_surface(X, Y, dens, cmap=jet_colors, edgecolor="none", shade=False)
            ax.set_zlim(zlim)
            ax.set_box_aspect([1, 1, 1])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
        fig.tight_layout()
        plt.draw_if_interactive()
        return fig, ax
