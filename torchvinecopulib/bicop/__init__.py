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
from fastkde import pdf as fkpdf
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats import kendalltau, norm

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
        self.is_tll = False
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
        is_tll: bool = True,
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
            is_tll (bool, optional): Using tll or fastKDE. Defaults to True (tll).
            mtd_tll (str, optional): fit method for the transformation local-likelihood (TLL) nonparametric family, one of ("constant", "linear", or "quadratic"). Defaults to "constant".
            num_iter_max (int, optional): num of Sinkhorn/IPF iters for grid normalization, used only when ``is_tll=False``. Defaults to 17.
            is_tau_est (bool, optional): If True, compute and store Kendall’s τ. Defaults to ``False``.
        """
        # ! device agnostic
        device, dtype = self.device, self.dtype
        self.is_indep = False
        self.is_tll = is_tll
        self.num_obs.copy_(obs.shape[0])
        # * assuming already in [0, 1]
        obs = obs.clamp(min=0.0, max=1.0)
        if is_tau_est:
            self.tau.copy_(
                torch.as_tensor(
                    kendalltau(obs[:, 0].cpu(), obs[:, 1].cpu()),
                    device=device,
                    dtype=dtype,
                )
            )
        self._target = self.num_step_grid - 1.0  # * marginal target
        self.step_grid = 1.0 / self._target
        # ! pdf
        if is_tll:
            controls = pv.FitControlsBicop(
                family_set=[pv.tll],
                num_threads=torch.get_num_threads(),
                nonparametric_method=mtd_tll,
            )
            cop = pv.Bicop.from_data(data=obs.cpu().numpy(), controls=controls)
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
        else:
            pdf_grid = torch.from_numpy(
                fkpdf(
                    obs[:, 0].cpu(),
                    obs[:, 1].cpu(),
                    num_points=self.num_step_grid * 2 + 1,
                ).values
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
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs.prod(dim=1, keepdim=True)
        return self._interp(grid=self._cdf_grid, obs=obs).unsqueeze(dim=1)

    def hfunc_l(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the left h-function at given points. Computes H(u₂ | u₁):= ∂/∂u₁ C(u₁,u₂) for
        the fitted copula. For independent copula, returns u₂.

        Args:
            obs (torch.Tensor): Points in [0,1]² where to evaluate the left h-function (rows are (u₁,u₂)), shape (n,2).

        Returns:
            torch.Tensor: Left h-function values at each observation, shape (n,1).
        """
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [1]]
        return self._interp(grid=self._hfunc_l_grid, obs=obs).unsqueeze(dim=1)

    def hfunc_r(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the right h-function at given points. Computes H(u₁ | u₂):= ∂/∂u₂ C(u₁,u₂) for
        the fitted copula. For independent copula, returns u₁.

        Args:
            obs (torch.Tensor): Points in [0,1]² where to evaluate the right h-function (rows are (u₁,u₂)), shape (n,2).

        Returns:
            torch.Tensor: Right h-function values at each observation, shape (n,1).
        """
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [0]]
        return self._interp(grid=self._hfunc_r_grid, obs=obs).unsqueeze(dim=1)

    @torch.no_grad()
    def hinv_l(self, obs: torch.Tensor) -> torch.Tensor:
        """Invert the left h‐function via root‐finding: find u₂ given (u₁, p). Solves H(u₂ | u₁) = p
        by ITP between 0 and 1.

        Args:
            obs (torch.Tensor): Points in [0,1]² where to evaluate the left h-function (rows are (u₁,u₂)), shape (n,2).

        Returns:
            torch.Tensor: Solutions u₂ ∈ [0,1], shape (n,1).
        """
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [1]]
        # * via root-finding
        u_l = obs[:, [0]]
        p = obs[:, [1]]
        return solve_ITP(
            fun=lambda u_r: self.hfunc_l(obs=torch.hstack([u_l, u_r])) - p,
            x_a=torch.zeros_like(p),
            x_b=torch.ones_like(p),
        ).clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def hinv_r(self, obs: torch.Tensor) -> torch.Tensor:
        """Invert the right h‐function via root‐finding: find u₁ given (u₂, p). Solves H(u₁ | u₂) =
        p by ITP between 0 and 1.

        Args:
            obs (torch.Tensor): Points in [0,1]² where to evaluate the right h-function (rows are (u₁,u₂)), shape (n,2).
        Returns:
            torch.Tensor: Solutions u₁ ∈ [0,1], shape (n,1).
        """
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [0]]
        # * via root-finding
        u_r = obs[:, [1]]
        p = obs[:, [0]]
        return solve_ITP(
            fun=lambda u_l: self.hfunc_r(obs=torch.hstack([u_l, u_r])) - p,
            x_a=torch.zeros_like(p),
            x_b=torch.ones_like(p),
        ).clamp(min=0.0, max=1.0)

    def pdf(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the copula PDF at given points. For independent copula, returns 1.

        Args:
            obs (torch.Tensor): Points in [0,1]² where to evaluate the PDF (rows are (u₁,u₂)), shape (n,2).
        Returns:
            torch.Tensor: PDF values at each observation, shape (n,1).
        """
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return torch.ones_like(obs[:, [0]])
        return self._interp(grid=self._pdf_grid, obs=obs).unsqueeze(dim=1)

    def log_pdf(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the copula log-PDF at given points, with safe handling of inf/nan. For
        independent copula, returns 0.

        Args:
            obs (torch.Tensor): Points in [0,1]² where to evaluate the log-PDF (rows are (u₁,u₂)), shape (n,2).
        Returns:
            torch.Tensor: log-PDF values at each observation, shape (n,1).
        """
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return torch.zeros_like(obs[:, [0]])
        return self.pdf(obs=obs).log().nan_to_num(posinf=0.0, neginf=-13.815510557964274)

    @torch.no_grad()
    def sample(self, num_sample: int = 100, seed: int = 42, is_sobol: bool = False) -> torch.Tensor:
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
        return f"{self.__class__.__name__}\n{
            pformat(
                object={
                    'is_indep': self.is_indep,
                    'num_obs': self.num_obs,
                    'negloglik': self.negloglik.round(decimals=4),
                    'num_step_grid': self.num_step_grid,
                    'tau': self.tau.round(decimals=4),
                    'is_tll': self.is_tll,
                    'dtype': self._dd.dtype,
                    'device': self._dd.device,
                },
                compact=True,
                sort_dicts=False,
                underscore_numbers=True,
            )
        }"

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
