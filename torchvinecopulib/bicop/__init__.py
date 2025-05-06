"""
* CamelCase naming convention for class
* grid stored in registered buffer after fit()
* torch.compile() for bilinear interpolation
* torch.no_grad() for fit(), hinv_l(), hinv_r(), sample(), and imshow()
"""

from pprint import pformat
from typing import Any, Optional, cast

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
        """
        num_step_grid has to be a power of 2
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
    def device(self):
        return self._dd.device

    @property
    def dtype(self):
        return self._dd.dtype

    @torch.no_grad()
    def reset(self) -> None:
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
        num_obs_max: int = None,
        seed: int = 42,
        num_iter_max: int = 17,
        is_tau_est: bool = False,
        is_tll: bool = True,
    ) -> None:
        # ! device agnostic
        device, dtype = self.device, self.dtype
        self.is_indep = False
        self.is_tll = is_tll
        self.num_obs.copy_(obs.shape[0])
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
        if num_obs_max and self.num_obs > num_obs_max:
            torch.manual_seed(seed=seed)
            idx = torch.randperm(self.num_obs, device=device)[:num_obs_max]
            obs = obs[idx]

        if is_tll:
            controls = pv.FitControlsBicop(family_set=[pv.tll])
            cop = pv.Bicop.from_data(data=obs.cpu().numpy(), controls=controls)
            axis = np.linspace(_EPS, 1 - _EPS, self.num_step_grid)
            grid = np.stack(np.meshgrid(axis, axis)).reshape(2, -1).T
            pdf_grid = torch.from_numpy(
                cop.pdf(grid).reshape(self.num_step_grid, self.num_step_grid)
            ).to(device=device, dtype=dtype)
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
                        torch.zeros(
                            self.num_step_grid - H, W, dtype=dtype, device=device
                        ),
                    ],
                    dim=0,
                )
            H, W = pdf_grid.shape
            if W < self.num_step_grid:
                pdf_grid = torch.cat(
                    [
                        pdf_grid,
                        torch.zeros(
                            H, self.num_step_grid - W, dtype=dtype, device=device
                        ),
                    ],
                    dim=1,
                )
            pdf_grid = pdf_grid[: self.num_step_grid, : self.num_step_grid].clamp_min(
                _EPS
            )
            pdf_grid = pdf_grid.view(self.num_step_grid, self.num_step_grid)
            # * normalization: Sinkhorn / iterative proportional fitting (IPF)
            for _ in range(num_iter_max):
                pdf_grid *= self._target / pdf_grid.sum(dim=0, keepdim=True)
                pdf_grid *= self._target / pdf_grid.sum(dim=1, keepdim=True)
            pdf_grid /= pdf_grid.sum() * self.step_grid**2
        self._pdf_grid = pdf_grid
        # * negloglik
        self.negloglik = -self.log_pdf(obs=obs).sum()
        # ! cdf
        self._cdf_grid = (
            (self._pdf_grid * self.step_grid**2).cumsum(dim=0).cumsum(dim=1)
        ).clamp_(0.0, 1.0)
        # ! h functions
        self._hfunc_l_grid = (
            (self._pdf_grid * self.step_grid).cumsum(dim=1).clamp_(0.0, 1.0)
        )
        self._hfunc_r_grid = (
            (self._pdf_grid * self.step_grid).cumsum(dim=0).clamp_(0.0, 1.0)
        )

    @torch.compile
    def _interp(self, grid: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
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
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs.prod(dim=1, keepdim=True)
        return self._interp(grid=self._cdf_grid, obs=obs).unsqueeze(dim=1)

    def hfunc_l(self, obs: torch.Tensor) -> torch.Tensor:
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [1]]
        return self._interp(grid=self._hfunc_l_grid, obs=obs).unsqueeze(dim=1)

    def hfunc_r(self, obs: torch.Tensor) -> torch.Tensor:
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [0]]
        return self._interp(grid=self._hfunc_r_grid, obs=obs).unsqueeze(dim=1)

    @torch.no_grad()
    def hinv_l(self, obs: torch.Tensor) -> torch.Tensor:
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [1]]
        # * via root-finding
        u_l = obs[:, [0]].clamp(self._EPS, 1 - self._EPS)
        p = obs[:, [1]].clamp(self._EPS, 1 - self._EPS)
        return solve_ITP(
            fun=lambda u_r: self.hfunc_l(obs=torch.hstack([u_l, u_r])) - p,
            x_a=torch.zeros_like(p),
            x_b=torch.ones_like(p),
        )

    @torch.no_grad()
    def hinv_r(self, obs: torch.Tensor) -> torch.Tensor:
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [0]]
        # * via root-finding
        u_r = obs[:, [1]].clamp(self._EPS, 1 - self._EPS)
        p = obs[:, [0]].clamp(self._EPS, 1 - self._EPS)
        return solve_ITP(
            fun=lambda u_l: self.hfunc_r(obs=torch.hstack([u_l, u_r])) - p,
            x_a=torch.zeros_like(p),
            x_b=torch.ones_like(p),
        )

    def pdf(self, obs: torch.Tensor) -> torch.Tensor:
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return torch.ones_like(obs[:, [0]])
        return self._interp(grid=self._pdf_grid, obs=obs).unsqueeze(dim=1)

    def log_pdf(self, obs: torch.Tensor) -> torch.Tensor:
        # ! device agnostic
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return torch.zeros_like(obs[:, [0]])
        return (
            self.pdf(obs=obs).log().nan_to_num(posinf=0.0, neginf=-13.815510557964274)
        )

    @torch.no_grad()
    def sample(
        self, num_sample: int = 100, seed: int = 42, is_sobol: bool = False
    ) -> torch.Tensor:
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
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        im = ax.imshow(
            X=self._pdf_grid.log()
            .nan_to_num(posinf=0.0, neginf=-13.815510557964274)
            .cpu()
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

    @staticmethod
    @torch.no_grad()
    def get_default_xylim(margin_type: str) -> tuple[float, float]:
        if margin_type == "unif":
            return (1e-2, 1 - 1e-2)
        elif margin_type == "norm":
            return (-3, 3)
        else:
            raise ValueError("Unknown margin type")

    @staticmethod
    @torch.no_grad()
    def get_default_grid_size(plot_type: str) -> int:
        if plot_type == "contour":
            return 100
        elif plot_type == "surface":
            return 40
        else:
            raise ValueError("Unknown plot type")

    @torch.no_grad()
    def plot(
        self,
        plot_type: str = "surface",
        margin_type: str = "unif",
        xylim: Optional[tuple[float, float]] = None,
        grid_size: Optional[int] = None,
    ) -> None:
        if plot_type not in ["contour", "surface"]:
            raise ValueError("Unknown type")

        if margin_type not in ["unif", "norm"]:
            raise ValueError("Unknown margin type")

        if xylim is None:
            xylim = self.get_default_xylim(margin_type)

        if grid_size is None:
            grid_size = self.get_default_grid_size(plot_type)

        if margin_type == "unif":
            if plot_type == "contour":
                points = np.linspace(1e-5, 1 - 1e-5, grid_size)
            else:
                points = np.linspace(1, grid_size, grid_size) / (grid_size + 1)

            g = np.meshgrid(points, points)
            points = g[0][0]
            adj = 1
            levels = [0.2, 0.6, 1, 1.5, 2, 3, 5, 10, 20]
            xlabel = "u1"
            ylabel = "u2"
        elif margin_type == "norm":
            points = norm.cdf(np.linspace(xylim[0], xylim[1], grid_size))
            g = np.meshgrid(points, points)
            points = norm.ppf(g[0][0])
            adj = np.outer(norm.pdf(points), norm.pdf(points))
            levels = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
            xlabel = "z1"
            ylabel = "z2"
        else:
            raise ValueError("Unknown margin type")

        ## evaluate on grid
        g_tensor = torch.from_numpy(np.stack(g, axis=-1).reshape(-1, 2)).to(
            device=self.device, dtype=self.dtype
        )
        vals = self.pdf(g_tensor).cpu().numpy()
        cop = np.reshape(vals, (grid_size, grid_size))

        ## adjust for margins
        dens = cop * adj
        if len(np.unique(dens)) == 1:
            dens[0] = 1.000001 * dens[0]

        if margin_type == "unif":
            zlim = (0, max(3, 1.1 * max(dens.flatten())))
        elif margin_type == "norm":
            zlim = (0, max(0.4, 1.1 * max(dens.flatten())))

        # Define the colors as in the R code
        colors = [
            "#00007F",
            "blue",
            "#007FFF",
            "cyan",
            "#7FFF7F",
            "yellow",
            "#FF7F00",
            "red",
            "#7F0000",
        ]

        # Create the custom colormap
        jet_colors = LinearSegmentedColormap.from_list("jet_colors", colors, N=100)

        ## plot
        if plot_type == "contour":
            fig, ax = plt.subplots()
            contour = ax.contour(points, points, dens, levels=levels, cmap="gray")
            ax.clabel(contour, inline=True, fontsize=8, fmt="%1.2f")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(xylim)
            ax.set_ylim(xylim)
            ax.set_aspect("equal")
            fig.tight_layout()
            plt.draw_if_interactive()
            return fig, ax
        elif plot_type == "surface":
            fig = plt.figure()
            ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
            ax.view_init(elev=30, azim=-110)
            X, Y = np.meshgrid(points, points)
            ax.plot_surface(X, Y, dens, cmap=jet_colors, edgecolor="none", shade=False)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(xylim)
            ax.set_ylim(xylim)
            ax.set_zlim(zlim)
            ax.set_box_aspect([1, 1, 1])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(False)
            fig.tight_layout()
            plt.draw_if_interactive()
            return fig, ax
        else:
            raise ValueError("Unknown plot type")
