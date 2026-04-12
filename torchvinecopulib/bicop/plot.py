from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import torch

from ..backends.common import _EPS

__all__ = ["BiCopPlotMixin"]

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _lazy_norm():
    try:
        from scipy.stats import norm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting with margin_type='norm' requires scipy.") from exc
    return norm


def _lazy_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from mpl_toolkits.mplot3d.axes3d import Axes3D
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib.") from exc
    return plt, LinearSegmentedColormap, Axes3D


class BiCopPlotMixin:
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
        plt, _, _ = _lazy_matplotlib()
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        grid = (
            self._pdf_grid
            if self._pdf_grid.numel() > 0
            else torch.ones(
                self.num_step_grid, self.num_step_grid, dtype=self.dtype, device=self.device
            )
        )
        grid_np = (grid.clamp_min(_EPS).log() if is_log_pdf else grid).detach().cpu().numpy()
        im = ax.imshow(
            X=grid_np,
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
        plt, LinearSegmentedColormap, Axes3D = _lazy_matplotlib()
        if plot_type not in ["contour", "surface"]:
            raise ValueError("Unknown type")
        elif plot_type == "contour" and grid_size is None:
            grid_size = 100
        elif plot_type == "surface" and grid_size is None:
            grid_size = 40
        if margin_type not in ["unif", "norm"]:
            raise ValueError("Unknown margin type")
        if margin_type == "unif":
            if xylim is None:
                xylim = (1e-2, 1 - 1e-2)
            if plot_type == "contour":
                points = np.linspace(1e-5, 1 - 1e-5, grid_size)
            else:
                points = np.linspace(1, grid_size, grid_size) / (grid_size + 1)
            g = np.meshgrid(points, points)
            points = g[0][0]
            adj = 1.0
            levels = [0.2, 0.6, 1, 1.5, 2, 3, 5, 10, 20]
            xlabel, ylabel = "u1", "u2"
        else:
            norm = _lazy_norm()
            if xylim is None:
                xylim = (-3, 3)
            points = norm.cdf(np.linspace(xylim[0], xylim[1], grid_size))
            g = np.meshgrid(points, points)
            points = norm.ppf(g[0][0])
            adj = np.outer(norm.pdf(points), norm.pdf(points))
            levels = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
            xlabel, ylabel = "z1", "z2"
        g_tensor = torch.from_numpy(np.stack(g, axis=-1).reshape(-1, 2)).to(
            device=self.device, dtype=self.dtype
        )
        vals = self.pdf(g_tensor).detach().cpu().numpy()
        cop = np.reshape(vals, (grid_size, grid_size))
        dens = cop * adj
        if len(np.unique(dens)) == 1:
            dens[0] = 1.000001 * dens[0]
        zlim = (0, max(3 if margin_type == "unif" else 0.4, 1.1 * max(dens.ravel())))
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
        if plot_type == "contour":
            fig, ax = plt.subplots()
            contour = ax.contour(points, points, dens, levels=levels, cmap="gray")
            ax.clabel(contour, inline=True, fontsize=8, fmt="%1.2f")
            ax.set_aspect("equal")
            ax.grid(True)
        else:
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
