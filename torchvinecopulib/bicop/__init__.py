"""
* CamelCase naming convention for class
* grid stored in registered buffer after fit()
* torch.compile() for bilinear interpolation
* torch.no_grad() for fit(), hinv_l(), hinv_r(), sample(), and imshow()
"""

from pprint import pformat

import matplotlib.pyplot as plt
import torch
from fastkde import pdf as fkpdf
from scipy.stats import kendalltau

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
    ) -> None:
        # ! device agnostic
        device, dtype = self.device, self.dtype
        # * fastkde works on cpu
        obs = obs.to(device="cpu", dtype=dtype)
        self.is_indep = False
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
            pdf_grid = torch.from_numpy(
                fkpdf(
                    obs[idx, 0].cpu(),
                    obs[idx, 1].cpu(),
                    num_points=self.num_step_grid * 2 + 1,
                ).values
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
        pdf_grid = pdf_grid[: self.num_step_grid, : self.num_step_grid].clamp_min(_EPS)
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
        )
        # ! h functions
        self._hfunc_l_grid = (self._pdf_grid * self.step_grid).cumsum(dim=1)
        self._hfunc_r_grid = (self._pdf_grid * self.step_grid).cumsum(dim=0)

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
            _, ax = plt.subplots()
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
        return ax
