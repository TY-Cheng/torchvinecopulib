"""Public bivariate copula interfaces.

This module exposes the `BiCop` façade plus diagnostics and torch-native KDE helpers used by the
grid backends.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import warnings
from pprint import pformat
from typing import Any, Literal, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..backends import API_VERSION, TorchCopulaKDE2D, build_bicop_estimator, normalize_bicop_backend
from ..backends.common import _EPS
from ..util import kendall_tau, solve_ITP

__all__ = [
    "BiCop",
    "BiCopDiagnostics",
    "TorchCopulaKDE2D",
]


@dataclass(frozen=True)
class BiCopDiagnostics:
    itp_failures_l: int
    itp_failures_r: int
    bisect_refinements_l: int
    bisect_refinements_r: int
    fallback_to_indep_l: int
    fallback_to_indep_r: int
    max_abs_hfunc_error_l: float
    max_abs_hfunc_error_r: float


def _lazy_norm():
    try:
        from scipy.stats import norm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting with margin_type='norm' requires scipy.") from exc
    return norm


class BiCop(torch.nn.Module):
    _EPS: float = _EPS

    def __init__(
        self,
        num_step_grid: int = 128,
        boundary_policy: Literal["hard", "st"] = "hard",
    ):
        super().__init__()
        self.api_version = API_VERSION
        self.is_indep = True
        self.bicop_backend = "grid_reflect"
        self.kde_backend = "grid_reflect"
        self.mtd_kde = "grid_reflect"
        self.backend_config: dict[str, Any] = {}
        self.boundary_policy: Literal["hard", "st"] = boundary_policy
        self.num_step_grid = int(num_step_grid)
        self._target = float(self.num_step_grid - 1)
        self.step_grid = 1.0 / max(self._target, 1.0)
        self.register_buffer("tau", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("num_obs", torch.zeros((), dtype=torch.int))
        self.register_buffer("negloglik", torch.zeros((), dtype=torch.float64))
        self.register_buffer("hinv_fallback_l", torch.zeros((), dtype=torch.int64))
        self.register_buffer("hinv_fallback_r", torch.zeros((), dtype=torch.int64))
        self.register_buffer("hinv_bisect_l", torch.zeros((), dtype=torch.int64))
        self.register_buffer("hinv_bisect_r", torch.zeros((), dtype=torch.int64))
        self.register_buffer("fallback_to_indep_l", torch.zeros((), dtype=torch.int64))
        self.register_buffer("fallback_to_indep_r", torch.zeros((), dtype=torch.int64))
        self.register_buffer("max_abs_hfunc_error_l", torch.zeros((), dtype=torch.float64))
        self.register_buffer("max_abs_hfunc_error_r", torch.zeros((), dtype=torch.float64))
        self.register_buffer("bandwidth", torch.empty(2, dtype=torch.float64))
        self.register_buffer("_pdf_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_cdf_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_hfunc_l_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_hfunc_r_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_dd", torch.tensor([], dtype=torch.float64))
        self.last_hinv_status_l: dict[str, torch.Tensor] | None = None
        self.last_hinv_status_r: dict[str, torch.Tensor] | None = None

    @property
    def device(self) -> torch.device:
        return self._dd.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dd.dtype

    def _empty_grid(self) -> torch.Tensor:
        return torch.empty(0, 0, device=self.device, dtype=self.dtype)

    def _ensure_geometry(self) -> None:
        if self._pdf_grid.numel() > 0:
            self.num_step_grid = int(self._pdf_grid.shape[0])
        self._target = float(self.num_step_grid - 1)
        self.step_grid = 1.0 / max(self._target, 1.0)

    def _assign_fit_bundle(self, estimator: torch.nn.Module) -> None:
        self._pdf_grid = estimator._pdf_grid.to(device=self.device, dtype=self.dtype)
        self._cdf_grid = estimator._cdf_grid.to(device=self.device, dtype=self.dtype)
        self._hfunc_l_grid = estimator._hfunc_l_grid.to(device=self.device, dtype=self.dtype)
        self._hfunc_r_grid = estimator._hfunc_r_grid.to(device=self.device, dtype=self.dtype)
        self.bandwidth = estimator.bandwidth.to(device=self.device, dtype=self.dtype)
        self.bicop_backend = estimator.backend_name
        self.kde_backend = estimator.backend_name
        self.mtd_kde = estimator.backend_name
        self.backend_config = dict(estimator.backend_config)
        self._ensure_geometry()

    def _normalize_fit_request(
        self,
        *,
        bicop_backend: str | None,
        kde_backend: str | None,
        mtd_kde: str | None,
        bicop_kwargs: dict[str, Any] | None,
        mtd_tll: str,
        num_iter_max: int,
        bandwidth: str | float | torch.Tensor,
        bandwidth_scale: float,
    ) -> tuple[str, dict[str, Any]]:
        if bicop_backend is None and kde_backend is not None:
            warnings.warn(
                "'kde_backend' is deprecated; use 'bicop_backend' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            bicop_backend = kde_backend
        if bicop_backend is None:
            if mtd_kde is None:
                bicop_backend = "grid_reflect"
            else:
                warnings.warn(
                    "'mtd_kde' is deprecated; use 'bicop_backend' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                bicop_backend = {
                    "fastKDE": "grid_reflect",
                    "torch_grid": "grid_reflect",
                    "tll": "tll_ref",
                }.get(mtd_kde, mtd_kde)
        normalized_backend = normalize_bicop_backend(bicop_backend)
        normalized_kwargs = dict(bicop_kwargs or {})
        if normalized_backend == "tll_ref":
            if "nonparametric_method" not in normalized_kwargs:
                normalized_kwargs["nonparametric_method"] = mtd_tll
            elif mtd_tll != "constant":
                warnings.warn(
                    "Both 'bicop_kwargs[\"nonparametric_method\"]' and legacy 'mtd_tll' were provided; "
                    "using bicop_kwargs.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        else:
            if "bandwidth" not in normalized_kwargs:
                normalized_kwargs["bandwidth"] = bandwidth
            elif bandwidth != "silverman":
                warnings.warn(
                    "Both 'bicop_kwargs[\"bandwidth\"]' and legacy top-level 'bandwidth' were provided; "
                    "using bicop_kwargs.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if "bandwidth_scale" not in normalized_kwargs:
                normalized_kwargs["bandwidth_scale"] = bandwidth_scale
            elif bandwidth_scale != 1.0:
                warnings.warn(
                    "Both 'bicop_kwargs[\"bandwidth_scale\"]' and legacy top-level 'bandwidth_scale' were "
                    "provided; using bicop_kwargs.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if "num_iter_max" not in normalized_kwargs:
                normalized_kwargs["num_iter_max"] = num_iter_max
            elif num_iter_max != 5:
                warnings.warn(
                    "Both 'bicop_kwargs[\"num_iter_max\"]' and legacy top-level 'num_iter_max' were provided; "
                    "using bicop_kwargs.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        return normalized_backend, normalized_kwargs

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "backend_name": self.bicop_backend,
            "normalized_backend_config": dict(self.backend_config),
            "boundary_policy": self.boundary_policy,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self.api_version = state.get("api_version", self.api_version)
        backend_name = state.get("backend_name")
        if backend_name:
            self.bicop_backend = backend_name
            self.kde_backend = backend_name
            self.mtd_kde = backend_name
        self.backend_config = dict(state.get("normalized_backend_config", {}))
        self.boundary_policy = state.get("boundary_policy", self.boundary_policy)

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
        for name in ("_pdf_grid", "_cdf_grid", "_hfunc_l_grid", "_hfunc_r_grid"):
            key = prefix + name
            if key in state_dict:
                tensor = state_dict[key]
                if self._buffers[name].shape != tensor.shape:
                    self._buffers[name] = torch.empty_like(tensor)
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
        self._ensure_geometry()
        self.is_indep = self._pdf_grid.numel() == 0
        if not self.is_indep and not self.bicop_backend:
            self.bicop_backend = "grid_reflect"
            self.kde_backend = "grid_reflect"
            self.mtd_kde = "grid_reflect"

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "_extra_state" not in state_dict:
            patched_state = OrderedDict(state_dict)
            patched_state["_extra_state"] = {
                "api_version": self.api_version,
                "backend_name": "grid_reflect",
                "normalized_backend_config": {},
            }
            state_dict = patched_state
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def reset(self) -> None:
        self.is_indep = True
        self.bicop_backend = "grid_reflect"
        self.kde_backend = "grid_reflect"
        self.mtd_kde = "grid_reflect"
        self.backend_config = {}
        self._ensure_geometry()
        self.tau.zero_()
        self.num_obs.zero_()
        self.negloglik.zero_()
        self.hinv_fallback_l.zero_()
        self.hinv_fallback_r.zero_()
        self.hinv_bisect_l.zero_()
        self.hinv_bisect_r.zero_()
        self.fallback_to_indep_l.zero_()
        self.fallback_to_indep_r.zero_()
        self.max_abs_hfunc_error_l.zero_()
        self.max_abs_hfunc_error_r.zero_()
        self.bandwidth.zero_()
        self._pdf_grid = self._empty_grid()
        self._cdf_grid = self._empty_grid()
        self._hfunc_l_grid = self._empty_grid()
        self._hfunc_r_grid = self._empty_grid()
        self.last_hinv_status_l = None
        self.last_hinv_status_r = None

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        mtd_kde: str | None = None,
        mtd_tll: str = "constant",
        num_iter_max: int = 5,
        is_tau_est: bool = False,
        *,
        bicop_backend: str | None = None,
        bicop_kwargs: dict[str, Any] | None = None,
        kde_backend: str | None = None,
        bandwidth: str | float | torch.Tensor = "silverman",
        bandwidth_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> None:
        del generator
        device, dtype = self.device, self.dtype
        obs = obs.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
        normalized_backend, normalized_kwargs = self._normalize_fit_request(
            bicop_backend=bicop_backend,
            kde_backend=kde_backend,
            mtd_kde=mtd_kde,
            bicop_kwargs=bicop_kwargs,
            mtd_tll=mtd_tll,
            num_iter_max=num_iter_max,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
        )
        self.reset()
        self.is_indep = False
        self.num_obs.copy_(torch.tensor(obs.shape[0], device=device, dtype=torch.int))
        if is_tau_est:
            self.tau.copy_(kendall_tau(obs[:, [0]], obs[:, [1]]).to(device=device, dtype=dtype))
        estimator = build_bicop_estimator(
            backend_name=normalized_backend,
            obs=obs.to(dtype=torch.float64),
            num_step_grid=self.num_step_grid,
            backend_kwargs=normalized_kwargs,
        )
        self._assign_fit_bundle(estimator)
        self.negloglik.copy_(-self.log_pdf(obs=obs).sum())

    def _clamp_unit(
        self,
        obs: torch.Tensor,
        *,
        eps: float,
        use_boundary_policy: bool,
    ) -> torch.Tensor:
        lower, upper = eps, 1.0 - eps
        clamped = obs.clamp(lower, upper)
        if use_boundary_policy and self.boundary_policy == "st":
            return obs + (clamped - obs).detach()
        return clamped

    def _interp(
        self,
        grid: torch.Tensor,
        obs: torch.Tensor,
        *,
        eps: float,
        use_boundary_policy: bool,
    ) -> torch.Tensor:
        idx = self._clamp_unit(obs, eps=eps, use_boundary_policy=use_boundary_policy) / self.step_grid
        i0 = idx.floor().long()
        di = idx - i0
        i1 = torch.minimum(
            i0 + 1,
            torch.full_like(input=i0, fill_value=int(self._target), device=idx.device),
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

    @torch.no_grad()
    def _bisect_hinv(
        self,
        *,
        fixed: torch.Tensor,
        target: torch.Tensor,
        mode: Literal["l", "r"],
        num_iter: int = 64,
    ) -> torch.Tensor:
        lo = torch.zeros_like(target)
        hi = torch.ones_like(target)
        for _ in range(num_iter):
            mid = 0.5 * (lo + hi)
            values = (
                self.hfunc_l(obs=torch.hstack([fixed, mid]))
                if mode == "l"
                else self.hfunc_r(obs=torch.hstack([mid, fixed]))
            )
            move_lo = values < target
            lo = torch.where(move_lo, mid, lo)
            hi = torch.where(move_lo, hi, mid)
        return 0.5 * (lo + hi)

    @torch.no_grad()
    def _stabilize_hinv(
        self,
        *,
        fixed: torch.Tensor,
        target: torch.Tensor,
        root: torch.Tensor,
        status: dict[str, torch.Tensor],
        mode: Literal["l", "r"],
        tol: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        used_fallback = status["used_fallback"]
        if used_fallback.any():
            refined = self._bisect_hinv(fixed=fixed, target=target, mode=mode)
            root = torch.where(used_fallback, refined, root)
        values = (
            self.hfunc_l(obs=torch.hstack([fixed, root]))
            if mode == "l"
            else self.hfunc_r(obs=torch.hstack([root, fixed]))
        )
        abs_err = (values - target).abs()
        bad = (~torch.isfinite(root)) | (~torch.isfinite(values)) | (abs_err > tol)
        if bad.any():
            # Independence fallback preserves a legal copula sample without creating a spurious mode.
            root = torch.where(bad, target, root)
            values = torch.where(bad, target, values)
            abs_err = (values - target).abs()
        root = root.clamp(min=0.0, max=1.0)
        return root, abs_err, bad

    @torch.no_grad()
    def diagnostics(self) -> BiCopDiagnostics:
        return BiCopDiagnostics(
            itp_failures_l=int(self.hinv_fallback_l),
            itp_failures_r=int(self.hinv_fallback_r),
            bisect_refinements_l=int(self.hinv_bisect_l),
            bisect_refinements_r=int(self.hinv_bisect_r),
            fallback_to_indep_l=int(self.fallback_to_indep_l),
            fallback_to_indep_r=int(self.fallback_to_indep_r),
            max_abs_hfunc_error_l=float(self.max_abs_hfunc_error_l),
            max_abs_hfunc_error_r=float(self.max_abs_hfunc_error_r),
        )

    def cdf(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs.prod(dim=1, keepdim=True)
        return self._interp(
            grid=self._cdf_grid,
            obs=obs,
            eps=0.0,
            use_boundary_policy=True,
        ).unsqueeze(dim=1)

    def hfunc_l(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [1]]
        return self._interp(
            grid=self._hfunc_l_grid,
            obs=obs,
            eps=0.0,
            use_boundary_policy=True,
        ).unsqueeze(dim=1)

    def hfunc_r(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [0]]
        return self._interp(
            grid=self._hfunc_r_grid,
            obs=obs,
            eps=0.0,
            use_boundary_policy=True,
        ).unsqueeze(dim=1)

    @torch.no_grad()
    def hinv_l(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [1]]
        u_l = self._clamp_unit(obs[:, [0]], eps=0.0, use_boundary_policy=False)
        p = self._clamp_unit(obs[:, [1]], eps=0.0, use_boundary_policy=False)
        root, status = solve_ITP(
            fun=lambda u_r: self.hfunc_l(obs=torch.hstack([u_l, u_r])) - p,
            x_a=torch.zeros_like(p),
            x_b=torch.ones_like(p),
            failure_policy="linear",
            return_status=True,
        )
        root, abs_err, bad = self._stabilize_hinv(
            fixed=u_l,
            target=p,
            root=root,
            status=status,
            mode="l",
        )
        self.hinv_fallback_l.add_(status["used_fallback"].sum().to(dtype=self.hinv_fallback_l.dtype))
        self.hinv_bisect_l.add_(status["used_fallback"].sum().to(dtype=self.hinv_bisect_l.dtype))
        self.fallback_to_indep_l.add_(bad.sum().to(dtype=self.fallback_to_indep_l.dtype))
        self.max_abs_hfunc_error_l.copy_(
            torch.maximum(
                self.max_abs_hfunc_error_l,
                torch.nan_to_num(abs_err, nan=0.0, posinf=float("inf"), neginf=float("inf")).max(),
            )
        )
        self.last_hinv_status_l = status
        return root

    @torch.no_grad()
    def hinv_r(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [0]]
        u_r = self._clamp_unit(obs[:, [1]], eps=0.0, use_boundary_policy=False)
        p = self._clamp_unit(obs[:, [0]], eps=0.0, use_boundary_policy=False)
        root, status = solve_ITP(
            fun=lambda u_l: self.hfunc_r(obs=torch.hstack([u_l, u_r])) - p,
            x_a=torch.zeros_like(p),
            x_b=torch.ones_like(p),
            failure_policy="linear",
            return_status=True,
        )
        root, abs_err, bad = self._stabilize_hinv(
            fixed=u_r,
            target=p,
            root=root,
            status=status,
            mode="r",
        )
        self.hinv_fallback_r.add_(status["used_fallback"].sum().to(dtype=self.hinv_fallback_r.dtype))
        self.hinv_bisect_r.add_(status["used_fallback"].sum().to(dtype=self.hinv_bisect_r.dtype))
        self.fallback_to_indep_r.add_(bad.sum().to(dtype=self.fallback_to_indep_r.dtype))
        self.max_abs_hfunc_error_r.copy_(
            torch.maximum(
                self.max_abs_hfunc_error_r,
                torch.nan_to_num(abs_err, nan=0.0, posinf=float("inf"), neginf=float("inf")).max(),
            )
        )
        self.last_hinv_status_r = status
        return root

    def pdf(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return torch.ones_like(obs[:, [0]])
        return self._interp(
            grid=self._pdf_grid,
            obs=obs,
            eps=self._EPS,
            use_boundary_policy=True,
        ).unsqueeze(dim=1)

    def log_pdf(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return torch.zeros_like(obs[:, [0]])
        return self.pdf(obs=obs).clamp_min(_EPS).log()

    @torch.no_grad()
    def sample(
        self,
        num_sample: int = 100,
        seed: int | None = 42,
        is_sobol: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device, dtype = self.device, self.dtype
        if is_sobol:
            obs = (
                torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
                .draw(n=num_sample, dtype=dtype)
                .to(device=device)
            )
        else:
            if generator is None:
                generator = torch.Generator(device=device)
                if seed is not None:
                    generator.manual_seed(seed)
            obs = torch.rand(size=(num_sample, 2), dtype=dtype, device=device, generator=generator)
        if not self.is_indep:
            obs[:, [1]] = self.hinv_l(obs=obs)
        return obs

    def __str__(self) -> str:
        return f"""{self.__class__.__name__}\n{
            pformat(
                object={
                    "is_indep": self.is_indep,
                    "num_obs": int(self.num_obs),
                    "negloglik": self.negloglik.round(decimals=4),
                    "num_step_grid": self.num_step_grid,
                    "tau": self.tau.round(decimals=4),
                    "bicop_backend": self.bicop_backend,
                    "backend_config": self.backend_config,
                    "boundary_policy": self.boundary_policy,
                    "bandwidth": self.bandwidth.round(decimals=6),
                    "hinv_fallback_l": int(self.hinv_fallback_l),
                    "hinv_fallback_r": int(self.hinv_fallback_r),
                    "hinv_bisect_l": int(self.hinv_bisect_l),
                    "hinv_bisect_r": int(self.hinv_bisect_r),
                    "fallback_to_indep_l": int(self.fallback_to_indep_l),
                    "fallback_to_indep_r": int(self.fallback_to_indep_r),
                    "max_abs_hfunc_error_l": float(self.max_abs_hfunc_error_l),
                    "max_abs_hfunc_error_r": float(self.max_abs_hfunc_error_r),
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
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        grid = self._pdf_grid if self._pdf_grid.numel() > 0 else torch.ones(
            self.num_step_grid, self.num_step_grid, dtype=self.dtype, device=self.device
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
        g_tensor = torch.from_numpy(np.stack(g, axis=-1).reshape(-1, 2)).to(device=self.device, dtype=self.dtype)
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
