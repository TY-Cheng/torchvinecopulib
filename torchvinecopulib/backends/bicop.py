from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from .common import (
    API_VERSION,
    BicopFitResult,
    _EPS,
    _build_pdf_buffers_2d,
    _normalize_backend_kwargs,
    fit_grid_probit_bicop,
    fit_grid_reflect_bicop,
)

__all__ = [
    "BaseBiCopEstimator",
    "GridProbitBicopEstimator",
    "GridReflectBicopEstimator",
    "TllRefBicopEstimator",
    "TorchCopulaKDE2D",
    "build_bicop_estimator",
    "normalize_bicop_backend",
]


def _lazy_pyvinecopulib():
    try:
        import pyvinecopulib as pv
    except ImportError as exc:
        raise ImportError(
            "The 'tll_ref' backend requires pyvinecopulib. Install torchvinecopulib[reference]."
        ) from exc
    return pv


def normalize_bicop_backend(name: str) -> str:
    aliases = {
        "torch_grid": "grid_reflect",
        "grid": "grid_reflect",
        "grid_reflect": "grid_reflect",
        "grid_probit": "grid_probit",
        "tll_ref": "tll_ref",
    }
    if name not in aliases:
        raise ValueError(
            "bicop_backend must be one of 'grid_reflect', 'grid_probit', or 'tll_ref'."
        )
    return aliases[name]


class BaseBiCopEstimator(torch.nn.Module, ABC):
    api_version = API_VERSION

    def __init__(self, *, num_step_grid: int = 128) -> None:
        super().__init__()
        self.num_step_grid = int(num_step_grid)
        self.register_buffer("_pdf_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_cdf_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_hfunc_l_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_hfunc_r_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("bandwidth", torch.empty(2, dtype=torch.float64))
        self.backend_name = "unknown"
        self.backend_config: dict[str, Any] = {}

    def _assign_result(self, result: BicopFitResult) -> None:
        self._pdf_grid = result.pdf_grid
        self._cdf_grid = result.cdf_grid
        self._hfunc_l_grid = result.hfunc_l_grid
        self._hfunc_r_grid = result.hfunc_r_grid
        self.bandwidth.copy_(result.bandwidth.reshape(-1)[:2].to(dtype=self.bandwidth.dtype))
        self.backend_name = result.backend_name
        self.backend_config = result.backend_config

    @abstractmethod
    def fit(self, obs: torch.Tensor, **kwargs: Any) -> None:
        raise NotImplementedError


class GridReflectBicopEstimator(BaseBiCopEstimator):
    def __init__(
        self,
        obs: torch.Tensor,
        *,
        num_step_grid: int = 128,
        bandwidth: str | float | torch.Tensor = "silverman",
        bandwidth_scale: float = 1.0,
        smoother: str = "auto",
        marginal_tol: float = 1e-3,
        num_iter_max: int = 5,
    ) -> None:
        super().__init__(num_step_grid=num_step_grid)
        self.fit(
            obs=obs,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            smoother=smoother,
            marginal_tol=marginal_tol,
            num_iter_max=num_iter_max,
        )

    @classmethod
    def from_result(cls, result: BicopFitResult) -> "GridReflectBicopEstimator":
        self = cls.__new__(cls)
        BaseBiCopEstimator.__init__(self, num_step_grid=result.pdf_grid.shape[0])
        self._assign_result(result)
        return self

    @staticmethod
    def fit_reference(
        obs: torch.Tensor,
        *,
        num_step_grid: int,
        bandwidth: str | float | torch.Tensor,
        bandwidth_scale: float,
        smoother: str,
        marginal_tol: float,
        num_iter_max: int,
    ) -> BicopFitResult:
        pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, h = fit_grid_reflect_bicop(
            obs=obs,
            num_step_grid=num_step_grid,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            smoother=smoother,
            marginal_tol=marginal_tol,
            num_iter_max=num_iter_max,
        )
        return BicopFitResult(
            pdf_grid=pdf_grid,
            cdf_grid=cdf_grid,
            hfunc_l_grid=hfunc_l_grid,
            hfunc_r_grid=hfunc_r_grid,
            bandwidth=h,
            backend_name="grid_reflect",
            backend_config={
                "bandwidth": bandwidth if isinstance(bandwidth, str) else "custom",
                "bandwidth_scale": float(bandwidth_scale),
                "smoother": smoother,
                "marginal_tol": float(marginal_tol),
                "num_iter_max": int(num_iter_max),
                "num_step_grid": int(num_step_grid),
            },
        )

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        *,
        bandwidth: str | float | torch.Tensor = "silverman",
        bandwidth_scale: float = 1.0,
        smoother: str = "auto",
        marginal_tol: float = 1e-3,
        num_iter_max: int = 5,
    ) -> None:
        self._assign_result(
            self.fit_reference(
                obs=obs,
                num_step_grid=self.num_step_grid,
                bandwidth=bandwidth,
                bandwidth_scale=bandwidth_scale,
                smoother=smoother,
                marginal_tol=marginal_tol,
                num_iter_max=num_iter_max,
            )
        )


class GridProbitBicopEstimator(BaseBiCopEstimator):
    def __init__(
        self,
        obs: torch.Tensor,
        *,
        num_step_grid: int = 128,
        bandwidth: str | float | torch.Tensor = "silverman",
        bandwidth_scale: float = 1.0,
        smoother: str = "auto",
        marginal_tol: float = 1e-3,
        num_iter_max: int = 5,
    ) -> None:
        super().__init__(num_step_grid=num_step_grid)
        self.fit(
            obs=obs,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            smoother=smoother,
            marginal_tol=marginal_tol,
            num_iter_max=num_iter_max,
        )

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        *,
        bandwidth: str | float | torch.Tensor = "silverman",
        bandwidth_scale: float = 1.0,
        smoother: str = "auto",
        marginal_tol: float = 1e-3,
        num_iter_max: int = 5,
    ) -> None:
        pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, h = fit_grid_probit_bicop(
            obs=obs,
            num_step_grid=self.num_step_grid,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            smoother=smoother,
            marginal_tol=marginal_tol,
            num_iter_max=num_iter_max,
        )
        self._assign_result(
            BicopFitResult(
                pdf_grid=pdf_grid,
                cdf_grid=cdf_grid,
                hfunc_l_grid=hfunc_l_grid,
                hfunc_r_grid=hfunc_r_grid,
                bandwidth=h,
                backend_name="grid_probit",
                backend_config={
                    "bandwidth": bandwidth if isinstance(bandwidth, str) else "custom",
                    "bandwidth_scale": float(bandwidth_scale),
                    "smoother": smoother,
                    "marginal_tol": float(marginal_tol),
                    "num_iter_max": int(num_iter_max),
                    "num_step_grid": int(self.num_step_grid),
                },
            )
        )


class TllRefBicopEstimator(BaseBiCopEstimator):
    def __init__(
        self,
        obs: torch.Tensor,
        *,
        num_step_grid: int = 128,
        nonparametric_method: str = "constant",
    ) -> None:
        super().__init__(num_step_grid=num_step_grid)
        self.fit(obs=obs, nonparametric_method=nonparametric_method)

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        *,
        nonparametric_method: str = "constant",
    ) -> None:
        pv = _lazy_pyvinecopulib()
        obs = obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS)
        controls = pv.FitControlsBicop(
            family_set=[pv.tll],
            num_threads=torch.get_num_threads(),
            nonparametric_method=nonparametric_method,
        )
        cop = pv.Bicop.from_data(data=obs.cpu().numpy(), controls=controls)
        axis = torch.linspace(_EPS, 1.0 - _EPS, steps=self.num_step_grid, dtype=torch.float64)
        pdf_grid = (
            torch.from_numpy(cop.pdf(torch.cartesian_prod(axis, axis).view(-1, 2).numpy()))
            .view(self.num_step_grid, self.num_step_grid)
            .to(dtype=torch.float64)
            .clamp_min(_EPS)
        )
        step = 1.0 / max(self.num_step_grid - 1, 1)
        pdf_grid /= pdf_grid.sum().clamp_min(_EPS) * step**2
        cdf_grid, hfunc_l_grid, hfunc_r_grid = _build_pdf_buffers_2d(pdf_grid=pdf_grid, step=step)
        self._assign_result(
            BicopFitResult(
                pdf_grid=pdf_grid,
                cdf_grid=cdf_grid,
                hfunc_l_grid=hfunc_l_grid,
                hfunc_r_grid=hfunc_r_grid,
                bandwidth=torch.zeros(2, dtype=torch.float64),
                backend_name="tll_ref",
                backend_config={
                    "nonparametric_method": nonparametric_method,
                    "num_step_grid": int(self.num_step_grid),
                },
            )
        )


BICOP_BACKENDS = {
    "grid_reflect": GridReflectBicopEstimator,
    "grid_probit": GridProbitBicopEstimator,
    "tll_ref": TllRefBicopEstimator,
}


def build_bicop_estimator(
    *,
    backend_name: str,
    obs: torch.Tensor,
    num_step_grid: int,
    backend_kwargs: dict[str, Any] | None = None,
) -> BaseBiCopEstimator:
    backend_name = normalize_bicop_backend(backend_name)
    defaults = (
        {
            "bandwidth": "silverman",
            "bandwidth_scale": 1.0,
            "num_step_grid": num_step_grid,
            "smoother": "auto",
            "marginal_tol": 1e-3,
            "num_iter_max": 5,
        }
        if backend_name != "tll_ref"
        else {
            "nonparametric_method": "constant",
            "num_step_grid": num_step_grid,
        }
    )
    allowed = set(defaults)
    normalized = _normalize_backend_kwargs(
        backend_name=backend_name,
        kwargs=backend_kwargs,
        defaults=defaults,
        allowed=allowed,
    )
    estimator_cls = BICOP_BACKENDS[backend_name]
    return estimator_cls(obs=obs, **normalized)


TorchCopulaKDE2D = GridReflectBicopEstimator
