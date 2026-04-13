"""Bivariate copula backend registry and estimator families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import sys

import torch

from ..bicop_native import (
    fit_beta_bicop,
    fit_beta_qt_bicop,
    fit_spline_pen_bicop,
    fit_tt_bicop,
    fit_tll_torch_bicop,
)
from ..common import (
    API_VERSION,
    BicopFitResult,
    _EPS,
    _build_pdf_buffers_2d,
    _copula_normalization_diagnostics,
    _normalize_backend_kwargs,
    _normalize_copula_pdf_grid,
    fit_grid_probit_bicop,
    fit_grid_reflect_bicop,
)

__all__ = [
    "BaseBiCopEstimator",
    "BetaBicopEstimator",
    "BetaQtBicopEstimator",
    "DEFAULT_BICOP_BACKEND",
    "GridProbitBicopEstimator",
    "GridReflectBicopEstimator",
    "PUBLIC_BICOP_BACKENDS",
    "SplinePenBicopEstimator",
    "TtCvBicopEstimator",
    "TtPiBicopEstimator",
    "Tll1BicopEstimator",
    "Tll1NnBicopEstimator",
    "Tll2BicopEstimator",
    "Tll2NnBicopEstimator",
    "TllRefBicopEstimator",
    "_lazy_pyvinecopulib",
    "build_bicop_estimator",
    "default_bicop_kwargs",
    "fit_beta_bicop",
    "fit_beta_qt_bicop",
    "fit_grid_probit_bicop",
    "fit_grid_reflect_bicop",
    "fit_spline_pen_bicop",
    "fit_tt_bicop",
    "fit_tll_torch_bicop",
    "normalize_bicop_backend",
]


class BaseBiCopEstimator(torch.nn.Module, ABC):
    api_version = API_VERSION

    def __init__(self, *, num_step_grid: int = 128) -> None:
        super().__init__()
        self.num_step_grid = int(num_step_grid)
        self.register_buffer("_pdf_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_cdf_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_hfunc_l_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("_hfunc_r_grid", torch.empty(0, 0, dtype=torch.float64))
        self.register_buffer("bandwidth", torch.empty(0, dtype=torch.float64))
        self.backend_name = "unknown"
        self.backend_config: dict[str, Any] = {}

    def _assign_result(self, result: BicopFitResult) -> None:
        self._pdf_grid = result.pdf_grid
        self._cdf_grid = result.cdf_grid
        self._hfunc_l_grid = result.hfunc_l_grid
        self._hfunc_r_grid = result.hfunc_r_grid
        self.bandwidth = result.bandwidth.to(
            device=self._pdf_grid.device, dtype=self._pdf_grid.dtype
        )
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
        num_iter_max: int = 2000,
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
        step = 1.0 / max(num_step_grid - 1, 1)
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
                **_copula_normalization_diagnostics(
                    pdf_grid,
                    step=step,
                    marginal_tol=marginal_tol,
                ),
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
        num_iter_max: int = 2000,
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
        num_iter_max: int = 2000,
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
        num_iter_max: int = 2000,
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
        step = 1.0 / max(self.num_step_grid - 1, 1)
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
                    **_copula_normalization_diagnostics(
                        pdf_grid,
                        step=step,
                        marginal_tol=marginal_tol,
                    ),
                },
            )
        )


class _BaseTllTorchBicopEstimator(BaseBiCopEstimator):
    degree = 1
    adaptive = False
    backend_name_default = "tll1"

    def __init__(
        self,
        obs: torch.Tensor,
        *,
        num_step_grid: int = 128,
        bandwidth: str | float | torch.Tensor | dict[str, Any] = "auto",
        mult: float | None = None,
        smoother: str = "auto",
        marginal_tol: float = 1e-3,
        num_iter_max: int = 2000,
        ridge: float = 1e-6,
        nn_k: int = 64,
        nn_alpha: float = 0.5,
        nn_min_scale: float = 0.65,
        nn_max_scale: float = 2.5,
        nn_chunk_size: int = 512,
    ) -> None:
        super().__init__(num_step_grid=num_step_grid)
        self.fit(
            obs=obs,
            bandwidth=bandwidth,
            mult=mult,
            smoother=smoother,
            marginal_tol=marginal_tol,
            num_iter_max=num_iter_max,
            ridge=ridge,
            nn_k=nn_k,
            nn_alpha=nn_alpha,
            nn_min_scale=nn_min_scale,
            nn_max_scale=nn_max_scale,
            nn_chunk_size=nn_chunk_size,
        )

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        *,
        bandwidth: str | float | torch.Tensor | dict[str, Any] = "auto",
        mult: float | None = None,
        smoother: str = "auto",
        marginal_tol: float = 1e-3,
        num_iter_max: int = 2000,
        ridge: float = 1e-6,
        nn_k: int = 64,
        nn_alpha: float = 0.5,
        nn_min_scale: float = 0.65,
        nn_max_scale: float = 2.5,
        nn_chunk_size: int = 512,
    ) -> None:
        pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, bandwidth_summary, config = (
            fit_tll_torch_bicop(
                obs=obs,
                num_step_grid=self.num_step_grid,
                bandwidth=bandwidth,
                mult=mult,
                smoother=smoother,
                marginal_tol=marginal_tol,
                num_iter_max=num_iter_max,
                degree=self.degree,
                ridge=ridge,
                nn_k=nn_k,
                nn_alpha=nn_alpha,
                nn_min_scale=nn_min_scale,
                nn_max_scale=nn_max_scale,
                nn_chunk_size=nn_chunk_size,
                adaptive=self.adaptive,
            )
        )
        self._assign_result(
            BicopFitResult(
                pdf_grid=pdf_grid,
                cdf_grid=cdf_grid,
                hfunc_l_grid=hfunc_l_grid,
                hfunc_r_grid=hfunc_r_grid,
                bandwidth=bandwidth_summary,
                backend_name=self.backend_name_default,
                backend_config=config,
            )
        )


class Tll1BicopEstimator(_BaseTllTorchBicopEstimator):
    degree = 1
    adaptive = False
    backend_name_default = "tll1"


class Tll2BicopEstimator(_BaseTllTorchBicopEstimator):
    degree = 2
    adaptive = False
    backend_name_default = "tll2"


class Tll1NnBicopEstimator(_BaseTllTorchBicopEstimator):
    degree = 1
    adaptive = True
    backend_name_default = "tll1nn"


class Tll2NnBicopEstimator(_BaseTllTorchBicopEstimator):
    degree = 2
    adaptive = True
    backend_name_default = "tll2nn"


class _BaseTtBicopEstimator(BaseBiCopEstimator):
    selector_kind = "cv"
    backend_name_default = "ttcv"

    def __init__(
        self,
        obs: torch.Tensor,
        *,
        num_step_grid: int = 128,
        bandwidth: str | float | torch.Tensor = "auto",
        mult: float | None = None,
        marginal_tol: float = 1e-3,
        num_iter_max: int = 2000,
        selector_grid_size: int = 17,
        selector_num_refine: int = 3,
        selector_sample_cap: int = 2048,
    ) -> None:
        super().__init__(num_step_grid=num_step_grid)
        self.fit(
            obs=obs,
            bandwidth=bandwidth,
            mult=mult,
            marginal_tol=marginal_tol,
            num_iter_max=num_iter_max,
            selector_grid_size=selector_grid_size,
            selector_num_refine=selector_num_refine,
            selector_sample_cap=selector_sample_cap,
        )

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        *,
        bandwidth: str | float | torch.Tensor = "auto",
        mult: float | None = None,
        marginal_tol: float = 1e-3,
        num_iter_max: int = 2000,
        selector_grid_size: int = 17,
        selector_num_refine: int = 3,
        selector_sample_cap: int = 2048,
    ) -> None:
        pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, params, config = fit_tt_bicop(
            obs=obs,
            num_step_grid=self.num_step_grid,
            bandwidth=bandwidth,
            mult=mult,
            marginal_tol=marginal_tol,
            num_iter_max=num_iter_max,
            selector_kind=self.selector_kind,
            selector_grid_size=selector_grid_size,
            selector_num_refine=selector_num_refine,
            selector_sample_cap=selector_sample_cap,
        )
        self._assign_result(
            BicopFitResult(
                pdf_grid=pdf_grid,
                cdf_grid=cdf_grid,
                hfunc_l_grid=hfunc_l_grid,
                hfunc_r_grid=hfunc_r_grid,
                bandwidth=params,
                backend_name=self.backend_name_default,
                backend_config=config,
            )
        )


class TtCvBicopEstimator(_BaseTtBicopEstimator):
    selector_kind = "cv"
    backend_name_default = "ttcv"


class TtPiBicopEstimator(_BaseTtBicopEstimator):
    selector_kind = "pi"
    backend_name_default = "ttpi"


class BetaBicopEstimator(BaseBiCopEstimator):
    def __init__(
        self,
        obs: torch.Tensor,
        *,
        num_step_grid: int = 128,
        bandwidth: str | float | torch.Tensor = "auto",
        mult: float | None = None,
        smoother: str = "auto",
        marginal_tol: float = 1e-3,
        num_iter_max: int = 2000,
    ) -> None:
        super().__init__(num_step_grid=num_step_grid)
        self.fit(
            obs=obs,
            bandwidth=bandwidth,
            mult=mult,
            smoother=smoother,
            marginal_tol=marginal_tol,
            num_iter_max=num_iter_max,
        )

    @torch.no_grad()
    def fit(
        self,
        obs: torch.Tensor,
        *,
        bandwidth: str | float | torch.Tensor = "auto",
        mult: float | None = None,
        smoother: str = "auto",
        marginal_tol: float = 1e-3,
        num_iter_max: int = 2000,
    ) -> None:
        pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, bandwidth_summary, config = fit_beta_bicop(
            obs=obs,
            num_step_grid=self.num_step_grid,
            bandwidth=bandwidth,
            mult=mult,
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
                bandwidth=bandwidth_summary,
                backend_name="beta",
                backend_config=config,
            )
        )


class BetaQtBicopEstimator(BaseBiCopEstimator):
    def __init__(
        self,
        obs: torch.Tensor,
        *,
        num_step_grid: int = 128,
        bandwidth: str | float | torch.Tensor = "silverman",
        bandwidth_scale: float = 1.0,
        smoother: str = "auto",
        marginal_tol: float = 1e-3,
        num_iter_max: int = 2000,
        transform_shape: float = 2.0,
    ) -> None:
        super().__init__(num_step_grid=num_step_grid)
        self.fit(
            obs=obs,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            smoother=smoother,
            marginal_tol=marginal_tol,
            num_iter_max=num_iter_max,
            transform_shape=transform_shape,
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
        num_iter_max: int = 2000,
        transform_shape: float = 2.0,
    ) -> None:
        pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, bandwidth_summary, config = (
            fit_beta_qt_bicop(
                obs=obs,
                num_step_grid=self.num_step_grid,
                bandwidth=bandwidth,
                bandwidth_scale=bandwidth_scale,
                smoother=smoother,
                marginal_tol=marginal_tol,
                num_iter_max=num_iter_max,
                transform_shape=transform_shape,
            )
        )
        self._assign_result(
            BicopFitResult(
                pdf_grid=pdf_grid,
                cdf_grid=cdf_grid,
                hfunc_l_grid=hfunc_l_grid,
                hfunc_r_grid=hfunc_r_grid,
                bandwidth=bandwidth_summary,
                backend_name="beta_qt",
                backend_config=config,
            )
        )


class SplinePenBicopEstimator(BaseBiCopEstimator):
    def __init__(
        self,
        obs: torch.Tensor,
        *,
        num_step_grid: int = 128,
        bandwidth: str | float | torch.Tensor = "silverman",
        bandwidth_scale: float = 1.0,
        smoother: str = "auto",
        marginal_tol: float = 1e-3,
        num_iter_max: int = 2000,
        num_basis: int = 17,
        penalty: float = 1e-2,
        spline_degree: int = 3,
    ) -> None:
        super().__init__(num_step_grid=num_step_grid)
        self.fit(
            obs=obs,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            smoother=smoother,
            marginal_tol=marginal_tol,
            num_iter_max=num_iter_max,
            num_basis=num_basis,
            penalty=penalty,
            spline_degree=spline_degree,
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
        num_iter_max: int = 2000,
        num_basis: int = 17,
        penalty: float = 1e-2,
        spline_degree: int = 3,
    ) -> None:
        pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, bandwidth_summary, config = (
            fit_spline_pen_bicop(
                obs=obs,
                num_step_grid=self.num_step_grid,
                bandwidth=bandwidth,
                bandwidth_scale=bandwidth_scale,
                smoother=smoother,
                marginal_tol=marginal_tol,
                num_iter_max=num_iter_max,
                num_basis=num_basis,
                penalty=penalty,
                spline_degree=spline_degree,
            )
        )
        self._assign_result(
            BicopFitResult(
                pdf_grid=pdf_grid,
                cdf_grid=cdf_grid,
                hfunc_l_grid=hfunc_l_grid,
                hfunc_r_grid=hfunc_r_grid,
                bandwidth=bandwidth_summary,
                backend_name="spline_pen",
                backend_config=config,
            )
        )


def _lazy_pyvinecopulib():
    bicop_pkg = sys.modules.get(__package__)
    if bicop_pkg is not None:
        override = bicop_pkg.__dict__.get("_lazy_pyvinecopulib")
        if override is not None and override is not _lazy_pyvinecopulib:
            return override()
    try:
        import pyvinecopulib as pv
    except ImportError as exc:
        raise ImportError(
            "The 'tll_ref' backend requires pyvinecopulib. Install torchvinecopulib[reference]."
        ) from exc
    return pv


class TllRefBicopEstimator(BaseBiCopEstimator):
    """Reference bicop backend delegated to `pyvinecopulib`'s TLL family.

    This backend fits `pyvinecopulib.Bicop` with `family_set=[pv.tll]`, evaluates the fitted
    reference model on the common unit-square grid, and then constructs the same cumulative buffers
    used by the native torch backends. It is intended primarily for reference comparisons and
    regression checks rather than as the lightweight production path.

    References:
        - https://vinecopulib.github.io/pyvinecopulib/_generate/pyvinecopulib.BicopFamily.html
        - https://vinecopulib.github.io/pyvinecopulib/_generate/pyvinecopulib.FitControlsBicop.__init__.html
        - Geenens, Charpentier, and Paindaveine (2017), Bernoulli 23(3), 1848-1873.
          https://doi.org/10.3150/15-BEJ798
        - Nagler (2018), Journal of Statistical Software 84(7), 1-22.
          https://doi.org/10.18637/jss.v084.i07
    """

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
        pdf_grid = _normalize_copula_pdf_grid(
            pdf_grid,
            step=step,
            marginal_tol=1e-3,
            num_iter_max=2000,
        )
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
                    "marginal_tol": 1e-3,
                    "num_iter_max": 2000,
                    **_copula_normalization_diagnostics(
                        pdf_grid,
                        step=step,
                        marginal_tol=1e-3,
                    ),
                },
            )
        )


DEFAULT_BICOP_BACKEND = "beta"
PUBLIC_BICOP_BACKENDS = (
    "grid_reflect",
    "grid_probit",
    "tll1",
    "tll2",
    "tll1nn",
    "tll2nn",
    "beta",
    "beta_qt",
    "ttcv",
    "ttpi",
    "spline_pen",
)


def normalize_bicop_backend(name: str) -> str:
    aliases = {
        "torch_grid": "grid_reflect",
        "grid": "grid_reflect",
        "grid_reflect": "grid_reflect",
        "grid_probit": "grid_probit",
        "tll1": "tll1",
        "tll2": "tll2",
        "tll1nn": "tll1nn",
        "tll2nn": "tll2nn",
        "beta": "beta",
        "beta_qt": "beta_qt",
        "ttcv": "ttcv",
        "ttpi": "ttpi",
        "spline_pen": "spline_pen",
        "tll_ref": "tll_ref",
    }
    if name not in aliases:
        raise ValueError(
            "bicop_backend must be one of 'grid_reflect', 'grid_probit', 'tll1', 'tll2', "
            "'tll1nn', 'tll2nn', 'beta', 'beta_qt', 'ttcv', 'ttpi', 'spline_pen', or 'tll_ref'."
        )
    return aliases[name]


def default_bicop_kwargs(backend_name: str, *, num_step_grid: int) -> dict[str, Any]:
    backend_name = normalize_bicop_backend(backend_name)
    generic = {
        "bandwidth": "silverman",
        "bandwidth_scale": 1.0,
        "num_step_grid": int(num_step_grid),
        "smoother": "auto",
        "marginal_tol": 1e-3,
        "num_iter_max": 2000,
    }
    if backend_name in {"grid_reflect", "grid_probit"}:
        return {
            **generic,
            "marginal_tol": 1e-3,
        }
    if backend_name in {"tll1", "tll2"}:
        return {
            "bandwidth": "auto",
            "mult": 1.0,
            "num_step_grid": int(num_step_grid),
            "smoother": "auto",
            "marginal_tol": 1e-3,
            "num_iter_max": 2000,
            "ridge": 1e-6,
        }
    if backend_name in {"tll1nn", "tll2nn"}:
        return {
            "bandwidth": "auto",
            "mult": 1.0,
            "num_step_grid": int(num_step_grid),
            "smoother": "auto",
            "marginal_tol": 1e-3,
            "num_iter_max": 2000,
            "ridge": 1e-6,
            "nn_k": 64,
            "nn_alpha": 0.5,
            "nn_min_scale": 0.65,
            "nn_max_scale": 2.5,
            "nn_chunk_size": 512,
        }
    if backend_name == "beta":
        return {
            "bandwidth": "auto",
            "mult": 1.0,
            "num_step_grid": int(num_step_grid),
            "smoother": "auto",
            "marginal_tol": 1e-3,
            "num_iter_max": 2000,
        }
    if backend_name == "beta_qt":
        return {
            **generic,
            "transform_shape": 2.0,
        }
    if backend_name in {"ttcv", "ttpi"}:
        return {
            "bandwidth": "auto",
            "mult": 1.0,
            "num_step_grid": int(num_step_grid),
            "marginal_tol": 1e-3,
            "num_iter_max": 2000,
            "selector_grid_size": 17,
            "selector_num_refine": 3,
            "selector_sample_cap": 2048,
        }
    if backend_name == "spline_pen":
        return {
            **generic,
            "num_basis": 17,
            "penalty": 1e-2,
            "spline_degree": 3,
        }
    return {
        "nonparametric_method": "constant",
        "num_step_grid": int(num_step_grid),
    }


BICOP_BACKENDS = {
    "grid_reflect": GridReflectBicopEstimator,
    "grid_probit": GridProbitBicopEstimator,
    "tll1": Tll1BicopEstimator,
    "tll2": Tll2BicopEstimator,
    "tll1nn": Tll1NnBicopEstimator,
    "tll2nn": Tll2NnBicopEstimator,
    "beta": BetaBicopEstimator,
    "beta_qt": BetaQtBicopEstimator,
    "ttcv": TtCvBicopEstimator,
    "ttpi": TtPiBicopEstimator,
    "spline_pen": SplinePenBicopEstimator,
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
    defaults = default_bicop_kwargs(backend_name, num_step_grid=num_step_grid)
    normalized = _normalize_backend_kwargs(
        backend_name=backend_name,
        kwargs=backend_kwargs,
        defaults=defaults,
        allowed=set(defaults),
    )
    estimator_cls = BICOP_BACKENDS[backend_name]
    return estimator_cls(obs=obs, **normalized)


_PUBLIC_MODULE = __name__.rsplit(".", 1)[0]
for _name in [
    "BaseBiCopEstimator",
    "GridReflectBicopEstimator",
    "GridProbitBicopEstimator",
    "Tll1BicopEstimator",
    "Tll2BicopEstimator",
    "Tll1NnBicopEstimator",
    "Tll2NnBicopEstimator",
    "TtCvBicopEstimator",
    "TtPiBicopEstimator",
    "BetaBicopEstimator",
    "BetaQtBicopEstimator",
    "SplinePenBicopEstimator",
    "TllRefBicopEstimator",
]:
    globals()[_name].__module__ = _PUBLIC_MODULE
