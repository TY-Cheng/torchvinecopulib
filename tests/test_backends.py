import numpy as np
import pytest
import torch

import torchvinecopulib.backends.bicop as bicop_mod
from torchvinecopulib.backends import (
    GridProbitBicopEstimator,
    GridReflectBicopEstimator,
    TllRefBicopEstimator,
    build_bicop_estimator,
    build_marginal_estimator,
    normalize_bicop_backend,
    normalize_marginal_backend,
)

from . import gaussian_copula


def test_backend_alias_normalization():
    assert normalize_marginal_backend("grid") == "grid"
    assert normalize_marginal_backend("torch_grid") == "grid"
    assert normalize_bicop_backend("grid_reflect") == "grid_reflect"
    assert normalize_bicop_backend("torch_grid") == "grid_reflect"


def test_backend_factories_build_expected_estimators():
    x = torch.randn(128, 1, dtype=torch.float64)
    marginal = build_marginal_estimator(
        backend_name="grid",
        x=x,
        backend_kwargs={"bandwidth": "isj", "num_step_grid": 65},
    )
    assert marginal.backend_name == "grid"
    obs = gaussian_copula(num_obs=256, rho=0.4)
    bicop = build_bicop_estimator(
        backend_name="grid_reflect",
        obs=obs,
        num_step_grid=65,
        backend_kwargs={"bandwidth": "silverman"},
    )
    assert bicop.backend_name == "grid_reflect"


def test_invalid_backend_names_raise():
    x = torch.randn(32, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        build_marginal_estimator(backend_name="unknown", x=x)
    obs = gaussian_copula(num_obs=64, rho=0.3)
    with pytest.raises(ValueError):
        build_bicop_estimator(backend_name="unknown", obs=obs, num_step_grid=33)


def test_grid_bicop_estimators_cover_recursive_and_custom_bandwidth():
    obs = gaussian_copula(num_obs=512, rho=0.45)
    reflect = GridReflectBicopEstimator(
        obs=obs,
        num_step_grid=65,
        bandwidth=torch.tensor(0.2, dtype=torch.float64),
        smoother="recursive",
    )
    assert reflect.backend_name == "grid_reflect"
    result = GridReflectBicopEstimator.fit_reference(
        obs=obs,
        num_step_grid=65,
        bandwidth="silverman",
        bandwidth_scale=1.0,
        smoother="auto",
        marginal_tol=1e-3,
        num_iter_max=5,
    )
    clone = GridReflectBicopEstimator.from_result(result)
    probit = GridProbitBicopEstimator(
        obs=obs,
        num_step_grid=65,
        bandwidth=torch.tensor(0.15, dtype=torch.float64),
    )
    assert clone._pdf_grid.shape == reflect._pdf_grid.shape
    assert probit.backend_name == "grid_probit"
    assert torch.isfinite(probit._pdf_grid).all()


def test_invalid_bicop_backend_kwargs_raise():
    obs = gaussian_copula(num_obs=128, rho=0.4)
    with pytest.raises(ValueError):
        build_bicop_estimator(
            backend_name="grid_reflect",
            obs=obs,
            num_step_grid=33,
            backend_kwargs={"smoother": "unknown"},
        )
    with pytest.raises(ValueError):
        GridProbitBicopEstimator(obs=obs, num_step_grid=33, bandwidth="unsupported")


def test_tll_ref_backend_with_fake_reference(monkeypatch):
    class FakeControls:
        def __init__(self, family_set, num_threads, nonparametric_method):
            self.family_set = family_set
            self.num_threads = num_threads
            self.nonparametric_method = nonparametric_method

    class FakeCopula:
        def pdf(self, data):
            return np.ones(len(data), dtype=np.float64)

    class FakeBicop:
        @staticmethod
        def from_data(data, controls):
            assert controls.nonparametric_method == "linear"
            return FakeCopula()

    class FakePV:
        tll = "tll"
        FitControlsBicop = FakeControls
        Bicop = FakeBicop

    monkeypatch.setattr(bicop_mod, "_lazy_pyvinecopulib", lambda: FakePV)
    obs = gaussian_copula(num_obs=64, rho=0.35)
    est = build_bicop_estimator(
        backend_name="tll_ref",
        obs=obs,
        num_step_grid=17,
        backend_kwargs={"nonparametric_method": "linear"},
    )
    assert isinstance(est, TllRefBicopEstimator)
    assert est.backend_name == "tll_ref"
    assert est._pdf_grid.shape == (17, 17)


def test_tll_ref_backend_missing_dependency_raises(monkeypatch):
    monkeypatch.setattr(
        bicop_mod,
        "_lazy_pyvinecopulib",
        lambda: (_ for _ in ()).throw(ImportError("missing pyvinecopulib")),
    )
    obs = gaussian_copula(num_obs=32, rho=0.25)
    with pytest.raises(ImportError):
        TllRefBicopEstimator(obs=obs, num_step_grid=17)
