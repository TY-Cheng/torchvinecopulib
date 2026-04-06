import pytest
import torch

from torchvinecopulib.backends import (
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
