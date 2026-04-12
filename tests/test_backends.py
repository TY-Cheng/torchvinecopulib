import builtins

import numpy as np
import pytest
import torch

import torchvinecopulib.backends.bicop as bicop_pkg
import torchvinecopulib.backends.common as common_pkg
import torchvinecopulib.backends.marginal as marginal_pkg
import torchvinecopulib.backends.bicop as bicop_mod
from torchvinecopulib.backends import (
    BetaBicopEstimator,
    BetaQtBicopEstimator,
    DEFAULT_BICOP_BACKEND,
    GridProbitBicopEstimator,
    GridReflectBicopEstimator,
    LocalPolynomialKDE1D,
    SplinePenBicopEstimator,
    TtCvBicopEstimator,
    TtPiBicopEstimator,
    Tll1BicopEstimator,
    Tll1NnBicopEstimator,
    Tll2BicopEstimator,
    Tll2NnBicopEstimator,
    TllRefBicopEstimator,
    build_bicop_estimator,
    build_marginal_estimator,
    default_bicop_kwargs,
    normalize_bicop_backend,
    normalize_marginal_backend,
)

from . import gaussian_copula


def test_backend_alias_normalization():
    assert normalize_marginal_backend("grid") == "grid"
    assert normalize_marginal_backend("torch_grid") == "grid"
    assert normalize_marginal_backend("lp_torch") == "lp"
    assert normalize_marginal_backend("kde1d_torch") == "lp"
    assert normalize_bicop_backend("grid_reflect") == "grid_reflect"
    assert normalize_bicop_backend("torch_grid") == "grid_reflect"
    assert normalize_bicop_backend("tll1") == "tll1"
    assert normalize_bicop_backend("tll2nn") == "tll2nn"
    assert normalize_bicop_backend("ttcv") == "ttcv"
    assert normalize_bicop_backend("ttpi") == "ttpi"
    assert DEFAULT_BICOP_BACKEND == "beta"


def test_backend_factories_build_expected_estimators():
    x = torch.randn(128, 1, dtype=torch.float64)
    marginal = build_marginal_estimator(
        backend_name="grid",
        x=x,
        backend_kwargs={"bandwidth": "isj", "num_step_grid": 65},
    )
    assert marginal.backend_name == "grid"
    lp = build_marginal_estimator(
        backend_name="lp",
        x=x,
        backend_kwargs={"num_step_grid": 65, "degree": 1},
    )
    assert isinstance(lp, LocalPolynomialKDE1D)
    obs = gaussian_copula(num_obs=256, rho=0.4)
    bicop = build_bicop_estimator(
        backend_name="grid_reflect",
        obs=obs,
        num_step_grid=65,
        backend_kwargs={"bandwidth": "silverman"},
    )
    assert bicop.backend_name == "grid_reflect"
    assert default_bicop_kwargs("tll2", num_step_grid=65)["num_step_grid"] == 65
    assert default_bicop_kwargs("ttcv", num_step_grid=65)["bandwidth"] == "auto"
    assert default_bicop_kwargs("ttcv", num_step_grid=65)["mult"] == 1.0
    assert default_bicop_kwargs("beta", num_step_grid=65)["bandwidth"] == "auto"


def test_backend_packages_reexport_expected_symbols():
    assert bicop_pkg.DEFAULT_BICOP_BACKEND == DEFAULT_BICOP_BACKEND
    assert bicop_pkg.normalize_bicop_backend("grid") == "grid_reflect"
    assert bicop_pkg.BaseBiCopEstimator is not None
    assert bicop_pkg.GridReflectBicopEstimator is GridReflectBicopEstimator
    assert bicop_pkg.GridProbitBicopEstimator is GridProbitBicopEstimator
    assert bicop_pkg.Tll1BicopEstimator is Tll1BicopEstimator
    assert bicop_pkg.Tll2BicopEstimator is Tll2BicopEstimator
    assert bicop_pkg.Tll1NnBicopEstimator is Tll1NnBicopEstimator
    assert bicop_pkg.Tll2NnBicopEstimator is Tll2NnBicopEstimator
    assert bicop_pkg.TtCvBicopEstimator is TtCvBicopEstimator
    assert bicop_pkg.TtPiBicopEstimator is TtPiBicopEstimator
    assert bicop_pkg.BetaBicopEstimator is BetaBicopEstimator
    assert bicop_pkg.SplinePenBicopEstimator is SplinePenBicopEstimator
    assert bicop_pkg.TllRefBicopEstimator is TllRefBicopEstimator
    assert callable(bicop_pkg._lazy_pyvinecopulib)
    assert callable(common_pkg.fit_grid_reflect_bicop)
    assert callable(common_pkg.fit_grid_probit_bicop)
    assert common_pkg.BicopFitResult is not None
    assert common_pkg.MarginalFitResult is not None
    assert common_pkg.API_VERSION
    assert callable(common_pkg._smooth_1d)
    assert callable(common_pkg._normalize_backend_kwargs)
    assert marginal_pkg.normalize_marginal_backend("lp_torch") == "lp"
    assert marginal_pkg.BaseMarginal1D is not None
    assert marginal_pkg.GridKDE1D is not None
    assert marginal_pkg.LocalPolynomialKDE1D is LocalPolynomialKDE1D
    assert callable(marginal_pkg._resolve_grid_bandwidth)
    assert callable(marginal_pkg._select_lp_bandwidth)


def test_invalid_backend_names_raise():
    x = torch.randn(32, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        build_marginal_estimator(backend_name="unknown", x=x)
    obs = gaussian_copula(num_obs=64, rho=0.3)
    with pytest.raises(ValueError):
        build_bicop_estimator(backend_name="unknown", obs=obs, num_step_grid=33)


@pytest.mark.parametrize(
    ("backend_name", "expected_cls", "backend_kwargs"),
    [
        ("tll1", Tll1BicopEstimator, {"bandwidth": "auto"}),
        ("tll2", Tll2BicopEstimator, {"bandwidth": torch.tensor([[0.12, 0.01], [0.01, 0.08]])}),
        ("tll1nn", Tll1NnBicopEstimator, {"bandwidth": "auto", "nn_k": 16}),
        ("tll2nn", Tll2NnBicopEstimator, {"bandwidth": "auto", "nn_k": 16}),
        ("beta", BetaBicopEstimator, {"bandwidth": "auto"}),
        ("beta_qt", BetaQtBicopEstimator, {"bandwidth": 0.06, "transform_shape": 2.4}),
        (
            "ttcv",
            TtCvBicopEstimator,
            {
                "bandwidth": "auto",
                "mult": 0.9,
                "selector_grid_size": 9,
                "selector_num_refine": 2,
                "selector_sample_cap": 128,
            },
        ),
        (
            "ttpi",
            TtPiBicopEstimator,
            {
                "bandwidth": "auto",
                "mult": 0.9,
                "selector_grid_size": 9,
                "selector_num_refine": 2,
                "selector_sample_cap": 128,
            },
        ),
        ("spline_pen", SplinePenBicopEstimator, {"num_basis": 9, "penalty": 1e-2}),
    ],
)
def test_new_bicop_backends_smoke(backend_name, expected_cls, backend_kwargs):
    obs = gaussian_copula(num_obs=128, rho=0.45)
    est = build_bicop_estimator(
        backend_name=backend_name,
        obs=obs,
        num_step_grid=33,
        backend_kwargs=backend_kwargs,
    )
    assert isinstance(est, expected_cls)
    assert est.backend_name == backend_name
    assert est._pdf_grid.shape == (33, 33)
    assert torch.isfinite(est._pdf_grid).all()
    assert torch.isfinite(est._cdf_grid).all()
    assert torch.isfinite(est._hfunc_l_grid).all()
    assert torch.isfinite(est._hfunc_r_grid).all()


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
    with pytest.raises(ValueError):
        build_bicop_estimator(
            backend_name="ttcv",
            obs=obs,
            num_step_grid=33,
            backend_kwargs={"bandwidth": torch.tensor([0.1, 0.2, 0.3])},
        )
    with pytest.raises(ValueError):
        build_bicop_estimator(
            backend_name="ttpi",
            obs=obs,
            num_step_grid=33,
            backend_kwargs={"bandwidth_scale": 0.9},
        )


def test_aligned_backend_configs_use_canonical_bw_shapes():
    obs = gaussian_copula(num_obs=256, rho=0.4)
    tll = build_bicop_estimator(
        backend_name="tll2",
        obs=obs,
        num_step_grid=33,
        backend_kwargs={"bandwidth": "auto", "mult": 0.8},
    )
    assert tll.backend_config["bandwidth_kind"] == "auto_tll"
    assert isinstance(tll.backend_config["bw"], list)
    assert len(tll.backend_config["bw"]) == 2

    tllnn = build_bicop_estimator(
        backend_name="tll2nn",
        obs=obs,
        num_step_grid=33,
        backend_kwargs={"bandwidth": "auto", "mult": 0.8},
    )
    assert tllnn.backend_config["bandwidth_kind"] == "auto_tll_nn"
    assert set(tllnn.backend_config["bw"]) == {"B", "alpha", "kappa"}

    beta = build_bicop_estimator(
        backend_name="beta",
        obs=obs,
        num_step_grid=33,
        backend_kwargs={"bandwidth": "auto", "mult": 0.8},
    )
    assert beta.backend_config["bandwidth_kind"] == "auto_beta"
    assert isinstance(beta.backend_config["bw"], float)

    tt = build_bicop_estimator(
        backend_name="ttpi",
        obs=obs,
        num_step_grid=33,
        backend_kwargs={
            "bandwidth": "auto",
            "mult": 0.8,
            "selector_grid_size": 9,
            "selector_num_refine": 2,
            "selector_sample_cap": 128,
        },
    )
    assert tt.backend_config["bandwidth_kind"] == "auto_pi"
    assert tt.backend_config["mult"] == pytest.approx(0.8)
    assert len(tt.backend_config["tt_params"]) == 4


def test_tllnn_accepts_canonical_bw_mapping():
    obs = gaussian_copula(num_obs=256, rho=0.4)
    est = build_bicop_estimator(
        backend_name="tll1nn",
        obs=obs,
        num_step_grid=33,
        backend_kwargs={
            "bandwidth": {
                "B": [[0.2, 0.0], [0.0, 0.15]],
                "alpha": 0.3,
                "kappa": [1.0, 1.2],
            }
        },
    )
    assert est.backend_name == "tll1nn"
    assert est.backend_config["bandwidth_kind"] == "custom_nn"


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


def test_lazy_pyvinecopulib_error_message(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyvinecopulib":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="tll_ref"):
        bicop_mod._lazy_pyvinecopulib()


def test_base_bicop_estimator_fit_stub_raises():
    class DummyEstimator(bicop_mod.BaseBiCopEstimator):
        def fit(self, obs, **kwargs):
            return bicop_mod.BaseBiCopEstimator.fit(self, obs, **kwargs)

    est = DummyEstimator(num_step_grid=9)
    with pytest.raises(NotImplementedError):
        est.fit(torch.zeros(4, 2, dtype=torch.float64))
