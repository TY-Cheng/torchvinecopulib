import math

import pytest
import torch

import torchvinecopulib.backends.bicop_native as native

from . import gaussian_copula


def _obs(num_obs: int = 96, rho: float = 0.45) -> torch.Tensor:
    return gaussian_copula(num_obs=num_obs, rho=rho).to(dtype=torch.float64)


def test_bandwidth_matrix_helpers_cover_custom_and_invalid_specs():
    obs = _obs()
    regularized = native._regularize_bandwidth_matrix(
        torch.tensor([[0.1, 0.3], [0.0, -0.2]], dtype=torch.float64)
    )
    assert bool((torch.linalg.eigvalsh(regularized) > 0.0).all())

    H_diag, summary_diag, cfg_diag = native._bandwidth_matrix_from_spec(
        obs,
        bandwidth=torch.tensor([0.1, 0.2], dtype=torch.float64),
        bandwidth_scale=1.5,
    )
    assert cfg_diag["bandwidth_kind"] == "diag"
    assert summary_diag.shape == (4,)
    assert H_diag[0, 0] == pytest.approx((0.1 * 1.5) ** 2)

    H_matrix, _, cfg_matrix = native._bandwidth_matrix_from_spec(
        obs,
        bandwidth=torch.tensor([[0.2, 0.05], [0.05, -0.1]], dtype=torch.float64),
        bandwidth_scale=1.0,
    )
    assert cfg_matrix["bandwidth_kind"] == "matrix"
    assert bool((torch.linalg.eigvalsh(H_matrix) > 0.0).all())

    with pytest.raises(ValueError, match="Unsupported copula bandwidth"):
        native._bandwidth_matrix_from_spec(obs, bandwidth="unknown", bandwidth_scale=1.0)
    with pytest.raises(ValueError, match="copula bandwidth must be"):
        native._bandwidth_matrix_from_spec(
            obs, bandwidth=torch.ones(3, dtype=torch.float64), bandwidth_scale=1.0
        )


def test_resolve_mult_uses_only_canonical_mult():
    assert native._resolve_mult() == pytest.approx(1.0)
    assert native._resolve_mult(mult=0.4) == pytest.approx(0.4)


def test_tll_bandwidth_helper_paths_cover_auto_and_errors():
    obs = _obs(num_obs=80)
    nn_spec = native._bw_tll_nn_auto(obs, degree=2, mult=0.8, sample_cap=16)
    assert nn_spec["selector_sample_size"] <= 16

    with pytest.raises(ValueError, match="provide 'B', 'alpha', and 'kappa'"):
        native._tll_bandwidth_from_spec(
            obs,
            bandwidth={"B": [[0.2, 0.0], [0.0, 0.15]], "alpha": 0.2},
            mult=1.0,
            degree=2,
            adaptive=True,
            nn_k=8,
        )
    with pytest.raises(ValueError, match="must have length 2"):
        native._tll_bandwidth_from_spec(
            obs,
            bandwidth={"B": [[0.2, 0.0], [0.0, 0.15]], "alpha": 0.2, "kappa": [1.0]},
            mult=1.0,
            degree=2,
            adaptive=True,
            nn_k=8,
        )
    with pytest.raises(ValueError, match="support bandwidth='auto'"):
        native._tll_bandwidth_from_spec(
            obs,
            bandwidth="bad",
            mult=1.0,
            degree=2,
            adaptive=True,
            nn_k=8,
        )

    _, alpha, kappa, summary, cfg, _ = native._tll_bandwidth_from_spec(
        obs,
        bandwidth="auto",
        mult=0.9,
        degree=2,
        adaptive=True,
        nn_k=8,
    )
    assert cfg["bandwidth_kind"] == "auto_tll_nn"
    assert alpha > 0.0
    assert kappa.shape == (2,)
    assert summary.numel() == 7

    with pytest.raises(ValueError, match="support bandwidth='auto'"):
        native._tll_bandwidth_from_spec(
            obs,
            bandwidth=0.14,
            mult=1.0,
            degree=2,
            adaptive=True,
            nn_k=8,
        )

    B, _, _, _, cfg, _ = native._tll_bandwidth_from_spec(
        obs,
        bandwidth="auto",
        mult=1.0,
        degree=1,
        adaptive=False,
        nn_k=8,
    )
    assert cfg["bandwidth_kind"] == "auto_tll"
    assert B.shape == (2, 2)

    with pytest.raises(ValueError, match="custom 2x2 matrix"):
        native._tll_bandwidth_from_spec(
            obs,
            bandwidth=torch.tensor([0.12, 0.18], dtype=torch.float64),
            mult=1.0,
            degree=1,
            adaptive=False,
            nn_k=8,
        )

    with pytest.raises(ValueError, match="bandwidth='auto'"):
        native._tll_bandwidth_from_spec(
            obs,
            bandwidth="bad",
            mult=1.0,
            degree=1,
            adaptive=False,
            nn_k=8,
        )


def test_beta_bandwidth_helper_paths_cover_auto_and_errors():
    obs = _obs()
    bw, summary, cfg = native._beta_bandwidth_from_spec(obs, bandwidth="auto", mult=0.85)
    assert cfg["bandwidth_kind"] == "auto_beta"
    assert bw > 0.0
    assert summary.shape == (1,)

    with pytest.raises(ValueError, match="positive scalar"):
        native._beta_bandwidth_from_spec(obs, bandwidth="silverman", mult=1.0)

    with pytest.raises(ValueError, match="positive scalar"):
        native._beta_bandwidth_from_spec(
            obs,
            bandwidth=torch.tensor([0.08, 0.18], dtype=torch.float64),
            mult=1.0,
        )

    with pytest.raises(ValueError, match="positive scalar"):
        native._beta_bandwidth_from_spec(
            obs,
            bandwidth=torch.tensor([[0.04, 0.0], [0.0, 0.09]], dtype=torch.float64),
            mult=1.0,
        )

    with pytest.raises(ValueError, match="beta bandwidth must be"):
        native._beta_bandwidth_from_spec(
            obs, bandwidth=torch.ones(3, dtype=torch.float64), mult=1.0
        )


def test_tll_density_helper_fallback_paths(monkeypatch):
    empty = native._tll_density_at_point(
        torch.empty(0, 2, dtype=torch.float64),
        scale_prod=1.0,
        degree=1,
        ridge=1e-6,
        n_total=10,
    )
    assert empty.item() == pytest.approx(native._EPS)

    short = native._tll_density_at_point(
        torch.randn(4, 2, dtype=torch.float64),
        scale_prod=1.0,
        degree=2,
        ridge=1e-6,
        n_total=20,
    )
    assert short.item() > 0.0

    beta_bad = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    assert native._tll_integral_moments(beta_bad, degree=2) is None

    diffs = torch.randn(64, 2, dtype=torch.float64)
    monkeypatch.setattr(native, "_tll_integral_moments", lambda beta, degree: None)
    fallback = native._tll_density_at_point(
        diffs,
        scale_prod=1.0,
        degree=2,
        ridge=1e-6,
        n_total=64,
    )
    assert fallback.item() > 0.0


def test_tll_density_helper_uses_lstsq_and_nonfinite_density_fallback(monkeypatch):
    diffs = torch.randn(64, 2, dtype=torch.float64)

    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("force lstsq fallback")

    monkeypatch.setattr(native.torch.linalg, "solve", _raise_runtime)
    val = native._tll_density_at_point(
        diffs,
        scale_prod=1.0,
        degree=1,
        ridge=1e-6,
        n_total=64,
    )
    assert val.item() > 0.0

    real_exp = native.torch.exp

    def _inf_scalar_exp(x):
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            return torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        return real_exp(x)

    monkeypatch.setattr(native.torch, "exp", _inf_scalar_exp)
    val = native._tll_density_at_point(
        torch.randn(32, 2, dtype=torch.float64),
        scale_prod=1.0,
        degree=1,
        ridge=1e-6,
        n_total=32,
        max_iter=0,
    )
    assert math.isfinite(float(val.item()))


def test_tt_helper_error_paths_and_selector_fallbacks(monkeypatch):
    with pytest.raises(ValueError, match="must be positive"):
        native._tt_check_params(
            torch.tensor([0.0, 0.0, 0.1, 0.0], dtype=torch.float64), strict=True
        )
    with pytest.raises(ValueError, match="rho"):
        native._tt_check_params(
            torch.tensor([0.1, 1.0, 0.1, 0.0], dtype=torch.float64), strict=True
        )
    with pytest.raises(ValueError, match="theta1"):
        native._tt_check_params(
            torch.tensor([0.1, 0.0, 0.0, 0.0], dtype=torch.float64), strict=True
        )
    with pytest.raises(ValueError, match="at least three observations"):
        native._tt_selector_stats(_obs(num_obs=2), pilot_b=0.4)

    C1 = torch.eye(2, dtype=torch.float64)
    rhs = torch.ones(2, dtype=torch.float64)

    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("force lstsq fallback")

    monkeypatch.setattr(native.torch.linalg, "solve", _raise_runtime)
    solved = native._tt_solve_c1(C1, rhs)
    assert solved.shape == (2,)

    stats = native._tt_selector_stats(_obs(num_obs=32), pilot_b=0.4)
    assert math.isinf(native._tt_M_objective(1.2, stats))

    monkeypatch.setattr(
        native,
        "_tt_C2_C3",
        lambda stats_arg, rho: (
            torch.zeros(2, device=stats_arg.C1.device, dtype=stats_arg.C1.dtype),
            torch.tensor(0.0, device=stats_arg.C1.device, dtype=stats_arg.C1.dtype),
        ),
    )
    assert math.isinf(native._tt_M_objective(0.25, stats))


def test_tt_profile_and_param_helpers_cover_invalid_paths(monkeypatch):
    obs = _obs(num_obs=48)
    stats = native._tt_selector_stats(obs, pilot_b=0.4)
    theta_bad = torch.tensor([0.01, 10.0], dtype=torch.float64)
    assert math.isinf(native._tt_profile_part1(z_std=stats.z_std, h=1.0, rho=0.0, theta=theta_bad))
    assert math.isinf(
        native._tt_profile_part2(obs=stats.obs, z_raw=stats.z_raw, h=1.0, rho=0.0, theta=theta_bad)
    )

    params, cfg = native._tt_params_from_spec(
        obs,
        bandwidth="auto",
        mult=0.9,
        selector_kind="pi",
        selector_grid_size=5,
        selector_num_refine=1,
        selector_sample_cap=32,
    )
    assert cfg["bandwidth"] == "auto"
    assert params.shape == (4,)

    with pytest.raises(ValueError, match="only support bandwidth='auto'"):
        native._tt_params_from_spec(
            obs,
            bandwidth="bad",
            selector_kind="cv",
            selector_grid_size=5,
            selector_num_refine=1,
            selector_sample_cap=32,
        )

    monkeypatch.setattr(native, "_tt_profile_part1", lambda **kwargs: float("inf"))
    params_cv, cfg_cv = native._tt_cv_selector(
        obs,
        selector_grid_size=5,
        selector_num_refine=1,
        selector_sample_cap=24,
    )
    assert cfg_cv["bandwidth_kind"] == "auto_cv"
    assert params_cv.shape == (4,)


def test_tt_cv_selector_handles_nonfinite_part2(monkeypatch):
    obs = _obs(num_obs=40)
    monkeypatch.setattr(native, "_tt_profile_part2", lambda **kwargs: float("inf"))
    params_cv, cfg_cv = native._tt_cv_selector(
        obs,
        selector_grid_size=5,
        selector_num_refine=1,
        selector_sample_cap=24,
    )
    assert cfg_cv["bandwidth_kind"] == "auto_cv"
    assert params_cv.shape == (4,)


def test_beta_special_functions_and_qt_scalar_summary_path(monkeypatch):
    x = torch.tensor([0.95], dtype=torch.float64)
    inc = native._regularized_beta_inc(2.5, 3.0, x)
    assert 0.0 < float(inc.item()) < 1.0

    obs = _obs(num_obs=64)

    def _fake_bandwidth_matrix_from_spec(obs_arg, *, bandwidth, bandwidth_scale):
        return (
            torch.eye(2, dtype=obs_arg.dtype, device=obs_arg.device),
            torch.tensor([0.04], dtype=obs_arg.dtype, device=obs_arg.device),
            {"bandwidth_kind": "scalar_stub"},
        )

    monkeypatch.setattr(native, "_bandwidth_matrix_from_spec", _fake_bandwidth_matrix_from_spec)
    _, _, _, _, summary, cfg = native.fit_beta_qt_bicop(
        obs,
        num_step_grid=17,
        bandwidth=0.08,
        bandwidth_scale=1.0,
        smoother="beta",
        marginal_tol=1e-3,
        num_iter_max=5,
    )
    assert summary.numel() == 1
    assert cfg["bandwidth_kind"] == "scalar_stub"
