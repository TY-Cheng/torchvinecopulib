import importlib.util

import matplotlib.pyplot as plt
import pytest
import torch

import torchvinecopulib as tvc
import torchvinecopulib.bicop as bicop_mod

from . import DEVICE, DTYPE, EPS, gaussian_copula


HAS_REFERENCE = importlib.util.find_spec("pyvinecopulib") is not None


def _fit_bicop(bicop_backend: str = "grid_reflect") -> tuple[torch.Tensor, tvc.BiCop]:
    U = gaussian_copula(num_obs=1500, rho=0.65).to(DEVICE)
    cop = tvc.BiCop(num_step_grid=129).to(DEVICE)
    cop.fit(U, bicop_backend=bicop_backend, bicop_kwargs={"bandwidth": "silverman"})
    return U, cop


def test_device_and_dtype():
    cop = tvc.BiCop(num_step_grid=16)
    assert cop.device.type == "cpu"
    assert cop.dtype is torch.float64
    if torch.cuda.is_available():
        cop = tvc.BiCop(num_step_grid=16).cuda()
        assert cop.device.type == "cuda"


@pytest.mark.parametrize("backend_name", ["grid_reflect", "grid_probit"])
def test_grid_backends_monotonicity_and_range(backend_name):
    U, cop = _fit_bicop(backend_name)
    grid = torch.linspace(0.05, 0.95, 100, device=U.device, dtype=DTYPE).unsqueeze(1)
    pts = torch.hstack([grid, grid])
    cdf = cop.cdf(pts)
    assert cdf.min() >= -EPS and cdf.max() <= 1.0 + EPS
    assert cdf.squeeze(1).diff().min() >= -1e-5
    for fn in (cop.hfunc_l, cop.hfunc_r, cop.hinv_l, cop.hinv_r):
        vals = fn(pts)
        assert vals.min() >= -EPS and vals.max() <= 1.0 + EPS


@pytest.mark.parametrize("backend_name", ["grid_reflect", "grid_probit"])
def test_inversion_grid_backends(backend_name):
    _, cop = _fit_bicop(backend_name)
    grid = torch.linspace(0.1, 0.9, 30, device=cop.device, dtype=DTYPE).unsqueeze(1)
    pts = torch.hstack([grid, grid.flip(0)])
    rec_r = cop.hinv_r(torch.hstack([cop.hfunc_r(pts), pts[:, [1]]]))
    rec_l = cop.hinv_l(torch.hstack([pts[:, [0]], cop.hfunc_l(pts)]))
    assert torch.allclose(rec_r, pts[:, [0]], atol=3e-2)
    assert torch.allclose(rec_l, pts[:, [1]], atol=3e-2)


def test_pdf_integrates_to_one():
    _, cop = _fit_bicop("grid_reflect")
    delta = 1.0 / (cop.num_step_grid - 1)
    approx_mass = (cop._pdf_grid.sum() * delta**2).item()
    assert pytest.approx(1.0, rel=2e-2) == approx_mass
    assert torch.isfinite(cop._pdf_grid).all()
    assert (cop._pdf_grid >= 0.0).all()


def test_sample_shape_dtype_and_uniform_marginals():
    _, cop = _fit_bicop("grid_reflect")
    for is_sobol in (False, True):
        samp = cop.sample(1500, seed=7, is_sobol=is_sobol)
        assert samp.shape == (1500, 2)
        assert samp.dtype is cop.dtype
        assert samp.device == cop.device
        assert samp.min() >= 0.0 and samp.max() <= 1.0
        counts = torch.histc(samp[:, 0].cpu(), bins=10, min=0.0, max=1.0)
        assert counts.std().item() < 35


def test_independent_copula_properties():
    cop = tvc.BiCop(num_step_grid=64).to(DEVICE)
    us = torch.rand(500, 2, dtype=DTYPE, device=DEVICE)
    assert torch.allclose(cop.cdf(us), (us[:, 0] * us[:, 1]).unsqueeze(1))
    assert torch.allclose(cop.pdf(us), torch.ones_like(us[:, [0]]))
    assert torch.allclose(cop.log_pdf(us), torch.zeros_like(us[:, [0]]))
    assert torch.allclose(cop.hfunc_r(us), us[:, [0]])
    assert torch.allclose(cop.hfunc_l(us), us[:, [1]])


def test_boundary_invariants_are_exact_on_grid():
    _, cop = _fit_bicop("grid_reflect")
    pts = torch.tensor(
        [[0.0, 0.0], [0.0, 0.3], [0.3, 0.0], [1.0, 1.0]],
        dtype=DTYPE,
        device=cop.device,
    )
    assert torch.allclose(cop.cdf(pts[:3]), torch.zeros(3, 1, dtype=DTYPE, device=cop.device), atol=1e-10)
    assert torch.allclose(cop.hfunc_l(pts[[0, 2]]), torch.zeros(2, 1, dtype=DTYPE, device=cop.device), atol=1e-10)
    assert torch.allclose(cop.hfunc_r(pts[[0, 1]]), torch.zeros(2, 1, dtype=DTYPE, device=cop.device), atol=1e-10)
    assert torch.allclose(cop.cdf(pts[[3]]), torch.ones(1, 1, dtype=DTYPE, device=cop.device), atol=1e-10)
    assert torch.allclose(cop.hfunc_l(pts[[3]]), torch.ones(1, 1, dtype=DTYPE, device=cop.device), atol=1e-10)
    assert torch.allclose(cop.hfunc_r(pts[[3]]), torch.ones(1, 1, dtype=DTYPE, device=cop.device), atol=1e-10)


@pytest.mark.parametrize("fn_name", ["cdf", "hfunc_l", "hfunc_r", "log_pdf"])
def test_boundary_policy_st_preserves_query_gradient(fn_name):
    obs_fit = gaussian_copula(num_obs=1500, rho=0.55).to(DEVICE)
    hard = tvc.BiCop(num_step_grid=65, boundary_policy="hard").to(DEVICE)
    soft = tvc.BiCop(num_step_grid=65, boundary_policy="st").to(DEVICE)
    for cop in (hard, soft):
        cop.fit(obs_fit, bicop_backend="grid_reflect")
    hard_obs = torch.tensor([[-1e-3, 0.4]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    soft_obs = torch.tensor([[-1e-3, 0.4]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    getattr(hard, fn_name)(hard_obs).sum().backward()
    getattr(soft, fn_name)(soft_obs).sum().backward()
    assert hard_obs.grad is not None and soft_obs.grad is not None
    assert hard_obs.grad[0, 0].abs().item() == pytest.approx(0.0, abs=1e-12)
    assert soft_obs.grad[0, 0].abs().item() > 0.0


def test_mtd_kde_alias_and_deprecation():
    U = gaussian_copula(num_obs=500, rho=0.5).to(DEVICE)
    cop = tvc.BiCop(num_step_grid=65).to(DEVICE)
    with pytest.deprecated_call():
        cop.fit(U, mtd_kde="fastKDE")
    assert cop.bicop_backend == "grid_reflect"


def test_legacy_fit_conflicts_warn_and_tau_estimates():
    obs = gaussian_copula(num_obs=500, rho=0.5).to(DEVICE)
    cop = tvc.BiCop(num_step_grid=65).to(DEVICE)
    with pytest.deprecated_call():
        cop.fit(
            obs,
            bicop_kwargs={"bandwidth": "silverman", "bandwidth_scale": 0.9, "num_iter_max": 3},
            bandwidth=0.2,
            bandwidth_scale=1.1,
            num_iter_max=4,
            is_tau_est=True,
        )
    assert bool(torch.isfinite(cop.tau).all())


def test_boundary_policy_survives_reset_and_state_roundtrip():
    obs, cop = _fit_bicop("grid_reflect")
    del obs
    st = tvc.BiCop(num_step_grid=65, boundary_policy="st").to(DEVICE)
    st.fit(gaussian_copula(num_obs=500, rho=0.45).to(DEVICE), bicop_backend="grid_reflect")
    st.reset()
    assert st.boundary_policy == "st"
    st.fit(gaussian_copula(num_obs=500, rho=0.45).to(DEVICE), bicop_backend="grid_reflect")
    fresh = tvc.BiCop(num_step_grid=65, boundary_policy="hard").to(DEVICE)
    fresh.load_state_dict(st.state_dict())
    assert fresh.boundary_policy == "st"
    pts = torch.rand(32, 2, dtype=DTYPE, device=DEVICE)
    assert torch.allclose(fresh.pdf(pts), st.pdf(pts), atol=1e-6)


def test_legacy_state_without_extra_state_uses_defaults():
    obs = gaussian_copula(num_obs=300, rho=0.5).to(DEVICE)
    cop = tvc.BiCop(num_step_grid=17).to(DEVICE)
    cop.fit(obs, bicop_backend="grid_reflect")
    legacy = {k: v for k, v in cop.state_dict().items() if k != "_extra_state"}
    fresh = tvc.BiCop(num_step_grid=17).to(DEVICE)
    fresh.load_state_dict(legacy)
    assert fresh.bicop_backend == "grid_reflect"
    assert fresh.boundary_policy == "hard"
    assert torch.allclose(fresh.pdf(obs[:16]), cop.pdf(obs[:16]), atol=1e-6)


def test_blank_backend_name_restores_grid_reflect_default():
    obs = gaussian_copula(num_obs=300, rho=0.5).to(DEVICE)
    cop = tvc.BiCop(num_step_grid=17).to(DEVICE)
    cop.fit(obs, bicop_backend="grid_reflect")
    state = cop.state_dict()
    state["_extra_state"]["backend_name"] = ""
    fresh = tvc.BiCop(num_step_grid=17).to(DEVICE)
    fresh.bicop_backend = ""
    fresh.kde_backend = ""
    fresh.mtd_kde = ""
    fresh.load_state_dict(state)
    assert fresh.bicop_backend == "grid_reflect"
    assert fresh.kde_backend == "grid_reflect"
    assert fresh.mtd_kde == "grid_reflect"


def test_hinv_fallback_status_is_observable(fitted_bicop, monkeypatch):
    def fake_solve_itp(fun, x_a, x_b, **kwargs):
        status = {
            "bracketed": torch.zeros_like(x_a, dtype=torch.bool),
            "converged": torch.zeros_like(x_a, dtype=torch.bool),
            "used_fallback": torch.ones_like(x_a, dtype=torch.bool),
            "iterations": torch.zeros_like(x_a, dtype=torch.int),
        }
        return torch.full_like(x_a, 0.9), status

    monkeypatch.setattr(bicop_mod, "solve_ITP", fake_solve_itp)
    query = torch.tensor([[0.4, 0.2], [0.6, 0.8]], dtype=DTYPE, device=DEVICE)
    out_l = fitted_bicop.hinv_l(query)
    out_r = fitted_bicop.hinv_r(query)
    assert torch.isfinite(out_l).all() and torch.isfinite(out_r).all()
    assert bool(fitted_bicop.last_hinv_status_l["used_fallback"].all())
    assert bool(fitted_bicop.last_hinv_status_r["used_fallback"].all())
    assert int(fitted_bicop.hinv_fallback_l) == query.shape[0]
    assert int(fitted_bicop.hinv_fallback_r) == query.shape[0]


def test_sample_falls_back_to_independence_without_point_mass(fitted_bicop, monkeypatch):
    def fake_solve_itp(fun, x_a, x_b, **kwargs):
        status = {
            "bracketed": torch.zeros_like(x_a, dtype=torch.bool),
            "converged": torch.zeros_like(x_a, dtype=torch.bool),
            "used_fallback": torch.ones_like(x_a, dtype=torch.bool),
            "iterations": torch.zeros_like(x_a, dtype=torch.int),
        }
        return torch.zeros_like(x_a), status

    monkeypatch.setattr(bicop_mod, "solve_ITP", fake_solve_itp)
    monkeypatch.setattr(fitted_bicop, "_bisect_hinv", lambda fixed, target, mode: torch.zeros_like(target))
    samp = fitted_bicop.sample(32, seed=11)
    assert torch.isfinite(samp).all()
    assert samp.min() >= 0.0 and samp.max() <= 1.0
    assert not torch.all(samp[:, 1] == 0.5)
    diag = fitted_bicop.diagnostics()
    assert diag.fallback_to_indep_l == samp.shape[0]
    assert diag.itp_failures_l == samp.shape[0]


def test_negloglik_is_finite_after_fit(strong_dep_copula_obs):
    cop = tvc.BiCop(num_step_grid=65).to(DEVICE)
    cop.fit(
        strong_dep_copula_obs,
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
    )
    assert cop.negloglik.ndim == 0
    assert torch.isfinite(cop.negloglik)


def test_lazy_grid_allocation_and_str():
    cop = tvc.BiCop(num_step_grid=64)
    assert cop._pdf_grid.numel() == 0
    _, fitted = _fit_bicop("grid_reflect")
    assert fitted._pdf_grid.shape == (129, 129)
    assert "bicop_backend" in str(fitted)


def test_imshow_and_plot_api():
    _, cop = _fit_bicop("grid_reflect")
    fig, ax = cop.imshow(is_log_pdf=True)
    assert fig is not None and ax is not None
    plt.close(fig)
    fig2, ax2 = cop.plot(plot_type="contour", margin_type="unif")
    plt.close(fig2)
    fig3, ax3 = cop.plot(plot_type="surface", margin_type="norm")
    plt.close(fig3)


def test_plot_validation_and_independent_constant_density_branch():
    cop = tvc.BiCop(num_step_grid=33).to(DEVICE)
    with pytest.raises(ValueError):
        cop.plot(plot_type="invalid")
    with pytest.raises(ValueError):
        cop.plot(plot_type="surface", margin_type="invalid")
    fig, ax = cop.plot(plot_type="contour", margin_type="unif")
    plt.close(fig)


def test_legacy_backend_alias_and_deprecation():
    U = gaussian_copula(num_obs=500, rho=0.5).to(DEVICE)
    cop = tvc.BiCop(num_step_grid=65).to(DEVICE)
    with pytest.deprecated_call():
        cop.fit(U, kde_backend="torch_grid")
    assert cop.bicop_backend == "grid_reflect"


def test_invalid_backend_kwargs_raise():
    U = gaussian_copula(num_obs=500, rho=0.5).to(DEVICE)
    cop = tvc.BiCop(num_step_grid=65).to(DEVICE)
    with pytest.raises(ValueError):
        cop.fit(U, bicop_backend="grid_reflect", bicop_kwargs={"unknown": 1})


@pytest.mark.reference
@pytest.mark.skipif(not HAS_REFERENCE, reason="pyvinecopulib reference backend not installed")
def test_reference_backend_smoke():
    U = gaussian_copula(num_obs=1000, rho=0.5).to(DEVICE)
    cop = tvc.BiCop(num_step_grid=65).to(DEVICE)
    cop.fit(U, bicop_backend="tll_ref")
    pts = torch.rand(32, 2, dtype=DTYPE, device=DEVICE)
    assert torch.isfinite(cop.log_pdf(pts)).all()
