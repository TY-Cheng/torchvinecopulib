import importlib.util

import matplotlib.pyplot as plt
import pytest
import torch

import torchvinecopulib as tvc

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


@pytest.mark.skipif(not HAS_REFERENCE, reason="pyvinecopulib reference backend not installed")
def test_reference_backend_smoke():
    U = gaussian_copula(num_obs=1000, rho=0.5).to(DEVICE)
    cop = tvc.BiCop(num_step_grid=65).to(DEVICE)
    cop.fit(U, bicop_backend="tll_ref")
    pts = torch.rand(32, 2, dtype=DTYPE, device=DEVICE)
    assert torch.isfinite(cop.log_pdf(pts)).all()
