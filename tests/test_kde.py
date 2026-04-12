import pytest
import torch

from torchvinecopulib import GridKDE1D, GridReflectBicopEstimator
from torchvinecopulib.backends import GridProbitBicopEstimator

from . import DEVICE, DTYPE, gaussian_copula


def test_torch_copula_kde_integral_and_shapes():
    obs = gaussian_copula(num_obs=2000, rho=0.7).to(DEVICE)
    kde = GridReflectBicopEstimator(obs=obs, num_step_grid=129)
    delta = 1.0 / (kde.num_step_grid - 1)
    assert pytest.approx(1.0, rel=2e-2) == (kde._pdf_grid.sum() * delta**2).item()
    assert kde._cdf_grid.shape == kde._pdf_grid.shape
    assert kde._hfunc_l_grid.shape == kde._pdf_grid.shape
    assert kde._hfunc_r_grid.shape == kde._pdf_grid.shape


def test_torch_copula_kde_boundary_mass_is_finite():
    edge = torch.rand(1024, 2, dtype=DTYPE, device=DEVICE)
    edge[:, 0] = edge[:, 0] * 0.05 + 0.9
    edge[:, 1] = edge[:, 1] * 0.05 + 0.02
    kde = GridReflectBicopEstimator(obs=edge, num_step_grid=65)
    assert torch.isfinite(kde._pdf_grid).all()
    assert kde._pdf_grid[0].mean().item() > 0.0
    assert kde._pdf_grid[-1].mean().item() > 0.0


def test_grid_probit_boundary_mass_is_finite():
    edge = torch.rand(1024, 2, dtype=DTYPE, device=DEVICE)
    edge[:, 0] = edge[:, 0] * 0.02 + 0.98
    edge[:, 1] = edge[:, 1] * 0.02 + 0.01
    kde = GridProbitBicopEstimator(obs=edge, num_step_grid=65)
    assert torch.isfinite(kde._pdf_grid).all()
    assert (kde._pdf_grid >= 0.0).all()


def test_compile_friendly_query_paths():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")
    x = torch.randn(512, 1, dtype=DTYPE, device=DEVICE)
    kde = GridKDE1D(x, num_step_grid=129).to(DEVICE)
    query = torch.linspace(kde.x_min, kde.x_max, 64, dtype=DTYPE, device=DEVICE).view(-1, 1)
    try:
        compiled = torch.compile(kde.pdf, backend="eager")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch.compile unavailable: {exc}")
    out = compiled(query)
    assert out.shape == (64, 1)
    assert torch.isfinite(out).all()


def test_recursive_smoother_smoke():
    x = torch.randn(256, 1, dtype=DTYPE, device=DEVICE)
    kde = GridKDE1D(x, num_step_grid=129, smoother="recursive").to(DEVICE)
    vals = kde.pdf(x[:16])
    assert torch.isfinite(vals).all()
