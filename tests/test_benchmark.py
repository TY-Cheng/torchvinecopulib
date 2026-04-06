import pytest
import torch

import torchvinecopulib as tvc

from . import DEVICE, correlated_raw, gaussian_copula


@pytest.mark.parametrize("num_obs", [1_000, 10_000])
@pytest.mark.parametrize("num_dim", [10, 20])
@pytest.mark.parametrize("grid_size", [64, 128])
def test_benchmark_smoke(num_obs, num_dim, grid_size):
    obs = gaussian_copula(num_obs=num_obs, rho=0.35, dim=num_dim).to(DEVICE)
    model = tvc.VineCop(num_dim=num_dim, is_cop_scale=True, num_step_grid=grid_size + 1).to(DEVICE)
    model.fit(
        obs,
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        thresh_trunc=0.05,
        mtd_bidep="kendall_tau",
    )
    lp = model.log_pdf(obs[:64])
    samp = model.sample(num_sample=64, seed=9)
    cdf = model.cdf(obs[:8], num_sample=255, seed=5)
    assert lp.shape == (64, 1)
    assert samp.shape == (64, num_dim)
    assert cdf.shape == (8, 1)
    assert torch.isfinite(lp).all()


def test_raw_scale_benchmark_smoke():
    obs = correlated_raw(1_000, 0.45, 10).to(DEVICE)
    model = tvc.VineCop(num_dim=10, is_cop_scale=False, num_step_grid=65).to(DEVICE)
    model.fit(
        obs,
        marginal_kwargs={"bandwidth": "isj"},
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        thresh_trunc=0.05,
    )
    assert torch.isfinite(model.log_pdf(obs[:32])).all()
