import pytest
import torch

import torchvinecopulib as tvc

from . import DEVICE, correlated_raw, gaussian_copula


def test_bicop_log_pdf_gradient_wrt_obs():
    obs_fit = gaussian_copula(num_obs=1000, rho=0.5).to(DEVICE)
    obs = gaussian_copula(num_obs=32, rho=0.3).to(DEVICE).requires_grad_(True)
    cop = tvc.BiCop(num_step_grid=65).to(DEVICE)
    cop.fit(obs_fit, bicop_backend="grid_reflect")
    loss = cop.log_pdf(obs).sum()
    loss.backward()
    assert obs.grad is not None
    assert torch.isfinite(obs.grad).all()


def test_vinecop_log_pdf_gradient_wrt_obs():
    obs_fit = correlated_raw(256, 0.55, 4).to(DEVICE)
    obs = correlated_raw(16, 0.3, 4).to(DEVICE).requires_grad_(True)
    vc = tvc.VineCop(num_dim=4, is_cop_scale=False, num_step_grid=33).to(DEVICE)
    vc.fit(
        obs_fit,
        marginal_kwargs={"bandwidth": "isj"},
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        thresh_trunc=None,
    )
    loss = vc.log_pdf(obs).sum()
    loss.backward()
    assert obs.grad is not None
    assert torch.isfinite(obs.grad).all()


@pytest.mark.parametrize("fn_name", ["cdf", "hfunc_l", "hfunc_r"])
def test_bicop_query_paths_backward_do_not_graph_break(fn_name):
    obs_fit = gaussian_copula(num_obs=512, rho=0.55).to(DEVICE)
    obs = gaussian_copula(num_obs=32, rho=0.25).to(DEVICE).requires_grad_(True)
    cop = tvc.BiCop(num_step_grid=65).to(DEVICE)
    cop.fit(obs_fit, bicop_backend="grid_reflect")
    out = getattr(cop, fn_name)(obs).sum()
    out.backward()
    assert obs.grad is not None
    assert torch.isfinite(obs.grad).all()
