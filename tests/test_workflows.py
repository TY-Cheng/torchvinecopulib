import pytest
import torch

import torchvinecopulib as tvc

from . import DEVICE, DTYPE


@pytest.mark.integration
def test_raw_scale_workflow_roundtrip(fitted_vine_raw):
    obs, vc = fitted_vine_raw
    fresh = tvc.VineCop(num_dim=5, is_cop_scale=False, num_step_grid=65).to(DEVICE)
    fresh.load_state_dict(vc.state_dict())
    assert fresh.engine.artifact is not None
    assert torch.allclose(fresh.engine.log_pdf(obs[:16]), vc.log_pdf(obs[:16]), atol=1e-6)
    samp = fresh.engine.sample(num_sample=32, seed=7)
    cdf = fresh.engine.cdf(obs[:8], num_sample=255, seed=3)
    assert samp.shape == (32, 5)
    assert cdf.shape == (8, 1)
    assert torch.isfinite(samp).all()
    assert torch.isfinite(cdf).all()


@pytest.mark.integration
def test_copula_scale_artifact_engine_workflow(fitted_vine_copula):
    obs, vc = fitted_vine_copula
    artifact = vc.engine.artifact
    clone = tvc.VineCop.from_artifact(artifact).to(DEVICE)
    assert torch.allclose(clone.engine.log_pdf(obs[:16]), vc.engine.log_pdf(obs[:16]), atol=1e-6)
    assert torch.allclose(clone.engine(obs[:16]), clone.forward(obs[:16]), atol=1e-6)
    assert torch.allclose(clone.engine._sample_u(8, seed=7), clone._sample_u(8, seed=7), atol=1e-6)


@pytest.mark.integration
def test_vinecop_boundary_policy_gradient_integration(copula_scale_obs):
    hard = tvc.VineCop(num_dim=5, is_cop_scale=True, num_step_grid=65, boundary_policy="hard").to(DEVICE)
    soft = tvc.VineCop(num_dim=5, is_cop_scale=True, num_step_grid=65, boundary_policy="st").to(DEVICE)
    for vc in (hard, soft):
        vc.fit(
            copula_scale_obs,
            bicop_backend="grid_reflect",
            bicop_kwargs={"bandwidth": "silverman"},
            mtd_bidep="kendall_tau",
            thresh_trunc=None,
        )
    hard_obs = copula_scale_obs[:8].clone()
    soft_obs = copula_scale_obs[:8].clone()
    hard_obs[0, 0] = -1e-3
    soft_obs[0, 0] = -1e-3
    hard_obs.requires_grad_(True)
    soft_obs.requires_grad_(True)
    hard.log_pdf(hard_obs).sum().backward()
    soft.log_pdf(soft_obs).sum().backward()
    assert hard_obs.grad is not None and soft_obs.grad is not None
    assert hard_obs.grad[0, 0].abs().item() == pytest.approx(0.0, abs=1e-12)
    assert soft_obs.grad[0, 0].abs().item() > 0.0
