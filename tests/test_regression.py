import sys

import pytest
import torch

import torchvinecopulib as tvc

from . import DEVICE, DTYPE, correlated_raw


def test_sample_does_not_pollute_global_rng():
    obs = correlated_raw(200, 0.4, 3).to(DEVICE)
    vc = tvc.VineCop(num_dim=3, is_cop_scale=False, num_step_grid=33).to(DEVICE)
    vc.fit(
        obs,
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        marginal_kwargs={"bandwidth": "isj"},
        thresh_trunc=None,
    )
    torch.manual_seed(1234)
    before = torch.rand(5, dtype=DTYPE)
    torch.manual_seed(1234)
    vc.sample(10, seed=7)
    after = torch.rand(5, dtype=DTYPE)
    assert torch.allclose(before, after)


def test_vinecop_cdf_uses_copula_scale_consistently():
    obs = correlated_raw(200, 0.6, 4).to(DEVICE)
    vc = tvc.VineCop(num_dim=4, is_cop_scale=False, num_step_grid=33).to(DEVICE)
    vc.fit(
        obs,
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        marginal_kwargs={"bandwidth": "isj"},
        thresh_trunc=None,
    )
    vals = vc.cdf(obs[:5], num_sample=255, seed=3)
    assert vals.shape == (5, 1)
    assert torch.isfinite(vals).all()
    assert (vals >= 0.0).all() and (vals <= 1.0).all()


def test_bicop_state_roundtrip_preserves_query_behavior():
    obs = torch.rand(300, 2, dtype=DTYPE, device=DEVICE)
    cop = tvc.BiCop(num_step_grid=17).to(DEVICE)
    cop.fit(obs, bicop_backend="grid_reflect")
    fresh = tvc.BiCop(num_step_grid=17).to(DEVICE)
    fresh.load_state_dict(cop.state_dict())
    pts = torch.rand(32, 2, dtype=DTYPE, device=DEVICE)
    assert torch.allclose(cop.pdf(pts), fresh.pdf(pts), atol=1e-6)


def test_lazy_optional_imports_do_not_break_runtime():
    sys.modules.pop("pyvinecopulib", None)
    obs = torch.rand(256, 2, dtype=DTYPE, device=DEVICE)
    cop = tvc.BiCop(num_step_grid=33).to(DEVICE)
    cop.fit(obs, bicop_backend="grid_reflect")
    assert torch.isfinite(cop.log_pdf(obs[:8])).all()
    assert "pyvinecopulib" not in sys.modules


def test_vinecop_state_roundtrip_preserves_backend_metadata():
    obs = correlated_raw(128, 0.5, 4).to(DEVICE)
    vc = tvc.VineCop(num_dim=4, is_cop_scale=False, num_step_grid=33).to(DEVICE)
    vc.fit(
        obs,
        marginal_backend="grid",
        marginal_kwargs={"bandwidth": "isj"},
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        thresh_trunc=None,
    )
    fresh = tvc.VineCop(num_dim=4, is_cop_scale=False, num_step_grid=33).to(DEVICE)
    fresh.load_state_dict(vc.state_dict())
    assert fresh.marginal_backend == "grid"
    assert fresh.bicop_backend == "grid_reflect"
    assert torch.allclose(vc.log_pdf(obs[:16]), fresh.log_pdf(obs[:16]), atol=1e-6)


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


def test_copula_scale_artifact_engine_workflow(fitted_vine_copula):
    obs, vc = fitted_vine_copula
    artifact = vc.engine.artifact
    clone = tvc.VineCop.from_artifact(artifact).to(DEVICE)
    assert torch.allclose(clone.engine.log_pdf(obs[:16]), vc.engine.log_pdf(obs[:16]), atol=1e-6)
    assert torch.allclose(clone.engine(obs[:16]), clone.forward(obs[:16]), atol=1e-6)
    assert torch.allclose(clone.engine._sample_u(8, seed=7), clone._sample_u(8, seed=7), atol=1e-6)


def test_vinecop_boundary_policy_gradient_integration(copula_scale_obs):
    hard = tvc.VineCop(num_dim=5, is_cop_scale=True, num_step_grid=65, boundary_policy="hard").to(
        DEVICE
    )
    soft = tvc.VineCop(num_dim=5, is_cop_scale=True, num_step_grid=65, boundary_policy="st").to(
        DEVICE
    )
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
