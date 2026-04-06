import sys

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


def test_bicop_state_dict_backward_compatibility():
    obs = torch.rand(300, 2, dtype=DTYPE, device=DEVICE)
    cop = tvc.BiCop(num_step_grid=17).to(DEVICE)
    cop.fit(obs, bicop_backend="grid_reflect")
    state = cop.state_dict()
    legacy = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in state.items()
        if k != "_extra_state"
    }
    fresh = tvc.BiCop(num_step_grid=17).to(DEVICE)
    fresh.load_state_dict(legacy)
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
