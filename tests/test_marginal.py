import numpy as np
import pytest
import torch

from torchvinecopulib.backends.marginal import (
    GridKDE1D,
    LPRefMarginal1D,
    build_marginal_estimator,
    build_marginal_shell,
)

from . import DEVICE, DTYPE


@pytest.mark.parametrize("backend_name", ["grid", "lp_ref"])
def test_marginal_support_semantics(raw_scale_obs, backend_name):
    x = raw_scale_obs[:, 0]
    backend_kwargs = {"bandwidth": "silverman", "num_step_grid": 65} if backend_name == "grid" else {"num_step_grid": 65}
    marginal = build_marginal_estimator(
        backend_name=backend_name,
        x=x,
        backend_kwargs=backend_kwargs,
    ).to(DEVICE)
    below = torch.full((4, 1), marginal.x_min - 1.0, dtype=DTYPE, device=DEVICE)
    above = torch.full((4, 1), marginal.x_max + 1.0, dtype=DTYPE, device=DEVICE)
    assert torch.allclose(marginal.cdf(below), torch.zeros_like(below))
    assert torch.allclose(marginal.cdf(above), torch.ones_like(above))
    assert torch.allclose(marginal.pdf(below), torch.zeros_like(below))
    assert torch.allclose(marginal.pdf(above), torch.zeros_like(above))
    assert torch.allclose(
        marginal.ppf(torch.full((3, 1), -0.2, dtype=DTYPE, device=DEVICE)),
        torch.full((3, 1), marginal.x_min, dtype=DTYPE, device=DEVICE),
    )
    assert torch.allclose(
        marginal.ppf(torch.full((3, 1), 1.2, dtype=DTYPE, device=DEVICE)),
        torch.full((3, 1), marginal.x_max, dtype=DTYPE, device=DEVICE),
    )


def test_gridkde_forward_matches_negative_log_pdf_mean(raw_scale_obs):
    x = raw_scale_obs[:, [0]]
    kde = GridKDE1D(x=x, num_step_grid=65, bandwidth="silverman").to(DEVICE)
    query = x[:32]
    assert torch.allclose(kde.forward(query), -kde.log_pdf(query).mean(), atol=1e-10)


def test_gridkde_from_result_and_shell_roundtrip(raw_scale_obs):
    x = raw_scale_obs[:, [0]]
    fitted = GridKDE1D(x=x, num_step_grid=65, bandwidth="silverman").to(DEVICE)
    result = GridKDE1D.fit_reference(
        x=x,
        num_step_grid=65,
        x_min=fitted.x_min,
        x_max=fitted.x_max,
        bandwidth="silverman",
        bandwidth_scale=1.0,
        smoother="auto",
    )
    clone = GridKDE1D.from_result(result).to(DEVICE)
    shell = build_marginal_shell(backend_name="grid").to(DEVICE)
    shell.load_state_dict(fitted.state_dict())
    pts = x[:16]
    assert GridKDE1D.empty().num_step_grid == 0
    assert torch.allclose(clone.pdf(pts), fitted.pdf(pts), atol=1e-6)
    assert torch.allclose(shell.cdf(pts), fitted.cdf(pts), atol=1e-6)


def test_lp_ref_from_result_and_shell_roundtrip(raw_scale_obs, monkeypatch):
    x = raw_scale_obs[:, [0]]

    class FakeGaussianKDE:
        covariance = np.array([[0.04]])

        def __init__(self, data, bw_method=None, weights=None):
            self._data = np.asarray(data)

        def __call__(self, grid):
            grid = np.asarray(grid)
            centered = grid - grid.mean()
            return np.exp(-0.5 * centered**2)

    import torchvinecopulib.backends.marginal as marginal_mod

    monkeypatch.setattr(marginal_mod, "_lazy_reference_kde", lambda: ("scipy", FakeGaussianKDE))
    fitted = LPRefMarginal1D(x=x, num_step_grid=65, reference_kwargs={"bw_method": 0.2}).to(DEVICE)
    result = LPRefMarginal1D.fit_reference(
        x=x,
        num_step_grid=65,
        x_min=fitted.x_min,
        x_max=fitted.x_max,
        reference_kwargs={"bw_method": 0.2},
    )
    clone = LPRefMarginal1D.from_result(result).to(DEVICE)
    shell = build_marginal_shell(backend_name="lp_ref").to(DEVICE)
    shell.load_state_dict(fitted.state_dict())
    pts = x[:16]
    assert LPRefMarginal1D.empty().num_step_grid == 0
    assert torch.allclose(clone.pdf(pts), fitted.pdf(pts), atol=1e-6)
    assert torch.allclose(shell.cdf(pts), fitted.cdf(pts), atol=1e-6)


def test_marginal_load_state_dict_resizes_buffers(raw_scale_obs):
    x = raw_scale_obs[:, [0]]
    source = GridKDE1D(x=x, num_step_grid=129, bandwidth="silverman").to(DEVICE)
    target = GridKDE1D(x=x, num_step_grid=33, bandwidth="silverman").to(DEVICE)
    target.load_state_dict(source.state_dict())
    assert target.grid_x.shape == source.grid_x.shape
    assert target.bandwidth.shape == source.bandwidth.shape
    assert torch.allclose(target.pdf(x[:16]), source.pdf(x[:16]), atol=1e-6)


@pytest.mark.parametrize(
    ("bandwidth", "expected_method"),
    [("silverman", "silverman"), (0.2, "custom")],
)
def test_gridkde_bandwidth_variants_and_recursive_smoother(raw_scale_obs, bandwidth, expected_method):
    x = raw_scale_obs[:, [0]]
    kde = GridKDE1D(x=x, num_step_grid=65, bandwidth=bandwidth, smoother="recursive").to(DEVICE)
    assert torch.isfinite(kde.log_pdf(x[:32])).all()
    assert kde.bandwidth_method == expected_method


def test_lp_ref_reference_kwargs_validation(monkeypatch, raw_scale_obs):
    x = raw_scale_obs[:, 0]

    class FakeGaussianKDE:
        covariance = np.array([[0.09]])

        def __init__(self, data, bw_method=None, weights=None):
            self._data = np.asarray(data)

        def __call__(self, grid):
            grid = np.asarray(grid)
            return np.ones_like(grid, dtype=np.float64)

    import torchvinecopulib.backends.marginal as marginal_mod

    monkeypatch.setattr(marginal_mod, "_lazy_reference_kde", lambda: ("scipy", FakeGaussianKDE))
    marginal = build_marginal_estimator(
        backend_name="lp_ref",
        x=x,
        backend_kwargs={"reference_kwargs": {"bw_method": 0.2}, "num_step_grid": 65},
    )
    assert marginal.backend_config["reference_impl"] == "scipy"
    with pytest.raises(ValueError):
        build_marginal_estimator(
            backend_name="lp_ref",
            x=x,
            backend_kwargs={"reference_kwargs": {"unknown": 1}, "num_step_grid": 65},
        )


def test_gridkde_invalid_smoother_raises(raw_scale_obs):
    x = raw_scale_obs[:, [0]]
    with pytest.raises(ValueError):
        GridKDE1D(x=x, num_step_grid=65, smoother="unknown")


def test_gridkde_invalid_bandwidth_raises(raw_scale_obs):
    x = raw_scale_obs[:, [0]]
    with pytest.raises(ValueError):
        GridKDE1D(x=x, num_step_grid=65, bandwidth="unsupported")


def test_lp_ref_pyvinecopulib_path_and_marginal_str(monkeypatch, raw_scale_obs):
    x = raw_scale_obs[:, [0]]

    class FakeKde1d:
        def __init__(self, data, **kwargs):
            self._data = np.asarray(data)

        def pdf(self, grid):
            grid = np.asarray(grid)
            return np.ones_like(grid, dtype=np.float64)

        def cdf(self, grid):
            grid = np.asarray(grid)
            if grid.size == 1:
                return np.array([1.0], dtype=np.float64)
            return np.linspace(0.0, 1.0, grid.size, dtype=np.float64)

    import torchvinecopulib.backends.marginal as marginal_mod

    monkeypatch.setattr(marginal_mod, "_lazy_reference_kde", lambda: ("pyvinecopulib", FakeKde1d))
    marginal = LPRefMarginal1D(x=x, num_step_grid=65).to(DEVICE)
    marginal.set_extra_state({})
    summary = str(marginal)
    assert marginal.backend_config["reference_impl"] == "pyvinecopulib"
    assert "backend_name" in summary and "num_step_grid" in summary
