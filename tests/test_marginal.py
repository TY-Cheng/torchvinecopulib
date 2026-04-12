import pytest
import torch

import torchvinecopulib.backends.marginal as marginal_mod
from torchvinecopulib.backends.marginal import (
    GridKDE1D,
    LocalPolynomialKDE1D,
    build_marginal_estimator,
    build_marginal_shell,
)

from . import DEVICE, DTYPE


def _backend_kwargs_for(name: str) -> dict:
    if name == "grid":
        return {"bandwidth": "silverman", "num_step_grid": 65}
    return {"num_step_grid": 65, "degree": 1}


@pytest.mark.parametrize("backend_name", ["grid", "lp"])
def test_marginal_support_semantics(raw_scale_obs, backend_name):
    x = raw_scale_obs[:, 0]
    marginal = build_marginal_estimator(
        backend_name=backend_name,
        x=x,
        backend_kwargs=_backend_kwargs_for(backend_name),
    ).to(DEVICE)
    below = torch.full((4, 1), marginal.x_min - 1.0, dtype=DTYPE, device=DEVICE)
    above = torch.full((4, 1), marginal.x_max + 1.0, dtype=DTYPE, device=DEVICE)
    assert torch.allclose(marginal.cdf(below), torch.zeros_like(below))
    assert torch.allclose(marginal.cdf(above), torch.ones_like(above))
    assert torch.allclose(marginal.pdf(below), torch.zeros_like(below))
    assert torch.allclose(marginal.pdf(above), torch.zeros_like(above))


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


def test_lp_torch_from_result_and_shell_roundtrip(raw_scale_obs):
    x = raw_scale_obs[:, [0]]
    fitted = LocalPolynomialKDE1D(x=x, num_step_grid=65, degree=2).to(DEVICE)
    result = LocalPolynomialKDE1D.fit_reference(
        x=x,
        num_step_grid=65,
        x_min=None,
        x_max=None,
        degree=2,
        bandwidth="plugin",
        bandwidth_scale=1.0,
    )
    clone = LocalPolynomialKDE1D.from_result(result).to(DEVICE)
    shell = build_marginal_shell(backend_name="lp").to(DEVICE)
    shell.load_state_dict(fitted.state_dict())
    pts = x[:16]
    assert LocalPolynomialKDE1D.empty().num_step_grid == 0
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
def test_gridkde_bandwidth_variants_and_recursive_smoother(
    raw_scale_obs, bandwidth, expected_method
):
    x = raw_scale_obs[:, [0]]
    kde = GridKDE1D(x=x, num_step_grid=65, bandwidth=bandwidth, smoother="recursive").to(DEVICE)
    assert torch.isfinite(kde.log_pdf(x[:32])).all()
    assert kde.bandwidth_method == expected_method


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_lp_torch_supports_bounded_continuous_paths(raw_scale_obs, degree):
    x = raw_scale_obs[:, [0]].abs() + 0.05
    kde = LocalPolynomialKDE1D(
        x=x,
        num_step_grid=129,
        x_min=0.0,
        degree=degree,
        bandwidth="plugin",
    ).to(DEVICE)
    xs = torch.linspace(0.05, 2.0, 32, dtype=DTYPE, device=DEVICE).view(-1, 1)
    qs = kde.cdf(xs)
    xs_rec = kde.ppf(qs)
    assert torch.isfinite(kde.log_pdf(xs)).all()
    assert torch.allclose(xs, xs_rec, atol=1e-1)


def test_private_boundary_helpers_cover_supported_boundary_types():
    x = torch.linspace(0.1, 0.9, 9, dtype=DTYPE)
    assert marginal_mod._lp_boundary_kind(None, None) == "none"
    assert marginal_mod._lp_boundary_kind(0.0, None) == "left"
    assert marginal_mod._lp_boundary_kind(None, 1.0) == "right"
    assert marginal_mod._lp_boundary_kind(0.0, 1.0) == "two"

    z_two = marginal_mod._lp_boundary_transform(x, x_min=0.0, x_max=1.0)
    x_two = marginal_mod._lp_boundary_transform(z_two, x_min=0.0, x_max=1.0, inverse=True)
    assert torch.allclose(x, x_two, atol=2e-4)
    corr_two = marginal_mod._lp_boundary_correction(x, x_min=0.0, x_max=1.0)
    assert torch.isfinite(corr_two).all()

    z_right = marginal_mod._lp_boundary_transform(x, x_min=None, x_max=1.0)
    x_right = marginal_mod._lp_boundary_transform(z_right, x_min=None, x_max=1.0, inverse=True)
    assert torch.allclose(x, x_right, atol=2e-4)
    corr_right = marginal_mod._lp_boundary_correction(x, x_min=None, x_max=1.0)
    assert torch.isfinite(corr_right).all()


def test_private_support_helpers_cover_edge_cases():
    with pytest.raises(ValueError):
        marginal_mod._support_bounds(
            torch.tensor([0.0], dtype=DTYPE),
            x_min=1.0,
            x_max=1.0,
            pad=0.1,
        )


def test_lp_bandwidth_helper_variants_cover_custom_rules(raw_scale_obs):
    x = raw_scale_obs[:, 0]
    h_custom, method_custom = marginal_mod._select_lp_bandwidth(
        x, degree=1, bandwidth=0.2, bandwidth_scale=1.0
    )
    h_silverman, method_silverman = marginal_mod._select_lp_bandwidth(
        x, degree=1, bandwidth="silverman", bandwidth_scale=1.0
    )
    h_isj, method_isj = marginal_mod._select_lp_bandwidth(
        x, degree=1, bandwidth="isj", bandwidth_scale=1.0
    )
    h_small, method_small = marginal_mod._select_lp_bandwidth(
        x[:8], degree=1, bandwidth="plugin", bandwidth_scale=1.0
    )
    h_bad_degree, method_bad_degree = marginal_mod._select_lp_bandwidth(
        x, degree=3, bandwidth="plugin", bandwidth_scale=1.0
    )
    assert h_custom.item() > 0.0 and method_custom == "custom"
    assert h_silverman.item() > 0.0 and method_silverman == "silverman"
    assert h_isj.item() > 0.0 and method_isj in {"isj", "silverman"}
    assert h_small.item() > 0.0 and method_small == "plugin_fallback"
    assert h_bad_degree.item() > 0.0 and method_bad_degree == "plugin_fallback"
    with pytest.raises(ValueError):
        marginal_mod._select_lp_bandwidth(
            x, degree=1, bandwidth="unsupported", bandwidth_scale=1.0
        )


def test_lp_covers_explicit_support_and_default_grid_paths(raw_scale_obs):
    x = raw_scale_obs[:, [0]]
    lp_default = LocalPolynomialKDE1D(x=x[:32]).to(DEVICE)
    assert lp_default.num_step_grid == 401
    assert torch.isfinite(lp_default.log_pdf(x[:16])).all()

    lp_supported = LocalPolynomialKDE1D.fit_reference(
        x=x,
        num_step_grid=65,
        x_min=-4.0,
        x_max=4.0,
        degree=1,
        bandwidth="plugin",
        bandwidth_scale=1.0,
    )
    assert lp_supported.backend_config["support_min"] == -4.0
    assert lp_supported.backend_config["support_max"] == 4.0
    assert torch.isfinite(lp_supported.grid_pdf).all()


def test_lp_right_boundary_and_fit_updates_cover_reverse_path(raw_scale_obs):
    x = raw_scale_obs[:, [0]].abs() + 0.05
    result = LocalPolynomialKDE1D.fit_reference(
        x=x,
        num_step_grid=65,
        x_min=None,
        x_max=2.5,
        degree=1,
        bandwidth="plugin",
        bandwidth_scale=1.0,
    )
    assert (result.grid_x[1:] - result.grid_x[:-1]).min().item() > -1e-4
    assert result.grid_x[-1].item() == pytest.approx(2.5, abs=1e-8)

    kde = LocalPolynomialKDE1D(x=x, num_step_grid=33, degree=1).to(DEVICE)
    kde.fit(x=x, num_step_grid=65, x_max=2.5, degree=1)
    assert kde.grid_x.shape[0] == 65


def test_gridkde_invalid_smoother_raises(raw_scale_obs):
    x = raw_scale_obs[:, [0]]
    with pytest.raises(ValueError):
        GridKDE1D(x=x, num_step_grid=65, smoother="unknown")


def test_gridkde_invalid_bandwidth_raises(raw_scale_obs):
    x = raw_scale_obs[:, [0]]
    with pytest.raises(ValueError):
        GridKDE1D(x=x, num_step_grid=65, bandwidth="unsupported")
