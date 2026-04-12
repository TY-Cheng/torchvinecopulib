import pytest
import torch
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import torchvinecopulib as tvc

from . import DEVICE, DTYPE


UNIT_FLOATS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
INTERIOR_FLOATS = st.floats(
    min_value=1e-3, max_value=1 - 1e-3, allow_nan=False, allow_infinity=False
)


@pytest.mark.property
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(v=UNIT_FLOATS, u=UNIT_FLOATS)
def test_bicop_boundary_identities_hold(fitted_bicop, u, v):
    pts_zero_u = torch.tensor([[0.0, v]], dtype=DTYPE, device=DEVICE)
    pts_zero_v = torch.tensor([[u, 0.0]], dtype=DTYPE, device=DEVICE)
    pts_one = torch.tensor([[1.0, 1.0]], dtype=DTYPE, device=DEVICE)
    assert torch.allclose(
        fitted_bicop.cdf(pts_zero_u), torch.zeros(1, 1, dtype=DTYPE, device=DEVICE), atol=1e-10
    )
    assert torch.allclose(
        fitted_bicop.cdf(pts_zero_v), torch.zeros(1, 1, dtype=DTYPE, device=DEVICE), atol=1e-10
    )
    assert torch.allclose(
        fitted_bicop.cdf(pts_one), torch.ones(1, 1, dtype=DTYPE, device=DEVICE), atol=1e-10
    )


@pytest.mark.property
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(u=INTERIOR_FLOATS, p=INTERIOR_FLOATS)
def test_hfunc_hinv_are_near_inverses(fitted_bicop, u, p):
    query = torch.tensor([[u, p]], dtype=DTYPE, device=DEVICE)
    inv = fitted_bicop.hinv_l(query)
    recovered = fitted_bicop.hfunc_l(torch.hstack([query[:, [0]], inv]))
    assert torch.allclose(recovered, query[:, [1]], atol=1e-3, rtol=1e-3)


@pytest.mark.property
@given(u=INTERIOR_FLOATS, v=INTERIOR_FLOATS)
def test_independent_copula_closed_form(u, v):
    cop = tvc.BiCop(num_step_grid=33).to(DEVICE)
    obs = torch.tensor([[u, v]], dtype=DTYPE, device=DEVICE)
    assert torch.allclose(cop.pdf(obs), torch.ones(1, 1, dtype=DTYPE, device=DEVICE))
    assert torch.allclose(cop.log_pdf(obs), torch.zeros(1, 1, dtype=DTYPE, device=DEVICE))
    assert torch.allclose(
        cop.cdf(obs), torch.tensor([[u * v]], dtype=DTYPE, device=DEVICE), atol=1e-10
    )


@pytest.mark.property
def test_sample_margins_are_close_to_uniform(fitted_bicop):
    samp = fitted_bicop.sample(num_sample=2_048, seed=9)
    quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=DTYPE, device=samp.device)
    empirical = torch.quantile(samp[:, 0], quantiles)
    assert torch.allclose(empirical, quantiles, atol=0.06)
