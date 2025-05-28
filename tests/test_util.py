import pytest
import torch
from scipy.stats import kendalltau as scipy_tau

from torchvinecopulib.util import (
    ENUM_FUNC_BIDEP,
    chatterjee_xi,
    ferreira_tail_dep_coeff,
    kdeCDFPPF1D,
    kendall_tau,
    mutual_info,
    solve_ITP,
)

from . import EPS, U_tensor, bicop_pair, sample_1d


@pytest.mark.parametrize(
    "x,y,expected",
    [
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 3], [3, 2, 1], -1.0),
    ],
)
def test_kendall_tau_perfect(x, y, expected):
    x = torch.tensor(x).view(-1, 1).double()
    y = torch.tensor(y).view(-1, 1).double()
    tau, p = kendall_tau(x, y)
    assert pytest.approx(expected, abs=1e-6) == tau.item()


def test_kendall_tau_matches_scipy_random():
    torch.manual_seed(0)
    x = torch.rand(50, 1)
    y = torch.rand(50, 1)
    tau_torch, p = kendall_tau(x, y)
    tau_scipy, p_scipy = scipy_tau(x.flatten().numpy(), y.flatten().numpy())
    assert pytest.approx(tau_scipy, rel=1e-6) == tau_torch.item()


def test_mutual_info_independent():
    torch.manual_seed(0)
    mi = mutual_info(*torch.rand(2, 10000))
    assert abs(mi.item()) < 2e-2  # near zero


def test_mutual_info_dependent():
    torch.manual_seed(0)
    u = torch.rand(500, 1)
    v = u.clone()
    mi = mutual_info(u, v)
    assert mi.item() > 0.5  # significantly positive


def test_tail_dep_perfect():
    u = torch.linspace(0, 1, 500).view(-1, 1)
    # y=u → perfect
    lam = ferreira_tail_dep_coeff(u, u)
    assert pytest.approx(1.0, rel=1e-3) == lam.item()


def test_tail_dep_independent():
    torch.manual_seed(0)
    u = torch.rand(1000, 1)
    v = torch.rand(1000, 1)
    lam = ferreira_tail_dep_coeff(u, v)
    assert lam.item() < 0.2  # near zero


def test_xi_perfect():
    u = torch.arange(1, 101).view(-1, 1).double()
    xi = chatterjee_xi(u, u)
    assert pytest.approx(1.0, abs=3e-2) == xi.item()


def test_xi_independent():
    torch.manual_seed(1)
    u = torch.rand(1000, 1)
    v = torch.rand(1000, 1)
    xi = chatterjee_xi(u, v)
    assert abs(xi.item()) < 0.1


def test_enum_dispatches_correctly():
    u = torch.rand(50, 1)
    v = u.clone()
    # perfect correlation → tau=1, xi=1, tail=1, mi>0
    out_tau = ENUM_FUNC_BIDEP.kendall_tau(u, v)
    out_xi = ENUM_FUNC_BIDEP.chatterjee_xi(u, v)
    out_tail = ENUM_FUNC_BIDEP.ferreira_tail_dep_coeff(u, v)
    out_mi = ENUM_FUNC_BIDEP.mutual_info(u, v)
    assert pytest.approx(1.0, abs=3e-2) == out_tau[0].item()
    assert pytest.approx(1.0, abs=3e-2) == out_xi.item()
    assert pytest.approx(1.0, rel=3e-2) == out_tail.item()
    assert out_mi.item() > 0.5


def test_kde_cdf_ppf_inverse(sample_1d):
    kde = kdeCDFPPF1D(sample_1d, num_step_grid=257)
    range = kde.x_max - kde.x_min
    xs = torch.linspace(
        kde.x_min + range / 10, kde.x_max - range / 10, 50, dtype=torch.float64
    ).view(-1, 1)
    qs = kde.cdf(xs)
    xs_rec = kde.ppf(qs)
    assert torch.allclose(xs, xs_rec, atol=1e-3)


def test_kde_bounds_and_pdf(sample_1d):
    kde = kdeCDFPPF1D(sample_1d, num_step_grid=257)
    # cdf out-of-bounds
    oob = torch.tensor([[kde.x_min - 1.0], [kde.x_max + 1.0]], dtype=torch.float64)
    assert torch.all(kde.cdf(oob) == torch.tensor([[0.0], [1.0]], dtype=torch.float64))
    # ppf out-of-bounds
    assert torch.allclose(
        torch.tensor([[kde.x_min], [kde.x_max]], dtype=torch.float64),
        kde.ppf(torch.tensor([[-1.0], [2.0]])),
    )
    # pdf ≥ 0
    pts = torch.linspace(kde.x_min, kde.x_max, 100).view(-1, 1)
    assert (kde.pdf(pts) >= 0).all()
    # log_pdf finite
    assert torch.isfinite(kde.log_pdf(pts)).all()


def test_kde_negloglik_forward(sample_1d):
    kde = kdeCDFPPF1D(sample_1d, num_step_grid=None)
    val1 = kde.negloglik
    val2 = kde.forward(sample_1d)
    assert pytest.approx(val1.item(), rel=1e-6) == val2.item()


def test_kde_str(sample_1d):
    kde = kdeCDFPPF1D(sample_1d, num_step_grid=257)
    str_repr = str(kde)
    assert "kdeCDFPPF1D" in str_repr
    assert "num_step_grid" in str_repr
    assert "257" in str_repr
    assert "x_min" in str_repr
    assert "x_max" in str_repr


def test_solve_itp_scalar():
    # f(x)=x-0.3 has root at 0.3
    f = lambda x: x - 0.3
    root = solve_ITP(f, torch.tensor(0.0), torch.tensor(1.0))
    assert pytest.approx(0.3, abs=1e-6) == root.item()


def test_solve_itp_vectorized():
    # two independent eq’s: x-a=0, x-b=0
    a = torch.tensor([0.2, 0.7])
    b = torch.tensor([1.0, 1.0])
    f = lambda x: x - a
    roots = solve_ITP(f, torch.zeros_like(a), b)
    assert torch.allclose(roots, a, atol=1e-6)
