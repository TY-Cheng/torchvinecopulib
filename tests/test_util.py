import pytest
import torch

from torchvinecopulib.util import (
    ENUM_FUNC_BIDEP,
    TorchKDE1D,
    chatterjee_xi,
    empirical_pobs,
    ferreira_tail_dep_coeff,
    kendall_tau,
    mutual_info,
    solve_ITP,
)

from . import DTYPE


def test_kendall_tau_perfect():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE).view(-1, 1)
    tau, _ = kendall_tau(x, x)
    assert pytest.approx(1.0, abs=1e-6) == tau.item()


def test_mutual_info_independent_and_dependent():
    torch.manual_seed(0)
    u = torch.rand(5000, 1, dtype=DTYPE)
    v = torch.rand(5000, 1, dtype=DTYPE)
    dep = u.clone()
    assert abs(mutual_info(u, v).item()) < 0.05
    assert mutual_info(u, dep).item() > 0.2


def test_tail_dep_and_xi():
    u = torch.linspace(0.0, 1.0, 500, dtype=DTYPE).view(-1, 1)
    assert pytest.approx(1.0, rel=1e-2) == ferreira_tail_dep_coeff(u, u).item()
    assert pytest.approx(1.0, abs=5e-2) == chatterjee_xi(u, u).item()


def test_empirical_pobs_range():
    x = torch.randn(128, 1, dtype=DTYPE)
    u = empirical_pobs(x)
    assert u.min() > 0.0
    assert u.max() < 1.0


def test_torch_kde_inverse_and_integral():
    torch.manual_seed(0)
    x = torch.cat(
        [
            torch.randn(800, 1, dtype=DTYPE) * 0.5 - 1.0,
            torch.randn(800, 1, dtype=DTYPE) * 0.25 + 1.5,
        ],
        dim=0,
    )
    kde = TorchKDE1D(x, num_step_grid=257, bandwidth="isj")
    delta = (kde.x_max - kde.x_min) / (kde.num_step_grid - 1)
    assert pytest.approx(1.0, rel=2e-2) == (kde.grid_pdf.sum() * delta).item()
    xs = torch.linspace(kde.x_min + 0.3, kde.x_max - 0.3, 50, dtype=DTYPE).view(-1, 1)
    qs = kde.cdf(xs)
    xs_rec = kde.ppf(qs)
    assert torch.allclose(xs, xs_rec, atol=5e-2)
    assert torch.isfinite(kde.log_pdf(xs)).all()


def test_torch_kde_small_sample_fallback():
    x = torch.tensor([[0.0], [0.1], [0.2], [0.25]], dtype=DTYPE)
    kde = TorchKDE1D(x, num_step_grid=65, bandwidth="isj")
    assert kde.bandwidth.item() > 0.0
    assert kde.bandwidth_method in {"isj", "silverman"}


def test_enum_dispatches():
    u = torch.rand(128, 1, dtype=DTYPE)
    v = u.clone()
    assert ENUM_FUNC_BIDEP.kendall_tau(u, v)[0].item() > 0.9
    assert ENUM_FUNC_BIDEP.chatterjee_xi(u, v).item() > 0.9
    assert ENUM_FUNC_BIDEP.ferreira_tail_dep_coeff(u, v).item() > 0.9
    assert ENUM_FUNC_BIDEP.mutual_info(u, v).item() > 0.2


def test_solve_itp_scalar_and_vectorized():
    def f(x):
        return x - 0.3

    root = solve_ITP(f, torch.tensor(0.0), torch.tensor(1.0))
    assert pytest.approx(0.3, abs=1e-6) == root.item()
    a = torch.tensor([0.2, 0.7], dtype=DTYPE)
    roots = solve_ITP(lambda x: x - a, torch.zeros_like(a), torch.ones_like(a))
    assert torch.allclose(roots, a, atol=1e-6)
