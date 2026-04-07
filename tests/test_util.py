import pytest
import torch

import torchvinecopulib.util as util_mod
from torchvinecopulib.util import (
    ENUM_FUNC_BIDEP,
    TorchKDE1D,
    chatterjee_xi,
    empirical_pobs,
    ferreira_tail_dep_coeff,
    kendall_tau,
    kendall_tau_matrix,
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


def test_solve_itp_failure_policy_and_status():
    root, status = solve_ITP(
        lambda x: x.square() + 1.0,
        torch.zeros(3, 1, dtype=DTYPE),
        torch.ones(3, 1, dtype=DTYPE),
        failure_policy="linear",
        return_status=True,
    )
    assert root.shape == (3, 1)
    assert status["used_fallback"].all()
    assert (~status["bracketed"]).all()


def test_solve_itp_raise_failure_policy():
    with pytest.raises(RuntimeError):
        solve_ITP(
            lambda x: x.square() + 1.0,
            torch.zeros(2, 1, dtype=DTYPE),
            torch.ones(2, 1, dtype=DTYPE),
            failure_policy="raise",
        )


def test_solve_itp_midpoint_policy_for_nonconvergence():
    root, status = solve_ITP(
        lambda x: x - 0.3,
        torch.zeros(2, 1, dtype=DTYPE),
        torch.ones(2, 1, dtype=DTYPE),
        max_iter=0,
        failure_policy="midpoint",
        return_status=True,
    )
    assert torch.allclose(root, torch.full_like(root, 0.5))
    assert status["bracketed"].all()
    assert (~status["converged"]).all()
    assert status["used_fallback"].all()


def test_solve_itp_return_status_false_midpoint():
    root = solve_ITP(
        lambda x: x.square() + 1.0,
        torch.zeros(1, 1, dtype=DTYPE),
        torch.ones(1, 1, dtype=DTYPE),
        failure_policy="midpoint",
    )
    assert torch.allclose(root, torch.full_like(root, 0.5))


def test_kendall_tau_matrix_torch_matches_pairwise():
    u = torch.linspace(0.0, 1.0, 64, dtype=DTYPE).view(-1, 1)
    v = u.square()
    w = u.flip(0)
    obs = torch.hstack([u, v, w])
    tau_mat, p_mat = kendall_tau_matrix(obs, backend="torch")
    pair_uv = kendall_tau(u, v, backend="torch")
    pair_uw = kendall_tau(u, w, backend="torch")
    assert torch.allclose(tau_mat, tau_mat.T, atol=1e-8)
    assert torch.allclose(torch.diag(tau_mat), torch.ones(3, dtype=DTYPE), atol=1e-8)
    assert torch.allclose(tau_mat[0, 1], pair_uv[0], atol=1e-8)
    assert torch.allclose(tau_mat[0, 2], pair_uw[0], atol=1e-8)
    assert (p_mat >= 0.0).all() and (p_mat <= 1.0).all()


def test_kendall_tau_matrix_tie_heavy_parity(tie_heavy_obs):
    tau_torch, p_torch = kendall_tau_matrix(tie_heavy_obs, backend="torch")
    tau_scipy, p_scipy = kendall_tau_matrix(tie_heavy_obs.cpu(), backend="scipy")
    tau_scipy = tau_scipy.to(device=tau_torch.device, dtype=tau_torch.dtype)
    p_scipy = p_scipy.to(device=p_torch.device, dtype=p_torch.dtype)
    assert torch.allclose(tau_torch, tau_scipy, atol=1e-4, rtol=1e-4)
    assert torch.allclose(p_torch, p_scipy, atol=1e-4, rtol=1e-4)


def test_kendall_tau_auto_uses_scipy_on_cpu(monkeypatch):
    called = {"scipy": False}

    def fake_lazy():
        called["scipy"] = True
        return lambda x, y: (0.25, 0.75)

    monkeypatch.setattr(util_mod, "_lazy_kendalltau", fake_lazy)
    x = torch.tensor([[0.1], [0.2], [0.3]], dtype=DTYPE)
    y = torch.tensor([[0.3], [0.2], [0.1]], dtype=DTYPE)
    stat = kendall_tau(x, y, backend="auto")
    assert called["scipy"]
    assert torch.allclose(stat, torch.tensor([0.25, 0.75], dtype=DTYPE))


@pytest.mark.parametrize("scenario", ["independent", "weak", "strong"])
@pytest.mark.parametrize(
    "metric_name",
    ["kendall_tau", "chatterjee_xi", "ferreira_tail_dep_coeff", "mutual_info"],
)
def test_dependence_metrics_cover_common_scenarios(metric_name, scenario, cpu_generator):
    if scenario == "independent":
        x = torch.rand(256, 1, dtype=DTYPE, generator=cpu_generator)
        y = torch.rand(256, 1, dtype=DTYPE, generator=cpu_generator)
    elif scenario == "weak":
        base = torch.linspace(0.0, 1.0, 256, dtype=DTYPE).view(-1, 1)
        x = base
        y = (base + 0.08 * torch.sin(base * 10.0)).clamp(0.0, 1.0)
    else:
        x = torch.linspace(0.0, 1.0, 256, dtype=DTYPE).view(-1, 1)
        y = x.clone()
    out = ENUM_FUNC_BIDEP[metric_name](x, y)
    assert torch.isfinite(torch.as_tensor(out)).all()
