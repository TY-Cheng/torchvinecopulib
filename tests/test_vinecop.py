import importlib.util

import matplotlib.pyplot as plt
import networkx as nx
import pytest
import torch

from torchvinecopulib.vinecop import VineCop

from . import DEVICE, correlated_raw, gaussian_copula


HAS_REFERENCE = importlib.util.find_spec("pyvinecopulib") is not None


def _fit_vine(
    is_cop_scale: bool,
    mtd_vine: str,
    mtd_bidep: str,
    *,
    marginal_backend: str = "grid",
    bicop_backend: str = "grid_reflect",
) -> tuple[torch.Tensor, VineCop]:
    obs = gaussian_copula(num_obs=300, rho=0.55, dim=5) if is_cop_scale else correlated_raw(300, 0.55, 5)
    vc = VineCop(num_dim=5, is_cop_scale=is_cop_scale, num_step_grid=65).to(DEVICE)
    vc.fit(
        obs.to(DEVICE),
        mtd_vine=mtd_vine,
        mtd_bidep=mtd_bidep,
        marginal_backend=marginal_backend,
        marginal_kwargs={"bandwidth": "isj"} if marginal_backend == "grid" else None,
        bicop_backend=bicop_backend,
        bicop_kwargs={"bandwidth": "silverman"} if bicop_backend != "tll_ref" else None,
        thresh_trunc=None,
    )
    return obs.to(DEVICE), vc


def test_init_defaults():
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=32)
    assert vc.num_dim == 4
    assert vc.sample_order == (0, 1, 2, 3)
    assert len(vc.bicops) == 6


@pytest.mark.parametrize("mtd_vine", ["dvine", "cvine", "rvine"])
@pytest.mark.parametrize(
    "mtd_bidep",
    ["kendall_tau", "mutual_info", "chatterjee_xi", "ferreira_tail_dep_coeff"],
)
def test_fit_logpdf_sample_and_cdf(mtd_vine, mtd_bidep):
    obs, vc = _fit_vine(False, mtd_vine, mtd_bidep)
    assert vc.num_obs.item() == 300
    samp = vc.sample(num_sample=40, seed=1)
    assert samp.shape == (40, vc.num_dim)
    lp = vc.log_pdf(obs)
    assert lp.shape == (300, 1)
    assert torch.isfinite(lp).all()
    cdf = vc.cdf(obs[:8], num_sample=511, seed=2)
    assert cdf.shape == (8, 1)
    assert (cdf >= 0.0).all() and (cdf <= 1.0).all()


@pytest.mark.parametrize("mtd_vine", ["dvine", "cvine", "rvine"])
def test_matrix_diagonal_matches_sample_order(mtd_vine):
    obs, vc = _fit_vine(True, mtd_vine, "kendall_tau")
    M = vc.matrix
    assert M.shape == (vc.num_dim, vc.num_dim)
    for i in range(vc.num_dim):
        assert M[i, i].item() == vc.sample_order[i]


def test_fit_with_explicit_matrix():
    obs, vc0 = _fit_vine(True, "rvine", "kendall_tau")
    M = vc0.matrix
    vc1 = VineCop(num_dim=5, is_cop_scale=True, num_step_grid=65).to(DEVICE)
    vc1.fit(
        obs,
        is_dissmann=False,
        matrix=M,
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
    )
    for lv in range(vc1.num_dim - 1):
        assert len(vc1.tree_bidep[lv]) == vc1.num_dim - lv - 1


def test_reset_and_str():
    _, vc = _fit_vine(True, "rvine", "kendall_tau")
    assert "sample_order" in str(vc)
    assert "bicop_backend" in str(vc)
    vc.reset()
    assert vc.num_obs.item() == 0
    assert vc.tree_bidep == [{} for _ in range(vc.num_dim - 1)]


def test_ref_count_hfunc():
    _, vc = _fit_vine(True, "rvine", "kendall_tau")
    ref_cnt, sources, num_hfunc = VineCop.ref_count_hfunc(
        num_dim=vc.num_dim,
        struct_obs=vc.struct_obs,
        sample_order=vc.sample_order,
    )
    assert isinstance(ref_cnt, dict)
    assert isinstance(sources, list)
    assert isinstance(num_hfunc, int)


def test_draw_lv_and_draw_dag():
    _, vc = _fit_vine(True, "rvine", "kendall_tau")
    fig, ax, G = vc.draw_lv(lv=0, is_bcp=False)
    assert isinstance(G, nx.Graph)
    plt.close(fig)
    fig2, ax2, G2 = vc.draw_dag()
    assert isinstance(G2, nx.DiGraph)
    plt.close(fig2)


def test_lp_ref_marginal_backend_smoke():
    obs, vc = _fit_vine(False, "rvine", "kendall_tau", marginal_backend="lp_ref")
    lp = vc.log_pdf(obs[:32])
    assert torch.isfinite(lp).all()


def test_invalid_backend_kwargs_raise():
    obs = correlated_raw(200, 0.4, 4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=False, num_step_grid=33).to(DEVICE)
    with pytest.raises(ValueError):
        vc.fit(obs, marginal_kwargs={"unknown": 1})


def test_legacy_backend_alias_warns_and_maps():
    obs = gaussian_copula(num_obs=128, rho=0.35, dim=4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33).to(DEVICE)
    with pytest.deprecated_call():
        vc.fit(obs, mtd_kde="fastKDE", thresh_trunc=None)
    assert vc.bicop_backend == "grid_reflect"


def test_vinecop_loads_legacy_state_without_extra_state():
    obs, vc = _fit_vine(True, "rvine", "kendall_tau")
    legacy = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in vc.state_dict().items()
        if not k.endswith("._extra_state") and k != "_extra_state"
    }
    fresh = VineCop(num_dim=5, is_cop_scale=True, num_step_grid=65).to(DEVICE)
    fresh.load_state_dict(legacy)
    assert fresh.bicop_backend == "grid_reflect"
    assert fresh.marginal_backend == "grid"
    assert fresh.num_obs.item() == vc.num_obs.item()


@pytest.mark.skipif(not HAS_REFERENCE, reason="pyvinecopulib reference backend not installed")
def test_reference_backend_smoke():
    obs = gaussian_copula(num_obs=200, rho=0.4, dim=4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33).to(DEVICE)
    vc.fit(obs, bicop_backend="tll_ref", thresh_trunc=None)
    assert torch.isfinite(vc.log_pdf(obs)).all()
