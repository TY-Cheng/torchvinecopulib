import importlib.util

import matplotlib.pyplot as plt
import networkx as nx
import pytest
import torch

from torchvinecopulib.vinecop import VineCop
from torchvinecopulib.vinecop import VineBuilder

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


def test_unfitted_query_paths_raise():
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=32).to(DEVICE)
    obs = gaussian_copula(num_obs=8, rho=0.2, dim=4).to(DEVICE)
    with pytest.raises(RuntimeError):
        vc.log_pdf(obs)
    with pytest.raises(RuntimeError):
        vc.sample(num_sample=8, seed=1)
    with pytest.raises(RuntimeError):
        vc.cdf(obs[:4], num_sample=31, seed=1)
    with pytest.raises(RuntimeError):
        vc.rosenblatt(obs[:4])
    with pytest.raises(RuntimeError):
        vc.inverse_rosenblatt(obs[:4])
    with pytest.raises(RuntimeError):
        vc._sample_u(num_sample=4, seed=1)


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


def test_fit_with_explicit_matrix_non_kendall_path():
    obs, vc0 = _fit_vine(True, "rvine", "kendall_tau")
    M = vc0.matrix
    vc1 = VineCop(num_dim=5, is_cop_scale=True, num_step_grid=65).to(DEVICE)
    vc1.fit(
        obs,
        is_dissmann=False,
        matrix=M,
        mtd_bidep="chatterjee_xi",
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        thresh_trunc=None,
    )
    assert torch.isfinite(vc1.log_pdf(obs[:16])).all()


def test_invalid_mtd_vine_raises():
    obs = gaussian_copula(num_obs=64, rho=0.35, dim=4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33).to(DEVICE)
    with pytest.raises(ValueError, match="mtd_vine"):
        vc.fit(obs, mtd_vine="bad-vine", thresh_trunc=None)


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


@pytest.mark.reference
@pytest.mark.skipif(not HAS_REFERENCE, reason="pyvinecopulib reference backend not installed")
def test_reference_backend_smoke():
    obs = gaussian_copula(num_obs=200, rho=0.4, dim=4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33).to(DEVICE)
    vc.fit(obs, bicop_backend="tll_ref", thresh_trunc=None)
    assert torch.isfinite(vc.log_pdf(obs)).all()


def test_engine_proxy_and_from_artifact_roundtrip():
    obs, vc = _fit_vine(True, "rvine", "kendall_tau")
    assert vc.engine is not None
    assert not hasattr(vc.engine, "_owner")
    assert torch.allclose(vc.engine.log_pdf(obs[:16]), vc.log_pdf(obs[:16]), atol=1e-6)
    assert torch.allclose(vc.engine.sample(num_sample=8, seed=7), vc.sample(num_sample=8, seed=7), atol=1e-6)
    assert torch.allclose(vc.engine.cdf(obs[:4], num_sample=127, seed=5), vc.cdf(obs[:4], num_sample=127, seed=5), atol=1e-6)
    assert torch.allclose(vc.engine(obs[:16]), vc.forward(obs[:16]), atol=1e-6)
    clone = VineCop.from_artifact(vc.engine.artifact)
    clone = clone.to(DEVICE)
    assert torch.allclose(clone.log_pdf(obs[:16]), vc.log_pdf(obs[:16]), atol=1e-6)


def test_execution_plan_contains_static_slots_and_ops():
    obs, vc = _fit_vine(True, "rvine", "kendall_tau")
    del obs
    artifact = vc.engine.artifact
    assert artifact.vertex_order
    assert artifact.slot_by_vertex
    assert artifact.base_slots.shape == (vc.num_dim,)
    assert artifact.forward_modes.numel() > 0
    assert artifact.logpdf_bicop_ids.numel() > 0
    assert artifact.sample_modes.numel() > 0
    assert artifact.source_slots.shape[0] == vc.num_dim


def test_rosenblatt_inverse_roundtrip_on_copula_scale():
    obs, vc = _fit_vine(True, "rvine", "kendall_tau")
    transformed = vc.rosenblatt(obs[:24])
    recovered = vc.inverse_rosenblatt(transformed)
    assert transformed.shape == (24, vc.num_dim)
    assert torch.allclose(recovered, obs[:24], atol=1e-6)


def test_invalid_custom_sample_order_raises():
    obs, vc = _fit_vine(True, "rvine", "kendall_tau")
    sample_order = tuple(reversed(vc.sample_order))
    with pytest.raises(ValueError, match="sample_order"):
        vc.rosenblatt(obs[:24], sample_order=sample_order)


def test_engine_sample_u_supports_partial_source_observations():
    obs, vc = _fit_vine(True, "rvine", "kendall_tau")
    source_vertex = vc.engine.artifact.source_vertices[0]
    fixed = torch.full((16, 1), 0.25, dtype=obs.dtype, device=obs.device)
    sampled = vc.engine._sample_u(
        num_sample=16,
        seed=3,
        dct_v_s_obs={source_vertex: fixed},
    )
    assert sampled.shape == (16, vc.num_dim)
    assert torch.isfinite(sampled).all()


def test_export_inference_plan_casts_query_dtype():
    obs, vc = _fit_vine(True, "rvine", "kendall_tau")
    plan = vc.export_inference_plan(dtype=torch.float32)
    clone = VineCop.from_artifact(plan).to(DEVICE)
    score = clone.engine.log_pdf(obs[:16].to(dtype=torch.float32))
    assert clone.engine.dtype is torch.float32
    assert score.dtype is torch.float32
    assert torch.isfinite(score).all()


def test_vine_diagnostics_are_observable():
    obs, vc = _fit_vine(True, "rvine", "kendall_tau")
    del obs
    diag = vc.diagnostics()
    assert diag.num_edges == len(vc.bicops)
    assert diag.itp_failures >= 0
    assert diag.bisect_refinements >= 0
    assert diag.fallback_to_indep >= 0
    assert diag.max_abs_hfunc_error >= 0.0


def test_truncation_can_leave_independence_edges():
    obs = gaussian_copula(num_obs=96, rho=0.35, dim=4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33).to(DEVICE)
    vc.fit(
        obs,
        mtd_bidep="kendall_tau",
        thresh_trunc=-1.0,
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
    )
    assert all(bicop.is_indep for bicop in vc.bicops.values())
    assert torch.isfinite(vc.log_pdf(obs[:8])).all()


def test_engine_compile_smoke():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")
    obs, vc = _fit_vine(True, "rvine", "kendall_tau")
    compiled_log_pdf = torch.compile(vc.engine.log_pdf, backend="eager")
    compiled_rosenblatt = torch.compile(vc.engine.rosenblatt, backend="eager")
    assert torch.allclose(compiled_log_pdf(obs[:16]), vc.engine.log_pdf(obs[:16]), atol=1e-6)
    assert torch.allclose(compiled_rosenblatt(obs[:16]), vc.engine.rosenblatt(obs[:16]), atol=1e-6)


def test_fit_with_torch_kendall_backend_smoke():
    obs = gaussian_copula(num_obs=128, rho=0.45, dim=4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33).to(DEVICE)
    vc.fit(obs, mtd_bidep="kendall_tau", bidep_backend="torch", thresh_trunc=None)
    assert torch.isfinite(vc.log_pdf(obs[:16])).all()


def test_builder_build_exposes_artifact_metadata():
    obs = gaussian_copula(num_obs=128, rho=0.45, dim=4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33).to(DEVICE)
    artifact = VineBuilder(vc).build(
        obs,
        mtd_bidep="kendall_tau",
        bidep_backend="torch",
        thresh_trunc=None,
    )
    assert artifact.num_dim == 4
    assert artifact.boundary_policy == "hard"
    assert len(artifact.level_edge_tensors) == vc.num_dim - 1
    assert artifact.source_vertices
    assert artifact.bicop_backend == "grid_reflect"


def test_boundary_policy_propagates_via_state_and_artifact():
    obs = gaussian_copula(num_obs=128, rho=0.35, dim=4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33, boundary_policy="st").to(DEVICE)
    vc.fit(obs, mtd_bidep="kendall_tau", thresh_trunc=None)
    clone = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33, boundary_policy="hard").to(DEVICE)
    clone.load_state_dict(vc.state_dict())
    assert clone.boundary_policy == "st"
    assert all(bicop.boundary_policy == "st" for bicop in clone.bicops.values())
    from_artifact = VineCop.from_artifact(vc.engine.artifact).to(DEVICE)
    assert all(bicop.boundary_policy == "st" for bicop in from_artifact.bicops.values())


def test_fit_with_auto_kendall_backend_smoke():
    obs = gaussian_copula(num_obs=96, rho=0.4, dim=4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33).to(DEVICE)
    vc.fit(obs, mtd_bidep="kendall_tau", bidep_backend="auto", thresh_trunc=None)
    assert torch.isfinite(vc.log_pdf(obs[:16])).all()


def test_draw_helpers_can_save_files(tmp_path):
    _, vc = _fit_vine(True, "rvine", "kendall_tau")
    f_lv = tmp_path / "vine-level.png"
    f_dag = tmp_path / "vine-dag.png"
    fig, ax, G, out_lv = vc.draw_lv(lv=0, is_bcp=False, f_path=f_lv)
    assert out_lv == f_lv and f_lv.exists()
    plt.close(fig)
    fig_mid, ax_mid, G_mid = vc.draw_lv(lv=1, is_bcp=False)
    assert isinstance(G_mid, nx.Graph)
    plt.close(fig_mid)
    fig_bcp, ax_bcp, G_bcp = vc.draw_lv(lv=1, is_bcp=True)
    assert isinstance(G_bcp, nx.Graph)
    plt.close(fig_bcp)
    fig2, ax2, G2, out_dag = vc.draw_dag(f_path=f_dag)
    assert out_dag == f_dag and f_dag.exists()
    assert isinstance(G, nx.Graph) and isinstance(G2, nx.DiGraph)
    plt.close(fig2)


def test_conflicting_legacy_kwargs_warn():
    obs = correlated_raw(200, 0.4, 4).to(DEVICE)
    vc = VineCop(num_dim=4, is_cop_scale=False, num_step_grid=33).to(DEVICE)
    with pytest.deprecated_call():
        vc.fit(
            obs,
            marginal_kwargs={"bandwidth": "silverman", "bandwidth_scale": 1.1, "num_step_grid": 65},
            bicop_kwargs={"bandwidth_scale": 0.9, "num_iter_max": 3},
            bandwidth=0.2,
            bandwidth_scale=1.2,
            num_iter_max=4,
            num_step_grid_kde1d=129,
            smoother="recursive",
        )


def test_set_extra_state_propagates_boundary_policy():
    vc = VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33, boundary_policy="hard").to(DEVICE)
    state = vc.get_extra_state()
    state["boundary_policy"] = "st"
    vc.set_extra_state(state)
    assert vc.boundary_policy == "st"
    assert all(bicop.boundary_policy == "st" for bicop in vc.bicops.values())
