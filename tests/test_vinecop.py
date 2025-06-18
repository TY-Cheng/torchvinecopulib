import matplotlib.pyplot as plt
import networkx as nx
import pytest
import torch

from torchvinecopulib.vinecop import VineCop


def test_init_attributes_and_defaults():
    num_dim = 5
    vc = VineCop(num_dim=num_dim, is_cop_scale=True, num_step_grid=32)
    # basic attrs
    assert vc.num_dim == num_dim
    assert vc.is_cop_scale is True
    assert vc.num_step_grid == 32
    # empty structures
    assert isinstance(vc.marginals, torch.nn.ModuleList)
    assert len(vc.marginals) == num_dim
    assert isinstance(vc.bicops, torch.nn.ModuleDict)
    # number of pair-copulas = n(n-1)/2
    assert len(vc.bicops) == num_dim * (num_dim - 1) // 2
    assert vc.tree_bidep == [{} for _ in range(num_dim - 1)]
    assert vc.sample_order == tuple(range(num_dim))
    # no data yet
    assert vc.num_obs.item() == 0


def test_device_and_dtype_after_cuda():
    vc = VineCop(3).to("cpu")
    assert vc.device.type == "cpu"
    assert vc.dtype is torch.float64
    if torch.cuda.is_available():
        vc2 = VineCop(3).cuda()
        assert vc2.device.type == "cuda"


@pytest.mark.parametrize("mtd_vine", ["dvine", "cvine", "rvine"])
@pytest.mark.parametrize(
    "mtd_bidep",
    ["kendall_tau", "mutual_info", "chatterjee_xi", "ferreira_tail_dep_coeff"],
)
def test_fit_sample_logpdf_cdf_forward(mtd_vine, mtd_bidep):
    num_dim = 5
    torch.manual_seed(0)
    U = torch.special.ndtri(torch.rand(200, num_dim, dtype=torch.float64))
    vc = VineCop(num_dim=num_dim, is_cop_scale=False, num_step_grid=32)
    # fit in copula-scale
    vc.fit(
        obs=U,
        mtd_kde="tll",  # use TLL for bidep estimation
        is_dissmann=True,
        mtd_vine=mtd_vine,
        mtd_bidep=mtd_bidep,
        thresh_trunc=None,  # no truncation
    )
    # num_obs must be updated
    assert vc.num_obs.item() == 200

    # sampling
    samp = vc.sample(num_sample=50, seed=1, is_sobol=False)
    assert samp.shape == (50, num_dim)

    # log_pdf
    lp = vc.log_pdf(U)
    assert lp.shape == (200, 1)
    # forward = neg average log-lik
    fwd = vc.forward(U)
    assert fwd.dim() == 0

    # cdf approximation in [0,1]
    cdf_vals = vc.cdf(U[:10], num_sample=1000, seed=2)
    assert cdf_vals.shape == (10, 1)
    assert (cdf_vals >= 0).all() and (cdf_vals <= 1).all()


@pytest.mark.parametrize("mtd_vine", ["dvine", "cvine", "rvine"])
@pytest.mark.parametrize(
    "mtd_bidep",
    ["kendall_tau", "mutual_info", "chatterjee_xi", "ferreira_tail_dep_coeff"],
)
def test_matrix_diagonal_matches_sample_order(mtd_vine, mtd_bidep):
    num_dim = 5
    vc = VineCop(num_dim, is_cop_scale=True, num_step_grid=16)
    torch.manual_seed(1)
    U = torch.rand(100, num_dim, dtype=torch.float64)
    vc.fit(
        U,
        mtd_kde="tll",
        is_dissmann=True,
        mtd_vine=mtd_vine,
        mtd_bidep=mtd_bidep,
    )
    M = vc.matrix
    # must be square
    assert M.shape == (num_dim, num_dim)
    # * d unique elements in each row
    for i in range(num_dim):
        elems = set(M[i, :].tolist())
        elems.discard(-1)  # discard -1
        assert len(elems) == num_dim - i, f"Row {i} has incorrect unique elements"
    # diag = sample_order
    for i in range(num_dim):
        assert M[i, i].item() == vc.sample_order[i]


def test_fit_with_explicit_matrix_uses_exact_edges():
    num_dim = 5
    torch.manual_seed(1)
    U = torch.rand(200, num_dim, dtype=torch.float64)
    vc = VineCop(num_dim, is_cop_scale=True, num_step_grid=16)
    vc.fit(U, mtd_kde="tll", is_dissmann=True)
    M = vc.matrix
    # M is a square matrix with num_dim rows and columns
    vc = VineCop(num_dim, is_cop_scale=True, num_step_grid=16)
    # fit with our explicit matrix
    vc.fit(U, is_dissmann=False, matrix=M)
    # now verify that for each level lv, and each idx in [0..num_dim-lv-2],
    # the edge (v_l, v_r, *cond) comes out exactly as we encoded it in M
    for lv in range(num_dim - 1):
        tree = vc.tree_bidep[lv]
        # must have exactly num_dim-lv-1 edges
        assert len(tree) == num_dim - lv - 1

        for idx in range(num_dim - lv - 1):
            # the two “free” spots in row idx of M
            a = int(M[idx, idx])
            b = int(M[idx, num_dim - lv - 1])
            v_l, v_r = sorted((a, b))
            # any remaining entries in that row form the conditioning set
            cond = sorted([int(_) for _ in M[idx, num_dim - lv :].tolist()])
            expected_edge = (v_l, v_r, *cond)

            # check that this exact tuple is a key in tree_bidep[lv]
            assert expected_edge in tree, (
                f"Expected edge {expected_edge} at level {lv} but got {list(tree)}"
            )


def test_reset_clears_all_levels_and_bicops():
    num_dim = 5
    vc = VineCop(num_dim, is_cop_scale=True, num_step_grid=16)
    torch.manual_seed(0)
    U = torch.rand(50, num_dim, dtype=torch.float64)
    # fit with Dissmann algorithm
    vc.fit(
        U,
        is_dissmann=True,
        mtd_kde="tll",
        mtd_vine="rvine",
        mtd_bidep="kendall_tau",
        thresh_trunc=None,
    )
    assert vc.num_obs.item() > 0
    # now reset
    vc.reset()
    assert vc.num_obs.item() == 0
    assert vc.tree_bidep == [{} for _ in range(num_dim - 1)]
    # all BiCop should be independent again
    assert all(bc.is_indep for bc in vc.bicops.values())


def test_reset_and_str():
    num_dim = 5
    vc = VineCop(num_dim, is_cop_scale=True, num_step_grid=16)
    torch.manual_seed(0)
    U = torch.rand(50, num_dim, dtype=torch.float64)
    # fit with Dissmann algorithm
    vc.fit(
        U,
        is_dissmann=True,
        mtd_kde="tll",
        mtd_vine="rvine",
        mtd_bidep="kendall_tau",
        thresh_trunc=None,
    )
    # __str__ contains key fields
    s = str(vc)
    for key in [
        "num_dim",
        "num_obs",
        "is_cop_scale",
        "num_step_grid",
        "mtd_bidep",
        "negloglik",
        "dtype",
        "device",
        "sample_order",
    ]:
        assert key in s, f"'{key}' not found in VineCop string representation"


@pytest.mark.parametrize("mtd_vine", ["dvine", "cvine", "rvine"])
@pytest.mark.parametrize(
    "mtd_bidep",
    ["kendall_tau", "mutual_info", "chatterjee_xi", "ferreira_tail_dep_coeff"],
)
def test_ref_count_hfunc_on_fitted_vine(mtd_vine, mtd_bidep):
    num_dim = 5
    torch.manual_seed(0)
    U = torch.rand(100, num_dim, dtype=torch.float64)
    vc = VineCop(num_dim, is_cop_scale=True, num_step_grid=16)
    vc.fit(
        U,
        mtd_kde="tll",
        is_dissmann=True,
        mtd_vine=mtd_vine,
        mtd_bidep=mtd_bidep,
        thresh_trunc=0.1,
    )
    # test static ref_count_hfunc
    ref_cnt, sources, num_hfunc = VineCop.ref_count_hfunc(
        num_dim=vc.num_dim,
        struct_obs=vc.struct_obs,
        sample_order=vc.sample_order,
    )
    assert isinstance(ref_cnt, dict)
    assert isinstance(sources, list)
    assert isinstance(num_hfunc, int)
    if mtd_vine == "cvine":
        # for cvine, we expect 0 hfuncs
        assert num_hfunc == 0
    else:
        assert num_hfunc >= 0


def test_draw_lv_and_draw_dag(tmp_path):
    num_dim = 5
    torch.manual_seed(0)
    U = torch.rand(100, num_dim, dtype=torch.float64)
    vc = VineCop(num_dim, is_cop_scale=True, num_step_grid=32)
    vc.fit(
        U,
        mtd_kde="tll",
        is_dissmann=True,
        mtd_vine="rvine",
        mtd_bidep="kendall_tau",
        thresh_trunc=None,
    )

    # draw level-0 with pseudo-obs nodes
    fig, ax, G = vc.draw_lv(lv=0, is_bcp=False)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() > 0
    plt.close(fig)

    # save level-1 bicop view
    fpath_lv = tmp_path / "level1.png"
    fig2, ax2, G2, outpath_lv = vc.draw_lv(lv=1, is_bcp=True, f_path=fpath_lv)
    assert outpath_lv == fpath_lv
    assert fpath_lv.exists()
    plt.close(fig2)
    # save level-1 pseudo-obs view
    fpath_lv = tmp_path / "level1.png"
    fig2, ax2, G2, outpath_lv = vc.draw_lv(lv=1, is_bcp=False, f_path=fpath_lv)
    assert outpath_lv == fpath_lv
    assert fpath_lv.exists()
    plt.close(fig2)

    # draw the DAG
    fig3, ax3, G3 = vc.draw_dag()
    assert isinstance(G3, nx.DiGraph)
    assert len(G3.nodes) > 0
    plt.close(fig3)

    # save DAG
    fpath_dag = tmp_path / "dag.png"
    fig4, ax4, G4, outpath_dag = vc.draw_dag(f_path=fpath_dag)
    assert outpath_dag == fpath_dag
    assert fpath_dag.exists()
    plt.close(fig4)
