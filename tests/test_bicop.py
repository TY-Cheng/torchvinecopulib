import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

import torchvinecopulib as tvc

from . import EPS, U_tensor, bicop_pair


def test_device_and_dtype():
    cop = tvc.BiCop(num_step_grid=16)
    # by default on CPU float64
    assert cop.device.type == "cpu"
    assert cop.dtype is torch.float64

    if torch.cuda.is_available():
        cop = tvc.BiCop(num_step_grid=16).cuda()
        assert cop.device.type == "cuda"


def test_monotonicity_and_range(bicop_pair):
    family, params, rotation, U, bc_fast, bc_tll = bicop_pair

    # pick one of the two implementations or loop both
    for bicop in (bc_fast, bc_tll):
        # * simple diagonal check
        grid = torch.linspace(EPS, 1.0 - EPS, 100, device=U.device, dtype=torch.float64).unsqueeze(
            1
        )
        pts = torch.hstack([grid, grid])
        out = bicop.cdf(pts)
        assert out.min() >= -EPS and out.max() <= 1 + EPS
        assert out.diff(0).min() >= -EPS

        # * row slices + left/right variants
        for u in grid:
            pts = torch.hstack([grid, u.repeat(grid.size(0), 1)])
            for pts in (pts, pts.flip(1)):
                for fn in (bicop.cdf, bicop.hfunc_r, bicop.hinv_r):
                    v = fn(pts)
                    assert v.min() >= -EPS and v.max() <= 1 + EPS
                    assert v.diff(0).min() >= -EPS


def test_inversion(bicop_pair):
    family, params, rotation, U, bc_fast, bc_tll = bicop_pair

    for bicop in (bc_fast, bc_tll):
        grid = torch.linspace(0.1, 0.9, 50, device=U.device, dtype=torch.float64).unsqueeze(1)
        for u in grid:
            pts = torch.hstack([grid, u.repeat(grid.size(0), 1)])

            # * right‐side inverse
            rec0 = bicop.hinv_r(torch.hstack([bicop.hfunc_r(pts), pts[:, [1]]]))
            assert torch.allclose(rec0, pts[:, [0]], atol=1e-3)

            rec1 = bicop.hfunc_r(torch.hstack([bicop.hinv_r(pts), pts[:, [1]]]))
            assert torch.allclose(rec1, pts[:, [0]], atol=1e-3)

            # * left‐side inverse
            pts_rev = pts.flip(1)
            rec2 = bicop.hinv_l(torch.hstack([pts_rev[:, [0]], bicop.hfunc_l(pts_rev)]))
            assert torch.allclose(rec2, pts_rev[:, [1]], atol=1e-3)

            rec3 = bicop.hfunc_l(torch.hstack([pts_rev[:, [0]], bicop.hinv_l(pts_rev)]))
            assert torch.allclose(rec3, pts_rev[:, [1]], atol=1e-3)


def test_pdf_integrates_to_one(bicop_pair):
    family, params, rotation, U, bc_fast, bc_tll = bicop_pair
    for cop in (bc_fast, bc_tll):
        # our grid is uniform on [0,1]² with spacing Δ = 1/(N−1)
        Δ = 1.0 / (cop.num_step_grid - 1)
        # approximate ∫ pdf(u,v) du dv ≈ Σ_pdf_grid * Δ²
        approx_mass = (cop._pdf_grid.sum() * Δ**2).item()
        assert pytest.approx(expected=1.0, rel=1e-2) == approx_mass
        # non-negativity
        assert (cop._pdf_grid >= -EPS).all()


def test_log_pdf_matches_log_of_pdf(bicop_pair):
    family, params, rotation, U, bc_fast, bc_tll = bicop_pair
    for cop in (bc_fast, bc_tll):
        pts = torch.rand(500, 2, dtype=torch.float64, device=cop.device)
        pdf = cop.pdf(pts)
        logp = cop.log_pdf(pts)
        # where pdf>0, log_pdf == log(pdf)
        mask = pdf.squeeze(1) > 0
        assert torch.allclose(logp[mask], pdf[mask].log(), atol=1e-6)


def test_log_pdf_handles_zero():
    cop = tvc.BiCop(num_step_grid=4)
    cop.is_indep = False
    # ! monkey‐patch pdf to always return zero
    cop.pdf = lambda obs: torch.zeros(obs.shape[0], 1, dtype=torch.float64)
    pts = torch.rand(100, 2, dtype=torch.float64)
    logp = cop.log_pdf(pts)
    # every entry should equal the neg‐infinity replacement
    assert torch.all(logp == -13.815510557964274)


def test_sample_marginals(bicop_pair):
    family, params, rotation, U, bc_fast, bc_tll = bicop_pair
    for cop in (bc_fast, bc_tll):
        for is_sobol in (False, True):
            samp = cop.sample(2000, seed=0, is_sobol=is_sobol)
            # samples lie in [0,1]
            assert samp.min() >= 0.0 and samp.max() <= 1.0
            # * marginal histograms should be roughly uniform
            counts_u = torch.histc(samp[:, 0], bins=10, min=0, max=1)
            counts_v = torch.histc(samp[:, 1], bins=10, min=0, max=1)
            # each bin ~200 ± 5 σ  (σ≈√(N·p·(1−p))≈√(2000·0.1·0.9)≈13.4)
            assert counts_u.std() < 20
            assert counts_v.std() < 20


def test_internal_buffers_and_flags(bicop_pair):
    _, _, _, U, bc_fast, bc_tll = bicop_pair
    for cop, expect_tll in [(bc_fast, False), (bc_tll, True)]:
        print(cop)
        assert not cop.is_indep
        assert cop.is_tll is expect_tll
        assert cop.num_obs == U.shape[0]
        # all the pre‐computed grids are the right shape
        m = cop.num_step_grid
        for name in ("_pdf_grid", "_cdf_grid", "_hfunc_l_grid", "_hfunc_r_grid"):
            grid = getattr(cop, name)
            assert grid.shape == (m, m)


def test_tau_estimation(bicop_pair):
    _, _, _, U, bc_fast, bc_tll = bicop_pair
    # re‐fit with tau estimation
    bc = tvc.BiCop(num_step_grid=64)
    bc.fit(U, is_tll=True, is_tau_est=True)
    # kendalltau must be nonzero for dependent data
    assert bc.tau[0].abs().item() > 0
    assert bc.tau[1].abs().item() >= 0


def test_sample_shape_and_dtype_on_tll(bicop_pair):
    _, _, _, U, bc_fast, bc_tll = bicop_pair
    for cop in (bc_fast, bc_tll):
        s = cop.sample(123, seed=7, is_sobol=True)
        assert s.shape == (123, 2)
        assert s.dtype is cop.dtype
        assert s.device == cop.device


def test_imshow_and_plot_api(bicop_pair):
    family, params, rotation, U, bc_fast, bc_tll = bicop_pair
    cop = bc_fast
    # imshow
    fig, ax = cop.imshow(is_log_pdf=True)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    # contour
    fig2, ax2 = cop.plot(plot_type="contour", margin_type="unif")
    assert isinstance(fig2, matplotlib.figure.Figure)
    assert isinstance(ax2, matplotlib.axes.Axes)
    fig2, ax2 = cop.plot(plot_type="contour", margin_type="norm")
    assert isinstance(fig2, matplotlib.figure.Figure)
    assert isinstance(ax2, matplotlib.axes.Axes)
    # surface
    fig3, ax3 = cop.plot(plot_type="surface", margin_type="unif")
    assert isinstance(fig3, matplotlib.figure.Figure)
    assert isinstance(ax3, matplotlib.axes.Axes)
    fig3, ax3 = cop.plot(plot_type="surface", margin_type="norm")
    assert isinstance(fig3, matplotlib.figure.Figure)
    assert isinstance(ax3, matplotlib.axes.Axes)
    # invalid args
    with pytest.raises(ValueError):
        cop.plot(plot_type="foo")
    with pytest.raises(ValueError):
        cop.plot(margin_type="bar")


def test_plot_accepts_unused_kwargs(bicop_pair):
    _, _, _, U, bc_fast, _ = bicop_pair
    # just ensure it doesn’t crash
    bc_fast.plot(plot_type="contour", margin_type="norm", xylim=(0, 1), grid_size=50)
    bc_fast.plot(plot_type="surface", margin_type="unif", xylim=(0, 1), grid_size=20)


def test_reset_and_str(bicop_pair):
    # ! notice scope="module" so we put this test at the end
    family, params, rotation, U, bc_fast, bc_tll = bicop_pair
    for cop in (bc_fast, bc_tll):
        cop.reset()
        # should go back to independent
        assert cop.is_indep
        assert cop.num_obs == 0
        # __str__ contains key fields
        s = str(cop)
        assert "is_indep" in s and "num_obs" in s and "is_tll" in s


@pytest.mark.parametrize("method", ["constant", "linear", "quadratic"])
def test_tll_methods_do_not_crash(U_tensor, method):
    cop = tvc.BiCop(num_step_grid=32)
    # should _not_ raise for any of the valid nonparametric_method names
    cop.fit(U_tensor, is_tll=True, mtd_tll=method)


def test_fit_invalid_method_raises(U_tensor):
    cop = tvc.BiCop(num_step_grid=32)
    with pytest.raises(RuntimeError):
        # pick something bogus
        cop.fit(U_tensor, is_tll=True, mtd_tll="no_such_method")


def test_interp_on_trivial_grid():
    # make a BiCop with a 2×2 grid
    bc = tvc.BiCop(num_step_grid=2)
    # override the geometry so that step_grid == 1.0 and target == 1
    bc.step_grid = 1.0
    bc._target = 1.0
    bc._EPS = 0.0  # so we don't get any clamping at the edges

    # grid:
    #   g00 = 0,  g01 = 1
    #   g10 = 2,  g11 = 3
    grid = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)

    # corners should map exactly:
    pts = torch.tensor(
        [
            [0.0, 0.0],  # g00
            [0.0, 1.0],  # g01
            [1.0, 0.0],  # g10
            [1.0, 1.0],  # g11
        ],
        dtype=torch.float64,
    )
    out = bc._interp(grid, pts)
    assert torch.allclose(out, torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64))

    # center point should average correctly:
    center = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    val = bc._interp(grid, center)
    # manual bilinear: 0 + (2−0)*.5 + (1−0)*.5 + (3−1−2+0)*.5*.5 = 1.5
    assert torch.allclose(val, torch.tensor([[1.5]], dtype=torch.float64))

    # if you ask for out-of-bounds it should clamp to [0,1] and then pick the corner:
    # e.g. (−1, −1)→(0,0), (2,3)→(1,1)
    oob = torch.tensor([[-1.0, -1.0], [2.0, 3.0]], dtype=torch.float64)
    val_oob = bc._interp(grid, oob)
    assert torch.allclose(val_oob, torch.tensor([0.0, 3.0], dtype=torch.float64))


def test_imshow_with_existing_axes():
    cop = tvc.BiCop(num_step_grid=32)
    us = torch.rand(100, 2)
    cop.fit(us, is_tll=False)
    fig, outer_ax = plt.subplots()
    fig2, ax2 = cop.imshow(is_log_pdf=False, ax=outer_ax, cmap="viridis")
    # should have returned the same axes object
    assert ax2 is outer_ax


def test_independent_copula_properties():
    for cop in (tvc.BiCop(num_step_grid=64), tvc.BiCop(num_step_grid=16)):
        # before fit, should be independent
        #   CDF(u,v) = u·v,   PDF(u,v)=1,    hfunc_r(u,v)=u,  hfunc_l(u,v)=v
        us = torch.rand(1000, 2, dtype=torch.float64)
        cdf = cop.cdf(us)
        assert torch.allclose(cdf, (us[:, 0] * us[:, 1]).unsqueeze(1))
        pdf = cop.pdf(us)
        assert torch.allclose(pdf, torch.ones_like(pdf))
        logpdf = cop.log_pdf(us)
        assert torch.allclose(logpdf, torch.zeros_like(logpdf))
        hr = cop.hfunc_r(us)
        hl = cop.hfunc_l(us)
        assert torch.allclose(hr, us[:, [0]])
        assert torch.allclose(hl, us[:, [1]])
        hir = cop.hinv_r(us)
        hil = cop.hinv_l(us)
        assert torch.allclose(hir, us[:, [0]])
        assert torch.allclose(hil, us[:, [1]])
