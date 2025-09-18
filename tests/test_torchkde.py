import pytest
import torch
import torchvinecopulib as tvc
from scipy.stats import kstest, wasserstein_distance

from . import EPS, bicop_pair


def test_1dkde_internal_grid_is_finite():
    """
    1D KDE must build finite grids even if input contains NaN/±Inf.
    This ensures later interpolation/extrapolation is safe.
    """
    x_clean = torch.randn(2000, dtype=torch.float64)
    x_bad = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=torch.float64)
    x = torch.cat([x_clean, x_bad])

    # Build 1D KDE (adjust path if your class lives elsewhere)
    kde = tvc.kdeCDFPPF1D(x, bandwidth_method="auto")

    # All internal grids must be finite
    for name in ("grid_x", "grid_pdf", "grid_cdf"):
        grid = getattr(kde, name)
        assert torch.isfinite(grid).all(), f"{name} contains non-finite values"

    # Basic validity: pdf ≥ 0; cdf in [0,1] and non-decreasing
    assert (kde.grid_pdf >= -EPS).all()
    assert kde.grid_cdf.min() >= -EPS and kde.grid_cdf.max() <= 1 + EPS
    if kde.grid_cdf.numel() > 1:
        assert kde.grid_cdf.diff().min() >= -EPS


def test_bicop_grids_and_eval_are_finite_with_nan_inf(bicop_pair):
    """
    All BiCop modes (fastKDE, tll, torchKDE) must:
    - build fully finite internal grids even if training data contain NaN/±Inf/OOB,
    - evaluate safely on queries containing NaN/±Inf/OOB:
        * finite rows → finite outputs (and proper ranges),
        * non-finite rows → NaN (no crash).
    """
    _, _, _, U, bc_fast, bc_tll, bc_torch = bicop_pair

    # Inject some bad rows into otherwise valid data and refit fresh models
    bad = torch.tensor(
        [
            [float("nan"), 0.3],
            [0.4, float("inf")],
            [-float("inf"), 0.9],
            [1.1, -0.1],   # OOB but finite
        ],
        dtype=torch.float64,
        device=U.device,
    )
    U_dirty = torch.vstack([U, bad])

    modes = [("fastKDE", bc_fast), ("tll", bc_tll), ("torchKDE", bc_torch)]
    for name, _ in modes:
        cop = tvc.BiCop(num_step_grid=64).to(device=U.device)
        cop.fit(U_dirty, mtd_kde=name)

        # 1) Internal grids must be finite
        for gname in ("_pdf_grid", "_cdf_grid", "_hfunc_l_grid", "_hfunc_r_grid"):
            G = getattr(cop, gname)
            assert torch.isfinite(G).all(), f"{name}:{gname} contains non-finite values"

        # 2) Evaluation must be safe on queries with NaN/±Inf/OOB
        Q = torch.tensor(
            [
                [0.2, 0.7],                 # clean
                [float("nan"), 0.3],        # NaN
                [0.4, float("inf")],        # +Inf
                [-float("inf"), 0.9],       # -Inf
                [0.0, 1.0],                 # edge
                [1.1, -0.1],                # OOB (finite)
            ],
            dtype=torch.float64,
            device=cop.device,
        )
        finite = torch.isfinite(Q).all(dim=1)

        # pdf/log_pdf: finite rows must be finite; pdf non-negative
        pdf = cop.pdf(Q).squeeze(1)
        logp = cop.log_pdf(Q).squeeze(1)
        assert torch.isfinite(pdf[finite]).all()
        assert torch.isfinite(logp[finite]).all()
        assert (pdf[finite] >= -EPS).all()

        # Non-finite rows should come back as NaN (by design) and never crash
        if (~finite).any():
            assert torch.isnan(pdf[~finite]).all()
            assert torch.isnan(logp[~finite]).all()

        # cdf/hfuncs/hinvs: finite rows within [0,1]
        for fn in (cop.cdf, cop.hfunc_r, cop.hfunc_l):
            out = fn(Q).squeeze(1)
            assert (out[finite] >= -EPS).all() and (out[finite] <= 1 + EPS).all()
            if (~finite).any():
                assert torch.isnan(out[~finite]).all()

        # Inverses: finite rows → valid [0,1]; non-finite rows → NaN
        out_r = cop.hinv_r(Q).squeeze(1)
        out_l = cop.hinv_l(Q).squeeze(1)
        assert (out_r[finite] >= -EPS).all() and (out_r[finite] <= 1 + EPS).all()
        assert (out_l[finite] >= -EPS).all() and (out_l[finite] <= 1 + EPS).all()
        if (~finite).any():
            assert torch.isnan(out_r[~finite]).all()
            assert torch.isnan(out_l[~finite]).all()



def test_pit_goodness_of_fit_train_test(bicop_pair):
    """
    Split U into train/test.
    Fit on train; compute PIT on test via hfunc_r.
    PIT should be close to Uniform[0,1] (KS and Wasserstein).
    """

    _, _, _, U, bc_fast, bc_tll, bc_torch = bicop_pair
    n = U.shape[0]
    n_tr = int(0.6 * n)
    Utr, Ute = U[:n_tr], U[n_tr:]

    cases = [
        ("fastKDE", bc_fast),
        ("tll", bc_tll),
        ("torchKDE", bc_torch),
    ]

    for name, _ in cases:
        cop = tvc.BiCop(num_step_grid=64)
        cop.fit(Utr, mtd_kde=name)

        pit = cop.hfunc_r(Ute).squeeze(1).cpu().numpy()  # PIT should be ~ U(0,1)

        # KS vs Uniform(0,1)
        ks_stat, ks_p = kstest(pit, "uniform")

        # 1-Wasserstein vs iid Uniform sample of same size
        uni = torch.rand_like(Ute[:, 0]).cpu().numpy()
        wdist = wasserstein_distance(pit, uni)

        # Lenient but meaningful thresholds (tune if your grids/resolution differ)
        assert ks_stat < 0.12
        assert wdist < 0.06
        assert ks_p > 1e-3
