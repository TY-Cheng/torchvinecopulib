from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pyvinecopulib import BicopFamily

# build & install before test
from torchvinecopulib import bicop
from torchvinecopulib.bicop import SET_FAMnROT, bcp_from_obs
from torchvinecopulib.util import _TAU_MAX, _TAU_MIN

DIR_OUT_TEST = Path(".") / "out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#
LST_MTD_FIT = ["itau", "mle"]
LST_MTD_SEL = ["aic", "bic"]
LST_MTD_BIDEP = [
    "kendall_tau",
    "ferreira_tail_dep_coeff",
    "chatterjee_xi",
    "wasserstein_dist_ind",
    "mutual_info",
]
DCT_FAM = {
    "Clayton": (BicopFamily.clayton, bicop.Clayton),
    "Frank": (BicopFamily.frank, bicop.Frank),
    "Gaussian": (BicopFamily.gaussian, bicop.Gaussian),
    "Gumbel": (BicopFamily.gumbel, bicop.Gumbel),
    "Independent": (BicopFamily.indep, bicop.Independent),
    "Joe": (BicopFamily.joe, bicop.Joe),
    "StudentT": (BicopFamily.student, bicop.StudentT),
}


def sim_from_bcp(
    bcp_tvc,
    par: tuple | None = None,
    seed: int = 0,
    rot: int = 0,
    num_sim: int = 10000,
    device: str = DEVICE,
    dtype=torch.float64,
) -> torch.Tensor:
    """Simulates bivariate copula data."""
    if par is None:
        par = tuple((_ / 10 for _ in bcp_tvc._PAR_MAX))
    return bcp_tvc.sim(
        rot=rot, num_sim=num_sim, seed=seed, device=device, dtype=dtype, par=par
    )


def sim_vcp_from_bcp(
    bcp_tvc,
    num_dim: int = 6,
    num_sim: int = 10000,
    device: str = DEVICE,
    dtype=torch.float64,
) -> torch.Tensor:
    return torch.hstack(
        [
            sim_from_bcp(bcp_tvc, seed=_, num_sim=num_sim, device=device, dtype=dtype)
            for _ in range(num_dim // 2)
        ]
    )


def compare_chart_vec(
    vec_pvc: np.ndarray,
    vec_tvc: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    title: str | None = None,
    label: str | None = None,
    xlabel: str = "Data Points",
    ylabel: str = "Diff",
) -> None | AssertionError:
    """
    chart the differences with proper fig title
    mkdir 'out/test' only when necessary
    """
    if np.allclose(a=vec_pvc, b=vec_tvc, rtol=rtol, atol=atol):
        return None
    else:
        DIR_OUT_TEST.mkdir(parents=True, exist_ok=True)
        _, ax = plt.subplots(figsize=(10, 6))
        pd.Series(vec_pvc - vec_tvc).plot(
            ax=ax, title=title, label=label, grid=True, alpha=0.7, linewidth=0.5
        )
        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        plt.savefig(DIR_OUT_TEST / f"{title}.png")
        plt.close()
        return AssertionError(f"{title} failed!")
