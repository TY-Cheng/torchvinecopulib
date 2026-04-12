"""Vine copula example script."""

import math

import matplotlib.pyplot as plt
import torch
from torch.special import ndtr

import torchvinecopulib as tvc
from torchvinecopulib.util import _EPS

try:
    from examples._example_utils import configure_matplotlib, export_figure
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _example_utils import configure_matplotlib, export_figure


def main() -> dict:
    """Fit a vine copula and export the learned structure figure."""
    configure_matplotlib()
    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float64

    num_dim = 6
    rho = 0.9
    base = torch.randn(size=(2_048, 2), dtype=dtype)
    base[:, 1] = rho * base[:, 0] + math.sqrt(1 - rho**2) * base[:, 1]
    obs_mvcp = ndtr(base).clamp(_EPS, 1 - _EPS)
    for _ in range(1 + num_dim // 2):
        idx = torch.randperm(obs_mvcp.shape[0])
        obs_mvcp = torch.hstack([obs_mvcp, obs_mvcp[idx, :2]])
    obs = obs_mvcp[:, :num_dim].to(device)

    vine = tvc.VineCop(
        num_dim=num_dim,
        is_cop_scale=True,
        num_step_grid=64,
    ).to(device)
    vine.fit(
        obs=obs,
        mtd_vine="rvine",
        mtd_bidep="kendall_tau",
        first_tree_vertex=(0, 1),
        bicop_backend="beta",
    )

    summary = {
        "num_bicops": len(vine.bicops),
        "mean_log_pdf": float(vine.log_pdf(obs[:256]).mean()),
        "cdf_mean": float(vine.cdf(obs[:64]).mean()),
        "sample_shape": tuple(vine.sample(num_sample=128, seed=0).shape),
    }

    vine.draw_dag()
    fig = plt.gcf()
    path = export_figure(fig, "vinecop_structure.svg")
    plt.close(fig)
    summary["figure_path"] = str(path)
    return summary


if __name__ == "__main__":
    print(main())
