"""Bivariate copula example script."""

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
    """Fit a bicop and export the sample-comparison figure."""
    configure_matplotlib()
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float64

    rho = 0.8
    latent = torch.randn(size=(2_000, 2), dtype=dtype)
    latent[:, 1] = rho * latent[:, 0] + math.sqrt(1 - rho**2) * latent[:, 1]
    obs = ndtr(latent).clamp(_EPS, 1 - _EPS).to(device)

    bicop = tvc.BiCop(num_step_grid=64).to(device)
    bicop.fit(obs, bicop_backend="beta")

    probe = torch.tensor(
        [[0.2, 0.3], [0.5, 0.5], [0.8, 0.75]],
        dtype=dtype,
        device=device,
    )
    summary = {
        "tau": float(torch.as_tensor(bicop.tau).reshape(-1)[0].cpu()),
        "mean_log_pdf": float(bicop.log_pdf(obs[:256]).mean()),
        "probe_pdf": bicop.pdf(probe).flatten().tolist(),
        "probe_hfunc_r": bicop.hfunc_r(probe).flatten().tolist(),
    }

    sample = bicop.sample(2_000, seed=1).cpu()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    axes[0].scatter(obs[:, 0].cpu(), obs[:, 1].cpu(), s=6, alpha=0.25, color="tab:blue")
    axes[0].set_title("Training observations")
    axes[1].scatter(sample[:, 0], sample[:, 1], s=6, alpha=0.25, color="tab:orange")
    axes[1].set_title("Samples from fitted BiCop")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("u1")
        ax.set_ylabel("u2")
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    path = export_figure(fig, "bicop_sample.svg")
    plt.close(fig)
    summary["figure_path"] = str(path)
    return summary


if __name__ == "__main__":
    print(main())
