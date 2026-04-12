"""Bicop backend comparison example script."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.special import ndtr

import torchvinecopulib as tvc
from torchvinecopulib.util import _EPS

try:
    from examples._example_utils import configure_matplotlib, export_figure
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _example_utils import configure_matplotlib, export_figure


def main() -> dict:
    """Compare bicop backends and export a density-surface comparison figure."""
    configure_matplotlib()
    torch.manual_seed(7)
    device = "cpu"
    dtype = torch.float64

    rho = 0.7
    latent = torch.randn(size=(768, 2), dtype=dtype)
    latent[:, 1] = rho * latent[:, 0] + math.sqrt(1 - rho**2) * latent[:, 1]
    obs = ndtr(latent).clamp(_EPS, 1 - _EPS).to(device)
    train = obs[:512]
    holdout = obs[512:]

    backend_names = ("grid_reflect", "tll2nn", "ttpi", "beta")
    models: dict[str, tvc.BiCop] = {}
    rows: list[dict[str, float]] = []
    for backend in backend_names:
        bicop = tvc.BiCop(num_step_grid=64).to(device)
        bicop.fit(train, bicop_backend=backend)
        models[backend] = bicop
        rows.append(
            {
                "backend": backend,
                "mean_log_pdf": float(bicop.log_pdf(holdout).mean()),
                "tau": float(torch.as_tensor(bicop.tau).reshape(-1)[0].cpu()),
            }
        )

    grid = torch.linspace(_EPS, 1 - _EPS, 80, dtype=dtype, device=device)
    mesh_u, mesh_v = torch.meshgrid(grid, grid, indexing="ij")
    grid_obs = torch.stack((mesh_u.reshape(-1), mesh_v.reshape(-1)), dim=1)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for ax, backend in zip(axes.flat, backend_names, strict=True):
        pdf = models[backend].pdf(grid_obs).reshape(mesh_u.shape).cpu()
        contour = ax.contourf(
            mesh_u.cpu(),
            mesh_v.cpu(),
            pdf,
            levels=16,
            cmap="viridis",
        )
        ax.set_title(backend)
        ax.set_xlabel("u1")
        ax.set_ylabel("u2")
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.9, label="estimated density")

    path = export_figure(fig, "bicop_backend_comparison.svg")
    plt.close(fig)
    result_df = pd.DataFrame(rows).sort_values("mean_log_pdf", ascending=False)
    return {
        "figure_path": str(path),
        "best_backend": str(result_df.iloc[0]["backend"]),
        "results": rows,
    }


if __name__ == "__main__":
    print(main())
