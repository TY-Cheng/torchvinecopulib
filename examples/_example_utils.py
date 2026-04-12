"""Utilities shared by lightweight example scripts."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_FIGURES_DIR = Path(
    os.environ.get(
        "TVC_EXAMPLE_FIGURE_DIR",
        PROJECT_ROOT / "docs" / "_static" / "examples",
    )
).resolve()


def configure_matplotlib() -> None:
    """Set deterministic plot defaults for example asset generation."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "torchvinecopulib",
        }
    )


def docs_figure_path(filename: str) -> Path:
    """Return the canonical docs figure path for an exported example asset."""
    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return DOCS_FIGURES_DIR / filename


def export_figure(fig: plt.Figure, filename: str) -> Path:
    """Save an example figure into the docs static asset directory."""
    path = docs_figure_path(filename)
    fmt = path.suffix.lstrip(".")
    metadata = {"Date": None, "Creator": "torchvinecopulib"}
    fig.savefig(path, format=fmt, bbox_inches="tight", metadata=metadata)
    return path
