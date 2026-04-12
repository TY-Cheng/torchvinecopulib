#!/usr/bin/env python3
"""Run lightweight example scripts and verify their exported docs assets."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "_static" / "examples"
SCRIPT_MODULES = (
    "examples.0_bicop",
    "examples.1_vinecop",
    "examples.2_num_hfunc",
    "examples.3_bicop_tll",
)
EXPORTED_FIGURES = (
    "bicop_sample.svg",
    "vinecop_structure.svg",
    "num_hfunc.svg",
    "bicop_backend_comparison.svg",
)


def compare_expected_bytes(actual_path: Path, expected_path: Path) -> str | None:
    if not expected_path.exists():
        return f"missing figure: {expected_path}"
    if actual_path.read_bytes() != expected_path.read_bytes():
        return f"drift in figure: {expected_path}"
    return None


def run_module(module_name: str) -> None:
    module = __import__(module_name, fromlist=["main"])
    if not hasattr(module, "main"):
        raise AttributeError(f"{module_name} does not expose main()")
    module.main()


def run_once(check: bool) -> int:
    figure_root = DOCS_FIGURES_DIR
    drift: list[str] = []
    temp_root: tempfile.TemporaryDirectory[str] | None = None
    if check:
        temp_root = tempfile.TemporaryDirectory()
        figure_root = Path(temp_root.name) / "figures"
    previous = os.environ.get("TVC_EXAMPLE_FIGURE_DIR")
    os.environ["TVC_EXAMPLE_FIGURE_DIR"] = str(figure_root)
    try:
        for module_name in SCRIPT_MODULES:
            run_module(module_name)
        if check:
            for figure_name in EXPORTED_FIGURES:
                drift_msg = compare_expected_bytes(
                    figure_root / figure_name,
                    DOCS_FIGURES_DIR / figure_name,
                )
                if drift_msg:
                    drift.append(drift_msg)
    finally:
        if previous is None:
            os.environ.pop("TVC_EXAMPLE_FIGURE_DIR", None)
        else:
            os.environ["TVC_EXAMPLE_FIGURE_DIR"] = previous
        if temp_root is not None:
            temp_root.cleanup()
    if drift:
        print("\n".join(drift), file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed figures drift.")
    args = parser.parse_args()
    return run_once(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
