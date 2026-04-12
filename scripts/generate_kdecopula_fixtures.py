from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.stats
import torch

import torchvinecopulib as tvc


METHODS = ("ttcv", "ttpi", "tll1", "tll2", "tll1nn", "tll2nn", "beta")
DEFAULT_OUTPUT = Path("tests/fixtures/aligned_bicop/locked_regression.json")


def _reference_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260412)
    rho = 0.61
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    z = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=96)
    obs = scipy.stats.norm.cdf(z)
    obs = np.clip(obs, 1e-6, 1.0 - 1e-6)
    eval_points = np.array(
        [
            [0.07, 0.09],
            [0.12, 0.74],
            [0.21, 0.41],
            [0.33, 0.87],
            [0.47, 0.52],
            [0.61, 0.18],
            [0.76, 0.69],
            [0.91, 0.93],
        ],
        dtype=np.float64,
    )
    return obs, eval_points


def _fit_kwargs(method: str) -> dict[str, object]:
    if method in {"ttcv", "ttpi"}:
        return {"bandwidth": "auto", "mult": 1.0}
    if method == "beta":
        return {"bandwidth": "auto", "mult": 1.0}
    return {"bandwidth": "auto", "mult": 1.0}


def _canonical_bw_payload(method: str, backend_config: dict[str, object]) -> dict[str, object]:
    payload = {
        "bandwidth_kind": backend_config["bandwidth_kind"],
        "mult": float(backend_config.get("mult", 1.0)),
    }
    if method in {"ttcv", "ttpi"}:
        payload["bw"] = [float(x) for x in backend_config["tt_params"]]
    elif method in {"tll1", "tll2"}:
        payload["bw"] = backend_config["bw"]
    elif method in {"tll1nn", "tll2nn"}:
        payload["bw"] = backend_config["bw"]
    elif method == "beta":
        payload["bw"] = float(backend_config["bw"])
    else:
        raise ValueError(f"Unsupported method: {method}")
    return payload


def _torch_payload() -> dict[str, object]:
    obs_np, eval_np = _reference_dataset()
    obs = torch.as_tensor(obs_np, dtype=torch.float64)
    eval_points = torch.as_tensor(eval_np, dtype=torch.float64)
    payload = {
        "schema_version": 1,
        "fixture_family": "aligned_bicop",
        "provenance": {
            "mode": "torch-bootstrap",
            "note": (
                "Locked regression fixtures generated from the current torch-native aligned "
                "implementation. Use --source r-kdecopula on a machine with R + kdecopula to "
                "refresh the same payload from the external reference implementation."
            ),
        },
        "dataset": {
            "obs": obs_np.tolist(),
            "eval_points": eval_np.tolist(),
            "num_step_grid": 65,
        },
        "methods": {},
    }
    tolerances = {
        "ttcv": {"bw_atol": 5e-4, "value_atol": 7.5e-3},
        "ttpi": {"bw_atol": 5e-4, "value_atol": 7.5e-3},
        "tll1": {"bw_atol": 1e-3, "value_atol": 1e-2},
        "tll2": {"bw_atol": 1e-3, "value_atol": 1e-2},
        "tll1nn": {"bw_atol": 1e-3, "value_atol": 1e-2},
        "tll2nn": {"bw_atol": 1e-3, "value_atol": 1e-2},
        "beta": {"bw_atol": 5e-4, "value_atol": 7.5e-3},
    }
    for method in METHODS:
        cop = tvc.BiCop(num_step_grid=65)
        cop.fit(obs, bicop_backend=method, bicop_kwargs=_fit_kwargs(method))
        payload["methods"][method] = {
            "fit_kwargs": _fit_kwargs(method),
            "canonical": _canonical_bw_payload(method, cop.backend_config),
            "pdf": [float(x) for x in cop.pdf(eval_points).reshape(-1).tolist()],
            "cdf": [float(x) for x in cop.cdf(eval_points).reshape(-1).tolist()],
            "tolerances": tolerances[method],
        }
    return payload


def _r_driver() -> str:
    return r"""
args <- commandArgs(trailingOnly = TRUE)
dataset_path <- args[[1]]
output_path <- args[[2]]
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(kdecopula))
dataset <- fromJSON(dataset_path)
obs <- as.matrix(dataset$dataset$obs)
eval_points <- as.matrix(dataset$dataset$eval_points)
methods <- names(dataset$methods)
serialize_bw <- function(x) {
  if (is.matrix(x)) {
    return(unclass(split(x, row(x))))
  }
  if (is.list(x)) {
    return(x)
  }
  return(unname(x))
}
for (method in methods) {
  fit <- kdecop(obs, method = toupper(method))
  dataset$methods[[method]]$canonical <- list(
    bandwidth_kind = "reference",
    mult = 1.0,
    bw = serialize_bw(fit$bw)
  )
  dataset$methods[[method]]$pdf <- unname(dkdecop(eval_points, fit))
  dataset$methods[[method]]$cdf <- unname(pkdecop(eval_points, fit))
}
dataset$provenance <- list(
  mode = "r-kdecopula",
  note = "Generated with kdecopula through Rscript."
)
write_json(dataset, output_path, pretty = TRUE, auto_unbox = TRUE, digits = NA)
"""


def _r_payload(output: Path) -> None:
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript was not found; use --source torch-bootstrap or install R.")
    base = _torch_payload()
    with tempfile.TemporaryDirectory(prefix="kdecopula-fixtures-") as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_path = tmp_path / "dataset.json"
        script_path = tmp_path / "generate.R"
        dataset_path.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        script_path.write_text(_r_driver(), encoding="utf-8")
        subprocess.run(
            ["Rscript", str(script_path), str(dataset_path), str(output)],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate locked regression fixtures for kdecopula-aligned torch-native backends. "
            "With --source r-kdecopula, the payload becomes an external-reference fixture."
        )
    )
    parser.add_argument(
        "--source",
        choices=("torch-bootstrap", "r-kdecopula"),
        default="torch-bootstrap",
        help="Fixture source. 'r-kdecopula' requires Rscript + kdecopula.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON fixture path.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.source == "torch-bootstrap":
        payload = _torch_payload()
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        _r_payload(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
