from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import torch

import torchvinecopulib as tvc
from torchvinecopulib.backends import DEFAULT_BICOP_BACKEND, PUBLIC_BICOP_BACKENDS


TORCH_NATIVE_BACKENDS = PUBLIC_BICOP_BACKENDS


def _has_reference() -> bool:
    return importlib.util.find_spec("pyvinecopulib") is not None


def _lazy_pv():
    import pyvinecopulib as pv

    return pv


def _available_families() -> tuple[str, ...]:
    if _has_reference():
        return (
            "gaussian",
            "student",
            "clayton",
            "gumbel",
            "frank",
            "joe",
            "rot_clayton",
            "rot_gumbel",
            "mix_sym",
            "mix_asym",
        )
    return ("gaussian",)


@dataclass(frozen=True)
class TruthModel:
    name: str
    description: str
    parameters: dict[str, float | int | str]
    sample: Callable[[np.random.Generator, int], np.ndarray]
    pdf: Callable[[np.ndarray], np.ndarray]
    cdf: Callable[[np.ndarray], np.ndarray]
    hfunc_l: Callable[[np.ndarray], np.ndarray]
    hfunc_r: Callable[[np.ndarray], np.ndarray]


def _make_parametric_truth(
    family_name: str,
    *,
    rotation: int,
    parameters: np.ndarray,
    description: str,
    meta: dict[str, float | int | str],
) -> TruthModel:
    pv = _lazy_pv()
    family = getattr(pv.BicopFamily, family_name)
    cop = pv.Bicop.from_family(family=family, rotation=rotation, parameters=parameters)
    return TruthModel(
        name=f"{family_name}@{rotation}" if rotation else family_name,
        description=description,
        parameters={**meta, "rotation": int(rotation)},
        sample=lambda local_rng, n: cop.simulate(
            int(n), seeds=[int(local_rng.integers(0, 2**31 - 1))]
        ),
        pdf=lambda u: cop.pdf(np.asarray(u, dtype=np.float64)),
        cdf=lambda u: cop.cdf(np.asarray(u, dtype=np.float64)),
        hfunc_l=lambda u: cop.hfunc1(np.asarray(u, dtype=np.float64)),
        hfunc_r=lambda u: cop.hfunc2(np.asarray(u, dtype=np.float64)),
    )


def _make_gaussian_truth(rng: np.random.Generator) -> TruthModel:
    rho = float(rng.uniform(-0.92, 0.92))
    if _has_reference():
        return _make_parametric_truth(
            "gaussian",
            rotation=0,
            parameters=np.array([[rho]], dtype=np.float64),
            description="Randomized Gaussian copula.",
            meta={"rho": rho},
        )
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    mvn = scipy.stats.multivariate_normal(mean=np.zeros(2), cov=cov)

    def _sample(local_rng: np.random.Generator, n: int) -> np.ndarray:
        z = local_rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=int(n))
        return scipy.stats.norm.cdf(z)

    def _z(u: np.ndarray) -> np.ndarray:
        return scipy.stats.norm.ppf(np.clip(np.asarray(u, dtype=np.float64), 1e-12, 1.0 - 1e-12))

    def _pdf(u: np.ndarray) -> np.ndarray:
        z = _z(u)
        num = mvn.pdf(z)
        den = np.prod(scipy.stats.norm.pdf(z), axis=1)
        return num / np.clip(den, 1e-12, None)

    def _cdf(u: np.ndarray) -> np.ndarray:
        z = _z(u)
        return np.asarray([mvn.cdf(row) for row in z], dtype=np.float64)

    def _hfunc_l(u: np.ndarray) -> np.ndarray:
        z = _z(u)
        return scipy.stats.norm.cdf((z[:, 1] - rho * z[:, 0]) / np.sqrt(max(1.0 - rho**2, 1e-12)))

    def _hfunc_r(u: np.ndarray) -> np.ndarray:
        z = _z(u)
        return scipy.stats.norm.cdf((z[:, 0] - rho * z[:, 1]) / np.sqrt(max(1.0 - rho**2, 1e-12)))

    return TruthModel(
        name="gaussian",
        description="Randomized Gaussian copula.",
        parameters={"rho": rho},
        sample=_sample,
        pdf=_pdf,
        cdf=_cdf,
        hfunc_l=_hfunc_l,
        hfunc_r=_hfunc_r,
    )


def _make_student_truth(rng: np.random.Generator) -> TruthModel:
    rho = float(rng.uniform(-0.88, 0.88))
    nu = float(rng.uniform(2.2, 12.0))
    return _make_parametric_truth(
        "student",
        rotation=0,
        parameters=np.array([[rho], [nu]], dtype=np.float64),
        description="Randomized Student t copula.",
        meta={"rho": rho, "nu": nu},
    )


def _make_clayton_truth(rng: np.random.Generator) -> TruthModel:
    theta = float(rng.uniform(0.25, 10.0))
    return _make_parametric_truth(
        "clayton",
        rotation=0,
        parameters=np.array([[theta]], dtype=np.float64),
        description="Randomized Clayton copula.",
        meta={"theta": theta},
    )


def _make_gumbel_truth(rng: np.random.Generator) -> TruthModel:
    theta = float(rng.uniform(1.05, 8.0))
    return _make_parametric_truth(
        "gumbel",
        rotation=0,
        parameters=np.array([[theta]], dtype=np.float64),
        description="Randomized Gumbel copula.",
        meta={"theta": theta},
    )


def _make_frank_truth(rng: np.random.Generator) -> TruthModel:
    theta = float(rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 22.0))
    return _make_parametric_truth(
        "frank",
        rotation=0,
        parameters=np.array([[theta]], dtype=np.float64),
        description="Randomized Frank copula.",
        meta={"theta": theta},
    )


def _make_joe_truth(rng: np.random.Generator) -> TruthModel:
    theta = float(rng.uniform(1.05, 8.0))
    return _make_parametric_truth(
        "joe",
        rotation=0,
        parameters=np.array([[theta]], dtype=np.float64),
        description="Randomized Joe copula.",
        meta={"theta": theta},
    )


def _make_rot_clayton_truth(rng: np.random.Generator) -> TruthModel:
    theta = float(rng.uniform(0.25, 12.0))
    rotation = int(rng.choice([90, 180, 270]))
    return _make_parametric_truth(
        "clayton",
        rotation=rotation,
        parameters=np.array([[theta]], dtype=np.float64),
        description="Rotated Clayton corner/tail concentration scenario.",
        meta={"theta": theta},
    )


def _make_rot_gumbel_truth(rng: np.random.Generator) -> TruthModel:
    theta = float(rng.uniform(1.05, 10.0))
    rotation = int(rng.choice([90, 180, 270]))
    return _make_parametric_truth(
        "gumbel",
        rotation=rotation,
        parameters=np.array([[theta]], dtype=np.float64),
        description="Rotated Gumbel corner/tail concentration scenario.",
        meta={"theta": theta},
    )


def _mix_truth(
    left: TruthModel,
    right: TruthModel,
    *,
    weight_left: float,
    name: str,
    description: str,
) -> TruthModel:
    def _sample(local_rng: np.random.Generator, n: int) -> np.ndarray:
        selector = local_rng.random(int(n)) < weight_left
        out = np.empty((int(n), 2), dtype=np.float64)
        n_left = int(selector.sum())
        n_right = int((~selector).sum())
        if n_left:
            out[selector] = left.sample(local_rng, n_left)
        if n_right:
            out[~selector] = right.sample(local_rng, n_right)
        return out

    def _weighted(fn_left, fn_right, u: np.ndarray) -> np.ndarray:
        arr = np.asarray(u, dtype=np.float64)
        return weight_left * fn_left(arr) + (1.0 - weight_left) * fn_right(arr)

    return TruthModel(
        name=name,
        description=description,
        parameters={
            "weight_left": float(weight_left),
            "left": left.name,
            "right": right.name,
        },
        sample=_sample,
        pdf=lambda u: _weighted(left.pdf, right.pdf, u),
        cdf=lambda u: _weighted(left.cdf, right.cdf, u),
        hfunc_l=lambda u: _weighted(left.hfunc_l, right.hfunc_l, u),
        hfunc_r=lambda u: _weighted(left.hfunc_r, right.hfunc_r, u),
    )


def _make_mix_sym_truth(rng: np.random.Generator) -> TruthModel:
    left = _make_gaussian_truth(rng)
    right = _make_gumbel_truth(rng)
    weight_left = float(rng.uniform(0.35, 0.65))
    return _mix_truth(
        left,
        right,
        weight_left=weight_left,
        name="mix_sym",
        description="Two-regime symmetric mixture of Gaussian and Gumbel copulas.",
    )


def _make_mix_asym_truth(rng: np.random.Generator) -> TruthModel:
    left = _make_rot_clayton_truth(rng)
    right = _make_frank_truth(rng)
    weight_left = float(rng.uniform(0.15, 0.85))
    return _mix_truth(
        left,
        right,
        weight_left=weight_left,
        name="mix_asym",
        description="Asymmetric rotated-tail mixture for a deliberately awkward synthetic truth.",
    )


def _family_factories() -> dict[str, Callable[[np.random.Generator], TruthModel]]:
    return {
        "gaussian": _make_gaussian_truth,
        "student": _make_student_truth,
        "clayton": _make_clayton_truth,
        "gumbel": _make_gumbel_truth,
        "frank": _make_frank_truth,
        "joe": _make_joe_truth,
        "rot_clayton": _make_rot_clayton_truth,
        "rot_gumbel": _make_rot_gumbel_truth,
        "mix_sym": _make_mix_sym_truth,
        "mix_asym": _make_mix_asym_truth,
    }


def _parse_args() -> argparse.Namespace:
    families = _available_families()
    parser = argparse.ArgumentParser(
        description="Benchmark 2D bicop backends on randomized synthetic copula truths."
    )
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[256, 1024, 4096, 16384])
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=[33, 65, 129])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-size", type=int, default=512)
    parser.add_argument("--eval-grid-size", type=int, default=65)
    parser.add_argument("--tail-prob", type=float, default=0.05)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--families", nargs="+", choices=families, default=list(families))
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=list(TORCH_NATIVE_BACKENDS) + (["tll_ref"] if _has_reference() else []),
        default=list(TORCH_NATIVE_BACKENDS) + (["tll_ref"] if _has_reference() else []),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/compare_bicop_backends.bench.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("benchmarks/results/compare_bicop_backends.summary.md"),
    )
    return parser.parse_args()


def _environment(device: torch.device) -> dict[str, str | bool]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_type": device.type,
        "reference_available": _has_reference(),
        "default_bicop_backend": DEFAULT_BICOP_BACKEND,
    }


def _torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=device)


def _backend_kwargs(backend_name: str, grid_size: int) -> dict[str, object] | None:
    if backend_name in {"grid_reflect", "grid_probit", "tll1", "tll2"}:
        return (
            {"bandwidth": "auto", "mult": 1.0}
            if backend_name.startswith("tll")
            else {"bandwidth": "silverman"}
        )
    if backend_name in {"tll1nn", "tll2nn"}:
        return {"bandwidth": "auto", "mult": 1.0, "nn_k": max(24, grid_size // 2)}
    if backend_name == "beta":
        return {"bandwidth": "auto", "mult": 1.0}
    if backend_name == "beta_qt":
        return {"bandwidth": 0.06, "transform_shape": 2.2}
    if backend_name in {"ttcv", "ttpi"}:
        return {
            "bandwidth": "auto",
            "mult": 1.0,
            "selector_grid_size": 9,
            "selector_num_refine": 2,
            "selector_sample_cap": 512,
        }
    if backend_name == "spline_pen":
        return {"num_basis": max(9, min(17, grid_size // 4 + 1)), "penalty": 1e-2}
    return None


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x) - np.asarray(y)))))


def _iae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(x) - np.asarray(y))))


def _measure_seconds(fn: Callable[[], object]) -> tuple[object, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _tail_corner_error(
    cop: tvc.BiCop, truth: TruthModel, *, tail_prob: float, device: torch.device
) -> float:
    q = float(tail_prob)
    pts = np.asarray(
        [
            [q, q],
            [q, 1.0 - q],
            [1.0 - q, q],
            [1.0 - q, 1.0 - q],
            [0.5 * q, 0.5 * q],
            [1.0 - 0.5 * q, 1.0 - 0.5 * q],
        ],
        dtype=np.float64,
    )
    truth_cdf = truth.cdf(pts)
    fitted_cdf = cop.cdf(_torch(pts, device)).detach().cpu().numpy().reshape(-1)
    return float(np.max(np.abs(truth_cdf - fitted_cdf)))


def _backend_summary_rows(payload: dict[str, object]) -> list[tuple[str, dict[str, float]]]:
    return [
        (backend_name, stats)
        for backend_name, stats in payload["summary"]["backend_scores"].items()
    ]


def _score_lower_is_better(values: dict[str, float]) -> dict[str, float]:
    lo = min(values.values())
    hi = max(values.values())
    if abs(hi - lo) < 1e-12:
        return {key: 1.0 for key in values}
    return {key: (hi - value) / (hi - lo) for key, value in values.items()}


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    keys = [
        "fit_seconds",
        "pdf_seconds",
        "cdf_seconds",
        "hfunc_seconds",
        "hinv_seconds",
        "heldout_mean_logpdf",
        "oracle_mean_logpdf",
        "heldout_logpdf_gap",
        "pdf_iae",
        "pdf_rmse",
        "cdf_rmse",
        "hfunc_l_rmse",
        "hfunc_r_rmse",
        "hinv_roundtrip_rmse",
        "tail_corner_error",
        "itp_failures",
        "bisect_refinements",
        "fallback_to_indep",
        "max_abs_hfunc_error",
        "fit_failed",
        "nonfinite",
    ]
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(run[key]) for run in runs], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def _composite_scores(results: dict[str, dict[str, object]]) -> dict[str, object]:
    native = {k: v for k, v in results.items() if k in TORCH_NATIVE_BACKENDS}
    accuracy_gap = {
        name: float(v["summary"]["heldout_logpdf_gap"]["mean"]) for name, v in native.items()
    }
    accuracy_pdf = {name: float(v["summary"]["pdf_rmse"]["mean"]) for name, v in native.items()}
    accuracy_h = {
        name: 0.5
        * (
            float(v["summary"]["hfunc_l_rmse"]["mean"])
            + float(v["summary"]["hfunc_r_rmse"]["mean"])
        )
        for name, v in native.items()
    }
    stability_fail = {
        name: float(v["summary"]["fit_failed"]["mean"]) + float(v["summary"]["nonfinite"]["mean"])
        for name, v in native.items()
    }
    stability_roundtrip = {
        name: float(v["summary"]["hinv_roundtrip_rmse"]["mean"]) for name, v in native.items()
    }
    stability_fallback = {
        name: float(v["summary"]["fallback_to_indep"]["mean"])
        + float(v["summary"]["itp_failures"]["mean"])
        for name, v in native.items()
    }
    speed_fit = {name: float(v["summary"]["fit_seconds"]["mean"]) for name, v in native.items()}
    speed_query = {
        name: float(v["summary"]["pdf_seconds"]["mean"])
        + float(v["summary"]["cdf_seconds"]["mean"])
        + float(v["summary"]["hfunc_seconds"]["mean"])
        + float(v["summary"]["hinv_seconds"]["mean"])
        for name, v in native.items()
    }
    acc_scores = {
        name: np.mean(
            [
                _score_lower_is_better(accuracy_gap)[name],
                _score_lower_is_better(accuracy_pdf)[name],
                _score_lower_is_better(accuracy_h)[name],
            ]
        )
        for name in native
    }
    stab_scores = {
        name: np.mean(
            [
                _score_lower_is_better(stability_fail)[name],
                _score_lower_is_better(stability_roundtrip)[name],
                _score_lower_is_better(stability_fallback)[name],
            ]
        )
        for name in native
    }
    speed_scores = {
        name: np.mean(
            [
                _score_lower_is_better(speed_fit)[name],
                _score_lower_is_better(speed_query)[name],
            ]
        )
        for name in native
    }
    composite = {
        name: float(0.5 * acc_scores[name] + 0.3 * stab_scores[name] + 0.2 * speed_scores[name])
        for name in native
    }
    winner = max(composite, key=composite.get)
    return {
        "winner": winner,
        "backend_scores": {
            name: {
                "accuracy": float(acc_scores[name]),
                "stability": float(stab_scores[name]),
                "speed": float(speed_scores[name]),
                "composite": float(composite[name]),
            }
            for name in sorted(native)
        },
    }


def _run_single(
    *,
    truth: TruthModel,
    backend_name: str,
    train_size: int,
    grid_size: int,
    query_size: int,
    tail_prob: float,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, object]:
    train = truth.sample(rng, train_size)
    test = truth.sample(rng, max(query_size * 2, train_size))
    query = truth.sample(rng, query_size)
    model = tvc.BiCop(num_step_grid=grid_size).to(device)
    backend_kwargs = _backend_kwargs(backend_name, grid_size)
    try:
        _, fit_seconds = _measure_seconds(
            lambda: model.fit(
                _torch(train, device),
                bicop_backend=backend_name,
                bicop_kwargs=backend_kwargs,
            )
        )
        fit_failed = 0
    except Exception as exc:
        return {
            "backend_name": backend_name,
            "fit_seconds": float("inf"),
            "pdf_seconds": float("inf"),
            "cdf_seconds": float("inf"),
            "hfunc_seconds": float("inf"),
            "hinv_seconds": float("inf"),
            "heldout_mean_logpdf": float("-inf"),
            "oracle_mean_logpdf": float("-inf"),
            "heldout_logpdf_gap": float("inf"),
            "pdf_iae": float("inf"),
            "pdf_rmse": float("inf"),
            "cdf_rmse": float("inf"),
            "hfunc_l_rmse": float("inf"),
            "hfunc_r_rmse": float("inf"),
            "hinv_roundtrip_rmse": float("inf"),
            "tail_corner_error": float("inf"),
            "itp_failures": float("inf"),
            "bisect_refinements": float("inf"),
            "fallback_to_indep": float("inf"),
            "max_abs_hfunc_error": float("inf"),
            "fit_failed": 1.0,
            "nonfinite": 1.0,
            "error": str(exc),
            "backend_kwargs": backend_kwargs or {},
            "train_size": int(train_size),
            "grid_size": int(grid_size),
        }
    test_t = _torch(test, device)
    query_t = _torch(query, device)
    truth_pdf_test = truth.pdf(test)
    truth_lp = np.log(np.clip(truth_pdf_test, 1e-12, None))
    fitted_lp, pdf_seconds = _measure_seconds(lambda: model.log_pdf(test_t))
    fitted_lp_np = fitted_lp.detach().cpu().numpy().reshape(-1)
    u_axis = np.linspace(1e-6, 1.0 - 1e-6, grid_size)
    uu, vv = np.meshgrid(u_axis, u_axis, indexing="ij")
    grid_pts = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)
    truth_pdf_grid = truth.pdf(grid_pts).reshape(grid_size, grid_size)
    fitted_pdf_grid = model._pdf_grid.detach().cpu().numpy()
    fitted_cdf, cdf_seconds = _measure_seconds(lambda: model.cdf(query_t))
    truth_cdf = truth.cdf(query)
    fitted_hl, hfunc_l_seconds = _measure_seconds(lambda: model.hfunc_l(query_t))
    fitted_hr, hfunc_r_seconds = _measure_seconds(lambda: model.hfunc_r(query_t))
    truth_hl = truth.hfunc_l(query)
    truth_hr = truth.hfunc_r(query)
    hinv_input_l = torch.hstack([query_t[:, [0]], fitted_hl])
    hinv_input_r = torch.hstack([fitted_hr, query_t[:, [1]]])
    recovered_l, hinv_l_seconds = _measure_seconds(lambda: model.hinv_l(hinv_input_l))
    recovered_r, hinv_r_seconds = _measure_seconds(lambda: model.hinv_r(hinv_input_r))
    diag = model.diagnostics()
    nonfinite = int(
        not (
            torch.isfinite(model._pdf_grid).all()
            and torch.isfinite(model._cdf_grid).all()
            and torch.isfinite(model._hfunc_l_grid).all()
            and torch.isfinite(model._hfunc_r_grid).all()
        )
    )
    return {
        "backend_name": backend_name,
        "fit_seconds": float(fit_seconds),
        "pdf_seconds": float(pdf_seconds),
        "cdf_seconds": float(cdf_seconds),
        "hfunc_seconds": float(hfunc_l_seconds + hfunc_r_seconds),
        "hinv_seconds": float(hinv_l_seconds + hinv_r_seconds),
        "heldout_mean_logpdf": float(fitted_lp_np.mean()),
        "oracle_mean_logpdf": float(truth_lp.mean()),
        "heldout_logpdf_gap": float(truth_lp.mean() - fitted_lp_np.mean()),
        "pdf_iae": _iae(fitted_pdf_grid, truth_pdf_grid),
        "pdf_rmse": _rmse(fitted_pdf_grid, truth_pdf_grid),
        "cdf_rmse": _rmse(fitted_cdf.detach().cpu().numpy().reshape(-1), truth_cdf),
        "hfunc_l_rmse": _rmse(fitted_hl.detach().cpu().numpy().reshape(-1), truth_hl),
        "hfunc_r_rmse": _rmse(fitted_hr.detach().cpu().numpy().reshape(-1), truth_hr),
        "hinv_roundtrip_rmse": float(
            0.5
            * (
                _rmse(recovered_l.detach().cpu().numpy().reshape(-1), query[:, 1])
                + _rmse(recovered_r.detach().cpu().numpy().reshape(-1), query[:, 0])
            )
        ),
        "tail_corner_error": _tail_corner_error(
            model,
            truth,
            tail_prob=tail_prob,
            device=device,
        ),
        "itp_failures": float(diag.itp_failures_l + diag.itp_failures_r),
        "bisect_refinements": float(diag.bisect_refinements_l + diag.bisect_refinements_r),
        "fallback_to_indep": float(diag.fallback_to_indep_l + diag.fallback_to_indep_r),
        "max_abs_hfunc_error": float(max(diag.max_abs_hfunc_error_l, diag.max_abs_hfunc_error_r)),
        "fit_failed": float(fit_failed),
        "nonfinite": float(nonfinite),
        "backend_kwargs": backend_kwargs or {},
        "train_size": int(train_size),
        "grid_size": int(grid_size),
    }


def _markdown_summary(payload: dict[str, object]) -> str:
    lines = [
        "# Bicop backend benchmark summary",
        "",
        f"Winner: `{payload['summary']['winner']}`",
        f"Current default backend: `{payload['environment']['default_bicop_backend']}`",
        "",
        "| backend | accuracy | stability | speed | composite |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for backend_name, scores in _backend_summary_rows(payload):
        lines.append(
            f"| `{backend_name}` | {scores['accuracy']:.4f} | {scores['stability']:.4f} | "
            f"{scores['speed']:.4f} | {scores['composite']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    results: dict[str, dict[str, object]] = {}
    scenario_rng = np.random.default_rng(args.seed)
    family_factories = _family_factories()
    for family_name in args.families:
        per_backend: dict[str, object] = {}
        for backend_name in args.backends:
            runs: list[dict[str, object]] = []
            for train_size in args.train_sizes:
                for grid_size in args.grid_sizes:
                    for _ in range(args.repeats):
                        truth = family_factories[family_name](scenario_rng)
                        run = _run_single(
                            truth=truth,
                            backend_name=backend_name,
                            train_size=int(train_size),
                            grid_size=int(grid_size),
                            query_size=int(args.query_size),
                            tail_prob=float(args.tail_prob),
                            device=device,
                            rng=scenario_rng,
                        )
                        run["truth_name"] = truth.name
                        run["truth_description"] = truth.description
                        run["truth_parameters"] = truth.parameters
                        runs.append(run)
            per_backend[backend_name] = {
                "runs": runs,
                "summary": _aggregate_runs(runs),
            }
        results[family_name] = {
            "backends": per_backend,
        }
    combined_runs: dict[str, list[dict[str, object]]] = {}
    for family_report in results.values():
        for backend_name, report in family_report["backends"].items():
            combined_runs.setdefault(backend_name, []).extend(report["runs"])
    summary = _composite_scores(
        {
            backend_name: {"summary": _aggregate_runs(runs)}
            for backend_name, runs in combined_runs.items()
        }
    )
    payload = {
        "schema_version": 1,
        "benchmark": "bicop_backends",
        "config": {
            "train_sizes": [int(x) for x in args.train_sizes],
            "grid_sizes": [int(x) for x in args.grid_sizes],
            "repeats": int(args.repeats),
            "seed": int(args.seed),
            "query_size": int(args.query_size),
            "eval_grid_size": int(args.eval_grid_size),
            "tail_prob": float(args.tail_prob),
            "device": args.device,
            "families": list(args.families),
            "backends": list(args.backends),
        },
        "environment": _environment(device),
        "results": results,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown_summary(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
