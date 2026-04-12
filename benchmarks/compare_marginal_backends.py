from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import torch
from scipy import stats

from torchvinecopulib.backends import build_marginal_estimator


BACKENDS = ("grid", "lp")
SCENARIOS = ("normal", "gamma", "beta", "bimodal")


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    support_min: float | None
    support_max: float | None
    parameters: dict[str, float]
    sample: Callable[[np.random.Generator, int], np.ndarray]
    pdf: Callable[[np.ndarray], np.ndarray]
    cdf: Callable[[np.ndarray], np.ndarray]
    ppf: Callable[[np.ndarray], np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the surviving 1D marginal backends on randomized continuous families."
    )
    parser.add_argument("--train-size", type=int, default=4_096)
    parser.add_argument("--test-size", type=int, default=16_384)
    parser.add_argument("--fit-grid-size", type=int, default=257)
    parser.add_argument("--eval-grid-size", type=int, default=1_025)
    parser.add_argument("--quantile-size", type=int, default=511)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tail-prob", type=float, default=1e-4)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/compare_marginal_backends.bench.json"),
    )
    return parser.parse_args()


def _normal_scenario(rng: np.random.Generator) -> Scenario:
    loc = float(rng.uniform(-0.35, 0.35))
    scale = float(rng.uniform(0.75, 1.35))
    dist = stats.norm(loc=loc, scale=scale)
    return Scenario(
        name="normal",
        description="Randomized unbounded Gaussian family.",
        support_min=None,
        support_max=None,
        parameters={"loc": loc, "scale": scale},
        sample=lambda local_rng, n: dist.rvs(size=n, random_state=local_rng),
        pdf=dist.pdf,
        cdf=dist.cdf,
        ppf=dist.ppf,
    )


def _gamma_scenario(rng: np.random.Generator) -> Scenario:
    shape = float(rng.uniform(2.0, 5.0))
    scale = float(rng.uniform(0.45, 1.05))
    dist = stats.gamma(a=shape, scale=scale)
    return Scenario(
        name="gamma",
        description="Randomized left-bounded gamma family.",
        support_min=0.0,
        support_max=None,
        parameters={"shape": shape, "scale": scale},
        sample=lambda local_rng, n: dist.rvs(size=n, random_state=local_rng),
        pdf=dist.pdf,
        cdf=dist.cdf,
        ppf=dist.ppf,
    )


def _beta_scenario(rng: np.random.Generator) -> Scenario:
    a = float(rng.uniform(1.6, 4.8))
    b = float(rng.uniform(2.2, 6.8))
    dist = stats.beta(a=a, b=b)
    return Scenario(
        name="beta",
        description="Randomized beta family on [0, 1].",
        support_min=0.0,
        support_max=1.0,
        parameters={"a": a, "b": b},
        sample=lambda local_rng, n: dist.rvs(size=n, random_state=local_rng),
        pdf=dist.pdf,
        cdf=dist.cdf,
        ppf=dist.ppf,
    )


def _bimodal_scenario(rng: np.random.Generator) -> Scenario:
    weight_left = float(rng.uniform(0.3, 0.7))
    loc_left = float(rng.uniform(-2.4, -1.0))
    loc_right = float(rng.uniform(0.9, 2.3))
    scale_left = float(rng.uniform(0.25, 0.7))
    scale_right = float(rng.uniform(0.35, 0.9))
    left = stats.norm(loc=loc_left, scale=scale_left)
    right = stats.norm(loc=loc_right, scale=scale_right)

    def _pdf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return weight_left * left.pdf(x) + (1.0 - weight_left) * right.pdf(x)

    def _cdf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return weight_left * left.cdf(x) + (1.0 - weight_left) * right.cdf(x)

    def _ppf(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).clip(0.0, 1.0)
        low = np.full_like(q, min(loc_left - 8.0 * scale_left, loc_right - 8.0 * scale_right))
        high = np.full_like(q, max(loc_left + 8.0 * scale_left, loc_right + 8.0 * scale_right))
        for _ in range(80):
            mid = 0.5 * (low + high)
            cdf_mid = _cdf(mid)
            low = np.where(cdf_mid < q, mid, low)
            high = np.where(cdf_mid >= q, mid, high)
        return 0.5 * (low + high)

    def _sample(local_rng: np.random.Generator, n: int) -> np.ndarray:
        selector = local_rng.random(n) < weight_left
        left_sample = left.rvs(size=n, random_state=local_rng)
        right_sample = right.rvs(size=n, random_state=local_rng)
        return np.where(selector, left_sample, right_sample)

    return Scenario(
        name="bimodal",
        description="Randomized unbounded bimodal Gaussian mixture family.",
        support_min=None,
        support_max=None,
        parameters={
            "weight_left": weight_left,
            "loc_left": loc_left,
            "scale_left": scale_left,
            "loc_right": loc_right,
            "scale_right": scale_right,
        },
        sample=_sample,
        pdf=_pdf,
        cdf=_cdf,
        ppf=_ppf,
    )


def _scenario_factories() -> dict[str, Callable[[np.random.Generator], Scenario]]:
    return {
        "normal": _normal_scenario,
        "gamma": _gamma_scenario,
        "beta": _beta_scenario,
        "bimodal": _bimodal_scenario,
    }


def _torch(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64).view(-1, 1)


def _support_bounds(scenario: Scenario, tail_prob: float) -> tuple[float, float]:
    lo = (
        scenario.support_min
        if scenario.support_min is not None
        else float(scenario.ppf(tail_prob))
    )
    hi = (
        scenario.support_max
        if scenario.support_max is not None
        else float(scenario.ppf(1.0 - tail_prob))
    )
    return float(lo), float(hi)


def _backend_kwargs(
    *,
    backend_name: str,
    fit_grid_size: int,
    support_min: float | None,
    support_max: float | None,
) -> dict[str, object]:
    if backend_name == "grid":
        kwargs: dict[str, object] = {
            "num_step_grid": fit_grid_size,
            "bandwidth": "isj",
            "bandwidth_scale": 1.0,
            "smoother": "auto",
            "pad": 0.1,
        }
    else:
        kwargs = {
            "num_step_grid": fit_grid_size,
            "degree": 2,
            "bandwidth": "plugin",
            "bandwidth_scale": 1.0,
        }
    if support_min is not None:
        kwargs["x_min"] = float(support_min)
    if support_max is not None:
        kwargs["x_max"] = float(support_max)
    return kwargs


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x - y))))


def _time_call(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
    }


def _aggregate(runs: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    numeric_keys = [
        "fit_seconds",
        "log_pdf_seconds",
        "cdf_seconds",
        "ppf_seconds",
        "heldout_mean_logpdf",
        "oracle_mean_logpdf",
        "heldout_logpdf_gap",
        "pdf_iae",
        "cdf_rmse",
        "ppf_rmse",
    ]
    out: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = np.asarray([float(run[key]) for run in runs], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def _run_single(
    *,
    scenario_name: str,
    backend_name: str,
    train_size: int,
    test_size: int,
    fit_grid_size: int,
    eval_grid_size: int,
    quantile_size: int,
    tail_prob: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    scenario = _scenario_factories()[scenario_name](rng)
    train = np.asarray(scenario.sample(rng, train_size), dtype=float)
    test = np.asarray(scenario.sample(rng, test_size), dtype=float)
    eval_lo, eval_hi = _support_bounds(scenario, tail_prob=tail_prob)
    eval_x = np.linspace(eval_lo, eval_hi, eval_grid_size, dtype=float)
    q = np.linspace(tail_prob, 1.0 - tail_prob, quantile_size, dtype=float)
    backend_kwargs = _backend_kwargs(
        backend_name=backend_name,
        fit_grid_size=fit_grid_size,
        support_min=scenario.support_min,
        support_max=scenario.support_max,
    )

    t0 = time.perf_counter()
    estimator = build_marginal_estimator(
        backend_name=backend_name,
        x=_torch(train),
        backend_kwargs=backend_kwargs,
    )
    fit_seconds = time.perf_counter() - t0

    with torch.inference_mode():
        log_pdf_t, log_pdf_seconds = _time_call(lambda: estimator.log_pdf(_torch(test)))
        cdf_t, cdf_seconds = _time_call(lambda: estimator.cdf(_torch(eval_x)))
        ppf_t, ppf_seconds = _time_call(lambda: estimator.ppf(_torch(q)))
        pdf_eval = estimator.pdf(_torch(eval_x)).view(-1).cpu().numpy()

    log_pdf_est = log_pdf_t.view(-1).cpu().numpy()
    cdf_est = cdf_t.view(-1).cpu().numpy()
    ppf_est = ppf_t.view(-1).cpu().numpy()
    pdf_true_test = np.clip(scenario.pdf(test), 1e-300, None)
    pdf_true_eval = np.clip(scenario.pdf(eval_x), 1e-300, None)
    cdf_true_eval = scenario.cdf(eval_x)
    ppf_true = scenario.ppf(q)
    oracle_mean_logpdf = float(np.log(pdf_true_test).mean())
    heldout_mean_logpdf = float(log_pdf_est.mean())

    return {
        "seed": int(seed),
        "backend_name": backend_name,
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
        "scenario_parameters": dict(scenario.parameters),
        "fit_seconds": float(fit_seconds),
        "log_pdf_seconds": float(log_pdf_seconds),
        "cdf_seconds": float(cdf_seconds),
        "ppf_seconds": float(ppf_seconds),
        "heldout_mean_logpdf": heldout_mean_logpdf,
        "oracle_mean_logpdf": oracle_mean_logpdf,
        "heldout_logpdf_gap": float(oracle_mean_logpdf - heldout_mean_logpdf),
        "pdf_iae": float(np.trapezoid(np.abs(pdf_eval - pdf_true_eval), x=eval_x)),
        "cdf_rmse": _rmse(cdf_est, cdf_true_eval),
        "ppf_rmse": _rmse(ppf_est, ppf_true),
        "backend_config": estimator.backend_config,
    }


def _summary_table(report: dict[str, object]) -> str:
    lines = [
        "scenario backend  loglik_gap  pdf_iae  cdf_rmse  ppf_rmse  fit_s",
    ]
    results = report["results"]
    assert isinstance(results, dict)
    for scenario_name, scenario_payload in results.items():
        backends = scenario_payload["backends"]
        assert isinstance(backends, dict)
        for backend_name, payload in backends.items():
            summary = payload["summary"]
            lines.append(
                f"{scenario_name:8s} {backend_name:8s} "
                f"{summary['heldout_logpdf_gap']['mean']:.4f} "
                f"{summary['pdf_iae']['mean']:.4f} "
                f"{summary['cdf_rmse']['mean']:.4f} "
                f"{summary['ppf_rmse']['mean']:.4f} "
                f"{summary['fit_seconds']['mean']:.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    results: dict[str, object] = {}
    for scenario_name in args.scenarios:
        backend_results: dict[str, object] = {}
        for backend_name in args.backends:
            runs = [
                _run_single(
                    scenario_name=scenario_name,
                    backend_name=backend_name,
                    train_size=args.train_size,
                    test_size=args.test_size,
                    fit_grid_size=args.fit_grid_size,
                    eval_grid_size=args.eval_grid_size,
                    quantile_size=args.quantile_size,
                    tail_prob=args.tail_prob,
                    seed=args.seed + 10_000 * SCENARIOS.index(scenario_name) + 1_000 * rep,
                )
                for rep in range(args.repeats)
            ]
            first = runs[0]
            assert isinstance(first["scenario_description"], str)
            results.setdefault(
                scenario_name,
                {
                    "description": first["scenario_description"],
                    "backends": {},
                },
            )
            results[scenario_name]["backends"][backend_name] = {
                "summary": _aggregate(runs),
                "runs": runs,
            }

    report = {
        "schema_version": 1,
        "benchmark": "marginal_backends",
        "config": {
            "train_size": args.train_size,
            "test_size": args.test_size,
            "fit_grid_size": args.fit_grid_size,
            "eval_grid_size": args.eval_grid_size,
            "quantile_size": args.quantile_size,
            "repeats": args.repeats,
            "seed": args.seed,
            "tail_prob": args.tail_prob,
            "scenarios": list(args.scenarios),
            "backends": list(args.backends),
            "randomized_scenario_parameters": True,
        },
        "environment": _environment(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(_summary_table(report))
    print()
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
