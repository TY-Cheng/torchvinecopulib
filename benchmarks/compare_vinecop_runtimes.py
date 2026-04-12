from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.special import ndtr

import torchvinecopulib as tvc
from torchvinecopulib.backends import DEFAULT_BICOP_BACKEND, default_bicop_kwargs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare end-to-end VineCop runtime across torchvinecopulib CPU/CUDA and "
            "optional pyvinecopulib reference paths."
        )
    )
    parser.add_argument("--num-obs", nargs="+", type=int, default=[1_000, 5_000])
    parser.add_argument("--num-dim", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--grid-size", type=int, default=65)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument("--query-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-reference",
        choices=("auto", "yes", "no"),
        default="auto",
    )
    parser.add_argument(
        "--include-cuda",
        choices=("auto", "yes", "no"),
        default="auto",
    )
    parser.add_argument(
        "--reference-nonparametric-method",
        choices=("constant", "linear", "quadratic"),
        default="quadratic",
    )
    parser.add_argument(
        "--bicop-backend",
        default=DEFAULT_BICOP_BACKEND,
        help="torchvinecopulib bicop backend for the torch-native runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/compare_vinecop_runtimes.bench.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("benchmarks/results/compare_vinecop_runtimes.bench.md"),
    )
    return parser.parse_args()


def _has_reference() -> bool:
    return importlib.util.find_spec("pyvinecopulib") is not None


def _load_reference():
    import pyvinecopulib as pvc

    return pvc


def _resolve_flag(flag: str, *, available: bool, label: str) -> bool:
    if flag == "auto":
        return available
    if flag == "yes" and not available:
        raise RuntimeError(f"{label} was requested but is not available in this environment.")
    return flag == "yes"


def _make_obs(*, num_obs: int, num_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    corr = torch.rand(num_dim, num_dim, dtype=torch.float64, generator=generator)
    corr = corr / corr.norm(dim=1, keepdim=True).clamp_min(1e-12)
    corr = corr @ corr.T
    z = torch.randn(num_obs, num_dim, dtype=torch.float64, generator=generator)
    return ndtr(z @ torch.linalg.cholesky(corr, upper=True))


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn, *, sync_cuda: bool = False) -> float:
    if sync_cuda:
        _sync_cuda()
    t0 = time.perf_counter()
    fn()
    if sync_cuda:
        _sync_cuda()
    return time.perf_counter() - t0


def _summarize(times: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(times)),
        "min": float(min(times)),
        "max": float(max(times)),
        "std": float(statistics.pstdev(times) if len(times) > 1 else 0.0),
    }


def _measure_repeated(
    fn, *, repeats: int, warmup: int, sync_cuda: bool = False
) -> dict[str, object]:
    for _ in range(max(0, int(warmup))):
        fn()
        if sync_cuda:
            _sync_cuda()
    runs = [_time_call(fn, sync_cuda=sync_cuda) for _ in range(max(1, int(repeats)))]
    return {
        "runs": [float(x) for x in runs],
        "summary": _summarize(runs),
    }


def _environment(include_reference: bool, include_cuda: bool) -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "reference_available": _has_reference(),
        "default_bicop_backend": DEFAULT_BICOP_BACKEND,
        "include_reference": include_reference,
        "include_cuda": include_cuda,
    }


def _benchmark_pvc(
    *,
    obs: torch.Tensor,
    repeats: int,
    warmup: int,
    sample_size: int,
    query_size: int,
    reference_nonparametric_method: str,
) -> dict[str, object]:
    pvc = _load_reference()
    obs_np = obs.cpu().numpy().astype("float64")
    query_np = obs_np[: min(query_size, obs_np.shape[0])]
    controls = pvc.FitControlsVinecop(
        family_set=(pvc.BicopFamily.indep, pvc.BicopFamily.tll),
        nonparametric_method=reference_nonparametric_method,
        tree_criterion="tau",
        num_threads=1,
    )
    holder: dict[str, object] = {}

    def _fit_once():
        holder["model"] = pvc.Vinecop.from_data(data=obs_np, controls=controls)

    fit = _measure_repeated(_fit_once, repeats=repeats, warmup=warmup)
    model = holder["model"]
    sample = _measure_repeated(
        lambda: model.simulate(n=sample_size, num_threads=1),
        repeats=repeats,
        warmup=warmup,
    )
    density = _measure_repeated(
        lambda: model.pdf(query_np, num_threads=1),
        repeats=repeats,
        warmup=warmup,
    )
    return {
        "engine": "pvc_cpu",
        "fit_seconds": fit,
        "sample_seconds": sample,
        "density_query_seconds": density,
        "reference_nonparametric_method": reference_nonparametric_method,
    }


def _benchmark_tvc(
    *,
    obs: torch.Tensor,
    device: torch.device,
    engine_name: str,
    repeats: int,
    warmup: int,
    sample_size: int,
    query_size: int,
    grid_size: int,
    bicop_backend: str,
) -> dict[str, object]:
    obs_device = obs.to(device)
    query_obs = obs_device[: min(query_size, obs_device.shape[0])]
    holder: dict[str, object] = {}

    def _fit_once():
        model = tvc.VineCop(num_dim=obs.shape[1], is_cop_scale=True, num_step_grid=grid_size).to(
            device
        )
        model.fit(
            obs_device,
            mtd_bidep="kendall_tau",
            thresh_trunc=0.05,
            bicop_backend=bicop_backend,
            bicop_kwargs=default_bicop_kwargs(bicop_backend, num_step_grid=grid_size),
        )
        holder["model"] = model

    fit = _measure_repeated(
        _fit_once, repeats=repeats, warmup=warmup, sync_cuda=device.type == "cuda"
    )
    model = holder["model"]
    sample = _measure_repeated(
        lambda: model.sample(num_sample=sample_size, seed=123),
        repeats=repeats,
        warmup=warmup,
        sync_cuda=device.type == "cuda",
    )
    density = _measure_repeated(
        lambda: model.log_pdf(query_obs),
        repeats=repeats,
        warmup=warmup,
        sync_cuda=device.type == "cuda",
    )
    return {
        "engine": engine_name,
        "fit_seconds": fit,
        "sample_seconds": sample,
        "density_query_seconds": density,
        "bicop_backend": bicop_backend,
        "device": device.type,
    }


def _aggregate_workload_scores(payload: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for workload in payload["results"]:
        for engine_name, metrics in workload["engines"].items():
            rows.append(
                {
                    "workload": f"{workload['num_obs']}x{workload['num_dim']}",
                    "engine": engine_name,
                    "fit_mean": metrics["fit_seconds"]["summary"]["mean"],
                    "sample_mean": metrics["sample_seconds"]["summary"]["mean"],
                    "density_mean": metrics["density_query_seconds"]["summary"]["mean"],
                }
            )
    return rows


def _write_markdown(payload: dict[str, object], path: Path) -> None:
    rows = _aggregate_workload_scores(payload)
    lines = [
        "# VineCop runtime comparison",
        "",
        f"Current default bicop backend: `{payload['environment']['default_bicop_backend']}`",
        "",
        "| workload | engine | fit mean (s) | sample mean (s) | density mean (s) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['workload']}` | `{row['engine']}` | "
            f"{row['fit_mean']:.4f} | {row['sample_mean']:.4f} | {row['density_mean']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    include_reference = _resolve_flag(
        args.include_reference,
        available=_has_reference(),
        label="pyvinecopulib reference benchmark",
    )
    include_cuda = _resolve_flag(
        args.include_cuda,
        available=torch.cuda.is_available(),
        label="CUDA benchmark",
    )
    results: list[dict[str, object]] = []
    for num_dim in args.num_dim:
        for num_obs in args.num_obs:
            obs = _make_obs(
                num_obs=num_obs, num_dim=num_dim, seed=args.seed + num_dim * 1000 + num_obs
            )
            engines: dict[str, object] = {}
            if include_reference:
                engines["pvc_cpu"] = _benchmark_pvc(
                    obs=obs,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    sample_size=args.sample_size,
                    query_size=args.query_size,
                    reference_nonparametric_method=args.reference_nonparametric_method,
                )
            engines["tvc_cpu"] = _benchmark_tvc(
                obs=obs,
                device=torch.device("cpu"),
                engine_name="tvc_cpu",
                repeats=args.repeats,
                warmup=args.warmup,
                sample_size=args.sample_size,
                query_size=args.query_size,
                grid_size=args.grid_size,
                bicop_backend=args.bicop_backend,
            )
            if include_cuda:
                engines["tvc_cuda"] = _benchmark_tvc(
                    obs=obs,
                    device=torch.device("cuda"),
                    engine_name="tvc_cuda",
                    repeats=args.repeats,
                    warmup=args.warmup,
                    sample_size=args.sample_size,
                    query_size=args.query_size,
                    grid_size=args.grid_size,
                    bicop_backend=args.bicop_backend,
                )
            results.append(
                {
                    "num_obs": int(num_obs),
                    "num_dim": int(num_dim),
                    "engines": engines,
                }
            )

    report = {
        "schema_version": 1,
        "benchmark": "vinecop_runtimes",
        "config": {
            "num_obs": [int(x) for x in args.num_obs],
            "num_dim": [int(x) for x in args.num_dim],
            "grid_size": int(args.grid_size),
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "sample_size": int(args.sample_size),
            "query_size": int(args.query_size),
            "seed": int(args.seed),
            "include_reference": include_reference,
            "include_cuda": include_cuda,
            "reference_nonparametric_method": args.reference_nonparametric_method,
            "bicop_backend": args.bicop_backend,
        },
        "environment": _environment(include_reference, include_cuda),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, args.markdown_output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
