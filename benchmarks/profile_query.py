from __future__ import annotations

import argparse
import io
import json
import platform
import time
import tracemalloc
from pathlib import Path

import torch
from torch.special import ndtr

import torchvinecopulib as tvc
from torchvinecopulib.backends import (
    DEFAULT_BICOP_BACKEND,
    PUBLIC_BICOP_BACKENDS,
    default_bicop_kwargs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VineCop query-path performance.")
    parser.add_argument("--num-obs", type=int, default=10_000)
    parser.add_argument("--num-dim", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cdf-samples", type=int, default=511)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--bicop-backend",
        choices=PUBLIC_BICOP_BACKENDS + ("tll_ref",),
        default=DEFAULT_BICOP_BACKEND,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compile-backend",
        choices=("none", "eager"),
        default="none",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/profile_query.bench.json"),
    )
    return parser.parse_args()


def _make_obs(num_obs: int, num_dim: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    corr = torch.rand(num_dim, num_dim, dtype=torch.float64, generator=generator)
    corr = corr / corr.norm(dim=1, keepdim=True).clamp_min(1e-12)
    corr = corr @ corr.T
    z = torch.randn(num_obs, num_dim, dtype=torch.float64, generator=generator)
    obs = ndtr(z @ torch.linalg.cholesky(corr, upper=True))
    return obs.to(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_dict_nbytes(module: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.tell()


def _environment(device: torch.device) -> dict[str, str | bool]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_type": device.type,
    }


def _measure(label: str, fn, device: torch.device) -> dict[str, int | float | str]:
    metrics: dict[str, int | float | str] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t0 = time.perf_counter()
        result = fn()
        _sync(device)
        metrics["seconds"] = time.perf_counter() - t0
        metrics["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        metrics["peak_memory_kind"] = "cuda_bytes"
    else:
        tracemalloc.start()
        t0 = time.perf_counter()
        result = fn()
        metrics["seconds"] = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics["peak_memory_bytes"] = int(peak)
        metrics["peak_memory_kind"] = "tracemalloc_bytes"
    metrics["numel"] = int(result.numel())
    return metrics


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    obs = _make_obs(args.num_obs, args.num_dim, device, args.seed)
    query_obs = obs[: args.batch_size]
    model = tvc.VineCop(num_dim=args.num_dim, is_cop_scale=True, num_step_grid=args.grid_size).to(
        device
    )
    fit_kwargs = {
        "mtd_bidep": "kendall_tau",
        "thresh_trunc": 0.05,
        "bicop_backend": args.bicop_backend,
    }
    fit_kwargs["bicop_kwargs"] = default_bicop_kwargs(
        args.bicop_backend,
        num_step_grid=args.grid_size,
    )
    model.fit(obs, **fit_kwargs)

    uniforms = model.rosenblatt(query_obs)
    metrics: dict[str, object] = {
        "state_dict_bytes": _state_dict_nbytes(model),
        "num_edges": len(model.bicops),
        "diagnostics": {
            "itp_failures": model.diagnostics().itp_failures,
            "bisect_refinements": model.diagnostics().bisect_refinements,
            "fallback_to_indep": model.diagnostics().fallback_to_indep,
            "max_abs_hfunc_error": model.diagnostics().max_abs_hfunc_error,
        },
        "queries": {
            "log_pdf": _measure("log_pdf", lambda: model.log_pdf(query_obs), device),
            "sample": _measure(
                "sample",
                lambda: model.sample(num_sample=args.batch_size, seed=args.seed + 1),
                device,
            ),
            "cdf": _measure(
                "cdf",
                lambda: model.cdf(
                    query_obs[: min(16, args.batch_size)],
                    num_sample=args.cdf_samples,
                    seed=args.seed + 2,
                ),
                device,
            ),
            "rosenblatt": _measure("rosenblatt", lambda: model.rosenblatt(query_obs), device),
            "inverse_rosenblatt": _measure(
                "inverse_rosenblatt",
                lambda: model.inverse_rosenblatt(uniforms),
                device,
            ),
        },
    }
    if args.compile_backend != "none" and hasattr(torch, "compile"):
        compiled = torch.compile(model.engine.log_pdf, backend=args.compile_backend)
        compiled_metrics = _measure("compile_log_pdf", lambda: compiled(query_obs), device)
        base_seconds = float(metrics["queries"]["log_pdf"]["seconds"])
        compiled_metrics["speedup_vs_eager"] = (
            base_seconds / float(compiled_metrics["seconds"])
            if float(compiled_metrics["seconds"]) > 0.0
            else 0.0
        )
        metrics["queries"]["compile_log_pdf"] = compiled_metrics

    report = {
        "schema_version": 1,
        "benchmark": "query",
        "config": {
            "num_obs": args.num_obs,
            "num_dim": args.num_dim,
            "grid_size": args.grid_size,
            "batch_size": args.batch_size,
            "cdf_samples": args.cdf_samples,
            "device": args.device,
            "bicop_backend": args.bicop_backend,
            "seed": args.seed,
            "compile_backend": args.compile_backend,
        },
        "environment": _environment(device),
        "metrics": metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
