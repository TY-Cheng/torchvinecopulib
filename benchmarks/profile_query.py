from __future__ import annotations

import argparse
import io
import json
import time
import tracemalloc
from pathlib import Path

import torch
from torch.special import ndtr

import torchvinecopulib as tvc


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
        choices=("grid_reflect", "grid_probit", "tll_ref"),
        default="grid_reflect",
    )
    parser.add_argument("--seed", type=int, default=42)
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


def _measure(label: str, fn, device: torch.device, metrics: dict[str, int | float | str]) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t0 = time.perf_counter()
        result = fn()
        _sync(device)
        metrics[f"{label}_seconds"] = time.perf_counter() - t0
        metrics[f"{label}_peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated(device))
    else:
        tracemalloc.start()
        t0 = time.perf_counter()
        result = fn()
        metrics[f"{label}_seconds"] = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics[f"{label}_peak_tracemalloc_bytes"] = int(peak)
    metrics[f"{label}_numel"] = int(result.numel())


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    obs = _make_obs(args.num_obs, args.num_dim, device, args.seed)
    query_obs = obs[: args.batch_size]
    model = tvc.VineCop(num_dim=args.num_dim, is_cop_scale=True, num_step_grid=args.grid_size).to(device)
    fit_kwargs = {
        "mtd_bidep": "kendall_tau",
        "thresh_trunc": 0.05,
        "bicop_backend": args.bicop_backend,
    }
    if args.bicop_backend != "tll_ref":
        fit_kwargs["bicop_kwargs"] = {"bandwidth": "silverman"}
    model.fit(obs, **fit_kwargs)

    metrics: dict[str, int | float | str] = {
        "num_obs": args.num_obs,
        "num_dim": args.num_dim,
        "grid_size": args.grid_size,
        "batch_size": args.batch_size,
        "cdf_samples": args.cdf_samples,
        "device": args.device,
        "bicop_backend": args.bicop_backend,
        "seed": args.seed,
        "state_dict_bytes": _state_dict_nbytes(model),
    }
    _measure("log_pdf", lambda: model.log_pdf(query_obs), device, metrics)
    _measure("sample", lambda: model.sample(num_sample=args.batch_size, seed=args.seed + 1), device, metrics)
    _measure(
        "cdf",
        lambda: model.cdf(query_obs[: min(16, args.batch_size)], num_sample=args.cdf_samples, seed=args.seed + 2),
        device,
        metrics,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
