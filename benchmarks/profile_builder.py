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
    parser = argparse.ArgumentParser(description="Benchmark VineCop.fit() builder performance.")
    parser.add_argument("--num-obs", type=int, default=10_000)
    parser.add_argument("--num-dim", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=65)
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
        default=Path("benchmarks/results/profile_builder.bench.json"),
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


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    obs = _make_obs(args.num_obs, args.num_dim, device, args.seed)
    model = tvc.VineCop(num_dim=args.num_dim, is_cop_scale=True, num_step_grid=args.grid_size).to(device)
    fit_kwargs = {
        "mtd_bidep": "kendall_tau",
        "thresh_trunc": 0.05,
        "bicop_backend": args.bicop_backend,
    }
    if args.bicop_backend != "tll_ref":
        fit_kwargs["bicop_kwargs"] = {"bandwidth": "silverman"}

    metrics: dict[str, int | float | str] = {
        "num_obs": args.num_obs,
        "num_dim": args.num_dim,
        "grid_size": args.grid_size,
        "device": args.device,
        "bicop_backend": args.bicop_backend,
        "seed": args.seed,
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t0 = time.perf_counter()
        model.fit(obs, **fit_kwargs)
        _sync(device)
        metrics["fit_seconds"] = time.perf_counter() - t0
        metrics["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated(device))
    else:
        tracemalloc.start()
        t0 = time.perf_counter()
        model.fit(obs, **fit_kwargs)
        metrics["fit_seconds"] = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics["peak_tracemalloc_bytes"] = int(peak)

    metrics["state_dict_bytes"] = _state_dict_nbytes(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
