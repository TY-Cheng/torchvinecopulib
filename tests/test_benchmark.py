"""Smoke and schema checks for benchmark entrypoints.

These tests verify that benchmark scripts execute and emit stable structured outputs.
They do not validate the statistical or performance conclusions of the benchmarks.
"""

import json
import subprocess
import sys

import pytest
import torch

import torchvinecopulib as tvc

from . import DEVICE, correlated_raw, gaussian_copula


pytestmark = pytest.mark.slow


@pytest.mark.parametrize("num_obs", [1_000, 10_000])
@pytest.mark.parametrize("num_dim", [10, 20])
@pytest.mark.parametrize("grid_size", [64, 128])
def test_benchmark_smoke(num_obs, num_dim, grid_size):
    obs = gaussian_copula(num_obs=num_obs, rho=0.35, dim=num_dim).to(DEVICE)
    model = tvc.VineCop(num_dim=num_dim, is_cop_scale=True, num_step_grid=grid_size + 1).to(DEVICE)
    model.fit(
        obs,
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        thresh_trunc=0.05,
        mtd_bidep="kendall_tau",
    )
    lp = model.log_pdf(obs[:64])
    samp = model.sample(num_sample=64, seed=9)
    cdf = model.cdf(obs[:8], num_sample=255, seed=5)
    assert lp.shape == (64, 1)
    assert samp.shape == (64, num_dim)
    assert cdf.shape == (8, 1)
    assert torch.isfinite(lp).all()


def test_raw_scale_benchmark_smoke():
    obs = correlated_raw(1_000, 0.45, 10).to(DEVICE)
    model = tvc.VineCop(num_dim=10, is_cop_scale=False, num_step_grid=65).to(DEVICE)
    model.fit(
        obs,
        marginal_kwargs={"bandwidth": "isj"},
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        thresh_trunc=0.05,
    )
    assert torch.isfinite(model.log_pdf(obs[:32])).all()


def test_profile_builder_script_emits_schema(tmp_path):
    out = tmp_path / "builder.json"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/profile_builder.py",
            "--num-obs",
            "128",
            "--num-dim",
            "4",
            "--grid-size",
            "33",
            "--output",
            str(out),
        ],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "builder"
    assert payload["config"]["num_dim"] == 4
    assert payload["metrics"]["fit_seconds"] >= 0.0
    assert payload["metrics"]["state_dict_bytes"] > 0


def test_profile_query_script_emits_schema(tmp_path):
    out = tmp_path / "query.json"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/profile_query.py",
            "--num-obs",
            "128",
            "--num-dim",
            "4",
            "--grid-size",
            "33",
            "--batch-size",
            "16",
            "--cdf-samples",
            "63",
            "--compile-backend",
            "eager",
            "--output",
            str(out),
        ],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "query"
    assert payload["metrics"]["queries"]["log_pdf"]["seconds"] >= 0.0
    assert payload["metrics"]["queries"]["rosenblatt"]["numel"] == 16 * 4
    assert payload["metrics"]["queries"]["inverse_rosenblatt"]["numel"] == 16 * 4
    assert "compile_log_pdf" in payload["metrics"]["queries"]


def test_compare_marginal_backends_script_emits_schema(tmp_path):
    out = tmp_path / "marginal.json"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/compare_marginal_backends.py",
            "--train-size",
            "96",
            "--test-size",
            "192",
            "--fit-grid-size",
            "65",
            "--eval-grid-size",
            "129",
            "--quantile-size",
            "63",
            "--repeats",
            "1",
            "--scenarios",
            "normal",
            "--output",
            str(out),
        ],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "marginal_backends"
    assert payload["config"]["scenarios"] == ["normal"]
    backends = payload["results"]["normal"]["backends"]
    assert set(backends) == {"grid", "lp"}
    for backend_name, report in backends.items():
        assert report["runs"][0]["backend_name"] == backend_name
        assert report["summary"]["fit_seconds"]["mean"] >= 0.0


def test_compare_bicop_backends_script_emits_schema(tmp_path):
    out = tmp_path / "bicop.json"
    md = tmp_path / "bicop.md"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/compare_bicop_backends.py",
            "--train-sizes",
            "96",
            "--grid-sizes",
            "33",
            "--query-size",
            "48",
            "--repeats",
            "1",
            "--families",
            "gaussian",
            "--backends",
            "grid_reflect",
            "ttcv",
            "ttpi",
            "--output",
            str(out),
            "--markdown-output",
            str(md),
        ],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "bicop_backends"
    assert payload["config"]["families"] == ["gaussian"]
    assert payload["summary"]["winner"] in {"grid_reflect", "ttcv", "ttpi"}
    scores = payload["summary"]["backend_scores"]
    assert set(scores) == {"grid_reflect", "ttcv", "ttpi"}
    assert "composite" in scores["grid_reflect"]
    assert payload["environment"]["default_bicop_backend"] == "beta"
    assert md.exists()


def test_compare_vinecop_runtimes_script_emits_schema(tmp_path):
    out = tmp_path / "vinecop_runtimes.json"
    md = tmp_path / "vinecop_runtimes.md"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/compare_vinecop_runtimes.py",
            "--num-obs",
            "128",
            "--num-dim",
            "4",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--sample-size",
            "32",
            "--query-size",
            "32",
            "--include-reference",
            "no",
            "--include-cuda",
            "no",
            "--output",
            str(out),
            "--markdown-output",
            str(md),
        ],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "vinecop_runtimes"
    assert payload["config"]["include_reference"] is False
    assert payload["config"]["include_cuda"] is False
    assert payload["environment"]["default_bicop_backend"] == "beta"
    assert set(payload["results"][0]["engines"]) == {"tvc_cpu"}
    assert payload["results"][0]["engines"]["tvc_cpu"]["fit_seconds"]["summary"]["mean"] >= 0.0
    assert md.exists()
