"""Locked regression checks for kdecopula-aligned torch-native bicop backends.

These tests always validate the current implementation against the tracked fixture payload.
When the fixture provenance is ``r-kdecopula``, the same payload can also serve as an
external-reference fidelity target; otherwise it is only a locked regression target.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import torchvinecopulib as tvc


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aligned_bicop" / "locked_regression.json"


def _fixture_payload() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _canonical_bw(method: str, backend_config: dict[str, object]) -> object:
    if method in {"ttcv", "ttpi"}:
        return backend_config["tt_params"]
    if method in {"tll1", "tll2", "tll1nn", "tll2nn", "beta"}:
        return backend_config["bw"]
    raise ValueError(f"Unsupported method: {method}")


def _assert_nested_allclose(actual: object, expected: object, *, atol: float) -> None:
    if isinstance(actual, dict):
        assert set(actual) == set(expected)
        for key in actual:
            _assert_nested_allclose(actual[key], expected[key], atol=atol)
        return
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        atol=atol,
        rtol=atol,
    )


def test_locked_aligned_fixture_schema():
    payload = _fixture_payload()
    assert payload["schema_version"] == 1
    assert payload["fixture_family"] == "aligned_bicop"
    assert payload["provenance"]["mode"] in {"torch-bootstrap", "r-kdecopula"}
    assert set(payload["methods"]) == {"ttcv", "ttpi", "tll1", "tll2", "tll1nn", "tll2nn", "beta"}
    assert payload["dataset"]["num_step_grid"] == 65


@pytest.mark.parametrize("method", ["ttcv", "ttpi", "tll1", "tll2", "tll1nn", "tll2nn", "beta"])
def test_aligned_backends_match_locked_regression_fixtures(method):
    payload = _fixture_payload()
    method_payload = payload["methods"][method]
    obs = torch.as_tensor(payload["dataset"]["obs"], dtype=torch.float64)
    eval_points = torch.as_tensor(payload["dataset"]["eval_points"], dtype=torch.float64)
    cop = tvc.BiCop(num_step_grid=payload["dataset"]["num_step_grid"])
    cop.fit(obs, bicop_backend=method, bicop_kwargs=method_payload["fit_kwargs"])
    assert cop.backend_config["bandwidth_kind"] == method_payload["canonical"]["bandwidth_kind"]
    assert cop.backend_config.get("mult", 1.0) == pytest.approx(
        method_payload["canonical"]["mult"]
    )
    _assert_nested_allclose(
        _canonical_bw(method, cop.backend_config),
        method_payload["canonical"]["bw"],
        atol=float(method_payload["tolerances"]["bw_atol"]),
    )
    np.testing.assert_allclose(
        cop.pdf(eval_points).detach().cpu().numpy().reshape(-1),
        np.asarray(method_payload["pdf"], dtype=np.float64),
        atol=float(method_payload["tolerances"]["value_atol"]),
        rtol=float(method_payload["tolerances"]["value_atol"]),
    )
    np.testing.assert_allclose(
        cop.cdf(eval_points).detach().cpu().numpy().reshape(-1),
        np.asarray(method_payload["cdf"], dtype=np.float64),
        atol=float(method_payload["tolerances"]["value_atol"]),
        rtol=float(method_payload["tolerances"]["value_atol"]),
    )


@pytest.mark.parametrize("method", ["ttcv", "ttpi", "tll1", "tll2", "tll1nn", "tll2nn", "beta"])
def test_aligned_backends_match_external_kdecopula_reference_when_available(method):
    payload = _fixture_payload()
    if payload["provenance"]["mode"] != "r-kdecopula":
        pytest.skip("external kdecopula reference fixture not installed")
    method_payload = payload["methods"][method]
    obs = torch.as_tensor(payload["dataset"]["obs"], dtype=torch.float64)
    eval_points = torch.as_tensor(payload["dataset"]["eval_points"], dtype=torch.float64)
    cop = tvc.BiCop(num_step_grid=payload["dataset"]["num_step_grid"])
    cop.fit(obs, bicop_backend=method, bicop_kwargs=method_payload["fit_kwargs"])
    _assert_nested_allclose(
        _canonical_bw(method, cop.backend_config),
        method_payload["canonical"]["bw"],
        atol=float(method_payload["tolerances"]["bw_atol"]),
    )
    np.testing.assert_allclose(
        cop.pdf(eval_points).detach().cpu().numpy().reshape(-1),
        np.asarray(method_payload["pdf"], dtype=np.float64),
        atol=float(method_payload["tolerances"]["value_atol"]),
        rtol=float(method_payload["tolerances"]["value_atol"]),
    )
    np.testing.assert_allclose(
        cop.cdf(eval_points).detach().cpu().numpy().reshape(-1),
        np.asarray(method_payload["cdf"], dtype=np.float64),
        atol=float(method_payload["tolerances"]["value_atol"]),
        rtol=float(method_payload["tolerances"]["value_atol"]),
    )


def test_aligned_backends_reject_noncanonical_bandwidth_shapes():
    obs = torch.as_tensor(_fixture_payload()["dataset"]["obs"], dtype=torch.float64)
    with pytest.raises(ValueError, match="custom 2x2 matrix"):
        tvc.BiCop(num_step_grid=33).fit(
            obs,
            bicop_backend="tll2",
            bicop_kwargs={"bandwidth": [0.12, 0.18]},
        )
    with pytest.raises(ValueError, match="positive scalar"):
        tvc.BiCop(num_step_grid=33).fit(
            obs,
            bicop_backend="beta",
            bicop_kwargs={"bandwidth": torch.tensor([0.08, 0.18], dtype=torch.float64)},
        )
