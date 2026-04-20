# torchvinecopulib

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e8a7a7448b2043d9bbefafc5a3ec14f7)](https://app.codacy.com/gh/TY-Cheng/torchvinecopulib/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/e8a7a7448b2043d9bbefafc5a3ec14f7)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![CI](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/ci.yml)
[![Docs](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/docs.yml/badge.svg?branch=main)](https://ty-cheng.github.io/torchvinecopulib/)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchvinecopulib)
[![OS](https://img.shields.io/badge/OS-Windows%7CmacOS%7CUbuntu-blue)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/TY-Cheng/torchvinecopulib/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/torchvinecopulib)](https://pypi.org/project/torchvinecopulib/)

`torchvinecopulib` is a PyTorch-first vine copula library for fitting, evaluating, and sampling
high-dimensional dependence models on CPU or GPU. Version `1.3.0` standardizes the
backend API around `marginal_backend` and `bicop_backend`, uses a benchmark-selected torch-native
`ttpi` pair-copula path as the current production default, and isolates CPU-only reference oracles behind the optional
`reference` extra.

- C-, D-, and R-vine fitting with differentiable log-density evaluation
- Torch-native 1D marginals with `grid` and `lp` backends
- Torch-native 2D pair-copula backends: `beta`, `ttpi`, `grid_reflect`, `grid_probit`
- Research-facing torch-native 2D backends: `ttcv`, `tll1`, `tll2`, `tll1nn`, `tll2nn`, `beta_qt`, `spline_pen`
- Optional `tll_ref` CPU reference backend
- Differentiable query path via `log_pdf()`, `rosenblatt()`, `cdf()`, `hfunc()`, and bilinear interpolation
- Boundary-aware query policies via `boundary_policy="hard" | "st"`
- Explicit `VineBuilder` / `VineBuildArtifact` / `VineCopEngine` entrypoints
- Plan-backed execution engine with diagnostics export and dtype-controlled inference plan export

## Scope

`torchvinecopulib` currently targets continuous random variables.

- The fitting and query paths assume continuous marginals or data that can be treated as continuous.
- The current API is not designed for categorical, count, ordinal, or other genuinely discrete / mixed marginals.
- If your data are heavily rounded or contain many ties, treat them as an approximation to an underlying continuous variable first, for example by applying a suitable jitter or other continuous relaxation before fitting.

## Citation

If you use `torchvinecopulib` in your work, please cite:

> Cheng, Tuoyuan, Thibault Vatter, Thomas Nagler, and Kan Chen. "Vine Copulas as Differentiable Computational Graphs." arXiv preprint arXiv:2506.13318 (2025).

```latex
@article{cheng2025vine,
  title={Vine Copulas as Differentiable Computational Graphs},
  author={Cheng, Tuoyuan and Vatter, Thibault and Nagler, Thomas and Chen, Kan},
  journal={arXiv preprint arXiv:2506.13318},
  year={2025},
  url={https://arxiv.org/abs/2506.13318},
}
```

## Installation

Install from PyPI with a matching PyTorch build:

```bash
pip install torchvinecopulib torch
```

For local development with `uv` and a shared external environment:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/torchvinecopulib"
uv sync --extra cpu
```

For local docs work, add the contributor-only `docs` dependency group:

```bash
uv sync --extra cpu --group docs
```

For example scripts and docs asset generation, add the `examples` dependency group:

```bash
uv sync --extra cpu --group examples
```

Maintained example scripts can regenerate the docs figures with:

```bash
uv run --extra cpu --group examples python scripts/generate_example_assets.py
```

Install the optional reference backend only when you need `tll_ref` or oracle comparisons:

```bash
uv sync --extra cpu --extra reference
# or
pip install "torchvinecopulib[reference]"
```

### Dependencies

The dependency layout is intentionally split by who consumes it:

- Runtime core: `numpy`, `scipy`
- Published extras for end users: `cpu`, `cu126`, `cu128`, `reference`
- Local `uv` groups for contributors: `dev`, `docs`, `examples`

In other words, `reference` remains an installable package extra because it changes library
functionality, while `docs` and `examples` are maintained as repository-local dependency groups.

### Marginal backend roles

- `marginal_backend="grid"`: default `GridKDE1D` path; regular-grid Gaussian KDE with ISJ by default.
- `marginal_backend="lp"`: torch-native continuous local-polynomial KDE inspired by `kde1d`.

### Bicop backend fidelity policy

The current 2D bicop surface mixes two kinds of names:

- aligned `kdecopula` names with canonical `bandwidth`/`mult` semantics:
  `ttpi`, `ttcv`, `tll1`, `tll2`, `tll1nn`, `tll2nn`, `beta`
- repository-native engineering backends that are not claims of `kdecopula` method identity:
  `grid_reflect`, `grid_probit`, `beta_qt`, `spline_pen`

For the aligned names, `bandwidth` now mirrors the upstream `bw` object shape:

- `ttpi` / `ttcv`: length-4 `(h, rho, theta1, theta2)`
- `tll1` / `tll2`: `2x2` matrix
- `tll1nn` / `tll2nn`: mapping with `B`, `alpha`, `kappa`
- `beta`: positive scalar

`mult` is the canonical bandwidth multiplier for aligned bicop backends.

The repository now also ships locked aligned-bicop regression fixtures under
`tests/fixtures/aligned_bicop/` and a one-time regeneration script at
`scripts/generate_kdecopula_fixtures.py`.

`MR` and `bern` from the current `kdecopula` method table are not implemented at the moment.
The detailed fidelity audit lives in the systems docs.

For CUDA builds of PyTorch, install the matching wheel index from the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).

## Quickstart

```python
import torch
import torchvinecopulib as tvc

torch.manual_seed(0)
obs = torch.rand(128, 4, dtype=torch.float64)

vc = tvc.VineCop(num_dim=4, is_cop_scale=True, num_step_grid=65)
vc.fit(
    obs,
    mtd_vine="cvine",
    mtd_bidep="kendall_tau",
    bicop_backend="beta",
)
log_pdf = vc.log_pdf(obs[:16])
sample = vc.sample(num_sample=32, seed=0)
u = vc.rosenblatt(obs[:16])
recovered = vc.inverse_rosenblatt(u)
```

## Documentation and Tests

- Documentation: [GitHub Pages](https://ty-cheng.github.io/torchvinecopulib/)
- Quickstart, theory notes, systems notes, and API reference all live under the docs site.
- Docs are built with MkDocs Material and rendered API sections via `mkdocstrings`.
- Docs are built in GitHub Actions and deployed through the official GitHub Pages artifact
  workflow. The generated `site/` output is not tracked on `main`.
- Examples: see the docs examples page in `docs/examples_benchmarks.md` for the maintained
  script-backed examples, and use the repository `examples/` directory for heavier experimental
  workflows.
- Test suite:

```bash
just test cpu
# or, if you want the explicit repo-bound pytest command:
uv run --extra cpu --extra reference pytest \
  --cov=torchvinecopulib \
  --cov-branch \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  --cov-report=html \
  --cov-fail-under=94 \
  -W error::DeprecationWarning \
  -m "not cuda" \
  tests
```

- If you use [`just`](https://github.com/casey/just), the repository also ships a thin local task
  runner that mirrors the `uv` + GitHub Actions workflow. It reads `.env` and requires
  `UV_PROJECT_ENVIRONMENT` to be set there so local tasks use the shared external environment
  instead of falling back to a project-local `.venv`. This repository's examples already use
  `.env`, so the simplest setup is to keep project-local paths such as `DIR_WORK` and
  `UV_PROJECT_ENVIRONMENT` there.

```bash
cat > .env <<'EOF'
DIR_WORK="$PWD"
UV_PROJECT_ENVIRONMENT="$HOME/.venvs/torchvinecopulib"
EOF

just setup            # syncs cpu + reference + docs + examples on top of the default dev group
just test             # installs reference; auto-runs CUDA tests only when CUDA is available
just test cpu         # force the CPU-only suite
just examples         # regenerate docs-facing example figures from scripts
just examples check  # verify committed example assets are up to date
just docs             # MkDocs build + docs smoke tests
just bench            # benchmark smoke
just workflow
```

`just test` runs `pytest` with `--extra reference`, so the optional `pyvinecopulib`
dependency is installed by default and `@pytest.mark.reference` tests are included. It
also probes `torch.cuda.is_available()` first and only enables the `@pytest.mark.cuda`
suite when a CUDA device is actually present.

Build docs locally with:

```bash
uv run --extra cpu --group docs mkdocs serve
uv run --extra cpu --group docs mkdocs build --strict
uv run --extra cpu pytest tests/test_docs_smoke.py -q
```

GitHub Actions is split into:

- `ci.yml`: lint, MkDocs docs-check, fast/full tests, optional reference/CUDA jobs, benchmark
  smoke.
- `docs.yml`: cloud build + deploy of docs through the official GitHub Pages artifact workflow.
- `release.yml`: build distributions and publish tagged releases.

## Migration

Version `1.3.0` is a breaking release.

- `kdeCDFPPF1D` was removed. Use `GridKDE1D` instead.
- `TorchKDE1D`, `TorchCopulaKDE2D`, and `VineExecutionPlan` were removed. Use
  `GridKDE1D`, `GridReflectBicopEstimator`, and `VineBuildArtifact`.
- `pyvinecopulib` moved to the optional `reference` extra.
- `mtd_kde`, `kde_backend`, `mtd_tll`, `num_step_grid_kde1d`, and legacy top-level
  `bandwidth_scale` were removed from `BiCop.fit()` / `VineCop.fit()`, and aligned bicop
  backends now accept only canonical `mult`. Use `bicop_backend=...`, `marginal_backend=...`,
  and explicit `bicop_kwargs` / `marginal_kwargs`.
- `fit()` is a builder path, not a differentiable training layer. The differentiable path is
  `log_pdf()`, `rosenblatt()`, `cdf()`, `hfunc()`, and the interpolation kernels.
- `boundary_policy="st"` keeps straight-through query gradients on clamp-heavy boundary paths.

Run the optional benchmark/profiler scripts with:

```bash
uv run --extra cpu python benchmarks/profile_builder.py --device cpu
uv run --extra cpu python benchmarks/profile_query.py --device cpu
uv run --extra cpu --extra reference python benchmarks/compare_bicop_backends.py --device cpu
uv run --extra cpu --extra reference python benchmarks/compare_vinecop_runtimes.py --include-reference yes
```

## TODO

- vectorized MST / dependence scheduling in the builder path
- vectorized union-find for structure learning
- benchmark-driven auto-thresholding for builder/query backends

## Contributing

Contributions are welcome. Keep pull requests focused, include tests for new behavior, and note
any numerical, documentation, or API compatibility changes explicitly.

## Third-Party Notice

`torchvinecopulib` depends on PyTorch at runtime and offers an optional `pyvinecopulib`
reference backend for `tll_ref`. See [LICENSE](./LICENSE) for the full third-party license texts
and attribution notes.
