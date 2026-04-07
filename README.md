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
backend API around `marginal_backend` and `bicop_backend`, keeps a torch-native grid path as
the production default, and isolates CPU-only reference oracles behind the optional
`reference` extra.

- C-, D-, and R-vine fitting with differentiable log-density evaluation
- Torch-native `TorchKDE1D` marginals with ISJ bandwidth by default
- Torch-native `TorchCopulaKDE2D` pair-copula grids with `grid_reflect` and `grid_probit`
- Optional `lp_ref` and `tll_ref` CPU reference backends
- Differentiable query path via `log_pdf()`, `rosenblatt()`, `cdf()`, `hfunc()`, and bilinear interpolation
- Boundary-aware query policies via `boundary_policy="hard" | "st"`
- Explicit `VineBuilder` / `VineExecutionPlan` / `VineCopEngine` entrypoints
- Plan-backed execution engine with diagnostics export and dtype-controlled inference plan export

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

For local development with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv sync --extra cpu
```

For docs-only work with `pip` instead of `uv`:

```bash
pip install -e ".[cpu,docs]"
```

Install the optional reference backend only when you need `lp_ref`, `tll_ref`, or oracle
comparisons:

```bash
uv sync --extra cpu --extra reference
# or
pip install "torchvinecopulib[reference]"
```

### Dependencies

Current core dependencies are:

```toml
[project]
dependencies = [
  "numpy>=2",
  "scipy",
]

[project.optional-dependencies]
cpu = ["torch>=2"]
cu126 = ["torch>=2"]
cu128 = ["torch>=2"]
docs = ["furo", "myst-parser", "sphinx", "sphinx_pyproject"]
examples = ["pytorch-lightning", "tqdm"]
reference = ["pyvinecopulib"]
```

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
    bicop_backend="grid_reflect",
)
log_pdf = vc.log_pdf(obs[:16])
sample = vc.sample(num_sample=32, seed=0)
u = vc.rosenblatt(obs[:16])
recovered = vc.inverse_rosenblatt(u)
```

## Documentation and Tests

- Documentation: [GitHub Pages](https://ty-cheng.github.io/torchvinecopulib/)
- Quickstart, theory notes, systems notes, and API reference all live under the docs site.
- Docs are built in GitHub Actions and deployed from the `gh-pages` branch. Generated HTML is not
  tracked on `main`.
- `docs/_api_stubs/` contains tracked API navigation stubs for Sphinx; it is source content, not a
  build artifact, and should stay in git.
- Examples: [`examples/`](https://github.com/TY-Cheng/torchvinecopulib/tree/main/examples)
- Test suite:

```bash
uv run coverage run --source=torchvinecopulib -m pytest tests
uv run coverage report -m
```

Build docs locally with:

```bash
uv run sphinx-build -b html -n -W --keep-going docs/ docs/_build/html
uv run sphinx-build -b doctest docs/ docs/_build/doctest
```

GitHub Actions is split into:

- `ci.yml`: lint, docs-check, fast/full tests, optional reference/CUDA jobs, benchmark smoke.
- `docs.yml`: cloud build + deploy of docs to `gh-pages`.
- `release.yml`: build distributions and publish tagged releases.

## Migration

Version `1.3.0` is a breaking release.

- `kdeCDFPPF1D` was removed. Use `TorchKDE1D` instead.
- `fastKDE` was removed from runtime dependencies and from the default KDE path.
- `pyvinecopulib` moved to the optional `reference` extra.
- `mtd_kde` is deprecated. Use `bicop_backend="grid_reflect" | "grid_probit" | "tll_ref"` and
  `marginal_backend="grid" | "lp_ref"` instead.
- `BiCop.fit()` and `VineCop.fit()` now normalize legacy top-level bandwidth arguments into
  `bicop_kwargs` / `marginal_kwargs`.
- `fit()` is a builder path, not a differentiable training layer. The differentiable path is
  `log_pdf()`, `rosenblatt()`, `cdf()`, `hfunc()`, and the interpolation kernels.
- `boundary_policy="st"` keeps straight-through query gradients on clamp-heavy boundary paths.

Run the optional benchmark/profiler scripts with:

```bash
uv run --extra cpu python benchmarks/profile_builder.py --device cpu
uv run --extra cpu python benchmarks/profile_query.py --device cpu
```

## TODO

- ~~`fastkde.pdf` onto `torch.Tensor`~~
- ~~replace runtime `fastKDE` dependency with a torch-native KDE path~~
- vectorized MST / dependence scheduling in the builder path
- vectorized union-find for structure learning
- benchmark-driven auto-thresholding for builder/query backends

## Contributing

Contributions are welcome. Keep pull requests focused, include tests for new behavior, and note
any numerical, documentation, or API compatibility changes explicitly.

## Third-Party Notice

`torchvinecopulib` depends on PyTorch at runtime. The project also retains historical attribution
for FastKDE and offers an optional `pyvinecopulib` reference backend. See [LICENSE](./LICENSE)
for the full third-party license texts and attribution notes.
