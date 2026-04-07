# torchvinecopulib

[![Lint Pytest](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml)
[![Deploy Docs](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/static.yml/badge.svg?branch=main)](https://ty-cheng.github.io/torchvinecopulib/)
[![PyPI - Version](https://img.shields.io/pypi/v/torchvinecopulib)](https://pypi.org/project/torchvinecopulib/)

`torchvinecopulib` is a PyTorch-first vine copula library for fitting, evaluating, and sampling
high-dimensional dependence models on CPU or GPU. The project targets researchers and engineers
who need differentiable query paths, explicit builder/engine boundaries, and numerically stable
copula estimation without leaving the PyTorch ecosystem.

## What lives here

- **Quickstart**: shortest working paths for `fit -> log_pdf -> sample` and Rosenblatt roundtrips.
- **Theory**: vine decomposition, pseudo-observations, and Rosenblatt transforms.
- **Systems**: builder vs engine, dtype policy, boundary semantics, and benchmark interpretation.
- **API Reference**: curated entrypoints for `BiCop`, `VineCop`, execution plans, and utilities.
- **Examples / Benchmarks**: runnable scripts, benchmark JSON outputs, and profiling entrypoints.

## Install

```bash
pip install torchvinecopulib torch
```

For local development with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv sync --extra cpu
```

Install the optional reference backend only when you need `lp_ref`, `tll_ref`, or oracle
comparisons:

```bash
uv sync --extra cpu --extra reference
```

## Documentation map

```{toctree}
:maxdepth: 2
:caption: Documentation

quickstart
theory/index
systems/index
api/index
examples_benchmarks
```

## Project notes

- `fit()` is a builder path and does not preserve an autograd graph.
- Differentiable query methods include `log_pdf()`, `rosenblatt()`, `inverse_rosenblatt()`,
  `cdf()`, and pair-copula `hfunc()`/`hinv()` routines.
- The default production path uses torch-native grid backends. CPU-only reference backends remain
  optional through the `reference` extra.

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
