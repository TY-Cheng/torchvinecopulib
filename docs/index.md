# torchvinecopulib

[![CI](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/ci.yml)
[![Docs](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/docs.yml/badge.svg?branch=main)](https://ty-cheng.github.io/torchvinecopulib/)
[![PyPI - Version](https://img.shields.io/pypi/v/torchvinecopulib)](https://pypi.org/project/torchvinecopulib/)

`torchvinecopulib` is a PyTorch-first vine copula library for fitting, evaluating, and sampling
high-dimensional dependence models on CPU or GPU. It is designed for statistical computing
workflows that require differentiable query paths, explicit builder/engine boundaries, and
numerically stable copula estimation within the PyTorch ecosystem.

## Scope

`torchvinecopulib` currently targets continuous random variables.

- The public fitting and query APIs assume continuous marginals, or data that can reasonably be
  treated as continuous.
- Discrete or mixed marginals such as categorical, count, or ordinal variables are not a supported
  target for the current library design.
- If observations are heavily rounded or contain many ties, apply an appropriate jitter or other continuous relaxation before fitting.

## Contents

- [Quickstart](quickstart.md): shortest working paths for `fit -> log_pdf -> sample` and
  Rosenblatt roundtrips.
- [Theory](theory/vine_decomposition.md): implementation-facing notes on vine decomposition and
  Rosenblatt transforms.
- [Systems](systems/backends.md): builder vs engine, dtype policy, boundary semantics, and
  benchmark interpretation.
- [API Reference](api/index.md): curated entrypoints for `BiCop`, `VineCop`, execution plans, and
  utilities.
- [Examples and benchmarks](examples_benchmarks.md): maintained scripts, committed figures, and
  benchmark entrypoints.

## Install

```bash
pip install torchvinecopulib torch
```

For local development with `uv`:

```bash
uv sync --extra cpu
```

For local docs work, add the contributor-only `docs` dependency group:

```bash
uv sync --extra cpu --group docs
```

If you prefer an external project environment instead of a local `.venv`, set
`UV_PROJECT_ENVIRONMENT` before syncing:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/torchvinecopulib"
uv sync --extra cpu
```

Install the optional reference backend only when you need `tll_ref` or reference/oracle
comparisons:

```bash
uv sync --extra cpu --extra reference
```

## Runtime notes

- `fit()` is a builder path and does not preserve an autograd graph.
- Differentiable query methods include `log_pdf()`, `rosenblatt()`, `cdf()`, and pair-copula
  `hfunc()` routines.
- Sampling, inverse Rosenblatt transforms, and inverse pair-copula conditionals are stabilized
  query paths and are not documented as differentiable operators.
- The default production path uses torch-native grid backends. CPU-only reference backends remain
  optional through the `reference` extra.
