# Changelog

## Unreleased

### Cleanup

- Removed the experimental `fastkde` and `lp_ref` marginal backends after benchmark review.
- Simplified the public 1D marginal surface back to `marginal_backend="grid" | "lp"`.
- Reworked the marginal benchmark script so each scenario family now jitters its distribution
  parameters across repeats instead of benchmarking one fixed distribution per family.

### Infrastructure

- Split GitHub Actions into `ci.yml`, `docs.yml`, and `release.yml`.
- Migrated docs from Sphinx to MkDocs Material and moved deployment to the official GitHub Pages
  artifact workflow instead of relying on generated HTML in the main branch.
- Added a docs-only optional dependency set and public project URLs for docs, changelog, and issue
  tracking.

## 1.3.0 - 2026-04-06

### Breaking Changes

- Standardized the fitting API around `marginal_backend`, `marginal_kwargs`,
  `bicop_backend`, and `bicop_kwargs`.
- Changed the default production bicop backend naming from legacy `torch_grid` to
  `grid_reflect`.
- Clarified that `fit()` is a builder path and does not preserve a differentiable training
  graph.

### Features

- Added the internal backend registry/factory architecture under `torchvinecopulib.backends`.
- Added the `grid_probit` bicop backend alongside `grid_reflect` and `tll_ref`.
- Added the `recursive` smoother option for torch-native grid backends.
- Added `get_extra_state()` / `set_extra_state()` metadata for `BiCop` and `VineCop`.
- Added optional CUDA profiler and memory regression tests plus benchmark scripts under
  `benchmarks/`.

### Compatibility

- Kept one transition-release compatibility layer for legacy `mtd_kde`, `kde_backend`,
  `mtd_tll`, `bandwidth`, `bandwidth_scale`, and `num_step_grid_kde1d`.
- Added backward-compatible `load_state_dict()` handling for checkpoints without
  `_extra_state`.

### Fixes

- Preserved differentiable query behavior for `log_pdf()`, `cdf()`, `hfunc()`, and bilinear
  interpolation on the new backend-dispatch path.
- Kept CPU-only reference dependencies behind lazy imports so the default runtime no longer
  requires them to import the package.
- Updated examples, docs, and benchmark scripts to use the v1.3 backend API.

## 1.2.0 - 2026-04-06

### Breaking Changes

- Removed the legacy `kdeCDFPPF1D` API.
- Moved `pyvinecopulib` to the optional `reference` extra.
- Deprecated `mtd_kde` in favor of `bicop_backend` and `marginal_backend`.

### Features

- Added `TorchKDE1D` for torch-native marginal KDE with ISJ bandwidth selection.
- Added `TorchCopulaKDE2D` for torch-native pair-copula grid estimation.
- Added lazy grid allocation in `BiCop` and backward-compatible state loading for older
  checkpoints.
- Added internal copula-scale sampling in `VineCop` via `_sample_u()`.

### Fixes

- Fixed `VineCop.cdf()` so non-copula inputs are always compared in copula scale.
- Removed global RNG pollution from `BiCop.sample()` and `VineCop.sample()`.
- Replaced the old non-standard `mutual_info()` implementation with a copula-entropy
  formulation.

### Infrastructure

- Bumped package version from `1.1.2` to `1.2.0`.
- Updated CI to install the `reference` extra, enforce flake8 complexity checks, and treat
  `DeprecationWarning` as errors during tests.
- Added KDE, regression, and benchmark smoke tests.
- Updated README, docs, and license attribution notes for the torch-native KDE migration.
