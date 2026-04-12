# BiCop API

```{currentmodule} torchvinecopulib
```

## Overview

`BiCop` is the public bivariate copula interface. It owns the fitted density and conditional CDF
grids, exposes differentiable query methods, and handles stabilized inverse conditional sampling.

Backend formulas, `tll_ref` references, and backend-selection guidance live in
[Systems: estimator backends](../systems/backends.md).
The current default for `BiCop.fit(..., bicop_backend=None)` is `bicop_backend="beta"`.

| Object | Purpose |
| --- | --- |
| `BiCop` | Fit and query a bivariate copula on observations shaped `[num_obs, 2]`. |
| `BiCopDiagnostics` | Query-time counters for root-finding fallbacks and residual errors. |
| `GridReflectBicopEstimator` | Reflected-grid 2D KDE backend used by the default grid pair-copula estimator. |

```{toctree}
:hidden:

../_api_stubs/torchvinecopulib.BiCop
../_api_stubs/torchvinecopulib.BiCopDiagnostics
../_api_stubs/torchvinecopulib.GridReflectBicopEstimator
```

## `BiCop`

**Input contract**

- `fit(obs=...)`: `obs` must have shape `[num_obs, 2]`.
- Query methods expect shape `[batch, 2]`.
- The default runtime dtype is `float64`.
- `log_pdf()`, `cdf()`, and `hfunc_*()` are differentiable query paths.
- Public bicop backends are `grid_reflect`, `grid_probit`, `ttcv`, `ttpi`, `tll1`, `tll2`,
  `tll1nn`, `tll2nn`, `beta`, `beta_qt`, `spline_pen`, and `tll_ref`.
- The aligned `kdecopula` names now use canonical `bandwidth` object shapes plus `mult`:
  `ttpi` / `ttcv` use `(h, rho, theta1, theta2)`, `tll1` / `tll2` use `2x2` matrices,
  `tll1nn` / `tll2nn` use `{B, alpha, kappa}`, and `beta` uses a scalar `bw`.
- `beta_qt` and `spline_pen` remain repository-native research backends, and `tll_ref` remains the
  CPU reference path backed by `pyvinecopulib`.

Detailed reference:

- [BiCop detail page](../_api_stubs/torchvinecopulib.BiCop.rst)

## `BiCopDiagnostics`

`BiCopDiagnostics` is the immutable snapshot returned by `BiCop.diagnostics()`.

It records left/right query-path fallback counters and worst-case residual errors:

- `itp_failures_l`, `itp_failures_r`
- `bisect_refinements_l`, `bisect_refinements_r`
- `fallback_to_indep_l`, `fallback_to_indep_r`
- `max_abs_hfunc_error_l`, `max_abs_hfunc_error_r`

Detailed reference:

- [BiCopDiagnostics detail page](../_api_stubs/torchvinecopulib.BiCopDiagnostics.rst)

## `GridReflectBicopEstimator`

`GridReflectBicopEstimator` is the reflected-grid 2D KDE class used by
`bicop_backend="grid_reflect"`. It is not the same thing as the newer `tt*`, `beta`, `tll*`, or
`spline_pen` research backends.

Detailed reference:

- [GridReflectBicopEstimator detail page](../_api_stubs/torchvinecopulib.GridReflectBicopEstimator.rst)
