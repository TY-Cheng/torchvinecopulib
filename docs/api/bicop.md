# BiCop API

```{currentmodule} torchvinecopulib
```

## Overview

`BiCop` is the public bivariate copula interface. It owns the fitted density and conditional CDF
grids, exposes differentiable query methods, and handles stabilized inverse conditional sampling.

| Object | Purpose |
| --- | --- |
| `BiCop` | Fit and query a bivariate copula on observations shaped `[num_obs, 2]`. |
| `BiCopDiagnostics` | Query-time counters for root-finding fallbacks and residual errors. |
| `TorchCopulaKDE2D` | Torch-native 2D KDE backend used by grid-based pair-copula estimators. |

## Autosummary

```{autosummary}
BiCop
BiCopDiagnostics
TorchCopulaKDE2D
```

```{toctree}
:hidden:

../_api_stubs/torchvinecopulib.BiCop
../_api_stubs/torchvinecopulib.BiCopDiagnostics
../_api_stubs/torchvinecopulib.TorchCopulaKDE2D
```

## `BiCop`

**Input contract**

- `fit(obs=...)`: `obs` must have shape `[num_obs, 2]`.
- Query methods expect shape `[batch, 2]`.
- The default runtime dtype is `float64`.
- `log_pdf()`, `cdf()`, and `hfunc_*()` are differentiable query paths.

```{autoclass} torchvinecopulib.BiCop
:members: __init__, fit, reset, cdf, hfunc_l, hfunc_r, hinv_l, hinv_r, pdf, log_pdf, sample, diagnostics, imshow, plot
:show-inheritance:
```

## `BiCopDiagnostics`

```{autoclass} torchvinecopulib.BiCopDiagnostics
:members:
```

## `TorchCopulaKDE2D`

```{autoclass} torchvinecopulib.TorchCopulaKDE2D
:members: __init__, fit, pdf, cdf, hfunc_l, hfunc_r
:show-inheritance:
```
