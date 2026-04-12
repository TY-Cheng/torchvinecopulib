# Backends and utilities

```{currentmodule} torchvinecopulib
```

## Overview

These objects support the public vine API but are also useful on their own when you need
standalone marginals, dependence measures, or root-finding utilities.

Implementation notes for the torch-native marginal and bicop estimators live in the
[Systems: estimator backends](../systems/backends.md) page.

```{toctree}
:hidden:

../_api_stubs/torchvinecopulib.GridKDE1D
../_api_stubs/torchvinecopulib.backends.LocalPolynomialKDE1D
../_api_stubs/torchvinecopulib.util.ENUM_FUNC_BIDEP
../_api_stubs/torchvinecopulib.util.empirical_pobs
../_api_stubs/torchvinecopulib.util.kendall_tau
../_api_stubs/torchvinecopulib.util.kendall_tau_matrix
../_api_stubs/torchvinecopulib.util.solve_ITP
```

## `GridKDE1D`

`GridKDE1D` is the regular-grid torch-native 1D KDE class used by the default
`marginal_backend="grid"` path.

Detailed reference:

- [GridKDE1D detail page](../_api_stubs/torchvinecopulib.GridKDE1D.rst)

## `LocalPolynomialKDE1D`

`LocalPolynomialKDE1D` is the torch-native continuous local-polynomial marginal estimator exposed
under `marginal_backend="lp"`.

Detailed reference:

- [LocalPolynomialKDE1D detail page](../_api_stubs/torchvinecopulib.backends.LocalPolynomialKDE1D.rst)

## Dependence measures

| Object | Purpose |
| --- | --- |
| `ENUM_FUNC_BIDEP` | Enum wrapper for the built-in bivariate dependence measures. |
| `kendall_tau` | Pairwise Kendall's tau and p-value for two vectors. |
| `kendall_tau_matrix` | Kendall's tau and p-value matrices for a 2D observation tensor. |
| `empirical_pobs` | Rank-transform a 1D sample into pseudo-observations on `(0, 1)`. |

## `ENUM_FUNC_BIDEP`

`ENUM_FUNC_BIDEP` exposes the built-in pairwise dependence measures through a callable enum.

Current members:

- `chatterjee_xi`
- `ferreira_tail_dep_coeff`
- `kendall_tau`
- `mutual_info`

Detailed reference:

- [ENUM_FUNC_BIDEP detail page](../_api_stubs/torchvinecopulib.util.ENUM_FUNC_BIDEP.rst)

## `kendall_tau`

`kendall_tau(x, y)` returns a length-2 tensor containing the Kendall tau statistic and its
associated p-value. The backend can be selected explicitly or left on `backend="auto"`.

Detailed reference:

- [kendall_tau detail page](../_api_stubs/torchvinecopulib.util.kendall_tau.rst)

## `kendall_tau_matrix`

`kendall_tau_matrix(x)` computes square tau and p-value matrices for multivariate observations
shaped `[num_obs, num_dim]`.

Detailed reference:

- [kendall_tau_matrix detail page](../_api_stubs/torchvinecopulib.util.kendall_tau_matrix.rst)

## `empirical_pobs`

`empirical_pobs(x)` converts a 1D sample into empirical pseudo-observations using rank order and
the standard `rank / (n + 1)` normalization.

Detailed reference:

- [empirical_pobs detail page](../_api_stubs/torchvinecopulib.util.empirical_pobs.rst)

## Root finding

`solve_ITP()` is the torch-native scalar root finder used by inverse conditional CDF paths. It
implements an interval-preserving ITP iteration with optional fallback policies for unbracketed or
non-converged samples.

Detailed reference:

- [solve_ITP detail page](../_api_stubs/torchvinecopulib.util.solve_ITP.rst)
