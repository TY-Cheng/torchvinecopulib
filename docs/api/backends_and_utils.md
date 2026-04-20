# Backends and utilities

## Overview

These objects support the public vine API but are also useful on their own when you need
standalone marginals, dependence measures, or root-finding utilities.

Implementation notes for the torch-native marginal and bicop estimators live in the
[Systems: estimator backends](../systems/backends.md) page.

## `GridKDE1D`

`GridKDE1D` is the regular-grid torch-native 1D KDE class used by the default
`marginal_backend="grid"` path.

::: torchvinecopulib.GridKDE1D
    options:
      heading_level: 3
      show_root_heading: false

## `LocalPolynomialKDE1D`

`LocalPolynomialKDE1D` is the torch-native continuous local-polynomial marginal estimator exposed
under `marginal_backend="lp"`.

::: torchvinecopulib.backends.LocalPolynomialKDE1D
    options:
      heading_level: 3
      show_root_heading: false

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

::: torchvinecopulib.util.ENUM_FUNC_BIDEP
    options:
      heading_level: 3
      show_root_heading: false

## `kendall_tau`

`kendall_tau(x, y)` returns a length-2 tensor containing the Kendall tau statistic and its
associated p-value. The backend can be selected explicitly or left on `backend="auto"`.

::: torchvinecopulib.util.kendall_tau
    options:
      heading_level: 3
      show_root_heading: false

## `kendall_tau_matrix`

`kendall_tau_matrix(x)` computes square tau and p-value matrices for multivariate observations
shaped `[num_obs, num_dim]`.

::: torchvinecopulib.util.kendall_tau_matrix
    options:
      heading_level: 3
      show_root_heading: false

## `empirical_pobs`

`empirical_pobs(x)` converts a 1D sample into empirical pseudo-observations using rank order and
the standard `rank / (n + 1)` normalization.

::: torchvinecopulib.util.empirical_pobs
    options:
      heading_level: 3
      show_root_heading: false

## Root finding

`solve_ITP()` is the torch-native scalar root finder used by inverse conditional CDF paths. It
implements an interval-preserving ITP iteration with optional fallback policies for unbracketed or
non-converged samples.

::: torchvinecopulib.util.solve_ITP
    options:
      heading_level: 3
      show_root_heading: false
