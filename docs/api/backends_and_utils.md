# Backends and utilities

```{currentmodule} torchvinecopulib
```

## Overview

These objects support the public vine API but are also useful on their own when you need
standalone marginals, dependence measures, or root-finding utilities.

## Autosummary

```{autosummary}
TorchKDE1D
util.ENUM_FUNC_BIDEP
util.empirical_pobs
util.kendall_tau
util.kendall_tau_matrix
util.solve_ITP
```

```{toctree}
:hidden:

../_api_stubs/torchvinecopulib.TorchKDE1D
../_api_stubs/torchvinecopulib.util.ENUM_FUNC_BIDEP
../_api_stubs/torchvinecopulib.util.empirical_pobs
../_api_stubs/torchvinecopulib.util.kendall_tau
../_api_stubs/torchvinecopulib.util.kendall_tau_matrix
../_api_stubs/torchvinecopulib.util.solve_ITP
```

## `TorchKDE1D`

```{autoclass} torchvinecopulib.TorchKDE1D
:members: __init__, fit, cdf, pdf, ppf
:show-inheritance:
```

## Dependence measures

```{autoclass} torchvinecopulib.util.ENUM_FUNC_BIDEP
:members:
```

```{autofunction} torchvinecopulib.util.kendall_tau
```

```{autofunction} torchvinecopulib.util.kendall_tau_matrix
```

```{autofunction} torchvinecopulib.util.empirical_pobs
```

## Root finding

```{autofunction} torchvinecopulib.util.solve_ITP
```
