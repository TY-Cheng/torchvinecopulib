# VineCop API

```{currentmodule} torchvinecopulib
```

## Overview

The multivariate API is split between a public façade (`VineCop`), a builder (`VineBuilder`), a
plan-backed runtime (`VineCopEngine`), and serializable artifacts (`VineBuildArtifact` /
`VineExecutionPlan`).

| Object | Purpose |
| --- | --- |
| `VineCop` | Public multivariate façade for fitting, query-time evaluation, and serialization. |
| `VineBuilder` | Explicit structure-learning and plan-assembly entrypoint. |
| `VineCopEngine` | Query-time runtime for plan-backed execution. |
| `VineBuildArtifact` | Serializable fitted structure and execution plan. |
| `VineExecutionPlan` | Alias of `VineBuildArtifact`, emphasizing runtime usage. |
| `VineDiagnostics` | Aggregated diagnostics across pair-copula query paths. |

## Autosummary

```{autosummary}
VineCop
VineBuilder
VineCopEngine
VineBuildArtifact
VineDiagnostics
```

```{toctree}
:hidden:

../_api_stubs/torchvinecopulib.VineCop
../_api_stubs/torchvinecopulib.VineBuilder
../_api_stubs/torchvinecopulib.VineCopEngine
../_api_stubs/torchvinecopulib.VineBuildArtifact
../_api_stubs/torchvinecopulib.VineDiagnostics
```

## `VineCop`

**Input contract**

- `fit(obs=...)` expects shape `[num_obs, num_dim]`.
- When `is_cop_scale=False`, marginal transforms are part of the builder path.
- Query methods return batched tensors with a leading dimension equal to the query batch size.
- `fit()` is not a differentiable training step; `log_pdf()` and transform methods are.

```{autoclass} torchvinecopulib.VineCop
:members: __init__, fit, log_pdf, forward, rosenblatt, inverse_rosenblatt, sample, cdf, diagnostics, export_inference_plan, draw_lv, draw_dag
:show-inheritance:
```

## `VineBuilder`

```{autoclass} torchvinecopulib.VineBuilder
:members: __init__, build
```

## `VineCopEngine`

```{autoclass} torchvinecopulib.VineCopEngine
:members: __init__, log_pdf, rosenblatt, inverse_rosenblatt, sample, cdf, forward
:show-inheritance:
```

## `VineBuildArtifact` and diagnostics

`VineExecutionPlan` is an alias of `VineBuildArtifact`. Use the alias when you want to emphasize
runtime execution rather than builder provenance.

```{autoclass} torchvinecopulib.VineBuildArtifact
:members:
```

```{autoclass} torchvinecopulib.VineDiagnostics
:members:
```
