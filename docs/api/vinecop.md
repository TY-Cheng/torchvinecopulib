# VineCop API

## Overview

The multivariate API is split between a public façade (`VineCop`), a builder (`VineBuilder`), a
plan-backed runtime (`VineCopEngine`), and serializable artifacts (`VineBuildArtifact`).

| Object | Purpose |
| --- | --- |
| `VineCop` | Public multivariate façade for fitting, query-time evaluation, and serialization. |
| `VineBuilder` | Explicit structure-learning and plan-assembly entrypoint. |
| `VineCopEngine` | Query-time runtime for plan-backed execution. |
| `VineBuildArtifact` | Serializable fitted structure and execution plan. |
| `VineDiagnostics` | Aggregated diagnostics across pair-copula query paths. |

## `VineCop`

**Input contract**

- `fit(obs=...)` expects shape `[num_obs, num_dim]`.
- When `is_cop_scale=False`, marginal transforms are part of the builder path.
- Query methods return batched tensors with a leading dimension equal to the query batch size.
- `fit()` is not a differentiable training step.
- `log_pdf()`, `cdf()`, and `rosenblatt()` are differentiable query paths.
- `sample()` and `inverse_rosenblatt()` are stabilized runtime operators and are not documented as
  differentiable.

::: torchvinecopulib.VineCop
    options:
      heading_level: 3
      show_root_heading: false

## `VineBuilder`

::: torchvinecopulib.VineBuilder
    options:
      heading_level: 3
      show_root_heading: false

## `VineCopEngine`

::: torchvinecopulib.VineCopEngine
    options:
      heading_level: 3
      show_root_heading: false

## `VineBuildArtifact` and diagnostics

`VineBuildArtifact` is the serialized fit result and execution-plan payload used by
`VineCop.from_artifact()` and `export_inference_plan()`. It stores:

- fitted marginals and pair-copula modules
- learned structure metadata and edge ordering
- backend configuration and boundary policy
- static execution tensors for forward, log-density, and sampling paths

::: torchvinecopulib.VineBuildArtifact
    options:
      heading_level: 3
      show_root_heading: false

`VineDiagnostics` is the aggregate multivariate snapshot returned by `VineCop.diagnostics()`.
It records:

- `num_edges`
- `itp_failures`
- `bisect_refinements`
- `fallback_to_indep`
- `max_abs_hfunc_error`

::: torchvinecopulib.VineDiagnostics
    options:
      heading_level: 3
      show_root_heading: false
