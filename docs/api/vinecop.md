# VineCop API

```{currentmodule} torchvinecopulib
```

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
- `fit()` is not a differentiable training step.
- `log_pdf()`, `cdf()`, and `rosenblatt()` are differentiable query paths.
- `sample()` and `inverse_rosenblatt()` are stabilized runtime operators and are not documented as
  differentiable.

Detailed reference:

- [VineCop detail page](../_api_stubs/torchvinecopulib.VineCop.rst)

## `VineBuilder`

Detailed reference:

- [VineBuilder detail page](../_api_stubs/torchvinecopulib.VineBuilder.rst)

## `VineCopEngine`

Detailed reference:

- [VineCopEngine detail page](../_api_stubs/torchvinecopulib.VineCopEngine.rst)

## `VineBuildArtifact` and diagnostics

`VineBuildArtifact` is the serialized fit result and execution-plan payload used by
`VineCop.from_artifact()` and `export_inference_plan()`. It stores:

- fitted marginals and pair-copula modules
- learned structure metadata and edge ordering
- backend configuration and boundary policy
- static execution tensors for forward, log-density, and sampling paths

Detailed reference:

- [VineBuildArtifact detail page](../_api_stubs/torchvinecopulib.VineBuildArtifact.rst)

`VineDiagnostics` is the aggregate multivariate snapshot returned by `VineCop.diagnostics()`.
It records:

- `num_edges`
- `itp_failures`
- `bisect_refinements`
- `fallback_to_indep`
- `max_abs_hfunc_error`

Detailed reference:

- [VineDiagnostics detail page](../_api_stubs/torchvinecopulib.VineDiagnostics.rst)
