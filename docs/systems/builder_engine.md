# Builder and engine

The multivariate stack is intentionally split into three public integration points:

- `VineBuilder`: structure learning, backend dispatch, and execution-plan assembly.
- `VineBuildArtifact`: a serializable description of the fitted structure and static execution ops.
- `VineCopEngine`: the query-time runtime that executes `log_pdf()`, `rosenblatt()`,
  `inverse_rosenblatt()`, `cdf()`, and `sample()` from the plan.

## Design boundary

`fit()` is a builder path. It may allocate intermediate tensors, learn tree structure, and fit
pair-copula or marginal backends, but it does **not** preserve a training graph for later
backpropagation.

Once a vine is fitted, query calls execute through a plan-backed engine:

- pseudo-observations live in a flat slot-indexed tensor pool,
- op sequences are precomputed during export,
- and query routines no longer depend on dynamic `dict[tuple, Tensor]` traversal.

## Typical workflow

```python
import torchvinecopulib as tvc

vc = tvc.VineCop(num_dim=4, is_cop_scale=True)
artifact = tvc.VineBuilder(vc).build(obs, mtd_bidep="kendall_tau", bidep_backend="torch")
clone = tvc.VineCop.from_artifact(artifact)
scores = clone.engine.log_pdf(obs[:32])
u = clone.engine.rosenblatt(obs[:32])
recovered = clone.engine.inverse_rosenblatt(u)
```

## Deployment notes

- Use `export_inference_plan(dtype=...)` when you want a dtype-controlled copy for inference-only
  deployment.
- Use `diagnostics()` after heavy query workloads to inspect fallback counts and worst-case
  conditional CDF errors.
- Treat the exported plan as the stable runtime artifact; treat the fitted `VineCop` as the
  public façade and builder entrypoint.
