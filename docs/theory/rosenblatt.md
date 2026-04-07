# Rosenblatt transforms

`torchvinecopulib` exposes both forward and inverse Rosenblatt transforms through
[`VineCop.rosenblatt()`](../api/vinecop.md) and
[`VineCop.inverse_rosenblatt()`](../api/vinecop.md).

For a copula-distributed random vector $U$, the Rosenblatt transform maps correlated uniforms to
independent uniforms by chaining conditional CDF evaluations:

$$
z_1 = u_1,\qquad
z_2 = C_{2 \mid 1}(u_2 \mid u_1),\qquad
z_3 = C_{3 \mid 1,2}(u_3 \mid u_1, u_2), \dots
$$

The inverse Rosenblatt transform reverses the process by repeatedly solving inverse conditional
distribution problems. In this library, the inverse path uses plan-backed `hinv()` calls,
stabilized root finding, and independence fallbacks as a last-resort statistical degradation.

## Why it matters

- Sampling from a fitted vine copula is an inverse Rosenblatt problem.
- Conditional simulation is a Rosenblatt / inverse Rosenblatt problem with partial source values.
- Flow-style models and likelihood decompositions often need these transforms explicitly.

## Numerical notes

- The builder path does not preserve a training graph.
- The query path is differentiable where the underlying interpolation kernels are differentiable.
- Boundary handling is controlled by `boundary_policy` on query-time pair-copula evaluation; see
  [Dtype and boundary policy](../systems/dtype_and_boundary_policy.md).
