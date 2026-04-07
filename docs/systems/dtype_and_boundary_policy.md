# Dtype and boundary policy

## Dtype policy

`torchvinecopulib` defaults to `float64` for fitting and exported execution plans. This is a
deliberate statistical choice:

- KDE-based grid fitting accumulates numerical error faster in `float32`.
- `hinv()` routines and cumulative buffer construction are more stable in `float64`.
- Reference-parity checks are easier to interpret when the default path is high precision.

When you need a lighter runtime copy, export a plan with a different dtype:

```python
plan_fp32 = vc.export_inference_plan(dtype=torch.float32)
vc_fp32 = tvc.VineCop.from_artifact(plan_fp32)
```

## Boundary policy

`BiCop` and `VineCop` expose `boundary_policy="hard" | "st"` for query-time evaluation.

- `hard`: clamps query inputs to the valid copula domain.
- `st`: uses a straight-through clamp so the forward pass remains numerically safe while the
  backward pass preserves the interior gradient signal.

Sampling and builder-time routines still use hard bounds for stability.

## Buffer semantics

The 2D cumulative buffers used for `cdf()` and `hfunc()` are built with trapezoidal integration so
the discrete grid respects the expected copula boundary conditions:

$$
C(0, v) = 0,\qquad C(u, 0) = 0,\qquad C(1, 1) = 1.
$$

This is important because query-time interpolation should not have to compensate for a numerically
incorrect grid.
