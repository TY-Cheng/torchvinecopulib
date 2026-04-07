# Quickstart

This page keeps the shortest working paths aligned with the public API. The examples below are
executed in CI through the Sphinx doctest builder.

## Fit, score, and sample

```{testsetup} quickstart-fit
import torch
import torchvinecopulib as tvc

torch.manual_seed(0)
obs = torch.rand(96, 4, dtype=torch.float64)
```

```{testcode} quickstart-fit
vc = tvc.VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33)
vc.fit(
    obs,
    mtd_vine="cvine",
    mtd_bidep="kendall_tau",
    bicop_backend="grid_reflect",
)
log_pdf = vc.log_pdf(obs[:8])
samples = vc.sample(num_sample=16, seed=0)

assert tuple(log_pdf.shape) == (8, 1)
assert tuple(samples.shape) == (16, 4)
assert torch.isfinite(log_pdf).all()
assert torch.isfinite(samples).all()
```

This path is the standard entrypoint when observations are already on copula scale with shape
`[num_obs, num_dim]`.

## Rosenblatt roundtrip

```{testsetup} quickstart-rosenblatt
import torch
import torchvinecopulib as tvc

torch.manual_seed(1)
obs = torch.rand(96, 4, dtype=torch.float64)
vc = tvc.VineCop(num_dim=4, is_cop_scale=True, num_step_grid=33)
vc.fit(
    obs,
    mtd_vine="rvine",
    mtd_bidep="kendall_tau",
    bicop_backend="grid_reflect",
)
```

```{testcode} quickstart-rosenblatt
u = vc.rosenblatt(obs[:8])
recovered = vc.inverse_rosenblatt(u)

assert tuple(u.shape) == (8, 4)
assert tuple(recovered.shape) == (8, 4)
assert torch.isfinite(u).all()
assert torch.isfinite(recovered).all()
```

Use this path when you need conditional simulation, likelihood-based transforms, or copula-space
integration with other probabilistic models.

## Where to go next

- Read [Vine decomposition](theory/vine_decomposition.md) for the pair-copula factorization.
- Read [Builder and engine](systems/builder_engine.md) for execution-plan semantics.
- Jump to the [VineCop API](api/vinecop.md) for full method contracts.
