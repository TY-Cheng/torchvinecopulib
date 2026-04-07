# API Reference

The API reference is organized around stable user-facing entrypoints rather than raw module dumps.

```{toctree}
:maxdepth: 1

bicop
vinecop
backends_and_utils
```

## Stable public surface

- Pair copulas: `BiCop`, `BiCopDiagnostics`, `TorchCopulaKDE2D`
- Multivariate vines: `VineCop`, `VineBuilder`, `VineCopEngine`, `VineBuildArtifact`,
  `VineExecutionPlan`, `VineDiagnostics`
- Utilities: `TorchKDE1D`, `kendall_tau()`, `kendall_tau_matrix()`, `solve_ITP()`,
  `empirical_pobs()`, `ENUM_FUNC_BIDEP`
