# API Reference

The API reference is organized around stable user-facing entrypoints rather than raw module dumps.

## Stable public surface

- Pair copulas: `BiCop`, `BiCopDiagnostics`, `GridReflectBicopEstimator`
- Multivariate vines: `VineCop`, `VineBuilder`, `VineCopEngine`, `VineBuildArtifact`,
  `VineDiagnostics`
- Utilities and standalone backends: `GridKDE1D`,
  `torchvinecopulib.backends.LocalPolynomialKDE1D`, `kendall_tau()`, `kendall_tau_matrix()`,
  `solve_ITP()`, `empirical_pobs()`, `ENUM_FUNC_BIDEP`

## Entry points

- [BiCop API](bicop.md)
- [VineCop API](vinecop.md)
- [Backends and utilities](backends_and_utils.md)
