# Examples and benchmarks

## Examples

Repository examples live under
[`examples/`](https://github.com/TY-Cheng/torchvinecopulib/tree/main/examples). Use them when you
need fuller workflows than the quickstart pages.

Suggested reading order:

1. Fit a multivariate vine and evaluate `log_pdf()`.
2. Export an inference plan and reload it on a different dtype or device.
3. Integrate `rosenblatt()` / `inverse_rosenblatt()` into a larger probabilistic workflow.

## Benchmarks

The benchmark scripts live under
[`benchmarks/`](https://github.com/TY-Cheng/torchvinecopulib/tree/main/benchmarks).

```bash
uv run --extra cpu python benchmarks/profile_builder.py --device cpu
uv run --extra cpu python benchmarks/profile_query.py --device cpu
```

The JSON outputs include configuration, environment metadata, and metrics so they can be uploaded
as CI artifacts and compared across commits.
