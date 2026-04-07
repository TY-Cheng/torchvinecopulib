# Benchmark interpretation

The repository ships two benchmark entrypoints:

- `benchmarks/profile_builder.py`
- `benchmarks/profile_query.py`

Both emit a stable JSON schema with:

- `schema_version`
- `benchmark`
- `config`
- `environment`
- `metrics`

## What to compare

Builder metrics should be read as structure-learning costs:

- fit time,
- peak memory,
- state size,
- and dependence-measure backend behavior.

Query metrics should be read as execution costs:

- `log_pdf`,
- `sample`,
- `cdf`,
- `rosenblatt`,
- `inverse_rosenblatt`,
- and optional compile smoke measurements.

## How to run locally

```bash
uv run --extra cpu python benchmarks/profile_builder.py --device cpu
uv run --extra cpu python benchmarks/profile_query.py --device cpu
```

The GitHub Actions workflow uploads benchmark JSON artifacts from a non-blocking job so trend
tracking does not interfere with the default development loop.
