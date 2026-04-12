# Benchmark interpretation

The repository ships four benchmark entrypoints:

- `benchmarks/profile_builder.py`
- `benchmarks/profile_query.py`
- `benchmarks/compare_bicop_backends.py`
- `benchmarks/compare_vinecop_runtimes.py`

All of them emit a stable JSON schema with:

- `schema_version`
- `benchmark`
- `config`
- `environment`

The profiling scripts (`profile_builder.py`, `profile_query.py`) emit a top-level `metrics` object.
The comparison scripts (`compare_bicop_backends.py`, `compare_vinecop_runtimes.py`) emit
top-level `results`, and may also include `summary`.

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

`compare_bicop_backends.py` is the benchmark used to compare the torch-native 2D bicop backends.
It is intentionally synthetic and deliberately varied:

- elliptical truths: Gaussian and Student-t
- Archimedean truths: Clayton, Gumbel, Frank, Joe
- rotated tail scenarios
- awkward mixtures with randomized weights and parameters
- repeated sweeps across train sizes, grid sizes, and randomized seeds

For each backend it records:

- held-out log-density and oracle log-density gap
- grid-level density error
- `cdf` and `hfunc` error
- inverse-hfunction roundtrip error
- tail/corner mass error
- fit/query timings
- `BiCopDiagnostics` fallback counters

The summary score uses the current repository policy:

- 50% accuracy
- 30% numerical stability
- 20% speed

At the time of writing, that policy keeps `beta` as the default torch-native `bicop_backend`.

Only torch-native backends are eligible for the default `bicop_backend`. `tll_ref` remains a
reference/oracle path even when included in the benchmark.

`compare_vinecop_runtimes.py` compares end-to-end `VineCop` fit / sample / density-query runtime across
`torchvinecopulib` CPU, optional CUDA, and optional `pyvinecopulib` reference runs, and writes JSON
plus Markdown artifacts under `benchmarks/results/`.

## How to run locally

```bash
uv run --extra cpu python benchmarks/profile_builder.py --device cpu
uv run --extra cpu python benchmarks/profile_query.py --device cpu
uv run --extra cpu --extra reference python benchmarks/compare_bicop_backends.py --device cpu
uv run --extra cpu --extra reference python benchmarks/compare_vinecop_runtimes.py --include-reference yes
```

The GitHub Actions workflow uploads benchmark JSON artifacts from a non-blocking job so trend
tracking does not interfere with the default development loop.
