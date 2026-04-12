# Examples and benchmarks

## Examples

Repository examples live under `examples/`. Use them when you need fuller workflows than the
quickstart pages.

Maintained example scripts:

1. {download}`examples/0_bicop.py <../examples/0_bicop.py>`:
   fit a bivariate copula and compare fitted samples to the observations.
2. {download}`examples/1_vinecop.py <../examples/1_vinecop.py>`:
   fit a multivariate vine and inspect the learned dependency structure.
3. {download}`examples/2_num_hfunc.py <../examples/2_num_hfunc.py>`:
   study how sampling order changes the number of `hfunc` calls.
4. {download}`examples/3_bicop_tll.py <../examples/3_bicop_tll.py>`:
   compare several torch-native bicop backends on the same pseudo-observations.

These four scripts remain the single source of truth for the docs-facing example figures. The docs
page links directly to the tracked local source files above, and the same scripts regenerate the
committed figures below:

```bash
uv run --extra cpu --group examples python scripts/generate_example_assets.py
```

The generated docs figures below are exported by the same pipeline and committed under
`docs/_static/examples/`.

::::{dropdown} `examples/0_bicop.py`
:open: false

```{literalinclude} ../examples/0_bicop.py
:language: python
:lines: 18-61
:caption: examples/0_bicop.py
```
::::

![Bicop sample cloud](_static/examples/bicop_sample.svg)

::::{dropdown} `examples/1_vinecop.py`
:open: false

```{literalinclude} ../examples/1_vinecop.py
:language: python
:lines: 18-60
:caption: examples/1_vinecop.py
```
::::

![Vine structure](_static/examples/vinecop_structure.svg)

::::{dropdown} `examples/2_num_hfunc.py`
:open: false

```{literalinclude} ../examples/2_num_hfunc.py
:language: python
:lines: 45-112
:caption: examples/2_num_hfunc.py
```
::::

![Sampling-order h-function counts](_static/examples/num_hfunc.svg)

::::{dropdown} `examples/3_bicop_tll.py`
:open: false

```{literalinclude} ../examples/3_bicop_tll.py
:language: python
:lines: 19-75
:caption: examples/3_bicop_tll.py
```
::::

![Bicop backend comparison](_static/examples/bicop_backend_comparison.svg)

Heavier experiment code:

- `examples/pred_intvl/experiments.py`
- `examples/vcae/run_seeds.py`
- `examples/vcae/vcae/`

These stay outside the asset-generation pipeline because they depend on longer training runs,
dataset downloads, or project-local experiment loops. The maintained docs surface is based on
executable example scripts plus committed static figures.

## Benchmarks

The benchmark scripts live under
[`benchmarks/`](https://github.com/TY-Cheng/torchvinecopulib/tree/main/benchmarks).

```bash
uv run --extra cpu python benchmarks/profile_builder.py --device cpu
uv run --extra cpu python benchmarks/profile_query.py --device cpu
uv run --extra cpu --extra reference python benchmarks/compare_vinecop_runtimes.py --include-reference yes
```

The JSON outputs include configuration, environment metadata, and benchmark metrics or per-workload
results so they can be uploaded as CI artifacts and compared across commits.
