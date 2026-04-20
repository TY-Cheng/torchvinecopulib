# Examples and benchmarks

## Examples

Repository examples live under `examples/`. Use them when you need fuller workflows than the
quickstart pages.

Maintained example scripts:

1. [`examples/0_bicop.py`](https://github.com/TY-Cheng/torchvinecopulib/blob/main/examples/0_bicop.py):
   fit a bivariate copula and compare fitted samples to the observations.
2. [`examples/1_vinecop.py`](https://github.com/TY-Cheng/torchvinecopulib/blob/main/examples/1_vinecop.py):
   fit a multivariate vine and inspect the learned dependency structure.
3. [`examples/2_num_hfunc.py`](https://github.com/TY-Cheng/torchvinecopulib/blob/main/examples/2_num_hfunc.py):
   study how sampling order changes the number of `hfunc` calls.
4. [`examples/3_bicop_tll.py`](https://github.com/TY-Cheng/torchvinecopulib/blob/main/examples/3_bicop_tll.py):
   compare several torch-native bicop backends on the same pseudo-observations.

These four scripts remain the single source of truth for the docs-facing example figures. The docs
page links directly to the tracked local source files above, and the same scripts regenerate the
committed figures below:

```bash
uv run --extra cpu --group examples python scripts/generate_example_assets.py
```

The generated docs figures below are exported by the same pipeline and committed under
`docs/_static/examples/`.

??? example "`examples/0_bicop.py`"

    ```python
    --8<-- "examples/0_bicop.py"
    ```

![Bicop sample cloud](_static/examples/bicop_sample.svg)

??? example "`examples/1_vinecop.py`"

    ```python
    --8<-- "examples/1_vinecop.py"
    ```

![Vine structure](_static/examples/vinecop_structure.svg)

??? example "`examples/2_num_hfunc.py`"

    ```python
    --8<-- "examples/2_num_hfunc.py"
    ```

![Sampling-order h-function counts](_static/examples/num_hfunc.svg)

??? example "`examples/3_bicop_tll.py`"

    ```python
    --8<-- "examples/3_bicop_tll.py"
    ```

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
