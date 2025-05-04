## Specify project root

Put a `.env` file in project root folder to specify the project root folder location as an env var:

```bash
DIR_WORK='~/path/to/project/root/folder'
```

## Environment setup

In project root folder run in bash:

```bash
uv sync --extra cpu -U
```

For `LightGBM` benchmark it may need `libomp`:

```bash
brew install libomp
```

## Run

`cd` into folder `DIR_WORK/examples/` where `quantile_regression` is visible as a child folder, trigger by:

```bash
uv run -m quantile_regression
```
