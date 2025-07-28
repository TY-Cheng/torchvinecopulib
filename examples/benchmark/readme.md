## Specify project root

Put a `.env` file in project root folder to specify the project root folder location as an env var:

```bash
DIR_WORK='~/path/to/project/root/folder'
```

## Environment setup

In project root folder run in bash:

```bash
# cpu only
uv sync --extra cpu -U
# or with cuda
uv sync --extra cu126 -U
```

## Run

`cd` into project root folder, trigger by:

```bash
uv run ./examples/benchmark/__init__.py
```
