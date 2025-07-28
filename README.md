## Examples

Visit the `./examples/` folder for `.ipynb` Jupyter notebooks.

## Installation

### (Recommended) [uv](https://docs.astral.sh/uv/getting-started/) for Dependency Management and Packaging

```bash
# inside project root folder
uv sync --extra cpu -U
# or
uv sync --extra cu126 -U
```

## Dependencies

```toml
# inside the `./pyproject.toml` file;
fastkde = "*"
numpy = "*"
pyvinecopulib = "*"
python = ">=3.11"
scipy = "*"
# optional to facilitate customization
torch = [
    { index = "torch-cpu", extra = "cpu" },
    { index = "torch-cu126", extra = "cu126" },
]
```

For [PyTorch](https://pytorch.org/get-started/locally/) with `cuda`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
# check cuda availability
python -c "import torch; print(torch.cuda.is_available())"
```

> [!TIP]
> macOS users should set `device='cpu'` at this stage, for using `device='mps'` won't support `dtype=torch.float64`.


## License

This project is released under the MIT License (© 2024- Anonymous).  
See [LICENSE](./LICENSE) for the full text, including our own grant of rights and disclaimer.

### Third-Party Dependencies

See the “Third-Party Dependencies” section in [LICENSE](./LICENSE) for details on the `PyTorch`, `FastKDE`, and `pyvinecopulib` licenses that govern those components.
