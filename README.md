# TorchVineCopuLib

Yet another vine copula package, using [PyTorch](https://pytorch.org/get-started/locally/).

- C/D/R-Vine full-sampling/ quantile-regression/ conditional-sampling, all in one package
  - Flexible sampling order for experienced users
- Vectorized tensor computation with GPU (`device='cuda'`) support
- Shorter runtimes for higher dimension simulations
- Pure `Python` library, inspired by [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib/) on Windows, Linux, MacOS
- IO and visualization support

## Installation

### [uv](https://docs.astral.sh/uv/getting-started/) for Dependency Management and Packaging

`cd` into the project root where [`pyproject.toml`](./pyproject.toml) exists,

```bash
# inside project root folder
uv sync --extra cpu -U
# or
uv sync --extra cu126 -U
```

## Examples

Visit the [`./examples/`](./examples/) folder for `.ipynb` Jupyter notebooks.

## Dependencies

```toml
# inside the `./pyproject.toml` file;
python = ">=3.12"
numpy = ">=2"
scipy = "*"
fastkde = "*"
pyvinecopulib = "*"
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
