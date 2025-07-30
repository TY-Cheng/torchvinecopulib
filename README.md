# torchvinecopulib

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e8a7a7448b2043d9bbefafc5a3ec14f7)](https://app.codacy.com/gh/TY-Cheng/torchvinecopulib/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/e8a7a7448b2043d9bbefafc5a3ec14f7)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![Lint Pytest](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml)
[![Deploy Docs](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/static.yml/badge.svg?branch=main)](https://ty-cheng.github.io/torchvinecopulib/)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchvinecopulib)
[![OS](https://img.shields.io/badge/OS-Windows%7CmacOS%7CUbuntu-blue)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/TY-Cheng/torchvinecopulib/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/torchvinecopulib)](https://pypi.org/project/torchvinecopulib/)

A Python library for fitting and sampling vine copulas, using [PyTorch](https://pytorch.org/get-started/locally/).

- C/D/R-Vine full-sampling/ quantile-regression/ conditional-sampling, all in one package
  - Flexible sampling order for experienced users
- Vectorized tensor computation with GPU (`device='cuda'`) support
- Shorter runtimes for higher dimension simulations
- Pure `Python` library, inspired by [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib/) on Windows, Linux, MacOS
- IO and visualization support

## Citation
If you use `torchvinecopulib` in your work, please cite:

> Cheng, Tuoyuan, Thibault Vatter, Thomas Nagler, and Kan Chen. "Vine Copulas as Differentiable Computational Graphs." arXiv preprint arXiv:2506.13318 (2025).

```latex
@article{cheng2025vine,
  title={Vine Copulas as Differentiable Computational Graphs},
  author={Cheng, Tuoyuan and Vatter, Thibault and Nagler, Thomas and Chen, Kan},
  journal={arXiv preprint arXiv:2506.13318},
  year={2025},
  url={https://arxiv.org/abs/2506.13318},
}
```

## Examples

Visit the [`./examples/`](https://github.com/TY-Cheng/torchvinecopulib/tree/main/examples) folder for `.ipynb` Jupyter notebooks.

## Installation

- By `pip` from [`PyPI`](https://pypi.org/project/torchvinecopulib/):

```bash
pip install torchvinecopulib torch
```

- Or `pip` from `./dist/*.whl` or `./dist/*.tar.gz` in this repo.
  Need to use proper file name.

```bash
# inside project root folder
pip install ./dist/torchvinecopulib-1.1.0-py3-none-any.whl
# or
pip install ./dist/torchvinecopulib-1.1.0.tar.gz
```

### (Recommended) [uv](https://docs.astral.sh/uv/getting-started/) for Dependency Management and Packaging

After `git clone https://github.com/TY-Cheng/torchvinecopulib.git`, `cd` into the project root where [`pyproject.toml`](https://github.com/TY-Cheng/torchvinecopulib/blob/main/pyproject.toml) exists,

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

## Documentation

- Visit [GitHub Pages website](https://ty-cheng.github.io/torchvinecopulib/)

- Or build by yourself (need [`Sphinx`](https://github.com/sphinx-doc/sphinx), theme [`furo`](https://github.com/pradyunsg/furo) and [the GNU `make`](https://www.gnu.org/software/make/))

```bash
# inside project root folder
sphinx-apidoc -o ./docs ./torchvinecopulib && cd ./docs && make html && cd ..
# if using uv
uv run sphinx-apidoc -o docs torchvinecopulib/ --separate
uv run sphinx-build docs docs/_build/html
```

## Tests

```python
# inside project root folder
python -m pytest ./tests
# coverage report
coverage run -m pytest ./tests && coverage html
# if using uv
uv run coverage run --source=torchvinecopulib -m pytest ./tests
uv run coverage report -m
```

## TODO

- `VineCop.rosenblatt`
- streaming (partial) fitting with `torch.utils.data.DataLoader`
- replace `dict` with `torch.Tensor` using some `mod`
- vectorized union-find
- flatten `_visit` logic
- flatten dynamic nested dicts into tensors
- `examples/someapplications.ipynb`
- [`fastkde.pdf`](https://github.com/LBL-EESA/fastkde/blob/main/src/fastkde/fastKDE.py) onto `torch.Tensor`

## Contributing

We welcome contributions, whether it's a bug report, feature suggestion, code contribution, or documentation improvement.

- If you encounter any issues with the project or have ideas for new features, please [open an issue](https://github.com/TY-Cheng/torchvinecopulib/issues/new) on GitHub or [privately email us](mailto:cty120120@gmail.com). Make sure to include detailed information about the problem or feature request, including steps to reproduce for bugs.

### Code Contributions

1. Fork the repository and create a new branch from the `main` branch.
2. Make your changes and ensure they adhere to the project's coding style and conventions.
3. Write tests for any new functionality and ensure existing tests pass.
4. Commit your changes with clear and descriptive commit messages.
5. Push your changes to your fork and submit a pull request to the `main` branch of the original repository.

### Pull Request Guidelines

- Keep pull requests focused on addressing a single issue or feature.
- Include a clear and descriptive title and description for your pull request.
- Make sure all tests pass before submitting the pull request.
- If your pull request addresses an open issue, reference the issue number in the description using the syntax `#issue_number`.
- [in-place ops can be slower](https://discuss.pytorch.org/t/are-inplace-operations-faster/61209/4)
- [torch.jit.script can be slower](https://discuss.pytorch.org/t/why-is-torch-jit-script-slower/120131/6)

## License

This project is released under the MIT License (© 2024- Tuoyuan Cheng, Kan Chen).  
See [LICENSE](./LICENSE) for the full text, including our own grant of rights and disclaimer.

### Third-Party Dependencies

See the “Third-Party Dependencies” section in [LICENSE](./LICENSE) for details on the `PyTorch`, `FastKDE`, and `pyvinecopulib` licenses that govern those components.
