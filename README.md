# TorchVineCopuLib

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e8a7a7448b2043d9bbefafc5a3ec14f7)](https://app.codacy.com/gh/TY-Cheng/torchvinecopulib/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/e8a7a7448b2043d9bbefafc5a3ec14f7)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![Lint Pytest](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml)
[![Deploy Docs](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/static.yml/badge.svg?branch=main)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/static.yml)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchvinecopulib)
[![OS](https://img.shields.io/badge/OS-Windows%7CmacOS%7CUbuntu-blue)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml)

![GitHub License](https://img.shields.io/github/license/ty-cheng/torchvinecopulib)
![PyPI - Version](https://img.shields.io/pypi/v/torchvinecopulib)
[![DOI](https://zenodo.org/badge/768037665.svg)](https://zenodo.org/doi/10.5281/zenodo.10836953)

Yet another vine copula package, using [PyTorch](https://pytorch.org/get-started/locally/).

- C/D/R-Vine full-sampling/ quantile-regression/ conditional-sampling, all in one package
  - Flexible sampling order for experienced users
- Vectorized tensor computation with GPU (`device='cuda'`) support
- Shorter runtimes for higher dimension simulations
- Pure `Python` library, inspired by [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib/) on Windows, Linux, MacOS
- IO and visualization support

## Dependencies

```toml
# inside the `./pyproject.toml` file;
fastkde = "*"
numpy = "*"
python = ">=3.12"
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

## Installation

- By `pip` from [`PyPI`](https://pypi.org/project/torchvinecopulib/):

```bash
pip install torchvinecopulib torch
```

- Or `pip` from `./dist/*.whl` or `./dist/*.tar.gz` in this repo.
  Need to use proper file name.

```bash
# inside project root folder
pip install ./dist/torchvinecopulib-1.0.0-py3-none-any.whl
# or
pip install ./dist/torchvinecopulib-1.0.0.tar.gz
```

### (Optional) [uv](https://docs.astral.sh/uv/getting-started/) for Dependency Management and Packaging

After `git clone https://github.com/TY-Cheng/torchvinecopulib.git`, `cd` into the project root where [`pyproject.toml`](https://github.com/TY-Cheng/torchvinecopulib/blob/main/pyproject.toml) exists,

```bash
# inside project root folder
uv sync --extra cpu -U
# or
uv sync --extra cu126 -U
```

## Examples

Visit the [`./examples/`](https://github.com/TY-Cheng/torchvinecopulib/tree/main/examples) folder for `.ipynb` Jupyter notebooks.

## Documentation

- Visit [GitHub Pages website](https://ty-cheng.github.io/torchvinecopulib/)

- Or build by yourself (need [`Sphinx`](https://github.com/sphinx-doc/sphinx), theme [`furo`](https://github.com/pradyunsg/furo) and [the GNU `make`](https://www.gnu.org/software/make/))

```bash
# inside project root folder
sphinx-apidoc -o ./docs ./torchvinecopulib && cd ./docs && make html && cd ..
```

## Tests

```python
# inside project root folder
python -m pytest ./tests
# coverage report
coverage run -m pytest ./tests && coverage html
```

## TODO

- [`fastkde.pdf`](https://github.com/LBL-EESA/fastkde/blob/main/src/fastkde/fastKDE.py) onto `torch.Tensor`
- `VineCop.rosenblatt`
- `examples/someapplications.ipynb`
- flatten dynamic nested dicts into tensors

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

> MIT License
>
> Copyright (c) 2024- Tuoyuan Cheng, Kan Chen
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
