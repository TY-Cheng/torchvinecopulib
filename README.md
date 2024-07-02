# TorchVineCopuLib

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e8a7a7448b2043d9bbefafc5a3ec14f7)](https://app.codacy.com/gh/TY-Cheng/torchvinecopulib/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/e8a7a7448b2043d9bbefafc5a3ec14f7)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![Python package](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml)
[![Deploy Docs](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/static.yml/badge.svg?branch=main)](https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/static.yml)

![Static Badge](https://img.shields.io/badge/Python-%203.10%7C3.11%7C3.12-blue)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![PyPI - Version](https://img.shields.io/pypi/v/torchvinecopulib?style=flat)
[![DOI](https://zenodo.org/badge/768037665.svg)](https://zenodo.org/doi/10.5281/zenodo.10836953)


Yet another vine copula package, using [PyTorch](https://pytorch.org/get-started/locally/).

- C/D/R-Vine full simulation/ quantile-regression/ conditional-simulation, all in one package
  - Flexible simulation workflow for experienced users
- Vectorized tensor computation with GPU (`device='cuda'`) support
- Shorter runtimes for higher dimension simulations
- Decoupled dataclasses and factory methods
- Pure `Python` library, inspired by and tested against [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib/) on Windows, Linux, MacOS
- IO and visualization support

## Dependencies

```toml
# inside the `./pyproject.toml` file
python = "^3.10"
scipy = "*"
torch = "^2"
```

For [PyTorch](https://pytorch.org/get-started/locally/) with `cuda` support on Windows:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

> [!TIP]
> macOS users should set `device='cpu'` at this stage, for using `device='mps'` won't support `dtype=torch.float64`.

## Installation

By `pip` from `PyPI`:

```bash
pip install torchvinecopulib
```

with full drawing and bivariate dependency metric support:

```bash
pip install torchvinecopulib matplotlib pot scikit-learn
```

By `pip` from `./dist/*.whl` or `./dist/*.tar.gz` in this repo.
Need to use proper file name.

```bash
# inside project root folder
pip install ./dist/torchvinecopulib-2024.7.0-py3-none-any.whl
# or
pip install ./dist/torchvinecopulib-2024.7.0.tar.gz
```

## Examples

Visit the `./examples/` folder for `.ipynb` Jupyter notebooks.

## Documentation

- Visit [GitHub Pages website](https://ty-cheng.github.io/torchvinecopulib/)

- Or visit the `./docs/_build/html` subfolder for static `html` made with `Sphinx`

- Or build by yourself (need [`Sphinx`](https://github.com/sphinx-doc/sphinx), theme [`furo`](https://github.com/pradyunsg/furo) and [the GNU `make`](https://www.gnu.org/software/make/))

```bash
# inside project root folder
sphinx-apidoc -o ./docs ./torchvinecopulib && cd ./docs && make html && cd ..
```

## Tests

> [!TIP]
> the `./tests/test_vinecop.py` may take longer without `'cuda'`

```python
# inside project root folder
python -m pytest ./tests
# coverage report
coverage run -m pytest ./tests && coverage html
```

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

### TODO
1. more (non-parametric) `bicop` class in `torch`
2. potentially deprecating `'mle'` from `mtd_fit`

## License

> Copyright (C) 2024 Tuoyuan Cheng, Xiaosheng You, Kan Chen
>
> This file is part of torchvinecopulib.
> torchvinecopulib is free software: you can redistribute it and/or modify
> it under the terms of the GNU General Public License as published by
> the Free Software Foundation, either version 3 of the License, or
> (at your option) any later version.
>
> torchvinecopulib is distributed in the hope that it will be useful,
> but WITHOUT ANY WARRANTY; without even the implied warranty of
> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
> GNU General Public License for more details.
>
> You should have received a copy of the GNU General Public License
> along with torchvinecopulib. If not, see <http://www.gnu.org/licenses/>.
