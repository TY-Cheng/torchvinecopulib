.. Put a project logo here if you have one
.. .. image:: _static/logo.png
..    :alt: torchvinecopulib Logo

============================================
Welcome to torchvinecopulib's documentation!
============================================

.. Add badges here
.. raw:: html

   <p align="center">
     <a href="https://app.codacy.com/gh/TY-Cheng/torchvinecopulib/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade" target="_blank" rel="noopener noreferrer"><img src="https://app.codacy.com/project/badge/Grade/e8a7a7448b2043d9bbefafc5a3ec14f7" alt="Codacy Grade"/></a>
     <a href="https://app.codacy.com/gh/TY-Cheng/torchvinecopulib/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage" target="_blank" rel="noopener noreferrer"><img src="https://app.codacy.com/project/badge/Coverage/e8a7a7448b2043d9bbefafc5a3ec14f7" alt="Codacy Coverage"/></a>
     <a href="https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml" target="_blank" rel="noopener noreferrer"><img src="https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml/badge.svg?branch=main" alt="Lint Pytest"/></a>
     <a href="https://ty-cheng.github.io/torchvinecopulib/" target="_blank" rel="noopener noreferrer"><img src="https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/static.yml/badge.svg?branch=main" alt="Deploy Docs"/></a>
     <br/>
     <img src="https://img.shields.io/pypi/pyversions/torchvinecopulib" alt="PyPI - Python Version"/>
     <a href="https://github.com/TY-Cheng/torchvinecopulib/actions/workflows/python-package.yml" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/OS-Windows%7CmacOS%7CUbuntu-blue" alt="OS Compatibility"/></a>
     <br/>
     <a href="https://github.com/TY-Cheng/torchvinecopulib/blob/main/LICENSE" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="GitHub License"/></a>
     <a href="https://pypi.org/project/torchvinecopulib/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/pypi/v/torchvinecopulib" alt="PyPI - Version"/></a>
     <a href="https://zenodo.org/doi/10.5281/zenodo.10836953" target="_blank" rel="noopener noreferrer"><img src="https://zenodo.org/badge/768037665.svg" alt="DOI"/></a>
   </p>

``torchvinecopulib`` is a ``Python`` library for fitting and sampling vine copulas using ``PyTorch``. 
It is designed for researchers and practitioners in statistics, machine learning, and finance who need flexible, GPU-accelerated, and vectorized copula modeling and sampling.

**GitHub Repository:** https://github.com/TY-Cheng/torchvinecopulib

.. note::

   Version ``1.3.0`` standardizes the backend API around ``marginal_backend`` and
   ``bicop_backend``. The default production path is the torch-native grid implementation
   (``grid``, ``grid_reflect``, ``grid_probit``), while CPU-only reference backends
   (``lp_ref`` and ``tll_ref``) remain optional via ``uv sync --extra reference`` or
   ``pip install torchvinecopulib[reference]``.

.. note::

   ``fit()`` is a builder path and does not preserve a training graph. The differentiable
   query path is exposed through ``log_pdf()``, ``cdf()``, ``hfunc()``, and the underlying
   interpolation routines.

.. toctree::
   :maxdepth: 3
   :caption: Core Modules
   :titlesonly:

   modules.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
