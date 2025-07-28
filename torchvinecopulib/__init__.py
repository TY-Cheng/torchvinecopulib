from . import util
from .bicop import BiCop
from .vinecop import VineCop

__all__ = [
    "BiCop",
    "VineCop",
    "util",
]
# dynamically grab the version you just built & installed
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python <3.8 fallback
    from pkg_resources import (
        get_distribution as version,
        DistributionNotFound as PackageNotFoundError,
    )

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # this can happen if you run from a source checkout
    __version__ = "0+unknown"

__title__ = "torchvinecopulib"  # the canonical project name
__author__ = "Tuoyuan Cheng"
__url__ = "https://github.com/TY-Cheng/torchvinecopulib"  # the project homepage
__description__ = "Fitting and sampling vine copulas using PyTorch."
