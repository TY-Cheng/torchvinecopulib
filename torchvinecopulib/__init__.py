from . import util
from .bicop import BiCop, TorchCopulaKDE2D
from .util import TorchKDE1D
from .vinecop import VineCop

__all__ = [
    "BiCop",
    "TorchCopulaKDE2D",
    "TorchKDE1D",
    "VineCop",
    "util",
]

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from pkg_resources import (  # type: ignore[assignment]
        DistributionNotFound as PackageNotFoundError,
        get_distribution as version,
    )

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0+unknown"

__title__ = "torchvinecopulib"
__author__ = "Tuoyuan Cheng"
__url__ = "https://github.com/TY-Cheng/torchvinecopulib"
__description__ = "Fitting and sampling vine copulas using PyTorch."
