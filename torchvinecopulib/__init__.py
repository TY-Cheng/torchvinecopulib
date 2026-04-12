from . import util
from .backends import GridKDE1D
from .bicop import BiCop, BiCopDiagnostics, GridReflectBicopEstimator
from .vinecop import (
    VineBuildArtifact,
    VineBuilder,
    VineCop,
    VineCopEngine,
    VineDiagnostics,
)

__all__ = [
    "BiCop",
    "BiCopDiagnostics",
    "GridKDE1D",
    "GridReflectBicopEstimator",
    "VineBuildArtifact",
    "VineBuilder",
    "VineCop",
    "VineCopEngine",
    "VineDiagnostics",
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
