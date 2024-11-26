import importlib.metadata

from . import bicop, util, vinecop

__all__ = [
    "bicop",
    "util",
    "vinecop",
]

__version__ = importlib.metadata.version("torchvinecopulib")
