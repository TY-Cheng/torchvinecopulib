import importlib
import importlib.metadata
import sys

import torchvinecopulib as tvc


def test___all___imports_everything():
    for name in tvc.__all__:
        assert hasattr(tvc, name), f"{name!r} is missing from torchvinecopulib"


def test_version_matches_distribution():
    dist_version = importlib.metadata.version("torchvinecopulib")
    assert tvc.__version__ == dist_version


def test_metadata_fields():
    assert isinstance(tvc.__version__, str) and len(tvc.__version__) > 0
    assert tvc.__title__ == "torchvinecopulib"
    assert isinstance(tvc.__author__, str) and len(tvc.__author__) > 10
    assert tvc.__url__.startswith("https://github.com/TY-Cheng/torchvinecopulib")
    assert isinstance(tvc.__description__, str) and len(tvc.__description__) > 10


def test_top_level_torch_kde_exports():
    assert tvc.GridKDE1D is not None
    assert tvc.GridReflectBicopEstimator is not None


def test_version_fallback_when_distribution_missing(monkeypatch):
    original_module = sys.modules["torchvinecopulib"]
    sys.modules.pop("torchvinecopulib", None)

    def missing_version(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing_version)
    try:
        reloaded = importlib.import_module("torchvinecopulib")
        assert reloaded.__version__ == "0+unknown"
    finally:
        sys.modules["torchvinecopulib"] = original_module
