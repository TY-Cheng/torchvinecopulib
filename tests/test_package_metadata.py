import importlib.metadata
import torchvinecopulib as tvc


def test___all___imports_everything():
    # make sure every name in __all__ actually exists on the package
    for name in tvc.__all__:
        assert hasattr(tvc, name), f"{name!r} is missing from torchvinecopulib"


def test_version_matches_distribution():
    # this will throw if the package isn’t actually installed under that name
    dist_version = importlib.metadata.version("torchvinecopulib")
    assert tvc.__version__ == dist_version


def test_metadata_fields():
    # simple sanity‐checks of your metadata dunders
    assert isinstance(tvc.__version__, str) and len(tvc.__version__) > 0
    assert tvc.__title__ == "torchvinecopulib"
    assert isinstance(tvc.__author__, str) and len(tvc.__author__) > 10
    assert tvc.__url__.startswith("https://github.com/TY-Cheng/torchvinecopulib")
    assert isinstance(tvc.__description__, str) and len(tvc.__description__) > 10
