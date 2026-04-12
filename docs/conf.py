import sys
from pathlib import Path

from sphinx_pyproject import SphinxConfig

sys.path.append(".")
sys.path.insert(0, str(Path(__file__).parents[1]))
# * load the pyproject.toml file using SphinxConfig
# * using Path for better cross-platform compatibility
try:
    config = SphinxConfig()
except FileNotFoundError as err:
    raise FileNotFoundError("pyproject.toml not found") from err

# * project metadata
project = config.name
author = config.author
maintainer = config.get("maintainer", author)
copyright = config.get("copyright", f"2024-, {author}")
version = release = config.version
documentation_summary = config.description
extensions = config.get("extensions", [])
if "sphinx_design" not in extensions:
    extensions.append("sphinx_design")
html_theme = config.get("html_theme", "furo")
html_title = f"{project} v{version}"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    # "light_logo": "../torchvinecopulib.png",
    # "dark_logo": "../torchvinecopulib.png",
    # "sticky_navigation": True,
    # "navigation_with_keys": True,
    # "footer_text": f"© {copyright}",
    # "navigation_depth": 4,
    # "titles_only": False,
}
autosummary_generate = True
autosummary_imported_members = False
autosectionlabel_prefix_document = True
autodoc_typehints = "description"
autodoc_default_options = {
    "undoc-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
nitpick_ignore = [
    ("py:class", "torchvinecopulib.backends.bicop.GridReflectBicopEstimator"),
    ("py:class", "torchvinecopulib.backends.marginal.GridKDE1D"),
]
