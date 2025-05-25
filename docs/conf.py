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
version = release = config.version
documentation_summary = config.description
extensions = config.get("extensions", [])
html_theme = config.get("html_theme", "furo")
html_title = f"{project} v{version}"
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
