from sphinx_pyproject import SphinxConfig


# * load the pyproject.toml file using SphinxConfig
# * using Path for better cross-platform compatibility
try:
    config = SphinxConfig("../pyproject.toml", globalns=globals(), style="poetry")
except FileNotFoundError as err:
    raise FileNotFoundError("pyproject.toml not found") from err

# * project metadata
project = config.name
author = config.author
copyright = "2024-, Tuoyuan Cheng, Kan Chen"
version = release = config.version
documentation_summary = config.description
extensions = config.get("extensions", [])
html_theme = config.get("html_theme", "furo")
html_title = f"{project} v{version}"
html_theme_options = {
    "sidebar_hide_name": True,
    # "light_logo": "../torchvinecopulib.png",
    # "dark_logo": "../torchvinecopulib.png",
    # "sticky_navigation": True,
    "navigation_with_keys": True,
    # "footer_text": f"© {copyright}",
    # "navigation_depth": 4,
    # "titles_only": False,
}
autosummary_generate = True
