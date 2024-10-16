import sys

from sphinx_pyproject import SphinxConfig

sys.path.append('.')

config = SphinxConfig('../pyproject.toml', globalns=globals(), style = 'poetry' )

project = config.name
author = config.author
version = release = config.version
documentation_summary = config.description
extensions = config["extensions"]
html_theme = 'furo'