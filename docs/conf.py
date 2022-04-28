import os
import sys

sys.path.insert(0, os.path.abspath("."))

project = "route-distances"
copyright = "2021, Molecular AI group"
author = "Molecular AI group"
release = "1.0.1"

extensions = [
    "sphinx.ext.autodoc",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "alabaster"
html_theme_options = {
    "description": "Routines for calculating route-distances",
    "fixed_sidebar": True,
}
