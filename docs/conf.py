# -*- coding: utf-8 -*-
"""Sphinx configuration for Orange3-Biblium documentation."""
project = "Orange3-Biblium"
author = "Lan Umek"
copyright = "2026, Lan Umek"
release = "0.2.1"

extensions = ["myst_parser", "sphinx.ext.autosectionlabel"]
myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Make each page's top-level (H1) heading a cross-reference label so Orange's
# in-app "?" help can resolve a widget by its display name via objects.inv.
autosectionlabel_prefix_document = False
autosectionlabel_maxdepth = 1

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Orange3-Biblium"
html_css_files = ["custom.css"]
