# -*- coding: utf-8 -*-
"""Orange3-Biblium widgets — bibliometric analysis category for Orange.

This package defines the "Biblium" category shown in the Orange canvas
toolbox. Category metadata (NAME, DESCRIPTION, ICON, BACKGROUND) is read
by Orange's widget discovery via the `orange.widgets` entry point.
"""

import os as _os

# Category title in the Orange toolbox.
NAME = "Biblium"

# Tooltip / description for the category.
DESCRIPTION = "Bibliometric and scientometric analysis (powered by Biblium)."

# Icon shown for the category (relative to this package).
ICON = "icons/category.svg"

# Toolbox section background colour (light blue).
BACKGROUND = "#E6F3FF"

# Sort order of the category relative to other Orange categories.
PRIORITY = 100

# In-app help ("?" button). Resolved by Orange's intersphinx help provider
# (orange.canvas.help entry point). Each widget page's H1 title is a Sphinx
# label equal to the widget's display name, so the widget resolves to its page.
# NOTE: the base paths MUST end with a separator, otherwise urljoin() drops the
# last path segment and the resolved help URL points to a non-existent file.
WIDGET_HELP_PATH = (
    # Development: built docs in the source tree (used for editable installs).
    ("{DEVELOP_ROOT}/docs/_build/html/", None),
    # Installed: HTML help bundled inside the package (trailing sep required).
    (_os.path.join(_os.path.dirname(__file__), "help") + _os.sep, None),
)
