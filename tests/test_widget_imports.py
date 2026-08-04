"""Smoke tests: import every Orange3-Biblium widget module.

Requires Orange3 (and Qt); skipped automatically when Orange is not
installed. Importing each module exercises the full widget definition
(class construction, settings declarations, input/output signals) against
the currently installed Biblium release, so incompatibilities introduced
by upstream changes surface immediately.
"""
import glob
import importlib
import os

import pytest

pytest.importorskip("Orange")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

WIDGET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "orangebib", "widgets",
)

MODULES = sorted(
    os.path.splitext(os.path.basename(f))[0]
    for f in glob.glob(os.path.join(WIDGET_DIR, "ow*.py"))
    if "_deprecated" not in f
)


@pytest.mark.parametrize("module", MODULES)
def test_widget_module_imports(module):
    importlib.import_module(f"orangebib.widgets.{module}")
