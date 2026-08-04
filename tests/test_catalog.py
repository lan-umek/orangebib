"""Metadata tests for the Orange3-Biblium widget catalogue.

These tests do not require Orange or Qt: they statically parse every widget
module and verify that the catalogue is complete and consistent (each widget
declares a name, an icon that exists on disk, and a priority, and that widget
names are unique).
"""
import glob
import os
import re

import pytest

WIDGET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "orangebib", "widgets",
)

WIDGET_FILES = sorted(
    f for f in glob.glob(os.path.join(WIDGET_DIR, "ow*.py"))
    if "_deprecated" not in f
)

HEADER = re.compile(
    r'class\s+OW\w+\([^)]*\).*?'
    r'name\s*=\s*["\']([^"\']+)["\'].*?'
    r'icon\s*=\s*["\']([^"\']+)["\'].*?'
    r'priority\s*=\s*(\d+)',
    re.S,
)


def parse(path):
    with open(path, encoding="utf8") as fh:
        return HEADER.search(fh.read())


def test_catalogue_size():
    assert len(WIDGET_FILES) >= 81


@pytest.mark.parametrize(
    "path", WIDGET_FILES, ids=[os.path.basename(f) for f in WIDGET_FILES]
)
def test_widget_declares_metadata(path):
    m = parse(path)
    assert m is not None, f"{os.path.basename(path)}: missing name/icon/priority"
    name, icon, priority = m.group(1), m.group(2), int(m.group(3))
    assert name.strip(), "empty widget name"
    assert priority > 0
    icon_path = os.path.join(WIDGET_DIR, icon)
    assert os.path.isfile(icon_path), f"icon not found: {icon}"


def test_widget_names_unique():
    names = [parse(f).group(1) for f in WIDGET_FILES if parse(f)]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate widget names: {dupes}"
