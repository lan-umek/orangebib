# -*- coding: utf-8 -*-
"""
HTML Dashboard Widget
=====================
Generate a single self-contained, interactive HTML dashboard (overview,
production, sources, authors, keywords, word cloud, networks, citations, …)
from bibliographic data using biblium's Dashboard engine.
"""

import os
import logging
import webbrowser
from typing import Optional

import pandas as pd

from AnyQt.QtCore import QThread, pyqtSignal
from AnyQt.QtWidgets import (QLabel, QLineEdit, QComboBox, QPushButton,
                             QGridLayout, QFileDialog)

from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from biblium.biblium_main import BiblioAnalysis
    from biblium.dashboard import Dashboard, DashboardConfig
    HAS_DASH = True
except Exception:  # noqa: BLE001
    HAS_DASH = False
    BiblioAnalysis = Dashboard = DashboardConfig = None

THEMES = ["light", "dark"]


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in (list(table.domain.attributes) + list(table.domain.class_vars)
                + list(table.domain.metas)):
        try:
            col = table.get_column(var)
        except Exception:  # noqa: BLE001
            continue
        if getattr(var, "is_discrete", False):
            vals = var.values
            data[var.name] = [vals[int(v)] if (v == v and 0 <= int(v) < len(vals))
                              else "" for v in col]
        else:
            data[var.name] = col
    return pd.DataFrame(data)


def _detect_list_separator(df):
    """Pick the list separator actually used (OpenAlex '|' vs Scopus '; ')."""
    try:
        cols = [c for c in ("Author Keywords", "Index Keywords", "Keywords",
                            "Authors", "Affiliations") if c in df.columns]
        sample = ""
        for c in cols:
            vals = df[c].dropna().astype(str)
            if len(vals):
                sample += " ".join(vals.head(100).tolist()) + " "
        for cand in ["||", "|", "; ", ";"]:
            if cand in sample:
                return cand
    except Exception:  # noqa: BLE001
        pass
    return None


class _Worker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, df, db, title, subtitle, theme, top_n, path):
        super().__init__()
        self.df, self.db = df, db
        self.title, self.subtitle = title, subtitle
        self.theme, self.top_n, self.path = theme, top_n, path

    def run(self):
        try:
            try:
                import matplotlib
                matplotlib.use("Agg", force=True)
            except Exception:  # noqa: BLE001
                pass
            ba = BiblioAnalysis(df=self.df, db=self.db or "", verbose=False)
            sep = _detect_list_separator(self.df)
            if sep:
                try:
                    ba.default_separator = sep
                except Exception:  # noqa: BLE001
                    pass
            cfg = DashboardConfig(title=self.title or "Bibliometric Analysis Dashboard",
                                  theme=self.theme, top_n=self.top_n)
            if self.subtitle:
                try:
                    cfg.subtitle = self.subtitle
                except Exception:  # noqa: BLE001
                    pass
            Dashboard(ba, cfg).create(self.path, title=self.title or None,
                                      theme=self.theme)
            self.finished_ok.emit(self.path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("dashboard build failed")
            self.failed.emit(str(exc))


class OWDashboard(OWWidget):
    """Build an interactive single-file HTML dashboard."""

    name = "HTML Dashboard"
    description = "Generate a self-contained interactive HTML dashboard"
    icon = "icons/dashboard.svg"
    priority = 910
    keywords = ["dashboard", "html", "report", "interactive", "overview"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table)

    title = settings.Setting("")
    subtitle = settings.Setting("")
    theme = settings.Setting(0)
    top_n = settings.Setting(15)
    db_code = settings.Setting("")
    last_path = settings.Setting("")

    want_main_area = False
    resizing_enabled = False

    class Error(OWWidget.Error):
        no_dash = Msg("biblium dashboard module not available")
        no_data = Msg("No input data")
        failed = Msg("Dashboard failed: {}")

    class Information(OWWidget.Information):
        done = Msg("Dashboard saved: {}")

    def __init__(self):
        super().__init__()
        self._df = None
        self._worker = None
        self._path = None
        if not HAS_DASH:
            self.Error.no_dash()

        box = gui.widgetBox(self.controlArea, "Dashboard")
        g = QGridLayout()
        g.addWidget(QLabel("Title:"), 0, 0)
        self.title_edit = QLineEdit(self.title)
        self.title_edit.textChanged.connect(lambda t: setattr(self, "title", t))
        g.addWidget(self.title_edit, 0, 1)
        g.addWidget(QLabel("Subtitle:"), 1, 0)
        self.sub_edit = QLineEdit(self.subtitle)
        self.sub_edit.textChanged.connect(lambda t: setattr(self, "subtitle", t))
        g.addWidget(self.sub_edit, 1, 1)
        box.layout().addLayout(g)
        gui.comboBox(box, self, "theme", items=THEMES, label="Theme:",
                     orientation="horizontal", sendSelectedValue=False)
        gui.spin(box, self, "top_n", 5, 50, label="Top N per section:")

        self.build_btn = QPushButton("Create dashboard…")
        self.build_btn.clicked.connect(self._create)
        self.controlArea.layout().addWidget(self.build_btn)
        self.open_btn = QPushButton("Open in browser")
        self.open_btn.clicked.connect(self._open)
        self.open_btn.setEnabled(False)
        self.controlArea.layout().addWidget(self.open_btn)

        self.status = QLabel("Connect data, then create."); self.status.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status)
        self.controlArea.layout().addStretch(1)

    @Inputs.data
    def set_data(self, data):
        self._df = _table_to_df(data) if data is not None else None
        self.Error.clear()
        if data is None:
            self.status.setText("Connect data, then create.")
        else:
            self.status.setText(f"{len(self._df)} documents ready. Click 'Create'.")

    def _create(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_DASH:
            self.Error.no_dash(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        start = self.last_path or os.path.join(os.path.expanduser("~"), "dashboard.html")
        path, _ = QFileDialog.getSaveFileName(self, "Save dashboard", start,
                                              "HTML files (*.html)")
        if not path:
            return
        if not path.lower().endswith(".html"):
            path += ".html"
        self.last_path = path
        self.build_btn.setEnabled(False)
        self.status.setText("Building dashboard… (this can take a moment)")
        self._worker = _Worker(self._df, self.db_code, self.title, self.subtitle,
                               THEMES[self.theme], self.top_n, path)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_ok(self, path):
        self._path = path
        self.build_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.Information.done(path)
        self.status.setText(f"Saved: {path}")
        try:
            webbrowser.open(f"file://{path}")
        except Exception:  # noqa: BLE001
            pass

    def _on_fail(self, msg):
        self.build_btn.setEnabled(True)
        self.Error.failed(msg)
        self.status.setText("Failed.")

    def _open(self):
        if self._path and os.path.exists(self._path):
            webbrowser.open(f"file://{self._path}")


if __name__ == "__main__":
    WidgetPreview(OWDashboard).run()
