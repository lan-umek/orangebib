# -*- coding: utf-8 -*-
"""
Open Science Widget
==================
Detect open-science practices per paper — open access, data availability,
code/software availability, preprints and preregistration — from abstracts /
DOIs, using `biblium.addons.open_science.analyze_open_science`. Reports
per-paper flags and the corpus-level shares.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import QLabel, QComboBox, QPushButton, QGridLayout, QProgressBar

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.addons.open_science import analyze_open_science
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    analyze_open_science = None

logger = logging.getLogger(__name__)


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas):
        try:
            col = table.get_column(var)
        except Exception:  # noqa: BLE001
            continue
        if var.is_discrete:
            vals = var.values
            data[var.name] = [vals[int(v)] if (v == v and 0 <= int(v) < len(vals)) else "" for v in col]
        else:
            data[var.name] = col
    return pd.DataFrame(data)


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    df = df.copy()
    for c in df.columns:
        if df[c].apply(lambda v: isinstance(v, (list, tuple))).any():
            df[c] = df[c].apply(lambda v: "; ".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
    attrs, metas, ac, mc = [], [], [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            attrs.append(ContinuousVariable(str(c))); ac.append(c)
        else:
            metas.append(StringVariable(str(c))); mc.append(c)
    domain = Domain(attrs, metas=metas)
    n = len(df)
    X = np.empty((n, len(attrs)))
    for i, c in enumerate(ac):
        X[:, i] = pd.to_numeric(df[c], errors="coerce").values
    M = np.empty((n, len(metas)), dtype=object)
    for i, c in enumerate(mc):
        M[:, i] = [("" if (v is None or (isinstance(v, float) and v != v)) else str(v)) for v in df[c]]
    return Table.from_numpy(domain, X, metas=M)


def _pick(df, cands, default):
    for c in cands:
        if c in df.columns:
            return c
    return default


class OSWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, str)  # summary_df, stats, error

    def __init__(self, df, doi, title, abstract, year, oa):
        super().__init__()
        self._df = df; self._doi = doi; self._title = title
        self._abstract = abstract; self._year = year; self._oa = oa

    def run(self):
        try:
            self.progress.emit("Detecting open-science practices...")
            res = analyze_open_science(
                self._df, doi_col=self._doi, title_col=self._title,
                abstract_col=self._abstract, year_col=self._year,
                oa_col=self._oa, verbose=False)
            self.finished.emit(res.summary_df, res.aggregate_stats, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("open science failed")
            self.finished.emit(None, None, f"{type(exc).__name__}: {exc}")


class OWOpenScience(OWWidget):
    """Detect open-science practices per paper."""

    name = "Open Science"
    description = "Open access, data/code availability, preprints, preregistration"
    icon = "icons/open_science.svg"
    priority = 810
    keywords = ["open science", "open access", "data availability", "code",
                "preprint", "reproducibility", "fair"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data (needs Abstract/DOI)")

    class Outputs:
        per_paper = Output("Per-paper", Table, doc="Open-science flags per paper")
        summary = Output("Summary", Table, doc="Corpus-level shares")

    doi_col = settings.Setting("")
    title_col = settings.Setting("")
    abstract_col = settings.Setting("")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("Analyzed {} papers")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Columns")
        grid = QGridLayout()
        grid.addWidget(QLabel("Abstract:"), 0, 0)
        self.abs_combo = QComboBox()
        self.abs_combo.currentTextChanged.connect(lambda t: setattr(self, "abstract_col", t))
        grid.addWidget(self.abs_combo, 0, 1)
        grid.addWidget(QLabel("Title:"), 1, 0)
        self.title_combo = QComboBox()
        self.title_combo.currentTextChanged.connect(lambda t: setattr(self, "title_col", t))
        grid.addWidget(self.title_combo, 1, 1)
        grid.addWidget(QLabel("DOI:"), 2, 0)
        self.doi_combo = QComboBox()
        self.doi_combo.currentTextChanged.connect(lambda t: setattr(self, "doi_col", t))
        grid.addWidget(self.doi_combo, 2, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Analyze")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.summary_label = QLabel("No data"); self.summary_label.setWordWrap(True)
        self.mainArea.layout().addWidget(self.summary_label)
        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.graph.setLabel("bottom", "Share / value")
        self.mainArea.layout().addWidget(self.graph)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py",
                                  "publication_year", "oa_publication_year"):
                return c
        return "Year"

    def _oa_col(self):
        for c in ("oa_is_oa", "oa_status", "Open Access", "open_access"):
            if self._df is not None and c in self._df.columns:
                return c
        return None

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        cols = list(self._df.columns) if self._df is not None else []
        self._fill(self.abs_combo, ["Processed Abstract", "Abstract", "AB"], cols, self.abstract_col)
        self._fill(self.title_combo, ["Title", "TI", "Document Title"], cols, self.title_col)
        self._fill(self.doi_combo, ["DOI", "doi", "oa_doi"], cols, self.doi_col)
        if data is None:
            self.Error.no_data()

    @staticmethod
    def _fill(combo, prefer, cols, current):
        ordered = [c for c in prefer if c in cols] + [c for c in cols if c not in prefer]
        combo.blockSignals(True); combo.clear(); combo.addItems(ordered)
        if current in ordered:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = OSWorker(
            self._df, self.doi_combo.currentText() or "DOI",
            self.title_combo.currentText() or "Title",
            self.abs_combo.currentText() or "Abstract",
            self._year_col(), self._oa_col())
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, summary_df, stats, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or summary_df is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            self.Outputs.per_paper.send(None); self.Outputs.summary.send(None)
            return
        n = len(summary_df)
        self.status_label.setText(f"Done — {n} papers")
        self.Information.done(n)
        stat_rows = []
        for k, v in (stats or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                stat_rows.append({"Metric": str(k), "Value": float(v)})
        stat_df = pd.DataFrame(stat_rows)
        self._render(stat_df)
        if not stat_df.empty:
            top = "; ".join(f"{r['Metric']}: {r['Value']:.2f}"
                            for _, r in stat_df.head(6).iterrows())
            self.summary_label.setText(f"<b>Open-science indicators</b> — {top}")
        self.Outputs.per_paper.send(_df_to_table(summary_df))
        self.Outputs.summary.send(_df_to_table(stat_df))

    def _render(self, stat_df):
        self.graph.clear()
        if stat_df is None or stat_df.empty:
            return
        d = stat_df.head(15).reset_index(drop=True)
        ys = list(range(len(d)))
        bar = pg.BarGraphItem(x0=0, y=ys, height=0.6,
                              width=list(d["Value"].astype(float)),
                              brush=pg.mkBrush("#5aa454"))
        self.graph.addItem(bar)
        self.graph.getAxis("left").setTicks([[(i, str(d.iloc[i]["Metric"])[:26]) for i in ys]])
        self.graph.setYRange(-1, len(d))

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWOpenScience).run()
