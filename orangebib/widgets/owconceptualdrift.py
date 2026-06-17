# -*- coding: utf-8 -*-
"""
Conceptual Drift Widget
======================
Track how the *meaning/context* of chosen terms shifts over time — the
co-occurrence context of each term is compared across time windows
(Jensen–Shannon divergence) via
`biblium.addons.conceptual_drift.compute_conceptual_drift`.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QLineEdit, QPushButton, QSpinBox, QGridLayout, QProgressBar,
)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.addons.conceptual_drift import compute_conceptual_drift
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    compute_conceptual_drift = None

logger = logging.getLogger(__name__)
TEXT_CANDIDATES = ["Processed Abstract", "Abstract", "Processed Combined Text", "Title"]
PALETTE = ["#4a90d9", "#e8743b", "#5aa454", "#c0504d", "#8064a2", "#1aa8a8"]


def _table_to_df(table):
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


def _df_to_table(df):
    if df is None or df.empty:
        return None
    attrs, metas, ac, mc = [], [], [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
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


class DriftWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, df, terms, text_col, year_col, window):
        super().__init__()
        self._df = df; self._terms = terms; self._tc = text_col
        self._yc = year_col; self._w = window

    def run(self):
        try:
            analysis = compute_conceptual_drift(
                self._df, target_terms=self._terms, text_col=self._tc,
                year_col=self._yc, window_size=self._w)
            self.finished.emit(analysis.get_summary_df(), "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("conceptual drift failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWConceptualDrift(OWWidget):
    """Track semantic/context drift of terms over time."""

    name = "Conceptual Drift"
    description = "How the meaning/context of terms shifts over time"
    icon = "icons/conceptual_drift.svg"
    priority = 380
    keywords = ["conceptual drift", "semantic", "meaning", "context", "drift",
                "temporal"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with text + Year")

    class Outputs:
        drift = Output("Drift", Table, doc="Drift scores per term over windows")

    terms = settings.Setting("")
    text_col = settings.Setting("")
    window_size = settings.Setting(5)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        no_terms = Msg("Enter one or more target terms")
        compute_error = Msg("{}")

    class Information(OWWidget.Information):
        done = Msg("Drift for {} terms")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Options")
        grid = QGridLayout()
        grid.addWidget(QLabel("Terms (comma-sep):"), 0, 0)
        self.terms_edit = QLineEdit(self.terms)
        self.terms_edit.setPlaceholderText("e.g. governance, big data")
        self.terms_edit.textChanged.connect(lambda t: setattr(self, "terms", t))
        grid.addWidget(self.terms_edit, 0, 1)
        grid.addWidget(QLabel("Text column:"), 1, 0)
        self.text_combo = QComboBox()
        self.text_combo.currentTextChanged.connect(lambda t: setattr(self, "text_col", t))
        grid.addWidget(self.text_combo, 1, 1)
        grid.addWidget(QLabel("Window (yrs):"), 2, 0)
        self.win = QSpinBox(); self.win.setRange(2, 20); self.win.setValue(self.window_size)
        self.win.valueChanged.connect(lambda v: setattr(self, "window_size", v))
        grid.addWidget(self.win, 2, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Compute Drift")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        self.graph.addLegend()
        self.graph.setLabel("left", "Drift vs previous")
        self.graph.setLabel("bottom", "Window")
        self.mainArea.layout().addWidget(self.graph)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py", "publication_year", "oa_publication_year"):
                return c
        return "Year"

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.text_combo.blockSignals(True); self.text_combo.clear()
        if self._df is not None:
            cols = [c for c in TEXT_CANDIDATES if c in self._df.columns]
            cols += [c for c in self._df.columns if c not in cols and self._df[c].dtype == object]
            self.text_combo.addItems(cols)
            if self.text_col in cols:
                self.text_combo.setCurrentText(self.text_col)
            elif cols:
                self.text_col = cols[0]
        self.text_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        terms = [t.strip() for t in self.terms.split(",") if t.strip()]
        if not terms:
            self.Error.no_terms(); return
        if not self.text_combo.currentText():
            self.Error.compute_error("No text column"); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Computing drift...")
        self._worker = DriftWorker(self._df, terms, self.text_combo.currentText(),
                                   self._year_col(), self.window_size)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, summary, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or summary is None or summary.empty:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "no drift computed")
            self.Outputs.drift.send(None)
            return
        self._render(summary)
        nterms = summary["Term"].nunique() if "Term" in summary.columns else 1
        self.status_label.setText(f"Done — {nterms} terms")
        self.Information.done(nterms)
        self.Outputs.drift.send(_df_to_table(summary))

    def _render(self, summary):
        self.graph.clear()
        if "Term" not in summary.columns:
            return
        drift_col = next((c for c in summary.columns if "drift" in c.lower()), None)
        if drift_col is None:
            return
        for ti, term in enumerate(summary["Term"].unique()):
            sub = summary[summary["Term"] == term].reset_index(drop=True)
            y = pd.to_numeric(sub[drift_col], errors="coerce").values
            x = list(range(len(y)))
            color = PALETTE[ti % len(PALETTE)]
            self.graph.plot(x, np.nan_to_num(y, nan=0.0), pen=pg.mkPen(color, width=2),
                            symbol="o", symbolBrush=color, symbolSize=7, name=str(term)[:20])

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWConceptualDrift).run()
