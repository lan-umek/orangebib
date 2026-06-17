# -*- coding: utf-8 -*-
"""
Dynamic Topic Models Widget
==========================
Track how topics evolve across time periods (sequential LDA) using
`biblium.addons.dynamic_topic_models.analyze_dynamic_topics`. Reports a label
and trajectory (growing / declining / emerging / ...) per topic plus
document-level topic assignments.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QPushButton, QSpinBox, QGridLayout, QProgressBar,
    QTableWidget, QTableWidgetItem,
)

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.addons.dynamic_topic_models import analyze_dynamic_topics
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    analyze_dynamic_topics = None

logger = logging.getLogger(__name__)

TEXT_CANDIDATES = ["Processed Abstract", "Abstract", "Processed Combined Text",
                   "Processed Title", "Title", "Processed Author Keywords",
                   "Author Keywords"]


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


class DTMWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object, str)

    def __init__(self, df, text_col, year_col, n_topics, period_size):
        super().__init__()
        self._df = df; self._tc = text_col; self._yc = year_col
        self._n = n_topics; self._ps = period_size

    def run(self):
        try:
            self.progress.emit("Fitting topic models...")
            res = analyze_dynamic_topics(
                self._df, text_column=self._tc, year_column=self._yc,
                method="sequential_lda", n_topics=self._n, period_size=self._ps)
            topics = []
            for tid, evo in (res.topic_evolutions or {}).items():
                topics.append({
                    "Topic": tid,
                    "Label": getattr(evo, "label", ""),
                    "Trajectory": getattr(evo, "trajectory_type", ""),
                    "First": getattr(evo, "first_appearance", ""),
                    "Last": getattr(evo, "last_appearance", ""),
                })
            topics_df = pd.DataFrame(topics)
            self.finished.emit(topics_df, res.document_topics, res.global_metrics, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("dynamic topics failed")
            self.finished.emit(None, None, None, f"{type(exc).__name__}: {exc}")


class OWDynamicTopics(OWWidget):
    """Track topic evolution over time (sequential LDA)."""

    name = "Dynamic Topic Models"
    description = "Topic evolution over time with trajectories (sequential LDA)"
    icon = "icons/dynamic_topics.svg"
    priority = 365
    keywords = ["topic", "lda", "dynamic", "evolution", "trajectory",
                "emerging", "temporal"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with text + Year")

    class Outputs:
        topics = Output("Topics", Table, doc="Topic labels & trajectories")
        document_topics = Output("Document Topics", Table)
        metrics = Output("Period Metrics", Table)

    text_col = settings.Setting("")
    n_topics = settings.Setting(10)
    period_size = settings.Setting(5)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons + scikit-learn required.")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("{} topics over time")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Options")
        grid = QGridLayout()
        grid.addWidget(QLabel("Text column:"), 0, 0)
        self.text_combo = QComboBox()
        self.text_combo.currentTextChanged.connect(lambda t: setattr(self, "text_col", t))
        grid.addWidget(self.text_combo, 0, 1)
        grid.addWidget(QLabel("N topics:"), 1, 0)
        self.nt = QSpinBox(); self.nt.setRange(2, 50); self.nt.setValue(self.n_topics)
        self.nt.valueChanged.connect(lambda v: setattr(self, "n_topics", v))
        grid.addWidget(self.nt, 1, 1)
        grid.addWidget(QLabel("Period size (yrs):"), 2, 0)
        self.ps = QSpinBox(); self.ps.setRange(1, 20); self.ps.setValue(self.period_size)
        self.ps.valueChanged.connect(lambda v: setattr(self, "period_size", v))
        grid.addWidget(self.ps, 2, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Model Topics")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        box2 = gui.widgetBox(self.mainArea, "Topics over time")
        self.table = QTableWidget()
        box2.layout().addWidget(self.table)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py",
                                  "publication_year", "oa_publication_year"):
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
        tc = self.text_combo.currentText()
        if not tc:
            self.Error.compute_error("No text column"); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = DTMWorker(self._df, tc, self._year_col(),
                                 self.n_topics, self.period_size)
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, topics_df, doc_topics, metrics, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or topics_df is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            for o in (self.Outputs.topics, self.Outputs.document_topics, self.Outputs.metrics):
                o.send(None)
            return
        self._fill_table(topics_df)
        n = len(topics_df)
        self.status_label.setText(f"Done — {n} topics")
        self.Information.done(n)
        self.Outputs.topics.send(_df_to_table(topics_df))
        self.Outputs.document_topics.send(_df_to_table(doc_topics))
        self.Outputs.metrics.send(_df_to_table(metrics))

    def _fill_table(self, df):
        self.table.clear()
        if df is None or df.empty:
            self.table.setRowCount(0); self.table.setColumnCount(0); return
        self.table.setColumnCount(len(df.columns)); self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
        self.table.resizeColumnsToContents()

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWDynamicTopics).run()
