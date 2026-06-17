# -*- coding: utf-8 -*-
"""
Hot Topics Widget
================
Identify emerging / "hot" topics — keywords whose recent activity and citation
momentum are rising — using `biblium.addons.predictive_analytics.predict_hot_topics`.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import QLabel, QComboBox, QPushButton, QSpinBox, QGridLayout, QProgressBar

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.addons.predictive_analytics import predict_hot_topics
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    predict_hot_topics = None

logger = logging.getLogger(__name__)
KW_CANDIDATES = ["Author Keywords", "Author keywords", "DE", "Index Keywords",
                 "Keywords", "oa_concepts", "oa_topics"]


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


class HotWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, df, kw, year, cit, lookback, top_n):
        super().__init__()
        self._df = df; self._kw = kw; self._year = year; self._cit = cit
        self._lb = lookback; self._top = top_n

    def run(self):
        try:
            out = predict_hot_topics(
                self._df, keywords_col=self._kw, year_col=self._year,
                citations_col=self._cit, lookback_years=self._lb,
                top_n=self._top, verbose=False)
            self.finished.emit(out, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("hot topics failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWHotTopics(OWWidget):
    """Emerging / hot topics by recent momentum."""

    name = "Hot Topics"
    description = "Emerging topics by recent activity and citation momentum"
    icon = "icons/hot_topics.svg"
    priority = 270
    keywords = ["hot topics", "emerging", "trending", "momentum", "predictive",
                "rising"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with keywords + Year")

    class Outputs:
        hot_topics = Output("Hot Topics", Table, doc="Ranked emerging topics")

    kw_col = settings.Setting("")
    lookback = settings.Setting(5)
    top_n = settings.Setting(20)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        compute_error = Msg("{}")

    class Information(OWWidget.Information):
        done = Msg("{} hot topics")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Options")
        grid = QGridLayout()
        grid.addWidget(QLabel("Keywords column:"), 0, 0)
        self.kw_combo = QComboBox()
        self.kw_combo.currentTextChanged.connect(lambda t: setattr(self, "kw_col", t))
        grid.addWidget(self.kw_combo, 0, 1)
        grid.addWidget(QLabel("Lookback (yrs):"), 1, 0)
        self.lb = QSpinBox(); self.lb.setRange(2, 20); self.lb.setValue(self.lookback)
        self.lb.valueChanged.connect(lambda v: setattr(self, "lookback", v))
        grid.addWidget(self.lb, 1, 1)
        grid.addWidget(QLabel("Top N:"), 2, 0)
        self.tn = QSpinBox(); self.tn.setRange(5, 100); self.tn.setValue(self.top_n)
        self.tn.valueChanged.connect(lambda v: setattr(self, "top_n", v))
        grid.addWidget(self.tn, 2, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Find Hot Topics")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.mainArea.layout().addWidget(self.graph)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py", "publication_year", "oa_publication_year"):
                return c
        return "Year"

    def _cit_col(self):
        for c in ("Cited by", "Times Cited", "cited_by_count", "oa_cited_by_count", "TC"):
            if self._df is not None and c in self._df.columns:
                return c
        return "Cited by"

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.kw_combo.blockSignals(True); self.kw_combo.clear()
        if self._df is not None:
            cols = [c for c in KW_CANDIDATES if c in self._df.columns]
            cols += [c for c in self._df.columns if c not in cols and "keyword" in str(c).lower()]
            self.kw_combo.addItems(cols or list(self._df.columns))
            if self.kw_col in cols:
                self.kw_combo.setCurrentText(self.kw_col)
            elif cols:
                self.kw_col = cols[0]
        self.kw_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        if not self.kw_combo.currentText():
            self.Error.compute_error("No keywords column"); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Analyzing momentum...")
        self._worker = HotWorker(self._df, self.kw_combo.currentText(),
                                 self._year_col(), self._cit_col(),
                                 self.lookback, self.top_n)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, out, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or out is None or out.empty:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "no hot topics found")
            self.Outputs.hot_topics.send(None)
            return
        self._render(out)
        self.status_label.setText(f"Done — {len(out)} topics")
        self.Information.done(len(out))
        self.Outputs.hot_topics.send(_df_to_table(out))

    def _render(self, out):
        self.graph.clear()
        label_col = out.columns[0]
        num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        if not num_cols:
            return
        val_col = num_cols[-1]
        d = out.head(15).reset_index(drop=True)
        ys = list(range(len(d)))
        bar = pg.BarGraphItem(x0=0, y=ys, height=0.6,
                              width=list(pd.to_numeric(d[val_col], errors="coerce").fillna(0)),
                              brush=pg.mkBrush("#e8743b"))
        self.graph.addItem(bar)
        self.graph.getAxis("left").setTicks([[(i, str(d.iloc[i][label_col])[:28]) for i in ys]])
        self.graph.setLabel("bottom", str(val_col))
        self.graph.setYRange(-1, len(d))

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWHotTopics).run()
