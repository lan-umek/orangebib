# -*- coding: utf-8 -*-
"""
Comparative Analysis Widget
==========================
Compare groups defined by a column (e.g. journals, countries, clusters) across
bibliometric metrics — output, citations, h-index, collaboration, etc. — using
`biblium.addons.comparative_analysis.compare_by_column`.
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
    from biblium.addons.comparative_analysis import compare_by_column
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    compare_by_column = None

logger = logging.getLogger(__name__)


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


def _result_to_df(result):
    """Best-effort extraction of a comparison DataFrame from the result."""
    mc = getattr(result, "multiple_comparison", None)
    if mc is not None:
        dfc = getattr(mc, "metrics_comparison", None)
        if isinstance(dfc, pd.DataFrame) and not dfc.empty:
            out = dfc.copy()
            if out.index.name or not isinstance(out.index, pd.RangeIndex):
                out.insert(0, "Group", [str(i) for i in out.index])
            return out.reset_index(drop=True)
    # fallback: from entity_metrics dict of dataclasses
    em = getattr(result, "entity_metrics", None)
    if isinstance(em, dict) and em:
        rows = []
        for name, m in em.items():
            row = {"Group": name}
            for attr in dir(m):
                if attr.startswith("_"):
                    continue
                v = getattr(m, attr)
                if isinstance(v, (int, float, str)) and not callable(v):
                    row[attr] = v
            rows.append(row)
        return pd.DataFrame(rows)
    return None


class CompWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, df, group_col, top_n):
        super().__init__()
        self._df = df; self._group = group_col; self._top = top_n

    def run(self):
        try:
            res = compare_by_column(self._df, group_col=self._group,
                                    top_n=self._top, verbose=False)
            out = _result_to_df(res)
            self.finished.emit(out, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("comparative analysis failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWComparative(OWWidget):
    """Compare groups across bibliometric metrics."""

    name = "Comparative Analysis"
    description = "Compare groups (by a column) across bibliometric metrics"
    icon = "icons/comparative.svg"
    priority = 670
    keywords = ["comparative", "compare", "groups", "benchmark", "metrics"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    class Outputs:
        comparison = Output("Comparison", Table, doc="Metrics by group")

    group_col = settings.Setting("")
    top_n = settings.Setting(10)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        compute_error = Msg("{}")

    class Information(OWWidget.Information):
        done = Msg("Compared {} groups")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Compare")
        grid = QGridLayout()
        grid.addWidget(QLabel("Group by column:"), 0, 0)
        self.group_combo = QComboBox()
        self.group_combo.currentTextChanged.connect(lambda t: setattr(self, "group_col", t))
        grid.addWidget(self.group_combo, 0, 1)
        grid.addWidget(QLabel("Top N groups:"), 1, 0)
        self.tn = QSpinBox(); self.tn.setRange(2, 50); self.tn.setValue(self.top_n)
        self.tn.valueChanged.connect(lambda v: setattr(self, "top_n", v))
        grid.addWidget(self.tn, 1, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Compare")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        b = gui.widgetBox(self.mainArea, "Comparison")
        self.table = QTableWidget()
        b.layout().addWidget(self.table)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.group_combo.blockSignals(True); self.group_combo.clear()
        if self._df is not None:
            # prefer low-cardinality categorical columns
            cats = []
            for c in self._df.columns:
                nun = self._df[c].nunique(dropna=True)
                if 2 <= nun <= 100:
                    cats.append(c)
            cols = cats + [c for c in self._df.columns if c not in cats]
            self.group_combo.addItems(cols)
            if self.group_col in cols:
                self.group_combo.setCurrentText(self.group_col)
            elif cols:
                self.group_col = cols[0]
        self.group_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        g = self.group_combo.currentText()
        if not g or g not in self._df.columns:
            self.Error.compute_error("Select a grouping column"); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Comparing groups...")
        self._worker = CompWorker(self._df, g, self.top_n)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, out, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or out is None or out.empty:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "no comparison produced")
            self.Outputs.comparison.send(None)
            return
        self._fill_table(out)
        self.status_label.setText(f"Done — {len(out)} groups")
        self.Information.done(len(out))
        self.Outputs.comparison.send(_df_to_table(out))

    def _fill_table(self, df):
        self.table.clear()
        self.table.setColumnCount(len(df.columns)); self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                v = df.iloc[r, c]
                txt = f"{v:,.3f}" if isinstance(v, float) else str(v)
                self.table.setItem(r, c, QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWComparative).run()
