# -*- coding: utf-8 -*-
"""
Research Gaps Widget
===================
Identify under-studied combinations (SDG pairs, geographic, methodological,
temporal) using `biblium.addons.research_gaps.run_gap_analysis`. Each gap is
scored and prioritised, with recommendations.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import QLabel, QPushButton, QProgressBar, QTableWidget, QTableWidgetItem

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.biblium_main import BiblioAnalysis
    from biblium.addons.research_gaps import run_gap_analysis
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    BiblioAnalysis = None
    run_gap_analysis = None

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


class GapWorker(QThread):
    finished = pyqtSignal(object, object, str)

    def __init__(self, df, db):
        super().__init__()
        self._df = df; self._db = db

    def run(self):
        try:
            ba = BiblioAnalysis(df=self._df, db=self._db or "", res_folder=None, verbose=False)
            analysis = run_gap_analysis(ba, verbose=False)
            gaps = analysis.all_gaps or []
            rows = []
            for g in gaps:
                rows.append({
                    "Type": g.gap_type,
                    "Description": g.description,
                    "Entities": "; ".join(map(str, g.entities or []))[:120],
                    "Current": g.current_count,
                    "Expected": round(g.expected_count, 2),
                    "Gap score": round(g.gap_score, 3),
                    "Priority": round(g.priority_score, 3),
                    "Recommendation": "; ".join(g.recommendations or [])[:200],
                })
            gaps_df = pd.DataFrame(rows).sort_values("Priority", ascending=False) if rows else pd.DataFrame()
            summary = pd.DataFrame(
                [{"Gap type": k, "Count": v} for k, v in (analysis.gap_summary or {}).items()])
            self.finished.emit(gaps_df, summary, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("gap analysis failed")
            self.finished.emit(None, None, f"{type(exc).__name__}: {exc}")


class OWResearchGaps(OWWidget):
    """Find under-studied research-gap combinations."""

    name = "Research Gaps"
    description = "Identify under-studied (SDG/geographic/methodological/temporal) gaps"
    icon = "icons/research_gaps.svg"
    priority = 398
    keywords = ["research gaps", "gap", "under-studied", "sdg", "priority",
                "opportunity"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    class Outputs:
        gaps = Output("Gaps", Table, doc="Ranked research gaps")
        summary = Output("Summary", Table, doc="Gap counts by type")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("Found {} gaps")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        self.run_btn = QPushButton("Find Gaps")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        box = gui.widgetBox(self.mainArea, "Ranked gaps")
        self.table = QTableWidget()
        box.layout().addWidget(self.table)

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
        if data is None:
            self.Error.no_data()

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        db = "oa" if any(str(c).startswith("oa_") for c in self._df.columns) else ""
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Analyzing gaps...")
        self._worker = GapWorker(self._df, db)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, gaps_df, summary, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or gaps_df is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            self.Outputs.gaps.send(None); self.Outputs.summary.send(None)
            return
        self._fill_table(gaps_df)
        n = len(gaps_df)
        self.status_label.setText(f"Done — {n} gaps")
        self.Information.done(n)
        self.Outputs.gaps.send(_df_to_table(gaps_df))
        self.Outputs.summary.send(_df_to_table(summary))

    def _fill_table(self, df):
        self.table.clear()
        if df is None or df.empty:
            self.table.setRowCount(0); self.table.setColumnCount(0); return
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
        self.table.resizeColumnsToContents()

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWResearchGaps).run()
