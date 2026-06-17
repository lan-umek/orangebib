# -*- coding: utf-8 -*-
"""
Integrity Audit Widget
=====================
Screen a corpus for research-integrity red flags — tortured phrases, retracted
papers (OpenAlex), abnormal author velocity, suspicious co-author cliques,
missing institutions and excessive self-citation — using
`biblium.addons.integrity_audit.integrity_audit_report`. Produces a
per-paper flags table. Flags are screening signals, not proof of misconduct.
"""

import os
import tempfile
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
    from biblium.addons.integrity_audit import integrity_audit_report
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    integrity_audit_report = None

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
    if df is None or len(df) == 0:
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


class AuditWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, str)

    def __init__(self, df, author_col, year_col):
        super().__init__()
        self._df = df; self._author = author_col; self._year = year_col

    def run(self):
        try:
            self.progress.emit("Running integrity checks...")
            out_folder = os.path.join(tempfile.gettempdir(), "biblium_integrity")
            os.makedirs(out_folder, exist_ok=True)
            results = integrity_audit_report(
                self._df, out_folder=out_folder,
                author_col=self._author, year_col=self._year)
            papers = results.get("papers_with_flags")
            counts = []
            for key in ("tortured", "retracted", "velocity", "coauthor",
                        "missing_inst", "self_cit"):
                v = results.get(key)
                n = len(v) if isinstance(v, pd.DataFrame) else (
                    int(v) if isinstance(v, (int, float)) else 0)
                counts.append({"Check": key, "Flagged": n})
            summary = pd.DataFrame(counts)
            self.finished.emit(papers, summary, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("integrity audit failed")
            self.finished.emit(None, None, f"{type(exc).__name__}: {exc}")


class OWIntegrityAudit(OWWidget):
    """Screen a corpus for research-integrity red flags."""

    name = "Integrity Audit"
    description = "Screen for integrity red flags (tortured phrases, retractions, anomalies)"
    icon = "icons/integrity_audit.svg"
    priority = 820
    keywords = ["integrity", "audit", "retraction", "tortured phrases",
                "self-citation", "anomaly", "misconduct"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    class Outputs:
        flagged = Output("Papers with flags", Table, doc="Per-paper integrity flags")
        summary = Output("Summary", Table, doc="Flag counts per check")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("{} papers flagged")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        info = QLabel("<small>Flags are <b>screening signals</b> for review, "
                      "not proof of misconduct.</small>")
        info.setWordWrap(True)
        self.controlArea.layout().addWidget(info)
        self.run_btn = QPushButton("Run Audit")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        box = gui.widgetBox(self.mainArea, "Flag summary")
        self.table = QTableWidget()
        box.layout().addWidget(self.table)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _author_col(self):
        for c in ("Author full names", "Authors", "Author", "AU"):
            if self._df is not None and c in self._df.columns:
                return c
        return "Author full names"

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
        if data is None:
            self.Error.no_data()

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Auditing...")
        self._worker = AuditWorker(self._df, self._author_col(), self._year_col())
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, papers, summary, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or summary is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            self.Outputs.flagged.send(None); self.Outputs.summary.send(None)
            return
        self._fill_table(summary)
        n = len(papers) if isinstance(papers, pd.DataFrame) else 0
        self.status_label.setText(f"Done — {n} papers with flags")
        self.Information.done(n)
        self.Outputs.flagged.send(_df_to_table(papers))
        self.Outputs.summary.send(_df_to_table(summary))

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
    WidgetPreview(OWIntegrityAudit).run()
