# -*- coding: utf-8 -*-
"""
Open Access Finder Widget
========================
Find free, legal full-text PDFs for papers by DOI, via the OpenAlex
best open-access location. Outputs each paper's OA status and PDF/landing URL;
double-click a row to open it. (No SciHub or other unauthorised sources.)
"""

import os
import time
import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QLineEdit, QPushButton, QGridLayout, QProgressBar,
    QTableWidget, QTableWidgetItem,
)

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    import requests
    HAS_REQUESTS = True
except Exception:  # noqa: BLE001
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)
DOI_CANDIDATES = ["DOI", "doi", "DI", "oa_doi"]
TITLE_CANDIDATES = ["Title", "TI", "Document Title"]


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
    metas = [StringVariable(str(c)) for c in df.columns]
    domain = Domain([], metas=metas)
    return Table.from_numpy(domain, np.empty((len(df), 0)), metas=df.astype(str).values)


def _norm_doi(doi):
    if not isinstance(doi, str):
        return None
    d = doi.strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p):]
    d = d.strip()
    return d if d.startswith("10.") and "/" in d else None


class OAWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, doi_col, title_col, email):
        super().__init__()
        self._df = df; self._doi = doi_col; self._title = title_col
        self._email = email; self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            session = requests.Session()
            session.headers.update({"User-Agent": "Biblium-Orange-OAFinder"})
            rows = []
            dois = []
            for _, r in self._df.iterrows():
                nd = _norm_doi(str(r.get(self._doi, "")))
                if nd:
                    dois.append((nd, str(r.get(self._title, "")) if self._title else ""))
            dois = list(dict.fromkeys(dois))
            n = len(dois)
            batch = 50
            done = 0
            for i in range(0, n, batch):
                if self._stop:
                    self.finished.emit(pd.DataFrame(rows), "Cancelled"); return
                chunk = dois[i:i + batch]
                filt = "doi:" + "|".join(d for d, _ in chunk)
                params = {"filter": filt, "per-page": 200,
                          "select": "doi,title,open_access,best_oa_location"}
                if self._email:
                    params["mailto"] = self._email
                try:
                    resp = session.get("https://api.openalex.org/works",
                                       params=params, timeout=30)
                    works = resp.json().get("results", []) if resp.status_code == 200 else []
                except Exception:  # noqa: BLE001
                    works = []
                by_doi = {}
                for w in works:
                    d = _norm_doi(w.get("doi") or "")
                    if d:
                        by_doi[d] = w
                for d, title in chunk:
                    w = by_doi.get(d)
                    oa = (w or {}).get("open_access") or {}
                    loc = (w or {}).get("best_oa_location") or {}
                    rows.append({
                        "Title": (title or (w or {}).get("title") or "")[:120],
                        "DOI": d,
                        "Is OA": "yes" if oa.get("is_oa") else "no",
                        "OA status": oa.get("oa_status") or "",
                        "PDF URL": loc.get("pdf_url") or "",
                        "Landing URL": loc.get("landing_page_url") or "",
                    })
                done += len(chunk)
                self.progress.emit(int(done * 100 / max(n, 1)), f"{done}/{n} DOIs")
                time.sleep(0.1)
            self.finished.emit(pd.DataFrame(rows), "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("OA finder failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWOAFinder(OWWidget):
    """Find free full-text PDFs by DOI via OpenAlex."""

    name = "Open Access Finder"
    description = "Find free full-text (PDF) for papers by DOI via OpenAlex"
    icon = "icons/oa_finder.svg"
    priority = 40
    keywords = ["open access", "pdf", "full text", "doi", "unpaywall",
                "free", "download"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Data with a DOI column")

    class Outputs:
        results = Output("OA Links", Table, doc="OA status + PDF/landing URLs")

    doi_col = settings.Setting("")
    title_col = settings.Setting("")
    email = settings.Setting("")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_requests = Msg("The 'requests' package is required.")
        no_doi = Msg("Select the DOI column")
        finder_error = Msg("{}")

    class Information(OWWidget.Information):
        done = Msg("{} of {} papers have a free PDF")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None
        self._result = None

        box = gui.widgetBox(self.controlArea, "Source")
        grid = QGridLayout()
        grid.addWidget(QLabel("DOI column:"), 0, 0)
        self.doi_combo = QComboBox()
        self.doi_combo.currentTextChanged.connect(lambda t: setattr(self, "doi_col", t))
        grid.addWidget(self.doi_combo, 0, 1)
        grid.addWidget(QLabel("Email (polite pool):"), 1, 0)
        self.email_edit = QLineEdit(self.email)
        self.email_edit.setPlaceholderText("you@university.edu")
        self.email_edit.textChanged.connect(lambda t: setattr(self, "email", t))
        grid.addWidget(self.email_edit, 1, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Find PDFs")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.controlArea.layout().addWidget(self.cancel_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        b = gui.widgetBox(self.mainArea, "Open-access links (double-click to open)")
        self.table = QTableWidget()
        self.table.cellDoubleClicked.connect(self._open_row)
        b.layout().addWidget(self.table)

        if not HAS_REQUESTS:
            self.Error.no_requests()
            self.run_btn.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_REQUESTS:
            self.Error.no_requests()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        cols = list(self._df.columns) if self._df is not None else []
        self._fill(self.doi_combo, DOI_CANDIDATES, cols, self.doi_col)
        if data is None:
            self.Error.no_data()

    @staticmethod
    def _fill(combo, prefer, cols, current):
        ordered = [c for c in prefer if c in cols] + [c for c in cols if c not in prefer]
        combo.blockSignals(True); combo.clear(); combo.addItems(ordered)
        if current in ordered:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _title_col(self):
        for c in TITLE_CANDIDATES:
            if self._df is not None and c in self._df.columns:
                return c
        return ""

    def _cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.status_label.setText("Cancelling...")

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_REQUESTS:
            self.Error.no_requests(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        doi = self.doi_combo.currentText()
        if not doi or doi not in self._df.columns:
            self.Error.no_doi(); return
        self.run_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0); self.status_label.setText("Starting...")
        self._worker = OAWorker(self._df, doi, self._title_col(), self.email)
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct); self.status_label.setText(msg)

    def _on_finished(self, out, error):
        self.run_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        if error and error != "Cancelled":
            self.status_label.setText("Failed"); self.Error.finder_error(error)
            self.Outputs.results.send(None); return
        if out is None or out.empty:
            self.status_label.setText("No results"); self.Outputs.results.send(None); return
        self._result = out
        self._fill_table(out)
        free = int((out["Is OA"] == "yes").sum())
        self.status_label.setText(f"Done — {free}/{len(out)} free")
        self.Information.done(free, len(out))
        self.Outputs.results.send(_df_to_table(out))

    def _fill_table(self, df):
        self.table.clear()
        self.table.setColumnCount(len(df.columns)); self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
        self.table.resizeColumnsToContents()

    def _open_row(self, r, _c):
        if self._result is None or r >= len(self._result):
            return
        row = self._result.iloc[r]
        url = row.get("PDF URL") or row.get("Landing URL")
        if not url:
            return
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            logger.warning("could not open %s", url)

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop(); self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWOAFinder).run()
