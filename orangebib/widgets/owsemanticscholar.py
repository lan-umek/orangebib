# -*- coding: utf-8 -*-
"""
Semantic Scholar Widget
======================
Enrich papers with Semantic Scholar data by DOI: citation counts, influential
citation counts, fields of study, the AI-generated TLDR summary and an
open-access PDF link. Uses the Semantic Scholar Graph API batch endpoint
(an API key is optional but raises the rate limit).
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
S2_FIELDS = ("title,year,citationCount,influentialCitationCount,"
             "fieldsOfStudy,tldr,openAccessPdf,externalIds")
BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"


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


def _norm_doi(doi):
    if not isinstance(doi, str):
        return None
    d = doi.strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p):]
    d = d.strip()
    return d if d.startswith("10.") and "/" in d else None


class S2Worker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, doi_col, api_key):
        super().__init__()
        self._df = df; self._doi = doi_col; self._key = api_key; self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            session = requests.Session()
            session.headers.update({"User-Agent": "Biblium-Orange-S2"})
            if self._key:
                session.headers.update({"x-api-key": self._key})
            norm = self._df[self._doi].map(_norm_doi)
            valid_idx = [i for i, d in zip(self._df.index, norm) if d]
            dois = [norm[i] for i in valid_idx]
            uniq = list(dict.fromkeys(dois))
            n = len(uniq)
            by_doi = {}
            batch = 100
            done = 0
            for i in range(0, n, batch):
                if self._stop:
                    self.finished.emit(None, "Cancelled"); return
                chunk = uniq[i:i + batch]
                ids = [f"DOI:{d}" for d in chunk]
                for attempt in range(4):
                    try:
                        resp = session.post(BATCH, params={"fields": S2_FIELDS},
                                            json={"ids": ids}, timeout=30)
                    except Exception as exc:  # noqa: BLE001
                        self.finished.emit(None, f"Network error: {exc}"); return
                    if resp.status_code == 200:
                        results = resp.json()
                        for d, item in zip(chunk, results):
                            if item:
                                by_doi[d] = item
                        break
                    if resp.status_code == 429:
                        time.sleep(2 * (attempt + 1)); continue
                    break
                done += len(chunk)
                self.progress.emit(int(done * 100 / max(n, 1)), f"{done}/{n} DOIs")
                time.sleep(1.0 if not self._key else 0.2)

            def _enrich_row(d):
                item = by_doi.get(_norm_doi(str(d)) or "")
                if not item:
                    return {}
                tldr = (item.get("tldr") or {}).get("text") or ""
                oa = (item.get("openAccessPdf") or {}).get("url") or ""
                fos = item.get("fieldsOfStudy") or []
                ext = item.get("externalIds") or {}
                return {
                    "s2_paper_id": item.get("paperId") or "",
                    "s2_citation_count": item.get("citationCount"),
                    "s2_influential_citations": item.get("influentialCitationCount"),
                    "s2_fields_of_study": "; ".join(fos),
                    "s2_tldr": tldr,
                    "s2_oa_pdf": oa,
                    "s2_corpus_id": str(ext.get("CorpusId") or ""),
                }
            out = self._df.copy()
            enr = out[self._doi].map(_enrich_row)
            for col in ["s2_paper_id", "s2_citation_count", "s2_influential_citations",
                        "s2_fields_of_study", "s2_tldr", "s2_oa_pdf", "s2_corpus_id"]:
                out[col] = [e.get(col) for e in enr]
            self.finished.emit((out, len(by_doi), n), "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("semantic scholar failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWSemanticScholar(OWWidget):
    """Enrich papers with Semantic Scholar metadata by DOI."""

    name = "Semantic Scholar"
    description = "Enrich by DOI with S2 citations, influential cites, fields, TLDR"
    icon = "icons/semantic_scholar.svg"
    priority = 30
    keywords = ["semantic scholar", "s2", "tldr", "influential", "enrich",
                "citations", "fields of study"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Data with a DOI column")

    class Outputs:
        data = Output("Enriched Data", Table, doc="Input + s2_ columns")

    doi_col = settings.Setting("")
    api_key = settings.Setting("")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_requests = Msg("The 'requests' package is required.")
        no_doi = Msg("Select the DOI column")
        s2_error = Msg("{}")

    class Information(OWWidget.Information):
        done = Msg("Matched {} of {} DOIs on Semantic Scholar")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Source")
        grid = QGridLayout()
        grid.addWidget(QLabel("DOI column:"), 0, 0)
        self.doi_combo = QComboBox()
        self.doi_combo.currentTextChanged.connect(lambda t: setattr(self, "doi_col", t))
        grid.addWidget(self.doi_combo, 0, 1)
        grid.addWidget(QLabel("API key (optional):"), 1, 0)
        self.key_edit = QLineEdit(self.api_key)
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("or set SEMANTIC_SCHOLAR_API_KEY")
        self.key_edit.textChanged.connect(lambda t: setattr(self, "api_key", t))
        grid.addWidget(self.key_edit, 1, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Enrich from Semantic Scholar")
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

        self.summary_label = QLabel("No data"); self.summary_label.setWordWrap(True)
        self.mainArea.layout().addWidget(self.summary_label)

        if not HAS_REQUESTS:
            self.Error.no_requests()
            self.run_btn.setEnabled(False)

    def _resolved_key(self):
        return self.api_key.strip() or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_REQUESTS:
            self.Error.no_requests()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        cols = list(self._df.columns) if self._df is not None else []
        ordered = [c for c in DOI_CANDIDATES if c in cols] + [c for c in cols if c not in DOI_CANDIDATES]
        self.doi_combo.blockSignals(True); self.doi_combo.clear(); self.doi_combo.addItems(ordered)
        if self.doi_col in ordered:
            self.doi_combo.setCurrentText(self.doi_col)
        self.doi_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop(); self.status_label.setText("Cancelling...")

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
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self._worker = S2Worker(self._df, doi, self._resolved_key())
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct); self.status_label.setText(msg)

    def _on_finished(self, result, error):
        self.run_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        if error and error != "Cancelled":
            self.status_label.setText("Failed"); self.Error.s2_error(error)
            self.Outputs.data.send(None); return
        if result is None:
            self.status_label.setText("Cancelled"); return
        out, matched, n = result
        self.summary_label.setText(
            f"Matched <b>{matched}</b> of {n} DOIs on Semantic Scholar. "
            f"Added s2_ columns (citations, influential, fields, TLDR, OA PDF).")
        self.status_label.setText(f"Done — {matched}/{n} matched")
        self.Information.done(matched, n)
        self.Outputs.data.send(_df_to_table(out))

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop(); self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWSemanticScholar).run()
