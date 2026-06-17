# -*- coding: utf-8 -*-
"""
OpenAlex Enrichment Widget
=========================
Enrich any bibliographic table with OpenAlex metadata, matched by DOI.

Wraps :meth:`biblium.openalex_api.OpenAlexClient.enrich_dataframe`, which
fetches DOIs in batches (OpenAlex OR-filter, up to 50 per request) and adds
``oa_`` columns: citations, open-access status, topics/fields/domains,
concepts, SDGs, referenced works, yearly counts and institutions. The run is
resumable via an on-disk cache.
"""

import os
import time
import json
import logging
import tempfile
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGridLayout, QListWidget, QListWidgetItem, QHBoxLayout,
    QProgressBar, QTableWidget, QTableWidgetItem,
)

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.openalex_api import OpenAlexClient
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001 - any import-chain failure
    HAS_BIBLIUM = False
    OpenAlexClient = None

logger = logging.getLogger(__name__)

# oa_ enrichment fields (without the "oa_" prefix) that enrich_dataframe adds.
OA_FIELDS = [
    "openalex_id", "cited_by_count", "is_oa", "oa_status",
    "publication_year", "type", "primary_topic", "subfield", "field",
    "domain", "topics", "fields", "subfields", "domains", "concepts",
    "sdgs", "referenced_works", "n_referenced_works", "counts_by_year",
    "institutions", "institution_rors", "institution_countries",
    "n_institutions", "is_retracted", "is_paratext",
]

# Multi-valued oa_ columns that benefit from a single, consistent separator
# so the Bibliometric Counts widget (and other consumers) can split them.
MULTIVALUE_OA_COLS = [
    "oa_topics", "oa_fields", "oa_subfields", "oa_domains", "oa_concepts",
    "oa_sdgs", "oa_referenced_works", "oa_institutions",
    "oa_institution_rors", "oa_institution_countries",
]
CANONICAL_SEP = "; "


def _add_timeseries_columns(df: pd.DataFrame):
    """Derive Sleeping-Beauty-ready columns from oa_counts_by_year.

    Adds pipe-separated 'counts_by_year.year' and 'Citations by Year' columns
    (the format biblium's Sleeping Beauty / time-series code expects), and
    fills 'Year' / 'Cited by' from OpenAlex when missing. Returns (df, ok).
    """
    if "oa_counts_by_year" not in df.columns:
        return df, False
    years_col, cites_col = [], []
    for v in df["oa_counts_by_year"]:
        ys, cs = [], []
        try:
            data = (json.loads(v) if isinstance(v, str) and v.strip()
                    else (v if isinstance(v, list) else []))
        except Exception:  # noqa: BLE001
            data = []
        for e in data or []:
            y, c = (e or {}).get("year"), (e or {}).get("cited_by_count")
            if y is not None and c is not None:
                ys.append(str(int(y)))
                cs.append(str(int(c)))
        years_col.append("|".join(ys))
        cites_col.append("|".join(cs))
    df = df.copy()
    df["counts_by_year.year"] = years_col
    df["Citations by Year"] = cites_col
    if "Year" not in df.columns and "oa_publication_year" in df.columns:
        df["Year"] = df["oa_publication_year"]
    if "Cited by" not in df.columns and "oa_cited_by_count" in df.columns:
        df["Cited by"] = df["oa_cited_by_count"]
    return df, True


def _normalize_multivalue(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite mixed list separators ('|', '; ', ';') to a single '; '."""
    for col in MULTIVALUE_OA_COLS:
        if col not in df.columns:
            continue
        def _fix(v):
            if v is None or (isinstance(v, float) and v != v):
                return v
            text = str(v)
            parts = []
            for chunk in text.replace("|", "\x1f").replace("; ", "\x1f").replace(";", "\x1f").split("\x1f"):
                c = chunk.strip()
                if c:
                    parts.append(c)
            return CANONICAL_SEP.join(parts)
        df[col] = df[col].map(_fix)
    return df

DOI_CANDIDATES = ["DOI", "doi", "DI", "Doi", "DOIs", "oa_doi"]


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    domain = table.domain
    for var in list(domain.attributes) + list(domain.class_vars) + list(domain.metas):
        try:
            col = table.get_column(var)
        except Exception:  # noqa: BLE001
            continue
        if var.is_discrete:
            values = var.values
            data[var.name] = [
                values[int(v)] if (v == v and 0 <= int(v) < len(values)) else ""
                for v in col
            ]
        else:
            data[var.name] = col
    return pd.DataFrame(data)


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    attrs, metas, acols, mcols = [], [], [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            attrs.append(ContinuousVariable(str(c))); acols.append(c)
        else:
            metas.append(StringVariable(str(c))); mcols.append(c)
    domain = Domain(attrs, metas=metas)
    n = len(df)
    X = np.empty((n, len(attrs)), dtype=float)
    for i, c in enumerate(acols):
        X[:, i] = pd.to_numeric(df[c], errors="coerce").values
    M = np.empty((n, len(metas)), dtype=object)
    for i, c in enumerate(mcols):
        col = df[c]
        M[:, i] = [("" if (v is None or (isinstance(v, float) and v != v))
                    else str(v)) for v in col]
    return Table.from_numpy(domain, X, metas=M)


def _oa_request(session, email, endpoint, params, timeout=30, max_retries=5,
                stop_cb=None):
    """OpenAlex GET with bounded 429 back-off (honours Retry-After).

    Returns parsed JSON on success; raises RuntimeError on cancel, network
    failure, persistent rate-limiting (429) or other HTTP errors. This avoids
    biblium's built-in *infinite* 1-second 429 retry loop.
    """
    url = "https://api.openalex.org/" + endpoint.lstrip("/")
    p = dict(params or {})
    if email:
        p["mailto"] = email
    backoff = 1.0
    for attempt in range(max_retries + 1):
        if stop_cb and stop_cb():
            raise RuntimeError("Cancelled")
        try:
            r = session.get(url, params=p, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Network error: {type(exc).__name__}: {exc}")
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            if attempt >= max_retries:
                raise RuntimeError(
                    "OpenAlex rate limit (HTTP 429) persisted after retries. "
                    "Your network's shared IP is being throttled. Try later, "
                    "increase 'Delay (s)', or use a different connection.")
            ra = r.headers.get("Retry-After", "")
            wait = float(ra) if ra.isdigit() else backoff
            slept = 0.0
            while slept < wait:
                if stop_cb and stop_cb():
                    raise RuntimeError("Cancelled")
                time.sleep(0.2)
                slept += 0.2
            backoff = min(backoff * 2, 30.0)
            continue
        raise RuntimeError(f"OpenAlex HTTP {r.status_code}: {r.text[:150]}")
    return None


class EnrichWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str)  # DataFrame, error

    def __init__(self, df, doi_column, email, fields, batch_size, delay, cache_dir):
        super().__init__()
        self._df = df
        self._doi_column = doi_column
        self._email = email
        self._fields = fields
        self._batch_size = max(int(batch_size), 1)
        self._delay = delay
        self._cache_dir = cache_dir
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            client = OpenAlexClient(email=self._email or None)
            # Route every OpenAlex call through a bounded-retry wrapper so a
            # 429 can't trigger biblium's infinite 1s-retry loop.
            def _patched(endpoint, params=None, timeout=30):
                return _oa_request(client.session, self._email, endpoint,
                                   params, timeout=timeout, max_retries=5,
                                   stop_cb=lambda: self._stop)
            client._make_request = _patched

            df = self._df
            n = len(df)
            chunk = self._batch_size
            n_chunks = (n + chunk - 1) // chunk
            parts = []
            for i in range(n_chunks):
                if self._stop:
                    self.finished.emit(None, "Cancelled")
                    return
                sub = df.iloc[i * chunk:(i + 1) * chunk]
                self.progress.emit(int(i * 100 / max(n_chunks, 1)),
                                   f"Batch {i + 1}/{n_chunks} ({len(sub)} rows)...")
                sub_cache = (os.path.join(self._cache_dir, f"chunk_{i:05d}")
                             if self._cache_dir else None)
                if sub_cache:
                    os.makedirs(sub_cache, exist_ok=True)
                try:
                    enriched = client.enrich_dataframe(
                        sub, doi_column=self._doi_column,
                        fields=self._fields or None, progress=False,
                        delay=self._delay, batch_size=self._batch_size,
                        cache_dir=sub_cache)
                except RuntimeError as exc:
                    if str(exc) == "Cancelled":
                        self.finished.emit(None, "Cancelled")
                        return
                    if parts:  # return what we have so far
                        out = pd.concat(parts, ignore_index=True)
                        self.finished.emit(out, "PARTIAL:" + str(exc))
                    else:
                        self.finished.emit(None, str(exc))
                    return
                parts.append(enriched)
            out = pd.concat(parts, ignore_index=True) if parts else df
            self.progress.emit(100, "Done")
            self.finished.emit(out, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAlex enrichment failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWOpenAlexEnrichment(OWWidget):
    """Enrich a bibliographic table with OpenAlex metadata by DOI."""

    name = "OpenAlex Enrichment"
    description = "Add OpenAlex metadata (citations, OA, topics, SDGs, refs) by DOI"
    icon = "icons/openalex_enrichment.svg"
    priority = 20
    keywords = ["openalex", "enrich", "enrichment", "doi", "citations",
                "open access", "topics", "concepts", "augment"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table (needs a DOI column)")

    class Outputs:
        data = Output("Enriched Data", Table, doc="Input data plus oa_ columns")

    doi_column = settings.Setting("")
    email = settings.Setting("")
    batch_size = settings.Setting(50)
    delay = settings.Setting(0.15)
    use_cache = settings.Setting(True)
    normalize_lists = settings.Setting(True)
    sb_columns = settings.Setting(True)
    selected_fields = settings.Setting([])  # empty => all

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium is required. Install biblium>=2.16.")
        no_doi = Msg("Select the DOI column")
        enrich_error = Msg("Enrichment failed: {}")

    class Warning(OWWidget.Warning):
        partial = Msg("{}")
        no_timeseries = Msg("Time-series columns need the 'oa_counts_by_year' "
                            "field enabled.")

    class Information(OWWidget.Information):
        done = Msg("{}")

    def __init__(self):
        super().__init__()
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._worker: Optional[EnrichWorker] = None

        self._setup_controls()
        self._setup_main_area()

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _setup_controls(self):
        src = gui.widgetBox(self.controlArea, "Source")
        grid = QGridLayout()
        grid.addWidget(QLabel("DOI column:"), 0, 0)
        self.doi_combo = QComboBox()
        self.doi_combo.currentTextChanged.connect(
            lambda t: setattr(self, "doi_column", t))
        grid.addWidget(self.doi_combo, 0, 1)

        grid.addWidget(QLabel("Email (polite pool):"), 1, 0)
        self.email_edit = QLineEdit(self.email)
        self.email_edit.setPlaceholderText("you@university.edu")
        self.email_edit.textChanged.connect(lambda t: setattr(self, "email", t))
        grid.addWidget(self.email_edit, 1, 1)
        src.layout().addLayout(grid)

        opt = gui.widgetBox(self.controlArea, "Options")
        ogrid = QGridLayout()
        ogrid.addWidget(QLabel("Batch size:"), 0, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 50)
        self.batch_spin.setValue(self.batch_size)
        self.batch_spin.valueChanged.connect(lambda v: setattr(self, "batch_size", v))
        ogrid.addWidget(self.batch_spin, 0, 1)

        ogrid.addWidget(QLabel("Delay (s):"), 1, 0)
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 5.0)
        self.delay_spin.setSingleStep(0.05)
        self.delay_spin.setDecimals(2)
        self.delay_spin.setValue(self.delay)
        self.delay_spin.valueChanged.connect(lambda v: setattr(self, "delay", v))
        ogrid.addWidget(self.delay_spin, 1, 1)
        opt.layout().addLayout(ogrid)

        self.cache_cb = QCheckBox("Resume from cache (recommended)")
        self.cache_cb.setChecked(self.use_cache)
        self.cache_cb.toggled.connect(lambda c: setattr(self, "use_cache", c))
        opt.layout().addWidget(self.cache_cb)

        self.norm_cb = QCheckBox("Normalize topics/concepts/... to '; ' (count-ready)")
        self.norm_cb.setChecked(self.normalize_lists)
        self.norm_cb.toggled.connect(lambda c: setattr(self, "normalize_lists", c))
        opt.layout().addWidget(self.norm_cb)

        self.sb_cb = QCheckBox("Add citation time-series cols (for Sleeping Beauty)")
        self.sb_cb.setChecked(self.sb_columns)
        self.sb_cb.setToolTip(
            "Derives 'counts_by_year.year' and 'Citations by Year' from "
            "oa_counts_by_year (requires that field enabled).")
        self.sb_cb.toggled.connect(lambda c: setattr(self, "sb_columns", c))
        opt.layout().addWidget(self.sb_cb)

        fbox = gui.widgetBox(self.controlArea, "Fields to add")
        self.fields_list = QListWidget()
        self.fields_list.setMaximumHeight(170)
        for f in OA_FIELDS:
            it = QListWidgetItem("oa_" + f)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            checked = (not self.selected_fields) or (f in self.selected_fields)
            it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            it.setData(Qt.UserRole, f)
            self.fields_list.addItem(it)
        fbox.layout().addWidget(self.fields_list)
        brow = QHBoxLayout()
        all_btn = QPushButton("Select All")
        all_btn.clicked.connect(lambda: self._set_all_fields(True))
        none_btn = QPushButton("Deselect All")
        none_btn.clicked.connect(lambda: self._set_all_fields(False))
        brow.addWidget(all_btn); brow.addWidget(none_btn)
        fbox.layout().addLayout(brow)

        self.run_btn = QPushButton("Enrich from OpenAlex")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._enrich)
        self.controlArea.layout().addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.controlArea.layout().addWidget(self.cancel_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)

    def _setup_main_area(self):
        box = gui.widgetBox(self.mainArea, "Summary")
        self.summary_label = QLabel("No data")
        self.summary_label.setWordWrap(True)
        box.layout().addWidget(self.summary_label)

        prev = gui.widgetBox(self.mainArea, "Enriched columns (preview)")
        self.preview_table = QTableWidget()
        self.preview_table.setMinimumHeight(320)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        prev.layout().addWidget(self.preview_table)

    def _set_all_fields(self, state: bool):
        for i in range(self.fields_list.count()):
            self.fields_list.item(i).setCheckState(
                Qt.Checked if state else Qt.Unchecked)

    def _checked_fields(self) -> List[str]:
        out = []
        for i in range(self.fields_list.count()):
            it = self.fields_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self._populate_doi_combo()
        if data is None:
            self.Error.no_data()

    def _populate_doi_combo(self):
        self.doi_combo.blockSignals(True)
        self.doi_combo.clear()
        if self._df is not None and not self._df.empty:
            cols = list(self._df.columns)
            self.doi_combo.addItems(cols)
            chosen = None
            if self.doi_column in cols:
                chosen = self.doi_column
            else:
                for cand in DOI_CANDIDATES:
                    if cand in cols:
                        chosen = cand
                        break
            if chosen:
                self.doi_combo.setCurrentText(chosen)
                self.doi_column = chosen
        self.doi_combo.blockSignals(False)

    def _enrich(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            return
        if self._df is None or self._df.empty:
            self.Error.no_data()
            return
        doi_col = self.doi_combo.currentText()
        if not doi_col or doi_col not in self._df.columns:
            self.Error.no_doi()
            return

        checked = self._checked_fields()
        self.selected_fields = checked
        # If everything is selected, pass None so biblium keeps all columns
        # (including any it may add in future versions).
        fields_arg = None if len(checked) == len(OA_FIELDS) else checked
        cache_dir = None
        if self.use_cache:
            cache_dir = os.path.join(tempfile.gettempdir(), "biblium_oa_enrich_cache")
            os.makedirs(cache_dir, exist_ok=True)

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting enrichment...")

        self._worker = EnrichWorker(
            self._df, doi_col, self.email, fields_arg,
            self.batch_size, self.delay, cache_dir)
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, out: Optional[pd.DataFrame], error: str):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        if error == "Cancelled":
            self.status_label.setText("Cancelled")
            return
        partial_msg = ""
        if isinstance(error, str) and error.startswith("PARTIAL:"):
            partial_msg = error[len("PARTIAL:"):]
            error = ""
        if error or out is None:
            self.status_label.setText("Failed")
            self.Error.enrich_error(error or "unknown error")
            self.Outputs.data.send(None)
            return
        if self.normalize_lists:
            out = _normalize_multivalue(out)
        if self.sb_columns:
            out, ok = _add_timeseries_columns(out)
            if not ok:
                self.Warning.no_timeseries()
        oa_cols = [c for c in out.columns if str(c).startswith("oa_")]
        matched = 0
        if "oa_openalex_id" in out.columns:
            matched = int(out["oa_openalex_id"].astype(str).str.len().gt(0).sum())
        self.summary_label.setText(
            f"Enriched <b>{len(out)}</b> rows, matched <b>{matched}</b> on OpenAlex.<br>"
            f"Added {len(oa_cols)} columns: {', '.join(oa_cols[:8])}"
            + (" ..." if len(oa_cols) > 8 else ""))
        if partial_msg:
            self.status_label.setText("Partial results — stopped early")
            self.Warning.partial(partial_msg)
        else:
            self.status_label.setText(f"Done — {matched}/{len(out)} matched")
        self.Information.done(f"Matched {matched}/{len(out)} rows on OpenAlex")
        self._fill_preview(out, oa_cols)
        self.Outputs.data.send(_df_to_table(out))

    def _fill_preview(self, df: pd.DataFrame, oa_cols, max_rows: int = 500):
        """Show identifier columns + the oa_ columns in the main-area table."""
        id_cols = [c for c in ("DOI", "Title", "Authors", "Year", "Cited by",
                               "counts_by_year.year", "Citations by Year")
                   if c in df.columns]
        cols = id_cols + [c for c in oa_cols if c not in id_cols]
        if not cols:
            cols = list(df.columns)
        view = df[cols].head(max_rows)
        t = self.preview_table
        t.clear()
        t.setColumnCount(len(cols))
        t.setRowCount(len(view))
        t.setHorizontalHeaderLabels([str(c) for c in cols])
        for r in range(len(view)):
            for c in range(len(cols)):
                v = view.iloc[r, c]
                if v is None or (isinstance(v, float) and v != v):
                    txt = ""
                elif isinstance(v, float):
                    txt = f"{v:,.3f}"
                elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                    txt = f"{int(v):,}"
                else:
                    txt = str(v)
                t.setItem(r, c, QTableWidgetItem(txt))
        t.resizeColumnsToContents()

    def _cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.status_label.setText("Cancelling after current batch...")

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWOpenAlexEnrichment).run()
