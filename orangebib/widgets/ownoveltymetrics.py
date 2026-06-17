# -*- coding: utf-8 -*-
"""
Novelty Metrics Widget
======================
Compute Uzzi-style combinatorial novelty / atypicality for each paper from the
unusual combinations in its references (or keywords), using
`biblium.addons.impact_metrics.compute_novelty_metrics`.
"""

import re
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QGridLayout,
                             QProgressBar, QTabWidget, QWidget, QVBoxLayout,
                             QTableWidget, QTableWidgetItem)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from biblium.addons.impact_metrics import compute_novelty_metrics
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    compute_novelty_metrics = None
    HAS_BIBLIUM = False

_STOPWORD_CACHE = {}


def _load_stopword_sets():
    """Load (general_stopwords, boilerplate_words) from the bundled
    orangebib/data/stopwords.xlsx. Cached. Returns (set, set)."""
    if "sets" in _STOPWORD_CACHE:
        return _STOPWORD_CACHE["sets"]
    general, boiler = set(), set()
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "data", "stopwords.xlsx")
        if os.path.exists(path):
            xl = pd.ExcelFile(path)
            if "general" in xl.sheet_names:
                g = pd.read_excel(xl, sheet_name="general")
                col = "english" if "english" in g.columns else g.columns[0]
                general = {str(w).strip().lower() for w in g[col].dropna()}
            if "specific" in xl.sheet_names:
                sp = pd.read_excel(xl, sheet_name="specific")
                wc = "Word" if "Word" in sp.columns else sp.columns[-1]
                boiler = {str(w).strip().lower() for w in sp[wc].dropna()}
    except Exception:  # noqa: BLE001
        pass
    _STOPWORD_CACHE["sets"] = (general, boiler)
    return general, boiler


COMBO_CANDIDATES = ["References", "Cited References", "CR", "oa_referenced_works",
                    "referenced_works", "Author Keywords", "Index Keywords",
                    "Keywords", "DE", "ID",
                    # free-text columns (tokenized into words automatically)
                    "Abstract", "Processed Abstract", "Title", "Document Title",
                    "AB", "TI"]


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas):
        data[var.name] = table.get_column(var)
    return pd.DataFrame(data)


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    attrs, metas, X, M = [], [], [], []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.6:
            attrs.append(ContinuousVariable(str(c))); X.append(s.fillna(0).values)
        else:
            metas.append(StringVariable(str(c)))
            M.append(df[c].astype(str).values)
    n = len(df)
    Xarr = np.column_stack(X) if X else np.empty((n, 0))
    Marr = np.column_stack(M) if M else np.empty((n, 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), Xarr, metas=Marr)


class NoveltyWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, combo_col, year_col, sep, baseline):
        super().__init__()
        self._df = df; self._combo = combo_col; self._yc = year_col
        self._sep = sep; self._baseline = baseline

    def run(self):
        try:
            self.progress.emit("Computing novelty...")
            df = self._df.copy()
            if "unique-id" not in df.columns:
                df["unique-id"] = [f"doc{i}" for i in range(len(df))]
            out = compute_novelty_metrics(
                df, combination_col=self._combo, id_col="unique-id",
                year_col=self._yc, sep=self._sep,
                baseline_years=self._baseline, verbose=False)
            self.finished.emit(out, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("novelty failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class _NumItem(QTableWidgetItem):
    def __init__(self, text, value=None):
        super().__init__(text)
        self._value = value

    def __lt__(self, other):
        try:
            if self._value is not None and getattr(other, "_value", None) is not None:
                return self._value < other._value
        except Exception:  # noqa: BLE001
            pass
        return super().__lt__(other)


class OWNoveltyMetrics(OWWidget):
    """Combinatorial novelty / atypicality per paper."""

    name = "Novelty Metrics"
    description = "Uzzi-style combinatorial novelty and atypicality per paper"
    icon = "icons/novelty_metrics.svg"
    priority = 385
    keywords = ["novelty", "atypicality", "uzzi", "disruption", "combination",
                "originality"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    class Outputs:
        per_document = Output("Per-document", Table, doc="Novelty metrics per paper")
        novel_subset = Output("Novel Documents", Table,
                              doc="Input rows for the most novel papers (>= percentile)")

    combo_col = settings.Setting("")
    sep_choice = settings.Setting("; ")
    baseline_years = settings.Setting(5)
    novelty_percentile = settings.Setting(75)
    remove_stopwords = settings.Setting(True)
    remove_general_words = settings.Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("Scored {} papers")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Options")
        grid = QGridLayout()
        grid.addWidget(QLabel("Combinations from:"), 0, 0)
        self.combo = QComboBox()
        self.combo.currentTextChanged.connect(self._on_combo_changed)
        grid.addWidget(self.combo, 0, 1)
        box.layout().addLayout(grid)
        # separator row — only relevant for list columns; hidden for free text
        self.sep_row = QWidget()
        sep_l = QGridLayout(self.sep_row); sep_l.setContentsMargins(0, 0, 0, 0)
        sep_l.addWidget(QLabel("Separator:"), 0, 0)
        self.sep_combo = QComboBox(); self.sep_combo.addItems(["; ", "|", ";", ", "])
        self.sep_combo.setCurrentText(self.sep_choice)
        self.sep_combo.currentTextChanged.connect(lambda t: setattr(self, "sep_choice", t))
        sep_l.addWidget(self.sep_combo, 0, 1)
        box.layout().addWidget(self.sep_row)
        gui.spin(box, self, "baseline_years", 1, 20, label="Baseline years:")
        gui.spin(box, self, "novelty_percentile", 0, 100,
                 label="Novel subset percentile:")
        cbox = gui.widgetBox(self.controlArea, "Word cleaning (text columns)")
        gui.checkBox(cbox, self, "remove_stopwords",
                     "Remove stop words (basic + my extended list)")
        gui.checkBox(cbox, self, "remove_general_words",
                     "Remove general/boilerplate words (concept Excel)")

        self.run_btn = QPushButton("Compute"); self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.summary_label = QLabel("No data"); self.summary_label.setWordWrap(True)
        self.mainArea.layout().addWidget(self.summary_label)
        self.view_tabs = QTabWidget()
        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.graph.setLabel("bottom", "Combinatorial novelty")
        self.graph.setLabel("left", "Papers")
        self.view_tabs.addTab(self.graph, "Distribution")
        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        self.view_tabs.addTab(self.table, "Per-document")
        self.mainArea.layout().addWidget(self.view_tabs)

        if not HAS_BIBLIUM:
            self.Error.no_biblium(); self.run_btn.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.combo.blockSignals(True)
        self.combo.clear()
        if self._df is not None and not self._df.empty:
            # offer only columns suitable for combinatorial novelty: delimited
            # list columns or free-text prose (Title/Abstract). Single-value
            # categoricals (Source Title, Language, …) are excluded.
            cols = [c for c in self._df.columns if self._classify(c) is not None]
            self.combo.addItems(cols)
            if self.combo_col in cols:
                self.combo.setCurrentText(self.combo_col)
            elif cols:
                self.combo_col = cols[0]
        self.combo.blockSignals(False)
        if hasattr(self, "sep_row"):
            self._on_combo_changed(self.combo.currentText())
        if data is None:
            self.Error.no_data()

    def _on_combo_changed(self, text):
        self.combo_col = text
        # separator only matters for list columns (free text is word-tokenized)
        if hasattr(self, "sep_row"):
            self.sep_row.setVisible(self._classify(text) == "list")

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py", "oa_publication_year"):
                return c
        return "Year"

    @staticmethod
    def _clean_cell(v):
        if v is None:
            return ""
        try:
            if isinstance(v, float) and pd.isna(v):
                return ""
        except Exception:  # noqa: BLE001
            pass
        sx = str(v).strip()
        return "" if sx.lower() in ("nan", "none", "<na>") else sx

    # known multi-valued list columns (delimited items)
    _LIST_COLS = {"references", "cited references", "cr", "oa_referenced_works",
                  "referenced_works", "author keywords", "index keywords",
                  "keywords", "de", "id", "keywords plus"}

    def _classify(self, col):
        """Return 'list' (delimited multi-value), 'text' (free prose to tokenize)
        or None (unsuitable: numeric or single-value categorical like Source
        Title / Language)."""
        if self._df is None or col not in self._df.columns:
            return None
        ser = self._df[col]
        if pd.api.types.is_numeric_dtype(ser):
            return None
        clean = ser.map(self._clean_cell)
        nonempty = clean[clean != ""]
        if len(nonempty) == 0:
            return None
        if col.lower() in self._LIST_COLS:
            return "list"
        # genuine list = uses a LIST delimiter ('|' or '; ') in most cells
        frac_listsep = nonempty.apply(
            lambda v: ("|" in v) or ("; " in v)).mean()
        if frac_listsep >= 0.5:
            return "list"
        # free text = (almost) unique per document AND reasonably long prose
        uniq_ratio = nonempty.nunique() / len(nonempty)
        avg_words = nonempty.apply(lambda v: len(v.split())).mean()
        if uniq_ratio >= 0.6 and avg_words >= 6:
            return "text"
        return None  # single-value categorical (Source Title, Language, ...)

    def _prepare_column(self, col):
        """Return (prepared_series, separator). List columns keep their item
        separator; free-text columns are tokenized into words. 'nan'/empty -> ''."""
        ser = self._df[col].map(self._clean_cell)
        if self._classify(col) == "list":
            sample = " ".join(ser.head(80).tolist())
            sep = next((c for c in ["||", "|", "; ", ";", ", "] if c in sample), "; ")
            return ser, sep
        # free text -> word tokens (>=3 letters), deduplicated, joined by "; "
        stops = set()
        if self.remove_stopwords or self.remove_general_words:
            general, boiler = _load_stopword_sets()
            if self.remove_stopwords:
                stops |= general
            if self.remove_general_words:
                stops |= boiler

        def _tok(t):
            if not t:
                return ""
            words = re.findall(r"[A-Za-z\u00C0-\u024F]{3,}", t.lower())
            words = [w for w in dict.fromkeys(words) if w not in stops]
            return "; ".join(words)
        return ser.map(_tok), "; "

    def _compute(self):
        self.Error.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        col = self.combo.currentText()
        if not col or col not in self._df.columns:
            return
        prepared, sep = self._prepare_column(col)
        work_df = self._df.copy()
        work_df["_novelty_src"] = prepared.values
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = NoveltyWorker(work_df, "_novelty_src",
                                     self._year_col(), sep,
                                     self.baseline_years)
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, out, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or out is None or (hasattr(out, "empty") and out.empty):
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "no novelty computed")
            self.Outputs.per_document.send(None)
            return
        col = "combinatorial_novelty" if "combinatorial_novelty" in out.columns else None
        if col is not None:
            vals = pd.to_numeric(out[col], errors="coerce").dropna()
            self.summary_label.setText(
                f"<b>{len(out)}</b> papers. Mean novelty: {vals.mean():.3f}, "
                f"median: {vals.median():.3f}.")
            self._render_hist(vals)
        out = self._clean_output(out)
        self._fill_table(out)
        self.status_label.setText(f"Done — {len(out)} papers")
        self.Information.done(len(out))
        self.Outputs.per_document.send(_df_to_table(out))
        # transfer the most-novel subset of the ORIGINAL input rows
        try:
            ncol = "combinatorial_novelty" if "combinatorial_novelty" in out.columns else None
            if ncol is not None and self._data is not None:
                vals = pd.to_numeric(out[ncol], errors="coerce").fillna(0).values
                thr = float(np.percentile(vals, self.novelty_percentile)) if len(vals) else 0
                # out rows align with the first len(out) processed input rows
                idx = [i for i in range(min(len(vals), len(self._data)))
                       if vals[i] >= thr]
                self.Outputs.novel_subset.send(self._data[idx] if idx else None)
            else:
                self.Outputs.novel_subset.send(None)
        except Exception:  # noqa: BLE001
            self.Outputs.novel_subset.send(None)

    @staticmethod
    def _clean_output(df):
        """Make list-valued cells (e.g. top novel combinations) readable;
        empty lists -> '' instead of '[]'."""
        df = df.copy()
        for c in df.columns:
            if df[c].apply(lambda v: isinstance(v, (list, tuple))).any():
                def _fmt(v):
                    if not isinstance(v, (list, tuple)) or len(v) == 0:
                        return ""
                    parts = []
                    for item in v[:5]:
                        if isinstance(item, (list, tuple)):
                            parts.append(" \u2013 ".join(str(x) for x in item[:2]))
                        else:
                            parts.append(str(item))
                    return "; ".join(parts)
                df[c] = df[c].apply(_fmt)
        return df

    def _fill_table(self, df):
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                v = df.iloc[r, c]
                num = None
                if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                    num = float(v)
                it = _NumItem(f"{v:g}" if num is not None else str(v), num)
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def _render_hist(self, vals):
        self.graph.clear()
        if vals.empty:
            return
        y, x = np.histogram(vals.values, bins=20)
        self.graph.addItem(pg.BarGraphItem(
            x0=x[:-1], x1=x[1:], height=y, y0=0, brush=pg.mkBrush("#9b59b6")))

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWNoveltyMetrics).run()
