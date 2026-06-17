# -*- coding: utf-8 -*-
"""
Self-Citation Rate Widget
========================
Self-citation is an author/journal-level phenomenon, not a per-document one, so
this widget aggregates it across the corpus. It builds the *within-corpus*
citation network (a document cites another document of the same dataset, matched
by DOI or title), and for every internal citation checks whether the citing and
cited papers share an author (author self-citation) or the same source / journal
(journal self-citation).

Outputs author-level and journal-level self-citation rates.
"""

import re
import logging
from collections import defaultdict
from typing import Optional, List

import numpy as np
import pandas as pd


from AnyQt.QtWidgets import (QLabel, QPushButton, QProgressBar, QTabWidget,
                             QTableWidget, QTableWidgetItem)

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


class _NumItem(QTableWidgetItem):
    """Table item that sorts numerically when given a value."""

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

_SEPS = ["||", "|", "; ", ";", ", "]
DOI_RE = re.compile(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+')


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
            metas.append(StringVariable(str(c))); M.append(df[c].astype(str).values)
    n = len(df)
    Xarr = np.column_stack(X) if X else np.empty((n, 0))
    Marr = np.column_stack(M) if M else np.empty((n, 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), Xarr, metas=Marr)


def _split(val) -> List[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    for sep in _SEPS:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def _norm_title(t) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', str(t).lower()).strip()


class OWSelfCitation(OWWidget):
    """Author- and journal-level self-citation rates."""

    name = "Self-Citation Rate"
    description = "Author and journal self-citation rates from the within-corpus citation network"
    icon = "icons/self_citation.svg"
    priority = 340
    keywords = ["self-citation", "self citation", "author", "journal",
                "source", "citation"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with references")

    class Outputs:
        authors = Output("Author Self-Citation", Table)
        journals = Output("Journal Self-Citation", Table)

    min_citations = settings.Setting(1)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_refs = Msg("No references / DOI columns found to build internal citations")

    class Information(OWWidget.Information):
        done = Msg("{} internal citations found")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None

        box = gui.widgetBox(self.controlArea, "Options")
        gui.spin(box, self, "min_citations", 1, 50,
                 label="Min citations given (author/journal):")
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
        self.tabs = QTabWidget()
        self.author_table = QTableWidget()
        self.journal_table = QTableWidget()
        self.tabs.addTab(self.author_table, "By author")
        self.tabs.addTab(self.journal_table, "By journal")
        self.mainArea.layout().addWidget(self.tabs)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        if data is None:
            self.Error.no_data()

    def _col(self, *names):
        if self._df is None:
            return None
        low = {str(c).lower(): c for c in self._df.columns}
        for n in names:
            if n in self._df.columns:
                return n
            if n.lower() in low:
                return low[n.lower()]
        return None

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        df = self._df.reset_index(drop=True)
        doi_col = self._col("DOI", "doi")
        ref_col = self._col("References", "Cited References", "CR",
                            "oa_referenced_works", "referenced_works")
        title_col = self._col("Title", "TI", "Document Title")
        auth_col = self._col("Authors", "Author", "AU", "Author full names")
        src_col = self._col("Source title", "Source", "Journal", "SO")
        if ref_col is None and doi_col is None:
            self.Error.no_refs(); return

        # build lookup keys for each document
        doc_dois = [None] * len(df)
        doc_titles = [None] * len(df)
        if doi_col is not None:
            for i, v in enumerate(df[doi_col]):
                m = DOI_RE.search(str(v).lower()) if v is not None else None
                doc_dois[i] = m.group(0) if m else None
        if title_col is not None:
            for i, v in enumerate(df[title_col]):
                doc_titles[i] = _norm_title(v) if v is not None else None
        doi_to_doc = {d: i for i, d in enumerate(doc_dois) if d}
        title_to_doc = {t: i for i, t in enumerate(doc_titles) if t and len(t) > 15}

        # authors / sources per doc
        doc_authors = [set(_split(df[auth_col].iloc[i])) if auth_col else set()
                       for i in range(len(df))]
        doc_source = [str(df[src_col].iloc[i]).strip().lower() if src_col else ""
                      for i in range(len(df))]

        # find internal citations: citing -> cited
        edges = []
        for i in range(len(df)):
            refs_text = str(df[ref_col].iloc[i]) if ref_col is not None else ""
            cited = set()
            if ref_col is not None and refs_text and refs_text.lower() != "nan":
                low = refs_text.lower()
                for d, j in doi_to_doc.items():
                    if d and d in low and j != i:
                        cited.add(j)
                if not cited and title_to_doc:
                    for t, j in title_to_doc.items():
                        if j != i and t in _norm_title(refs_text):
                            cited.add(j)
            for j in cited:
                edges.append((i, j))

        if not edges:
            self.summary_label.setText(
                "No internal citations could be matched (need DOIs in references "
                "or matchable titles).")
            self.Outputs.authors.send(None); self.Outputs.journals.send(None)
            return

        # author-level
        a_given = defaultdict(int); a_self = defaultdict(int)
        for (i, j) in edges:
            shared = doc_authors[i] & doc_authors[j]
            for a in doc_authors[i]:
                a_given[a] += 1
                if a in shared:
                    a_self[a] += 1
        a_rows = []
        for a, g in a_given.items():
            if g >= self.min_citations:
                a_rows.append({"Author": a, "Citations given": g,
                               "Self-citations": a_self[a],
                               "Self-citation rate": round(a_self[a] / g, 4)})
        a_df = pd.DataFrame(a_rows).sort_values(
            "Self-citation rate", ascending=False).reset_index(drop=True) \
            if a_rows else pd.DataFrame()

        # journal-level
        j_given = defaultdict(int); j_self = defaultdict(int)
        for (i, j) in edges:
            s = doc_source[i]
            if not s:
                continue
            j_given[s] += 1
            if doc_source[j] == s:
                j_self[s] += 1
        j_rows = []
        for s, g in j_given.items():
            if g >= self.min_citations:
                j_rows.append({"Source": s, "Citations given": g,
                               "Self-citations": j_self[s],
                               "Self-citation rate": round(j_self[s] / g, 4)})
        j_df = pd.DataFrame(j_rows).sort_values(
            "Self-citation rate", ascending=False).reset_index(drop=True) \
            if j_rows else pd.DataFrame()

        self.Information.done(len(edges))
        self.summary_label.setText(
            f"<b>{len(edges)}</b> internal citations · "
            f"{len(a_df)} authors · {len(j_df)} journals.")
        self._fill(self.author_table, a_df)
        self._fill(self.journal_table, j_df)
        self.Outputs.authors.send(_df_to_table(a_df))
        self.Outputs.journals.send(_df_to_table(j_df))

    @staticmethod
    def _fill(table, df):
        table.setSortingEnabled(False)
        table.clear()
        if df is None or df.empty:
            table.setRowCount(0); table.setColumnCount(0); return
        table.setColumnCount(len(df.columns))
        table.setRowCount(len(df))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                v = df.iloc[r, c]
                if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                    table.setItem(r, c, _NumItem(f"{v:g}", float(v)))
                else:
                    table.setItem(r, c, QTableWidgetItem(str(v)))
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)


if __name__ == "__main__":
    WidgetPreview(OWSelfCitation).run()
