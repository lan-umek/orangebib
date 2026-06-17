# -*- coding: utf-8 -*-
"""
Main Information Widget
=======================
Orange widget for computing main bibliometric information and statistics.

Uses biblium 2.16+ for:
- Performance indicators (H-index, G-index, A/R/W/T/Pi/HG indices, Gini, ...)
- Time series analysis (growth rates, trends)
- Descriptive statistics for various entity types

Provides comprehensive dataset overview with user-selectable options.
Falls back to a slim local implementation when biblium isn't available.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from AnyQt.QtGui import QFont
from AnyQt.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)

from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output

from orangebib.base import (
    BaseBibliumWidget,
    fmt_value,
    get_biblium_submodule,
    safe_numeric,
    split_list_cell,
    unique_items_in_list_column,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Tiny fallback implementations of bibliometric indicators
# (only used when biblium isn't installed)
# =============================================================================

def _h_index(citations: list[float]) -> int:
    if not citations:
        return 0
    s = sorted((int(c) for c in citations if pd.notna(c) and c > 0), reverse=True)
    h = 0
    for i, c in enumerate(s, 1):
        if c >= i:
            h = i
        else:
            break
    return h


def _g_index(citations: list[float]) -> int:
    if not citations:
        return 0
    s = sorted((int(c) for c in citations if pd.notna(c)), reverse=True)
    cumsum = 0
    g = 0
    for i, c in enumerate(s, 1):
        cumsum += c
        if cumsum >= i * i:
            g = i
    return g


def _a_index(citations: list[float]) -> float:
    h = _h_index(citations)
    if h == 0:
        return 0.0
    s = sorted((int(c) for c in citations if pd.notna(c)), reverse=True)
    return float(np.mean(s[:h])) if s else 0.0


def _r_index(citations: list[float]) -> float:
    h = _h_index(citations)
    if h == 0:
        return 0.0
    s = sorted((int(c) for c in citations if pd.notna(c)), reverse=True)
    return float(np.sqrt(sum(s[:h]))) if s else 0.0


def _m_index_median(citations: list[float]) -> float:
    h = _h_index(citations)
    if h == 0:
        return 0.0
    s = sorted((int(c) for c in citations if pd.notna(c)), reverse=True)
    return float(np.median(s[:h])) if s else 0.0


def _e_index(citations: list[float]) -> float:
    h = _h_index(citations)
    if h == 0:
        return 0.0
    s = sorted((int(c) for c in citations if pd.notna(c)), reverse=True)
    excess = sum(c - h for c in s[:h])
    return float(np.sqrt(excess)) if excess > 0 else 0.0


def _q2_index(citations: list[float]) -> float:
    h = _h_index(citations); m = _m_index_median(citations)
    return float(np.sqrt(h * m)) if h and m else 0.0


def _v_index(citations: list[float], n_papers: int) -> float:
    h = _h_index(citations)
    return float(100.0 * h / n_papers) if n_papers else 0.0


def _pi_index(citations: list[float], n_papers: int) -> float:
    s = sorted((int(c) for c in citations if pd.notna(c)), reverse=True)
    if n_papers <= 0:
        return 0.0
    elite = int(np.floor(np.sqrt(n_papers)))
    return float(sum(s[:elite]) / 100.0) if elite > 0 else 0.0


def _p_index(citations: list[float], n_papers: int) -> float:
    s = sorted((int(c) for c in citations if pd.notna(c)), reverse=True)
    c = sum(s)
    return float((c * c / n_papers) ** (1.0 / 3.0)) if (n_papers and c) else 0.0


def _rational_h_index(citations: list[float]) -> float:
    s = sorted((int(c) for c in citations if pd.notna(c)), reverse=True)
    h = _h_index(citations)
    if not s:
        return 0.0
    needed = sum(max(0, (h + 1) - (s[i] if i < len(s) else 0)) for i in range(h + 1))
    return float(h + 1 - needed / (2 * h + 1))


def _m_quotient(citations: list[float], years) -> float:
    h = _h_index(citations)
    yy = [int(y) for y in years if pd.notna(y) and 1500 < int(y) < 2100]
    if h <= 0 or not yy:
        return 0.0
    age = (max(yy) - min(yy)) + 1
    return float(h / age) if age > 0 else 0.0


def _gini_index(citations: list[float]) -> float:
    valid = [c for c in citations if pd.notna(c) and c >= 0]
    if not valid or len(valid) < 2:
        return 0.0
    s = sorted(valid)
    n = len(s)
    cumsum = np.cumsum(s)
    if cumsum[-1] == 0:
        return 0.0
    return float(
        (2 * sum((i + 1) * v for i, v in enumerate(s)) - (n + 1) * cumsum[-1])
        / (n * cumsum[-1])
    )


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWMainInfo(BaseBibliumWidget):
    """Compute and display main bibliometric information."""

    name = "Main Information"
    description = (
        "Compute comprehensive bibliometric statistics and dataset overview"
    )
    icon = "icons/main_info.svg"
    priority = 100
    keywords = ["main", "info", "statistics", "summary", "overview",
                "h-index", "bibliometric"]

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")

    class Outputs:
        summary = Output("Summary", Table, doc="Dataset summary table")
        performance = Output("Performance", Table,
                             doc="Performance indicators table")
        timeseries = Output("Time Series", Table,
                            doc="Time series analysis table")
        descriptives = Output("Descriptives", Table,
                              doc="Descriptive statistics table")
        all_stats = Output("All Statistics", Table,
                           doc="Combined statistics table")

    # Settings — what to compute
    compute_summary = settings.Setting(True)
    compute_performance = settings.Setting(True)
    compute_timeseries = settings.Setting(True)
    compute_descriptives = settings.Setting(True)

    # Settings — performance options
    performance_mode = settings.Setting("extended")  # core, extended, full

    # Settings — time series options
    exclude_last_year = settings.Setting(True)

    # Settings — descriptives options
    desc_year = settings.Setting(True)
    desc_source = settings.Setting(True)
    desc_doctype = settings.Setting(True)
    desc_citations = settings.Setting(True)
    desc_keywords = settings.Setting(True)
    desc_language = settings.Setting(False)
    desc_openaccess = settings.Setting(False)
    extra_stats = settings.Setting(False)

    auto_apply = settings.Setting(True)

    want_main_area = True
    resizing_enabled = True

    def __init__(self):
        super().__init__()

        self._data: Table | None = None
        self._df: pd.DataFrame | None = None
        self._summary_df: pd.DataFrame | None = None
        self._performance_df: pd.DataFrame | None = None
        self._timeseries_df: pd.DataFrame | None = None
        self._descriptives_df: pd.DataFrame | None = None

        self._setup_control_area()
        self._setup_main_area()

    # =========================================================================
    # GUI SETUP
    # =========================================================================

    def _setup_control_area(self):
        # Statistics selection
        stats_box = gui.widgetBox(self.controlArea, "Statistics to Compute")

        gui.checkBox(stats_box, self, "compute_summary", "Dataset Summary",
                     tooltip="Basic counts: documents, sources, authors, etc.",
                     callback=self._on_option_changed)
        gui.checkBox(stats_box, self, "compute_performance",
                     "Performance Indicators",
                     tooltip="H-index, G-index, citations statistics",
                     callback=self._on_option_changed)
        gui.checkBox(stats_box, self, "compute_timeseries",
                     "Time Series Analysis",
                     tooltip="Publication trends, growth rates",
                     callback=self._on_option_changed)
        gui.checkBox(stats_box, self, "compute_descriptives",
                     "Descriptive Statistics",
                     tooltip="Detailed statistics per column type",
                     callback=self._on_option_changed)

        # Performance options
        perf_box = gui.widgetBox(self.controlArea, "Performance Options")
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Detail level:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Core (H-index, citations)",
            "Extended (+G-index, quartiles)",
            "Full (+A,R,W indices, Gini)",
        ])
        mode_map = {"core": 0, "extended": 1, "full": 2}
        self.mode_combo.setCurrentIndex(mode_map.get(self.performance_mode, 1))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        perf_box.layout().addLayout(mode_layout)

        # Time series options
        ts_box = gui.widgetBox(self.controlArea, "Time Series Options")
        gui.checkBox(ts_box, self, "exclude_last_year",
                     "Exclude last year for growth rates",
                     tooltip="Last year may be incomplete",
                     callback=self._on_option_changed)

        # Descriptives options
        desc_box = gui.widgetBox(self.controlArea, "Descriptive Statistics")
        desc_grid = QGridLayout()
        desc_grid.setSpacing(2)

        self._make_desc_checkbox(desc_grid, 0, 0, "_desc_cb_year",
                                 "desc_year", "Publication years")
        self._make_desc_checkbox(desc_grid, 0, 1, "_desc_cb_source",
                                 "desc_source", "Sources/Journals")
        self._make_desc_checkbox(desc_grid, 1, 0, "_desc_cb_doctype",
                                 "desc_doctype", "Document types")
        self._make_desc_checkbox(desc_grid, 1, 1, "_desc_cb_citations",
                                 "desc_citations", "Citations")
        self._make_desc_checkbox(desc_grid, 2, 0, "_desc_cb_keywords",
                                 "desc_keywords", "Keywords")
        self._make_desc_checkbox(desc_grid, 2, 1, "_desc_cb_language",
                                 "desc_language", "Language")
        self._make_desc_checkbox(desc_grid, 3, 0, "_desc_cb_openaccess",
                                 "desc_openaccess", "Open Access")
        self._make_desc_checkbox(desc_grid, 3, 1, "_desc_cb_extra",
                                 "extra_stats", "Extra statistics",
                                 tooltip="Additional metrics like entropy, "
                                         "concentration")

        desc_box.layout().addLayout(desc_grid)

        # Apply button
        self.apply_btn = gui.button(
            self.controlArea, self, "Compute Statistics",
            callback=self.commit, autoDefault=False,
        )
        self.apply_btn.setMinimumHeight(35)
        gui.checkBox(self.controlArea, self, "auto_apply",
                     "Apply Automatically")
        self.controlArea.layout().addStretch(1)

    def _make_desc_checkbox(self, grid: QGridLayout, row: int, col: int,
                            attr_name: str, setting_name: str, label: str,
                            tooltip: str = "") -> None:
        cb = QCheckBox(label)
        cb.setChecked(getattr(self, setting_name))
        if tooltip:
            cb.setToolTip(tooltip)
        # Capture setting_name in default arg to avoid late-binding bug
        cb.toggled.connect(
            lambda checked, name=setting_name: (
                setattr(self, name, checked) or self._on_option_changed()
            )
        )
        setattr(self, attr_name, cb)
        grid.addWidget(cb, row, col)

    def _setup_main_area(self):
        self.tabs = QTabWidget()
        self.mainArea.layout().addWidget(self.tabs)

        # Summary tab
        self.summary_widget = QWidget()
        summary_layout = QVBoxLayout(self.summary_widget)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Consolas", 10))
        summary_layout.addWidget(self.summary_text)
        self.tabs.addTab(self.summary_widget, "Summary")

        # Performance tab
        self.performance_widget = QWidget()
        perf_layout = QVBoxLayout(self.performance_widget)
        self.performance_table = QTableWidget()
        self.performance_table.setSelectionBehavior(QTableWidget.SelectRows)
        perf_layout.addWidget(self.performance_table)
        self.tabs.addTab(self.performance_widget, "Performance")

        # Time Series tab
        self.timeseries_widget = QWidget()
        ts_layout = QVBoxLayout(self.timeseries_widget)
        self.timeseries_table = QTableWidget()
        self.timeseries_table.setSelectionBehavior(QTableWidget.SelectRows)
        ts_layout.addWidget(self.timeseries_table)
        self.tabs.addTab(self.timeseries_widget, "Time Series")

        # Descriptives tab
        self.descriptives_widget = QWidget()
        desc_layout = QVBoxLayout(self.descriptives_widget)
        self.descriptives_table = QTableWidget()
        self.descriptives_table.setSelectionBehavior(QTableWidget.SelectRows)
        desc_layout.addWidget(self.descriptives_table)
        self.tabs.addTab(self.descriptives_widget, "Descriptives")

    def _on_option_changed(self):
        # Recompute right away so toggling an option (Language, Extra
        # statistics, ...) has an immediate, visible effect.
        self.commit()

    def _on_mode_changed(self, index: int):
        modes = ["core", "extended", "full"]
        self.performance_mode = modes[index]
        if self.auto_apply:
            self.commit()

    # =========================================================================
    # DATA HANDLING
    # =========================================================================

    @Inputs.data
    def set_data(self, data: Table | None):
        self.clear_messages()
        self._data = data
        self._df = None
        self._clear_results()

        if data is None:
            self.Error.no_data()
            return

        self._df = self._table_to_df(data)
        if self.auto_apply:
            self.commit()

    def _clear_results(self):
        self._summary_df = None
        self._performance_df = None
        self._timeseries_df = None
        self._descriptives_df = None

        self.summary_text.clear()
        for tbl in (self.performance_table, self.timeseries_table,
                    self.descriptives_table):
            tbl.clear()
            tbl.setRowCount(0)

    def commit(self):
        self._compute_all()

    # =========================================================================
    # COMPUTATION ORCHESTRATION
    # =========================================================================

    def _compute_all(self):
        self.clear_messages()
        self._clear_results()

        if self._df is None or self._df.empty:
            self.Error.no_data()
            self._send_outputs()
            return

        try:
            if self.compute_summary:
                self._compute_summary()
            if self.compute_performance:
                self._compute_performance()
            if self.compute_timeseries:
                self._compute_timeseries()
            if self.compute_descriptives:
                self._compute_descriptives()
            self._send_outputs()
            self.Information.computed(len(self._df))
        except Exception as exc:  # noqa: BLE001
            import traceback
            logger.error("Computation error: %s\n%s", exc,
                         traceback.format_exc())
            self.Error.compute_error(str(exc))

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def _compute_summary(self):
        df = self._df
        rows: list[tuple[str, str, Any]] = []
        sep = "; "

        author_col = self._find_column(df, "authors")
        if author_col:
            sep = self._detect_separator(df[author_col])

        rows.append(("Dataset", "Number of documents", len(df)))

        # Timespan
        year_col = self._find_column(df, "year")
        if year_col:
            years = safe_numeric(df[year_col]).dropna()
            if len(years) > 0:
                min_y = int(years.min())
                max_y = int(years.max())
                rows.append(("Dataset", "Timespan", f"{min_y} - {max_y}"))
                rows.append(("Dataset", "Number of years", max_y - min_y + 1))

        # Sources
        source_col = self._find_column(df, "source")
        if source_col:
            rows.append(("Dataset", "Number of sources",
                         int(df[source_col].dropna().nunique())))

        # Authors
        if author_col:
            n_authors = len(unique_items_in_list_column(df[author_col], sep))
            rows.append(("Dataset", "Number of authors", n_authors))
            if len(df) > 0:
                rows.append(("Dataset", "Authors per document",
                             round(n_authors / len(df), 2)))

        # Geo / affiliations / keywords / index keywords / references
        for ind_label, key in [
            ("Number of countries", "country"),
            ("Number of affiliations", "affiliation"),
            ("Number of author keywords", "keywords"),
            ("Number of index keywords", "index_kw"),
        ]:
            col = self._find_column(df, key)
            if col:
                local_sep = self._detect_separator(df[col])
                n = len(unique_items_in_list_column(df[col], local_sep))
                rows.append(("Dataset", ind_label, n))

        ref_col = self._find_column(df, "references")
        if ref_col:
            ref_sep = self._detect_separator(df[ref_col])
            total_refs = sum(len(split_list_cell(v, ref_sep))
                             for v in df[ref_col].dropna())
            rows.append(("Dataset", "Total references", total_refs))
            if len(df) > 0:
                rows.append(("Dataset", "References per document",
                             round(total_refs / len(df), 2)))

        # Languages (most frequent language, document count and share)
        if self.desc_language:
            lang_col = self._find_column(df, "language")
            if lang_col:
                langs = df[lang_col].dropna().astype(str).str.strip()
                langs = langs[~langs.str.lower().isin(["nan", "none", ""])]
                if len(langs) > 0:
                    vc = langs.value_counts()
                    rows.append(("Languages", "Number of languages",
                                 int(langs.nunique())))
                    top, cnt = str(vc.index[0]), int(vc.iloc[0])
                    pct = cnt / len(df) * 100 if len(df) else 0
                    rows.append(("Languages", "Most frequent language",
                                 f"{top} ({cnt}, {pct:.1f}%)"))

        # Citation summary
        cite_col = self._find_column(df, "citations")
        if cite_col:
            citations = safe_numeric(df[cite_col]).fillna(0)
            rows.append(("Citations", "Total citations", int(citations.sum())))
            rows.append(("Citations", "Average citations",
                         round(citations.mean(), 2)))
            rows.append(("Citations", "Median citations",
                         int(citations.median())))
            rows.append(("Citations", "Max citations", int(citations.max())))
            cited_docs = int((citations > 0).sum())
            rows.append(("Citations", "Cited documents", cited_docs))
            rows.append(("Citations", "Uncited documents",
                         len(df) - cited_docs))
            if len(df) > 0:
                rows.append(("Citations", "Citation rate (%)",
                             round(cited_docs / len(df) * 100, 1)))

        self._summary_df = pd.DataFrame(
            rows, columns=["Category", "Indicator", "Value"]
        )
        self._update_summary_display()

    def _update_summary_display(self):
        if self._summary_df is None:
            return
        lines: list[str] = ["=" * 60,
                            "BIBLIOMETRIC DATASET SUMMARY",
                            "=" * 60, ""]
        current = None
        for _, row in self._summary_df.iterrows():
            if row["Category"] != current:
                if current is not None:
                    lines.append("")
                current = row["Category"]
                lines.append(f"[{current}]")
                lines.append("-" * 40)
            lines.append(f"  {row['Indicator']}: {fmt_value(row['Value'])}")
        lines.extend(["", "=" * 60])
        self.summary_text.setPlainText("\n".join(lines))

    # =========================================================================
    # PERFORMANCE INDICATORS
    # =========================================================================

    def _extra_index_rows(self):
        """Additional h-index variants from the SCI2S reference, computed on the
        corpus citation list (full mode only)."""
        df = self._df
        cite_col = self._find_column(df, "citations")
        if not cite_col:
            return []
        citations = safe_numeric(df[cite_col]).fillna(0).tolist()
        if not citations:
            return []
        n = len(df)
        year_col = self._find_column(df, "year")
        years = safe_numeric(df[year_col]).dropna().tolist() if year_col else []
        return [
            ("Full (variants)", "M-index (median core)", round(_m_index_median(citations), 3)),
            ("Full (variants)", "E-index", round(_e_index(citations), 3)),
            ("Full (variants)", "Q2-index", round(_q2_index(citations), 3)),
            ("Full (variants)", "V-index (%)", round(_v_index(citations, n), 2)),
            ("Full (variants)", "Pi-index", round(_pi_index(citations, n), 3)),
            ("Full (variants)", "P-index", round(_p_index(citations, n), 3)),
            ("Full (variants)", "Rational H-index", round(_rational_h_index(citations), 3)),
            ("Full (variants)", "M-quotient", round(_m_quotient(citations, years), 4)),
        ]

    def _compute_performance(self):
        df = self._df
        utilsbib = get_biblium_submodule("utilsbib")
        if self.has_biblium and utilsbib is not None:
            try:
                # Advanced (full) indices (g, hg, a, r, ...) are only computed
                # when a 'Cited by' column is present, so alias the citations
                # column to that name when it differs (WoS/OpenAlex/Scopus).
                work = df
                cite_col = self._find_column(df, "citations")
                if (cite_col and cite_col != "Cited by"
                        and "Cited by" not in df.columns):
                    work = df.copy()
                    work["Cited by"] = safe_numeric(df[cite_col])
                indicators = utilsbib.get_performance_indicators(
                    work, mode=self.performance_mode
                )
                rows = [("Performance", name, value)
                        for name, value in indicators]
                if self.performance_mode == "full":
                    rows = rows + self._extra_index_rows()
                self._performance_df = pd.DataFrame(
                    rows, columns=["Category", "Indicator", "Value"]
                )
                self._update_table(self.performance_table,
                                   self._performance_df)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Biblium performance failed: %s — using fallback",
                               exc)
        self._compute_performance_fallback()

    def _compute_performance_fallback(self):
        df = self._df
        cite_col = self._find_column(df, "citations")
        year_col = self._find_column(df, "year")

        rows: list[tuple[str, str, Any]] = []
        citations: list[float] = []
        years = pd.Series(dtype=float)

        rows.append(("Core", "Number of documents", len(df)))

        if cite_col:
            citations = safe_numeric(df[cite_col]).fillna(0).tolist()
            rows.append(("Core", "Total citations", int(sum(citations))))
            rows.append(("Core", "H-index", _h_index(citations)))

        if year_col:
            years = safe_numeric(df[year_col]).dropna()
            if len(years) > 0:
                rows.append(("Core", "Average year", round(years.mean(), 1)))

        if self.performance_mode in ("extended", "full"):
            if cite_col and citations:
                rows.append(("Extended", "G-index", _g_index(citations)))
                for threshold in (1, 5, 10, 25, 50, 100):
                    cnt = sum(1 for c in citations if c >= threshold)
                    if cnt > 0:
                        rows.append(("Extended", f"C{threshold}", cnt))
            if year_col and len(years) > 0:
                rows.append(("Extended", "First year", int(years.min())))
                rows.append(("Extended", "Last year", int(years.max())))
                rows.append(("Extended", "Q1 year",
                             int(years.quantile(0.25))))
                rows.append(("Extended", "Median year",
                             int(years.median())))
                rows.append(("Extended", "Q3 year",
                             int(years.quantile(0.75))))

        if self.performance_mode == "full" and cite_col and citations:
            cited_docs = sum(1 for c in citations if c > 0)
            rows.append(("Full", "Cited documents", cited_docs))
            rows.append(("Full", "A-index", round(_a_index(citations), 2)))
            rows.append(("Full", "R-index", round(_r_index(citations), 2)))
            rows.append(("Full", "Gini index",
                         round(_gini_index(citations), 3)))

        author_col = self._find_column(df,
                                       ["Authors", "Author(s) ID",
                                        "Author full names"])
        if author_col:
            sep = self._detect_separator(df[author_col])
            counts = [len(split_list_cell(v, sep))
                      for v in df[author_col].dropna()]
            if counts:
                rows.append(("Collaboration", "Collaboration index",
                             round(float(np.mean(counts)), 2)))

        if year_col and len(years) > 0:
            year_range = int(years.max()) - int(years.min()) + 1
            if year_range > 0:
                rows.append(("Activity", "Documents per year",
                             round(len(df) / year_range, 2)))
                if cite_col and citations:
                    rows.append(("Activity", "Citations per year",
                                 round(sum(citations) / year_range, 2)))

        if self.performance_mode == "full":
            rows = rows + self._extra_index_rows()
        self._performance_df = pd.DataFrame(
            rows, columns=["Category", "Indicator", "Value"]
        )
        self._update_table(self.performance_table, self._performance_df)

    # =========================================================================
    # TIME SERIES
    # =========================================================================

    def _compute_timeseries(self):
        df = self._df
        year_col = self._find_column(df, "year")
        cite_col = self._find_column(df, "citations")
        if year_col is None:
            return

        years = safe_numeric(df[year_col])
        df_work = df.copy()
        df_work["_year"] = years
        df_work = df_work.dropna(subset=["_year"])
        if len(df_work) == 0:
            return

        production = df_work.groupby("_year").agg(
            n_docs=("_year", "count")
        ).reset_index()
        production.columns = ["Year", "Number of Documents"]
        production = production.sort_values("Year").reset_index(drop=True)

        if cite_col:
            df_work["_citations"] = safe_numeric(df_work[cite_col]).fillna(0)
            cite_by_year = df_work.groupby("_year")["_citations"].sum() \
                                  .reset_index()
            cite_by_year.columns = ["Year", "Total Citations"]
            production = production.merge(cite_by_year, on="Year", how="left")

        production["Percentage Change Documents"] = (
            production["Number of Documents"].pct_change() * 100
        )

        utilsbib = get_biblium_submodule("utilsbib")
        if self.has_biblium and utilsbib is not None:
            try:
                ts_df = utilsbib.summarize_publication_timeseries(
                    production,
                    exclude_last_year_for_growth=self.exclude_last_year,
                )
                self._timeseries_df = ts_df
                self._update_table(self.timeseries_table,
                                   self._timeseries_df)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Biblium time series failed: %s — using fallback",
                               exc)
        self._compute_timeseries_fallback(production)

    def _compute_timeseries_fallback(self, production: pd.DataFrame):
        rows: list[tuple[str, str, Any]] = []
        min_year = int(production["Year"].min())
        max_year = int(production["Year"].max())
        rows.append(("Time Series", "Timespan", f"{min_year} - {max_year}"))
        rows.append(("Time Series", "Number of years", len(production)))

        max_idx = production["Number of Documents"].idxmax()
        max_row = production.loc[max_idx]
        rows.append((
            "Time Series", "Most productive year",
            f"{int(max_row['Year'])} "
            f"({int(max_row['Number of Documents'])} documents)",
        ))

        if "Percentage Change Documents" in production.columns:
            growth_df = (production.iloc[:-1]
                         if self.exclude_last_year and len(production) > 1
                         else production)
            valid_growth = (growth_df["Percentage Change Documents"]
                            .replace([np.inf, -np.inf], np.nan).dropna())
            if len(valid_growth) > 0:
                rates = 1 + valid_growth / 100
                gmean = np.prod(rates) ** (1 / len(rates)) - 1
                rows.append(("Growth", "Average annual growth",
                             f"{gmean * 100:.1f}%"))

                max_idx = valid_growth.idxmax()
                min_idx = valid_growth.idxmin()
                rows.append(("Growth", "Highest growth year",
                             f"{int(production.loc[max_idx, 'Year'])} "
                             f"({valid_growth.loc[max_idx]:.1f}%)"))
                rows.append(("Growth", "Lowest growth year",
                             f"{int(production.loc[min_idx, 'Year'])} "
                             f"({valid_growth.loc[min_idx]:.1f}%)"))

                for n in (3, 5, 10):
                    if len(valid_growth) >= n:
                        recent = valid_growth.tail(n)
                        rates = 1 + recent / 100
                        rgmean = np.prod(rates) ** (1 / len(rates)) - 1
                        rows.append((
                            "Growth", f"Average growth (last {n} years)",
                            f"{rgmean * 100:.1f}%",
                        ))

        if "Total Citations" in production.columns:
            max_cite_idx = production["Total Citations"].idxmax()
            max_cite_row = production.loc[max_cite_idx]
            rows.append((
                "Citations", "Most cited year",
                f"{int(max_cite_row['Year'])} "
                f"({int(max_cite_row['Total Citations']):,} citations)",
            ))
            rows.append(("Citations", "Average citations per year",
                         f"{production['Total Citations'].mean():,.0f}"))

        self._timeseries_df = pd.DataFrame(
            rows, columns=["Category", "Indicator", "Value"]
        )
        self._update_table(self.timeseries_table, self._timeseries_df)

    # =========================================================================
    # DESCRIPTIVES
    # =========================================================================

    def _compute_descriptives(self):
        df = self._df
        desc_cols: list[tuple[str, str]] = []

        def _add(setting: bool, key: str, kind: str):
            if not setting:
                return
            col = self._find_column(df, key)
            if col:
                desc_cols.append((col, kind))

        _add(self.desc_year, "year", "numeric")
        _add(self.desc_source, "source", "string")
        _add(self.desc_doctype, "doctype", "string")
        _add(self.desc_citations, "citations", "numeric")
        _add(self.desc_keywords, "keywords", "list")
        _add(self.desc_keywords, "index_kw", "list")
        _add(self.desc_language, "language", "string")
        _add(self.desc_openaccess, "open_access", "string")

        if not desc_cols:
            return

        # Detect separator from any list column
        sep = "; "
        for col, _ in desc_cols:
            sample = df[col].dropna()
            if len(sample) > 0 and "|" in str(sample.iloc[0]):
                sep = "|"
                break

        utilsbib = get_biblium_submodule("utilsbib")
        if self.has_biblium and utilsbib is not None:
            try:
                desc_df = utilsbib.compute_descriptive_statistics(
                    df, desc_cols,
                    stopwords=None,
                    extra_stats=self.extra_stats,
                    sep=sep,
                )
                self._descriptives_df = desc_df
                self._update_table(self.descriptives_table,
                                   self._descriptives_df)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Biblium descriptives failed: %s — using fallback",
                               exc)
        self._compute_descriptives_fallback(desc_cols, sep)

    def _compute_descriptives_fallback(self,
                                       desc_cols: list[tuple[str, str]],
                                       sep: str):
        df = self._df
        rows: list[tuple[str, str, Any]] = []

        for col, col_type in desc_cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()

            if col_type == "numeric":
                values = safe_numeric(series).dropna()
                if len(values) == 0:
                    continue
                rows.append((col, "Count", len(values)))
                rows.append((col, "Mean", round(values.mean(), 2)))
                rows.append((col, "Median", round(values.median(), 2)))
                rows.append((col, "Std Dev", round(values.std(), 2)))
                rows.append((col, "Min", round(values.min(), 2)))
                rows.append((col, "Max", round(values.max(), 2)))
                rows.append((col, "Sum", round(values.sum(), 2)))
                if self.extra_stats:
                    rows.append((col, "Skewness", round(values.skew(), 3)))
                    rows.append((col, "Kurtosis", round(values.kurtosis(), 3)))
                    for p in (25, 75, 90, 95):
                        rows.append((col, f"P{p}",
                                     round(values.quantile(p / 100), 2)))

            elif col_type == "string":
                n_unique = series.nunique()
                rows.append((col, "Count", len(series)))
                rows.append((col, "Unique", n_unique))
                rows.append((col, "Missing", len(df) - len(series)))
                if n_unique > 0:
                    for i, (val, cnt) in enumerate(
                            series.value_counts().head(3).items(), 1):
                        pct = cnt / len(series) * 100
                        s = str(val)
                        label = (f"{s[:50]}... ({cnt}, {pct:.1f}%)"
                                 if len(s) > 50
                                 else f"{s} ({cnt}, {pct:.1f}%)")
                        rows.append((col, f"Top {i}", label))

            elif col_type == "list":
                all_items: list[str] = []
                for val in series:
                    all_items.extend(split_list_cell(val, sep))
                if all_items:
                    item_counts = Counter(all_items)
                    rows.append((col, "Total occurrences", len(all_items)))
                    rows.append((col, "Unique items", len(item_counts)))
                    rows.append((col, "Items per document",
                                 round(len(all_items) / len(series), 2)))
                    for i, (item, cnt) in enumerate(
                            item_counts.most_common(3), 1):
                        label = (f"{item[:40]}... ({cnt})"
                                 if len(item) > 40 else f"{item} ({cnt})")
                        rows.append((col, f"Top {i}", label))

        self._descriptives_df = pd.DataFrame(
            rows, columns=["Variable", "Indicator", "Value"]
        )
        self._update_table(self.descriptives_table, self._descriptives_df)

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _update_table(table: QTableWidget, df: pd.DataFrame | None):
        if df is None or df.empty:
            table.clear()
            table.setRowCount(0)
            return
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for row_idx in range(len(df)):
            for col_idx in range(len(df.columns)):
                v = df.iloc[row_idx, col_idx]
                table.setItem(row_idx, col_idx, QTableWidgetItem(fmt_value(v)))
        table.resizeColumnsToContents()

    def _send_outputs(self):
        self.Outputs.summary.send(
            self._df_to_table(self._summary_df, prefer_strings=True)
        )
        self.Outputs.performance.send(
            self._df_to_table(self._performance_df, prefer_strings=True)
        )
        self.Outputs.timeseries.send(
            self._df_to_table(self._timeseries_df, prefer_strings=True)
        )
        self.Outputs.descriptives.send(
            self._df_to_table(self._descriptives_df, prefer_strings=True)
        )

        all_dfs: list[pd.DataFrame] = []
        for d in (self._summary_df, self._performance_df,
                  self._timeseries_df, self._descriptives_df):
            if d is None:
                continue
            d2 = d.copy()
            if "Variable" in d2.columns:
                d2 = d2.rename(columns={"Variable": "Category"})
            all_dfs.append(d2)
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            self.Outputs.all_stats.send(
                self._df_to_table(combined, prefer_strings=True)
            )
        else:
            self.Outputs.all_stats.send(None)

    def send_report(self):
        self.report_items([
            ("Compute summary", self.compute_summary),
            ("Compute performance", self.compute_performance),
            ("Performance mode", self.performance_mode),
            ("Compute time series", self.compute_timeseries),
            ("Compute descriptives", self.compute_descriptives),
        ])
        if self._df is not None:
            self.report_items([("Documents", len(self._df))])


if __name__ == "__main__":
    WidgetPreview(OWMainInfo).run()
