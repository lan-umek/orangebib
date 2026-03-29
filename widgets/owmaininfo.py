# -*- coding: utf-8 -*-
"""
Main Information Widget
=======================
Orange widget for computing main bibliometric information and statistics.

Uses biblium's implementation for:
- Performance indicators (H-index, G-index, etc.)
- Time series analysis (growth rates, trends)
- Descriptive statistics for various entity types

Provides comprehensive dataset overview with user-selectable options.
"""

import os
import logging
from typing import Optional, Dict, List, Any, Tuple
from collections import Counter

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QTabWidget, QTextEdit,
    QScrollArea, QFrame, QSplitter,
)
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QFont

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

# Try to import biblium
try:
    import biblium
    from biblium import utilsbib
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    utilsbib = None

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS (fallback if biblium not available)
# =============================================================================

def h_index(citations):
    """Compute H-index from list of citations."""
    if not citations:
        return 0
    sorted_cites = sorted([int(c) for c in citations if pd.notna(c) and c > 0], reverse=True)
    h = 0
    for i, c in enumerate(sorted_cites, 1):
        if c >= i:
            h = i
        else:
            break
    return h


def g_index(citations):
    """Compute G-index from list of citations."""
    if not citations:
        return 0
    sorted_cites = sorted([int(c) for c in citations if pd.notna(c)], reverse=True)
    cumsum = 0
    g = 0
    for i, c in enumerate(sorted_cites, 1):
        cumsum += c
        if cumsum >= i * i:
            g = i
    return g


def a_index(citations):
    """Compute A-index (average citations of h-core papers)."""
    h = h_index(citations)
    if h == 0:
        return 0
    sorted_cites = sorted([int(c) for c in citations if pd.notna(c)], reverse=True)
    return np.mean(sorted_cites[:h]) if sorted_cites else 0


def r_index(citations):
    """Compute R-index (square root of sum of h-core citations)."""
    h = h_index(citations)
    if h == 0:
        return 0
    sorted_cites = sorted([int(c) for c in citations if pd.notna(c)], reverse=True)
    return np.sqrt(sum(sorted_cites[:h])) if sorted_cites else 0


def gini_index(citations):
    """Compute Gini index for citation inequality."""
    valid = [c for c in citations if pd.notna(c) and c >= 0]
    if not valid or len(valid) < 2:
        return 0
    sorted_vals = sorted(valid)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    if cumsum[-1] == 0:
        return 0
    return (2 * sum((i + 1) * v for i, v in enumerate(sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWMainInfo(OWWidget):
    """Compute and display main bibliometric information."""
    
    name = "Main Information"
    description = "Compute comprehensive bibliometric statistics and dataset overview"
    icon = "icons/main_info.svg"
    priority = 5
    keywords = ["main", "info", "statistics", "summary", "overview", "h-index", "bibliometric"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        summary = Output("Summary", Table, doc="Dataset summary table")
        performance = Output("Performance", Table, doc="Performance indicators table")
        timeseries = Output("Time Series", Table, doc="Time series analysis table")
        descriptives = Output("Descriptives", Table, doc="Descriptive statistics table")
        all_stats = Output("All Statistics", Table, doc="Combined statistics table")
    
    # Settings - What to compute
    compute_summary = settings.Setting(True)
    compute_performance = settings.Setting(True)
    compute_timeseries = settings.Setting(True)
    compute_descriptives = settings.Setting(True)
    
    # Settings - Performance options
    performance_mode = settings.Setting("extended")  # core, extended, full
    
    # Settings - Time series options
    exclude_last_year = settings.Setting(True)
    
    # Settings - Descriptives options
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
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        compute_error = Msg("Computation error: {}")
    
    class Warning(OWWidget.Warning):
        no_biblium = Msg("Biblium not installed - using basic implementation")
        missing_columns = Msg("Some columns not found: {}")
    
    class Information(OWWidget.Information):
        computed = Msg("Computed statistics for {:,} documents")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._summary_df: Optional[pd.DataFrame] = None
        self._performance_df: Optional[pd.DataFrame] = None
        self._timeseries_df: Optional[pd.DataFrame] = None
        self._descriptives_df: Optional[pd.DataFrame] = None
        
        if not HAS_BIBLIUM:
            self.Warning.no_biblium()
        
        self._setup_control_area()
        self._setup_main_area()
    
    # =========================================================================
    # GUI SETUP
    # =========================================================================
    
    def _setup_control_area(self):
        """Build control area."""
        # Statistics Selection
        stats_box = gui.widgetBox(self.controlArea, "Statistics to Compute")
        
        gui.checkBox(stats_box, self, "compute_summary", "Dataset Summary",
                     tooltip="Basic counts: documents, sources, authors, etc.",
                     callback=self._on_option_changed)
        gui.checkBox(stats_box, self, "compute_performance", "Performance Indicators",
                     tooltip="H-index, G-index, citations statistics",
                     callback=self._on_option_changed)
        gui.checkBox(stats_box, self, "compute_timeseries", "Time Series Analysis",
                     tooltip="Publication trends, growth rates",
                     callback=self._on_option_changed)
        gui.checkBox(stats_box, self, "compute_descriptives", "Descriptive Statistics",
                     tooltip="Detailed statistics per column type",
                     callback=self._on_option_changed)
        
        # Performance Options
        perf_box = gui.widgetBox(self.controlArea, "Performance Options")
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Detail level:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Core (H-index, citations)", 
            "Extended (+G-index, quartiles)",
            "Full (+A,R,W indices, Gini)"
        ])
        mode_map = {"core": 0, "extended": 1, "full": 2}
        self.mode_combo.setCurrentIndex(mode_map.get(self.performance_mode, 1))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        perf_box.layout().addLayout(mode_layout)
        
        # Time Series Options
        ts_box = gui.widgetBox(self.controlArea, "Time Series Options")
        gui.checkBox(ts_box, self, "exclude_last_year", 
                     "Exclude last year for growth rates",
                     tooltip="Last year may be incomplete",
                     callback=self._on_option_changed)
        
        # Descriptives Options
        desc_box = gui.widgetBox(self.controlArea, "Descriptive Statistics")
        
        desc_grid = QGridLayout()
        desc_grid.setSpacing(2)
        
        row = 0
        self._desc_cb_year = QCheckBox("Publication years")
        self._desc_cb_year.setChecked(self.desc_year)
        self._desc_cb_year.toggled.connect(lambda c: setattr(self, 'desc_year', c) or self._on_option_changed())
        desc_grid.addWidget(self._desc_cb_year, row, 0)
        
        self._desc_cb_source = QCheckBox("Sources/Journals")
        self._desc_cb_source.setChecked(self.desc_source)
        self._desc_cb_source.toggled.connect(lambda c: setattr(self, 'desc_source', c) or self._on_option_changed())
        desc_grid.addWidget(self._desc_cb_source, row, 1)
        
        row += 1
        self._desc_cb_doctype = QCheckBox("Document types")
        self._desc_cb_doctype.setChecked(self.desc_doctype)
        self._desc_cb_doctype.toggled.connect(lambda c: setattr(self, 'desc_doctype', c) or self._on_option_changed())
        desc_grid.addWidget(self._desc_cb_doctype, row, 0)
        
        self._desc_cb_citations = QCheckBox("Citations")
        self._desc_cb_citations.setChecked(self.desc_citations)
        self._desc_cb_citations.toggled.connect(lambda c: setattr(self, 'desc_citations', c) or self._on_option_changed())
        desc_grid.addWidget(self._desc_cb_citations, row, 1)
        
        row += 1
        self._desc_cb_keywords = QCheckBox("Keywords")
        self._desc_cb_keywords.setChecked(self.desc_keywords)
        self._desc_cb_keywords.toggled.connect(lambda c: setattr(self, 'desc_keywords', c) or self._on_option_changed())
        desc_grid.addWidget(self._desc_cb_keywords, row, 0)
        
        self._desc_cb_language = QCheckBox("Language")
        self._desc_cb_language.setChecked(self.desc_language)
        self._desc_cb_language.toggled.connect(lambda c: setattr(self, 'desc_language', c) or self._on_option_changed())
        desc_grid.addWidget(self._desc_cb_language, row, 1)
        
        row += 1
        self._desc_cb_openaccess = QCheckBox("Open Access")
        self._desc_cb_openaccess.setChecked(self.desc_openaccess)
        self._desc_cb_openaccess.toggled.connect(lambda c: setattr(self, 'desc_openaccess', c) or self._on_option_changed())
        desc_grid.addWidget(self._desc_cb_openaccess, row, 0)
        
        self._desc_cb_extra = QCheckBox("Extra statistics")
        self._desc_cb_extra.setChecked(self.extra_stats)
        self._desc_cb_extra.toggled.connect(lambda c: setattr(self, 'extra_stats', c) or self._on_option_changed())
        self._desc_cb_extra.setToolTip("Additional metrics like entropy, concentration")
        desc_grid.addWidget(self._desc_cb_extra, row, 1)
        
        desc_box.layout().addLayout(desc_grid)
        
        # Apply button
        self.apply_btn = gui.button(
            self.controlArea, self, "Compute Statistics",
            callback=self.commit, autoDefault=False
        )
        self.apply_btn.setMinimumHeight(35)
        
        gui.checkBox(self.controlArea, self, "auto_apply", "Apply Automatically")
        
        self.controlArea.layout().addStretch(1)
    
    def _setup_main_area(self):
        """Build main area with tabbed results."""
        # Create tab widget
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
        """Handle option changes."""
        if self.auto_apply:
            self.commit()
    
    def _on_mode_changed(self, index):
        """Handle performance mode change."""
        modes = ["core", "extended", "full"]
        self.performance_mode = modes[index]
        if self.auto_apply:
            self.commit()
    
    # =========================================================================
    # DATA HANDLING
    # =========================================================================
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive input data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._clear_results()
        
        if data is None:
            self.Error.no_data()
            return
        
        # Convert to DataFrame
        self._df = self._table_to_df(data)
        
        if self.auto_apply:
            self.commit()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        """Convert Orange Table to pandas DataFrame."""
        data = {}
        
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        
        return pd.DataFrame(data)
    
    def _clear_results(self):
        """Clear all results."""
        self._summary_df = None
        self._performance_df = None
        self._timeseries_df = None
        self._descriptives_df = None
        
        self.summary_text.clear()
        self.performance_table.clear()
        self.performance_table.setRowCount(0)
        self.timeseries_table.clear()
        self.timeseries_table.setRowCount(0)
        self.descriptives_table.clear()
        self.descriptives_table.setRowCount(0)
    
    def commit(self):
        """Compute statistics."""
        self._compute_all()
    
    # =========================================================================
    # COMPUTATION
    # =========================================================================
    
    def _compute_all(self):
        """Compute all selected statistics."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
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
            
        except Exception as e:
            import traceback
            logger.error(f"Computation error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
    
    def _find_column(self, candidates: List[str]) -> Optional[str]:
        """Find first available column from candidates."""
        for col in candidates:
            if col in self._df.columns:
                return col
        return None
    
    def _detect_separator(self, series) -> str:
        """Detect separator used in a column."""
        sample = series.dropna()
        if len(sample) == 0:
            return "; "
        sample_str = str(sample.iloc[0])
        if "|" in sample_str:
            return "|"
        return "; "
    
    # =========================================================================
    # SUMMARY COMPUTATION
    # =========================================================================
    
    def _compute_summary(self):
        """Compute dataset summary."""
        df = self._df
        rows = []
        sep = "; "
        
        # Auto-detect separator
        author_col = self._find_column(["Authors", "Author", "AU", "Author full names"])
        if author_col:
            sep = self._detect_separator(df[author_col])
        
        # Basic counts
        rows.append(("Dataset", "Number of documents", len(df)))
        
        # Timespan
        year_col = self._find_column(["Year", "Publication Year", "publication_year", "PY"])
        if year_col:
            years = pd.to_numeric(df[year_col], errors='coerce').dropna()
            if len(years) > 0:
                min_year = int(years.min())
                max_year = int(years.max())
                rows.append(("Dataset", "Timespan", f"{min_year} - {max_year}"))
                rows.append(("Dataset", "Number of years", max_year - min_year + 1))
        
        # Sources
        source_col = self._find_column(["Source title", "Source", "Journal", "SO"])
        if source_col:
            n_sources = df[source_col].dropna().nunique()
            rows.append(("Dataset", "Number of sources", n_sources))
        
        # Authors
        if author_col:
            all_authors = set()
            for val in df[author_col].dropna():
                authors = [a.strip() for a in str(val).split(sep) if a.strip()]
                all_authors.update(authors)
            rows.append(("Dataset", "Number of authors", len(all_authors)))
            if len(df) > 0:
                rows.append(("Dataset", "Authors per document", round(len(all_authors) / len(df), 2)))
        
        # Countries
        country_col = self._find_column(["Countries of Authors", "Countries", "Country", "authorships.countries"])
        if country_col:
            country_sep = self._detect_separator(df[country_col])
            all_countries = set()
            for val in df[country_col].dropna():
                countries = [c.strip() for c in str(val).split(country_sep) if c.strip()]
                all_countries.update(countries)
            rows.append(("Dataset", "Number of countries", len(all_countries)))
        
        # Affiliations
        aff_col = self._find_column(["Affiliations", "Affiliation", "C1"])
        if aff_col:
            aff_sep = self._detect_separator(df[aff_col])
            all_affs = set()
            for val in df[aff_col].dropna():
                affs = [a.strip() for a in str(val).split(aff_sep) if a.strip()]
                all_affs.update(affs)
            rows.append(("Dataset", "Number of affiliations", len(all_affs)))
        
        # Keywords
        kw_col = self._find_column(["Author Keywords", "Keywords", "DE"])
        if kw_col:
            kw_sep = self._detect_separator(df[kw_col])
            all_kw = set()
            for val in df[kw_col].dropna():
                kws = [k.strip() for k in str(val).split(kw_sep) if k.strip()]
                all_kw.update(kws)
            rows.append(("Dataset", "Number of author keywords", len(all_kw)))
        
        # Index keywords
        ik_col = self._find_column(["Index Keywords", "Index keywords", "ID"])
        if ik_col:
            ik_sep = self._detect_separator(df[ik_col])
            all_ik = set()
            for val in df[ik_col].dropna():
                iks = [k.strip() for k in str(val).split(ik_sep) if k.strip()]
                all_ik.update(iks)
            rows.append(("Dataset", "Number of index keywords", len(all_ik)))
        
        # References
        ref_col = self._find_column(["References", "Cited References", "CR"])
        if ref_col:
            ref_sep = self._detect_separator(df[ref_col])
            total_refs = 0
            for val in df[ref_col].dropna():
                refs = [r.strip() for r in str(val).split(ref_sep) if r.strip()]
                total_refs += len(refs)
            rows.append(("Dataset", "Total references", total_refs))
            if len(df) > 0:
                rows.append(("Dataset", "References per document", round(total_refs / len(df), 2)))
        
        # Citations summary
        cite_col = self._find_column(["Cited by", "Times Cited", "Citation Count", "cited_by_count", "TC"])
        if cite_col:
            citations = pd.to_numeric(df[cite_col], errors='coerce').fillna(0)
            rows.append(("Citations", "Total citations", int(citations.sum())))
            rows.append(("Citations", "Average citations", round(citations.mean(), 2)))
            rows.append(("Citations", "Median citations", int(citations.median())))
            rows.append(("Citations", "Max citations", int(citations.max())))
            cited_docs = (citations > 0).sum()
            rows.append(("Citations", "Cited documents", cited_docs))
            rows.append(("Citations", "Uncited documents", len(df) - cited_docs))
            if len(df) > 0:
                rows.append(("Citations", "Citation rate (%)", round(cited_docs / len(df) * 100, 1)))
        
        # Create DataFrame
        self._summary_df = pd.DataFrame(rows, columns=["Category", "Indicator", "Value"])
        
        # Update display
        self._update_summary_display()
    
    def _update_summary_display(self):
        """Update summary text display."""
        if self._summary_df is None:
            return
        
        lines = []
        lines.append("=" * 60)
        lines.append("BIBLIOMETRIC DATASET SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        
        current_category = None
        for _, row in self._summary_df.iterrows():
            if row["Category"] != current_category:
                if current_category is not None:
                    lines.append("")
                current_category = row["Category"]
                lines.append(f"[{current_category}]")
                lines.append("-" * 40)
            
            value = row["Value"]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float):
                    value = f"{value:,.2f}" if value < 1000 else f"{value:,.0f}"
                else:
                    value = f"{value:,}"
            lines.append(f"  {row['Indicator']}: {value}")
        
        lines.append("")
        lines.append("=" * 60)
        
        self.summary_text.setPlainText("\n".join(lines))
    
    # =========================================================================
    # PERFORMANCE COMPUTATION (uses biblium)
    # =========================================================================
    
    def _compute_performance(self):
        """Compute performance indicators using biblium if available."""
        df = self._df
        
        # Try biblium first
        if HAS_BIBLIUM and utilsbib:
            try:
                indicators = utilsbib.get_performance_indicators(
                    df, mode=self.performance_mode
                )
                rows = [("Performance", name, value) for name, value in indicators]
                self._performance_df = pd.DataFrame(rows, columns=["Category", "Indicator", "Value"])
                self._update_table(self.performance_table, self._performance_df)
                return
            except Exception as e:
                logger.warning(f"Biblium performance failed: {e}, using fallback")
        
        # Fallback implementation
        self._compute_performance_fallback()
    
    def _compute_performance_fallback(self):
        """Fallback performance computation without biblium."""
        df = self._df
        cite_col = self._find_column(["Cited by", "Times Cited", "Citation Count", "cited_by_count", "TC"])
        year_col = self._find_column(["Year", "Publication Year", "publication_year", "PY"])
        
        rows = []
        citations = []
        years = pd.Series(dtype=float)
        
        # Core indicators
        rows.append(("Core", "Number of documents", len(df)))
        
        if cite_col:
            citations = pd.to_numeric(df[cite_col], errors='coerce').fillna(0).tolist()
            rows.append(("Core", "Total citations", int(sum(citations))))
            rows.append(("Core", "H-index", h_index(citations)))
        
        if year_col:
            years = pd.to_numeric(df[year_col], errors='coerce').dropna()
            if len(years) > 0:
                rows.append(("Core", "Average year", round(years.mean(), 1)))
        
        # Extended indicators
        if self.performance_mode in ["extended", "full"]:
            if cite_col and citations:
                rows.append(("Extended", "G-index", g_index(citations)))
                
                # C-index (documents with >= N citations)
                for threshold in [1, 5, 10, 25, 50, 100]:
                    count = sum(1 for c in citations if c >= threshold)
                    if count > 0:
                        rows.append(("Extended", f"C{threshold}", count))
            
            if year_col and len(years) > 0:
                rows.append(("Extended", "First year", int(years.min())))
                rows.append(("Extended", "Last year", int(years.max())))
                rows.append(("Extended", "Q1 year", int(years.quantile(0.25))))
                rows.append(("Extended", "Median year", int(years.median())))
                rows.append(("Extended", "Q3 year", int(years.quantile(0.75))))
        
        # Full indicators
        if self.performance_mode == "full" and cite_col and citations:
            cited_docs = sum(1 for c in citations if c > 0)
            rows.append(("Full", "Cited documents", cited_docs))
            rows.append(("Full", "A-index", round(a_index(citations), 2)))
            rows.append(("Full", "R-index", round(r_index(citations), 2)))
            rows.append(("Full", "Gini index", round(gini_index(citations), 3)))
        
        # Collaboration index
        author_col = self._find_column(["Authors", "Author(s) ID", "Author full names"])
        if author_col:
            sep = self._detect_separator(df[author_col])
            author_counts = []
            for val in df[author_col].dropna():
                authors = [a.strip() for a in str(val).split(sep) if a.strip()]
                author_counts.append(len(authors))
            if author_counts:
                collab_idx = np.mean(author_counts)
                rows.append(("Collaboration", "Collaboration index", round(collab_idx, 2)))
        
        # Activity metrics
        if year_col and len(years) > 0:
            year_range = int(years.max()) - int(years.min()) + 1
            if year_range > 0:
                rows.append(("Activity", "Documents per year", round(len(df) / year_range, 2)))
                if cite_col and citations:
                    rows.append(("Activity", "Citations per year", round(sum(citations) / year_range, 2)))
        
        self._performance_df = pd.DataFrame(rows, columns=["Category", "Indicator", "Value"])
        self._update_table(self.performance_table, self._performance_df)
    
    # =========================================================================
    # TIME SERIES COMPUTATION (uses biblium)
    # =========================================================================
    
    def _compute_timeseries(self):
        """Compute time series analysis using biblium if available."""
        df = self._df
        year_col = self._find_column(["Year", "Publication Year", "publication_year", "PY"])
        cite_col = self._find_column(["Cited by", "Times Cited", "Citation Count", "cited_by_count", "TC"])
        
        if year_col is None:
            return
        
        # First, compute production_df (needed by biblium)
        years = pd.to_numeric(df[year_col], errors='coerce')
        df_work = df.copy()
        df_work["_year"] = years
        df_work = df_work.dropna(subset=["_year"])
        
        if len(df_work) == 0:
            return
        
        # Compute production by year
        production = df_work.groupby("_year").agg(
            n_docs=("_year", "count")
        ).reset_index()
        production.columns = ["Year", "Number of Documents"]
        production = production.sort_values("Year").reset_index(drop=True)
        
        # Add citations if available
        if cite_col:
            df_work["_citations"] = pd.to_numeric(df_work[cite_col], errors='coerce').fillna(0)
            cite_by_year = df_work.groupby("_year")["_citations"].sum().reset_index()
            cite_by_year.columns = ["Year", "Total Citations"]
            production = production.merge(cite_by_year, on="Year", how="left")
        
        # Calculate percentage change
        production["Percentage Change Documents"] = production["Number of Documents"].pct_change() * 100
        
        # Try biblium summarize_publication_timeseries
        if HAS_BIBLIUM and utilsbib:
            try:
                ts_df = utilsbib.summarize_publication_timeseries(
                    production, 
                    exclude_last_year_for_growth=self.exclude_last_year
                )
                self._timeseries_df = ts_df
                self._update_table(self.timeseries_table, self._timeseries_df)
                return
            except Exception as e:
                logger.warning(f"Biblium time series failed: {e}, using fallback")
        
        # Fallback implementation
        self._compute_timeseries_fallback(production)
    
    def _compute_timeseries_fallback(self, production):
        """Fallback time series computation."""
        rows = []
        
        # Timespan
        min_year = int(production["Year"].min())
        max_year = int(production["Year"].max())
        rows.append(("Time Series", "Timespan", f"{min_year} - {max_year}"))
        rows.append(("Time Series", "Number of years", len(production)))
        
        # Most productive year
        max_idx = production["Number of Documents"].idxmax()
        max_row = production.loc[max_idx]
        rows.append(("Time Series", "Most productive year", 
                     f"{int(max_row['Year'])} ({int(max_row['Number of Documents'])} documents)"))
        
        # Growth statistics
        if "Percentage Change Documents" in production.columns:
            # Optionally exclude last year
            if self.exclude_last_year and len(production) > 1:
                growth_df = production.iloc[:-1]
            else:
                growth_df = production
            
            valid_growth = growth_df["Percentage Change Documents"].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(valid_growth) > 0:
                # Average growth (geometric mean)
                rates = 1 + valid_growth / 100
                gmean = np.prod(rates) ** (1 / len(rates)) - 1
                rows.append(("Growth", "Average annual growth", f"{gmean * 100:.1f}%"))
                
                # Highest/lowest growth years
                max_growth_idx = valid_growth.idxmax()
                min_growth_idx = valid_growth.idxmin()
                max_growth_row = production.loc[max_growth_idx]
                min_growth_row = production.loc[min_growth_idx]
                rows.append(("Growth", "Highest growth year", 
                             f"{int(max_growth_row['Year'])} ({valid_growth.loc[max_growth_idx]:.1f}%)"))
                rows.append(("Growth", "Lowest growth year", 
                             f"{int(min_growth_row['Year'])} ({valid_growth.loc[min_growth_idx]:.1f}%)"))
                
                # Recent growth windows
                for n in [3, 5, 10]:
                    if len(valid_growth) >= n:
                        recent = valid_growth.tail(n)
                        rates = 1 + recent / 100
                        recent_gmean = np.prod(rates) ** (1 / len(rates)) - 1
                        rows.append(("Growth", f"Average growth (last {n} years)", f"{recent_gmean * 100:.1f}%"))
        
        # Citation trends
        if "Total Citations" in production.columns:
            max_cite_idx = production["Total Citations"].idxmax()
            max_cite_row = production.loc[max_cite_idx]
            rows.append(("Citations", "Most cited year", 
                         f"{int(max_cite_row['Year'])} ({int(max_cite_row['Total Citations']):,} citations)"))
            
            avg_citations = production["Total Citations"].mean()
            rows.append(("Citations", "Average citations per year", f"{avg_citations:,.0f}"))
        
        self._timeseries_df = pd.DataFrame(rows, columns=["Category", "Indicator", "Value"])
        self._update_table(self.timeseries_table, self._timeseries_df)
    
    # =========================================================================
    # DESCRIPTIVES COMPUTATION (uses biblium)
    # =========================================================================
    
    def _compute_descriptives(self):
        """Compute descriptive statistics using biblium if available."""
        df = self._df
        
        # Build list of columns to analyze based on user selection
        desc_cols = []
        
        if self.desc_year:
            year_col = self._find_column(["Year", "Publication Year", "publication_year", "PY"])
            if year_col:
                desc_cols.append((year_col, "numeric"))
        
        if self.desc_source:
            source_col = self._find_column(["Source title", "Source", "Journal", "SO"])
            if source_col:
                desc_cols.append((source_col, "string"))
        
        if self.desc_doctype:
            dt_col = self._find_column(["Document Type", "Document type", "type", "DT"])
            if dt_col:
                desc_cols.append((dt_col, "string"))
        
        if self.desc_citations:
            cite_col = self._find_column(["Cited by", "Times Cited", "Citation Count", "cited_by_count", "TC"])
            if cite_col:
                desc_cols.append((cite_col, "numeric"))
        
        if self.desc_keywords:
            kw_col = self._find_column(["Author Keywords", "Keywords", "DE"])
            if kw_col:
                desc_cols.append((kw_col, "list"))
            
            ik_col = self._find_column(["Index Keywords", "Index keywords", "ID"])
            if ik_col:
                desc_cols.append((ik_col, "list"))
        
        if self.desc_language:
            lang_col = self._find_column(["Language of Original Document", "Language", "LA"])
            if lang_col:
                desc_cols.append((lang_col, "string"))
        
        if self.desc_openaccess:
            oa_col = self._find_column(["Open Access", "open_access", "OA"])
            if oa_col:
                desc_cols.append((oa_col, "string"))
        
        if not desc_cols:
            return
        
        # Detect separator
        sep = "; "
        for col, _ in desc_cols:
            sample = df[col].dropna()
            if len(sample) > 0 and "|" in str(sample.iloc[0]):
                sep = "|"
                break
        
        # Try biblium compute_descriptive_statistics
        if HAS_BIBLIUM and utilsbib:
            try:
                desc_df = utilsbib.compute_descriptive_statistics(
                    df, desc_cols, 
                    stopwords=None, 
                    extra_stats=self.extra_stats,
                    sep=sep
                )
                self._descriptives_df = desc_df
                self._update_table(self.descriptives_table, self._descriptives_df)
                return
            except Exception as e:
                logger.warning(f"Biblium descriptives failed: {e}, using fallback")
        
        # Fallback implementation
        self._compute_descriptives_fallback(desc_cols, sep)
    
    def _compute_descriptives_fallback(self, desc_cols, sep):
        """Fallback descriptive statistics computation."""
        df = self._df
        rows = []
        
        for col, col_type in desc_cols:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            
            if col_type == "numeric":
                values = pd.to_numeric(series, errors='coerce').dropna()
                if len(values) > 0:
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
                        for p in [25, 75, 90, 95]:
                            rows.append((col, f"P{p}", round(values.quantile(p/100), 2)))
            
            elif col_type == "string":
                n_unique = series.nunique()
                rows.append((col, "Count", len(series)))
                rows.append((col, "Unique", n_unique))
                rows.append((col, "Missing", len(df) - len(series)))
                
                if n_unique > 0:
                    top = series.value_counts().head(3)
                    for i, (val, cnt) in enumerate(top.items(), 1):
                        pct = cnt / len(series) * 100
                        rows.append((col, f"Top {i}", f"{val[:50]}... ({cnt}, {pct:.1f}%)" if len(str(val)) > 50 else f"{val} ({cnt}, {pct:.1f}%)"))
            
            elif col_type == "list":
                all_items = []
                for val in series:
                    items = [i.strip() for i in str(val).split(sep) if i.strip()]
                    all_items.extend(items)
                
                if all_items:
                    item_counts = Counter(all_items)
                    rows.append((col, "Total occurrences", len(all_items)))
                    rows.append((col, "Unique items", len(item_counts)))
                    rows.append((col, "Items per document", round(len(all_items) / len(series), 2)))
                    
                    # Top items
                    for i, (item, cnt) in enumerate(item_counts.most_common(3), 1):
                        rows.append((col, f"Top {i}", f"{item[:40]}... ({cnt})" if len(item) > 40 else f"{item} ({cnt})"))
        
        self._descriptives_df = pd.DataFrame(rows, columns=["Variable", "Indicator", "Value"])
        self._update_table(self.descriptives_table, self._descriptives_df)
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _update_table(self, table: QTableWidget, df: pd.DataFrame):
        """Update a table widget with DataFrame data."""
        if df is None or df.empty:
            table.clear()
            table.setRowCount(0)
            return
        
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(list(df.columns))
        
        for row_idx in range(len(df)):
            for col_idx, col in enumerate(df.columns):
                value = df.iloc[row_idx, col_idx]
                if pd.isna(value):
                    text = ""
                elif isinstance(value, float):
                    text = f"{value:.2f}" if abs(value) < 1000 else f"{value:,.0f}"
                else:
                    text = str(value)
                
                item = QTableWidgetItem(text)
                table.setItem(row_idx, col_idx, item)
        
        table.resizeColumnsToContents()
    
    def _df_to_table(self, df: pd.DataFrame) -> Optional[Table]:
        """Convert DataFrame to Orange Table."""
        if df is None or df.empty:
            return None
        
        # All columns as metas (string/mixed data)
        metas = [StringVariable(str(col)) for col in df.columns]
        domain = Domain([], metas=metas)
        
        meta_data = df.astype(str).values
        
        return Table.from_numpy(domain, np.empty((len(df), 0)), metas=meta_data)
    
    def _send_outputs(self):
        """Send outputs."""
        self.Outputs.summary.send(self._df_to_table(self._summary_df))
        self.Outputs.performance.send(self._df_to_table(self._performance_df))
        self.Outputs.timeseries.send(self._df_to_table(self._timeseries_df))
        self.Outputs.descriptives.send(self._df_to_table(self._descriptives_df))
        
        # Combine all stats
        all_dfs = []
        if self._summary_df is not None:
            all_dfs.append(self._summary_df)
        if self._performance_df is not None:
            all_dfs.append(self._performance_df)
        if self._timeseries_df is not None:
            # Ensure consistent columns
            ts_df = self._timeseries_df.copy()
            if "Variable" in ts_df.columns:
                ts_df = ts_df.rename(columns={"Variable": "Category"})
            all_dfs.append(ts_df)
        if self._descriptives_df is not None:
            desc_df = self._descriptives_df.copy()
            if "Variable" in desc_df.columns:
                desc_df = desc_df.rename(columns={"Variable": "Category"})
            all_dfs.append(desc_df)
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            self.Outputs.all_stats.send(self._df_to_table(combined))
        else:
            self.Outputs.all_stats.send(None)
    
    def send_report(self):
        """Generate widget report."""
        self.report_items([
            ("Compute summary", self.compute_summary),
            ("Compute performance", self.compute_performance),
            ("Performance mode", self.performance_mode),
            ("Compute time series", self.compute_timeseries),
            ("Compute descriptives", self.compute_descriptives),
        ])
        
        if self._df is not None:
            self.report_items([
                ("Documents", len(self._df)),
            ])


if __name__ == "__main__":
    WidgetPreview(OWMainInfo).run()
