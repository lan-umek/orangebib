# -*- coding: utf-8 -*-
"""
Top Cited Widget
================
Orange widget for finding top-cited documents in bibliographic data.

Supports:
- Global Citations: Most cited in external databases
- Local Citations: Most cited within the dataset
- Citations per Year: Normalized by publication age
- All: Combined view
"""

import os
import logging
from typing import Optional, List, Any
from collections import Counter

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy,
)
from AnyQt.QtCore import Qt

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
# CONSTANTS
# =============================================================================

ANALYSIS_TYPES = {
    "Global Citations": "global",
    "Local Citations": "local",
    "Citations per Year": "per_year",
}

# Default columns to display
DEFAULT_COLUMNS = ["Authors", "Title", "Source title", "Year", "Document Type"]


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWTopCited(OWWidget):
    """Find top-cited documents."""
    
    name = "Top Cited"
    description = "Global and local top-cited documents"
    icon = "icons/top_cited.svg"
    priority = 40
    keywords = ["citations", "top", "cited", "ranking", "impact"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        top_cited = Output("Top Cited", Table, doc="Top cited documents")
        selected = Output("Selected", Table, doc="Selected documents from preview")
    
    # Settings
    analysis_type = settings.Setting("Global Citations")
    top_n = settings.Setting(10)
    include_ties = settings.Setting(False)
    
    auto_apply = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_citations = Msg("Citations column not found")
        compute_error = Msg("Error: {}")
    
    class Warning(OWWidget.Warning):
        no_references = Msg("References column not found - local citations unavailable")
        empty_result = Msg("No documents found")
    
    class Information(OWWidget.Information):
        found = Msg("Found {:,} top-cited documents")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._result_df: Optional[pd.DataFrame] = None
        self._result_indices: Optional[List[int]] = None  # Map result rows to original data
        
        self._setup_control_area()
        self._setup_main_area()
    
    # =========================================================================
    # GUI SETUP
    # =========================================================================
    
    def _setup_control_area(self):
        """Build control area."""
        # Analysis Type
        type_box = gui.widgetBox(self.controlArea, "Analysis Type")
        
        type_layout = QGridLayout()
        type_layout.addWidget(QLabel("Type:"), 0, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems(list(ANALYSIS_TYPES.keys()))
        self.type_combo.setCurrentText(self.analysis_type)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo, 0, 1)
        type_box.layout().addLayout(type_layout)
        
        # Options
        options_box = gui.widgetBox(self.controlArea, "Options")
        
        options_layout = QGridLayout()
        
        options_layout.addWidget(QLabel("Top N:"), 0, 0)
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 10000)
        self.top_n_spin.setValue(self.top_n)
        self.top_n_spin.valueChanged.connect(lambda v: setattr(self, 'top_n', v))
        options_layout.addWidget(self.top_n_spin, 0, 1)
        
        options_box.layout().addLayout(options_layout)
        
        self.ties_cb = QCheckBox("Include ties")
        self.ties_cb.setChecked(self.include_ties)
        self.ties_cb.setToolTip("Include documents with same citation count as the Nth document")
        self.ties_cb.toggled.connect(lambda c: setattr(self, 'include_ties', c))
        options_box.layout().addWidget(self.ties_cb)
        
        # Info about analysis types
        info_box = gui.widgetBox(self.controlArea, "Information")
        info_label = QLabel(
            "<small>"
            "<b>Global Citations:</b> External citations (e.g., from Scopus/WoS)<br>"
            "<b>Local Citations:</b> Citations within this dataset<br>"
            "<b>Citations per Year:</b> Normalized by document age"
            "</small>"
        )
        info_label.setWordWrap(True)
        info_box.layout().addWidget(info_label)
        
        # Apply button
        self.apply_btn = QPushButton("Find Top Cited")
        self.apply_btn.setMinimumHeight(35)
        self.apply_btn.clicked.connect(self.commit)
        self.controlArea.layout().addWidget(self.apply_btn)
        
        gui.auto_apply(self.controlArea, self, "auto_apply")
    
    def _setup_main_area(self):
        """Build main area with results preview."""
        # Summary
        summary_box = gui.widgetBox(self.mainArea, "Results Summary")
        self.summary_label = QLabel("No data")
        self.summary_label.setWordWrap(True)
        summary_box.layout().addWidget(self.summary_label)
        
        # Selection info
        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        summary_box.layout().addWidget(self.selection_label)
        
        # Results table
        results_box = gui.widgetBox(self.mainArea, "Top Cited Documents (select rows to output)")
        self.results_table = QTableWidget()
        self.results_table.setMinimumHeight(350)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.itemSelectionChanged.connect(self._on_selection_changed)
        results_box.layout().addWidget(self.results_table)
        
        # Selection buttons
        btn_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.results_table.selectAll)
        btn_row.addWidget(self.select_all_btn)
        
        self.clear_selection_btn = QPushButton("Clear Selection")
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        btn_row.addWidget(self.clear_selection_btn)
        
        btn_row.addStretch()
        results_box.layout().addLayout(btn_row)
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_type_changed(self, analysis_type):
        self.analysis_type = analysis_type
        if self.auto_apply:
            self.commit()
    
    def _clear_selection(self):
        """Clear table selection and reset outputs."""
        self.results_table.clearSelection()
        self.selection_label.setText("")
        self.Outputs.selected.send(None)
    
    def _on_selection_changed(self):
        """Handle selection in results table - output selected documents."""
        selected_rows = set(item.row() for item in self.results_table.selectedItems())
        
        if not selected_rows or self._result_indices is None or self._data is None:
            self.selection_label.setText("")
            self.Outputs.selected.send(None)
            return
        
        # Map selected result rows to original data indices
        original_indices = []
        for row in sorted(selected_rows):
            if row < len(self._result_indices):
                original_indices.append(self._result_indices[row])
        
        if original_indices:
            selected_data = self._data[original_indices]
            n_selected = len(original_indices)
            self.selection_label.setText(f"✓ {n_selected} documents selected")
            self.Outputs.selected.send(selected_data)
        else:
            self.selection_label.setText("")
            self.Outputs.selected.send(None)
    
    def commit(self):
        """Compute top cited documents."""
        self._compute()
    
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
        self._result_df = None
        
        if data is None:
            self.Error.no_data()
            self._update_results_display()
            return
        
        # Convert to DataFrame
        self._df = self._table_to_df(data)
        
        if self.auto_apply:
            self.commit()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        """Convert Orange Table to pandas DataFrame."""
        data = {}
        
        for var in table.domain.attributes:
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else "" for v in col]
            else:
                data[var.name] = col
        
        for var in table.domain.metas:
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else "" for v in col]
            elif isinstance(var, StringVariable):
                data[var.name] = col
            else:
                data[var.name] = col
        
        if table.domain.class_var:
            var = table.domain.class_var
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else "" for v in col]
            else:
                data[var.name] = col
        
        return pd.DataFrame(data)
    
    # =========================================================================
    # COMPUTATION
    # =========================================================================
    
    def _compute(self):
        """Compute top cited documents."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None or self._df.empty:
            self.Error.no_data()
            self._update_results_display()
            self.Outputs.top_cited.send(None)
            return
        
        try:
            mode = ANALYSIS_TYPES.get(self.analysis_type, "global")
            
            # Find citation column
            cite_col = self._find_column(["Cited by", "Times Cited", "Citation Count", 
                                          "cited_by_count", "TC", "Citations"])
            if cite_col is None:
                self.Error.no_citations()
                self._update_results_display()
                self.Outputs.top_cited.send(None)
                return
            
            # Check for references column (needed for local citations)
            ref_col = self._find_column(["References", "Cited References", "CR"])
            if mode == "local" and ref_col is None:
                self.Warning.no_references()
                if mode == "local":
                    mode = "global"  # Fall back to global
            
            # Compute based on mode
            if mode == "local":
                result_df, indices = self._compute_local(cite_col, ref_col)
            elif mode == "per_year":
                result_df, indices = self._compute_per_year(cite_col)
            else:  # global
                result_df, indices = self._compute_global(cite_col)
            
            if result_df is None or result_df.empty:
                self.Warning.empty_result()
                self._result_df = None
                self._result_indices = None
                self._update_results_display()
                self.Outputs.top_cited.send(None)
                self.Outputs.selected.send(None)
                return
            
            self._result_df = result_df
            self._result_indices = indices
            
            # Update display
            self._update_results_display()
            
            # Send output
            output_table = self._df_to_table(result_df)
            self.Outputs.top_cited.send(output_table)
            
            self.Information.found(len(result_df))
            
        except Exception as e:
            import traceback
            logger.error(f"Computation error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
            self._result_df = None
            self._update_results_display()
            self.Outputs.top_cited.send(None)
    
    def _find_column(self, candidates):
        """Find first available column from candidates."""
        for col in candidates:
            if col in self._df.columns:
                return col
        return None
    
    def _get_display_columns(self):
        """Get columns to display in output."""
        cols = []
        for c in DEFAULT_COLUMNS:
            if c in self._df.columns:
                cols.append(c)
        return cols
    
    def _compute_global(self, cite_col):
        """Compute global top cited."""
        if HAS_BIBLIUM and utilsbib:
            try:
                result = utilsbib.select_global_top_cited_documents(
                    self._df,
                    top_n=self.top_n,
                    cite_col=cite_col,
                    include_ties=self.include_ties,
                )
                # Try to get indices from result
                if result is not None and not result.empty:
                    indices = result.index.tolist() if hasattr(result, 'index') else list(range(len(result)))
                    return result.reset_index(drop=True), indices
                return result, []
            except Exception as e:
                logger.warning(f"Biblium global citations failed: {e}")
        return self._compute_global_basic(cite_col)
    
    def _compute_global_basic(self, cite_col):
        """Basic global top cited computation."""
        df = self._df.copy()
        df["_orig_idx"] = range(len(df))  # Track original indices
        df[cite_col] = pd.to_numeric(df[cite_col], errors='coerce').fillna(0)
        
        # Sort by citations (descending), then year (ascending), then title
        sort_cols = [cite_col]
        ascending = [False]
        
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
            sort_cols.append("Year")
            ascending.append(True)
        
        if "Title" in df.columns:
            sort_cols.append("Title")
            ascending.append(True)
        
        df = df.sort_values(sort_cols, ascending=ascending)
        
        # Handle ties
        if self.include_ties and self.top_n < len(df):
            cutoff = df[cite_col].iloc[self.top_n - 1]
            df = df[df[cite_col] >= cutoff]
        else:
            df = df.head(self.top_n)
        
        # Extract original indices
        indices = df["_orig_idx"].tolist()
        
        # Select columns
        display_cols = self._get_display_columns()
        out_cols = list(dict.fromkeys(display_cols + [cite_col]))
        out_cols = [c for c in out_cols if c in df.columns]
        
        result = df[out_cols].reset_index(drop=True)
        result = result.rename(columns={cite_col: "Global Citations"})
        
        return result, indices
    
    def _compute_local(self, cite_col, ref_col):
        """Compute local top cited (citations within dataset)."""
        if HAS_BIBLIUM and utilsbib:
            try:
                result = utilsbib.select_local_top_cited_documents(
                    self._df,
                    top_n=self.top_n,
                    ref_col=ref_col,
                    cite_col=cite_col,
                )
                if result is not None and not result.empty:
                    indices = result.index.tolist() if hasattr(result, 'index') else list(range(len(result)))
                    return result.reset_index(drop=True), indices
                return result, []
            except Exception as e:
                logger.warning(f"Biblium local citations failed: {e}")
        return self._compute_local_basic(cite_col, ref_col)
    
    def _compute_local_basic(self, cite_col, ref_col):
        """Basic local citations computation."""
        df = self._df.copy()
        df["_orig_idx"] = range(len(df))  # Track original indices
        
        # Find title column
        title_col = self._find_column(["Title", "Article Title", "TI"])
        if title_col is None:
            # Can't compute local without titles
            return self._compute_global_basic(cite_col)
        
        # Get all titles
        titles = df[title_col].dropna().unique()
        title_counts = {t: 0 for t in titles}
        
        # Count how many times each title appears in references
        for ref_text in df[ref_col].dropna():
            ref_str = str(ref_text)
            for title in titles:
                if str(title) in ref_str:
                    title_counts[title] += 1
        
        # Add local citation count
        df["Local Citations"] = df[title_col].map(title_counts).fillna(0).astype(int)
        df["Global Citations"] = pd.to_numeric(df[cite_col], errors='coerce').fillna(0)
        
        # Sort by local citations
        df = df.sort_values("Local Citations", ascending=False)
        
        # Handle ties
        if self.include_ties and self.top_n < len(df):
            cutoff = df["Local Citations"].iloc[self.top_n - 1]
            df = df[df["Local Citations"] >= cutoff]
        else:
            df = df.head(self.top_n)
        
        # Extract original indices
        indices = df["_orig_idx"].tolist()
        
        # Select columns
        display_cols = self._get_display_columns()
        out_cols = list(dict.fromkeys(display_cols + ["Local Citations", "Global Citations"]))
        out_cols = [c for c in out_cols if c in df.columns]
        
        return df[out_cols].reset_index(drop=True), indices
    
    def _compute_per_year(self, cite_col):
        """Compute citations per year (normalized)."""
        df = self._df.copy()
        df["_orig_idx"] = range(len(df))  # Track original indices
        
        year_col = self._find_column(["Year", "Publication Year", "publication_year", "PY"])
        if year_col is None:
            # Fall back to global if no year
            return self._compute_global_basic(cite_col)
        
        df[cite_col] = pd.to_numeric(df[cite_col], errors='coerce').fillna(0)
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        
        # Calculate citations per year
        current_year = df[year_col].max()
        age = (current_year - df[year_col] + 1).clip(lower=1)
        df["Citations per Year"] = df[cite_col] / age
        
        # Sort
        df = df.sort_values("Citations per Year", ascending=False)
        
        # Handle ties
        if self.include_ties and self.top_n < len(df):
            cutoff = df["Citations per Year"].iloc[self.top_n - 1]
            df = df[df["Citations per Year"] >= cutoff]
        else:
            df = df.head(self.top_n)
        
        # Extract original indices
        indices = df["_orig_idx"].tolist()
        
        # Select columns
        display_cols = self._get_display_columns()
        out_cols = list(dict.fromkeys(display_cols + ["Citations per Year", cite_col]))
        out_cols = [c for c in out_cols if c in df.columns]
        
        result = df[out_cols].reset_index(drop=True)
        result = result.rename(columns={cite_col: "Global Citations"})
        
        return result, indices
    
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    def _df_to_table(self, df: pd.DataFrame) -> Table:
        """Convert DataFrame to Orange Table."""
        attrs = []
        metas = []
        
        for col in df.columns:
            col_data = df[col]
            if pd.api.types.is_numeric_dtype(col_data.dtype):
                attrs.append(ContinuousVariable(str(col)))
            else:
                metas.append(StringVariable(str(col)))
        
        domain = Domain(attrs, metas=metas)
        
        X = np.zeros((len(df), len(attrs)), dtype=float)
        M = np.zeros((len(df), len(metas)), dtype=object)
        
        for i, var in enumerate(attrs):
            X[:, i] = pd.to_numeric(df[var.name], errors='coerce').fillna(np.nan).values
        
        for i, var in enumerate(metas):
            M[:, i] = df[var.name].fillna("").astype(str).values
        
        return Table.from_numpy(domain, X, metas=M if metas else None)
    
    def _update_results_display(self):
        """Update the results preview."""
        if self._result_df is None or self._result_df.empty:
            self.summary_label.setText("No results")
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return
        
        df = self._result_df
        
        # Summary
        n_docs = len(df)
        
        # Find citation columns for summary
        global_col = "Global Citations" if "Global Citations" in df.columns else None
        local_col = "Local Citations" if "Local Citations" in df.columns else None
        per_year_col = "Citations per Year" if "Citations per Year" in df.columns else None
        
        summary_parts = [
            f"<b>Analysis:</b> {self.analysis_type}",
            f"<b>Documents:</b> {n_docs:,}",
        ]
        
        if global_col:
            total = df[global_col].sum()
            max_val = df[global_col].max()
            summary_parts.append(f"<b>Total global citations:</b> {total:,.0f}")
            summary_parts.append(f"<b>Max global citations:</b> {max_val:,.0f}")
        
        if local_col:
            total = df[local_col].sum()
            summary_parts.append(f"<b>Total local citations:</b> {total:,.0f}")
        
        self.summary_label.setText("<br>".join(summary_parts))
        
        # Preview table
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i, j]
                if isinstance(val, float):
                    if pd.isna(val):
                        text = ""
                    elif val == int(val):
                        text = f"{int(val):,}"
                    else:
                        text = f"{val:.2f}"
                else:
                    text = str(val)[:80]
                self.results_table.setItem(i, j, QTableWidgetItem(text))
        
        self.results_table.resizeColumnsToContents()


if __name__ == "__main__":
    WidgetPreview(OWTopCited).run()
