# -*- coding: utf-8 -*-
"""
Bibliometric Statistics Widget
==============================
Orange widget for computing bibliometric performance indicators.

Supports three depth levels:
- Core: Number of documents, Total citations, H-index, Average year
- Extended: + G-index, C-index, Year quartiles
- Full: + A, R, W, T, Pi, HG, Chi indices, Gini

Includes entity filtering via include/exclude lists and regex patterns.
Geographic filtering supports EU, continent names, and country codes/names.
"""

import os
import re
import logging
from typing import Optional, Dict, List, Any, Set
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QPlainTextEdit, QLineEdit,
    QFileDialog, QFrame,
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

ENTITY_TYPES = {
    "Authors": {
        "columns": ["Authors", "Author full names", "Author(s) ID", "AU"],
        "value_type": "list",
        "label": "Author",
    },
    "Sources": {
        "columns": ["Source title", "Source", "Journal", "SO"],
        "value_type": "string",
        "label": "Source",
    },
    "Countries": {
        "columns": ["Countries of Authors", "Countries", "Country", "countries", "CA Country", "authorships.countries"],
        "value_type": "list",
        "label": "Country",
    },
    "Affiliations": {
        "columns": ["Affiliations", "Affiliation", "C1"],
        "value_type": "list",
        "label": "Affiliation",
    },
    "Author Keywords": {
        "columns": ["Author Keywords", "Author keywords", "DE"],
        "value_type": "list",
        "label": "Keyword",
    },
    "Index Keywords": {
        "columns": ["Index Keywords", "Index keywords", "ID"],
        "value_type": "list",
        "label": "Keyword",
    },
    "All Keywords": {
        "columns": ["All Keywords", "Keywords", "Processed Keywords"],
        "value_type": "list",
        "label": "Keyword",
    },
    "References": {
        "columns": ["References", "Cited References", "CR"],
        "value_type": "list",
        "label": "Reference",
    },
    "Document Types": {
        "columns": ["Document Type", "Document type", "DT"],
        "value_type": "string",
        "label": "Document Type",
    },
    "Subject Fields": {
        "columns": ["Subject Area", "Field", "Research Areas", "WC", "SC"],
        "value_type": "list",
        "label": "Field",
    },
}

DEPTH_LEVELS = {
    "Core (H-index, citations)": "core",
    "Extended (+G-index, C-index)": "extended",
    "Full (+A,R,W,T,Pi,HG,Chi indices)": "full",
}


# =============================================================================
# BIBLIOMETRIC FUNCTIONS (fallback if biblium not available)
# =============================================================================

def h_index(citations):
    """Compute H-index from list of citations."""
    if not citations:
        return 0
    sorted_cites = sorted([int(c) for c in citations if pd.notna(c)], reverse=True)
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


def w_index(citations):
    """Compute W-index (weighted h-index)."""
    if not citations:
        return 0
    sorted_cites = sorted([int(c) for c in citations if pd.notna(c)], reverse=True)
    w = 0
    for i, c in enumerate(sorted_cites, 1):
        if c >= 10 * i:
            w = i
        else:
            break
    return w


def hg_index(citations):
    """Compute HG-index (geometric mean of h and g)."""
    h = h_index(citations)
    g = g_index(citations)
    return np.sqrt(h * g) if h and g else 0


def average_citations(citations):
    """Compute average citations."""
    valid = [c for c in citations if pd.notna(c)]
    return np.mean(valid) if valid else 0


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWBibliometricStats(OWWidget):
    """Compute bibliometric performance indicators."""
    
    name = "Bibliometric Statistics"
    description = "Compute performance indicators (H-index, G-index, etc.) for bibliometric entities"
    icon = "icons/bibliometric_stats.svg"
    priority = 30
    keywords = ["h-index", "g-index", "citations", "performance", "bibliometric", "statistics"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        stats = Output("Statistics", Table, doc="Entity statistics table")
        selected_data = Output("Selected Documents", Table, doc="Data for selected entities")
    
    # Settings - Entity
    entity_type = settings.Setting("Authors")
    
    # Settings - Options
    top_n = settings.Setting(50)
    depth = settings.Setting("Core (H-index, citations)")
    
    # Settings - Filtering
    include_items = settings.Setting("")
    exclude_items = settings.Setting("")
    regex_include = settings.Setting("")
    regex_exclude = settings.Setting("")
    max_items = settings.Setting(0)
    skip_header = settings.Setting(True)
    
    # Settings - Geographic filter
    country_filter = settings.Setting("")
    
    auto_apply = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_column = Msg("Required column not found: {}")
        no_citations = Msg("Citations column ('Cited by') not found")
        compute_error = Msg("Computation error: {}")
    
    class Warning(OWWidget.Warning):
        empty_result = Msg("No entities found matching criteria")
        no_biblium = Msg("Biblium not installed - using basic implementation")
        geo_filter_applied = Msg("Geographic filter: {} documents from {} total")
    
    class Information(OWWidget.Information):
        computed = Msg("Computed statistics for {:,} entities")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._stats_df: Optional[pd.DataFrame] = None
        self._available_columns: List[str] = []
        self._current_column: Optional[str] = None
        self._entity_doc_indices: Dict[str, List[int]] = {}
        self._filtered_doc_indices: Optional[List[int]] = None
        
        if not HAS_BIBLIUM:
            self.Warning.no_biblium()
        
        self._setup_control_area()
        self._setup_main_area()
    
    # =========================================================================
    # GUI SETUP
    # =========================================================================
    
    def _setup_control_area(self):
        """Build control area."""
        # Entity Type
        entity_box = gui.widgetBox(self.controlArea, "Entity Type")
        
        entity_layout = QGridLayout()
        entity_layout.addWidget(QLabel("Analyze:"), 0, 0)
        self.entity_combo = QComboBox()
        self.entity_combo.addItems(list(ENTITY_TYPES.keys()))
        self.entity_combo.setCurrentText(self.entity_type)
        self.entity_combo.currentTextChanged.connect(self._on_entity_changed)
        entity_layout.addWidget(self.entity_combo, 0, 1)
        entity_box.layout().addLayout(entity_layout)
        
        # Options
        options_box = gui.widgetBox(self.controlArea, "Options")
        
        options_layout = QGridLayout()
        
        options_layout.addWidget(QLabel("Top N:"), 0, 0)
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 10000)
        self.top_n_spin.setValue(self.top_n)
        self.top_n_spin.valueChanged.connect(lambda v: setattr(self, 'top_n', v))
        options_layout.addWidget(self.top_n_spin, 0, 1)
        
        options_layout.addWidget(QLabel("Depth:"), 1, 0)
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(list(DEPTH_LEVELS.keys()))
        self.depth_combo.setCurrentText(self.depth)
        self.depth_combo.currentTextChanged.connect(lambda d: setattr(self, 'depth', d))
        options_layout.addWidget(self.depth_combo, 1, 1)
        
        options_box.layout().addLayout(options_layout)
        
        # Entity Filtering
        filter_box = gui.widgetBox(self.controlArea, "Entity Filtering", flat=False)
        
        # Include items
        inc_layout = QHBoxLayout()
        inc_layout.addWidget(QLabel("Include only (one per line):"))
        inc_load_btn = QPushButton("Load")
        inc_load_btn.setMaximumWidth(50)
        inc_load_btn.clicked.connect(lambda: self._load_list("include"))
        inc_layout.addWidget(inc_load_btn)
        filter_box.layout().addLayout(inc_layout)
        
        self.include_edit = QPlainTextEdit()
        self.include_edit.setMaximumHeight(60)
        self.include_edit.setPlainText(self.include_items)
        self.include_edit.textChanged.connect(
            lambda: setattr(self, 'include_items', self.include_edit.toPlainText())
        )
        filter_box.layout().addWidget(self.include_edit)
        
        # Exclude items
        exc_layout = QHBoxLayout()
        exc_layout.addWidget(QLabel("Exclude (one per line):"))
        exc_load_btn = QPushButton("Load")
        exc_load_btn.setMaximumWidth(50)
        exc_load_btn.clicked.connect(lambda: self._load_list("exclude"))
        exc_layout.addWidget(exc_load_btn)
        filter_box.layout().addLayout(exc_layout)
        
        self.exclude_edit = QPlainTextEdit()
        self.exclude_edit.setMaximumHeight(60)
        self.exclude_edit.setPlainText(self.exclude_items)
        self.exclude_edit.textChanged.connect(
            lambda: setattr(self, 'exclude_items', self.exclude_edit.toPlainText())
        )
        filter_box.layout().addWidget(self.exclude_edit)
        
        # Regex filters
        regex_layout = QGridLayout()
        regex_layout.addWidget(QLabel("Regex include:"), 0, 0)
        self.regex_inc_edit = QLineEdit(self.regex_include)
        self.regex_inc_edit.setPlaceholderText("e.g., ^Smith|^Jones (regex pa...")
        self.regex_inc_edit.textChanged.connect(lambda t: setattr(self, 'regex_include', t))
        regex_layout.addWidget(self.regex_inc_edit, 0, 1)
        
        regex_layout.addWidget(QLabel("Regex exclude:"), 1, 0)
        self.regex_exc_edit = QLineEdit(self.regex_exclude)
        self.regex_exc_edit.setPlaceholderText("e.g., Unknown|Anonymous")
        self.regex_exc_edit.textChanged.connect(lambda t: setattr(self, 'regex_exclude', t))
        regex_layout.addWidget(self.regex_exc_edit, 1, 1)
        
        regex_layout.addWidget(QLabel("Max items:"), 2, 0)
        self.max_items_combo = QComboBox()
        self.max_items_combo.addItems(["No limit", "10", "25", "50", "100", "250", "500", "1000"])
        if self.max_items == 0:
            self.max_items_combo.setCurrentText("No limit")
        else:
            self.max_items_combo.setCurrentText(str(self.max_items))
        self.max_items_combo.currentTextChanged.connect(self._on_max_items_changed)
        regex_layout.addWidget(self.max_items_combo, 2, 1)
        
        filter_box.layout().addLayout(regex_layout)
        
        gui.checkBox(filter_box, self, "skip_header", "Skip header row when loading files")
        
        note = QLabel("<small>Note: 'Include only' overrides Top N. Max items=0 means no limit.<br>Load from .txt, .csv, or .xlsx files.</small>")
        note.setWordWrap(True)
        filter_box.layout().addWidget(note)
        
        # Geographic Filter
        geo_box = gui.widgetBox(self.controlArea, "Geographic Filter (Countries)", flat=False)
        
        geo_layout = QHBoxLayout()
        geo_layout.addWidget(QLabel("Countries:"))
        self.country_edit = QLineEdit()
        self.country_edit.setPlaceholderText("EU, US, DE, ...")
        self.country_edit.setText(self.country_filter)
        self.country_edit.textChanged.connect(lambda t: setattr(self, 'country_filter', t))
        geo_layout.addWidget(self.country_edit)
        geo_box.layout().addLayout(geo_layout)
        
        geo_note = QLabel("<small>Filter entities by associated countries.<br>Supports: EU, country codes (US, DE), full names, comma-separated.</small>")
        geo_note.setWordWrap(True)
        geo_box.layout().addWidget(geo_note)
        
        # Apply button
        self.apply_btn = gui.button(
            self.controlArea, self, "Compute Statistics",
            callback=self.commit, autoDefault=False
        )
        self.apply_btn.setMinimumHeight(35)
        
        gui.checkBox(self.controlArea, self, "auto_apply", "Apply Automatically")
        
        self.controlArea.layout().addStretch(1)
    
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
        results_box = gui.widgetBox(self.mainArea, "Statistics Preview (select rows to filter documents)")
        self.results_table = QTableWidget()
        self.results_table.setMinimumHeight(300)
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
    
    def _clear_selection(self):
        """Clear table selection."""
        self.results_table.clearSelection()
        self.selection_label.setText("")
        self.Outputs.selected_data.send(None)
    
    def _on_max_items_changed(self, text):
        if text == "No limit":
            self.max_items = 0
        else:
            self.max_items = int(text)
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_entity_changed(self, entity_type):
        self.entity_type = entity_type
        if self.auto_apply:
            self.commit()
    
    def _on_selection_changed(self):
        """Handle selection change in results table."""
        selected_rows = set(item.row() for item in self.results_table.selectedItems())
        
        if not selected_rows or self._stats_df is None or self._data is None:
            self.selection_label.setText("")
            self.Outputs.selected_data.send(None)
            return
        
        # Get selected entity names (first column is always the entity name)
        entity_col = self._stats_df.columns[0]
        selected_entities = set()
        for row in selected_rows:
            if row < len(self._stats_df):
                entity = str(self._stats_df.iloc[row][entity_col])
                selected_entities.add(entity)
        
        if not selected_entities:
            self.selection_label.setText("")
            self.Outputs.selected_data.send(None)
            return
        
        # Collect all document indices for selected entities
        selected_indices = []
        for entity in selected_entities:
            if entity in self._entity_doc_indices:
                selected_indices.extend(self._entity_doc_indices[entity])
        
        selected_indices = sorted(set(selected_indices))
        
        if selected_indices:
            selected_data = self._data[selected_indices]
            n_entities = len(selected_entities)
            n_docs = len(selected_indices)
            self.selection_label.setText(f"✓ {n_entities} entities selected → {n_docs} documents")
            self.Outputs.selected_data.send(selected_data)
        else:
            self.selection_label.setText(f"✓ {len(selected_entities)} entities selected → 0 documents")
            self.Outputs.selected_data.send(None)
    
    def _load_list(self, list_type):
        """Load items from file."""
        filters = "Text Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, f"Load {list_type} items", "", filters)
        if not path:
            return
        
        try:
            items = self._read_items_file(path)
            text = "\n".join(items)
            
            if list_type == "include":
                self.include_edit.setPlainText(text)
                self.include_items = text
            else:
                self.exclude_edit.setPlainText(text)
                self.exclude_items = text
                
        except Exception as e:
            logger.error(f"Error loading file: {e}")
    
    def _read_items_file(self, path):
        """Read items from file."""
        ext = os.path.splitext(path)[1].lower()
        items = []
        
        if ext == ".xlsx" or ext == ".xls":
            df = pd.read_excel(path, header=0 if self.skip_header else None)
            items = df.iloc[:, 0].dropna().astype(str).tolist()
        elif ext == ".csv":
            df = pd.read_csv(path, header=0 if self.skip_header else None)
            items = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                start = 1 if self.skip_header else 0
                items = [line.strip() for line in lines[start:] if line.strip()]
        
        return items
    
    def commit(self):
        """Compute statistics."""
        self._compute_stats()
    
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
        self._stats_df = None
        self._entity_doc_indices = {}
        self._filtered_doc_indices = None
        self.selection_label.setText("")
        
        if data is None:
            self.Error.no_data()
            self._update_results_display()
            self.Outputs.stats.send(None)
            self.Outputs.selected_data.send(None)
            return
        
        # Convert to DataFrame
        self._df = self._table_to_df(data)
        
        # Update summary
        self.summary_label.setText(f"Loaded {len(self._df):,} documents")
        
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
    
    # =========================================================================
    # GEOGRAPHIC FILTERING
    # =========================================================================
    
    def _get_target_countries(self, filter_text: str) -> Set[str]:
        """Get set of target countries from filter text.
        
        Returns both country codes AND full names to match different data formats.
        """
        if not filter_text.strip():
            return set()
        
        filter_text = filter_text.strip()
        result = set()
        
        if HAS_BIBLIUM and utilsbib:
            try:
                # Get country data from biblium
                eu_countries = utilsbib._get_eu_countries() or []
                code_dct = utilsbib._get_code_dct() or {}  # Name -> Code
                code_dct_r = utilsbib._get_code_dct_r() or {}  # Code -> Name
                l_countries = utilsbib._get_l_countries() or []  # All country names
                continent_dct = utilsbib._get_continent_dct() or {}  # Name -> Continent
                
                # Handle multiple comma-separated values
                parts = [p.strip() for p in filter_text.split(",") if p.strip()]
                
                for part in parts:
                    part_upper = part.upper()
                    
                    # Handle EU
                    if part_upper == "EU":
                        for country in eu_countries:
                            result.add(country)  # Full name
                            if country in code_dct:
                                result.add(code_dct[country])  # Code
                        continue
                    
                    # Handle continent names
                    continent_names = ["EUROPE", "ASIA", "AFRICA", "NORTH AMERICA", "SOUTH AMERICA", "OCEANIA", "ANTARCTICA"]
                    if part_upper in continent_names:
                        for country, continent in continent_dct.items():
                            if continent and continent.upper() == part_upper:
                                result.add(country)  # Full name
                                if country in code_dct:
                                    result.add(code_dct[country])  # Code
                        continue
                    
                    # Handle country code
                    if part_upper in code_dct_r:
                        result.add(part_upper)  # Code
                        result.add(code_dct_r[part_upper])  # Full name
                        continue
                    
                    # Handle country name (case-insensitive)
                    for country in l_countries:
                        if country.upper() == part_upper:
                            result.add(country)  # Full name
                            if country in code_dct:
                                result.add(code_dct[country])  # Code
                            break
                    else:
                        # Just add it as-is (might be a partial match)
                        result.add(part)
                
                return result
                
            except Exception as e:
                logger.warning(f"Error getting country data from biblium: {e}")
        
        # Fallback: treat as comma-separated list of country names/codes
        parts = [p.strip() for p in filter_text.split(",") if p.strip()]
        return set(parts)
    
    def _apply_geographic_filter(self, df: pd.DataFrame) -> tuple:
        """Apply geographic filter and return filtered df with index mapping.
        
        Returns (filtered_df, original_indices) where original_indices maps 
        filtered df row positions to original data indices.
        """
        if not self.country_filter.strip():
            # No filter - return original df with identity mapping
            return df, list(range(len(df)))
        
        # Find countries column
        country_cols = ["Countries of Authors", "Countries", "Country", "countries", "CA Country", "authorships.countries"]
        country_col = None
        for col in country_cols:
            if col in df.columns:
                country_col = col
                break
        
        if country_col is None:
            # Can't filter by country - return original
            return df, list(range(len(df)))
        
        # Get target countries (includes both codes and full names)
        target_countries = self._get_target_countries(self.country_filter)
        
        if not target_countries:
            return df, list(range(len(df)))
        
        # Determine separator
        sep = "; "
        sample = df[country_col].dropna()
        if len(sample) > 0:
            sample_str = str(sample.iloc[0])
            if "|" in sample_str:
                sep = "|"
        
        # Filter rows that have at least one target country
        matching_indices = []
        for idx in range(len(df)):
            val = df.iloc[idx][country_col]
            if pd.isna(val):
                continue
            
            val_str = str(val)
            countries = [c.strip() for c in val_str.split(sep) if c.strip()]
            
            # Check if any country matches (case-insensitive)
            for c in countries:
                if c in target_countries or c.upper() in {t.upper() for t in target_countries}:
                    matching_indices.append(idx)
                    break
        
        if not matching_indices:
            return pd.DataFrame(), []
        
        filtered_df = df.iloc[matching_indices].copy().reset_index(drop=True)
        return filtered_df, matching_indices
    
    # =========================================================================
    # COMPUTATION
    # =========================================================================
    
    def _compute_stats(self):
        """Compute bibliometric statistics."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        self._entity_doc_indices = {}
        self._current_column = None
        self._filtered_doc_indices = None
        self.selection_label.setText("")
        
        if self._df is None or self._df.empty:
            self.Error.no_data()
            self._update_results_display()
            self.Outputs.stats.send(None)
            self.Outputs.selected_data.send(None)
            return
        
        try:
            # Apply geographic filter
            work_df, filtered_indices = self._apply_geographic_filter(self._df)
            self._filtered_doc_indices = filtered_indices
            
            if work_df.empty:
                self.Warning.empty_result()
                self._stats_df = None
                self._update_results_display()
                self.Outputs.stats.send(None)
                self.Outputs.selected_data.send(None)
                return
            
            # Show filter info
            if self.country_filter.strip() and len(filtered_indices) < len(self._df):
                self.Warning.geo_filter_applied(len(filtered_indices), len(self._df))
            
            # Find entity column
            entity_config = ENTITY_TYPES.get(self.entity_type, {})
            entity_col = self._find_column(work_df, entity_config.get("columns", []))
            
            if entity_col is None:
                self.Error.no_column(", ".join(entity_config.get("columns", [])[:3]))
                self._update_results_display()
                self.Outputs.stats.send(None)
                self.Outputs.selected_data.send(None)
                return
            
            self._current_column = entity_col
            
            # Check for citations column
            cite_col = self._find_column(work_df, ["Cited by", "Times Cited", "Citation Count", "cited_by_count", "TC"])
            if cite_col is None:
                self.Error.no_citations()
                self._update_results_display()
                self.Outputs.stats.send(None)
                self.Outputs.selected_data.send(None)
                return
            
            # Get depth mode
            mode = DEPTH_LEVELS.get(self.depth, "core")
            
            # Get entity label and value type
            entity_label = entity_config.get("label", "Item")
            value_type = entity_config.get("value_type", "list")
            
            # Use biblium if available
            if HAS_BIBLIUM and utilsbib:
                stats_df = self._compute_with_biblium(
                    work_df, entity_col, entity_label, value_type, cite_col, mode
                )
            else:
                stats_df = self._compute_basic(
                    work_df, entity_col, entity_label, value_type, cite_col, mode
                )
            
            if stats_df is None or stats_df.empty:
                self.Warning.empty_result()
                self._stats_df = None
                self._update_results_display()
                self.Outputs.stats.send(None)
                self.Outputs.selected_data.send(None)
                return
            
            self._stats_df = stats_df
            
            # Build entity-document index mapping using filtered indices
            self._build_entity_doc_indices(work_df, entity_col, value_type, stats_df, filtered_indices)
            
            # Update display
            self._update_results_display()
            
            # Send output
            output_table = self._df_to_table(stats_df)
            self.Outputs.stats.send(output_table)
            self.Outputs.selected_data.send(None)  # Clear selection output
            
            self.Information.computed(len(stats_df))
            
        except Exception as e:
            import traceback
            logger.error(f"Computation error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
            self._stats_df = None
            self._update_results_display()
            self.Outputs.stats.send(None)
            self.Outputs.selected_data.send(None)
    
    def _build_entity_doc_indices(self, work_df: pd.DataFrame, entity_col: str, 
                                   value_type: str, stats_df: pd.DataFrame,
                                   original_indices: List[int]):
        """Build mapping from entities to ORIGINAL document indices.
        
        Uses the actual entity names from the stats output to ensure matching.
        Maps through filtered indices to get back to original data indices.
        """
        self._entity_doc_indices = {}
        
        # Get the entity names that appear in stats
        stats_entity_col = stats_df.columns[0]
        stats_entities = set(str(e) for e in stats_df[stats_entity_col].dropna())
        
        # Determine separator
        sep = "; "
        if len(work_df) > 0:
            sample = work_df[entity_col].dropna()
            if len(sample) > 0:
                sample_str = str(sample.iloc[0])
                if "|" in sample_str:
                    sep = "|"
        
        # Build mapping - work_df position i corresponds to original_indices[i]
        for work_idx in range(len(work_df)):
            val = work_df.iloc[work_idx][entity_col]
            if pd.isna(val):
                continue
            
            original_idx = original_indices[work_idx]
            val_str = str(val)
            
            if value_type == "list":
                entities = [e.strip() for e in val_str.split(sep) if e.strip()]
            else:
                entities = [val_str.strip()] if val_str.strip() else []
            
            for entity in entities:
                # Only include entities that are in the stats output
                if entity in stats_entities:
                    if entity not in self._entity_doc_indices:
                        self._entity_doc_indices[entity] = []
                    self._entity_doc_indices[entity].append(original_idx)
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first available column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _get_items_of_interest(self):
        """Get filtered list of items to analyze."""
        if self.include_items.strip():
            return [i.strip() for i in self.include_items.strip().split("\n") if i.strip()]
        return None
    
    def _get_exclude_items(self):
        """Get list of items to exclude."""
        if self.exclude_items.strip():
            return [i.strip() for i in self.exclude_items.strip().split("\n") if i.strip()]
        return None
    
    def _compute_with_biblium(self, df, entity_col, entity_label, value_type, cite_col, mode):
        """Compute statistics using biblium."""
        items_of_interest = self._get_items_of_interest()
        exclude_items = self._get_exclude_items()
        
        # Determine separator
        sep = "; "
        if len(df) > 0 and "|" in str(df[entity_col].iloc[0]):
            sep = "|"
        
        regex_inc = self.regex_include.strip() or None
        regex_exc = self.regex_exclude.strip() or None
        
        # First count occurrences
        count_type = "single" if value_type == "string" else value_type
        counts_df = utilsbib.count_occurrences(
            df,
            entity_col,
            count_type=count_type,
            item_column_name=entity_label,
            sep=sep,
        )
        
        # Use biblium's get_entity_stats
        stats_df, _ = utilsbib.get_entity_stats(
            df=df,
            entity_col=entity_col,
            entity_label=entity_label,
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=self.top_n,
            counts_df=counts_df,
            regex_include=regex_inc,
            regex_exclude=regex_exc,
            value_type=value_type,
            mode=mode,
            sep=sep,
            max_items=self.max_items if self.max_items > 0 else 0,
        )
        
        return stats_df
    
    def _compute_basic(self, df, entity_col, entity_label, value_type, cite_col, mode):
        """Basic statistics computation without biblium."""
        df = df.copy()
        
        # Ensure citations are numeric
        df[cite_col] = pd.to_numeric(df[cite_col], errors='coerce').fillna(0)
        
        # Determine separator
        sep = "; "
        if len(df) > 0:
            sample = df[entity_col].dropna()
            if len(sample) > 0 and "|" in str(sample.iloc[0]):
                sep = "|"
        
        items_of_interest = self._get_items_of_interest()
        exclude_items = self._get_exclude_items()
        
        # Count entities
        if value_type == "list":
            all_items = []
            for val in df[entity_col].dropna().astype(str):
                items = [i.strip() for i in val.split(sep) if i.strip()]
                all_items.extend(items)
            counts = Counter(all_items)
        else:
            counts = Counter(df[entity_col].dropna().astype(str))
        
        # Apply filters
        if items_of_interest:
            counts = {k: v for k, v in counts.items() if k in items_of_interest}
        
        if self.regex_include.strip():
            pattern = re.compile(self.regex_include.strip(), re.IGNORECASE)
            counts = {k: v for k, v in counts.items() if pattern.search(k)}
        
        if self.regex_exclude.strip():
            pattern = re.compile(self.regex_exclude.strip(), re.IGNORECASE)
            counts = {k: v for k, v in counts.items() if not pattern.search(k)}
        
        if exclude_items:
            counts = {k: v for k, v in counts.items() if k not in exclude_items}
        
        # Sort and limit
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if self.max_items > 0:
            sorted_items = sorted_items[:self.max_items]
        elif not items_of_interest:
            sorted_items = sorted_items[:self.top_n]
        
        # Compute statistics for each entity
        stats_list = []
        for entity, doc_count in sorted_items:
            if value_type == "list":
                mask = df[entity_col].fillna("").str.contains(re.escape(entity), na=False)
            else:
                mask = df[entity_col] == entity
            
            entity_df = df[mask]
            citations = entity_df[cite_col].tolist()
            years = entity_df["Year"].dropna().tolist() if "Year" in entity_df.columns else []
            
            stats = {
                entity_label: entity,
                "Number of documents": doc_count,
                "Total citations": sum(citations),
                "H-index": h_index(citations),
                "Average year": np.mean(years) if years else np.nan,
            }
            
            if mode in ("extended", "full"):
                stats["G-index"] = g_index(citations)
            
            if mode == "full":
                stats["W-index"] = w_index(citations)
                stats["HG-index"] = hg_index(citations)
                stats["Average citations"] = average_citations(citations)
            
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list) if stats_list else None
    
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
        
        attr_data = np.column_stack([
            df[attr.name].values for attr in attrs
        ]) if attrs else np.empty((len(df), 0))
        
        meta_data = np.column_stack([
            df[meta.name].astype(str).values for meta in metas
        ]) if metas else np.empty((len(df), 0), dtype=object)
        
        return Table.from_numpy(domain, attr_data, metas=meta_data)
    
    def _update_results_display(self):
        """Update the results table display."""
        self.results_table.clear()
        
        if self._stats_df is None or self._stats_df.empty:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.summary_label.setText("No results")
            return
        
        # Set up table
        df = self._stats_df.head(100)  # Limit preview
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        
        # Populate table
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
                self.results_table.setItem(row_idx, col_idx, item)
        
        self.results_table.resizeColumnsToContents()
        
        # Update summary
        entity_label = ENTITY_TYPES.get(self.entity_type, {}).get("label", "Entity")
        n_docs = len(self._filtered_doc_indices) if self._filtered_doc_indices else len(self._df) if self._df is not None else 0
        total_cites = int(self._stats_df["Total citations"].sum()) if "Total citations" in self._stats_df.columns else 0
        
        summary = (
            f"<b>Entity:</b> {entity_label}<br>"
            f"<b>Entities analyzed:</b> {len(self._stats_df)}<br>"
            f"<b>Total documents:</b> {n_docs:,}<br>"
            f"<b>Total citations:</b> {total_cites:,}<br>"
            f"<b>Depth:</b> {self.depth}"
        )
        self.summary_label.setText(summary)
    
    def send_report(self):
        """Generate widget report."""
        self.report_items([
            ("Entity type", self.entity_type),
            ("Top N", self.top_n),
            ("Depth", self.depth),
            ("Geographic filter", self.country_filter or "None"),
        ])
        
        if self._stats_df is not None:
            self.report_items([
                ("Entities found", len(self._stats_df)),
            ])


if __name__ == "__main__":
    WidgetPreview(OWBibliometricStats).run()
