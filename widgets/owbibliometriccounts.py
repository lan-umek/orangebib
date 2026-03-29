# -*- coding: utf-8 -*-
"""
Bibliometric Counts Widget
==========================
Orange widget for counting entities in bibliographic data.

Supports counting:
- Single value fields (Sources, Document Types, Countries)
- Multi-valued fields (Authors, Keywords, Affiliations, References)
- Text fields with n-gram extraction (Titles, Abstracts)

Outputs:
- Counts table with entity frequencies
- Selected documents matching selected entities
"""

import os
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
    QHeaderView, QSizePolicy, QRadioButton, QButtonGroup,
    QAbstractItemView,
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

# Try sklearn for text n-grams
try:
    from sklearn.feature_extraction.text import CountVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


# =============================================================================
# ENTITY DEFINITIONS
# =============================================================================

ENTITY_TYPES = {
    "Sources": {
        "columns": ["Source title", "Source", "Journal", "source", "SO"],
        "count_type": "single",
        "label": "Source",
        "description": "Journal/conference names",
    },
    "Authors": {
        "columns": ["Authors", "Author", "authors", "AU"],
        "count_type": "list",
        "label": "Author",
        "description": "Individual authors (multi-valued)",
    },
    "Author Keywords": {
        "columns": ["Author Keywords", "Author keywords", "author_keywords", "DE", "Keywords"],
        "count_type": "list",
        "label": "Keyword",
        "description": "Author-assigned keywords",
    },
    "Index Keywords": {
        "columns": ["Index Keywords", "Index keywords", "index_keywords", "ID", "Indexed Keywords"],
        "count_type": "list",
        "label": "Keyword",
        "description": "Database-indexed keywords",
    },
    "All Keywords": {
        "columns": ["All Keywords", "Keywords", "keywords", "Processed Keywords"],
        "count_type": "list",
        "label": "Keyword",
        "description": "All keywords combined",
    },
    "Affiliations": {
        "columns": ["Affiliations", "Affiliation", "affiliations", "C1"],
        "count_type": "list",
        "label": "Affiliation",
        "description": "Author affiliations/institutions",
    },
    "Countries": {
        "columns": ["Countries of Authors", "Countries", "Country", "countries", "CA Country", "authorships.countries"],
        "count_type": "list",
        "label": "Country",
        "description": "Author countries",
    },
    "Document Types": {
        "columns": ["Document Type", "Document type", "type", "DT", "Type"],
        "count_type": "single",
        "label": "Document Type",
        "description": "Article, Review, Conference paper, etc.",
    },
    "Publication Years": {
        "columns": ["Year", "Publication Year", "publication_year", "PY"],
        "count_type": "single",
        "label": "Year",
        "description": "Publication years",
    },
    "References": {
        "columns": ["References", "Cited References", "references", "CR"],
        "count_type": "list",
        "label": "Reference",
        "description": "Cited references",
    },
    "Subject Areas": {
        "columns": ["Subject Area", "Subject Areas", "Research Areas", "WC", "SC", "Topics"],
        "count_type": "list",
        "label": "Subject Area",
        "description": "Research/subject areas",
    },
    "Funding Sponsors": {
        "columns": ["Funding Sponsor", "Funding", "Funders", "FU"],
        "count_type": "list",
        "label": "Sponsor",
        "description": "Funding sponsors/agencies",
    },
    "Title (N-grams)": {
        "columns": ["Title", "title", "TI", "Article Title"],
        "count_type": "text",
        "label": "Term",
        "description": "Terms extracted from titles",
    },
    "Abstract (N-grams)": {
        "columns": ["Abstract", "abstract", "AB", "Description"],
        "count_type": "text",
        "label": "Term",
        "description": "Terms extracted from abstracts",
    },
    "Custom Column": {
        "columns": [],
        "count_type": "auto",
        "label": "Item",
        "description": "Select any column manually",
    },
}

SEPARATORS = {
    "Auto-detect": None,
    "; ": "; ",
    "|": "|",
    ",": ",",
    "||": "||",
}


class OWBibliometricCounts(OWWidget):
    """Count entities in bibliographic data."""
    
    name = "Bibliometric Counts"
    description = "Count occurrences of entities (authors, keywords, sources, etc.) in bibliographic data"
    icon = "icons/bibliometric_counts.svg"
    priority = 20
    keywords = ["count", "frequency", "authors", "keywords", "sources", "bibliometric"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        counts = Output("Counts", Table, doc="Entity counts table")
        selected_data = Output("Selected Documents", Table, doc="Documents matching selected entities")
    
    # Settings
    entity_type = settings.Setting("Sources")
    custom_column = settings.Setting("")
    count_type_override = settings.Setting("Auto")
    separator = settings.Setting("Auto-detect")
    
    top_n = settings.Setting(0)  # 0 = all
    min_count = settings.Setting(1)
    
    normalize_counts = settings.Setting(False)
    show_cumulative = settings.Setting(True)
    compute_growth = settings.Setting(False)
    
    ngram_min = settings.Setting(1)
    ngram_max = settings.Setting(2)
    
    output_mode = settings.Setting(0)  # 0=counts, 1=documents
    auto_apply = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_column = Msg("Column not found: {}")
        count_error = Msg("Counting error: {}")
    
    class Warning(OWWidget.Warning):
        empty_result = Msg("No items found for counting")
        sklearn_missing = Msg("sklearn not installed - text n-grams disabled")
    
    class Information(OWWidget.Information):
        counted = Msg("Counted {:,} unique items from {:,} records")
        selected = Msg("Selected {:,} items matching {:,} documents")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._counts_df: Optional[pd.DataFrame] = None
        self._available_columns: List[str] = []
        self._current_column: Optional[str] = None
        self._selected_entities: Set[str] = set()
        
        if not HAS_SKLEARN:
            self.Warning.sklearn_missing()
        
        self._setup_control_area()
        self._setup_main_area()
    
    def _setup_control_area(self):
        """Build control area."""
        # Entity Selection
        entity_box = gui.widgetBox(self.controlArea, "Entity Selection")
        
        self.entity_combo = QComboBox()
        self.entity_combo.addItems(list(ENTITY_TYPES.keys()))
        self.entity_combo.setCurrentText(self.entity_type)
        self.entity_combo.currentTextChanged.connect(self._on_entity_changed)
        
        entity_layout = QGridLayout()
        entity_layout.addWidget(QLabel("Entity Type:"), 0, 0)
        entity_layout.addWidget(self.entity_combo, 0, 1)
        
        # Separator
        self.sep_label = QLabel("Separator:")
        self.sep_combo = QComboBox()
        self.sep_combo.addItems(list(SEPARATORS.keys()))
        self.sep_combo.setCurrentText(self.separator)
        self.sep_combo.currentTextChanged.connect(lambda s: setattr(self, 'separator', s))
        entity_layout.addWidget(self.sep_label, 1, 0)
        entity_layout.addWidget(self.sep_combo, 1, 1)
        
        # Custom column selection
        self.custom_label = QLabel("Column:")
        self.custom_combo = QComboBox()
        self.custom_combo.currentTextChanged.connect(self._on_custom_column_changed)
        entity_layout.addWidget(self.custom_label, 2, 0)
        entity_layout.addWidget(self.custom_combo, 2, 1)
        
        # Count type override for custom
        self.count_type_label = QLabel("Count as:")
        self.count_type_combo = QComboBox()
        self.count_type_combo.addItems(["Auto", "Single Value", "Multi-Value (List)", "Text (N-grams)"])
        self.count_type_combo.currentTextChanged.connect(self._on_count_type_changed)
        entity_layout.addWidget(self.count_type_label, 3, 0)
        entity_layout.addWidget(self.count_type_combo, 3, 1)
        
        entity_box.layout().addLayout(entity_layout)
        
        # Description
        self.desc_label = QLabel()
        self.desc_label.setStyleSheet("color: gray; font-size: 10px;")
        entity_box.layout().addWidget(self.desc_label)
        self._update_description()
        
        # Options
        options_box = gui.widgetBox(self.controlArea, "Options")
        opt_layout = QGridLayout()
        
        opt_layout.addWidget(QLabel("Top N:"), 0, 0)
        self.top_n_combo = QComboBox()
        self.top_n_combo.addItems(["All", "10", "20", "50", "100", "200", "500"])
        self.top_n_combo.setCurrentText("All" if self.top_n == 0 else str(self.top_n))
        self.top_n_combo.currentTextChanged.connect(self._on_top_n_changed)
        opt_layout.addWidget(self.top_n_combo, 0, 1)
        
        opt_layout.addWidget(QLabel("Min Count:"), 1, 0)
        self.min_count_spin = QSpinBox()
        self.min_count_spin.setRange(1, 10000)
        self.min_count_spin.setValue(self.min_count)
        self.min_count_spin.valueChanged.connect(lambda v: setattr(self, 'min_count', v))
        opt_layout.addWidget(self.min_count_spin, 1, 1)
        
        options_box.layout().addLayout(opt_layout)
        
        # Advanced options
        adv_box = gui.widgetBox(self.controlArea, "Advanced Options")
        self.normalize_check = QCheckBox("Normalize counts (fractions)")
        self.normalize_check.setChecked(self.normalize_counts)
        self.normalize_check.toggled.connect(lambda c: setattr(self, 'normalize_counts', c))
        adv_box.layout().addWidget(self.normalize_check)
        
        self.cumulative_check = QCheckBox("Show cumulative percentage")
        self.cumulative_check.setChecked(self.show_cumulative)
        self.cumulative_check.toggled.connect(lambda c: setattr(self, 'show_cumulative', c))
        adv_box.layout().addWidget(self.cumulative_check)
        
        self.growth_check = QCheckBox("Compute growth rates")
        self.growth_check.setChecked(self.compute_growth)
        self.growth_check.toggled.connect(lambda c: setattr(self, 'compute_growth', c))
        adv_box.layout().addWidget(self.growth_check)
        
        # N-gram options
        ngram_box = gui.widgetBox(self.controlArea, "N-gram Options")
        ngram_layout = QGridLayout()
        
        ngram_layout.addWidget(QLabel("Min n-gram:"), 0, 0)
        self.ngram_min_spin = QSpinBox()
        self.ngram_min_spin.setRange(1, 5)
        self.ngram_min_spin.setValue(self.ngram_min)
        self.ngram_min_spin.valueChanged.connect(lambda v: setattr(self, 'ngram_min', v))
        ngram_layout.addWidget(self.ngram_min_spin, 0, 1)
        
        ngram_layout.addWidget(QLabel("Max n-gram:"), 1, 0)
        self.ngram_max_spin = QSpinBox()
        self.ngram_max_spin.setRange(1, 5)
        self.ngram_max_spin.setValue(self.ngram_max)
        self.ngram_max_spin.valueChanged.connect(lambda v: setattr(self, 'ngram_max', v))
        ngram_layout.addWidget(self.ngram_max_spin, 1, 1)
        
        info = QLabel("<small>N-gram range: 1=unigrams, 2=bigrams, 3=trigrams.<br>Only applies to text entity types.</small>")
        info.setStyleSheet("color: gray;")
        ngram_layout.addWidget(info, 2, 0, 1, 2)
        
        ngram_box.layout().addLayout(ngram_layout)
        
        # Output Mode Selection
        output_box = gui.widgetBox(self.controlArea, "Selection Output")
        
        output_info = QLabel("<small>Select rows in the table to filter output</small>")
        output_info.setStyleSheet("color: gray;")
        output_box.layout().addWidget(output_info)
        
        # Apply button
        self.apply_btn = QPushButton("Count Entities")
        self.apply_btn.setMinimumHeight(35)
        self.apply_btn.clicked.connect(self.commit)
        self.controlArea.layout().addWidget(self.apply_btn)
        
        # Auto-apply checkbox
        gui.auto_apply(self.controlArea, self, "auto_apply")
        
        self._update_ui_visibility()
    
    def _on_top_n_changed(self, text):
        self.top_n = 0 if text == "All" else int(text)
        if self.auto_apply:
            self.commit()
    
    def commit(self):
        """Perform the counting."""
        self._count()
    
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
        
        # Results table with selection
        results_box = gui.widgetBox(self.mainArea, "Counts Preview (select rows to filter documents)")
        self.results_table = QTableWidget()
        self.results_table.setMinimumHeight(300)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
        """Clear table selection and reset outputs."""
        self.results_table.clearSelection()
        self._selected_entities = set()
        self.selection_label.setText("")
        self.Outputs.selected_data.send(None)
    
    def _on_selection_changed(self):
        """Handle table row selection."""
        selected_rows = set(item.row() for item in self.results_table.selectedItems())
        
        # Get selected entity names
        self._selected_entities = set()
        if self._counts_df is not None and len(selected_rows) > 0:
            entity_col = self._counts_df.columns[0]
            for row in selected_rows:
                if row < len(self._counts_df):
                    entity = self._counts_df.iloc[row][entity_col]
                    self._selected_entities.add(str(entity))
        
        # Update selection label and send output
        n_selected = len(self._selected_entities)
        
        if n_selected > 0 and self._data is not None:
            matching_docs = self._filter_documents_by_entities(self._selected_entities)
            n_docs = len(matching_docs) if matching_docs is not None else 0
            
            if matching_docs is not None and n_docs > 0:
                self.selection_label.setText(f"✓ {n_selected} entities selected → {n_docs} documents")
                self.Information.selected(n_selected, n_docs)
                self.Outputs.selected_data.send(matching_docs)
            else:
                self.selection_label.setText(f"✓ {n_selected} entities selected → 0 documents")
                self.Outputs.selected_data.send(None)
        else:
            self.selection_label.setText("")
            self.Information.clear()
            self.Outputs.selected_data.send(None)
    
    def _filter_documents_by_entities(self, entities: Set[str]) -> Optional[Table]:
        """Filter input documents that contain any of the selected entities."""
        if self._df is None or self._current_column is None:
            return None
        
        col = self._current_column
        if col not in self._df.columns:
            return None
        
        # Get count type to know how to parse the column
        if self.entity_type == "Custom Column":
            ct_map = {
                "Auto": "auto",
                "Single Value": "single",
                "Multi-Value (List)": "list",
                "Text (N-grams)": "text"
            }
            count_type = ct_map.get(self.count_type_override, "auto")
            if count_type == "auto":
                count_type = self._detect_count_type(col)
        else:
            entity_config = ENTITY_TYPES.get(self.entity_type, {})
            count_type = entity_config.get("count_type", "single")
        
        # Normalize entities for matching
        entities_normalized = {str(e).strip() for e in entities}
        
        # For text n-grams, search for terms in text
        if count_type == "text":
            matching_indices = []
            for idx, val in enumerate(self._df[col]):
                if pd.isna(val) or str(val).strip() == "":
                    continue
                val_lower = str(val).lower()
                if any(e.lower() in val_lower for e in entities_normalized):
                    matching_indices.append(idx)
        else:
            # Detect separator for list types
            sep = self._get_separator(col) if count_type == "list" else None
            
            # Find matching rows
            matching_indices = []
            for idx, val in enumerate(self._df[col]):
                if pd.isna(val) or str(val).strip() == "":
                    continue
                
                val_str = str(val)
                
                if sep:
                    # Multi-valued field - split and check each part
                    parts = [p.strip() for p in val_str.split(sep) if p.strip()]
                    if any(p in entities_normalized for p in parts):
                        matching_indices.append(idx)
                else:
                    # Single value field - check exact match
                    val_stripped = val_str.strip()
                    if val_stripped in entities_normalized:
                        matching_indices.append(idx)
        
        if not matching_indices:
            return None
        
        # Return subset of original data
        if self._data is not None:
            return self._data[matching_indices]
        return None
    
    def _get_separator(self, col: str) -> Optional[str]:
        """Get the separator for a column."""
        if self.separator != "Auto-detect":
            return SEPARATORS.get(self.separator)
        
        # Auto-detect from data
        sample = self._df[col].dropna()
        if len(sample) == 0:
            return None
        sample = sample.iloc[:100]
        sample_str = " ".join(str(s) for s in sample)
        
        for sep in ["|", "; ", "||", ","]:
            if sep in sample_str:
                return sep
        return None
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_entity_changed(self, entity_type):
        self.entity_type = entity_type
        self._update_ui_visibility()
        self._update_description()
        if self.auto_apply:
            self.commit()
    
    def _on_custom_column_changed(self, column):
        self.custom_column = column
        if self.auto_apply and self.entity_type == "Custom Column":
            self.commit()
    
    def _on_count_type_changed(self, count_type):
        self.count_type_override = count_type
        self._update_ui_visibility()
        if self.auto_apply:
            self.commit()
    
    def _update_ui_visibility(self):
        """Show/hide UI elements based on selections."""
        is_custom = self.entity_type == "Custom Column"
        self.custom_label.setVisible(is_custom)
        self.custom_combo.setVisible(is_custom)
        self.count_type_label.setVisible(is_custom)
        self.count_type_combo.setVisible(is_custom)
        
        # Get effective count type
        if is_custom:
            ct = self.count_type_override
        else:
            entity_config = ENTITY_TYPES.get(self.entity_type, {})
            ct = entity_config.get("count_type", "single")
        
        is_text = ct == "text" or (is_custom and self.count_type_override == "Text (N-grams)")
        is_list = ct == "list" or (is_custom and self.count_type_override == "Multi-Value (List)")
        
        self.sep_label.setVisible(is_list or is_custom)
        self.sep_combo.setVisible(is_list or is_custom)
    
    def _update_description(self):
        """Update entity description label."""
        config = ENTITY_TYPES.get(self.entity_type, {})
        desc = config.get("description", "")
        self.desc_label.setText(f"<i>{desc}</i>")
    
    # =========================================================================
    # INPUT HANDLING
    # =========================================================================
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive input data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._counts_df = None
        self._selected_entities = set()
        self._current_column = None
        self.selection_label.setText("")
        
        if data is None:
            self._available_columns = []
            self.custom_combo.clear()
            self.Outputs.counts.send(None)
            self.Outputs.selected_data.send(None)
            return
        
        # Convert to DataFrame
        self._df = self._table_to_df(data)
        self._available_columns = list(self._df.columns)
        
        # Update custom column combo
        self.custom_combo.clear()
        self.custom_combo.addItems(self._available_columns)
        if self.custom_column in self._available_columns:
            self.custom_combo.setCurrentText(self.custom_column)
        
        if self.auto_apply:
            self.commit()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        """Convert Orange Table to DataFrame."""
        data = {}
        
        # Attributes
        for var in table.domain.attributes:
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else "" for v in col]
            else:
                data[var.name] = col
        
        # Class variable
        if table.domain.class_var:
            var = table.domain.class_var
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else "" for v in col]
            else:
                data[var.name] = col
        
        # Metas
        for var in table.domain.metas:
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else "" for v in col]
            else:
                data[var.name] = [str(v) if v is not None else "" for v in col]
        
        return pd.DataFrame(data)
    
    # =========================================================================
    # COUNTING LOGIC
    # =========================================================================
    
    def _count(self):
        """Main counting logic."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        self._counts_df = None
        self._selected_entities = set()
        self.selection_label.setText("")
        
        if self._df is None:
            self.Error.no_data()
            self.Outputs.counts.send(None)
            self.Outputs.selected_data.send(None)
            self._update_results_display()
            return
        
        # Find the column to count
        col = self._find_entity_column()
        if col is None:
            columns = ENTITY_TYPES.get(self.entity_type, {}).get("columns", [])
            self.Error.no_column(", ".join(columns) if columns else self.entity_type)
            self.Outputs.counts.send(None)
            self.Outputs.selected_data.send(None)
            self._update_results_display()
            return
        
        self._current_column = col
        
        try:
            # Get count type
            if self.entity_type == "Custom Column":
                ct_map = {
                    "Auto": "auto",
                    "Single Value": "single",
                    "Multi-Value (List)": "list",
                    "Text (N-grams)": "text"
                }
                count_type = ct_map.get(self.count_type_override, "auto")
                if count_type == "auto":
                    count_type = self._detect_count_type(col)
            else:
                entity_config = ENTITY_TYPES.get(self.entity_type, {})
                count_type = entity_config.get("count_type", "single")
            
            # Perform counting
            if count_type == "text":
                if not HAS_SKLEARN:
                    self.Error.count_error("sklearn required for text n-grams")
                    self.Outputs.counts.send(None)
                    self._update_results_display()
                    return
                counts = self._count_ngrams(col)
            elif count_type == "list":
                counts = self._count_list_values(col)
            else:
                counts = self._count_single_values(col)
            
            if not counts:
                self.Warning.empty_result()
                self.Outputs.counts.send(None)
                self.Outputs.selected_data.send(None)
                self._update_results_display()
                return
            
            # Build result DataFrame
            entity_config = ENTITY_TYPES.get(self.entity_type, {})
            label = entity_config.get("label", "Item")
            
            df = pd.DataFrame([
                {label: k, "Number of documents": v}
                for k, v in counts.items()
            ])
            
            df = df.sort_values("Number of documents", ascending=False).reset_index(drop=True)
            
            # Apply filtering
            df = self._apply_filters(df)
            
            # Add computed columns
            df = self._add_optional_columns(df)
            
            self._counts_df = df
            
            self.Information.counted(len(df), len(self._df))
            
            # Update display
            self._update_results_display()
            
            # Send output
            output_table = self._df_to_table(df)
            self.Outputs.counts.send(output_table)
            
            # Initially no selection, so no documents output
            self.Outputs.selected_data.send(None)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.Error.count_error(str(e))
            self.Outputs.counts.send(None)
            self.Outputs.selected_data.send(None)
    
    def _find_entity_column(self) -> Optional[str]:
        """Find the column to count."""
        if self.entity_type == "Custom Column":
            if self.custom_column in self._df.columns:
                return self.custom_column
            return None
        
        entity_config = ENTITY_TYPES.get(self.entity_type, {})
        possible_columns = entity_config.get("columns", [])
        
        for col in possible_columns:
            if col in self._df.columns:
                return col
        
        return None
    
    def _detect_count_type(self, col: str) -> str:
        """Auto-detect whether column is single, list, or text."""
        sample = self._df[col].dropna()
        if len(sample) == 0:
            return "single"
        sample = sample.iloc[:100]
        
        for val in sample:
            val_str = str(val)
            for sep in ["|", "; ", "||"]:
                if sep in val_str:
                    return "list"
        
        avg_len = np.mean([len(str(v)) for v in sample])
        if avg_len > 100:
            return "text"
        
        return "single"
    
    def _count_single_values(self, col: str) -> Dict[str, int]:
        """Count single values."""
        counts = Counter()
        for val in self._df[col].dropna():
            val_str = str(val).strip()
            if val_str:
                counts[val_str] += 1
        return dict(counts)
    
    def _count_list_values(self, col: str) -> Dict[str, int]:
        """Count multi-valued (list) fields."""
        sep = self._get_separator(col)
        
        counts = Counter()
        for val in self._df[col].dropna():
            val_str = str(val)
            if sep:
                parts = [p.strip() for p in val_str.split(sep) if p.strip()]
            else:
                parts = [val_str.strip()] if val_str.strip() else []
            
            for part in parts:
                counts[part] += 1
        
        return dict(counts)
    
    def _count_ngrams(self, col: str) -> Dict[str, int]:
        """Extract and count n-grams from text."""
        texts = self._df[col].dropna().astype(str).tolist()
        if not texts:
            return {}
        
        try:
            vectorizer = CountVectorizer(
                ngram_range=(self.ngram_min, self.ngram_max),
                stop_words='english',
                min_df=1,
                max_features=10000
            )
            X = vectorizer.fit_transform(texts)
            
            feature_names = vectorizer.get_feature_names_out()
            counts = X.sum(axis=0).A1
            
            return dict(zip(feature_names, counts.astype(int)))
        except Exception as e:
            logger.error(f"N-gram extraction failed: {e}")
            return {}
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min_count and top_n filters."""
        if df.empty:
            return df
        
        count_col = "Number of documents"
        
        # Min count filter
        df = df[df[count_col] >= self.min_count].copy()
        
        # Add ranks
        if len(df) > 0:
            df["Rank"] = range(1, len(df) + 1)
            n = len(df)
            if n > 1:
                df["Percentrank"] = (n - df["Rank"]) / (n - 1)
            else:
                df["Percentrank"] = 1.0
        
        # Apply top_n
        if self.top_n > 0:
            df = df.head(self.top_n).copy()
        
        # Re-rank after filtering
        if len(df) > 0:
            df["Rank"] = range(1, len(df) + 1)
            n = len(df)
            if n > 1:
                df["Percentrank"] = (n - df["Rank"]) / (n - 1)
            else:
                df["Percentrank"] = 1.0
        
        return df
    
    def _add_optional_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add optional computed columns."""
        if df.empty:
            return df
        
        count_col = "Number of documents"
        
        if self.normalize_counts and count_col in df.columns:
            total = df[count_col].sum()
            if total > 0:
                df["Normalized Count"] = df[count_col] / total
        
        if self.show_cumulative and count_col in df.columns:
            total = df[count_col].sum()
            if total > 0:
                df["Cumulative Count"] = df[count_col].cumsum()
                df["Cumulative %"] = (df["Cumulative Count"] / total) * 100
        
        return df
    
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
        if self._counts_df is None or self._counts_df.empty:
            self.summary_label.setText("No results")
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return
        
        df = self._counts_df
        
        # Summary
        item_col = df.columns[0]
        count_col = "Number of documents"
        total_items = len(df)
        total_docs = df[count_col].sum() if count_col in df.columns else 0
        
        self.summary_label.setText(
            f"<b>Entity:</b> {self.entity_type}<br>"
            f"<b>Unique items:</b> {total_items:,}<br>"
            f"<b>Total occurrences:</b> {total_docs:,.0f}"
        )
        
        # Preview table (first 100 rows)
        preview = df.head(100)
        self.results_table.setRowCount(len(preview))
        self.results_table.setColumnCount(len(preview.columns))
        self.results_table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        
        for i in range(len(preview)):
            for j, col in enumerate(preview.columns):
                val = preview.iloc[i, j]
                if isinstance(val, float):
                    if val == int(val):
                        text = f"{int(val):,}"
                    else:
                        text = f"{val:.4f}"
                else:
                    text = str(val)[:50]
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.results_table.setItem(i, j, item)
        
        self.results_table.resizeColumnsToContents()


if __name__ == "__main__":
    WidgetPreview(OWBibliometricCounts).run()
