# -*- coding: utf-8 -*-
"""
SDG Identifier Widget
=====================
Identify Sustainable Development Goals (SDGs) in bibliographic data.

Uses keyword matching based on SDG queries to classify documents
according to the 17 UN Sustainable Development Goals.
"""

import logging
import re
import json
from typing import Optional, Dict, List, Tuple, Set
from pathlib import Path

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QFrame, QLineEdit,
    QFileDialog, QMessageBox, QProgressBar,
)
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QFont, QColor

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER CLASSES
# =============================================================================

class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically by stored value."""
    
    def __init__(self, display_text: str, sort_value: float):
        super().__init__(display_text)
        self._sort_value = sort_value
    
    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


# =============================================================================
# SDG DEFINITIONS
# =============================================================================

SDG_INFO = {
    1: {"name": "No Poverty", "color": "#E5243B", "short": "SDG1"},
    2: {"name": "Zero Hunger", "color": "#DDA63A", "short": "SDG2"},
    3: {"name": "Good Health and Well-being", "color": "#4C9F38", "short": "SDG3"},
    4: {"name": "Quality Education", "color": "#C5192D", "short": "SDG4"},
    5: {"name": "Gender Equality", "color": "#FF3A21", "short": "SDG5"},
    6: {"name": "Clean Water and Sanitation", "color": "#26BDE2", "short": "SDG6"},
    7: {"name": "Affordable and Clean Energy", "color": "#FCC30B", "short": "SDG7"},
    8: {"name": "Decent Work and Economic Growth", "color": "#A21942", "short": "SDG8"},
    9: {"name": "Industry, Innovation and Infrastructure", "color": "#FD6925", "short": "SDG9"},
    10: {"name": "Reduced Inequalities", "color": "#DD1367", "short": "SDG10"},
    11: {"name": "Sustainable Cities and Communities", "color": "#FD9D24", "short": "SDG11"},
    12: {"name": "Responsible Consumption and Production", "color": "#BF8B2E", "short": "SDG12"},
    13: {"name": "Climate Action", "color": "#3F7E44", "short": "SDG13"},
    14: {"name": "Life Below Water", "color": "#0A97D9", "short": "SDG14"},
    15: {"name": "Life on Land", "color": "#56C02B", "short": "SDG15"},
    16: {"name": "Peace, Justice and Strong Institutions", "color": "#00689D", "short": "SDG16"},
    17: {"name": "Partnerships for the Goals", "color": "#19486A", "short": "SDG17"},
}

# Default SDG keywords based on Scopus/Elsevier SDG mapping methodology
# These are simplified keyword sets - real implementation would use more comprehensive queries
DEFAULT_SDG_KEYWORDS = {
    1: [  # No Poverty
        "poverty", "poor", "low-income", "economic vulnerability", "social protection",
        "basic services", "microfinance", "welfare", "deprivation", "slum",
        "homeless", "food insecurity", "income inequality", "basic needs"
    ],
    2: [  # Zero Hunger
        "hunger", "food security", "malnutrition", "nutrition", "agricultural",
        "farming", "crop", "food production", "sustainable agriculture", "famine",
        "food supply", "dietary", "undernutrition", "food access", "smallholder"
    ],
    3: [  # Good Health and Well-being
        "health", "disease", "mortality", "medical", "healthcare", "epidemic",
        "pandemic", "vaccination", "maternal health", "child health", "mental health",
        "well-being", "pharmaceutical", "hospital", "clinical", "therapy"
    ],
    4: [  # Quality Education
        "education", "school", "learning", "literacy", "teacher", "student",
        "curriculum", "educational", "academic", "university", "vocational",
        "scholarship", "training", "skills development", "inclusive education"
    ],
    5: [  # Gender Equality
        "gender", "women", "female", "girl", "feminist", "gender equality",
        "women empowerment", "gender-based violence", "maternal", "reproductive rights",
        "gender discrimination", "gender gap", "women's rights", "gender mainstreaming"
    ],
    6: [  # Clean Water and Sanitation
        "water", "sanitation", "hygiene", "drinking water", "wastewater", "sewage",
        "water quality", "water scarcity", "water management", "groundwater",
        "freshwater", "water treatment", "water supply", "water pollution"
    ],
    7: [  # Affordable and Clean Energy
        "renewable energy", "solar", "wind energy", "clean energy", "energy access",
        "energy efficiency", "sustainable energy", "electricity", "power generation",
        "biofuel", "geothermal", "hydropower", "energy transition", "fossil fuel"
    ],
    8: [  # Decent Work and Economic Growth
        "economic growth", "employment", "labor", "job", "unemployment", "worker",
        "decent work", "productivity", "entrepreneurship", "small business",
        "economic development", "GDP", "labor rights", "workplace", "income"
    ],
    9: [  # Industry, Innovation and Infrastructure
        "infrastructure", "industrialization", "innovation", "technology", "manufacturing",
        "industry", "research and development", "R&D", "digital", "internet",
        "transport", "connectivity", "engineering", "technological", "industrial"
    ],
    10: [  # Reduced Inequalities
        "inequality", "discrimination", "social inclusion", "migration", "refugee",
        "marginalized", "vulnerable", "disparity", "equal opportunity", "social justice",
        "inclusion", "xenophobia", "racism", "disadvantaged", "accessibility"
    ],
    11: [  # Sustainable Cities and Communities
        "urban", "city", "sustainable city", "housing", "transport", "public space",
        "urban planning", "urbanization", "slum", "resilient", "smart city",
        "public transport", "urban development", "heritage", "air quality"
    ],
    12: [  # Responsible Consumption and Production
        "sustainable consumption", "waste", "recycling", "circular economy", "production",
        "resource efficiency", "sustainable production", "food waste", "chemical",
        "life cycle", "corporate sustainability", "green procurement", "eco-label"
    ],
    13: [  # Climate Action
        "climate change", "global warming", "greenhouse gas", "carbon", "emission",
        "climate adaptation", "climate mitigation", "climate resilience", "Paris Agreement",
        "carbon footprint", "decarbonization", "climate policy", "temperature rise"
    ],
    14: [  # Life Below Water
        "ocean", "marine", "sea", "coastal", "fishery", "fish", "aquatic",
        "coral", "marine pollution", "overfishing", "marine ecosystem", "oceanography",
        "marine biodiversity", "seafood", "marine conservation", "plastic pollution"
    ],
    15: [  # Life on Land
        "biodiversity", "forest", "deforestation", "ecosystem", "wildlife", "species",
        "land degradation", "desertification", "habitat", "conservation", "protected area",
        "endangered species", "terrestrial", "soil", "afforestation", "poaching"
    ],
    16: [  # Peace, Justice and Strong Institutions
        "peace", "justice", "governance", "corruption", "violence", "crime",
        "rule of law", "human rights", "institution", "transparency", "accountability",
        "democracy", "conflict", "security", "legal", "court", "judiciary"
    ],
    17: [  # Partnerships for the Goals
        "partnership", "cooperation", "international", "global", "multilateral",
        "development assistance", "capacity building", "technology transfer", "trade",
        "sustainable development", "SDG", "financing", "collaboration", "stakeholder"
    ],
}

# SDG Dimensions (Environmental, Social, Economic)
SDG_DIMENSIONS = {
    "Environmental": [6, 7, 12, 13, 14, 15],
    "Social": [1, 2, 3, 4, 5, 10, 11, 16],
    "Economic": [8, 9, 17],
}

# SDG Perspectives (People, Planet, Prosperity, Peace, Partnership)
SDG_PERSPECTIVES = {
    "People": [1, 2, 3, 4, 5],
    "Planet": [6, 12, 13, 14, 15],
    "Prosperity": [7, 8, 9, 10, 11],
    "Peace": [16],
    "Partnership": [17],
}


# =============================================================================
# TEXT FIELD OPTIONS
# =============================================================================

TEXT_FIELD_OPTIONS = [
    ("Abstract", ["Abstract", "AB", "abstract"]),
    ("Title", ["Title", "TI", "title", "Document Title"]),
    ("Title + Abstract", None),  # Special: combines both
    ("Author Keywords", ["Author Keywords", "Keywords", "DE", "author_keywords"]),
    ("Index Keywords", ["Index Keywords", "Keywords Plus", "ID", "indexed_keywords"]),
    ("Full Text", ["Full Text", "full_text", "Text", "Body"]),
]


# =============================================================================
# SDG MATCHING
# =============================================================================

def keyword_to_pattern(keyword: str) -> str:
    """Convert keyword to regex pattern with word boundaries."""
    # Escape special characters
    escaped = re.escape(keyword.lower())
    # Add word boundaries
    return r'\b' + escaped + r'\b'


def identify_sdgs(text: str, sdg_keywords: Dict[int, List[str]]) -> Dict[int, bool]:
    """
    Identify which SDGs are present in text.
    
    Args:
        text: Text to analyze
        sdg_keywords: Dictionary mapping SDG number to keyword list
    
    Returns:
        Dictionary mapping SDG number to boolean (True if present)
    """
    results = {}
    
    if not text or pd.isna(text):
        return {sdg: False for sdg in range(1, 18)}
    
    text_lower = str(text).lower()
    
    for sdg_num, keywords in sdg_keywords.items():
        found = False
        for keyword in keywords:
            try:
                pattern = keyword_to_pattern(keyword)
                if re.search(pattern, text_lower):
                    found = True
                    break
            except re.error:
                # Fallback to simple contains
                if keyword.lower() in text_lower:
                    found = True
                    break
        results[sdg_num] = found
    
    return results


def count_sdg_keywords(text: str, sdg_keywords: Dict[int, List[str]]) -> Dict[int, int]:
    """
    Count keyword matches for each SDG.
    
    Args:
        text: Text to analyze
        sdg_keywords: Dictionary mapping SDG number to keyword list
    
    Returns:
        Dictionary mapping SDG number to match count
    """
    results = {}
    
    if not text or pd.isna(text):
        return {sdg: 0 for sdg in range(1, 18)}
    
    text_lower = str(text).lower()
    
    for sdg_num, keywords in sdg_keywords.items():
        count = 0
        for keyword in keywords:
            try:
                pattern = keyword_to_pattern(keyword)
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            except re.error:
                count += text_lower.count(keyword.lower())
        results[sdg_num] = count
    
    return results


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWSDGIdentifier(OWWidget):
    """Identify Sustainable Development Goals in documents."""
    
    name = "SDG Identifier"
    description = "Identify Sustainable Development Goals in your dataset"
    icon = "icons/sdg_identifier.svg"
    priority = 66
    keywords = ["SDG", "sustainable", "development", "goals", "UN", "2030"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        data = Output("Data", Table, doc="Data with SDG variables")
        sdg_summary = Output("SDG Summary", Table, doc="Summary of SDG distribution")
        sdg_documents = Output("SDG Documents", Table, doc="Documents with at least one SDG")
    
    # Settings
    field_index = settings.Setting(0)
    include_perspectives = settings.Setting(True)
    include_dimensions = settings.Setting(True)
    add_to_dataset = settings.Setting(True)
    use_numeric_labels = settings.Setting(True)  # True = 0/1, False = No/Yes
    custom_queries_path = settings.Setting("")
    auto_apply = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_field = Msg("Selected text field not found in data")
        compute_error = Msg("Computation error: {}")
    
    class Warning(OWWidget.Warning):
        low_coverage = Msg("Only {:.1f}% of documents have SDG matches")
    
    class Information(OWWidget.Information):
        identified = Msg("Identified SDGs in {:,} of {:,} documents ({:.1f}%)")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._sdg_keywords: Dict[int, List[str]] = DEFAULT_SDG_KEYWORDS.copy()
        self._results: Optional[pd.DataFrame] = None
        
        self._setup_control_area()
        self._setup_main_area()
    
    def _setup_control_area(self):
        """Build control area."""
        # Text Analysis
        text_box = gui.widgetBox(self.controlArea, "📝 Text Analysis")
        
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Text Column:"))
        self.field_combo = QComboBox()
        for name, _ in TEXT_FIELD_OPTIONS:
            self.field_combo.addItem(name)
        self.field_combo.setCurrentIndex(self.field_index)
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        field_layout.addWidget(self.field_combo)
        text_box.layout().addLayout(field_layout)
        
        hint = QLabel("<small>Select the column containing text to analyze for SDG keywords</small>")
        hint.setStyleSheet("color: #6c757d;")
        text_box.layout().addWidget(hint)
        
        # SDG Queries
        queries_box = gui.widgetBox(self.controlArea, "📋 SDG Queries")
        
        queries_layout = QHBoxLayout()
        queries_layout.addWidget(QLabel("Queries File:"))
        self.queries_edit = QLineEdit("(default)")
        self.queries_edit.setReadOnly(True)
        queries_layout.addWidget(self.queries_edit)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_queries)
        queries_layout.addWidget(self.browse_btn)
        queries_box.layout().addLayout(queries_layout)
        
        queries_hint = QLabel("<small>Uses Scopus SDG metadata by default</small>")
        queries_hint.setStyleSheet("color: #6c757d;")
        queries_box.layout().addWidget(queries_hint)
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self._reset_queries)
        queries_box.layout().addWidget(reset_btn)
        
        # Output Options
        output_box = gui.widgetBox(self.controlArea, "⚙️ Output Options")
        
        gui.checkBox(output_box, self, "include_perspectives", "Include Perspectives",
                     tooltip="Add columns for 5P framework (People, Planet, Prosperity, Peace, Partnership)")
        gui.checkBox(output_box, self, "include_dimensions", "Include Dimensions",
                     tooltip="Add columns for dimensions (Environmental, Social, Economic)")
        gui.checkBox(output_box, self, "add_to_dataset", "Add results to dataset",
                     tooltip="Add SDG columns to the original dataset")
        gui.checkBox(output_box, self, "use_numeric_labels", "Use numeric labels (0/1)",
                     tooltip="Use 0/1 instead of No/Yes for SDG variables")
        
        # Identify button
        self.identify_btn = gui.button(
            self.controlArea, self, "🔍 Identify SDGs",
            callback=self.commit, autoDefault=False
        )
        self.identify_btn.setMinimumHeight(40)
        self.identify_btn.setStyleSheet("background-color: #4a90d9; color: white; font-weight: bold;")
        
        gui.checkBox(self.controlArea, self, "auto_apply", "Auto apply")
        
        self.controlArea.layout().addStretch(1)
    
    def _setup_main_area(self):
        """Build main area."""
        main_layout = QVBoxLayout()
        self.mainArea.layout().addLayout(main_layout)
        
        # Summary header
        self.summary_label = QLabel("Load data and click 'Identify SDGs' to analyze")
        self.summary_label.setStyleSheet("font-size: 14px; color: #6c757d;")
        main_layout.addWidget(self.summary_label)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)
        
        # Results table
        results_box = QGroupBox("📊 SDG Distribution")
        results_layout = QVBoxLayout(results_box)
        
        self.results_table = QTableWidget()
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSortingEnabled(True)
        results_layout.addWidget(self.results_table)
        
        main_layout.addWidget(results_box)
        
        # Export button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_btn = QPushButton("📥 Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        main_layout.addLayout(export_layout)
    
    def _on_field_changed(self, index):
        self.field_index = index
        if self.auto_apply and self._df is not None:
            self.commit()
    
    def _browse_queries(self):
        """Browse for custom SDG queries file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load SDG Queries", "",
            "JSON files (*.json);;All files (*)"
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate format
            if isinstance(data, dict):
                new_keywords = {}
                for key, keywords in data.items():
                    sdg_num = int(key)
                    if 1 <= sdg_num <= 17 and isinstance(keywords, list):
                        new_keywords[sdg_num] = keywords
                
                if new_keywords:
                    self._sdg_keywords = new_keywords
                    self.custom_queries_path = filepath
                    self.queries_edit.setText(Path(filepath).name)
                    
                    if self.auto_apply and self._df is not None:
                        self.commit()
                else:
                    QMessageBox.warning(self, "Invalid Format", 
                                       "File must contain SDG keywords in format: {\"1\": [\"keyword1\", ...], ...}")
            
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load file: {e}")
    
    def _reset_queries(self):
        """Reset to default SDG queries."""
        self._sdg_keywords = DEFAULT_SDG_KEYWORDS.copy()
        self.custom_queries_path = ""
        self.queries_edit.setText("(default)")
        
        if self.auto_apply and self._df is not None:
            self.commit()
    
    def _find_text_column(self) -> Optional[str]:
        """Find the text column to analyze."""
        if self._df is None:
            return None
        
        name, candidates = TEXT_FIELD_OPTIONS[self.field_index]
        
        if name == "Title + Abstract":
            # Will be handled specially
            return "Title + Abstract"
        
        if candidates is None:
            return None
        
        for col in self._df.columns:
            if col in candidates:
                return col
            for candidate in candidates:
                if col.lower() == candidate.lower():
                    return col
        
        return None
    
    def _get_text_for_row(self, row_idx: int, text_col: str) -> str:
        """Get text for a row, handling combined columns."""
        if text_col == "Title + Abstract":
            parts = []
            
            # Find title column
            for col in self._df.columns:
                if col in ["Title", "TI", "title", "Document Title"]:
                    val = self._df.iloc[row_idx][col]
                    if val and not pd.isna(val):
                        parts.append(str(val))
                    break
            
            # Find abstract column
            for col in self._df.columns:
                if col in ["Abstract", "AB", "abstract"]:
                    val = self._df.iloc[row_idx][col]
                    if val and not pd.isna(val):
                        parts.append(str(val))
                    break
            
            return " ".join(parts)
        else:
            val = self._df.iloc[row_idx][text_col]
            return str(val) if val and not pd.isna(val) else ""
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive input data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._results = None
        
        self._clear_results()
        
        if data is None:
            self.Error.no_data()
            return
        
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
        """Clear results display."""
        self.summary_label.setText("Load data and click 'Identify SDGs' to analyze")
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.export_btn.setEnabled(False)
    
    def commit(self):
        """Identify SDGs in data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self.Error.no_data()
            self._send_outputs(None, None, None)
            return
        
        text_col = self._find_text_column()
        if text_col is None and self.field_index != 2:  # Not "Title + Abstract"
            self.Error.no_field()
            self._send_outputs(None, None, None)
            return
        
        if text_col is None:
            text_col = "Title + Abstract"
        
        try:
            self.progress.setVisible(True)
            self.progress.setMaximum(len(self._df))
            self.progress.setValue(0)
            
            # Identify SDGs for each document
            sdg_results = []
            
            for i in range(len(self._df)):
                text = self._get_text_for_row(i, text_col)
                sdg_matches = identify_sdgs(text, self._sdg_keywords)
                sdg_results.append(sdg_matches)
                
                if i % 100 == 0:
                    self.progress.setValue(i)
                    QApplication = __import__('AnyQt.QtWidgets', fromlist=['QApplication']).QApplication
                    QApplication.processEvents()
            
            self.progress.setValue(len(self._df))
            self.progress.setVisible(False)
            
            # Build results DataFrame
            results_df = pd.DataFrame(sdg_results)
            results_df.columns = [f"SDG{i}" for i in range(1, 18)]
            
            # Add perspectives
            if self.include_perspectives:
                for perspective, sdgs in SDG_PERSPECTIVES.items():
                    results_df[f"P_{perspective}"] = results_df[[f"SDG{s}" for s in sdgs]].any(axis=1).astype(int)
            
            # Add dimensions
            if self.include_dimensions:
                for dimension, sdgs in SDG_DIMENSIONS.items():
                    results_df[f"D_{dimension}"] = results_df[[f"SDG{s}" for s in sdgs]].any(axis=1).astype(int)
            
            # Count SDGs per document
            results_df["SDG_Count"] = results_df[[f"SDG{i}" for i in range(1, 18)]].sum(axis=1)
            results_df["Has_SDG"] = (results_df["SDG_Count"] > 0).astype(int)
            
            self._results = results_df
            
            # Update display
            self._update_results_display()
            
            # Build outputs
            self._send_outputs_from_results()
            
        except Exception as e:
            import traceback
            logger.error(f"SDG error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
            self.progress.setVisible(False)
            self._send_outputs(None, None, None)
    
    def _update_results_display(self):
        """Update the results display table."""
        if self._results is None:
            return
        
        n_total = len(self._results)
        n_with_sdg = self._results["Has_SDG"].sum()
        pct = n_with_sdg / n_total * 100 if n_total > 0 else 0
        
        self.summary_label.setText(
            f"<b>{n_with_sdg:,}</b> of <b>{n_total:,}</b> documents ({pct:.1f}%) have SDG matches"
        )
        
        self.Information.identified(n_with_sdg, n_total, pct)
        
        if pct < 10:
            self.Warning.low_coverage(pct)
        
        # Disable sorting while populating to avoid issues
        self.results_table.setSortingEnabled(False)
        
        # Build results table
        self.results_table.clear()
        self.results_table.setRowCount(17)
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["SDG", "Name", "Documents", "Percentage"])
        
        for i in range(1, 18):
            col_name = f"SDG{i}"
            count = int(self._results[col_name].sum())
            pct_sdg = count / n_total * 100 if n_total > 0 else 0
            
            # SDG number with color - use NumericTableWidgetItem for proper sorting
            sdg_item = NumericTableWidgetItem(f"SDG {i}", i)
            color = QColor(SDG_INFO[i]["color"])
            sdg_item.setBackground(color)
            sdg_item.setForeground(QColor("white") if color.lightness() < 128 else QColor("black"))
            self.results_table.setItem(i - 1, 0, sdg_item)
            
            # Name
            self.results_table.setItem(i - 1, 1, QTableWidgetItem(SDG_INFO[i]["name"]))
            
            # Count - use NumericTableWidgetItem for proper sorting
            count_item = NumericTableWidgetItem(f"{count:,}", count)
            self.results_table.setItem(i - 1, 2, count_item)
            
            # Percentage - use NumericTableWidgetItem for proper sorting
            pct_item = NumericTableWidgetItem(f"{pct_sdg:.1f}%", pct_sdg)
            self.results_table.setItem(i - 1, 3, pct_item)
        
        self.results_table.resizeColumnsToContents()
        
        # Re-enable sorting
        self.results_table.setSortingEnabled(True)
        self.export_btn.setEnabled(True)
    
    def _send_outputs_from_results(self):
        """Send outputs based on computed results."""
        if self._results is None or self._data is None:
            self._send_outputs(None, None, None)
            return
        
        # Build output data table
        if self.add_to_dataset:
            # Determine label values based on setting
            label_values = ['0', '1'] if self.use_numeric_labels else ['No', 'Yes']
            
            # Create SDG variables
            sdg_vars = []
            for i in range(1, 18):
                var = DiscreteVariable(f"SDG{i}", values=label_values)
                sdg_vars.append(var)
            
            # Add perspective variables
            if self.include_perspectives:
                for perspective in SDG_PERSPECTIVES.keys():
                    var = DiscreteVariable(f"P_{perspective}", values=label_values)
                    sdg_vars.append(var)
            
            # Add dimension variables
            if self.include_dimensions:
                for dimension in SDG_DIMENSIONS.keys():
                    var = DiscreteVariable(f"D_{dimension}", values=label_values)
                    sdg_vars.append(var)
            
            # Add count and has_sdg
            count_var = ContinuousVariable("SDG_Count")
            has_sdg_var = DiscreteVariable("Has_SDG", values=label_values)
            
            # Build new domain
            new_domain = Domain(
                list(self._data.domain.attributes) + sdg_vars + [count_var, has_sdg_var],
                self._data.domain.class_vars,
                self._data.domain.metas
            )
            
            # Build X array
            n_rows = len(self._df)
            n_orig = len(self._data.domain.attributes)
            n_new = len(sdg_vars) + 2  # +2 for count and has_sdg
            
            new_X = np.zeros((n_rows, n_orig + n_new))
            new_X[:, :n_orig] = self._data.X
            
            # Fill SDG columns
            col_idx = n_orig
            for i in range(1, 18):
                new_X[:, col_idx] = self._results[f"SDG{i}"].values
                col_idx += 1
            
            if self.include_perspectives:
                for perspective in SDG_PERSPECTIVES.keys():
                    new_X[:, col_idx] = self._results[f"P_{perspective}"].values
                    col_idx += 1
            
            if self.include_dimensions:
                for dimension in SDG_DIMENSIONS.keys():
                    new_X[:, col_idx] = self._results[f"D_{dimension}"].values
                    col_idx += 1
            
            new_X[:, col_idx] = self._results["SDG_Count"].values
            col_idx += 1
            new_X[:, col_idx] = self._results["Has_SDG"].values
            
            output_table = Table.from_numpy(
                new_domain, new_X,
                self._data.Y if self._data.Y.size > 0 else None,
                self._data.metas if self._data.metas.size > 0 else None
            )
        else:
            output_table = self._data
        
        # SDG Summary table
        summary_data = []
        n_total = len(self._results)
        for i in range(1, 18):
            count = int(self._results[f"SDG{i}"].sum())
            pct = count / n_total * 100 if n_total > 0 else 0
            summary_data.append({
                "SDG": f"SDG{i}",
                "Name": SDG_INFO[i]["name"],
                "Documents": count,
                "Percentage": f"{pct:.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_metas = [StringVariable(col) for col in summary_df.columns]
        summary_domain = Domain([], metas=summary_metas)
        summary_table = Table.from_numpy(
            summary_domain, 
            np.empty((len(summary_df), 0)), 
            metas=summary_df.astype(str).values
        )
        
        # SDG Documents (documents with at least one SDG)
        sdg_indices = np.where(self._results["Has_SDG"] == 1)[0].tolist()
        sdg_docs_table = output_table[sdg_indices] if sdg_indices else None
        
        self._send_outputs(output_table, summary_table, sdg_docs_table)
    
    def _send_outputs(self, data: Optional[Table], summary: Optional[Table], sdg_docs: Optional[Table]):
        """Send outputs."""
        self.Outputs.data.send(data)
        self.Outputs.sdg_summary.send(summary)
        self.Outputs.sdg_documents.send(sdg_docs)
    
    def _export_results(self):
        """Export results to CSV."""
        if self._results is None:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export SDG Results", "sdg_results.csv",
            "CSV files (*.csv);;All files (*)"
        )
        
        if not filepath:
            return
        
        try:
            self._results.to_csv(filepath, index=False)
            QMessageBox.information(self, "Exported", f"Results saved to {filepath}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not export: {e}")


if __name__ == "__main__":
    WidgetPreview(OWSDGIdentifier).run()
