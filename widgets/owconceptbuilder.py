# -*- coding: utf-8 -*-
"""
Concept Builder Widget
======================
Create binary concept variables from keywords.

Define concepts using keywords (with wildcard * support) and search
in specified text fields to create binary variables indicating
whether each document matches the concept.
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
    QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QAbstractItemView,
)
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QFont

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# TEXT FIELD OPTIONS
# =============================================================================

TEXT_FIELD_OPTIONS = [
    ("Auto-detect", None),
    ("Abstract", ["Abstract", "AB", "abstract"]),
    ("Title", ["Title", "TI", "title", "Document Title"]),
    ("Author Keywords", ["Author Keywords", "Keywords", "DE", "author_keywords"]),
    ("Index Keywords", ["Index Keywords", "Keywords Plus", "ID", "indexed_keywords"]),
    ("Processed Text", ["Processed Text", "processed_text", "Text"]),
]

# Auto-detect priority
AUTO_DETECT_PRIORITY = [
    ["Abstract", "AB", "abstract"],
    ["Title", "TI", "title", "Document Title"],
    ["Author Keywords", "Keywords", "DE", "author_keywords"],
    ["Index Keywords", "Keywords Plus", "ID", "indexed_keywords"],
]


# =============================================================================
# CONCEPT MATCHING
# =============================================================================

def keyword_to_regex(keyword: str, use_regex: bool = False) -> str:
    """
    Convert a keyword pattern to regex.
    
    If use_regex is False, only * wildcard is supported.
    If use_regex is True, the keyword is treated as a regex pattern.
    
    Args:
        keyword: Keyword pattern (e.g., "govern*" matches government, governance, etc.)
        use_regex: If True, treat keyword as regex
    
    Returns:
        Regex pattern string
    """
    if use_regex:
        return keyword
    else:
        # Escape special regex characters except *
        escaped = re.escape(keyword)
        # Convert * wildcard to regex .*
        pattern = escaped.replace(r'\*', r'\w*')
        # Word boundary matching
        return r'\b' + pattern + r'\b'


def matches_concept(text: str, keywords: List[str], use_regex: bool = False) -> bool:
    """
    Check if text matches any of the concept keywords.
    
    Args:
        text: Text to search in
        keywords: List of keyword patterns
        use_regex: If True, treat keywords as regex patterns
    
    Returns:
        True if any keyword matches
    """
    if not text or pd.isna(text):
        return False
    
    text = str(text).lower()
    
    for keyword in keywords:
        keyword = keyword.strip().lower()
        if not keyword:
            continue
        
        try:
            pattern = keyword_to_regex(keyword, use_regex)
            if re.search(pattern, text, re.IGNORECASE):
                return True
        except re.error:
            # Invalid regex, try literal match
            if keyword in text:
                return True
    
    return False


# =============================================================================
# CONCEPT CLASS
# =============================================================================

class Concept:
    """Represents a concept definition."""
    
    def __init__(self, name: str, keywords: List[str]):
        self.name = name
        self.keywords = keywords
    
    def to_dict(self) -> Dict:
        return {"name": self.name, "keywords": self.keywords}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Concept':
        return cls(data["name"], data["keywords"])
    
    def __str__(self):
        return f"{self.name}: {'; '.join(self.keywords)}"


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWConceptBuilder(OWWidget):
    """Create binary concept variables from keywords."""
    
    name = "Concept Builder"
    description = "Create binary concept variables from keywords"
    icon = "icons/concept_builder.svg"
    priority = 65
    keywords = ["concept", "keyword", "binary", "variable", "text", "search"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        data = Output("Data", Table, doc="Data with concept variables")
        concept_matches = Output("Concept Matches", Table, doc="Documents matching any concept")
    
    # Settings
    field_index = settings.Setting(0)
    use_regex = settings.Setting(False)
    concepts_json = settings.Setting("[]")
    auto_apply = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_field = Msg("Selected text field not found in data")
        no_concepts = Msg("No concepts defined")
        compute_error = Msg("Computation error: {}")
    
    class Warning(OWWidget.Warning):
        field_not_found = Msg("Field '{}' not found, using auto-detect")
    
    class Information(OWWidget.Information):
        concepts_applied = Msg("Applied {} concepts to {} documents")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._concepts: List[Concept] = []
        
        # Load saved concepts
        self._load_concepts_from_settings()
        
        self._setup_control_area()
        self._setup_main_area()
        self._update_concepts_list()
    
    def _load_concepts_from_settings(self):
        """Load concepts from settings JSON."""
        try:
            data = json.loads(self.concepts_json)
            self._concepts = [Concept.from_dict(c) for c in data]
        except:
            self._concepts = []
    
    def _save_concepts_to_settings(self):
        """Save concepts to settings JSON."""
        self.concepts_json = json.dumps([c.to_dict() for c in self._concepts])
    
    def _setup_control_area(self):
        """Build control area."""
        # Info box
        info_box = gui.widgetBox(self.controlArea, "")
        info_label = QLabel(
            "<b>Create binary concept variables</b><br>"
            "<small>Define concepts using keywords.<br>"
            "Use * as wildcard (e.g., govern* matches<br>"
            "government, governance, governing).</small>"
        )
        info_label.setStyleSheet("color: #0277bd; background-color: #e1f5fe; padding: 8px; border-radius: 4px;")
        info_box.layout().addWidget(info_label)
        
        # Text Source
        source_box = gui.widgetBox(self.controlArea, "📄 Text Source")
        
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Search in:"))
        self.field_combo = QComboBox()
        for name, _ in TEXT_FIELD_OPTIONS:
            self.field_combo.addItem(name)
        self.field_combo.setCurrentIndex(self.field_index)
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        field_layout.addWidget(self.field_combo)
        source_box.layout().addLayout(field_layout)
        
        gui.checkBox(source_box, self, "use_regex", "Use regular expressions",
                     callback=self._on_settings_changed)
        
        # Define Concept
        define_box = gui.widgetBox(self.controlArea, "✏️ Define Concept")
        
        # Load from file
        file_layout = QHBoxLayout()
        self.load_btn = QPushButton("📂 Load from File")
        self.load_btn.clicked.connect(self._load_from_file)
        file_layout.addWidget(self.load_btn)
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #6c757d; font-size: 10px;")
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        define_box.layout().addLayout(file_layout)
        
        # Manual definition
        manual_label = QLabel("Or define manually:")
        manual_label.setStyleSheet("margin-top: 8px;")
        define_box.layout().addWidget(manual_label)
        
        # Concept name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Concept name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Sustainability")
        name_layout.addWidget(self.name_edit)
        define_box.layout().addLayout(name_layout)
        
        # Keywords
        kw_layout = QHBoxLayout()
        kw_layout.addWidget(QLabel("Keywords:"))
        self.keywords_edit = QLineEdit()
        self.keywords_edit.setPlaceholderText("e.g., sustainab*; green; eco-friendly")
        kw_layout.addWidget(self.keywords_edit)
        define_box.layout().addLayout(kw_layout)
        
        hint_label = QLabel("<small>(separate keywords with semicolon ;)</small>")
        hint_label.setStyleSheet("color: #6c757d;")
        define_box.layout().addWidget(hint_label)
        
        # Add button
        self.add_btn = QPushButton("➕ Add Concept")
        self.add_btn.setStyleSheet("background-color: #4a90d9; color: white; font-weight: bold; padding: 8px;")
        self.add_btn.clicked.connect(self._add_concept)
        define_box.layout().addWidget(self.add_btn)
        
        # Apply button
        self.apply_btn = gui.button(
            self.controlArea, self, "Apply Concepts",
            callback=self.commit, autoDefault=False
        )
        self.apply_btn.setMinimumHeight(35)
        
        gui.checkBox(self.controlArea, self, "auto_apply", "Auto apply")
        
        self.controlArea.layout().addStretch(1)
    
    def _setup_main_area(self):
        """Build main area."""
        main_layout = QVBoxLayout()
        self.mainArea.layout().addLayout(main_layout)
        
        # Defined Concepts section
        concepts_box = QGroupBox("📋 Defined Concepts")
        concepts_layout = QVBoxLayout(concepts_box)
        
        self.concepts_list = QListWidget()
        self.concepts_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.concepts_list.itemDoubleClicked.connect(self._edit_concept)
        concepts_layout.addWidget(self.concepts_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.edit_btn = QPushButton("✏️ Edit")
        self.edit_btn.clicked.connect(self._edit_concept)
        btn_layout.addWidget(self.edit_btn)
        
        self.remove_btn = QPushButton("🗑️ Remove")
        self.remove_btn.clicked.connect(self._remove_concept)
        btn_layout.addWidget(self.remove_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self._clear_concepts)
        btn_layout.addWidget(self.clear_btn)
        
        concepts_layout.addLayout(btn_layout)
        
        # Save/Export
        export_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 Save to File")
        self.save_btn.clicked.connect(self._save_to_file)
        export_layout.addWidget(self.save_btn)
        export_layout.addStretch()
        concepts_layout.addLayout(export_layout)
        
        main_layout.addWidget(concepts_box)
        
        # Results preview
        results_box = QGroupBox("📊 Results Preview")
        results_layout = QVBoxLayout(results_box)
        
        self.results_label = QLabel("No results yet")
        self.results_label.setStyleSheet("color: #6c757d;")
        results_layout.addWidget(self.results_label)
        
        self.results_table = QTableWidget()
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setMaximumHeight(200)
        results_layout.addWidget(self.results_table)
        
        main_layout.addWidget(results_box)
        
        main_layout.addStretch()
    
    def _on_field_changed(self, index):
        self.field_index = index
        if self.auto_apply and self._df is not None and self._concepts:
            self.commit()
    
    def _on_settings_changed(self):
        if self.auto_apply and self._df is not None and self._concepts:
            self.commit()
    
    def _update_concepts_list(self):
        """Update the concepts list widget."""
        self.concepts_list.clear()
        
        for concept in self._concepts:
            item = QListWidgetItem(str(concept))
            item.setData(Qt.UserRole, concept)
            self.concepts_list.addItem(item)
        
        # Update button states
        has_concepts = len(self._concepts) > 0
        self.edit_btn.setEnabled(has_concepts)
        self.remove_btn.setEnabled(has_concepts)
        self.clear_btn.setEnabled(has_concepts)
        self.save_btn.setEnabled(has_concepts)
    
    def _add_concept(self):
        """Add a new concept from the input fields."""
        name = self.name_edit.text().strip()
        keywords_text = self.keywords_edit.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a concept name.")
            return
        
        if not keywords_text:
            QMessageBox.warning(self, "Missing Keywords", "Please enter at least one keyword.")
            return
        
        # Parse keywords
        keywords = [k.strip() for k in keywords_text.split(';') if k.strip()]
        
        if not keywords:
            QMessageBox.warning(self, "Invalid Keywords", "Please enter valid keywords separated by semicolons.")
            return
        
        # Check for duplicate name
        for concept in self._concepts:
            if concept.name.lower() == name.lower():
                QMessageBox.warning(self, "Duplicate Name", f"A concept named '{name}' already exists.")
                return
        
        # Add concept
        concept = Concept(name, keywords)
        self._concepts.append(concept)
        self._save_concepts_to_settings()
        self._update_concepts_list()
        
        # Clear inputs
        self.name_edit.clear()
        self.keywords_edit.clear()
        
        if self.auto_apply and self._df is not None:
            self.commit()
    
    def _edit_concept(self, item=None):
        """Edit selected concept."""
        if item is None:
            item = self.concepts_list.currentItem()
        
        if item is None:
            return
        
        concept = item.data(Qt.UserRole)
        if concept is None:
            return
        
        # Populate fields
        self.name_edit.setText(concept.name)
        self.keywords_edit.setText('; '.join(concept.keywords))
        
        # Remove the concept (will be re-added when user clicks Add)
        self._concepts.remove(concept)
        self._save_concepts_to_settings()
        self._update_concepts_list()
    
    def _remove_concept(self):
        """Remove selected concept."""
        item = self.concepts_list.currentItem()
        if item is None:
            return
        
        concept = item.data(Qt.UserRole)
        if concept is None:
            return
        
        self._concepts.remove(concept)
        self._save_concepts_to_settings()
        self._update_concepts_list()
        
        if self.auto_apply and self._df is not None:
            self.commit()
    
    def _clear_concepts(self):
        """Clear all concepts."""
        if not self._concepts:
            return
        
        reply = QMessageBox.question(
            self, "Clear All Concepts",
            "Are you sure you want to remove all concepts?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._concepts = []
            self._save_concepts_to_settings()
            self._update_concepts_list()
            
            if self.auto_apply and self._df is not None:
                self.commit()
    
    def _load_from_file(self):
        """Load concepts from JSON file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Concepts", "",
            "JSON files (*.json);;Text files (*.txt);;All files (*)"
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try JSON format first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'name' in item and 'keywords' in item:
                            concept = Concept.from_dict(item)
                            # Check for duplicates
                            if not any(c.name.lower() == concept.name.lower() for c in self._concepts):
                                self._concepts.append(concept)
                
                self.file_label.setText(Path(filepath).name)
                
            except json.JSONDecodeError:
                # Try text format: name: keyword1; keyword2; keyword3
                lines = content.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        name, kw_part = line.split(':', 1)
                        name = name.strip()
                        keywords = [k.strip() for k in kw_part.split(';') if k.strip()]
                        if name and keywords:
                            if not any(c.name.lower() == name.lower() for c in self._concepts):
                                self._concepts.append(Concept(name, keywords))
                
                self.file_label.setText(Path(filepath).name)
            
            self._save_concepts_to_settings()
            self._update_concepts_list()
            
            if self.auto_apply and self._df is not None:
                self.commit()
            
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load file: {e}")
    
    def _save_to_file(self):
        """Save concepts to JSON file."""
        if not self._concepts:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Concepts", "concepts.json",
            "JSON files (*.json);;All files (*)"
        )
        
        if not filepath:
            return
        
        try:
            data = [c.to_dict() for c in self._concepts]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, "Saved", f"Concepts saved to {filepath}")
            
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not save file: {e}")
    
    def _find_text_column(self) -> Optional[str]:
        """Find the text column to search in."""
        if self._df is None:
            return None
        
        _, candidates = TEXT_FIELD_OPTIONS[self.field_index]
        
        if candidates is None:
            # Auto-detect
            for field_candidates in AUTO_DETECT_PRIORITY:
                for col in self._df.columns:
                    if col in field_candidates:
                        return col
                    for candidate in field_candidates:
                        if col.lower() == candidate.lower():
                            return col
            return None
        else:
            for col in self._df.columns:
                if col in candidates:
                    return col
                for candidate in candidates:
                    if col.lower() == candidate.lower():
                        return col
            return None
    
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
        
        self._df = self._table_to_df(data)
        
        if self.auto_apply and self._concepts:
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
        self.results_label.setText("No results yet")
        self.results_table.clear()
        self.results_table.setRowCount(0)
    
    def commit(self):
        """Apply concepts to data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self.Error.no_data()
            self._send_outputs(None, None)
            return
        
        if not self._concepts:
            self.Error.no_concepts()
            self._send_outputs(None, None)
            return
        
        # Find text column
        text_col = self._find_text_column()
        if text_col is None:
            self.Error.no_field()
            self._send_outputs(None, None)
            return
        
        try:
            # Create concept columns
            concept_data = {}
            any_match = np.zeros(len(self._df), dtype=bool)
            
            for concept in self._concepts:
                matches = np.array([
                    matches_concept(text, concept.keywords, self.use_regex)
                    for text in self._df[text_col]
                ])
                concept_data[concept.name] = matches.astype(int)
                any_match |= matches
            
            # Update results display
            self._update_results(concept_data)
            
            # Build output table
            output_table = self._build_output_table(concept_data)
            
            # Build matches table
            match_indices = np.where(any_match)[0].tolist()
            matches_table = self._data[match_indices] if match_indices else None
            
            n_matches = sum(any_match)
            self.Information.concepts_applied(len(self._concepts), n_matches)
            
            self._send_outputs(output_table, matches_table)
            
        except Exception as e:
            import traceback
            logger.error(f"Concept error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
            self._send_outputs(None, None)
    
    def _update_results(self, concept_data: Dict[str, np.ndarray]):
        """Update results display."""
        n_docs = len(self._df)
        
        # Summary
        summary_parts = []
        for name, matches in concept_data.items():
            count = sum(matches)
            pct = count / n_docs * 100 if n_docs > 0 else 0
            summary_parts.append(f"{name}: {count:,} ({pct:.1f}%)")
        
        self.results_label.setText(f"Matches in {n_docs:,} documents")
        
        # Table
        self.results_table.clear()
        self.results_table.setRowCount(len(concept_data))
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(['Concept', 'Matches', 'Percentage', 'Keywords'])
        
        for i, (concept, matches) in enumerate(zip(self._concepts, concept_data.values())):
            count = sum(matches)
            pct = count / n_docs * 100 if n_docs > 0 else 0
            
            self.results_table.setItem(i, 0, QTableWidgetItem(concept.name))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{count:,}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{pct:.1f}%"))
            self.results_table.setItem(i, 3, QTableWidgetItem('; '.join(concept.keywords[:5])))
        
        self.results_table.resizeColumnsToContents()
    
    def _build_output_table(self, concept_data: Dict[str, np.ndarray]) -> Table:
        """Build output table with concept variables."""
        # Create new variables for concepts
        concept_vars = [
            DiscreteVariable(name, values=['No', 'Yes'])
            for name in concept_data.keys()
        ]
        
        # Build new domain
        new_domain = Domain(
            list(self._data.domain.attributes) + concept_vars,
            self._data.domain.class_vars,
            self._data.domain.metas
        )
        
        # Build X array
        n_rows = len(self._df)
        n_orig_attrs = len(self._data.domain.attributes)
        n_concepts = len(concept_data)
        
        new_X = np.zeros((n_rows, n_orig_attrs + n_concepts))
        new_X[:, :n_orig_attrs] = self._data.X
        
        for i, matches in enumerate(concept_data.values()):
            new_X[:, n_orig_attrs + i] = matches
        
        # Create new table
        new_table = Table.from_numpy(
            new_domain,
            new_X,
            self._data.Y if self._data.Y.size > 0 else None,
            self._data.metas if self._data.metas.size > 0 else None
        )
        
        return new_table
    
    def _send_outputs(self, data: Optional[Table], matches: Optional[Table]):
        """Send outputs."""
        self.Outputs.data.send(data)
        self.Outputs.concept_matches.send(matches)


if __name__ == "__main__":
    WidgetPreview(OWConceptBuilder).run()
