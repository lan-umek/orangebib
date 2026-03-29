# -*- coding: utf-8 -*-
"""
PA Concepts Widget
==================
Create Public Administration paradigm indicators.

Specialized widget for identifying Public Administration research paradigms
(Weber, NPM, Good Governance) in bibliographic data.
"""

import logging
import re
import json
from typing import Optional, Dict, List, Set
from pathlib import Path

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QFrame, QLineEdit,
    QFileDialog, QMessageBox, QListWidget, QListWidgetItem,
    QAbstractItemView,
)
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QFont, QColor

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# PA CONCEPT DEFINITIONS (from PA_concepts.xlsx)
# =============================================================================

# Default PA concepts from biblium additional files
# Format: column name -> list of keywords (rows in that column)
DEFAULT_PA_CONCEPTS = {
    "Weber": ["weberian", "neo-weberian", "law"],
    "Npm": ["new public management", "npm"],
    "Good Governance": ["good governance", "gg", "transparency"],
}


# =============================================================================
# TEXT FIELD OPTIONS
# =============================================================================

TEXT_FIELD_OPTIONS = [
    ("Auto-detect", None),
    ("Abstract", ["Abstract", "AB", "abstract"]),
    ("Title", ["Title", "TI", "title", "Document Title"]),
    ("Author Keywords", ["Author Keywords", "Keywords", "DE", "author_keywords"]),
    ("Index Keywords", ["Index Keywords", "Keywords Plus", "ID", "indexed_keywords"]),
    ("Processed Text", ["Processed Text", "processed_text", "Text", "Combined Text"]),
]

# Auto-detect priority
AUTO_DETECT_PRIORITY = [
    ["Abstract", "AB", "abstract"],
    ["Processed Text", "Combined Text", "processed_text"],
    ["Title", "TI", "title", "Document Title"],
    ["Author Keywords", "Keywords", "DE", "author_keywords"],
]


# =============================================================================
# CONCEPT MATCHING
# =============================================================================

def keyword_to_pattern(keyword: str, use_regex: bool = False) -> str:
    """Convert keyword to regex pattern."""
    if use_regex:
        return keyword
    else:
        escaped = re.escape(keyword.lower())
        return r'\b' + escaped + r'\b'


def matches_concept(text: str, keywords: List[str], use_regex: bool = False) -> bool:
    """Check if text matches any keyword."""
    if not text or pd.isna(text):
        return False
    
    text_lower = str(text).lower()
    
    for keyword in keywords:
        keyword = keyword.strip().lower()
        if not keyword:
            continue
        
        try:
            pattern = keyword_to_pattern(keyword, use_regex)
            if re.search(pattern, text_lower):
                return True
        except re.error:
            if keyword in text_lower:
                return True
    
    return False


# =============================================================================
# NUMERIC TABLE ITEM FOR SORTING
# =============================================================================

class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically."""
    
    def __init__(self, display_text: str, sort_value: float):
        super().__init__(display_text)
        self._sort_value = sort_value
    
    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWPAConcepts(OWWidget):
    """Create Public Administration paradigm indicators."""
    
    name = "PA Concepts"
    description = "Create Public Administration paradigm indicators"
    icon = "icons/pa_concepts.svg"
    priority = 67
    keywords = ["PA", "public administration", "weber", "npm", "governance", "paradigm"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        data = Output("Data", Table, doc="Data with PA concept variables")
        summary = Output("Summary", Table, doc="Summary of PA concept distribution")
        pa_documents = Output("PA Documents", Table, doc="Documents with PA concepts")
    
    # Settings
    field_index = settings.Setting(0)
    use_regex = settings.Setting(False)
    use_numeric_labels = settings.Setting(True)
    selected_concepts = settings.Setting(["Weber", "Npm", "Good Governance"])
    custom_concepts_path = settings.Setting("")
    auto_apply = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_field = Msg("Selected text field not found in data")
        no_concepts = Msg("No concepts selected")
        compute_error = Msg("Computation error: {}")
    
    class Warning(OWWidget.Warning):
        low_coverage = Msg("Only {:.1f}% of documents have PA concept matches")
    
    class Information(OWWidget.Information):
        identified = Msg("Identified PA concepts in {:,} of {:,} documents ({:.1f}%)")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._pa_concepts: Dict[str, List[str]] = DEFAULT_PA_CONCEPTS.copy()
        self._results: Optional[pd.DataFrame] = None
        
        self._setup_control_area()
        self._setup_main_area()
    
    def _setup_control_area(self):
        """Build control area."""
        # Info header
        info_box = gui.widgetBox(self.controlArea, "")
        info_label = QLabel(
            "<b>🏛️ PA Concepts</b><br>"
            "<small>Create Public Administration paradigm indicators</small>"
        )
        info_label.setStyleSheet("color: #1565c0; background-color: #e3f2fd; padding: 8px; border-radius: 4px;")
        info_box.layout().addWidget(info_label)
        
        # Text Source
        text_box = gui.widgetBox(self.controlArea, "📄 Text Source")
        
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Search in:"))
        self.field_combo = QComboBox()
        for name, _ in TEXT_FIELD_OPTIONS:
            self.field_combo.addItem(name)
        self.field_combo.setCurrentIndex(self.field_index)
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        field_layout.addWidget(self.field_combo)
        text_box.layout().addLayout(field_layout)
        
        gui.checkBox(text_box, self, "use_regex", "Use regular expressions")
        
        # Concept Selection
        concept_box = gui.widgetBox(self.controlArea, "🏛️ Select PA Concepts")
        
        avail_label = QLabel(f"Available concepts ({len(self._pa_concepts)}):")
        avail_label.setStyleSheet("font-weight: bold;")
        concept_box.layout().addWidget(avail_label)
        self._avail_label = avail_label
        
        # Concept list with checkboxes
        self.concept_list = QListWidget()
        self.concept_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._populate_concept_list()
        concept_box.layout().addWidget(self.concept_list)
        
        # Select All / None buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        self.select_all_btn.setStyleSheet("background-color: #4a90d9; color: white;")
        btn_layout.addWidget(self.select_all_btn)
        
        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self._select_none)
        self.select_none_btn.setStyleSheet("background-color: #4a90d9; color: white;")
        btn_layout.addWidget(self.select_none_btn)
        concept_box.layout().addLayout(btn_layout)
        
        # Keywords Preview (collapsible)
        self._preview_visible = False
        self.preview_btn = QPushButton("▶ Keywords Preview")
        self.preview_btn.setFlat(True)
        self.preview_btn.setStyleSheet("text-align: left; font-weight: bold;")
        self.preview_btn.clicked.connect(self._toggle_preview)
        self.controlArea.layout().addWidget(self.preview_btn)
        
        self._preview_box = gui.widgetBox(self.controlArea, "")
        self.preview_text = QLabel()
        self.preview_text.setWordWrap(True)
        self.preview_text.setStyleSheet("font-size: 10px; color: #666; padding: 8px; background: #f5f5f5; border-radius: 4px;")
        self._preview_box.layout().addWidget(self.preview_text)
        self._preview_box.setVisible(False)
        self._update_preview()
        
        # Custom Concepts File
        custom_box = gui.widgetBox(self.controlArea, "📁 Custom Concepts")
        
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit("(default)")
        self.file_edit.setReadOnly(True)
        file_layout.addWidget(self.file_edit)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_concepts)
        file_layout.addWidget(browse_btn)
        custom_box.layout().addLayout(file_layout)
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self._reset_concepts)
        custom_box.layout().addWidget(reset_btn)
        
        # Output Options
        output_box = gui.widgetBox(self.controlArea, "⚙️ Output Options")
        gui.checkBox(output_box, self, "use_numeric_labels", "Use numeric labels (0/1)")
        
        # Action button
        self.create_btn = gui.button(
            self.controlArea, self, "🏛️ Create PA Concept Variables",
            callback=self.commit, autoDefault=False
        )
        self.create_btn.setMinimumHeight(40)
        self.create_btn.setStyleSheet("background-color: #4a90d9; color: white; font-weight: bold;")
        
        gui.checkBox(self.controlArea, self, "auto_apply", "Auto apply")
        
        self.controlArea.layout().addStretch(1)
    
    def _populate_concept_list(self):
        """Populate the concept list with checkboxes."""
        self.concept_list.clear()
        self._concept_items = {}
        
        for concept, keywords in self._pa_concepts.items():
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if concept in self.selected_concepts else Qt.Unchecked)
            item.setText(f"{concept}  ({len(keywords)} keywords)")
            item.setData(Qt.UserRole, concept)
            self.concept_list.addItem(item)
            self._concept_items[concept] = item
        
        self.concept_list.itemChanged.connect(self._on_concept_toggled)
        self._avail_label.setText(f"Available concepts ({len(self._pa_concepts)}):")
    
    def _browse_concepts(self):
        """Browse for custom PA concepts file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load PA Concepts", "",
            "Excel files (*.xlsx);;All files (*)"
        )
        
        if not filepath:
            return
        
        try:
            df = pd.read_excel(filepath)
            
            # Build concepts from columns
            new_concepts = {}
            for col in df.columns:
                keywords = df[col].dropna().astype(str).str.strip().tolist()
                keywords = [k for k in keywords if k and k.lower() != 'nan']
                if keywords:
                    # Capitalize concept name
                    concept_name = col.title()
                    new_concepts[concept_name] = keywords
            
            if not new_concepts:
                QMessageBox.warning(self, "Invalid File", "No valid concepts found in file.")
                return
            
            self._pa_concepts = new_concepts
            self.custom_concepts_path = filepath
            self.file_edit.setText(Path(filepath).name)
            
            # Update concept list
            self.concept_list.itemChanged.disconnect()
            self._populate_concept_list()
            self.selected_concepts = list(new_concepts.keys())
            self._update_preview()
            
            QMessageBox.information(
                self, "Loaded",
                f"Loaded {len(new_concepts)} concepts:\n" + 
                "\n".join([f"  • {c} ({len(kw)} keywords)" for c, kw in new_concepts.items()])
            )
            
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load file: {e}")
    
    def _reset_concepts(self):
        """Reset to default PA concepts."""
        self._pa_concepts = DEFAULT_PA_CONCEPTS.copy()
        self.custom_concepts_path = ""
        self.file_edit.setText("(default)")
        
        # Update concept list
        self.concept_list.itemChanged.disconnect()
        self._populate_concept_list()
        self.selected_concepts = list(DEFAULT_PA_CONCEPTS.keys())
        self._update_preview()
    
    def _setup_main_area(self):
        """Build main area."""
        main_layout = QVBoxLayout()
        self.mainArea.layout().addLayout(main_layout)
        
        # Summary
        self.summary_label = QLabel("Load data and select concepts to analyze")
        self.summary_label.setStyleSheet("font-size: 14px; color: #6c757d;")
        main_layout.addWidget(self.summary_label)
        
        # Results table
        results_box = QGroupBox("📊 PA Concept Distribution")
        results_layout = QVBoxLayout(results_box)
        
        self.results_table = QTableWidget()
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSortingEnabled(True)
        results_layout.addWidget(self.results_table)
        
        main_layout.addWidget(results_box)
        
        # Export
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_btn = QPushButton("📥 Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        main_layout.addLayout(export_layout)
    
    def _toggle_preview(self):
        """Toggle keywords preview visibility."""
        self._preview_visible = not self._preview_visible
        self._preview_box.setVisible(self._preview_visible)
        self.preview_btn.setText("▼ Keywords Preview" if self._preview_visible else "▶ Keywords Preview")
    
    def _update_preview(self):
        """Update keywords preview text."""
        lines = []
        for concept, keywords in self._pa_concepts.items():
            kw_text = ", ".join(keywords[:8])
            if len(keywords) > 8:
                kw_text += f"... (+{len(keywords)-8} more)"
            lines.append(f"<b>{concept}:</b> {kw_text}")
        self.preview_text.setText("<br>".join(lines))
    
    def _on_field_changed(self, index):
        self.field_index = index
        if self.auto_apply and self._df is not None:
            self.commit()
    
    def _on_concept_toggled(self, item):
        """Update selected concepts list from list widget."""
        self.selected_concepts = []
        for i in range(self.concept_list.count()):
            item = self.concept_list.item(i)
            if item.checkState() == Qt.Checked:
                concept = item.data(Qt.UserRole)
                self.selected_concepts.append(concept)
        
        if self.auto_apply and self._df is not None:
            self.commit()
    
    def _select_all(self):
        """Select all concepts."""
        self.concept_list.itemChanged.disconnect()
        for i in range(self.concept_list.count()):
            self.concept_list.item(i).setCheckState(Qt.Checked)
        self.selected_concepts = list(self._pa_concepts.keys())
        self.concept_list.itemChanged.connect(self._on_concept_toggled)
        if self.auto_apply and self._df is not None:
            self.commit()
    
    def _select_none(self):
        """Deselect all concepts."""
        self.concept_list.itemChanged.disconnect()
        for i in range(self.concept_list.count()):
            self.concept_list.item(i).setCheckState(Qt.Unchecked)
        self.selected_concepts = []
        self.concept_list.itemChanged.connect(self._on_concept_toggled)
        if self.auto_apply and self._df is not None:
            self.commit()
    
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
        self._results = None
        
        self._clear_results()
        
        if data is None:
            self.Error.no_data()
            return
        
        self._df = self._table_to_df(data)
        
        if self.auto_apply and self.selected_concepts:
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
        self.summary_label.setText("Load data and select concepts to analyze")
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.export_btn.setEnabled(False)
    
    def commit(self):
        """Create PA concept variables."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self.Error.no_data()
            self._send_outputs(None, None, None)
            return
        
        if not self.selected_concepts:
            self.Error.no_concepts()
            self._send_outputs(None, None, None)
            return
        
        text_col = self._find_text_column()
        if text_col is None:
            self.Error.no_field()
            self._send_outputs(None, None, None)
            return
        
        try:
            # Identify concepts for each document
            concept_results = {}
            any_match = np.zeros(len(self._df), dtype=bool)
            
            for concept in self.selected_concepts:
                if concept not in self._pa_concepts:
                    continue
                
                keywords = self._pa_concepts[concept]
                matches = np.array([
                    matches_concept(text, keywords, self.use_regex)
                    for text in self._df[text_col]
                ])
                concept_results[concept] = matches.astype(int)
                any_match |= matches
            
            # Store results
            results_df = pd.DataFrame(concept_results)
            results_df["PA_Count"] = results_df.sum(axis=1)
            results_df["Has_PA"] = (results_df["PA_Count"] > 0).astype(int)
            self._results = results_df
            
            # Update display
            self._update_results_display()
            
            # Send outputs
            self._send_outputs_from_results()
            
        except Exception as e:
            import traceback
            logger.error(f"PA concepts error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
            self._send_outputs(None, None, None)
    
    def _update_results_display(self):
        """Update results table."""
        if self._results is None:
            return
        
        n_total = len(self._results)
        n_with_pa = self._results["Has_PA"].sum()
        pct = n_with_pa / n_total * 100 if n_total > 0 else 0
        
        self.summary_label.setText(
            f"<b>{n_with_pa:,}</b> of <b>{n_total:,}</b> documents ({pct:.1f}%) have PA concept matches"
        )
        
        self.Information.identified(n_with_pa, n_total, pct)
        
        if pct < 5:
            self.Warning.low_coverage(pct)
        
        # Disable sorting while populating
        self.results_table.setSortingEnabled(False)
        
        # Build table
        concepts = [c for c in self.selected_concepts if c in self._results.columns]
        self.results_table.clear()
        self.results_table.setRowCount(len(concepts))
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Concept", "Keywords", "Documents", "Percentage"])
        
        for i, concept in enumerate(concepts):
            count = int(self._results[concept].sum())
            pct_concept = count / n_total * 100 if n_total > 0 else 0
            n_keywords = len(self._pa_concepts.get(concept, []))
            
            # Concept name
            self.results_table.setItem(i, 0, QTableWidgetItem(concept))
            
            # Keywords count
            self.results_table.setItem(i, 1, NumericTableWidgetItem(str(n_keywords), n_keywords))
            
            # Documents count
            self.results_table.setItem(i, 2, NumericTableWidgetItem(f"{count:,}", count))
            
            # Percentage
            self.results_table.setItem(i, 3, NumericTableWidgetItem(f"{pct_concept:.1f}%", pct_concept))
        
        self.results_table.resizeColumnsToContents()
        self.results_table.setSortingEnabled(True)
        self.export_btn.setEnabled(True)
    
    def _send_outputs_from_results(self):
        """Send outputs."""
        if self._results is None or self._data is None:
            self._send_outputs(None, None, None)
            return
        
        # Determine label values
        label_values = ['0', '1'] if self.use_numeric_labels else ['No', 'Yes']
        
        # Create concept variables
        concept_vars = []
        for concept in self.selected_concepts:
            if concept in self._results.columns:
                var = DiscreteVariable(concept, values=label_values)
                concept_vars.append(var)
        
        # Add count and has_pa
        count_var = ContinuousVariable("PA_Count")
        has_pa_var = DiscreteVariable("Has_PA", values=label_values)
        
        # Build new domain
        new_domain = Domain(
            list(self._data.domain.attributes) + concept_vars + [count_var, has_pa_var],
            self._data.domain.class_vars,
            self._data.domain.metas
        )
        
        # Build X array
        n_rows = len(self._df)
        n_orig = len(self._data.domain.attributes)
        n_new = len(concept_vars) + 2
        
        new_X = np.zeros((n_rows, n_orig + n_new))
        new_X[:, :n_orig] = self._data.X
        
        col_idx = n_orig
        for concept in self.selected_concepts:
            if concept in self._results.columns:
                new_X[:, col_idx] = self._results[concept].values
                col_idx += 1
        
        new_X[:, col_idx] = self._results["PA_Count"].values
        col_idx += 1
        new_X[:, col_idx] = self._results["Has_PA"].values
        
        output_table = Table.from_numpy(
            new_domain, new_X,
            self._data.Y if self._data.Y.size > 0 else None,
            self._data.metas if self._data.metas.size > 0 else None
        )
        
        # Summary table
        summary_data = []
        n_total = len(self._results)
        for concept in self.selected_concepts:
            if concept in self._results.columns:
                count = int(self._results[concept].sum())
                pct = count / n_total * 100 if n_total > 0 else 0
                summary_data.append({
                    "Concept": concept,
                    "Keywords": len(self._pa_concepts.get(concept, [])),
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
        
        # PA Documents
        pa_indices = np.where(self._results["Has_PA"] == 1)[0].tolist()
        pa_docs_table = output_table[pa_indices] if pa_indices else None
        
        self._send_outputs(output_table, summary_table, pa_docs_table)
    
    def _send_outputs(self, data: Optional[Table], summary: Optional[Table], pa_docs: Optional[Table]):
        """Send outputs."""
        self.Outputs.data.send(data)
        self.Outputs.summary.send(summary)
        self.Outputs.pa_documents.send(pa_docs)
    
    def _export_results(self):
        """Export results to CSV."""
        if self._results is None:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export PA Concept Results", "pa_concepts.csv",
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
    WidgetPreview(OWPAConcepts).run()
