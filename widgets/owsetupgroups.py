# -*- coding: utf-8 -*-
"""
Setup Groups Widget for Orange3-Biblium.

Provides comprehensive document grouping methods for comparative bibliometric analysis.
Uses Biblium's generate_group_matrix for flexible group definitions.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, pyqtSignal
from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QLineEdit, QTextEdit,
    QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem,
    QTabWidget, QGroupBox, QCheckBox, QProgressBar, QHeaderView,
    QSplitter, QScrollArea, QFrame, QSizePolicy, QStackedWidget,
    QListWidget, QListWidgetItem, QAbstractItemView, QFileDialog
)

from Orange.data import Table, Domain, StringVariable, ContinuousVariable, DiscreteVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output, Msg

logger = logging.getLogger(__name__)

# Try to import biblium
try:
    from biblium import utilsbib
    from biblium.bibstats import BiblioStats
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    logger.info("Biblium not available - install with: pip install biblium")


class GroupPreviewTable(QTableWidget):
    """Table widget for group preview."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.horizontalHeader().setStretchLastSection(True)
        self.setStyleSheet("""
            QTableWidget {
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                font-weight: bold;
            }
        """)


class ConceptEditor(QWidget):
    """Editor for defining concept-based groups."""
    
    changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Instructions
        instr = QLabel(
            "<b>Define Groups by Concepts/Terms</b><br>"
            "<small>Format: <code>GroupName: term1, term2, term3</code><br>"
            "Use * as wildcard (e.g., <code>machine*</code> matches machine learning)</small>"
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)
        
        # Text editor
        self.editor = QTextEdit()
        self.editor.setPlaceholderText(
            "AI Methods: machine learning, deep learning, neural network*\n"
            "Big Data: big data, data mining, analytics\n"
            "IoT: internet of things, IoT, smart devices"
        )
        self.editor.setMinimumHeight(120)
        self.editor.textChanged.connect(self.changed.emit)
        layout.addWidget(self.editor)
    
    def get_group_dict(self) -> Dict[str, List[str]]:
        """Parse editor text into group dictionary."""
        groups = {}
        text = self.editor.toPlainText()
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            parts = line.split(':', 1)
            group_name = parts[0].strip()
            terms = [t.strip() for t in parts[1].split(',') if t.strip()]
            
            if group_name and terms:
                groups[group_name] = terms
        
        return groups
    
    def set_group_dict(self, groups: Dict[str, List[str]]):
        """Set editor text from group dictionary."""
        lines = []
        for name, terms in groups.items():
            lines.append(f"{name}: {', '.join(terms)}")
        self.editor.setPlainText('\n'.join(lines))


class OWSetupGroups(OWWidget):
    """
    Setup Groups Widget for comparative bibliometric analysis.
    
    Supports multiple grouping methods:
    - By Column: Each unique value becomes a group
    - By Year Periods: Group by time periods
    - By Multi-Item Column: Explode multi-value columns
    - By Clustering: Automatic document clustering
    - By Concept DataFrame: External concept definitions
    - By Dictionary/Regex: Pattern-based grouping
    - Random Groups: For testing/validation
    """
    
    name = "Setup Groups"
    description = "Define document groups for comparative analysis"
    icon = "icons/setup_groups.svg"
    priority = 5
    keywords = ["groups", "comparison", "subgroups", "classification", "clustering"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")
        concepts = Input("Concepts", Table, doc="Concept definitions (optional)")
    
    class Outputs:
        data = Output("Data", Table, doc="Data with group columns")
        group_matrix = Output("Group Matrix", Table, doc="Document × Group binary matrix")
        group_summary = Output("Group Summary", Table, doc="Summary statistics per group")
    
    # Grouping methods
    METHODS = [
        ("By Column", "Each unique value in a column becomes a group"),
        ("By Year Periods", "Group documents by time periods"),
        ("By Multi-Item Column", "Explode multi-value fields (authors, keywords, etc.)"),
        ("By Clustering", "Automatic document clustering"),
        ("By Concept DataFrame", "Use external concept definitions"),
        ("By Dictionary/Regex", "Define groups with search terms/patterns"),
        ("Random Groups", "Create random groups for testing"),
    ]
    
    # Settings
    method_idx = Setting(0)
    column_name = Setting("")
    multiitem_column_name = Setting("")
    year_column = Setting("")
    n_periods = Setting(3)
    use_cutpoints = Setting(False)
    cutpoints_str = Setting("")
    top_n_items = Setting(20)
    separator_idx = Setting(0)  # 0=auto, 1="; ", 2="|", 3=";"
    cluster_method = Setting("kmeans")
    n_clusters = Setting(5)
    auto_clusters = Setting(True)
    text_column = Setting("")
    concept_text = Setting("")
    n_random_groups = Setting(3)
    include_items_str = Setting("")
    exclude_items_str = Setting("")
    whole_word_match = Setting(False)
    auto_apply = Setting(True)
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium library not installed")
        no_column = Msg("Selected column not found in data")
        grouping_failed = Msg("Grouping failed: {}")
        no_groups_created = Msg("No groups were created")
    
    class Warning(OWWidget.Warning):
        few_groups = Msg("Only {} group(s) created")
        many_groups = Msg("{} groups created - consider filtering")
        empty_groups = Msg("{} empty group(s) removed")
    
    class Information(OWWidget.Information):
        groups_created = Msg("Created {} groups with {} documents")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._concepts_df: Optional[pd.DataFrame] = None
        self._group_matrix: Optional[pd.DataFrame] = None
        self._preview_df: Optional[pd.DataFrame] = None
        
        self._setup_control_area()
        self._setup_main_area()
    
    def _setup_control_area(self):
        """Build control area."""
        # Grouping Method Selection
        method_box = gui.widgetBox(self.controlArea, "🔀 Grouping Method")
        
        self.method_group = QButtonGroup(self)
        
        for i, (name, desc) in enumerate(self.METHODS):
            radio = QRadioButton(name)
            radio.setToolTip(desc)
            if i == self.method_idx:
                radio.setChecked(True)
            self.method_group.addButton(radio, i)
            method_box.layout().addWidget(radio)
        
        self.method_group.buttonClicked.connect(self._on_method_changed)
        
        # Method Options (stacked widget)
        options_box = gui.widgetBox(self.controlArea, "⚙️ Method Options")
        
        self.options_stack = QStackedWidget()
        options_box.layout().addWidget(self.options_stack)
        
        # Create option panels for each method
        self._create_column_options()
        self._create_year_options()
        self._create_multiitem_options()
        self._create_clustering_options()
        self._create_concept_df_options()
        self._create_dict_regex_options()
        self._create_random_options()
        
        # Sync stack to saved method_idx
        self.options_stack.setCurrentIndex(self.method_idx)
        
        # Advanced Options (collapsible)
        advanced_box = gui.widgetBox(self.controlArea, "🔧 Advanced Options")
        advanced_box.setVisible(True)
        
        # Include/Exclude items
        inc_layout = QHBoxLayout()
        inc_layout.addWidget(QLabel("Include:"))
        self.include_edit = QLineEdit()
        self.include_edit.setPlaceholderText("group1, group2, ...")
        self.include_edit.setText(self.include_items_str)
        self.include_edit.textChanged.connect(self._on_filter_changed)
        inc_layout.addWidget(self.include_edit)
        advanced_box.layout().addLayout(inc_layout)
        
        exc_layout = QHBoxLayout()
        exc_layout.addWidget(QLabel("Exclude:"))
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("group3, group4, ...")
        self.exclude_edit.setText(self.exclude_items_str)
        self.exclude_edit.textChanged.connect(self._on_filter_changed)
        exc_layout.addWidget(self.exclude_edit)
        advanced_box.layout().addLayout(exc_layout)
        
        # Whole word matching
        self.whole_word_cb = QCheckBox("Whole word matching (for text search)")
        self.whole_word_cb.setChecked(self.whole_word_match)
        self.whole_word_cb.toggled.connect(self._on_option_changed)
        advanced_box.layout().addWidget(self.whole_word_cb)
        
        # Group Configuration Preview
        config_box = gui.widgetBox(self.controlArea, "📊 Group Configuration")
        
        self.config_label = QLabel("No groups configured")
        self.config_label.setWordWrap(True)
        self.config_label.setStyleSheet("color: #666;")
        config_box.layout().addWidget(self.config_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("👁️ Preview Groups")
        self.preview_btn.clicked.connect(self._preview_groups)
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e7ff;
                border: 1px solid #6366f1;
                border-radius: 4px;
                padding: 8px 16px;
                color: #4338ca;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c7d2fe;
            }
        """)
        btn_layout.addWidget(self.preview_btn)
        
        self.create_btn = QPushButton("✨ Create Groups")
        self.create_btn.clicked.connect(self._create_groups)
        self.create_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        btn_layout.addWidget(self.create_btn)
        
        self.controlArea.layout().addLayout(btn_layout)
        
        # Auto apply checkbox (without gui.auto_apply since we use manual buttons)
        self.auto_apply_cb = QCheckBox("Auto apply on data change")
        self.auto_apply_cb.setChecked(self.auto_apply)
        self.auto_apply_cb.toggled.connect(self._on_auto_apply_changed)
        self.controlArea.layout().addWidget(self.auto_apply_cb)
    
    def _on_auto_apply_changed(self, checked):
        """Handle auto apply checkbox change."""
        self.auto_apply = checked
        if checked and self._df is not None:
            self._create_groups()
    
    def commit(self):
        """Commit changes - alias for _create_groups."""
        self._create_groups()
    
    def _create_column_options(self):
        """Create options panel for By Column method."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        desc = QLabel(
            "<small>Select a column to group by. Each unique value becomes a group.</small>"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        form = QFormLayout()
        self.column_combo = QComboBox()
        self.column_combo.currentTextChanged.connect(self._on_column_changed)
        form.addRow("Column:", self.column_combo)
        
        self.unique_label = QLabel("Unique values: -")
        self.unique_label.setStyleSheet("color: #666;")
        form.addRow("", self.unique_label)
        
        layout.addLayout(form)
        layout.addStretch()
        
        self.options_stack.addWidget(panel)
    
    def _create_year_options(self):
        """Create options panel for By Year Periods method."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        desc = QLabel(
            "<small>Group documents by publication year periods.</small>"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        form = QFormLayout()
        
        self.year_combo = QComboBox()
        self.year_combo.currentTextChanged.connect(self._on_year_column_changed)
        form.addRow("Year Column:", self.year_combo)
        
        self.year_range_label = QLabel("Range: -")
        self.year_range_label.setStyleSheet("color: #666;")
        form.addRow("", self.year_range_label)
        
        # Number of periods
        self.periods_spin = QSpinBox()
        self.periods_spin.setRange(2, 20)
        self.periods_spin.setValue(self.n_periods)
        self.periods_spin.valueChanged.connect(self._on_periods_changed)
        form.addRow("Number of periods:", self.periods_spin)
        
        # Or custom cutpoints
        self.cutpoints_cb = QCheckBox("Use custom cutpoints:")
        self.cutpoints_cb.setChecked(self.use_cutpoints)
        self.cutpoints_cb.toggled.connect(self._on_cutpoints_toggled)
        form.addRow(self.cutpoints_cb)
        
        self.cutpoints_edit = QLineEdit()
        self.cutpoints_edit.setPlaceholderText("e.g., 2000, 2010, 2020")
        self.cutpoints_edit.setText(self.cutpoints_str)
        self.cutpoints_edit.setEnabled(self.use_cutpoints)
        self.cutpoints_edit.textChanged.connect(self._on_option_changed)
        form.addRow("Cutpoints:", self.cutpoints_edit)
        
        layout.addLayout(form)
        layout.addStretch()
        
        self.options_stack.addWidget(panel)
    
    def _create_multiitem_options(self):
        """Create options panel for By Multi-Item Column method."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        desc = QLabel(
            "<small>Explode multi-value columns (e.g., Authors, Keywords). "
            "Each unique item becomes a group.</small>"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        form = QFormLayout()
        
        # Separator selection (determines which DB convention to use)
        self.separator_combo = QComboBox()
        self.separator_combo.addItems([
            'Auto-detect',
            '"; " (Scopus, WoS, PubMed, …)',
            '"| " (OpenAlex)',
            '";" (Scopus areas, fields, …)',
        ])
        self.separator_combo.setCurrentIndex(self.separator_idx)
        self.separator_combo.currentIndexChanged.connect(self._on_separator_changed)
        form.addRow("Separator:", self.separator_combo)
        
        self.detected_sep_label = QLabel("")
        self.detected_sep_label.setStyleSheet("color: #666; font-size: 11px;")
        form.addRow("", self.detected_sep_label)
        
        self.multiitem_combo = QComboBox()
        self.multiitem_combo.currentTextChanged.connect(self._on_multiitem_changed)
        form.addRow("Column:", self.multiitem_combo)
        
        self.items_label = QLabel("Unique items: -")
        self.items_label.setStyleSheet("color: #666;")
        form.addRow("", self.items_label)
        
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 1000)
        self.top_n_spin.setValue(self.top_n_items)
        self.top_n_spin.valueChanged.connect(self._on_option_changed)
        form.addRow("Top N items:", self.top_n_spin)
        
        layout.addLayout(form)
        layout.addStretch()
        
        self.options_stack.addWidget(panel)
    
    # Separator map: index → string
    SEPARATOR_OPTIONS = {0: None, 1: "; ", 2: "|", 3: ";"}
    
    def _get_active_separator(self):
        """Return the active separator string, auto-detecting if needed."""
        idx = self.separator_idx
        if idx != 0:
            return self.SEPARATOR_OPTIONS[idx]
        
        # Auto-detect from data
        return self._detect_separator()
    
    def _detect_separator(self):
        """
        Detect the dominant separator in the data.
        
        Checks well-known multi-item columns and returns the separator
        that matches the most column values.
        """
        if self._df is None:
            return "; "
        
        cols = list(self._df.columns)
        
        # Check a handful of well-known multi-item columns
        probe_cols = [
            "Author Keywords", "Processed Author Keywords",
            "Index Keywords", "Processed Index Keywords",
            "Authors", "Affiliations", "References",
        ]
        probes = [c for c in probe_cols if c in cols]
        if not probes:
            # Fallback: sample first few object columns
            probes = [c for c in cols if self._df[c].dtype == 'object'][:5]
        
        # Count how many probe columns contain each separator
        scores = {"; ": 0, "|": 0, ";": 0}
        for col in probes:
            try:
                sample = self._df[col].dropna().head(200).astype(str)
                for sep in scores:
                    if sample.str.contains(sep, regex=False).mean() > 0.15:
                        scores[sep] += 1
            except Exception:
                pass
        
        # "|" is unambiguous — if it wins, it's OpenAlex-style
        if scores["|"] >= scores["; "] and scores["|"] > 0:
            return "|"
        # "; " (with space) is the standard Scopus/WoS separator
        if scores["; "] > 0:
            return "; "
        # ";" (no space) as last resort
        if scores[";"] > 0:
            return ";"
        return "; "
    
    def _on_separator_changed(self, idx):
        """Handle separator selection change."""
        self.separator_idx = idx
        sep = self._get_active_separator()
        if self.separator_idx == 0:
            self.detected_sep_label.setText(
                f'Detected: "{sep}"'
            )
        else:
            self.detected_sep_label.setText("")
        # Refresh column list and item count with new separator
        self._update_column_combos()
        self._update_items_count()
        if self.method_idx == 2:
            self._auto_create()
    
    def _create_clustering_options(self):
        """Create options panel for By Clustering method."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        desc = QLabel(
            "<small>Automatically cluster documents based on text content. "
            "Requires Biblium's clustering functionality.</small>"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        form = QFormLayout()
        
        # Text column
        self.cluster_text_combo = QComboBox()
        self.cluster_text_combo.currentTextChanged.connect(self._on_option_changed)
        form.addRow("Text Field:", self.cluster_text_combo)
        
        # Clustering method
        self.cluster_method_combo = QComboBox()
        self.cluster_method_combo.addItems(["kmeans", "hierarchical", "spectral"])
        self.cluster_method_combo.setCurrentText(self.cluster_method)
        self.cluster_method_combo.currentTextChanged.connect(self._on_option_changed)
        form.addRow("Method:", self.cluster_method_combo)
        
        # Number of clusters
        self.auto_clusters_cb = QCheckBox("Auto-detect optimal K")
        self.auto_clusters_cb.setChecked(self.auto_clusters)
        self.auto_clusters_cb.toggled.connect(self._on_auto_clusters_toggled)
        form.addRow(self.auto_clusters_cb)
        
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 50)
        self.n_clusters_spin.setValue(self.n_clusters)
        self.n_clusters_spin.setEnabled(not self.auto_clusters)
        self.n_clusters_spin.valueChanged.connect(self._on_option_changed)
        form.addRow("Number of clusters:", self.n_clusters_spin)
        
        layout.addLayout(form)
        layout.addStretch()
        
        self.options_stack.addWidget(panel)
    
    def _create_concept_df_options(self):
        """Create options panel for By Concept DataFrame method."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        desc = QLabel(
            "<small>Define groups with a concept table (Excel/CSV). "
            "Each column = a group; cells = search terms.<br>"
            "Load a file below or connect a table to the <i>Concepts</i> input.</small>"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        form = QFormLayout()
        
        # Text column to search
        self.concept_text_combo = QComboBox()
        self.concept_text_combo.currentTextChanged.connect(self._on_option_changed)
        form.addRow("Search in:", self.concept_text_combo)
        
        layout.addLayout(form)
        
        # File loading section
        file_box = QGroupBox("Concept File")
        file_layout = QVBoxLayout(file_box)
        
        # File path display + browse button
        path_row = QHBoxLayout()
        self.concept_file_label = QLabel("No file loaded")
        self.concept_file_label.setStyleSheet("color: #666; font-size: 11px;")
        self.concept_file_label.setWordWrap(True)
        path_row.addWidget(self.concept_file_label, 1)
        
        browse_btn = QPushButton("📂 Browse…")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._load_concept_file)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e7ff;
                border: 1px solid #6366f1;
                border-radius: 3px;
                padding: 4px 8px;
                color: #4338ca;
            }
            QPushButton:hover { background-color: #c7d2fe; }
        """)
        path_row.addWidget(browse_btn)
        file_layout.addLayout(path_row)
        
        # Preview of loaded concepts
        self.concept_preview = QLabel("")
        self.concept_preview.setWordWrap(True)
        self.concept_preview.setStyleSheet("color: #444; font-size: 11px;")
        file_layout.addWidget(self.concept_preview)
        
        layout.addWidget(file_box)
        
        # Status
        self.concept_status = QLabel("No concept data loaded")
        self.concept_status.setStyleSheet("color: #f59e0b;")
        layout.addWidget(self.concept_status)
        
        layout.addStretch()
        self.options_stack.addWidget(panel)
    
    def _load_concept_file(self):
        """Open file dialog to load a concept definition file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Concept Definitions",
            "",
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv *.tsv);;All Files (*)"
        )
        if not path:
            return
        
        try:
            import os
            ext = os.path.splitext(path)[1].lower()
            
            if ext in ('.xlsx', '.xls'):
                df = pd.read_excel(path)
            elif ext == '.tsv':
                df = pd.read_csv(path, sep='\t')
            else:
                df = pd.read_csv(path)
            
            if df.empty:
                self.concept_status.setText("File is empty")
                self.concept_status.setStyleSheet("color: #ef4444;")
                return
            
            self._concepts_df = df
            
            fname = os.path.basename(path)
            n_groups = len(df.columns)
            n_terms = df.notna().sum().sum()
            
            self.concept_file_label.setText(fname)
            self.concept_status.setText(f"✓ {n_groups} groups, {n_terms} terms")
            self.concept_status.setStyleSheet("color: #22c55e;")
            
            # Build preview: show group names and first few terms
            previews = []
            for col in df.columns:
                terms = df[col].dropna().astype(str).tolist()
                sample = ", ".join(terms[:3])
                if len(terms) > 3:
                    sample += f", … ({len(terms)} total)"
                previews.append(f"<b>{col}</b>: {sample}")
            self.concept_preview.setText("<br>".join(previews[:6]))
            if len(previews) > 6:
                self.concept_preview.setText(
                    self.concept_preview.text() + f"<br><i>… and {len(previews)-6} more</i>"
                )
            
        except Exception as e:
            self.concept_status.setText(f"Error: {e}")
            self.concept_status.setStyleSheet("color: #ef4444;")
            logger.exception("Failed to load concept file")
    
    def _create_dict_regex_options(self):
        """Create options panel for By Dictionary/Regex method."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Text column
        form = QFormLayout()
        self.dict_text_combo = QComboBox()
        self.dict_text_combo.currentTextChanged.connect(self._on_option_changed)
        form.addRow("Search in:", self.dict_text_combo)
        layout.addLayout(form)
        
        # Concept editor
        self.concept_editor = ConceptEditor()
        self.concept_editor.changed.connect(self._on_option_changed)
        layout.addWidget(self.concept_editor)
        
        self.options_stack.addWidget(panel)
    
    def _create_random_options(self):
        """Create options panel for Random Groups method."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        desc = QLabel(
            "<small>Create random overlapping groups for testing and validation. "
            "Documents may belong to multiple groups.</small>"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        form = QFormLayout()
        
        self.n_random_spin = QSpinBox()
        self.n_random_spin.setRange(2, 20)
        self.n_random_spin.setValue(self.n_random_groups)
        self.n_random_spin.valueChanged.connect(self._on_option_changed)
        form.addRow("Number of groups:", self.n_random_spin)
        
        layout.addLayout(form)
        layout.addStretch()
        
        self.options_stack.addWidget(panel)
    
    def _setup_main_area(self):
        """Build main area."""
        main_layout = QVBoxLayout()
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.mainArea.layout().addWidget(main_widget)
        
        # Status
        self.status_label = QLabel("Connect bibliographic data to begin")
        self.status_label.setStyleSheet("color: #666; padding: 4px;")
        main_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Tabs
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, 1)
        
        # Preview tab
        self.preview_table = GroupPreviewTable()
        self.tab_widget.addTab(self.preview_table, "👁️ Preview")
        
        # Summary tab
        self.summary_table = GroupPreviewTable()
        self.tab_widget.addTab(self.summary_table, "📊 Summary")
        
        # Info tab
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        info_header = QLabel("🔀 Setup Groups")
        info_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #3b82f6;")
        info_layout.addWidget(info_header)
        
        info_text = QLabel(
            "<p>Define document groups for comparative bibliometric analysis.</p>"
            "<p><b>Grouping Methods:</b></p>"
            "<ul>"
            "<li><b>By Column</b>: Each unique value becomes a group</li>"
            "<li><b>By Year Periods</b>: Split by publication years</li>"
            "<li><b>By Multi-Item Column</b>: Explode authors, keywords, etc.</li>"
            "<li><b>By Clustering</b>: Automatic text-based clustering</li>"
            "<li><b>By Concept DataFrame</b>: External concept definitions</li>"
            "<li><b>By Dictionary/Regex</b>: Pattern-based grouping</li>"
            "<li><b>Random Groups</b>: For testing purposes</li>"
            "</ul>"
            "<p><b>Output:</b></p>"
            "<ul>"
            "<li><b>Data</b>: Original data with group membership columns</li>"
            "<li><b>Group Matrix</b>: Binary document × group matrix</li>"
            "<li><b>Group Summary</b>: Statistics per group</li>"
            "</ul>"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #444;")
        info_layout.addWidget(info_text)
        info_layout.addStretch()
        
        self.tab_widget.addTab(info_widget, "ℹ️ Info")
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_method_changed(self, button):
        """Handle method selection change."""
        # QButtonGroup.buttonClicked can emit button or int
        if isinstance(button, int):
            idx = button
        else:
            idx = self.method_group.id(button)
        if idx < 0 or idx >= len(self.METHODS):
            return
        self.method_idx = idx
        self.options_stack.setCurrentIndex(idx)
        self._update_config_label()
        self._auto_create()
    
    def _on_column_changed(self, col_name):
        """Handle column selection change (By Column method)."""
        self.column_name = col_name
        self._update_unique_count()
        self._update_config_label()
        if self.method_idx == 0:
            self._auto_create()
    
    def _on_year_column_changed(self, col_name):
        """Handle year column selection change."""
        self.year_column = col_name
        self._update_year_range()
        self._update_config_label()
        if self.method_idx == 1:
            self._auto_create()
    
    def _on_multiitem_changed(self, col_name):
        """Handle multi-item column selection change."""
        self.multiitem_column_name = col_name
        self._update_items_count()
        self._update_config_label()
        if self.method_idx == 2:
            self._auto_create()
    
    def _on_periods_changed(self, value):
        """Handle periods spin change."""
        self.n_periods = value
        self._update_config_label()
        if self.method_idx == 1:
            self._auto_create()
    
    def _on_cutpoints_toggled(self, checked):
        """Handle cutpoints checkbox toggle."""
        self.use_cutpoints = checked
        self.cutpoints_edit.setEnabled(checked)
        self.periods_spin.setEnabled(not checked)
        self._update_config_label()
        if self.method_idx == 1:
            self._auto_create()
    
    def _on_auto_clusters_toggled(self, checked):
        """Handle auto clusters checkbox toggle."""
        self.auto_clusters = checked
        self.n_clusters_spin.setEnabled(not checked)
        self._update_config_label()
    
    def _on_filter_changed(self):
        """Handle include/exclude filter change."""
        self.include_items_str = self.include_edit.text()
        self.exclude_items_str = self.exclude_edit.text()
    
    def _on_option_changed(self, *args):
        """Handle general option change."""
        self._update_config_label()
        self._auto_create()
    
    def _auto_create(self):
        """Trigger group creation if auto-apply is enabled and data is loaded."""
        if self.auto_apply and self._df is not None:
            self._create_groups()
    
    # =========================================================================
    # INPUT HANDLERS
    # =========================================================================
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Handle data input."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._group_matrix = None
        
        if data is None:
            self._clear_outputs()
            self.status_label.setText("No data connected")
            return
        
        # Convert to DataFrame
        self._df = self._table_to_df(data)
        
        # Update column combos
        self._update_column_combos()
        
        # Show detected separator if in auto mode
        if self.separator_idx == 0:
            sep = self._detect_separator()
            self.detected_sep_label.setText(f'Detected: "{sep}"')
        
        self.status_label.setText(f"Data: {len(self._df)} documents, {len(self._df.columns)} columns")
        self._update_config_label()
        
        # Auto apply if enabled
        if self.auto_apply:
            self._create_groups()
    
    @Inputs.concepts
    def set_concepts(self, concepts: Optional[Table]):
        """Handle concepts input (from connected widget)."""
        if concepts is None:
            # Don't clear file-loaded concepts
            if self._concepts_df is None:
                self.concept_status.setText("No concept data loaded")
                self.concept_status.setStyleSheet("color: #f59e0b;")
            return
        
        self._concepts_df = self._table_to_df(concepts)
        n_groups = len(self._concepts_df.columns)
        n_terms = self._concepts_df.notna().sum().sum()
        
        self.concept_file_label.setText("(from input port)")
        self.concept_status.setText(f"✓ {n_groups} groups, {n_terms} terms")
        self.concept_status.setStyleSheet("color: #22c55e;")
        
        # Build preview
        df = self._concepts_df
        previews = []
        for col in df.columns:
            terms = df[col].dropna().astype(str).tolist()
            sample = ", ".join(terms[:3])
            if len(terms) > 3:
                sample += f", … ({len(terms)} total)"
            previews.append(f"<b>{col}</b>: {sample}")
        self.concept_preview.setText("<br>".join(previews[:6]))
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
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
    
    def _df_to_table(self, df: pd.DataFrame) -> Table:
        """Convert pandas DataFrame to Orange Table."""
        attrs = []
        metas = []
        
        for col in df.columns:
            series = df[col]
            
            # Check if boolean/binary
            unique = series.dropna().unique()
            if set(unique).issubset({True, False, 0, 1, 0.0, 1.0}):
                var = DiscreteVariable(col, values=["No", "Yes"])
                metas.append(var)
            elif pd.api.types.is_numeric_dtype(series):
                var = ContinuousVariable(col)
                attrs.append(var)
            else:
                var = StringVariable(col)
                metas.append(var)
        
        domain = Domain(attrs, metas=metas)
        
        # Build data arrays
        X = np.zeros((len(df), len(attrs)), dtype=float)
        metas_arr = np.zeros((len(df), len(metas)), dtype=object)
        
        for j, var in enumerate(attrs):
            X[:, j] = pd.to_numeric(df[var.name], errors='coerce').fillna(np.nan)
        
        for j, var in enumerate(metas):
            if isinstance(var, DiscreteVariable):
                # Map bool/binary to 0/1
                vals = df[var.name].fillna(False).astype(bool).astype(int)
                metas_arr[:, j] = vals
            else:
                metas_arr[:, j] = df[var.name].fillna("").astype(str)
        
        return Table.from_numpy(domain, X, metas=metas_arr)
    
    def _update_column_combos(self):
        """Update all column combo boxes."""
        if self._df is None:
            return
        
        cols = list(self._df.columns)
        
        # Helper: resolve first matching candidate from df columns
        def _resolve(candidates):
            for c in candidates:
                if c in cols:
                    return c
            return None
        
        # =================================================================
        # Multi-item combo: ONLY columns that actually contain separators
        # Ordered: keyword fields first, then other detected multi-item cols
        # =================================================================
        
        # Get active separator for detection
        active_sep = self._get_active_separator()
        
        # Priority order for multi-item fields (keywords at top)
        MULTIITEM_PRIORITY = [
            ["Processed Author Keywords", "Author Keywords"],
            ["Processed Index Keywords", "Index Keywords"],
            ["Processed Author and Index Keywords", "Author and Index Keywords"],
            ["Authors", "Author full names"],
            ["Affiliations", "Authors with affiliations"],
            ["Countries of Authors"],
            ["References", "Cited References", "referenced_works"],
            ["Area", "Subject Area", "Research Areas",
             "Web of Science Categories"],
            ["Science"],
            ["Field"],
            ["Funding Orgs", "Funding Details"],
            ["SDG"],
            ["MeSH Terms", "MeSH Headings"],
        ]
        
        # Detect which columns actually contain the active separator
        def _has_separator(col_name):
            """Check if a column contains the active separator."""
            try:
                sample = self._df[col_name].dropna().head(200).astype(str)
                return sample.str.contains(active_sep, regex=False).mean() > 0.05
            except Exception:
                return False
        
        self.multiitem_combo.blockSignals(True)
        self.multiitem_combo.clear()
        
        # 1. Priority fields that exist and have separators
        priority_cols = []
        for candidates in MULTIITEM_PRIORITY:
            match = _resolve(candidates)
            if match and match not in priority_cols and _has_separator(match):
                priority_cols.append(match)
        
        # 2. Other columns with separators (auto-detected)
        other_sep_cols = []
        for col in cols:
            if col not in priority_cols and _has_separator(col):
                other_sep_cols.append(col)
        
        mi_cols = priority_cols + other_sep_cols
        
        if mi_cols:
            self.multiitem_combo.addItems(mi_cols)
        else:
            # Fallback: all object columns
            self.multiitem_combo.addItems(
                [c for c in cols if self._df[c].dtype == 'object'] or cols
            )
        
        self.multiitem_combo.blockSignals(False)
        # Update dependent labels (without triggering auto_create)
        self.multiitem_column_name = self.multiitem_combo.currentText()
        self._update_items_count()
        
        # =================================================================
        # By Column combo: single-value categorical columns, recommended first
        # =================================================================
        SINGLE_PRIORITY = [
            ["CA Country", "Correspondence Author Country"],
            ["Document Type"],
            ["Document Type 2"],
            ["Language of Original Document", "Language", "language"],
            ["Source title", "Source Title", "source", "Publication Name"],
            ["Open Access"],
            ["Publication Stage"],
            ["Indexed In"],
        ]
        
        self.column_combo.blockSignals(True)
        self.column_combo.clear()
        
        recommended = []
        for candidates in SINGLE_PRIORITY:
            match = _resolve(candidates)
            if match and match not in recommended:
                recommended.append(match)
        
        other = [c for c in cols if c not in recommended]
        self.column_combo.addItems(recommended + other)
        
        self.column_combo.blockSignals(False)
        # Update dependent labels (without triggering auto_create)
        self.column_name = self.column_combo.currentText()
        self._update_unique_count()
        
        # =================================================================
        # Year columns (numeric in 1900-2100 range)
        # =================================================================
        year_cols = []
        for col in cols:
            try:
                numeric = pd.to_numeric(self._df[col], errors='coerce')
                valid = numeric.dropna()
                if len(valid) > 0 and valid.between(1900, 2100).mean() > 0.8:
                    year_cols.append(col)
            except Exception:
                pass
        
        self.year_combo.blockSignals(True)
        self.year_combo.clear()
        self.year_combo.addItems(year_cols if year_cols else cols)
        self.year_combo.blockSignals(False)
        
        # =================================================================
        # Text columns (for clustering / concept / dict search)
        # =================================================================
        KNOWN_TEXT = [
            "Abstract", "Processed Abstract", "Title", "Processed Title",
            "Author Keywords", "Index Keywords", "Author and Index Keywords",
            "Processed Author Keywords", "Processed Index Keywords",
            "Processed Author and Index Keywords",
        ]
        
        text_priority = [c for c in KNOWN_TEXT if c in cols]
        text_auto = []
        for col in cols:
            if col in text_priority:
                continue
            try:
                if self._df[col].dtype == 'object':
                    avg_len = self._df[col].dropna().astype(str).str.len().mean()
                    if avg_len > 50:
                        text_auto.append(col)
            except Exception:
                pass
        
        text_cols = text_priority + text_auto
        text_cols = text_cols if text_cols else cols
        
        for combo in [self.cluster_text_combo, self.concept_text_combo,
                       self.dict_text_combo]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(text_cols)
            combo.blockSignals(False)
    
    def _update_unique_count(self):
        """Update unique value count for selected column."""
        if self._df is None or not self.column_name:
            self.unique_label.setText("Unique values: -")
            return
        
        if self.column_name in self._df.columns:
            n_unique = self._df[self.column_name].nunique()
            self.unique_label.setText(f"Unique values: {n_unique}")
    
    def _update_year_range(self):
        """Update year range for selected year column."""
        if self._df is None or not self.year_column:
            self.year_range_label.setText("Range: -")
            return
        
        if self.year_column in self._df.columns:
            try:
                numeric = pd.to_numeric(self._df[self.year_column], errors='coerce')
                valid = numeric.dropna()
                if len(valid) > 0:
                    self.year_range_label.setText(f"Range: {int(valid.min())} - {int(valid.max())}")
                else:
                    self.year_range_label.setText("Range: No valid years")
            except Exception:
                self.year_range_label.setText("Range: Error")
    
    def _update_items_count(self):
        """Update item count for multi-item column."""
        if self._df is None or not self.column_name:
            self.items_label.setText("Unique items: -")
            return
        
        if self.column_name in self._df.columns:
            sep = self._get_active_separator()
            all_items = set()
            for val in self._df[self.column_name].dropna():
                for item in str(val).split(sep):
                    item = item.strip()
                    if item:
                        all_items.add(item)
            self.items_label.setText(f"Unique items: {len(all_items)}")
    
    def _update_config_label(self):
        """Update configuration summary label."""
        method_name = self.METHODS[self.method_idx][0]
        
        if self.method_idx == 0:  # By Column
            col = self.column_combo.currentText() or "(none)"
            self.config_label.setText(f"Method: {method_name}\nColumn: {col}")
        elif self.method_idx == 1:  # By Year Periods
            col = self.year_combo.currentText() or "(none)"
            if self.use_cutpoints:
                self.config_label.setText(f"Method: {method_name}\nColumn: {col}\nCutpoints: {self.cutpoints_edit.text()}")
            else:
                self.config_label.setText(f"Method: {method_name}\nColumn: {col}\nPeriods: {self.periods_spin.value()}")
        elif self.method_idx == 2:  # By Multi-Item
            col = self.multiitem_combo.currentText() or "(none)"
            self.config_label.setText(f"Method: {method_name}\nColumn: {col}\nTop N: {self.top_n_spin.value()}")
        elif self.method_idx == 3:  # By Clustering
            text = self.cluster_text_combo.currentText() or "(none)"
            n = "auto" if self.auto_clusters else self.n_clusters_spin.value()
            self.config_label.setText(f"Method: {method_name}\nText: {text}\nClusters: {n}")
        elif self.method_idx == 4:  # By Concept DF
            text = self.concept_text_combo.currentText() or "(none)"
            status = "connected" if self._concepts_df is not None else "not connected"
            self.config_label.setText(f"Method: {method_name}\nText: {text}\nConcepts: {status}")
        elif self.method_idx == 5:  # By Dict/Regex
            groups = self.concept_editor.get_group_dict()
            n = len(groups)
            self.config_label.setText(f"Method: {method_name}\nGroups defined: {n}")
        elif self.method_idx == 6:  # Random
            self.config_label.setText(f"Method: {method_name}\nGroups: {self.n_random_spin.value()}")
    
    def _clear_outputs(self):
        """Clear all outputs."""
        self.Outputs.data.send(None)
        self.Outputs.group_matrix.send(None)
        self.Outputs.group_summary.send(None)
        self.preview_table.clear()
        self.preview_table.setRowCount(0)
        self.summary_table.clear()
        self.summary_table.setRowCount(0)
    
    # =========================================================================
    # GROUP CREATION
    # =========================================================================
    
    def _preview_groups(self):
        """Preview groups without sending outputs."""
        self.Error.clear()
        self.Warning.clear()
        
        if self._df is None:
            self.Error.no_data()
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.status_label.setText("Generating group preview...")
            
            # Generate group matrix
            group_matrix = self._generate_group_matrix()
            
            if group_matrix is None or group_matrix.empty:
                self.Error.no_groups_created()
                return
            
            self._preview_df = group_matrix
            
            # Update preview table
            self._update_preview_table(group_matrix)
            
            # Update summary table
            self._update_summary_table(group_matrix)
            
            # Show summary
            n_groups = len(group_matrix.columns)
            n_docs = len(group_matrix)
            self.status_label.setText(f"Preview: {n_groups} groups, {n_docs} documents")
            
            # Switch to preview tab
            self.tab_widget.setCurrentIndex(0)
            
        except Exception as e:
            self.Error.grouping_failed(str(e))
            logger.exception("Group preview failed")
        finally:
            self.progress_bar.setVisible(False)
    
    def _create_groups(self):
        """Create groups and send outputs."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self.Error.no_data()
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.status_label.setText("Creating groups...")
            
            # Generate group matrix
            group_matrix = self._generate_group_matrix()
            
            if group_matrix is None or group_matrix.empty:
                self.Error.no_groups_created()
                self._clear_outputs()
                return
            
            self._group_matrix = group_matrix
            
            # Check for warnings
            n_groups = len(group_matrix.columns)
            n_docs = group_matrix.any(axis=1).sum()
            
            if n_groups == 1:
                self.Warning.few_groups(1)
            elif n_groups > 50:
                self.Warning.many_groups(n_groups)
            
            # Check for empty groups
            empty_groups = (group_matrix.sum() == 0).sum()
            if empty_groups > 0:
                self.Warning.empty_groups(empty_groups)
                # Remove empty groups
                group_matrix = group_matrix.loc[:, group_matrix.sum() > 0]
            
            # Update tables
            self._update_preview_table(group_matrix)
            self._update_summary_table(group_matrix)
            
            # Create outputs
            self._send_outputs(group_matrix)
            
            self.Information.groups_created(n_groups, n_docs)
            self.status_label.setText(f"Created {n_groups} groups with {n_docs} documents")
            
        except Exception as e:
            self.Error.grouping_failed(str(e))
            logger.exception("Group creation failed")
            self._clear_outputs()
        finally:
            self.progress_bar.setVisible(False)
    
    def _generate_group_matrix(self) -> Optional[pd.DataFrame]:
        """Generate group matrix based on selected method."""
        if self._df is None:
            return None
        
        method = self.method_idx
        
        # Parse include/exclude items
        include_items = None
        exclude_items = None
        if self.include_items_str.strip():
            include_items = [s.strip() for s in self.include_items_str.split(',') if s.strip()]
        if self.exclude_items_str.strip():
            exclude_items = [s.strip() for s in self.exclude_items_str.split(',') if s.strip()]
        
        whole_word = self.whole_word_cb.isChecked()
        
        if HAS_BIBLIUM:
            # Use Biblium's generate_group_matrix
            return self._generate_with_biblium(
                method, include_items, exclude_items, whole_word
            )
        else:
            # Fallback implementation
            return self._generate_fallback(
                method, include_items, exclude_items
            )
    
    def _generate_with_biblium(
        self,
        method: int,
        include_items: Optional[List[str]],
        exclude_items: Optional[List[str]],
        whole_word: bool
    ) -> Optional[pd.DataFrame]:
        """Generate groups using Biblium."""
        df = self._df.copy()
        
        if method == 0:  # By Column
            col = self.column_combo.currentText()
            if not col or col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
            
            return utilsbib.generate_group_matrix(
                df, col,
                force_type="column",
                include_items=include_items,
                exclude_items=exclude_items
            )
        
        elif method == 1:  # By Year Periods
            col = self.year_combo.currentText()
            if not col or col not in df.columns:
                raise ValueError(f"Year column '{col}' not found")
            
            cutpoints = None
            n_periods = None
            
            if self.use_cutpoints and self.cutpoints_edit.text().strip():
                try:
                    cutpoints = [int(x.strip()) for x in self.cutpoints_edit.text().split(',')]
                except ValueError:
                    raise ValueError("Invalid cutpoints format. Use comma-separated integers.")
            else:
                n_periods = self.periods_spin.value()
            
            return utilsbib.generate_group_matrix(
                df, col,
                force_type="year",
                cutpoints=cutpoints,
                n_periods=n_periods,
                include_items=include_items,
                exclude_items=exclude_items
            )
        
        elif method == 2:  # By Multi-Item Column
            col = self.multiitem_combo.currentText()
            if not col or col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
            
            return utilsbib.generate_group_matrix(
                df, col,
                force_type="multiitem",
                top_n=self.top_n_spin.value(),
                include_items=include_items,
                exclude_items=exclude_items
            )
        
        elif method == 3:  # By Clustering
            text_col = self.cluster_text_combo.currentText()
            if not text_col or text_col not in df.columns:
                raise ValueError(f"Text column '{text_col}' not found")
            
            # Use BiblioStats for clustering
            ba = BiblioStats(df=df, db="", label_docs=False, res_folder=None)
            
            n_clusters = None if self.auto_clusters else self.n_clusters_spin.value()
            method_name = self.cluster_method_combo.currentText()
            
            ba.cluster_documents(
                text_field=text_col,
                method=method_name,
                n_clusters=n_clusters,
                k_range=range(2, 11)
            )
            
            # Get cluster column
            cluster_col = ba.new_column if hasattr(ba, 'new_column') else None
            if cluster_col is None:
                # Find cluster column
                for c in ba.df.columns:
                    if 'cluster' in c.lower():
                        cluster_col = c
                        break
            
            if cluster_col is None:
                raise ValueError("Clustering did not produce cluster labels")
            
            return utilsbib.generate_group_matrix(
                ba.df, cluster_col,
                force_type="column",
                include_items=include_items,
                exclude_items=exclude_items
            )
        
        elif method == 4:  # By Concept DataFrame
            if self._concepts_df is None:
                raise ValueError("No concept DataFrame connected")
            
            text_col = self.concept_text_combo.currentText()
            if not text_col or text_col not in df.columns:
                raise ValueError(f"Text column '{text_col}' not found")
            
            return utilsbib.generate_group_matrix(
                df, self._concepts_df,
                force_type="concept",
                text_column=text_col,
                concept_whole_word=whole_word,
                include_items=include_items,
                exclude_items=exclude_items
            )
        
        elif method == 5:  # By Dictionary/Regex
            groups_dict = self.concept_editor.get_group_dict()
            if not groups_dict:
                raise ValueError("No groups defined. Add group definitions.")
            
            text_col = self.dict_text_combo.currentText()
            if not text_col or text_col not in df.columns:
                raise ValueError(f"Text column '{text_col}' not found")
            
            return utilsbib.generate_group_matrix(
                df, groups_dict,
                force_type="regex",
                text_column=text_col,
                concept_whole_word=whole_word,
                include_items=include_items,
                exclude_items=exclude_items
            )
        
        elif method == 6:  # Random Groups
            n_groups = self.n_random_spin.value()
            return utilsbib.generate_group_matrix(
                df, n_groups,
                include_items=include_items,
                exclude_items=exclude_items
            )
        
        return None
    
    def _generate_fallback(
        self,
        method: int,
        include_items: Optional[List[str]],
        exclude_items: Optional[List[str]]
    ) -> Optional[pd.DataFrame]:
        """Fallback group generation without Biblium."""
        df = self._df.copy()
        
        if method == 0:  # By Column
            col = self.column_combo.currentText()
            if not col or col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
            
            # One-hot encode
            result = pd.get_dummies(df[col].astype(str), dtype=bool)
            
        elif method == 1:  # By Year Periods
            col = self.year_combo.currentText()
            if not col or col not in df.columns:
                raise ValueError(f"Year column '{col}' not found")
            
            years = pd.to_numeric(df[col], errors='coerce')
            
            if self.use_cutpoints and self.cutpoints_edit.text().strip():
                cutpoints = [int(x.strip()) for x in self.cutpoints_edit.text().split(',')]
                bins = [-np.inf] + cutpoints + [np.inf]
                labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
            else:
                n = self.periods_spin.value()
                mn, mx = int(years.min()), int(years.max())
                bins = np.linspace(mn, mx + 1, n + 1).astype(int).tolist()
                labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
            
            cats = pd.cut(years, bins=bins, labels=labels, include_lowest=True)
            result = pd.get_dummies(cats, dtype=bool)
            
        elif method == 2:  # By Multi-Item
            col = self.multiitem_combo.currentText()
            if not col or col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
            
            sep = self._get_active_separator()
            exploded = df[col].fillna("").str.split(sep).explode().str.strip()
            exploded = exploded[exploded != ""]
            result = pd.get_dummies(exploded, dtype=bool).groupby(level=0).max()
            
            # Top N
            top_n = self.top_n_spin.value()
            if result.shape[1] > top_n:
                top_cols = result.sum().nlargest(top_n).index
                result = result[top_cols]
            
        elif method == 5:  # By Dict/Regex
            groups_dict = self.concept_editor.get_group_dict()
            if not groups_dict:
                raise ValueError("No groups defined")
            
            text_col = self.dict_text_combo.currentText()
            if not text_col or text_col not in df.columns:
                raise ValueError(f"Text column '{text_col}' not found")
            
            text = df[text_col].fillna("")
            result = pd.DataFrame(index=df.index)
            
            for group_name, terms in groups_dict.items():
                pattern = '|'.join(re.escape(t).replace(r'\*', '.*') for t in terms)
                result[group_name] = text.str.contains(pattern, case=False, regex=True)
            
        elif method == 6:  # Random
            n_groups = self.n_random_spin.value()
            result = pd.DataFrame(
                np.random.rand(len(df), n_groups) < 0.5,
                index=df.index,
                columns=[f"Random {i+1}" for i in range(n_groups)]
            )
        
        else:
            raise ValueError(f"Method {method} requires Biblium library")
        
        # Apply include/exclude filters
        if include_items:
            result = result[[c for c in include_items if c in result.columns]]
        if exclude_items:
            result = result[[c for c in result.columns if c not in exclude_items]]
        
        return result.reindex(df.index, fill_value=False)
    
    def _update_preview_table(self, group_matrix: pd.DataFrame):
        """Update preview table with sample data."""
        self.preview_table.clear()
        
        if group_matrix.empty:
            return
        
        # Show first 100 rows with group assignments
        n_show = min(100, len(group_matrix))
        sample = group_matrix.head(n_show)
        
        # Add document index
        display_df = sample.copy()
        display_df.insert(0, 'Document', range(1, len(display_df) + 1))
        
        # Populate table
        self.preview_table.setRowCount(len(display_df))
        self.preview_table.setColumnCount(len(display_df.columns))
        self.preview_table.setHorizontalHeaderLabels(display_df.columns.tolist())
        
        for i, (idx, row) in enumerate(display_df.iterrows()):
            for j, col in enumerate(display_df.columns):
                val = row[col]
                if isinstance(val, (bool, np.bool_)):
                    item = QTableWidgetItem("✓" if val else "")
                    if val:
                        item.setBackground(Qt.green)
                else:
                    item = QTableWidgetItem(str(val))
                self.preview_table.setItem(i, j, item)
        
        self.preview_table.resizeColumnsToContents()
    
    def _update_summary_table(self, group_matrix: pd.DataFrame):
        """Update summary table with group statistics."""
        self.summary_table.clear()
        
        if group_matrix.empty:
            return
        
        # Compute statistics
        stats = []
        for col in group_matrix.columns:
            count = group_matrix[col].sum()
            pct = count / len(group_matrix) * 100
            stats.append({
                'Group': col,
                'Documents': int(count),
                'Percentage': f"{pct:.1f}%",
                'Rank': 0
            })
        
        # Add ranks
        stats = sorted(stats, key=lambda x: x['Documents'], reverse=True)
        for i, s in enumerate(stats):
            s['Rank'] = i + 1
        
        # Populate table
        self.summary_table.setRowCount(len(stats))
        self.summary_table.setColumnCount(4)
        self.summary_table.setHorizontalHeaderLabels(['Rank', 'Group', 'Documents', 'Percentage'])
        
        for i, s in enumerate(stats):
            self.summary_table.setItem(i, 0, QTableWidgetItem(str(s['Rank'])))
            self.summary_table.setItem(i, 1, QTableWidgetItem(s['Group']))
            self.summary_table.setItem(i, 2, QTableWidgetItem(str(s['Documents'])))
            self.summary_table.setItem(i, 3, QTableWidgetItem(s['Percentage']))
        
        self.summary_table.resizeColumnsToContents()
    
    def _send_outputs(self, group_matrix: pd.DataFrame):
        """Send outputs."""
        # Data with group columns
        if self._data is not None:
            enhanced_df = self._df.copy()
            for col in group_matrix.columns:
                enhanced_df[f"Group: {col}"] = group_matrix[col].astype(int)
            
            enhanced_table = self._df_to_table(enhanced_df)
            self.Outputs.data.send(enhanced_table)
        else:
            self.Outputs.data.send(None)
        
        # Group matrix
        matrix_table = self._df_to_table(group_matrix.astype(int))
        self.Outputs.group_matrix.send(matrix_table)
        
        # Group summary
        summary_data = []
        for col in group_matrix.columns:
            count = group_matrix[col].sum()
            pct = count / len(group_matrix) * 100
            summary_data.append({
                'Group': col,
                'Documents': int(count),
                'Percentage': pct
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_table = self._df_to_table(summary_df)
        self.Outputs.group_summary.send(summary_table)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWSetupGroups).run()
