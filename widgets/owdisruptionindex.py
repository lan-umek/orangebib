# -*- coding: utf-8 -*-
"""
Disruption Index Widget
=======================
Compute the Disruption Index (CD Index) to measure whether papers
consolidate or disrupt existing knowledge.

Based on Funk & Owen-Smith (2017) and Wu, Wang & Evans (2019).

CD Index ranges from -1 (consolidating) to +1 (disruptive):
- Positive: Disruptive work that redirects the field
- Negative: Consolidating work that builds on existing foundations

Uses Biblium's BiblioStats class for proper database-specific handling.
"""

import logging
import re
from typing import Optional, List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem,
    QTabWidget, QFrame, QProgressBar, QRadioButton, QButtonGroup,
    QFileDialog, QMessageBox, QTextEdit, QWidget,
)
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor

from Orange.data import Table, Domain, ContinuousVariable, StringVariable, DiscreteVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT BIBLIUM
# =============================================================================

try:
    from biblium.bibstats import BiblioStats
    from biblium.disruption import (
        compute_document_disruption,
        aggregate_disruption_by_entity,
    )
    HAS_BIBLIUM = True
    logger.info("Biblium BiblioStats available")
except ImportError:
    HAS_BIBLIUM = False
    logger.warning("Biblium not available - install with: pip install biblium")

# =============================================================================
# PYQTGRAPH FOR VISUALIZATION
# =============================================================================

try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background='w', foreground='k')
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


# =============================================================================
# PLOT WIDGETS
# =============================================================================

from AnyQt.QtWidgets import QToolTip
from AnyQt.QtCore import pyqtSignal

class DisruptionDistributionPlot(pg.PlotWidget):
    """Distribution plot for disruption index with hover and selection."""
    
    selectionChanged = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=False, y=False)
        
        self._bins = []
        self._bar_items = []
        self._selected_bin = None
        self._df = None
        
        self.scene().sigMouseMoved.connect(self._on_hover)
        self.scene().sigMouseClicked.connect(self._on_click)
    
    def plot_distribution(self, df: pd.DataFrame, metric: str = "cd_index",
                          show_stats: bool = True):
        """Plot histogram of disruption values with interactivity."""
        self.clear()
        self._bins = []
        self._bar_items = []
        self._selected_bin = None
        self._df = df
        
        if df is None or df.empty or metric not in df.columns:
            return
        
        values = df[metric].dropna()
        if values.empty:
            return
        
        # Compute histogram
        counts, bin_edges = np.histogram(values, bins=30, range=(-1, 1))
        
        # Build bins with document indices
        for i in range(len(counts)):
            left = bin_edges[i]
            right = bin_edges[i + 1]
            center = (left + right) / 2
            
            # Find documents in this bin
            if i == len(counts) - 1:
                mask = (values >= left) & (values <= right)
            else:
                mask = (values >= left) & (values < right)
            
            indices = df[df[metric].notna()][mask].index.tolist()
            
            # Color based on value
            if center < -0.25:
                color = '#ef4444'  # Red - consolidating
            elif center > 0.25:
                color = '#22c55e'  # Green - disruptive
            else:
                color = '#6b7280'  # Gray - neutral
            
            self._bins.append({
                'left': left,
                'right': right,
                'center': center,
                'count': counts[i],
                'color': color,
                'indices': indices,
            })
            
            # Create bar
            bar = pg.BarGraphItem(
                x=[center], height=[counts[i]], width=(right - left) * 0.9,
                brush=pg.mkBrush(color),
                pen=pg.mkPen('w', width=1)
            )
            self.addItem(bar)
            self._bar_items.append(bar)
        
        # Add zero line
        zero_line = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen('#000000', width=2, style=Qt.DashLine)
        )
        self.addItem(zero_line)
        
        # Add statistics
        if show_stats and len(values) > 0:
            mean_val = values.mean()
            median_val = values.median()
            
            # Mean line
            mean_line = pg.InfiniteLine(
                pos=mean_val, angle=90,
                pen=pg.mkPen('#3b82f6', width=2)
            )
            self.addItem(mean_line)
            
            # Stats text
            max_count = max(counts) if max(counts) > 0 else 1
            stats_text = pg.TextItem(
                f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nN: {len(values)}",
                color='k', anchor=(0, 1)
            )
            stats_text.setPos(0.5, max_count * 0.95)
            self.addItem(stats_text)
        
        self.setLabel('bottom', 'CD Index')
        self.setLabel('left', 'Count')
        self.setTitle('Distribution of CD Index (click bin to select)')
        self.setXRange(-1.1, 1.1)
    
    def _on_hover(self, pos):
        """Show tooltip on hover."""
        mouse = self.plotItem.vb.mapSceneToView(pos)
        
        for b in self._bins:
            if b['left'] <= mouse.x() <= b['right'] and 0 <= mouse.y() <= b['count']:
                interp = "consolidating" if b['center'] < -0.25 else ("disruptive" if b['center'] > 0.25 else "neutral")
                tip = f"Range: {b['left']:.2f} to {b['right']:.2f}\nCount: {b['count']}\nType: {interp}\nClick to select {len(b['indices'])} papers"
                QToolTip.showText(self.mapToGlobal(self.mapFromScene(pos)), tip)
                return
        
        QToolTip.hideText()
    
    def _on_click(self, event):
        """Select documents in clicked bin."""
        mouse = self.plotItem.vb.mapSceneToView(event.scenePos())
        
        for i, b in enumerate(self._bins):
            if b['left'] <= mouse.x() <= b['right'] and 0 <= mouse.y() <= b['count']:
                # Toggle selection
                if self._selected_bin == i:
                    self._selected_bin = None
                    self._update_visual()
                    self.selectionChanged.emit([])
                else:
                    self._selected_bin = i
                    self._update_visual()
                    self.selectionChanged.emit(b['indices'])
                return
    
    def _update_visual(self):
        """Highlight selected bin."""
        for i, bar in enumerate(self._bar_items):
            if i == self._selected_bin:
                bar.setOpts(pen=pg.mkPen('#000000', width=3))
            else:
                bar.setOpts(pen=pg.mkPen('w', width=1))
    
    def clear_selection(self):
        """Clear selection."""
        self._selected_bin = None
        self._update_visual()


class DisruptionRankingPlot(pg.PlotWidget):
    """Horizontal bar chart ranking entities by disruption with hover."""
    
    selectionChanged = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=False, y=False)
        
        self._bars = []
        self._bar_items = []
        self._selected_bar = None
        
        self.scene().sigMouseMoved.connect(self._on_hover)
        self.scene().sigMouseClicked.connect(self._on_click)
    
    def plot_ranking(self, df: pd.DataFrame, entity_col: str, metric_col: str,
                     top_n: int = 20, doc_indices: Dict = None):
        """Plot top/bottom entities by disruption."""
        self.clear()
        self._bars = []
        self._bar_items = []
        self._selected_bar = None
        
        if df is None or df.empty:
            return
        
        if entity_col not in df.columns or metric_col not in df.columns:
            return
        
        # Get top N by value
        plot_df = df.dropna(subset=[metric_col]).copy()
        plot_df = plot_df.nlargest(top_n, metric_col)
        
        entities = plot_df[entity_col].astype(str).tolist()[::-1]  # Reverse for bottom-to-top
        values = plot_df[metric_col].tolist()[::-1]
        indices_list = plot_df.index.tolist()[::-1]
        
        if not entities:
            return
        
        # Create bars with colors based on value
        for i, (entity, val, idx) in enumerate(zip(entities, values, indices_list)):
            if val < -0.1:
                color = '#ef4444'  # Red
            elif val > 0.1:
                color = '#22c55e'  # Green
            else:
                color = '#6b7280'  # Gray
            
            self._bars.append({
                'entity': entity,
                'value': val,
                'color': color,
                'y': i,
                'index': idx,
            })
            
            bar = pg.BarGraphItem(
                x0=0, x1=val, y=i, height=0.7,
                brush=pg.mkBrush(color),
                pen=pg.mkPen('w', width=1)
            )
            self.addItem(bar)
            self._bar_items.append(bar)
        
        # Zero line
        zero_line = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen('#000000', width=1)
        )
        self.addItem(zero_line)
        
        # Set y-axis labels
        axis = self.getAxis('left')
        ticks = [(i, entities[i][:25]) for i in range(len(entities))]
        axis.setTicks([ticks])
        
        self.setLabel('bottom', 'CD Index')
        self.setTitle(f'Top {min(top_n, len(entities))} by Disruption (click to select)')
    
    def _on_hover(self, pos):
        """Show tooltip on hover."""
        mouse = self.plotItem.vb.mapSceneToView(pos)
        
        for b in self._bars:
            y_min = b['y'] - 0.35
            y_max = b['y'] + 0.35
            x_min = min(0, b['value'])
            x_max = max(0, b['value'])
            
            if x_min <= mouse.x() <= x_max and y_min <= mouse.y() <= y_max:
                interp = "consolidating" if b['value'] < -0.1 else ("disruptive" if b['value'] > 0.1 else "neutral")
                tip = f"{b['entity']}\nCD Index: {b['value']:.3f}\n({interp})"
                QToolTip.showText(self.mapToGlobal(self.mapFromScene(pos)), tip)
                return
        
        QToolTip.hideText()
    
    def _on_click(self, event):
        """Select clicked bar."""
        mouse = self.plotItem.vb.mapSceneToView(event.scenePos())
        
        for i, b in enumerate(self._bars):
            y_min = b['y'] - 0.35
            y_max = b['y'] + 0.35
            x_min = min(0, b['value'])
            x_max = max(0, b['value'])
            
            if x_min <= mouse.x() <= x_max and y_min <= mouse.y() <= y_max:
                if self._selected_bar == i:
                    self._selected_bar = None
                    self._update_visual()
                    self.selectionChanged.emit([])
                else:
                    self._selected_bar = i
                    self._update_visual()
                    self.selectionChanged.emit([b['index']])
                return
    
    def _update_visual(self):
        """Highlight selected bar."""
        for i, bar in enumerate(self._bar_items):
            if i == self._selected_bar:
                bar.setOpts(pen=pg.mkPen('#000000', width=3))
            else:
                bar.setOpts(pen=pg.mkPen('w', width=1))
    
    def clear_selection(self):
        """Clear selection."""
        self._selected_bar = None
        self._update_visual()


# =============================================================================
# NUMERIC TABLE ITEM
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

class OWDisruptionIndex(OWWidget):
    """Compute Disruption Index (CD Index) for bibliometric data.
    
    Uses Biblium's BiblioStats class for proper database handling.
    """
    
    name = "Disruption Index"
    description = "Measure whether papers consolidate or disrupt fields"
    icon = "icons/disruption_index.svg"
    priority = 72
    keywords = ["disruption", "cd index", "consolidation", "innovation", "impact"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with references")
    
    class Outputs:
        disruption = Output("Disruption", Table, doc="Papers with disruption metrics")
        aggregated = Output("Aggregated", Table, doc="Aggregated disruption by entity")
        enhanced = Output("Enhanced Data", Table, doc="Original data with disruption added")
        selected = Output("Selected Documents", Table, doc="Selected documents from plots")
    
    # Settings
    analysis_type = settings.Setting(0)  # 0=Documents, 1=Sources, 2=Authors, 3=Countries, 4=Affiliations, 5=Years
    min_documents = settings.Setting(3)
    top_n = settings.Setting(20)
    add_to_dataset = settings.Setting(True)
    
    plot_type_idx = settings.Setting(0)  # 0=Distribution, 1=Ranking, 2=Trend
    show_visualization = settings.Setting(True)
    show_statistics = settings.Setting(True)
    
    auto_apply = settings.Setting(False)
    
    want_main_area = True
    resizing_enabled = True
    
    ANALYSIS_TYPES = [
        ("Documents", "Document-level disruption"),
        ("Sources/Journals", "Average disruption by journal"),
        ("Authors", "Average disruption by author"),
        ("Countries", "Average disruption by country"),
        ("Affiliations", "Average disruption by institution"),
        ("Years", "Disruption trend over time"),
    ]
    
    PLOT_TYPES = ["Distribution", "Ranking", "Trend"]
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium library required - install with: pip install biblium")
        no_refs = Msg("References column not found - required for disruption computation")
        no_id = Msg("Document ID column not found")
        compute_error = Msg("Analysis error: {}")
    
    class Warning(OWWidget.Warning):
        few_internal = Msg("Few internal citations detected. CD Index works best when papers cite each other.")
        no_valid = Msg("No valid disruption values computed")
    
    class Information(OWWidget.Information):
        computed = Msg("Computed disruption for {} documents ({} with citations)")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._bib: Optional[BiblioStats] = None
        self._disruption_df: Optional[pd.DataFrame] = None
        self._aggregated_df: Optional[pd.DataFrame] = None
        self._db_type: str = ""
        self._selected_indices: List = []
        
        self._setup_control_area()
        self._setup_main_area()
    
    def _setup_control_area(self):
        """Build control area."""
        # Analysis Type
        type_box = gui.widgetBox(self.controlArea, "📊 Analysis Type")
        
        self.type_buttons = QButtonGroup(self)
        
        for i, (name, desc) in enumerate(self.ANALYSIS_TYPES):
            radio = QRadioButton(f"{name}")
            radio.setToolTip(desc)
            if i == self.analysis_type:
                radio.setChecked(True)
            self.type_buttons.addButton(radio, i)
            
            # Add description label
            layout = QHBoxLayout()
            layout.addWidget(radio)
            desc_label = QLabel(f"<small><i>({desc})</i></small>")
            desc_label.setStyleSheet("color: #666;")
            layout.addWidget(desc_label)
            layout.addStretch()
            
            container = QWidget()
            container.setLayout(layout)
            type_box.layout().addWidget(container)
        
        self.type_buttons.buttonClicked.connect(self._on_type_changed)
        
        # Parameters
        params_box = gui.widgetBox(self.controlArea, "⚙️ Parameters")
        
        # Min Documents
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min Documents:"))
        self.min_spin = QSpinBox()
        self.min_spin.setRange(1, 100)
        self.min_spin.setValue(self.min_documents)
        self.min_spin.valueChanged.connect(self._on_params_changed)
        min_layout.addWidget(self.min_spin)
        params_box.layout().addLayout(min_layout)
        
        # Top N
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Top N:"))
        self.top_spin = QSpinBox()
        self.top_spin.setRange(5, 200)
        self.top_spin.setValue(self.top_n)
        self.top_spin.valueChanged.connect(self._on_params_changed)
        top_layout.addWidget(self.top_spin)
        params_box.layout().addLayout(top_layout)
        
        # Add to dataset checkbox
        self.add_check = gui.checkBox(
            params_box, self, "add_to_dataset",
            "Add disruption to dataset",
            callback=self._on_params_changed
        )
        
        # View Data link (placeholder)
        view_btn = QPushButton("📋 View Data →")
        view_btn.setFlat(True)
        view_btn.setStyleSheet("color: #3b82f6; text-align: left;")
        view_btn.clicked.connect(self._view_data)
        params_box.layout().addWidget(view_btn)
        
        # Visualization
        viz_box = gui.widgetBox(self.controlArea, "📈 Visualization")
        
        # Plot type
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(QLabel("Plot Type:"))
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(self.PLOT_TYPES)
        self.plot_combo.setCurrentIndex(self.plot_type_idx)
        self.plot_combo.currentIndexChanged.connect(self._on_plot_changed)
        plot_layout.addWidget(self.plot_combo)
        viz_box.layout().addLayout(plot_layout)
        
        # Checkboxes
        gui.checkBox(viz_box, self, "show_visualization", "Show visualization",
                    callback=self._update_plots)
        gui.checkBox(viz_box, self, "show_statistics", "Show statistics on plot",
                    callback=self._update_plots)
        
        # Run button
        self.run_btn = gui.button(
            self.controlArea, self, "▶ Analyze",
            callback=self.commit, autoDefault=False
        )
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold;")
        
        gui.checkBox(self.controlArea, self, "auto_apply", "Auto apply")
        
        self.controlArea.layout().addStretch(1)
    
    def _setup_main_area(self):
        """Build main area."""
        main_layout = QVBoxLayout()
        self.mainArea.layout().addLayout(main_layout)
        
        # Results header
        header = QLabel("<b>Results</b>")
        header.setStyleSheet("font-size: 14px; padding: 4px;")
        main_layout.addWidget(header)
        
        # Status
        self.status_label = QLabel("Configure options and click Run to see results")
        self.status_label.setStyleSheet("color: #6c757d; padding: 4px;")
        main_layout.addWidget(self.status_label)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, 1)
        
        # Results tab (plot)
        if HAS_PYQTGRAPH:
            self.dist_plot = DisruptionDistributionPlot()
            self.rank_plot = DisruptionRankingPlot()
            
            # Connect selection signals
            self.dist_plot.selectionChanged.connect(self._on_selection)
            self.rank_plot.selectionChanged.connect(self._on_selection)
            
            # Stack plots
            self.plot_stack = QTabWidget()
            self.plot_stack.addTab(self.dist_plot, "Distribution")
            self.plot_stack.addTab(self.rank_plot, "Ranking")
            
            self.tab_widget.addTab(self.plot_stack, "📊 Results")
        else:
            placeholder = QLabel("PyQtGraph required for visualization")
            placeholder.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(placeholder, "📊 Results")
        
        # Info tab
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        # Info header
        info_header = QLabel("⚡ Disruption Index")
        info_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #3b82f6;")
        info_layout.addWidget(info_header)
        
        # Info text
        info_text = QLabel(
            "<p>Measure whether papers consolidate or disrupt fields.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>CD Index calculation</li>"
            "<li>DI Index variant</li>"
            "<li>Disruption classification</li>"
            "<li>Aggregation by entity</li>"
            "</ul>"
            "<p><b>Positive</b> = disruptive; <b>Negative</b> = consolidating work.</p>"
            "<p><b>Note:</b> CD Index requires papers to cite each other within the dataset. "
            "Works best with focused datasets from a specific research area.</p>"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #444;")
        info_layout.addWidget(info_text)
        info_layout.addStretch()
        
        self.tab_widget.addTab(info_widget, "ℹ️ Info")
        
        # Data table tab
        self.results_table = QTableWidget()
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSortingEnabled(True)
        self.tab_widget.addTab(self.results_table, "📋 Data")
        
        # Export
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_btn = QPushButton("📥 Export to Excel")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        main_layout.addLayout(export_layout)
    
    def _on_type_changed(self, button):
        """Handle analysis type change."""
        self.analysis_type = self.type_buttons.id(button)
    
    def _on_params_changed(self):
        """Handle parameter change."""
        self.min_documents = self.min_spin.value()
        self.top_n = self.top_spin.value()
    
    def _on_plot_changed(self, idx: int):
        """Handle plot type change."""
        self.plot_type_idx = idx
        if HAS_PYQTGRAPH:
            self.plot_stack.setCurrentIndex(idx)
    
    def _view_data(self):
        """Switch to data tab."""
        self.tab_widget.setCurrentIndex(2)  # Data tab
    
    def _detect_db_type(self, df: pd.DataFrame) -> str:
        """Detect database type from column names."""
        cols = set(df.columns)
        
        # OpenAlex - check for OpenAlex-specific columns
        if 'referenced_works' in cols or 'ids.openalex' in cols:
            return 'oa'
        
        # Check if IDs look like OpenAlex URLs
        for col in ['unique-id', 'id', 'DOI']:
            if col in cols:
                sample = df[col].dropna().head(5).astype(str)
                if any('openalex.org' in str(v) for v in sample):
                    return 'oa'
        
        # Scopus
        if 'EID' in cols or 'Source title' in cols:
            return 'scopus'
        
        # Web of Science
        if 'UT' in cols or 'WOS' in cols:
            return 'wos'
        
        # Default
        return ''
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _normalize_openalex_id(self, id_str: str) -> str:
        """Normalize OpenAlex ID to short form (W123456789)."""
        if pd.isna(id_str) or not id_str:
            return ""
        id_str = str(id_str).strip()
        # Extract W followed by digits
        match = re.search(r'(W\d+)', id_str)
        if match:
            return match.group(1).lower()
        return id_str.lower()
    
    def _compute_openalex_disruption(self, df: pd.DataFrame, id_col: str, 
                                      refs_col: str, sep: str) -> pd.DataFrame:
        """
        Compute disruption index for OpenAlex data with proper ID normalization.
        
        OpenAlex IDs can be in different formats:
        - Full URL: https://openalex.org/W123456789
        - Short form: W123456789
        
        This method normalizes all IDs to short form for matching.
        """
        # Step 1: Build set of all paper IDs in dataset (normalized)
        all_papers = {}  # normalized_id -> original_id
        for _, row in df.iterrows():
            orig_id = str(row.get(id_col, ""))
            if orig_id and orig_id != "nan":
                norm_id = self._normalize_openalex_id(orig_id)
                if norm_id:
                    all_papers[norm_id] = orig_id
        
        # Step 2: Build citation network (who cites whom)
        # citation_network[paper] = set of papers it cites (that are in dataset)
        citation_network = {}
        
        for _, row in df.iterrows():
            doc_id = str(row.get(id_col, ""))
            if not doc_id or doc_id == "nan":
                continue
            
            doc_norm = self._normalize_openalex_id(doc_id)
            refs_str = row.get(refs_col, "")
            
            if pd.isna(refs_str) or not refs_str:
                citation_network[doc_norm] = set()
                continue
            
            refs = set()
            for ref in str(refs_str).split(sep):
                ref_norm = self._normalize_openalex_id(ref.strip())
                if ref_norm and ref_norm in all_papers:
                    refs.add(ref_norm)
            
            citation_network[doc_norm] = refs
        
        # Step 3: Build reverse citation index (who is cited by whom)
        cited_by = defaultdict(set)
        for paper, refs in citation_network.items():
            for ref in refs:
                cited_by[ref].add(paper)
        
        # Diagnostic
        internal_citations = sum(1 for p in all_papers if cited_by.get(p))
        logger.info(f"OpenAlex disruption: {len(all_papers)} papers, {internal_citations} with internal citations")
        
        # Step 4: Compute disruption for each paper
        results = []
        
        for norm_id, orig_id in all_papers.items():
            # Get focal paper's references (only internal)
            focal_refs = citation_network.get(norm_id, set())
            
            # Get papers that cite the focal paper (internal only)
            citing_papers = cited_by.get(norm_id, set())
            
            # CD Index components
            n_i = 0  # cite focal but NOT its references
            n_j = 0  # cite BOTH focal AND its references
            n_k = 0  # cite focal's references but NOT focal
            
            # For each paper that cites focal paper
            for citer in citing_papers:
                citer_refs = citation_network.get(citer, set())
                
                # Does citer also cite any of focal's references?
                cites_focal_refs = bool(citer_refs & focal_refs)
                
                if cites_focal_refs:
                    n_j += 1
                else:
                    n_i += 1
            
            # For n_k: papers that cite focal's refs but not focal
            for ref in focal_refs:
                ref_citers = cited_by.get(ref, set())
                for citer in ref_citers:
                    if citer != norm_id and citer not in citing_papers:
                        n_k += 1
            
            # Compute indices
            n_citing = len(citing_papers)
            denominator = n_i + n_j + n_k
            
            if n_citing == 0:
                cd_index = np.nan
                di_index = np.nan
                interpretation = "uncited"
            elif denominator == 0:
                cd_index = np.nan
                di_index = np.nan
                interpretation = "no context"
            else:
                cd_index = (n_i - n_j) / denominator
                di_index = n_i / denominator if denominator > 0 else np.nan
                
                if cd_index > 0.25:
                    interpretation = "disruptive"
                elif cd_index < -0.25:
                    interpretation = "consolidating"
                else:
                    interpretation = "neutral"
            
            results.append({
                'doc_id': orig_id,
                'cd_index': cd_index,
                'di_index': di_index,
                'n_citing': n_citing,
                'n_i': n_i,
                'n_j': n_j,
                'n_k': n_k,
                'interpretation': interpretation,
            })
        
        return pd.DataFrame(results)
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive input data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._bib = None
        self._disruption_df = None
        self._aggregated_df = None
        
        self._clear_results()
        
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            return
        
        if data is None:
            self.Error.no_data()
            return
        
        # Convert to DataFrame
        self._df = self._table_to_df(data)
        
        # Detect database type
        self._db_type = self._detect_db_type(self._df)
        
        self.status_label.setText(
            f"Loaded {len(self._df)} papers (detected: {self._db_type or 'unknown'}) - click Analyze"
        )
        
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
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.export_btn.setEnabled(False)
        
        if HAS_PYQTGRAPH:
            self.dist_plot.clear()
            self.rank_plot.clear()
        
        self.status_label.setText("Configure options and click Run to see results")
    
    def commit(self):
        """Run disruption analysis."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self._send_outputs(None, None, None)
            return
        
        if self._df is None:
            self.Error.no_data()
            self._send_outputs(None, None, None)
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.status_label.setText("Computing disruption index...")
            
            # Determine columns based on database type
            if self._db_type == 'oa':
                # OpenAlex - specific column names and separator
                id_col = self._find_column(self._df, [
                    'unique-id', 'id', 'ids.openalex', 'work_id'
                ])
                refs_col = self._find_column(self._df, [
                    'referenced_works', 'References'
                ])
                sep = '|'
            else:
                # Scopus/WoS/other
                id_col = self._find_column(self._df, [
                    'unique-id', 'DOI', 'EID', 'UT'
                ])
                refs_col = self._find_column(self._df, [
                    'References', 'Cited References', 'CR'
                ])
                sep = '; '
            
            if id_col is None:
                self.Error.no_id()
                self._send_outputs(None, None, None)
                return
            
            if refs_col is None:
                self.Error.no_refs()
                self._send_outputs(None, None, None)
                return
            
            self.status_label.setText(
                f"Computing disruption... (ID: {id_col}, Refs: {refs_col}, Sep: '{sep}')"
            )
            
            # For OpenAlex, normalize IDs to short form (W123456789)
            if self._db_type == 'oa':
                self._disruption_df = self._compute_openalex_disruption(
                    self._df, id_col, refs_col, sep
                )
            else:
                # Compute disruption using low-level function
                self._disruption_df = compute_document_disruption(
                    self._df,
                    id_col=id_col,
                    refs_col=refs_col,
                    sep=sep,
                    verbose=False,
                )
            
            self.progress_bar.setVisible(False)
            
            if self._disruption_df is None or self._disruption_df.empty:
                self.Warning.no_valid()
                self._send_outputs(None, None, None)
                return
            
            # Check for valid values
            valid_count = self._disruption_df['cd_index'].notna().sum()
            cited_count = (self._disruption_df['n_citing'] > 0).sum()
            
            if cited_count == 0:
                self.Warning.few_internal()
            
            # Step 2: Aggregate if needed
            self._aggregated_df = None
            analysis_type = self.analysis_type
            
            if analysis_type > 0:
                self._aggregated_df = self._compute_aggregation(analysis_type, id_col, sep)
            
            # Update display
            self._update_results_display()
            self._update_plots()
            
            # Send outputs
            self._send_outputs_from_results(id_col)
            
            self.Information.computed(len(self._disruption_df), cited_count)
            self.status_label.setText(
                f"Computed disruption for {len(self._disruption_df)} documents ({cited_count} with internal citations)"
            )
            
        except Exception as e:
            import traceback
            logger.error(f"Disruption error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
            self._send_outputs(None, None, None)
        finally:
            self.progress_bar.setVisible(False)
    
    def _compute_aggregation(self, analysis_type: int, id_col: str, sep: str) -> Optional[pd.DataFrame]:
        """Compute aggregated disruption."""
        if self._df is None or self._disruption_df is None:
            return None
        
        df = self._df
        
        try:
            if analysis_type == 1:  # Sources
                entity_col = self._find_column(df, [
                    'Source title', 'Source Title', 'SO', 'Journal',
                    'primary_location.source.display_name'
                ])
            elif analysis_type == 2:  # Authors
                entity_col = self._find_column(df, [
                    'Authors', 'AU', 'Author', 'authorships.author.display_name'
                ])
            elif analysis_type == 3:  # Countries
                entity_col = self._find_column(df, [
                    'Countries', 'Country', 'CU', 'authorships.countries'
                ])
            elif analysis_type == 4:  # Affiliations
                entity_col = self._find_column(df, [
                    'Affiliations', 'Affiliation', 'C1', 
                    'authorships.institutions.display_name'
                ])
            elif analysis_type == 5:  # Years
                entity_col = self._find_column(df, [
                    'Year', 'PY', 'Publication Year', 'publication_year'
                ])
            else:
                return None
            
            if entity_col is None:
                return None
            
            # Use different separator for entity types
            if analysis_type in [1, 5]:  # Sources and Years are single-value
                entity_sep = "|||"
            else:
                entity_sep = sep
            
            return aggregate_disruption_by_entity(
                self._disruption_df, df, entity_col, id_col,
                sep=entity_sep, min_docs=self.min_documents if analysis_type != 5 else 1
            )
        
        except Exception as e:
            logger.warning(f"Aggregation error: {e}")
            return None
    
    def _update_results_display(self):
        """Update results table."""
        # Decide which dataframe to show
        if self.analysis_type == 0:
            df = self._disruption_df
        else:
            df = self._aggregated_df if self._aggregated_df is not None else self._disruption_df
        
        if df is None or df.empty:
            return
        
        self.results_table.setSortingEnabled(False)
        self.results_table.clear()
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        for i, (idx, row) in enumerate(df.iterrows()):
            for j, col in enumerate(df.columns):
                val = row[col]
                
                if pd.isna(val):
                    item = QTableWidgetItem("")
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    if isinstance(val, (float, np.floating)):
                        item = NumericTableWidgetItem(f"{val:.4f}", float(val))
                    else:
                        item = NumericTableWidgetItem(str(val), float(val))
                    
                    # Color CD index cells
                    if 'cd_index' in col.lower():
                        if val < -0.25:
                            item.setBackground(QColor('#fecaca'))
                        elif val > 0.25:
                            item.setBackground(QColor('#bbf7d0'))
                else:
                    item = QTableWidgetItem(str(val)[:50])
                
                self.results_table.setItem(i, j, item)
        
        self.results_table.resizeColumnsToContents()
        self.results_table.setSortingEnabled(True)
        self.export_btn.setEnabled(True)
    
    def _update_plots(self):
        """Update visualizations."""
        if not HAS_PYQTGRAPH or not self.show_visualization:
            return
        
        # Distribution plot - pass full DataFrame for selection support
        if self._disruption_df is not None:
            self.dist_plot.plot_distribution(self._disruption_df, "cd_index", self.show_statistics)
        
        # Ranking plot
        if self.analysis_type == 0 and self._disruption_df is not None:
            # Show top documents
            df = self._disruption_df.dropna(subset=['cd_index']).copy()
            df = df.reset_index(drop=True)  # Ensure clean indices
            self.rank_plot.plot_ranking(df, 'doc_id', 'cd_index', self.top_n)
        elif self._aggregated_df is not None:
            # Find entity and metric columns
            entity_col = self._aggregated_df.columns[0]
            metric_col = None
            for col in self._aggregated_df.columns:
                col_str = str(col).lower()
                if 'cd_index' in col_str and 'mean' in col_str:
                    metric_col = col
                    break
            # Fallback
            if metric_col is None:
                for col in self._aggregated_df.columns:
                    if 'cd_index' in str(col).lower():
                        metric_col = col
                        break
            if metric_col:
                self.rank_plot.plot_ranking(self._aggregated_df, entity_col, metric_col, self.top_n)
    
    def _on_selection(self, indices: List):
        """Handle selection from plots."""
        self._selected_indices = indices
        
        # Clear other plots' selections
        if HAS_PYQTGRAPH:
            sender = self.sender()
            if sender != self.dist_plot and hasattr(self.dist_plot, 'clear_selection'):
                self.dist_plot.clear_selection()
            if sender != self.rank_plot and hasattr(self.rank_plot, 'clear_selection'):
                self.rank_plot.clear_selection()
        
        # Send selected documents
        self._send_selected()
    
    def _send_selected(self):
        """Send selected documents to output."""
        if not self._selected_indices or self._disruption_df is None:
            self.Outputs.selected.send(None)
            return
        
        try:
            # Get selected rows from disruption_df
            selected_df = self._disruption_df.iloc[self._selected_indices].copy()
            selected_table = self._df_to_table(selected_df)
            self.Outputs.selected.send(selected_table)
        except Exception as e:
            logger.warning(f"Selection error: {e}")
            self.Outputs.selected.send(None)
    
    def _send_outputs_from_results(self, id_col: str):
        """Create and send output tables."""
        # Disruption output
        disruption_table = None
        if self._disruption_df is not None:
            disruption_table = self._df_to_table(self._disruption_df)
        
        # Aggregated output
        aggregated_table = None
        if self._aggregated_df is not None:
            aggregated_table = self._df_to_table(self._aggregated_df)
        
        # Enhanced data output - merge disruption into original
        enhanced_table = None
        if self.add_to_dataset and self._disruption_df is not None and self._df is not None:
            from biblium.disruption import add_disruption_to_df
            enhanced_df = add_disruption_to_df(
                self._df, self._disruption_df, id_col,
                columns=['cd_index', 'di_index', 'interpretation']
            )
            enhanced_table = self._df_to_table(enhanced_df)
        
        self._send_outputs(disruption_table, aggregated_table, enhanced_table)
    
    def _df_to_table(self, df: pd.DataFrame) -> Table:
        """Convert DataFrame to Orange Table."""
        # Separate numeric and string columns
        attrs = []
        metas = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                attrs.append(ContinuousVariable(col))
            else:
                metas.append(StringVariable(col))
        
        domain = Domain(attrs, metas=metas)
        
        # Build arrays
        X = np.zeros((len(df), len(attrs)))
        for i, attr in enumerate(attrs):
            col_data = df[attr.name].values
            X[:, i] = np.where(pd.isna(col_data), np.nan, col_data)
        
        if metas:
            M = np.array([
                [str(row.get(m.name, "")) for m in metas]
                for _, row in df.iterrows()
            ], dtype=object)
        else:
            M = None
        
        return Table.from_numpy(domain, X, metas=M)
    
    def _send_outputs(self, disruption: Optional[Table], aggregated: Optional[Table], 
                      enhanced: Optional[Table]):
        """Send all outputs."""
        self.Outputs.disruption.send(disruption)
        self.Outputs.aggregated.send(aggregated)
        self.Outputs.enhanced.send(enhanced)
    
    def _export_results(self):
        """Export results to Excel."""
        df = self._disruption_df if self.analysis_type == 0 else self._aggregated_df
        
        if df is None or df.empty:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Disruption Index", "disruption_index.xlsx",
            "Excel files (*.xlsx);;All files (*)"
        )
        
        if not filepath:
            return
        
        try:
            df.to_excel(filepath, index=False)
            QMessageBox.information(self, "Exported", f"Results saved to {filepath}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not export: {e}")


if __name__ == "__main__":
    WidgetPreview(OWDisruptionIndex).run()
