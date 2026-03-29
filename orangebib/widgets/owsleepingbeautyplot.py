# -*- coding: utf-8 -*-
"""
Sleeping Beauty Plot Widget
===========================
Interactive PyQtGraph visualization of Sleeping Beauty detection results.

Features:
- Native Qt plotting with PyQtGraph (no matplotlib)
- Separate individual plots in tabs
- Hover tooltips showing paper details
- Click selection to output selected papers
- No gridlines, clean visualization
"""

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QAbstractItemView, QWidget,
    QTabWidget, QToolTip, QSplitter, QFrame,
)
from AnyQt.QtCore import Qt, QPointF, pyqtSignal
from AnyQt.QtGui import QFont, QColor, QPen, QBrush

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT PYQTGRAPH
# =============================================================================

try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background='w', foreground='k')
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    logger.warning("PyQtGraph not available for plotting")


# =============================================================================
# INTERACTIVE SCATTER PLOT ITEM
# =============================================================================

class HoverScatterPlot(pg.PlotWidget):
    """Scatter plot with hover and selection capabilities."""
    
    selectionChanged = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Clean styling
        self.setBackground('w')
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=False, y=False)
        
        self._data_points = []
        self._scatter = None
        self._selected_indices = set()
        self._original_brushes = []
        self._original_sizes = []
        
        # Selection color
        self._selection_color = '#ef4444'
        self._default_size = 10
        self._selected_size = 14
        
        # Enable mouse tracking
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
    
    def set_data(self, x_data: List[float], y_data: List[float],
                 colors: Optional[List[str]] = None,
                 sizes: Optional[List[float]] = None,
                 tooltips: Optional[List[str]] = None,
                 data_indices: Optional[List[int]] = None):
        """Set scatter plot data."""
        self.clear()
        self._data_points = []
        self._selected_indices = set()
        
        if not x_data or not y_data:
            return
        
        n = len(x_data)
        
        # Store data for hover/selection
        for i in range(n):
            self._data_points.append({
                'x': x_data[i],
                'y': y_data[i],
                'tooltip': tooltips[i] if tooltips else f"({x_data[i]:.2f}, {y_data[i]:.2f})",
                'data_index': data_indices[i] if data_indices else i
            })
        
        # Default colors and sizes
        if colors is None:
            colors = ['#7c3aed'] * n
        if sizes is None:
            sizes = [self._default_size] * n
        
        self._original_brushes = [pg.mkBrush(c) for c in colors]
        self._original_sizes = sizes.copy()
        
        # Create scatter
        self._scatter = pg.ScatterPlotItem(
            x=x_data, y=y_data,
            size=sizes,
            brush=self._original_brushes,
            pen=pg.mkPen('w', width=1),
            hoverable=True,
        )
        self.addItem(self._scatter)
    
    def _on_mouse_moved(self, pos):
        """Show tooltip on hover."""
        if self._scatter is None or not self._data_points:
            return
        
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        
        # Find closest point
        min_dist = float('inf')
        closest = None
        
        for pt in self._data_points:
            dist = ((pt['x'] - mouse_point.x()) ** 2 + 
                   (pt['y'] - mouse_point.y()) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest = pt
        
        # Check threshold
        view_range = self.viewRange()
        x_range = max(0.01, view_range[0][1] - view_range[0][0])
        y_range = max(0.01, view_range[1][1] - view_range[1][0])
        threshold = max(x_range, y_range) * 0.03
        
        if closest and min_dist < threshold:
            global_pos = self.mapToGlobal(self.mapFromScene(pos))
            QToolTip.showText(global_pos, closest['tooltip'])
        else:
            QToolTip.hideText()
    
    def _on_mouse_clicked(self, event):
        """Handle click for selection."""
        if self._scatter is None or not self._data_points:
            return
        
        pos = event.scenePos()
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        
        # Find clicked point
        min_dist = float('inf')
        clicked = None
        clicked_idx = None
        
        for i, pt in enumerate(self._data_points):
            dist = ((pt['x'] - mouse_point.x()) ** 2 + 
                   (pt['y'] - mouse_point.y()) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                clicked = pt
                clicked_idx = i
        
        # Check threshold
        view_range = self.viewRange()
        x_range = max(0.01, view_range[0][1] - view_range[0][0])
        y_range = max(0.01, view_range[1][1] - view_range[1][0])
        threshold = max(x_range, y_range) * 0.03
        
        if clicked and min_dist < threshold:
            data_idx = clicked['data_index']
            
            # Toggle selection
            if data_idx in self._selected_indices:
                self._selected_indices.remove(data_idx)
            else:
                self._selected_indices.add(data_idx)
            
            self._update_selection_visual()
            self.selectionChanged.emit(list(self._selected_indices))
    
    def _update_selection_visual(self):
        """Update scatter appearance for selection."""
        if self._scatter is None:
            return
        
        spots = []
        for i, pt in enumerate(self._data_points):
            if pt['data_index'] in self._selected_indices:
                spots.append({
                    'pos': (pt['x'], pt['y']),
                    'brush': pg.mkBrush(self._selection_color),
                    'size': self._selected_size,
                    'pen': pg.mkPen('w', width=2)
                })
            else:
                spots.append({
                    'pos': (pt['x'], pt['y']),
                    'brush': self._original_brushes[i],
                    'size': self._original_sizes[i],
                    'pen': pg.mkPen('w', width=1)
                })
        
        self._scatter.setData(spots=spots)
    
    def clear_selection(self):
        """Clear all selections."""
        self._selected_indices = set()
        self._update_selection_visual()
        self.selectionChanged.emit([])
    
    def get_selected_indices(self) -> List[int]:
        """Get selected data indices."""
        return list(self._selected_indices)


# =============================================================================
# INTERACTIVE HISTOGRAM
# =============================================================================

class HoverHistogram(pg.PlotWidget):
    """Histogram with hover tooltips."""
    
    selectionChanged = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setBackground('w')
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=False, y=False)
        
        self._bins = []
        self._bar_item = None
        
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
    
    def set_histogram(self, values: List[float], n_bins: int = 20,
                      color: str = '#3b82f6', median_line: bool = True):
        """Create histogram from values."""
        self.clear()
        self._bins = []
        
        if not values:
            return
        
        values = [v for v in values if pd.notna(v)]
        if not values:
            return
        
        # Compute histogram
        counts, bin_edges = np.histogram(values, bins=n_bins)
        
        # Store bin info for tooltips
        for i in range(len(counts)):
            self._bins.append({
                'left': bin_edges[i],
                'right': bin_edges[i + 1],
                'count': counts[i],
                'center': (bin_edges[i] + bin_edges[i + 1]) / 2
            })
        
        # Create bar graph
        width = bin_edges[1] - bin_edges[0]
        x = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))]
        
        self._bar_item = pg.BarGraphItem(
            x=x, height=counts, width=width * 0.9,
            brush=pg.mkBrush(color),
            pen=pg.mkPen('w', width=1)
        )
        self.addItem(self._bar_item)
        
        # Add median line
        if median_line and values:
            median_val = np.median(values)
            median_line_item = pg.InfiniteLine(
                pos=median_val, angle=90,
                pen=pg.mkPen('#ef4444', width=2, style=Qt.DashLine)
            )
            self.addItem(median_line_item)
            
            # Median label
            text = pg.TextItem(f'Median: {median_val:.1f}', color='#ef4444', anchor=(0, 1))
            text.setPos(median_val + width * 0.5, max(counts) * 0.95)
            self.addItem(text)
    
    def _on_mouse_moved(self, pos):
        """Show tooltip on hover."""
        if not self._bins:
            return
        
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        
        for bin_info in self._bins:
            if bin_info['left'] <= mouse_point.x() <= bin_info['right']:
                if 0 <= mouse_point.y() <= bin_info['count']:
                    global_pos = self.mapToGlobal(self.mapFromScene(pos))
                    tooltip = (f"Range: {bin_info['left']:.2f} - {bin_info['right']:.2f}\n"
                              f"Count: {bin_info['count']}")
                    QToolTip.showText(global_pos, tooltip)
                    return
        
        QToolTip.hideText()


# =============================================================================
# INTERACTIVE BAR CHART
# =============================================================================

class HoverBarChart(pg.PlotWidget):
    """Horizontal bar chart with hover and selection."""
    
    selectionChanged = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setBackground('w')
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=False, y=False)
        
        self._bars = []
        self._bar_items = []
        self._selected_indices = set()
        self._default_color = '#f97316'
        
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
    
    def set_bars(self, values: List[float], labels: List[str],
                 tooltips: Optional[List[str]] = None,
                 data_indices: Optional[List[int]] = None,
                 color: str = '#f97316'):
        """Set bar chart data (vertical bars)."""
        self.clear()
        self._bars = []
        self._bar_items = []
        self._selected_indices = set()
        self._default_color = color
        
        if not values:
            return
        
        n = len(values)
        
        # Store bar info
        for i in range(n):
            self._bars.append({
                'x': i,
                'value': values[i],
                'label': labels[i],
                'tooltip': tooltips[i] if tooltips else f"{labels[i]}: {values[i]}",
                'data_index': data_indices[i] if data_indices else i
            })
        
        # Create bars
        bar_item = pg.BarGraphItem(
            x=list(range(n)), height=values, width=0.7,
            brush=pg.mkBrush(color),
            pen=pg.mkPen('w', width=1)
        )
        self.addItem(bar_item)
        self._bar_items.append(bar_item)
        
        # Set x-axis tick labels (rotate for readability)
        axis = self.getAxis('bottom')
        ticks = [(i, str(labels[i])[:10]) for i in range(n)]
        axis.setTicks([ticks])
    
    def _on_mouse_moved(self, pos):
        """Show tooltip on hover."""
        if not self._bars:
            return
        
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        
        for bar in self._bars:
            if (bar['x'] - 0.4 <= mouse_point.x() <= bar['x'] + 0.4 and
                0 <= mouse_point.y() <= bar['value']):
                global_pos = self.mapToGlobal(self.mapFromScene(pos))
                QToolTip.showText(global_pos, bar['tooltip'])
                return
        
        QToolTip.hideText()
    
    def _on_mouse_clicked(self, event):
        """Handle click for selection."""
        if not self._bars:
            return
        
        pos = event.scenePos()
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        
        for bar in self._bars:
            if (bar['x'] - 0.4 <= mouse_point.x() <= bar['x'] + 0.4 and
                0 <= mouse_point.y() <= bar['value']):
                data_idx = bar['data_index']
                
                if data_idx in self._selected_indices:
                    self._selected_indices.remove(data_idx)
                else:
                    self._selected_indices.add(data_idx)
                
                self.selectionChanged.emit(list(self._selected_indices))
                return
    
    def get_selected_indices(self) -> List[int]:
        return list(self._selected_indices)
    
    def clear_selection(self):
        self._selected_indices = set()


# =============================================================================
# TRAJECTORY PLOT
# =============================================================================

class TrajectoryPlot(pg.PlotWidget):
    """Citation trajectory plot."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setBackground('w')
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=False, y=False)
        
        self._year_data = []
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
    
    def plot_single(self, years: List[int], citations: List[int],
                    title: str = "", pub_year: int = None, 
                    awakening_year: int = None, b_coef: float = None):
        """Plot single paper trajectory."""
        self.clear()
        self._year_data = list(zip(years, citations))
        
        if not years:
            return
        
        # Bar plot
        bar = pg.BarGraphItem(
            x=years, height=citations, width=0.7,
            brush=pg.mkBrush('#7c3aed'),
            pen=pg.mkPen('w', width=1)
        )
        self.addItem(bar)
        
        # Line
        line = pg.PlotDataItem(
            x=years, y=citations,
            pen=pg.mkPen('#4c1d95', width=2),
            symbol='o', symbolSize=6,
            symbolBrush='#4c1d95', symbolPen='w'
        )
        self.addItem(line)
        
        # Mark awakening year with star
        if awakening_year and awakening_year in years:
            idx = years.index(awakening_year)
            marker = pg.ScatterPlotItem(
                x=[awakening_year], y=[citations[idx]],
                size=18, symbol='star',
                brush=pg.mkBrush('#22c55e'),
                pen=pg.mkPen('w', width=2)
            )
            self.addItem(marker)
            
            # Vertical line
            vline = pg.InfiniteLine(
                pos=awakening_year, angle=90,
                pen=pg.mkPen('#22c55e', width=2, style=Qt.DashLine)
            )
            self.addItem(vline)
        
        # Publication year line
        if pub_year:
            pub_line = pg.InfiniteLine(
                pos=pub_year, angle=90,
                pen=pg.mkPen('#6b7280', width=1.5, style=Qt.DotLine)
            )
            self.addItem(pub_line)
        
        # Title
        title_str = title[:70] + "..." if len(title) > 70 else title
        if b_coef is not None:
            title_str += f" (B={b_coef:.1f})"
        self.setTitle(title_str)
        self.setLabel('bottom', 'Year')
        self.setLabel('left', 'Citations')
    
    def plot_multi(self, trajectories: List[Dict], normalize: bool = True):
        """Plot multiple trajectories."""
        self.clear()
        self._year_data = []
        
        colors = ['#7c3aed', '#ef4444', '#22c55e', '#3b82f6', '#f59e0b',
                  '#ec4899', '#06b6d4', '#8b5cf6', '#10b981', '#f97316']
        
        legend = self.addLegend(offset=(10, 10))
        
        for i, traj in enumerate(trajectories[:10]):
            years = traj.get('years', [])
            citations = traj.get('citations', [])
            title = traj.get('title', f'Paper {i+1}')[:25]
            
            if not years:
                continue
            
            # Normalize
            y_vals = citations
            if normalize and max(citations) > 0:
                y_vals = [c / max(citations) for c in citations]
            
            color = colors[i % len(colors)]
            
            line = pg.PlotDataItem(
                x=years, y=y_vals,
                pen=pg.mkPen(color, width=2),
                symbol='o', symbolSize=5,
                symbolBrush=color, symbolPen='w',
                name=title
            )
            self.addItem(line)
            
            # Star at awakening
            aw_year = traj.get('awakening_year')
            if aw_year and aw_year in years:
                idx = years.index(aw_year)
                marker = pg.ScatterPlotItem(
                    x=[aw_year], y=[y_vals[idx]],
                    size=10, symbol='star',
                    brush=pg.mkBrush(color), pen=pg.mkPen('w')
                )
                self.addItem(marker)
        
        self.setTitle('Citation Trajectories (★ = awakening)')
        self.setLabel('bottom', 'Year')
        self.setLabel('left', 'Normalized Citations' if normalize else 'Citations')
    
    def _on_mouse_moved(self, pos):
        """Show tooltip."""
        if not self._year_data:
            return
        
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        
        for year, cites in self._year_data:
            if abs(mouse_point.x() - year) < 0.5:
                global_pos = self.mapToGlobal(self.mapFromScene(pos))
                QToolTip.showText(global_pos, f"Year: {year}\nCitations: {cites}")
                return
        
        QToolTip.hideText()


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWSleepingBeautyPlot(OWWidget):
    """Interactive PyQtGraph visualization of Sleeping Beauty results."""
    
    name = "Sleeping Beauty Plot"
    description = "Interactive visualization of sleeping beauty detection results"
    icon = "icons/sleeping_beauty_plot.svg"
    priority = 70
    keywords = ["sleeping beauty", "plot", "visualization", "citation", "trajectory"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Sleeping beauty detection results")
    
    class Outputs:
        selected = Output("Selected", Table, doc="Selected sleeping beauties")
    
    # Settings
    selected_paper_index = settings.Setting(0)
    normalize_trajectories = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_pyqtgraph = Msg("PyQtGraph required for plotting")
        missing_columns = Msg("Data must have beauty_coefficient column")
    
    class Warning(OWWidget.Warning):
        no_citation_history = Msg("No citation_history - trajectory plots unavailable")
    
    class Information(OWWidget.Information):
        loaded = Msg("Loaded {} sleeping beauties")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._selected_indices: List[int] = []
        
        self._setup_control_area()
        self._setup_main_area()
    
    def _setup_control_area(self):
        """Build control area."""
        # Info header
        info_box = gui.widgetBox(self.controlArea, "")
        status = "✓ PyQtGraph" if HAS_PYQTGRAPH else "⚠ PyQtGraph required"
        
        info_label = QLabel(
            f"<b>📊 Sleeping Beauty Plot</b><br>"
            f"<small>Interactive visualization<br>"
            f"Click to select, hover for details<br>"
            f"<i>{status}</i></small>"
        )
        info_label.setStyleSheet("color: #7c3aed; background-color: #ede9fe; padding: 8px; border-radius: 4px;")
        info_box.layout().addWidget(info_label)
        
        # Options
        options_box = gui.widgetBox(self.controlArea, "⚙️ Options")
        
        gui.checkBox(
            options_box, self, "normalize_trajectories",
            "Normalize trajectories (0-1)",
            callback=self._update_multi_trajectory
        )
        
        # Paper selection for single trajectory
        paper_box = gui.widgetBox(self.controlArea, "📄 Single Paper")
        
        self.paper_list = QListWidget()
        self.paper_list.setMaximumHeight(180)
        self.paper_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.paper_list.currentRowChanged.connect(self._on_paper_selected)
        paper_box.layout().addWidget(self.paper_list)
        
        # Selection controls
        sel_box = gui.widgetBox(self.controlArea, "✓ Selection")
        
        self.sel_label = QLabel("Selected: 0 papers")
        sel_box.layout().addWidget(self.sel_label)
        
        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self._clear_selection)
        sel_box.layout().addWidget(clear_btn)
        
        export_btn = QPushButton("Export Selected")
        export_btn.clicked.connect(self._export_selected)
        sel_box.layout().addWidget(export_btn)
        
        self.controlArea.layout().addStretch(1)
    
    def _setup_main_area(self):
        """Build main area with tabbed plots."""
        main_layout = QVBoxLayout()
        self.mainArea.layout().addLayout(main_layout)
        
        # Status
        self.status_label = QLabel("Load sleeping beauty data to visualize")
        self.status_label.setStyleSheet("font-size: 13px; color: #6c757d; padding: 4px;")
        main_layout.addWidget(self.status_label)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, 1)
        
        if HAS_PYQTGRAPH:
            # Tab 1: B Coefficient Distribution
            self.hist_b = HoverHistogram()
            self.tab_widget.addTab(self.hist_b, "📊 B Distribution")
            
            # Tab 2: Sleep Duration vs B Coefficient
            self.scatter_sleep_b = HoverScatterPlot()
            self.scatter_sleep_b.selectionChanged.connect(self._on_selection_changed)
            self.tab_widget.addTab(self.scatter_sleep_b, "🔵 Sleep vs B")
            
            # Tab 3: Publication Year
            self.bar_pubyear = HoverBarChart()
            self.bar_pubyear.selectionChanged.connect(self._on_selection_changed)
            self.tab_widget.addTab(self.bar_pubyear, "📅 By Year")
            
            # Tab 4: Awakening Intensity Distribution
            self.hist_intensity = HoverHistogram()
            self.tab_widget.addTab(self.hist_intensity, "⚡ Intensity")
            
            # Tab 5: Multi Trajectories
            self.traj_multi = TrajectoryPlot()
            self.tab_widget.addTab(self.traj_multi, "📈 Trajectories")
            
            # Tab 6: Single Paper
            self.traj_single = TrajectoryPlot()
            self.tab_widget.addTab(self.traj_single, "📄 Single")
        else:
            placeholder = QLabel("PyQtGraph required\n\npip install pyqtgraph")
            placeholder.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(placeholder, "Error")
    
    def _on_paper_selected(self, row: int):
        """Handle paper selection."""
        self.selected_paper_index = row
        self._update_single_trajectory()
    
    def _on_selection_changed(self, indices: List[int]):
        """Handle selection from plots."""
        self._selected_indices = indices
        self.sel_label.setText(f"Selected: {len(indices)} papers")
        self._send_selected()
    
    def _clear_selection(self):
        """Clear all selections."""
        self._selected_indices = []
        
        if HAS_PYQTGRAPH:
            self.scatter_sleep_b.clear_selection()
            self.bar_pubyear.clear_selection()
        
        self.sel_label.setText("Selected: 0 papers")
        self._send_selected()
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive input data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._selected_indices = []
        
        if not HAS_PYQTGRAPH:
            self.Error.no_pyqtgraph()
            return
        
        if data is None:
            self.Error.no_data()
            self._clear_plots()
            return
        
        # Convert to DataFrame
        self._df = self._table_to_df(data)
        
        # Check required columns
        if "beauty_coefficient" not in self._df.columns:
            self.Error.missing_columns()
            return
        
        if "citation_history" not in self._df.columns:
            self.Warning.no_citation_history()
        
        # Populate paper list
        self._populate_paper_list()
        
        # Update all plots
        self._update_all_plots()
        
        self.Information.loaded(len(self._df))
        self.status_label.setText(f"Showing {len(self._df)} sleeping beauties")
        self.sel_label.setText("Selected: 0 papers")
    
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
    
    def _populate_paper_list(self):
        """Populate paper list."""
        self.paper_list.clear()
        
        if self._df is None:
            return
        
        for i, (idx, row) in enumerate(self._df.iterrows()):
            title = str(row.get("title", f"Paper {i}"))[:35]
            b = row.get("beauty_coefficient", 0)
            item = QListWidgetItem(f"{title}... (B={b:.1f})")
            item.setData(Qt.UserRole, i)
            self.paper_list.addItem(item)
    
    def _clear_plots(self):
        """Clear all plots."""
        if HAS_PYQTGRAPH:
            self.hist_b.clear()
            self.scatter_sleep_b.clear()
            self.bar_pubyear.clear()
            self.hist_intensity.clear()
            self.traj_multi.clear()
            self.traj_single.clear()
        
        self.status_label.setText("Load sleeping beauty data to visualize")
    
    def _update_all_plots(self):
        """Update all plots."""
        if self._df is None:
            return
        
        self._update_b_histogram()
        self._update_sleep_vs_b()
        self._update_pubyear_bars()
        self._update_intensity_histogram()
        self._update_multi_trajectory()
        self._update_single_trajectory()
    
    def _update_b_histogram(self):
        """Update B coefficient histogram."""
        if self._df is None:
            return
        
        values = self._df["beauty_coefficient"].dropna().tolist()
        self.hist_b.set_histogram(values, n_bins=20, color='#3b82f6', median_line=True)
        self.hist_b.setTitle('Distribution of Beauty Coefficients')
        self.hist_b.setLabel('bottom', 'Beauty Coefficient')
        self.hist_b.setLabel('left', 'Count')
    
    def _update_sleep_vs_b(self):
        """Update Sleep Duration vs B coefficient scatter."""
        if self._df is None:
            return
        
        df = self._df
        
        x = df.get("sleep_duration", pd.Series([0] * len(df))).fillna(0).tolist()
        y = df.get("beauty_coefficient", pd.Series([0] * len(df))).fillna(0).tolist()
        
        # Color by citations
        citations = df.get("total_citations", pd.Series([0] * len(df))).fillna(0)
        max_c = max(citations) if max(citations) > 0 else 1
        
        colors = []
        for c in citations:
            ratio = c / max_c
            # Dark purple to bright green gradient
            r = int(76 * (1 - ratio) + 34 * ratio)
            g = int(29 * (1 - ratio) + 197 * ratio)
            b = int(149 * (1 - ratio) + 94 * ratio)
            colors.append(f'#{r:02x}{g:02x}{b:02x}')
        
        # Tooltips
        tooltips = []
        for i, (idx, row) in enumerate(df.iterrows()):
            title = str(row.get("title", ""))[:50]
            tooltips.append(
                f"{title}...\n"
                f"B: {row.get('beauty_coefficient', 0):.1f}\n"
                f"Sleep: {row.get('sleep_duration', 0):.0f} yrs\n"
                f"Citations: {row.get('total_citations', 0):.0f}"
            )
        
        self.scatter_sleep_b.set_data(x, y, colors=colors, tooltips=tooltips)
        self.scatter_sleep_b.setTitle('Sleep Duration vs Beauty Coefficient')
        self.scatter_sleep_b.setLabel('bottom', 'Sleep Duration (Years)')
        self.scatter_sleep_b.setLabel('left', 'Beauty Coefficient')
    
    def _update_pubyear_bars(self):
        """Update publication year bar chart."""
        if self._df is None:
            return
        
        df = self._df
        
        if "publication_year" not in df.columns:
            return
        
        # Count by year
        year_counts = df["publication_year"].value_counts().sort_index()
        
        years = [int(y) for y in year_counts.index.tolist()]
        counts = year_counts.values.tolist()
        
        tooltips = [f"Year: {y}\nCount: {c}" for y, c in zip(years, counts)]
        
        self.bar_pubyear.set_bars(counts, years, tooltips=tooltips, color='#f97316')
        self.bar_pubyear.setTitle('Sleeping Beauties by Publication Year')
        self.bar_pubyear.setLabel('bottom', 'Publication Year')
        self.bar_pubyear.setLabel('left', 'Number of Sleeping Beauties')
    
    def _update_intensity_histogram(self):
        """Update awakening intensity histogram."""
        if self._df is None:
            return
        
        if "awakening_intensity" not in self._df.columns:
            return
        
        # Filter out inf values
        values = self._df["awakening_intensity"].replace([np.inf, -np.inf], np.nan).dropna().tolist()
        
        if values:
            self.hist_intensity.set_histogram(values, n_bins=15, color='#22c55e', median_line=True)
        
        self.hist_intensity.setTitle('Distribution of Awakening Intensity')
        self.hist_intensity.setLabel('bottom', 'Awakening Intensity (Citation Ratio)')
        self.hist_intensity.setLabel('left', 'Count')
    
    def _update_multi_trajectory(self):
        """Update multi-trajectory plot."""
        if self._df is None or "citation_history" not in self._df.columns:
            return
        
        trajectories = []
        for i, (idx, row) in enumerate(self._df.head(8).iterrows()):
            history = row.get("citation_history")
            if history is None:
                continue
            
            # Parse if string
            if isinstance(history, str):
                try:
                    import json
                    history = json.loads(history.replace("'", '"'))
                except:
                    continue
            
            if not isinstance(history, dict):
                continue
            
            years = sorted([int(y) for y in history.keys()])
            citations = [int(history[y]) for y in years]
            
            trajectories.append({
                'years': years,
                'citations': citations,
                'title': str(row.get("title", f"Paper {i}")),
                'awakening_year': int(row.get("awakening_year")) if pd.notna(row.get("awakening_year")) else None
            })
        
        if trajectories:
            self.traj_multi.plot_multi(trajectories, normalize=self.normalize_trajectories)
    
    def _update_single_trajectory(self):
        """Update single paper trajectory plot."""
        if self._df is None or "citation_history" not in self._df.columns:
            return
        
        idx = min(max(0, self.selected_paper_index), len(self._df) - 1)
        row = self._df.iloc[idx]
        
        history = row.get("citation_history")
        if history is None:
            self.traj_single.clear()
            return
        
        # Parse if string
        if isinstance(history, str):
            try:
                import json
                history = json.loads(history.replace("'", '"'))
            except:
                self.traj_single.clear()
                return
        
        if not isinstance(history, dict):
            self.traj_single.clear()
            return
        
        years = sorted([int(y) for y in history.keys()])
        citations = [int(history[y]) for y in years]
        
        self.traj_single.plot_single(
            years, citations,
            title=str(row.get("title", "")),
            pub_year=int(row.get("publication_year")) if pd.notna(row.get("publication_year")) else None,
            awakening_year=int(row.get("awakening_year")) if pd.notna(row.get("awakening_year")) else None,
            b_coef=row.get("beauty_coefficient")
        )
    
    def _send_selected(self):
        """Send selected papers to output."""
        if self._df is None or not self._selected_indices:
            self.Outputs.selected.send(None)
            return
        
        selected_df = self._df.iloc[self._selected_indices]
        if selected_df.empty:
            self.Outputs.selected.send(None)
            return
        
        # Build output table
        attr_names = ["publication_year", "beauty_coefficient", "sleep_duration",
                      "awakening_year", "total_citations", "awakening_intensity"]
        attrs = [ContinuousVariable(n) for n in attr_names if n in selected_df.columns]
        
        meta_names = ["title", "paper_id"]
        metas = [StringVariable(n) for n in meta_names if n in selected_df.columns]
        
        domain = Domain(attrs, metas=metas)
        
        X = np.zeros((len(selected_df), len(attrs)))
        for i, (idx, row) in enumerate(selected_df.iterrows()):
            for j, attr in enumerate(attrs):
                val = row.get(attr.name, np.nan)
                X[i, j] = val if pd.notna(val) and val != float('inf') else np.nan
        
        if metas:
            metas_arr = np.array([
                [str(row.get(m.name, "")) for m in metas]
                for idx, row in selected_df.iterrows()
            ], dtype=object)
        else:
            metas_arr = None
        
        sel_table = Table.from_numpy(domain, X, metas=metas_arr)
        self.Outputs.selected.send(sel_table)
    
    def _export_selected(self):
        """Export selected papers."""
        if self._df is None:
            QMessageBox.warning(self, "No Data", "No data loaded")
            return
        
        if not self._selected_indices:
            QMessageBox.warning(self, "No Selection", "No papers selected")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Selected", "selected_beauties.csv",
            "CSV files (*.csv);;All files (*)"
        )
        
        if not filepath:
            return
        
        try:
            sel_df = self._df.iloc[self._selected_indices]
            export_df = sel_df.drop(columns=["citation_history"], errors="ignore")
            export_df.to_csv(filepath, index=False)
            QMessageBox.information(self, "Exported", f"Saved {len(sel_df)} papers")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Export failed: {e}")


if __name__ == "__main__":
    WidgetPreview(OWSleepingBeautyPlot).run()
