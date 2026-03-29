# -*- coding: utf-8 -*-
"""
Citation Patterns Widget
========================
Classify papers by citation trajectory using Biblium's implementation.

Features:
- Interactive PyQtGraph visualization with hover tooltips and click selection
- Pattern-specific color scheme throughout
- Selected Documents output for downstream analysis
"""

import json
import logging
from typing import Optional, List, Dict, Set

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem,
    QTabWidget, QFrame, QProgressBar,
    QFileDialog, QMessageBox, QToolTip,
)
from AnyQt.QtCore import Qt, pyqtSignal
from AnyQt.QtGui import QColor

from Orange.data import Table, Domain, ContinuousVariable, StringVariable, DiscreteVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS
# =============================================================================

try:
    from biblium.citation_patterns import (
        analyze_citation_patterns,
        CitationPatternResult,
        CitationPattern,
    )
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False

try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background='w', foreground='k')
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


# =============================================================================
# PATTERN COLOR SCHEME
# =============================================================================

PATTERN_COLORS = {
    "Evergreen": "#22c55e",         # Green
    "Flash-in-the-pan": "#ef4444",  # Red
    "Delayed Recognition": "#3b82f6", # Blue
    "Sleeping Beauty": "#8b5cf6",   # Purple
    "Normal": "#6b7280",            # Gray
    "Uncited": "#d1d5db",           # Light gray
    "Too Recent": "#f59e0b",        # Amber
}

PATTERN_ORDER = ["Evergreen", "Flash-in-the-pan", "Delayed Recognition", 
                 "Sleeping Beauty", "Normal", "Uncited", "Too Recent"]


# =============================================================================
# INTERACTIVE DISTRIBUTION PLOT
# =============================================================================

class PatternDistributionPlot(pg.PlotWidget):
    """Horizontal bar chart for pattern distribution with hover and selection."""
    
    selectionChanged = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=False, y=False)
        
        self._bars = []
        self._pattern_indices = {}
        self._selected_pattern = None
        
        self.scene().sigMouseMoved.connect(self._on_hover)
        self.scene().sigMouseClicked.connect(self._on_click)
    
    def set_data(self, pattern_counts: Dict[str, int], pattern_indices: Dict[str, List[int]]):
        """Set distribution data."""
        self.clear()
        self._bars = []
        self._pattern_indices = pattern_indices
        self._selected_pattern = None
        
        if not pattern_counts:
            return
        
        # Filter and sort patterns
        patterns = [p for p in PATTERN_ORDER if p in pattern_counts and pattern_counts[p] > 0]
        counts = [pattern_counts[p] for p in patterns]
        
        if not patterns:
            return
        
        max_count = max(counts)
        
        for i, (pattern, count) in enumerate(zip(patterns, counts)):
            color = PATTERN_COLORS.get(pattern, '#6b7280')
            
            bar = pg.BarGraphItem(
                x0=0, x1=count, y=i, height=0.7,
                brush=pg.mkBrush(color),
                pen=pg.mkPen('w', width=1)
            )
            self.addItem(bar)
            
            self._bars.append({
                'item': bar,
                'pattern': pattern,
                'count': count,
                'y': i,
                'color': color
            })
            
            # Label
            label = pg.TextItem(f"{pattern}: {count}", color='k', anchor=(0, 0.5))
            label.setPos(count + max_count * 0.02, i)
            self.addItem(label)
        
        self.setLabel('bottom', 'Count')
        self.setTitle('Citation Pattern Distribution (click to select)')
        self.setXRange(0, max_count * 1.3)
        self.setYRange(-0.5, len(patterns) - 0.5)
        
        # Y-axis ticks
        axis = self.getAxis('left')
        axis.setTicks([[(i, '') for i in range(len(patterns))]])
    
    def _on_hover(self, pos):
        """Show tooltip on hover."""
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        
        for bar in self._bars:
            if (0 <= mouse_point.x() <= bar['count'] and
                bar['y'] - 0.4 <= mouse_point.y() <= bar['y'] + 0.4):
                n_papers = len(self._pattern_indices.get(bar['pattern'], []))
                tooltip = f"{bar['pattern']}: {bar['count']} papers\nClick to select all"
                QToolTip.showText(self.mapToGlobal(self.mapFromScene(pos)), tooltip)
                return
        
        QToolTip.hideText()
    
    def _on_click(self, event):
        """Select pattern on click."""
        pos = event.scenePos()
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        
        for bar in self._bars:
            if (0 <= mouse_point.x() <= bar['count'] and
                bar['y'] - 0.4 <= mouse_point.y() <= bar['y'] + 0.4):
                
                pattern = bar['pattern']
                
                if self._selected_pattern == pattern:
                    self._selected_pattern = None
                    self._update_visual()
                    self.selectionChanged.emit([])
                else:
                    self._selected_pattern = pattern
                    self._update_visual()
                    indices = self._pattern_indices.get(pattern, [])
                    self.selectionChanged.emit(indices)
                return
    
    def _update_visual(self):
        """Update bar highlighting."""
        for bar in self._bars:
            if bar['pattern'] == self._selected_pattern:
                bar['item'].setOpts(pen=pg.mkPen('#000000', width=3))
            else:
                bar['item'].setOpts(pen=pg.mkPen('w', width=1))
    
    def clear_selection(self):
        self._selected_pattern = None
        self._update_visual()


# =============================================================================
# SCATTER PLOT WITH SELECTION
# =============================================================================

class InteractiveScatterPlot(pg.PlotWidget):
    """Scatter plot with hover and click selection, colored by pattern."""
    
    selectionChanged = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=False, y=False)
        
        self._points = []
        self._selected: Set[int] = set()
        self._scatter_item = None
        self._x_col = ""
        self._y_col = ""
        
        self.scene().sigMouseMoved.connect(self._on_hover)
        self.scene().sigMouseClicked.connect(self._on_click)
    
    def set_data(self, df: pd.DataFrame, x_col: str, y_col: str):
        """Set scatter data."""
        self.clear()
        self._points = []
        self._selected = set()
        self._scatter_item = None
        self._x_col = x_col
        self._y_col = y_col
        
        if df is None or df.empty:
            return
        
        if x_col not in df.columns or y_col not in df.columns:
            return
        
        # Build points list - use positional index (i) not DataFrame index
        for i, (idx, row) in enumerate(df.iterrows()):
            x = row[x_col] if pd.notna(row[x_col]) else 0
            y = row[y_col] if pd.notna(row[y_col]) else 0
            pattern = row.get('Pattern', 'Normal')
            color = PATTERN_COLORS.get(pattern, '#6b7280')
            title = str(row.get('Title', ''))[:40]
            
            self._points.append({
                'x': float(x), 
                'y': float(y),
                'pattern': pattern,
                'color': color,
                'index': idx,  # Use DataFrame index for lookup back to original data
                'tooltip': f"{title}...\nPattern: {pattern}\n{x_col}: {x:.1f}\n{y_col}: {y:.1f}"
            })
        
        self._rebuild_scatter()
        
        self.setLabel('bottom', x_col)
        self.setLabel('left', y_col)
        self.setTitle(f'{x_col} vs {y_col} (click to select)')
    
    def _rebuild_scatter(self):
        """Rebuild scatter with current selection state."""
        if self._scatter_item is not None:
            self.removeItem(self._scatter_item)
        
        if not self._points:
            return
        
        spots = []
        for pt in self._points:
            if pt['index'] in self._selected:
                # Selected: black fill, colored border, larger
                spots.append({
                    'pos': (pt['x'], pt['y']),
                    'brush': pg.mkBrush('#000000'),
                    'pen': pg.mkPen(pt['color'], width=3),
                    'size': 14,
                })
            else:
                # Normal: colored fill, white border
                spots.append({
                    'pos': (pt['x'], pt['y']),
                    'brush': pg.mkBrush(pt['color']),
                    'pen': pg.mkPen('w', width=1),
                    'size': 10,
                })
        
        self._scatter_item = pg.ScatterPlotItem(spots=spots)
        self.addItem(self._scatter_item)
    
    def _on_hover(self, pos):
        """Show tooltip."""
        if not self._points:
            return
        
        mouse = self.plotItem.vb.mapSceneToView(pos)
        
        closest = None
        min_dist = float('inf')
        
        for pt in self._points:
            d = ((pt['x'] - mouse.x())**2 + (pt['y'] - mouse.y())**2)**0.5
            if d < min_dist:
                min_dist = d
                closest = pt
        
        vr = self.viewRange()
        threshold = max(vr[0][1] - vr[0][0], vr[1][1] - vr[1][0]) * 0.03
        
        if closest and min_dist < threshold:
            QToolTip.showText(self.mapToGlobal(self.mapFromScene(pos)), closest['tooltip'])
        else:
            QToolTip.hideText()
    
    def _on_click(self, event):
        """Toggle selection."""
        if not self._points:
            return
        
        mouse = self.plotItem.vb.mapSceneToView(event.scenePos())
        
        closest = None
        min_dist = float('inf')
        
        for pt in self._points:
            d = ((pt['x'] - mouse.x())**2 + (pt['y'] - mouse.y())**2)**0.5
            if d < min_dist:
                min_dist = d
                closest = pt
        
        vr = self.viewRange()
        threshold = max(vr[0][1] - vr[0][0], vr[1][1] - vr[1][0]) * 0.03
        
        if closest and min_dist < threshold:
            idx = closest['index']
            if idx in self._selected:
                self._selected.remove(idx)
            else:
                self._selected.add(idx)
            
            self._rebuild_scatter()
            self.selectionChanged.emit(list(self._selected))
    
    def clear_selection(self):
        self._selected = set()
        self._rebuild_scatter()
        self.selectionChanged.emit([])
    
    def get_selected(self) -> List[int]:
        return list(self._selected)


# =============================================================================
# BY YEAR STACKED BAR CHART
# =============================================================================

class ByYearPlot(pg.PlotWidget):
    """Stacked bar chart of patterns by year with selection."""
    
    selectionChanged = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=False, y=False)
        
        self._year_data = {}  # {year: {pattern: [indices]}}
        self._bar_items = []  # List of (bar_item, year, pattern, bottom, top)
        self._selected_year = None
        self._selected_pattern = None
        
        self.scene().sigMouseMoved.connect(self._on_hover)
        self.scene().sigMouseClicked.connect(self._on_click)
    
    def set_data(self, df: pd.DataFrame):
        """Plot patterns by year."""
        self.clear()
        self._year_data = {}
        self._bar_items = []
        self._selected_year = None
        self._selected_pattern = None
        
        if df is None or df.empty or 'Year' not in df.columns or 'Pattern' not in df.columns:
            return
        
        # Build year_data: {year: {pattern: [indices]}}
        for idx, row in df.iterrows():
            year = row['Year']
            pattern = row['Pattern']
            if pd.isna(year):
                continue
            year = int(year)
            
            if year not in self._year_data:
                self._year_data[year] = {}
            if pattern not in self._year_data[year]:
                self._year_data[year][pattern] = []
            self._year_data[year][pattern].append(idx)
        
        if not self._year_data:
            return
        
        years = sorted(self._year_data.keys())
        
        # Stack bars
        bottom = {y: 0 for y in years}
        
        for pattern in PATTERN_ORDER:
            color = PATTERN_COLORS.get(pattern, '#6b7280')
            
            for year in years:
                if pattern not in self._year_data.get(year, {}):
                    continue
                
                count = len(self._year_data[year][pattern])
                if count == 0:
                    continue
                
                bar = pg.BarGraphItem(
                    x=[year], height=[count], width=0.7,
                    y0=[bottom[year]],
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen('w', width=0.5),
                )
                self.addItem(bar)
                
                self._bar_items.append({
                    'item': bar,
                    'year': year,
                    'pattern': pattern,
                    'bottom': bottom[year],
                    'top': bottom[year] + count,
                    'color': color,
                    'indices': self._year_data[year][pattern]
                })
                
                bottom[year] += count
        
        self.setLabel('bottom', 'Publication Year')
        self.setLabel('left', 'Count')
        self.setTitle('Citation Patterns by Year (click to select)')
        
        # Add legend
        for pattern in PATTERN_ORDER:
            if any(b['pattern'] == pattern for b in self._bar_items):
                color = PATTERN_COLORS.get(pattern, '#6b7280')
                self.plot([0], [0], pen=None, symbol='s', symbolBrush=color,
                         symbolPen='w', symbolSize=10, name=pattern)
        self.addLegend()
    
    def _on_hover(self, pos):
        """Show tooltip on hover."""
        mouse = self.plotItem.vb.mapSceneToView(pos)
        
        for b in self._bar_items:
            if (b['year'] - 0.4 <= mouse.x() <= b['year'] + 0.4 and
                b['bottom'] <= mouse.y() <= b['top']):
                count = len(b['indices'])
                tip = f"Year: {b['year']}\nPattern: {b['pattern']}\nCount: {count}\nClick to select"
                QToolTip.showText(self.mapToGlobal(self.mapFromScene(pos)), tip)
                return
        
        QToolTip.hideText()
    
    def _on_click(self, event):
        """Select papers in clicked segment."""
        mouse = self.plotItem.vb.mapSceneToView(event.scenePos())
        
        for b in self._bar_items:
            if (b['year'] - 0.4 <= mouse.x() <= b['year'] + 0.4 and
                b['bottom'] <= mouse.y() <= b['top']):
                
                # Toggle selection
                if self._selected_year == b['year'] and self._selected_pattern == b['pattern']:
                    self._selected_year = None
                    self._selected_pattern = None
                    self._update_visual()
                    self.selectionChanged.emit([])
                else:
                    self._selected_year = b['year']
                    self._selected_pattern = b['pattern']
                    self._update_visual()
                    self.selectionChanged.emit(b['indices'])
                return
    
    def _update_visual(self):
        """Highlight selected segment."""
        for b in self._bar_items:
            if b['year'] == self._selected_year and b['pattern'] == self._selected_pattern:
                b['item'].setOpts(pen=pg.mkPen('#000000', width=3))
            else:
                b['item'].setOpts(pen=pg.mkPen('w', width=0.5))
    
    def clear_selection(self):
        self._selected_year = None
        self._selected_pattern = None
        self._update_visual()


# =============================================================================
# METRIC HISTOGRAM
# =============================================================================

class MetricHistogram(pg.PlotWidget):
    """Histogram for metrics with selection."""
    
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
        self._base_color = '#3b82f6'
        
        self.scene().sigMouseMoved.connect(self._on_hover)
        self.scene().sigMouseClicked.connect(self._on_click)
    
    def set_data(self, df: pd.DataFrame, metric: str, pattern_filter: str = "All"):
        """Plot histogram."""
        self.clear()
        self._bins = []
        self._bar_items = []
        self._selected_bin = None
        
        if df is None or df.empty or metric not in df.columns:
            return
        
        if pattern_filter != "All":
            df = df[df['Pattern'] == pattern_filter].copy()
            self._base_color = PATTERN_COLORS.get(pattern_filter, '#3b82f6')
        else:
            self._base_color = '#3b82f6'
        
        if df.empty:
            return
        
        # Keep track of which rows have valid metric values
        valid_mask = df[metric].notna()
        valid_df = df[valid_mask]
        values = valid_df[metric]
        
        if values.empty:
            return
        
        counts, edges = np.histogram(values, bins=20)
        
        # Build bins with indices - use valid_df for consistent indexing
        for i in range(len(counts)):
            if i == len(counts) - 1:  # Include right edge for last bin
                mask = (values >= edges[i]) & (values <= edges[i+1])
            else:
                mask = (values >= edges[i]) & (values < edges[i+1])
            
            # Get indices from valid_df where mask is True
            bin_indices = valid_df[mask].index.tolist()
            
            self._bins.append({
                'left': edges[i],
                'right': edges[i+1],
                'center': (edges[i] + edges[i+1]) / 2,
                'count': counts[i],
                'indices': bin_indices
            })
        
        # Create individual bars for each bin (so we can highlight them)
        width = edges[1] - edges[0]
        for i, b in enumerate(self._bins):
            bar = pg.BarGraphItem(
                x=[b['center']], height=[b['count']], width=width*0.9,
                brush=pg.mkBrush(self._base_color),
                pen=pg.mkPen('w', width=1)
            )
            self.addItem(bar)
            self._bar_items.append(bar)
        
        # Median line
        if len(values) > 0:
            med = values.median()
            line = pg.InfiniteLine(pos=med, angle=90, pen=pg.mkPen('#ef4444', width=2, style=Qt.DashLine))
            self.addItem(line)
            
            txt = pg.TextItem(f"Median: {med:.1f}", color='#ef4444', anchor=(0, 1))
            txt.setPos(med, max(counts) * 0.9 if max(counts) > 0 else 1)
            self.addItem(txt)
        
        self.setLabel('bottom', metric)
        self.setLabel('left', 'Count')
        title = f'{metric} Distribution (click bin to select)'
        if pattern_filter != "All":
            title = f'{metric} - {pattern_filter} (click to select)'
        self.setTitle(title)
    
    def _on_hover(self, pos):
        mouse = self.plotItem.vb.mapSceneToView(pos)
        for b in self._bins:
            if b['left'] <= mouse.x() <= b['right'] and 0 <= mouse.y() <= b['count']:
                tip = f"Range: {b['left']:.1f} - {b['right']:.1f}\nCount: {b['count']}\nClick to select {len(b['indices'])} papers"
                QToolTip.showText(self.mapToGlobal(self.mapFromScene(pos)), tip)
                return
        QToolTip.hideText()
    
    def _on_click(self, event):
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
                bar.setOpts(
                    brush=pg.mkBrush('#000000'),
                    pen=pg.mkPen(self._base_color, width=3)
                )
            else:
                bar.setOpts(
                    brush=pg.mkBrush(self._base_color),
                    pen=pg.mkPen('w', width=1)
                )
    
    def clear_selection(self):
        self._selected_bin = None
        self._update_visual()


# =============================================================================
# TRAJECTORY PLOT
# =============================================================================

class TrajectoryPlot(pg.PlotWidget):
    """Multi-trajectory comparison."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=False, y=False)
    
    def set_data(self, trajectories: List[Dict], normalize: bool = True):
        """Plot trajectories."""
        self.clear()
        
        if not trajectories:
            return
        
        for traj in trajectories[:8]:
            years = traj.get('years', [])
            cites = traj.get('citations', [])
            pattern = traj.get('pattern', 'Normal')
            title = traj.get('title', '')[:20]
            
            if not years or not cites:
                continue
            
            y = cites
            if normalize and max(cites) > 0:
                y = [c / max(cites) for c in cites]
            
            color = PATTERN_COLORS.get(pattern, '#6b7280')
            
            line = pg.PlotDataItem(
                x=years, y=y,
                pen=pg.mkPen(color, width=2),
                symbol='o', symbolSize=5,
                symbolBrush=color, symbolPen='w',
                name=f"{title}... ({pattern})"
            )
            self.addItem(line)
        
        self.setLabel('bottom', 'Year')
        self.setLabel('left', 'Normalized Citations' if normalize else 'Citations')
        self.setTitle('Citation Trajectories')
        self.addLegend()


# =============================================================================
# TABLE ITEM
# =============================================================================

class NumericTableItem(QTableWidgetItem):
    def __init__(self, text: str, value: float):
        super().__init__(text)
        self._value = value
    
    def __lt__(self, other):
        if isinstance(other, NumericTableItem):
            return self._value < other._value
        return super().__lt__(other)


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWCitationPatterns(OWWidget):
    """Classify papers by citation trajectory."""
    
    name = "Citation Patterns"
    description = "Classify papers by citation trajectory"
    icon = "icons/citation_patterns.svg"
    priority = 71
    keywords = ["citation", "pattern", "trajectory", "evergreen", "sleeping beauty"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table)
    
    class Outputs:
        classified = Output("Classified", Table)
        selected = Output("Selected Documents", Table)
    
    # Settings
    use_openalex = settings.Setting(True)
    max_papers = settings.Setting(500)
    min_age = settings.Setting(3)
    
    plot_type_idx = settings.Setting(0)
    pattern_filter_idx = settings.Setting(0)
    metric_idx = settings.Setting(0)
    normalize_traj = settings.Setting(True)
    
    auto_apply = settings.Setting(False)
    
    want_main_area = True
    resizing_enabled = True
    
    PLOT_TYPES = ["Distribution", "By Year", "Trajectories", "Metrics", "Scatter"]
    PATTERN_FILTERS = ["All", "Evergreen", "Flash-in-the-pan", "Delayed Recognition", 
                       "Sleeping Beauty", "Normal"]
    METRICS = ["Half-life", "Years to Peak", "Early Citations %", "Decay Rate"]
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium library required")
        no_year = Msg("Year column not found")
        error = Msg("{}")
    
    class Warning(OWWidget.Warning):
        estimated = Msg("Results estimated from total citations")
    
    class Information(OWWidget.Information):
        done = Msg("Analyzed {} papers")
    
    def __init__(self):
        super().__init__()
        
        self._data = None
        self._df = None
        self._result = None
        self._result_df = None
        self._selected_indices = []
        
        self._setup_gui()
    
    def _setup_gui(self):
        """Build GUI."""
        # === Control Area ===
        
        # About
        about_box = gui.widgetBox(self.controlArea, "ℹ️ About")
        about_text = "Classify papers by citation trajectory:<br>"
        for p in ["Evergreen", "Flash-in-the-pan", "Delayed Recognition", "Sleeping Beauty", "Normal"]:
            c = PATTERN_COLORS.get(p, '#666')
            about_text += f"<span style='color:{c}'>●</span> {p}<br>"
        about_label = QLabel(f"<small>{about_text}</small>")
        about_label.setWordWrap(True)
        about_box.layout().addWidget(about_label)
        
        # Warning
        warn = QFrame()
        warn.setStyleSheet("background-color:#fef3c7; padding:6px; border-radius:4px;")
        wl = QVBoxLayout(warn)
        wl.setContentsMargins(6,6,6,6)
        wl.addWidget(QLabel("<small>⚠️ <b>Best with OpenAlex API</b></small>"))
        about_box.layout().addWidget(warn)
        
        # Settings
        settings_box = gui.widgetBox(self.controlArea, "⚙️ Settings")
        gui.checkBox(settings_box, self, "use_openalex", "Use OpenAlex API (recommended)")
        
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Max papers:"))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(10, 5000)
        self.max_spin.setValue(self.max_papers)
        self.max_spin.valueChanged.connect(lambda v: setattr(self, 'max_papers', v))
        h1.addWidget(self.max_spin)
        settings_box.layout().addLayout(h1)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Min age (years):"))
        self.age_spin = QSpinBox()
        self.age_spin.setRange(1, 20)
        self.age_spin.setValue(self.min_age)
        self.age_spin.valueChanged.connect(lambda v: setattr(self, 'min_age', v))
        h2.addWidget(self.age_spin)
        settings_box.layout().addLayout(h2)
        
        # Visualization
        viz_box = gui.widgetBox(self.controlArea, "📊 Visualization")
        
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Plot Type:"))
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(self.PLOT_TYPES)
        self.plot_combo.currentIndexChanged.connect(self._on_plot_changed)
        h3.addWidget(self.plot_combo)
        viz_box.layout().addLayout(h3)
        
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Pattern:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(self.PATTERN_FILTERS)
        self.pattern_combo.currentIndexChanged.connect(self._on_filter_changed)
        h4.addWidget(self.pattern_combo)
        viz_box.layout().addLayout(h4)
        
        h5 = QHBoxLayout()
        h5.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(self.METRICS)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        h5.addWidget(self.metric_combo)
        viz_box.layout().addLayout(h5)
        
        gui.checkBox(viz_box, self, "normalize_traj", "Normalize trajectories",
                    callback=self._update_trajectory_plot)
        
        # Selection
        sel_box = gui.widgetBox(self.controlArea, "✓ Selection")
        self.sel_label = QLabel("Selected: 0 papers")
        sel_box.layout().addWidget(self.sel_label)
        
        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self._clear_selection)
        sel_box.layout().addWidget(clear_btn)
        
        # Run
        self.run_btn = gui.button(self.controlArea, self, "▶ Run Analysis",
                                  callback=self.commit, autoDefault=False)
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color:#3b82f6; color:white; font-weight:bold;")
        
        gui.checkBox(self.controlArea, self, "auto_apply", "Auto apply")
        self.controlArea.layout().addStretch(1)
        
        # === Main Area ===
        main_layout = QVBoxLayout()
        self.mainArea.layout().addLayout(main_layout)
        
        self.status_label = QLabel("Load data and click Run Analysis")
        self.status_label.setStyleSheet("color:#6c757d; padding:4px;")
        main_layout.addWidget(self.status_label)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)
        
        if HAS_PYQTGRAPH:
            self.dist_plot = PatternDistributionPlot()
            self.dist_plot.selectionChanged.connect(self._on_selection)
            self.tabs.addTab(self.dist_plot, "📊 Distribution")
            
            self.year_plot = ByYearPlot()
            self.year_plot.selectionChanged.connect(self._on_selection)
            self.tabs.addTab(self.year_plot, "📅 By Year")
            
            self.traj_plot = TrajectoryPlot()
            self.tabs.addTab(self.traj_plot, "📈 Trajectories")
            
            self.metric_plot = MetricHistogram()
            self.metric_plot.selectionChanged.connect(self._on_selection)
            self.tabs.addTab(self.metric_plot, "📉 Metrics")
            
            self.scatter_plot = InteractiveScatterPlot()
            self.scatter_plot.selectionChanged.connect(self._on_selection)
            self.tabs.addTab(self.scatter_plot, "🔵 Scatter")
        else:
            self.tabs.addTab(QLabel("PyQtGraph required"), "Plot")
        
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.tabs.addTab(self.table, "📋 Data")
        
        # Export
        exp_layout = QHBoxLayout()
        exp_layout.addStretch()
        self.export_btn = QPushButton("📥 Export")
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        exp_layout.addWidget(self.export_btn)
        main_layout.addLayout(exp_layout)
    
    def _on_plot_changed(self, idx):
        self.plot_type_idx = idx
        self.tabs.setCurrentIndex(idx)
    
    def _on_filter_changed(self, idx):
        self.pattern_filter_idx = idx
        # Clear selection since indices may no longer be valid
        self._clear_selection()
        self._update_metric_plot()
        self._update_scatter_plot()
    
    def _on_metric_changed(self, idx):
        self.metric_idx = idx
        self._update_metric_plot()
    
    def _on_selection(self, indices):
        """Handle selection from plots - sync across plots."""
        self._selected_indices = indices
        self.sel_label.setText(f"Selected: {len(indices)} papers")
        
        # Identify which plot triggered the selection and clear others' visual state
        sender = self.sender()
        if HAS_PYQTGRAPH:
            if sender != self.dist_plot:
                self.dist_plot._selected_pattern = None
                self.dist_plot._update_visual()
            if sender != self.year_plot and self.year_plot._bar_items:
                self.year_plot._selected_year = None
                self.year_plot._selected_pattern = None
                self.year_plot._update_visual()
            if sender != self.scatter_plot and self.scatter_plot._points:
                self.scatter_plot._selected = set()
                self.scatter_plot._rebuild_scatter()
            if sender != self.metric_plot and self.metric_plot._bar_items:
                self.metric_plot._selected_bin = None
                self.metric_plot._update_visual()
        
        self._send_selected()
    
    def _on_table_selection(self):
        """Handle table selection."""
        rows = set(item.row() for item in self.table.selectedItems())
        self._selected_indices = list(rows)
        self.sel_label.setText(f"Selected: {len(rows)} papers")
        self._send_selected()
    
    def _clear_selection(self):
        self._selected_indices = []
        self.sel_label.setText("Selected: 0 papers")
        if HAS_PYQTGRAPH:
            self.dist_plot.clear_selection()
            self.year_plot.clear_selection()
            self.scatter_plot.clear_selection()
            self.metric_plot.clear_selection()
        self.table.clearSelection()
        self._send_selected()
    
    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._result = None
        self._result_df = None
        self._selected_indices = []
        
        self._clear_display()
        
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            return
        
        if data is None:
            self.Error.no_data()
            return
        
        # Convert to DataFrame
        d = {}
        for v in data.domain.attributes:
            d[v.name] = data.get_column(v)
        for v in data.domain.class_vars:
            d[v.name] = data.get_column(v)
        for v in data.domain.metas:
            d[v.name] = data.get_column(v)
        self._df = pd.DataFrame(d)
        
        self.status_label.setText(f"Loaded {len(self._df)} papers")
        
        if self.auto_apply:
            self.commit()
    
    def _clear_display(self):
        self.table.clear()
        self.table.setRowCount(0)
        self.export_btn.setEnabled(False)
        if HAS_PYQTGRAPH:
            self.dist_plot.clear()
            self.year_plot.clear()
            self.traj_plot.clear()
            self.metric_plot.clear()
            self.scatter_plot.clear()
    
    def commit(self):
        """Run analysis."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            return
        
        if self._df is None:
            self.Error.no_data()
            return
        
        try:
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.status_label.setText("Analyzing...")
            
            self._result = analyze_citation_patterns(
                self._df,
                use_openalex=self.use_openalex,
                max_papers=self.max_papers,
                min_age=self.min_age,
                verbose=False
            )
            
            self.progress.setVisible(False)
            
            if not self._result or self._result.n_analyzed == 0:
                self._clear_display()
                return
            
            self._result_df = self._result.to_dataframe()
            self._result_df = self._result_df.reset_index(drop=True)  # Ensure sequential indices
            
            self._update_table()
            self._update_all_plots()
            self._send_classified()
            
            if self._result.data_source == "estimated":
                self.Warning.estimated()
            
            self.Information.done(self._result.n_analyzed)
            self.status_label.setText(f"Analyzed {self._result.n_analyzed} papers")
            
        except ValueError as e:
            if "Year" in str(e):
                self.Error.no_year()
            else:
                self.Error.error(str(e))
        except Exception as e:
            self.Error.error(str(e))
        finally:
            self.progress.setVisible(False)
    
    def _update_table(self):
        """Update results table."""
        if self._result_df is None:
            return
        
        df = self._result_df
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())
        
        for i, (idx, row) in enumerate(df.iterrows()):
            pattern = row.get("Pattern", "Normal")
            color = PATTERN_COLORS.get(pattern, '#6b7280')
            
            for j, col in enumerate(df.columns):
                val = row[col]
                
                if pd.isna(val):
                    item = QTableWidgetItem("")
                elif isinstance(val, float):
                    item = NumericTableItem(f"{val:.2f}", val)
                elif isinstance(val, (int, np.integer)):
                    item = NumericTableItem(str(val), float(val))
                else:
                    item = QTableWidgetItem(str(val)[:60])
                
                if col == "Pattern":
                    item.setBackground(QColor(color))
                    item.setForeground(QColor("white"))
                
                self.table.setItem(i, j, item)
        
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)
        self.export_btn.setEnabled(True)
    
    def _update_all_plots(self):
        if not HAS_PYQTGRAPH or self._result_df is None:
            return
        
        self._update_distribution_plot()
        self._update_by_year_plot()
        self._update_trajectory_plot()
        self._update_metric_plot()
        self._update_scatter_plot()
    
    def _update_distribution_plot(self):
        if self._result is None:
            return
        
        pattern_indices = {}
        for i, (idx, row) in enumerate(self._result_df.iterrows()):
            p = row.get("Pattern", "Normal")
            if p not in pattern_indices:
                pattern_indices[p] = []
            pattern_indices[p].append(i)
        
        self.dist_plot.set_data(self._result.pattern_counts, pattern_indices)
    
    def _update_by_year_plot(self):
        if self._result_df is not None:
            self.year_plot.set_data(self._result_df)
    
    def _update_trajectory_plot(self):
        if self._result is None:
            return
        
        trajs = []
        for t in self._result.trajectories[:8]:
            if not t.counts_by_year:
                continue
            years = sorted(t.counts_by_year.keys())
            cites = [t.counts_by_year[y] for y in years]
            trajs.append({
                'years': years,
                'citations': cites,
                'pattern': t.pattern.value,
                'title': t.title
            })
        
        self.traj_plot.set_data(trajs, normalize=self.normalize_traj)
    
    def _update_metric_plot(self):
        if self._result_df is None:
            return
        
        metric = self.METRICS[self.metric_idx]
        pattern = self.PATTERN_FILTERS[self.pattern_filter_idx]
        self.metric_plot.set_data(self._result_df, metric, pattern)
    
    def _update_scatter_plot(self):
        if self._result_df is None:
            return
        
        df = self._result_df
        pattern = self.PATTERN_FILTERS[self.pattern_filter_idx]
        
        if pattern != "All":
            df = df[df["Pattern"] == pattern].copy()
        
        self.scatter_plot.set_data(df, "Years to Peak", "Half-life")
    
    def _send_classified(self):
        """Send classified output."""
        if self._result_df is None:
            self.Outputs.classified.send(None)
            return
        
        df = self._result_df
        
        attrs = [
            ContinuousVariable("Year"),
            ContinuousVariable("Total Citations"),
            ContinuousVariable("Peak Year"),
            ContinuousVariable("Peak Citations"),
            ContinuousVariable("Years to Peak"),
            ContinuousVariable("Half-life"),
            ContinuousVariable("Decay Rate"),
            ContinuousVariable("Early Citations %"),
            ContinuousVariable("Late Citations %"),
            ContinuousVariable("Confidence"),
        ]
        
        class_var = DiscreteVariable("Pattern", values=PATTERN_ORDER)
        metas = [StringVariable("DOI"), StringVariable("Title")]
        domain = Domain(attrs, class_vars=[class_var], metas=metas)
        
        n = len(df)
        X = np.zeros((n, len(attrs)))
        Y = np.zeros((n,))
        M = np.empty((n, len(metas)), dtype=object)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            X[i, 0] = row.get("Year", np.nan)
            X[i, 1] = row.get("Total Citations", 0)
            X[i, 2] = row.get("Peak Year", np.nan)
            X[i, 3] = row.get("Peak Citations", 0)
            X[i, 4] = row.get("Years to Peak", 0)
            X[i, 5] = row.get("Half-life", 0)
            X[i, 6] = row.get("Decay Rate", 0)
            X[i, 7] = row.get("Early Citations %", 0)
            X[i, 8] = row.get("Late Citations %", 0)
            X[i, 9] = row.get("Confidence", 0)
            
            p = row.get("Pattern", "Normal")
            Y[i] = PATTERN_ORDER.index(p) if p in PATTERN_ORDER else 4
            
            M[i, 0] = str(row.get("DOI", ""))
            M[i, 1] = str(row.get("Title", ""))
        
        table = Table.from_numpy(domain, X, Y, metas=M)
        self.Outputs.classified.send(table)
    
    def _send_selected(self):
        """Send selected documents."""
        if self._result_df is None or not self._selected_indices:
            self.Outputs.selected.send(None)
            return
        
        # Use .loc for label-based indexing (indices come from DataFrame index labels)
        try:
            df = self._result_df.loc[self._selected_indices]
        except KeyError:
            # Fall back to iloc if indices are positional
            df = self._result_df.iloc[self._selected_indices]
        
        if df.empty:
            self.Outputs.selected.send(None)
            return
        
        attrs = [
            ContinuousVariable("Year"),
            ContinuousVariable("Total Citations"),
            ContinuousVariable("Years to Peak"),
            ContinuousVariable("Half-life"),
        ]
        metas = [StringVariable("DOI"), StringVariable("Title"), StringVariable("Pattern")]
        domain = Domain(attrs, metas=metas)
        
        X = np.zeros((len(df), len(attrs)))
        M = np.empty((len(df), len(metas)), dtype=object)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            X[i, 0] = row.get("Year", np.nan)
            X[i, 1] = row.get("Total Citations", 0)
            X[i, 2] = row.get("Years to Peak", 0)
            X[i, 3] = row.get("Half-life", 0)
            
            M[i, 0] = str(row.get("DOI", ""))
            M[i, 1] = str(row.get("Title", ""))
            M[i, 2] = str(row.get("Pattern", ""))
        
        sel_table = Table.from_numpy(domain, X, metas=M)
        self.Outputs.selected.send(sel_table)
    
    def _export(self):
        if self._result_df is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export", "citation_patterns.xlsx",
            "Excel (*.xlsx);;CSV (*.csv)"
        )
        
        if not path:
            return
        
        try:
            if path.endswith('.csv'):
                self._result_df.to_csv(path, index=False)
            else:
                self._result_df.to_excel(path, index=False)
            QMessageBox.information(self, "Done", f"Exported to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


if __name__ == "__main__":
    WidgetPreview(OWCitationPatterns).run()
