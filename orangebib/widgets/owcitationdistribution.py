# -*- coding: utf-8 -*-
"""
Citation Distribution Widget
============================
Orange widget for analyzing citation distribution and impact metrics.

Features:
- Interactive citation histogram with selection
- Log-scale distribution view with selection
- Citation classes breakdown with selection
- User-selectable bar color
- Output: Selected documents or aggregated bin data
"""

import logging
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QTabWidget, QFrame,
    QToolTip, QApplication,
)
from AnyQt.QtCore import Qt, pyqtSignal, QRectF
from AnyQt.QtGui import QFont, QColor, QCursor

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

# Import pyqtgraph
try:
    import pyqtgraph as pg
    from Orange.widgets.visualize.utils.plotutils import AxisItem
    from Orange.widgets.utils.plot import SELECT, PANNING, ZOOMING
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    SELECT, PANNING, ZOOMING = 0, 1, 2

# Import scipy for stats
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# =============================================================================
# COLOR OPTIONS
# =============================================================================

BAR_COLORS = {
    "Blue": "#4a90d9",
    "Teal": "#17a2b8",
    "Green": "#28a745",
    "Orange": "#fd7e14",
    "Red": "#dc3545",
    "Purple": "#6f42c1",
    "Gray": "#6c757d",
}

SELECTED_COLOR = "#ff9800"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def h_index(citations):
    """Compute H-index."""
    if len(citations) == 0:
        return 0
    sorted_cit = sorted(citations, reverse=True)
    h = 0
    for i, c in enumerate(sorted_cit, 1):
        if c >= i:
            h = i
        else:
            break
    return h


def g_index(citations):
    """Compute G-index."""
    if len(citations) == 0:
        return 0
    sorted_cit = sorted(citations, reverse=True)
    cumsum = 0
    g = 0
    for i, c in enumerate(sorted_cit, 1):
        cumsum += c
        if cumsum >= i * i:
            g = i
    return g


def gini_coefficient(citations):
    """Compute Gini coefficient for citation inequality."""
    if len(citations) == 0 or sum(citations) == 0:
        return 0
    n = len(citations)
    sorted_cit = sorted(citations)
    cumsum = np.cumsum(sorted_cit)
    return 1 - 2 * np.sum(cumsum) / (n * sum(citations))


def classify_citation(c, p90, p75, p50):
    """Classify a citation count into a class."""
    if c == 0:
        return "Uncited"
    elif c >= p90:
        return "Highly Cited (Top 10%)"
    elif c >= p75:
        return "Well Cited (Top 25%)"
    elif c >= p50:
        return "Average"
    else:
        return "Low Cited"


# =============================================================================
# CUSTOM VIEWBOX WITH SELECTION
# =============================================================================

class SelectableViewBox(pg.ViewBox):
    """Custom ViewBox with selection support."""
    
    def __init__(self, graph, enable_menu=False):
        super().__init__(enableMenu=enable_menu)
        self.graph = graph
        self.setMouseMode(self.RectMode)
    
    def mouseDragEvent(self, ev, axis=None):
        if self.graph.state == SELECT and axis is None:
            ev.accept()
            if ev.button() == Qt.LeftButton:
                self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                if ev.isFinish():
                    self.rbScaleBox.hide()
                    p1 = self.mapToView(ev.buttonDownPos(ev.button()))
                    p2 = self.mapToView(ev.pos())
                    self.graph.select_by_rectangle(QRectF(p1, p2))
                else:
                    self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        elif self.graph.state == ZOOMING or self.graph.state == PANNING:
            super().mouseDragEvent(ev, axis=axis)
        else:
            ev.ignore()
    
    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.graph.select_by_click(self.mapSceneToView(ev.scenePos()))
            ev.accept()


# =============================================================================
# BASE HISTOGRAM PLOT
# =============================================================================

class BaseHistogramPlot(pg.PlotWidget):
    """Base interactive histogram with selection support."""
    
    BAR_WIDTH_RATIO = 0.85
    
    selectionChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        self.selection: List[int] = []
        self.state: int = SELECT
        
        super().__init__(
            parent=parent,
            viewBox=SelectableViewBox(self),
            enableMenu=False,
        )
        
        self.bar_item: Optional[pg.BarGraphItem] = None
        self.mean_line: Optional[pg.InfiniteLine] = None
        self.median_line: Optional[pg.InfiniteLine] = None
        
        # Data storage
        self._bins = None
        self._counts = None
        self._bar_width = 1.0
        self._mean_val = 0
        self._median_val = 0
        self._bar_color = "#4a90d9"
        
        # Setup
        self.setBackground('w')
        self.showGrid(x=False, y=False)
        self.setLabel('left', 'Number of Papers')
        self.setLabel('bottom', 'Citations')
        
        # Mouse tracking for tooltips
        self.setMouseTracking(True)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
    
    def set_bar_color(self, color: str):
        """Set bar color."""
        self._bar_color = color
        self._update_selection_display()
    
    def clear_plot(self):
        """Clear all plot items."""
        self.selection = []
        self.clear()
        self.bar_item = None
        self.mean_line = None
        self.median_line = None
        self._bins = None
        self._counts = None
    
    def toggle_mean(self, visible: bool):
        if self.mean_line:
            self.mean_line.setVisible(visible)
    
    def toggle_median(self, visible: bool):
        if self.median_line:
            self.median_line.setVisible(visible)
    
    def _on_mouse_moved(self, pos):
        """Handle mouse move for tooltips."""
        if self.bar_item is None or self._bins is None:
            QToolTip.hideText()
            return
        
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()
        
        for i, (bin_start, bin_end, count) in enumerate(zip(self._bins[:-1], self._bins[1:], self._counts)):
            if bin_start <= x <= bin_end and 0 <= y <= count:
                tooltip = f"Citations: {bin_start:.0f} – {bin_end:.0f}\nPapers: {count:,}"
                if i in self.selection:
                    tooltip += "\n(Selected)"
                global_pos = self.mapToGlobal(self.mapFromScene(pos))
                QToolTip.showText(global_pos, tooltip)
                return
        
        QToolTip.hideText()
    
    def select_by_rectangle(self, rect: QRectF):
        """Select bars within rectangle."""
        if self.bar_item is None or self._bins is None:
            return
        
        x0, x1 = sorted((rect.topLeft().x(), rect.bottomRight().x()))
        y0, y1 = sorted((rect.topLeft().y(), rect.bottomRight().y()))
        
        indices = []
        for i, (bin_start, bin_end, count) in enumerate(zip(self._bins[:-1], self._bins[1:], self._counts)):
            if x0 <= bin_end and x1 >= bin_start and y0 <= count and y1 >= 0:
                indices.append(i)
        
        self.select_by_indices(indices)
    
    def select_by_click(self, p):
        """Select bar at click position."""
        if self.bar_item is None or self._bins is None:
            return
        
        x = p.x()
        y = p.y()
        
        for i, (bin_start, bin_end, count) in enumerate(zip(self._bins[:-1], self._bins[1:], self._counts)):
            if bin_start <= x <= bin_end and 0 <= y <= count:
                self.select_by_indices([i])
                return
        
        self.select_by_indices([])
    
    def select_by_indices(self, indices: List[int]):
        """Update selection with modifier key support."""
        keys = QApplication.keyboardModifiers()
        if keys & Qt.ControlModifier:
            self.selection = list(set(self.selection) ^ set(indices))
        elif keys & Qt.AltModifier:
            self.selection = list(set(self.selection) - set(indices))
        elif keys & Qt.ShiftModifier:
            self.selection = list(set(self.selection) | set(indices))
        else:
            self.selection = list(set(indices))
        
        self._update_selection_display()
        self.selectionChanged.emit()
    
    def _update_selection_display(self):
        """Update bar colors based on selection."""
        if self.bar_item is None or self._counts is None:
            return
        
        n = len(self._counts)
        brushes = []
        pens = []
        
        for i in range(n):
            if i in self.selection:
                brushes.append(pg.mkBrush(SELECTED_COLOR))
                pens.append(pg.mkPen('#e65100', width=2))
            else:
                brushes.append(pg.mkBrush(self._bar_color))
                pens.append(pg.mkPen('w', width=1))
        
        self.bar_item.setOpts(brushes=brushes, pens=pens)
    
    def clear_selection(self):
        """Clear selection."""
        self.selection = []
        self._update_selection_display()
        self.selectionChanged.emit()
    
    def get_selected_ranges(self) -> List[tuple]:
        """Get list of (min, max) citation ranges for selected bars."""
        if not self.selection or self._bins is None:
            return []
        
        ranges = []
        for i in self.selection:
            if i < len(self._bins) - 1:
                ranges.append((self._bins[i], self._bins[i + 1]))
        return ranges
    
    def set_select_mode(self):
        self.state = SELECT
        self.getViewBox().setMouseMode(pg.ViewBox.RectMode)
    
    def set_pan_mode(self):
        self.state = PANNING
        self.getViewBox().setMouseMode(pg.ViewBox.PanMode)
    
    def set_zoom_mode(self):
        self.state = ZOOMING
        self.getViewBox().setMouseMode(pg.ViewBox.RectMode)


# =============================================================================
# LINEAR HISTOGRAM
# =============================================================================

class HistogramPlot(BaseHistogramPlot):
    """Linear scale histogram."""
    
    def set_data(self, citations: np.ndarray, n_bins: int = 30, clip_percentile: int = 99,
                 show_mean: bool = True, show_median: bool = True):
        """Set data and draw histogram."""
        self.clear_plot()
        
        if citations is None or len(citations) == 0:
            return
        
        # Clip for visualization
        clip_val = np.percentile(citations, clip_percentile)
        clipped = np.clip(citations, 0, clip_val)
        
        # Compute histogram
        counts, bins = np.histogram(clipped, bins=n_bins)
        self._bins = bins
        self._counts = counts
        self._bar_width = bins[1] - bins[0]
        self._mean_val = np.mean(citations)
        self._median_val = np.median(citations)
        
        # Create bars
        x = bins[:-1] + self._bar_width / 2
        brushes = [pg.mkBrush(self._bar_color) for _ in counts]
        
        self.bar_item = pg.BarGraphItem(
            x=x, height=counts, width=self._bar_width * self.BAR_WIDTH_RATIO,
            brushes=brushes, pen=pg.mkPen('w', width=1)
        )
        self.addItem(self.bar_item)
        
        # Mean line
        if show_mean and self._mean_val <= clip_val:
            self.mean_line = pg.InfiniteLine(
                pos=self._mean_val, angle=90,
                pen=pg.mkPen('#dc3545', width=2, style=Qt.DashLine),
                label=f'Mean ({self._mean_val:.1f})',
                labelOpts={'position': 0.9, 'color': '#dc3545', 'fill': pg.mkBrush(255, 255, 255, 200)}
            )
            self.addItem(self.mean_line)
        
        # Median line
        if show_median and self._median_val <= clip_val:
            self.median_line = pg.InfiniteLine(
                pos=self._median_val, angle=90,
                pen=pg.mkPen('#28a745', width=2),
                label=f'Median ({int(self._median_val)})',
                labelOpts={'position': 0.8, 'color': '#28a745', 'fill': pg.mkBrush(255, 255, 255, 200)}
            )
            self.addItem(self.median_line)
        
        self.setTitle(f"Citation Distribution (clipped at P{clip_percentile})")
        
        # Reset view
        self.setXRange(bins[0] - self._bar_width, bins[-1] + self._bar_width, padding=0.02)
        y_max = max(counts) if len(counts) > 0 else 1
        self.setYRange(0, y_max * 1.1, padding=0.02)


# =============================================================================
# LOG SCALE HISTOGRAM
# =============================================================================

class LogHistogramPlot(BaseHistogramPlot):
    """Log scale histogram (excludes uncited papers)."""
    
    def set_data(self, citations: np.ndarray, n_bins: int = 30,
                 show_mean: bool = True, show_median: bool = True):
        """Set data and draw log-scale histogram."""
        self.clear_plot()
        
        if citations is None or len(citations) == 0:
            return
        
        # Filter positive citations only
        positive = citations[citations > 0]
        if len(positive) == 0:
            self.setTitle("Log Distribution (no cited papers)")
            return
        
        # Log-scale bins
        max_val = max(positive)
        log_bins = np.logspace(0, np.log10(max_val + 1), n_bins)
        counts, bins = np.histogram(positive, bins=log_bins)
        
        self._bins = bins
        self._counts = counts
        self._mean_val = np.mean(positive)
        self._median_val = np.median(positive)
        
        # Create bars - use linear x positions for display
        x_positions = np.arange(len(counts))
        self._bar_width = 0.85
        
        brushes = [pg.mkBrush(self._bar_color) for _ in counts]
        
        self.bar_item = pg.BarGraphItem(
            x=x_positions, height=counts, width=self._bar_width,
            brushes=brushes, pen=pg.mkPen('w', width=1)
        )
        self.addItem(self.bar_item)
        
        # X-axis labels (log scale values)
        ax = self.getAxis('bottom')
        ticks = []
        for i in range(0, len(bins) - 1, max(1, len(bins) // 8)):
            ticks.append((i, f"{bins[i]:.0f}"))
        ax.setTicks([ticks])
        
        # Mean line - find position in log space
        if show_mean:
            mean_pos = np.searchsorted(bins[:-1], self._mean_val) - 0.5
            if 0 <= mean_pos < len(counts):
                self.mean_line = pg.InfiniteLine(
                    pos=mean_pos, angle=90,
                    pen=pg.mkPen('#dc3545', width=2, style=Qt.DashLine),
                    label=f'Mean ({self._mean_val:.1f})',
                    labelOpts={'position': 0.9, 'color': '#dc3545', 'fill': pg.mkBrush(255, 255, 255, 200)}
                )
                self.addItem(self.mean_line)
        
        # Median line
        if show_median:
            median_pos = np.searchsorted(bins[:-1], self._median_val) - 0.5
            if 0 <= median_pos < len(counts):
                self.median_line = pg.InfiniteLine(
                    pos=median_pos, angle=90,
                    pen=pg.mkPen('#28a745', width=2),
                    label=f'Median ({int(self._median_val)})',
                    labelOpts={'position': 0.8, 'color': '#28a745', 'fill': pg.mkBrush(255, 255, 255, 200)}
                )
                self.addItem(self.median_line)
        
        n_uncited = len(citations) - len(positive)
        self.setTitle(f"Log Distribution ({n_uncited} uncited excluded)")
        self.setLabel('bottom', 'Citations (log scale)')
        
        # Reset view
        self.setXRange(-0.5, len(counts) - 0.5, padding=0.02)
        y_max = max(counts) if len(counts) > 0 else 1
        self.setYRange(0, y_max * 1.1, padding=0.02)
    
    def _on_mouse_moved(self, pos):
        """Handle mouse move for tooltips (log scale version)."""
        if self.bar_item is None or self._bins is None:
            QToolTip.hideText()
            return
        
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()
        
        # Find bar by x position (integer index)
        idx = int(round(x))
        if 0 <= idx < len(self._counts):
            count = self._counts[idx]
            if 0 <= y <= count:
                bin_start = self._bins[idx]
                bin_end = self._bins[idx + 1]
                tooltip = f"Citations: {bin_start:.0f} – {bin_end:.0f}\nPapers: {count:,}"
                if idx in self.selection:
                    tooltip += "\n(Selected)"
                global_pos = self.mapToGlobal(self.mapFromScene(pos))
                QToolTip.showText(global_pos, tooltip)
                return
        
        QToolTip.hideText()
    
    def select_by_rectangle(self, rect: QRectF):
        """Select bars within rectangle (log scale version)."""
        if self.bar_item is None or self._bins is None:
            return
        
        x0, x1 = sorted((rect.topLeft().x(), rect.bottomRight().x()))
        y0, y1 = sorted((rect.topLeft().y(), rect.bottomRight().y()))
        
        indices = []
        for i, count in enumerate(self._counts):
            if x0 <= i + 0.5 and x1 >= i - 0.5 and y0 <= count and y1 >= 0:
                indices.append(i)
        
        self.select_by_indices(indices)
    
    def select_by_click(self, p):
        """Select bar at click position (log scale version)."""
        if self.bar_item is None or self._counts is None:
            return
        
        x = p.x()
        y = p.y()
        
        idx = int(round(x))
        if 0 <= idx < len(self._counts):
            if 0 <= y <= self._counts[idx]:
                self.select_by_indices([idx])
                return
        
        self.select_by_indices([])


# =============================================================================
# CITATION CLASSES CHART
# =============================================================================

class CitationClassesChart(BaseHistogramPlot):
    """Bar chart for citation classes with selection."""
    
    CLASS_ORDER = ["Uncited", "Low Cited", "Average", "Well Cited (Top 25%)", "Highly Cited (Top 10%)"]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._class_data = None
        self._class_indices = {}  # class_name -> list of document indices
        self.setLabel('bottom', 'Citation Class')
    
    def set_data(self, class_distribution: Dict[str, Dict], class_indices: Dict[str, List[int]] = None):
        """Set citation class data."""
        self.clear_plot()
        self._class_data = class_distribution
        self._class_indices = class_indices or {}
        
        if not class_distribution:
            return
        
        # Build arrays
        x_vals = []
        heights = []
        labels = []
        
        for i, cls in enumerate(self.CLASS_ORDER):
            if cls in class_distribution:
                x_vals.append(i)
                heights.append(class_distribution[cls]['count'])
                labels.append(cls)
        
        if not x_vals:
            return
        
        self._bins = np.array(x_vals + [x_vals[-1] + 1], dtype=float) - 0.5
        self._counts = np.array(heights)
        self._bar_width = 0.7
        
        # Create bars - all same color
        brushes = [pg.mkBrush(self._bar_color) for _ in heights]
        
        self.bar_item = pg.BarGraphItem(
            x=x_vals, height=heights, width=self._bar_width,
            brushes=brushes, pen=pg.mkPen('w', width=1)
        )
        self.addItem(self.bar_item)
        
        # X-axis labels
        ax = self.getAxis('bottom')
        short_labels = [cls.split('(')[0].strip() for cls in labels]
        ticks = [[(x, lbl) for x, lbl in zip(x_vals, short_labels)]]
        ax.setTicks(ticks)
        
        self.setTitle("Citation Classes")
        
        # Reset view
        self.setXRange(min(x_vals) - 0.5, max(x_vals) + 0.5, padding=0.05)
        y_max = max(heights) if heights else 1
        self.setYRange(0, y_max * 1.1, padding=0.02)
    
    def _on_mouse_moved(self, pos):
        """Handle mouse move for tooltips."""
        if self.bar_item is None or self._class_data is None:
            QToolTip.hideText()
            return
        
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()
        
        # Find bar by x position
        idx = int(round(x))
        
        # Get corresponding class
        classes_present = [cls for cls in self.CLASS_ORDER if cls in self._class_data]
        if 0 <= idx < len(classes_present):
            cls = classes_present[idx]
            d = self._class_data[cls]
            count = d['count']
            
            if 0 <= y <= count:
                tooltip = f"{cls}\nCount: {count:,}\nPercentage: {d['percentage']:.1f}%"
                if idx in self.selection:
                    tooltip += "\n(Selected)"
                global_pos = self.mapToGlobal(self.mapFromScene(pos))
                QToolTip.showText(global_pos, tooltip)
                return
        
        QToolTip.hideText()
    
    def select_by_rectangle(self, rect: QRectF):
        """Select bars within rectangle."""
        if self.bar_item is None or self._class_data is None:
            return
        
        x0, x1 = sorted((rect.topLeft().x(), rect.bottomRight().x()))
        y0, y1 = sorted((rect.topLeft().y(), rect.bottomRight().y()))
        
        classes_present = [cls for cls in self.CLASS_ORDER if cls in self._class_data]
        
        indices = []
        for i, cls in enumerate(classes_present):
            count = self._class_data[cls]['count']
            if x0 <= i + 0.5 and x1 >= i - 0.5 and y0 <= count and y1 >= 0:
                indices.append(i)
        
        self.select_by_indices(indices)
    
    def select_by_click(self, p):
        """Select bar at click position."""
        if self.bar_item is None or self._class_data is None:
            return
        
        x = p.x()
        y = p.y()
        
        classes_present = [cls for cls in self.CLASS_ORDER if cls in self._class_data]
        idx = int(round(x))
        
        if 0 <= idx < len(classes_present):
            cls = classes_present[idx]
            count = self._class_data[cls]['count']
            if 0 <= y <= count:
                self.select_by_indices([idx])
                return
        
        self.select_by_indices([])
    
    def get_selected_class_names(self) -> List[str]:
        """Get names of selected classes."""
        if not self.selection or self._class_data is None:
            return []
        
        classes_present = [cls for cls in self.CLASS_ORDER if cls in self._class_data]
        return [classes_present[i] for i in self.selection if i < len(classes_present)]
    
    def get_selected_document_indices(self) -> List[int]:
        """Get document indices for selected classes."""
        if not self._class_indices:
            return []
        
        selected_classes = self.get_selected_class_names()
        indices = []
        for cls in selected_classes:
            if cls in self._class_indices:
                indices.extend(self._class_indices[cls])
        return sorted(set(indices))


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWCitationDistribution(OWWidget):
    """Analyze citation distribution and impact metrics."""
    
    name = "Citation Distribution"
    description = "Analyze citation distribution and impact metrics"
    icon = "icons/citation_distribution.svg"
    priority = 25
    keywords = ["citation", "distribution", "h-index", "g-index", "gini", "histogram"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        selected_documents = Output("Selected Documents", Table, doc="Selected papers (document-level)")
        aggregated = Output("Aggregated", Table, doc="Aggregated bin statistics")
        metrics = Output("Metrics", Table, doc="Citation metrics")
        classes = Output("Citation Classes", Table, doc="Papers with citation class")
    
    # Settings
    citations_column = settings.Setting("")
    clip_percentile = settings.Setting(99)
    n_bins = settings.Setting(30)
    show_mean = settings.Setting(True)
    show_median = settings.Setting(True)
    bar_color_index = settings.Setting(0)
    auto_apply = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_citations = Msg("No citation column found")
        compute_error = Msg("Computation error: {}")
    
    class Warning(OWWidget.Warning):
        no_pyqtgraph = Msg("PyQtGraph not available")
    
    class Information(OWWidget.Information):
        analyzed = Msg("Analyzed {:,} papers")
        selected = Msg("Selected {:,} papers")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._result: Optional[Dict] = None
        self._available_cite_cols: List[str] = []
        
        if not HAS_PYQTGRAPH:
            self.Warning.no_pyqtgraph()
        
        self._setup_control_area()
        self._setup_main_area()
    
    def _setup_control_area(self):
        """Build control area."""
        # Settings
        settings_box = gui.widgetBox(self.controlArea, "Settings")
        
        cite_layout = QHBoxLayout()
        cite_layout.addWidget(QLabel("Citations:"))
        self.cite_combo = QComboBox()
        self.cite_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cite_combo.currentTextChanged.connect(self._on_column_changed)
        cite_layout.addWidget(self.cite_combo)
        settings_box.layout().addLayout(cite_layout)
        
        # Histogram options
        hist_box = gui.widgetBox(self.controlArea, "Histogram")
        
        gui.spin(hist_box, self, "clip_percentile", 90, 100, label="Clip percentile:",
                 callback=self._on_settings_changed)
        gui.spin(hist_box, self, "n_bins", 10, 100, label="Number of bins:",
                 callback=self._on_settings_changed)
        
        gui.checkBox(hist_box, self, "show_mean", "Show mean line",
                     callback=self._on_line_toggle)
        gui.checkBox(hist_box, self, "show_median", "Show median line",
                     callback=self._on_line_toggle)
        
        # Color selection
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Bar color:"))
        self.color_combo = QComboBox()
        for name in BAR_COLORS.keys():
            self.color_combo.addItem(name)
        self.color_combo.setCurrentIndex(self.bar_color_index)
        self.color_combo.currentIndexChanged.connect(self._on_color_changed)
        color_layout.addWidget(self.color_combo)
        hist_box.layout().addLayout(color_layout)
        
        # Selection
        sel_box = gui.widgetBox(self.controlArea, "Selection")
        
        self.selection_label = QLabel("No selection")
        sel_box.layout().addWidget(self.selection_label)
        
        gui.button(sel_box, self, "Clear Selection", callback=self._clear_selection)
        
        # Apply
        self.apply_btn = gui.button(
            self.controlArea, self, "Analyze",
            callback=self.commit, autoDefault=False
        )
        self.apply_btn.setMinimumHeight(35)
        
        gui.checkBox(self.controlArea, self, "auto_apply", "Auto apply")
        
        self.controlArea.layout().addStretch(1)
    
    def _setup_main_area(self):
        """Build main area."""
        main_layout = QVBoxLayout()
        self.mainArea.layout().addLayout(main_layout)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Histogram tab
        if HAS_PYQTGRAPH:
            self.histogram = HistogramPlot()
            self.histogram.selectionChanged.connect(self._on_histogram_selection)
        else:
            self.histogram = QLabel("PyQtGraph required")
        self.tabs.addTab(self.histogram, "📊 Histogram")
        
        # Log histogram tab
        if HAS_PYQTGRAPH:
            self.log_histogram = LogHistogramPlot()
            self.log_histogram.selectionChanged.connect(self._on_log_histogram_selection)
        else:
            self.log_histogram = QLabel("PyQtGraph required")
        self.tabs.addTab(self.log_histogram, "📈 Log Scale")
        
        # Classes tab
        if HAS_PYQTGRAPH:
            self.classes_chart = CitationClassesChart()
            self.classes_chart.selectionChanged.connect(self._on_classes_selection)
        else:
            self.classes_chart = QLabel("PyQtGraph required")
        self.tabs.addTab(self.classes_chart, "🏷️ Classes")
        
        # Metrics tab
        self.metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_widget)
        self.metrics_table = QTableWidget()
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectRows)
        metrics_layout.addWidget(self.metrics_table)
        self.tabs.addTab(self.metrics_widget, "📋 Metrics")
        
        # Percentiles tab
        self.percentiles_widget = QWidget()
        perc_layout = QVBoxLayout(self.percentiles_widget)
        self.percentiles_table = QTableWidget()
        self.percentiles_table.setSelectionBehavior(QTableWidget.SelectRows)
        perc_layout.addWidget(self.percentiles_table)
        self.tabs.addTab(self.percentiles_widget, "📊 Percentiles")
    
    def _get_bar_color(self) -> str:
        """Get current bar color."""
        color_names = list(BAR_COLORS.keys())
        if 0 <= self.bar_color_index < len(color_names):
            return BAR_COLORS[color_names[self.bar_color_index]]
        return BAR_COLORS["Blue"]
    
    def _on_column_changed(self, column):
        self.citations_column = column
        if self.auto_apply and self._df is not None:
            self.commit()
    
    def _on_settings_changed(self):
        if self.auto_apply and self._result is not None:
            self._update_histograms()
    
    def _on_line_toggle(self):
        if HAS_PYQTGRAPH:
            self.histogram.toggle_mean(self.show_mean)
            self.histogram.toggle_median(self.show_median)
            self.log_histogram.toggle_mean(self.show_mean)
            self.log_histogram.toggle_median(self.show_median)
    
    def _on_color_changed(self, index):
        self.bar_color_index = index
        color = self._get_bar_color()
        if HAS_PYQTGRAPH:
            self.histogram.set_bar_color(color)
            self.log_histogram.set_bar_color(color)
            self.classes_chart.set_bar_color(color)
    
    def _on_histogram_selection(self):
        """Handle histogram selection change."""
        # Clear other selections
        if HAS_PYQTGRAPH:
            self.log_histogram.selection = []
            self.log_histogram._update_selection_display()
            self.classes_chart.selection = []
            self.classes_chart._update_selection_display()
        
        self._update_selection_from_histogram()
    
    def _on_log_histogram_selection(self):
        """Handle log histogram selection change."""
        # Clear other selections
        if HAS_PYQTGRAPH:
            self.histogram.selection = []
            self.histogram._update_selection_display()
            self.classes_chart.selection = []
            self.classes_chart._update_selection_display()
        
        self._update_selection_from_log_histogram()
    
    def _on_classes_selection(self):
        """Handle classes selection change."""
        # Clear other selections
        if HAS_PYQTGRAPH:
            self.histogram.selection = []
            self.histogram._update_selection_display()
            self.log_histogram.selection = []
            self.log_histogram._update_selection_display()
        
        self._update_selection_from_classes()
    
    def _update_selection_from_histogram(self):
        """Update outputs based on histogram selection."""
        if not HAS_PYQTGRAPH or self._result is None or self._data is None:
            return
        
        ranges = self.histogram.get_selected_ranges()
        if not ranges:
            self.selection_label.setText("No selection")
            self.Information.clear()
            self.Outputs.selected_documents.send(None)
            self.Outputs.aggregated.send(None)
            return
        
        self._send_selection_by_ranges(ranges)
    
    def _update_selection_from_log_histogram(self):
        """Update outputs based on log histogram selection."""
        if not HAS_PYQTGRAPH or self._result is None or self._data is None:
            return
        
        ranges = self.log_histogram.get_selected_ranges()
        if not ranges:
            self.selection_label.setText("No selection")
            self.Information.clear()
            self.Outputs.selected_documents.send(None)
            self.Outputs.aggregated.send(None)
            return
        
        self._send_selection_by_ranges(ranges)
    
    def _update_selection_from_classes(self):
        """Update outputs based on classes selection."""
        if not HAS_PYQTGRAPH or self._result is None or self._data is None:
            return
        
        indices = self.classes_chart.get_selected_document_indices()
        class_names = self.classes_chart.get_selected_class_names()
        
        if not indices:
            self.selection_label.setText("No selection")
            self.Information.clear()
            self.Outputs.selected_documents.send(None)
            self.Outputs.aggregated.send(None)
            return
        
        self.selection_label.setText(f"{len(indices):,} papers in {len(class_names)} classes")
        self.Information.selected(len(indices))
        
        selected_data = self._data[indices]
        self.Outputs.selected_documents.send(selected_data)
        
        # Aggregated output for classes
        agg_rows = []
        for cls in class_names:
            if cls in self._result['class_distribution']:
                d = self._result['class_distribution'][cls]
                class_indices = self._result.get('class_indices', {}).get(cls, [])
                if class_indices:
                    cite_col = self.citations_column
                    citations = pd.to_numeric(self._df[cite_col], errors='coerce').fillna(0)
                    class_citations = citations.iloc[class_indices]
                    agg_rows.append({
                        "Class": cls,
                        "Papers": d['count'],
                        "Percentage": f"{d['percentage']:.1f}%",
                        "Total Citations": int(class_citations.sum()),
                        "Mean Citations": round(class_citations.mean(), 2),
                    })
        
        if agg_rows:
            agg_df = pd.DataFrame(agg_rows)
            self.Outputs.aggregated.send(self._df_to_table(agg_df))
        else:
            self.Outputs.aggregated.send(None)
    
    def _send_selection_by_ranges(self, ranges):
        """Send selection outputs for citation ranges."""
        cite_col = self.citations_column
        citations = pd.to_numeric(self._df[cite_col], errors='coerce').fillna(0)
        
        # Document-level output
        mask = np.zeros(len(citations), dtype=bool)
        for min_val, max_val in ranges:
            mask |= (citations >= min_val) & (citations < max_val)
        
        n_selected = int(mask.sum())
        self.selection_label.setText(f"{n_selected:,} papers in {len(ranges)} bins")
        self.Information.selected(n_selected)
        
        if mask.any():
            indices = np.where(mask)[0].tolist()
            selected_data = self._data[indices]
            self.Outputs.selected_documents.send(selected_data)
        else:
            self.Outputs.selected_documents.send(None)
        
        # Aggregated output
        agg_rows = []
        for min_val, max_val in sorted(ranges):
            bin_mask = (citations >= min_val) & (citations < max_val)
            bin_citations = citations[bin_mask]
            
            if len(bin_citations) > 0:
                agg_rows.append({
                    "Bin": f"{min_val:.0f}-{max_val:.0f}",
                    "Papers": int(bin_mask.sum()),
                    "Total Citations": int(bin_citations.sum()),
                    "Mean Citations": round(bin_citations.mean(), 2),
                    "Median Citations": int(bin_citations.median()),
                    "Max Citations": int(bin_citations.max()),
                })
        
        if agg_rows:
            agg_df = pd.DataFrame(agg_rows)
            self.Outputs.aggregated.send(self._df_to_table(agg_df))
        else:
            self.Outputs.aggregated.send(None)
    
    def _clear_selection(self):
        if HAS_PYQTGRAPH:
            self.histogram.clear_selection()
            self.log_histogram.clear_selection()
            self.classes_chart.clear_selection()
        self.Information.clear()
        self.selection_label.setText("No selection")
        self.Outputs.selected_documents.send(None)
        self.Outputs.aggregated.send(None)
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive input data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._result = None
        self._clear_displays()
        
        if data is None:
            self.Error.no_data()
            return
        
        self._df = self._table_to_df(data)
        self._update_cite_columns()
        
        if self.auto_apply:
            self.commit()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)
    
    def _update_cite_columns(self):
        self.cite_combo.clear()
        self._available_cite_cols = []
        
        if self._df is None:
            return
        
        candidates = ["Cited by", "Times Cited", "Citation Count", "cited_by_count", "TC",
                      "Citations", "Total Citations", "Field-Weighted Citation Impact", "FWCI"]
        
        for col in self._df.columns:
            if col in candidates or "cited" in col.lower() or "citation" in col.lower():
                try:
                    pd.to_numeric(self._df[col], errors='coerce')
                    self._available_cite_cols.append(col)
                except:
                    pass
        
        for col in self._available_cite_cols:
            self.cite_combo.addItem(col)
        
        if self.citations_column in self._available_cite_cols:
            self.cite_combo.setCurrentText(self.citations_column)
        elif self._available_cite_cols:
            self.cite_combo.setCurrentIndex(0)
            self.citations_column = self._available_cite_cols[0]
    
    def _clear_displays(self):
        if HAS_PYQTGRAPH:
            self.histogram.clear_plot()
            self.log_histogram.clear_plot()
            self.classes_chart.clear_plot()
        
        self.metrics_table.clear()
        self.metrics_table.setRowCount(0)
        self.percentiles_table.clear()
        self.percentiles_table.setRowCount(0)
        self.selection_label.setText("No selection")
    
    def commit(self):
        self._analyze()
    
    def _analyze(self):
        self.Error.clear()
        self._result = None
        
        if self._df is None:
            self.Error.no_data()
            self._send_outputs()
            return
        
        cite_col = self.citations_column
        if not cite_col or cite_col not in self._df.columns:
            for col in ["Cited by", "Times Cited", "Citation Count", "cited_by_count", "TC"]:
                if col in self._df.columns:
                    cite_col = col
                    self.citations_column = col
                    break
            else:
                self.Error.no_citations()
                self._send_outputs()
                return
        
        try:
            citations = pd.to_numeric(self._df[cite_col], errors='coerce').fillna(0).astype(int).values
            valid_mask = citations >= 0
            citations = citations[valid_mask]
            
            if len(citations) == 0:
                self.Error.no_citations()
                self._send_outputs()
                return
            
            self._result = self._compute_metrics(citations)
            self._update_displays()
            self._send_outputs()
            
            self.Information.analyzed(len(citations))
            
        except Exception as e:
            import traceback
            logger.error(f"Analysis error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
    
    def _compute_metrics(self, citations: np.ndarray) -> Dict:
        n = len(citations)
        
        mean_cit = np.mean(citations)
        median_cit = np.median(citations)
        std_cit = np.std(citations)
        max_cit = np.max(citations)
        min_cit = np.min(citations)
        sum_cit = np.sum(citations)
        
        percentiles = {p: np.percentile(citations, p) for p in [5, 10, 25, 50, 75, 90, 95, 99]}
        
        n_uncited = np.sum(citations == 0)
        h_idx = h_index(citations)
        g_idx = g_index(citations)
        gini = gini_coefficient(citations)
        
        skewness = scipy_stats.skew(citations) if HAS_SCIPY else 0
        kurtosis = scipy_stats.kurtosis(citations) if HAS_SCIPY else 0
        
        p90, p75, p50 = percentiles[90], percentiles[75], percentiles[50]
        classes = [classify_citation(c, p90, p75, p50) for c in citations]
        
        # Build class indices
        class_indices = {}
        for i, cls in enumerate(classes):
            if cls not in class_indices:
                class_indices[cls] = []
            class_indices[cls].append(i)
        
        class_order = ["Uncited", "Low Cited", "Average", "Well Cited (Top 25%)", "Highly Cited (Top 10%)"]
        class_counts = pd.Series(classes).value_counts()
        class_distribution = {
            cls: {"count": int(class_counts.get(cls, 0)), "percentage": class_counts.get(cls, 0) / n * 100}
            for cls in class_order
        }
        
        return {
            "n_papers": n, "citations": citations,
            "basic_stats": {"mean": mean_cit, "median": median_cit, "std": std_cit,
                           "max": max_cit, "min": min_cit, "sum": sum_cit},
            "percentiles": percentiles,
            "uncited": {"count": n_uncited, "percentage": n_uncited / n * 100},
            "h_index": h_idx, "g_index": g_idx, "gini": gini,
            "skewness": skewness, "kurtosis": kurtosis,
            "class_distribution": class_distribution, "classes": classes,
            "class_indices": class_indices,
        }
    
    def _update_displays(self):
        if self._result is None:
            return
        
        self._update_histograms()
        self._update_metrics_table()
        self._update_percentiles_table()
    
    def _update_histograms(self):
        if not HAS_PYQTGRAPH or self._result is None:
            return
        
        color = self._get_bar_color()
        
        # Linear histogram
        self.histogram.set_bar_color(color)
        self.histogram.set_data(
            self._result['citations'], self.n_bins, self.clip_percentile,
            self.show_mean, self.show_median
        )
        
        # Log histogram
        self.log_histogram.set_bar_color(color)
        self.log_histogram.set_data(
            self._result['citations'], self.n_bins,
            self.show_mean, self.show_median
        )
        
        # Classes chart
        self.classes_chart.set_bar_color(color)
        self.classes_chart.set_data(
            self._result['class_distribution'],
            self._result.get('class_indices', {})
        )
    
    def _update_metrics_table(self):
        if self._result is None:
            return
        
        r = self._result
        metrics = [
            ("Number of papers", f"{r['n_papers']:,}"),
            ("Total citations", f"{int(r['basic_stats']['sum']):,}"),
            ("Mean", f"{r['basic_stats']['mean']:.2f}"),
            ("Median", f"{int(r['basic_stats']['median'])}"),
            ("Std deviation", f"{r['basic_stats']['std']:.2f}"),
            ("Max", f"{int(r['basic_stats']['max']):,}"),
            ("H-index", str(r['h_index'])),
            ("G-index", str(r['g_index'])),
            ("Gini coefficient", f"{r['gini']:.3f}"),
            ("Skewness", f"{r['skewness']:.2f}"),
            ("Kurtosis", f"{r['kurtosis']:.2f}"),
            ("Uncited", f"{r['uncited']['count']:,} ({r['uncited']['percentage']:.1f}%)"),
        ]
        
        self.metrics_table.clear()
        self.metrics_table.setRowCount(len(metrics))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        
        for i, (metric, value) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
        
        self.metrics_table.resizeColumnsToContents()
    
    def _update_percentiles_table(self):
        if self._result is None:
            return
        
        percentiles = self._result['percentiles']
        rows = [(f"P{p}", f"{int(v)}") for p, v in sorted(percentiles.items())]
        
        self.percentiles_table.clear()
        self.percentiles_table.setRowCount(len(rows))
        self.percentiles_table.setColumnCount(2)
        self.percentiles_table.setHorizontalHeaderLabels(["Percentile", "Citations"])
        
        for i, (label, value) in enumerate(rows):
            self.percentiles_table.setItem(i, 0, QTableWidgetItem(label))
            self.percentiles_table.setItem(i, 1, QTableWidgetItem(value))
        
        self.percentiles_table.resizeColumnsToContents()
    
    def _send_outputs(self):
        """Send all outputs."""
        if self._result is None:
            self.Outputs.selected_documents.send(None)
            self.Outputs.aggregated.send(None)
            self.Outputs.metrics.send(None)
            self.Outputs.classes.send(None)
            return
        
        # Metrics output
        r = self._result
        metrics_data = [
            ("n_papers", r['n_papers']), ("total_citations", r['basic_stats']['sum']),
            ("mean", r['basic_stats']['mean']), ("median", r['basic_stats']['median']),
            ("h_index", r['h_index']), ("g_index", r['g_index']), ("gini", r['gini']),
        ]
        metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
        self.Outputs.metrics.send(self._df_to_table(metrics_df))
        
        # Citation classes output
        if self._data is not None and len(self._result.get('classes', [])) == len(self._data):
            classes_list = self._result['classes']
            new_domain = Domain(
                self._data.domain.attributes,
                self._data.domain.class_vars,
                list(self._data.domain.metas) + [StringVariable("Citation Class")]
            )
            
            if self._data.metas.shape[1] > 0:
                new_metas = np.column_stack([self._data.metas, np.array(classes_list, dtype=object).reshape(-1, 1)])
            else:
                new_metas = np.array(classes_list, dtype=object).reshape(-1, 1)
            
            classes_table = Table.from_numpy(new_domain, self._data.X,
                                             self._data.Y if self._data.Y.size > 0 else None, new_metas)
            self.Outputs.classes.send(classes_table)
    
    def _df_to_table(self, df: pd.DataFrame) -> Table:
        metas = [StringVariable(str(col)) for col in df.columns]
        domain = Domain([], metas=metas)
        return Table.from_numpy(domain, np.empty((len(df), 0)), metas=df.astype(str).values)


if __name__ == "__main__":
    WidgetPreview(OWCitationDistribution).run()
