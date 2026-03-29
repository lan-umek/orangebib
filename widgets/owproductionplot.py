# -*- coding: utf-8 -*-
"""
Production Plot Widget
======================
Orange widget for visualizing scientific production trends.

Uses pyqtgraph for Qt-native visualization following Orange's patterns.
Features: bar/line charts, selection, tooltips, cutpoint with data merging.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QRectF, QEvent, QObject, QPointF
from AnyQt.QtGui import QColor, QFont
from AnyQt.QtWidgets import QApplication, QToolTip

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.visualize.utils.plotutils import AxisItem, PlotWidget
from Orange.widgets.utils.plot import SELECT, PANNING, ZOOMING
from Orange.widgets.utils.annotated_data import create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

BAR_OPTIONS = [
    ("Documents", "Number of Documents"),
    ("Citations", "Total Citations"),
    ("None", None),
]

LINE_OPTIONS = [
    ("None", None),
    ("Cumulative Documents", "Cumulative Documents"),
    ("Cumulative Citations", "Cumulative Citations"),
    ("Documents", "Number of Documents"),
    ("Citations", "Total Citations"),
    ("Trendline", "Trendline"),
]

BAR_COLORS = {
    "Blue": "#4a90d9",
    "Orange": "#e67e22",
    "Green": "#27ae60",
    "Teal": "#16a085",
    "Gray": "#7f8c8d",
    "Red": "#e74c3c",
    "Purple": "#9b59b6",
}

LINE_COLORS = {
    "Black": "#000000",
    "Red": "#e74c3c",
    "Blue": "#3498db",
    "Green": "#27ae60",
    "Orange": "#e67e22",
    "Purple": "#9b59b6",
    "Gray": "#7f8c8d",
}


# =============================================================================
# CUSTOM VIEWBOX WITH SELECTION
# =============================================================================

class ProductionViewBox(pg.ViewBox):
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
# PLOT GRAPH
# =============================================================================

class ProductionPlotGraph(PlotWidget):
    """Custom plot widget for production visualization with selection."""
    
    BAR_WIDTH = 0.7
    
    def __init__(self, master, parent=None):
        self.master = master
        self.selection: List[int] = []
        self.state: int = SELECT
        
        super().__init__(
            parent=parent,
            viewBox=ProductionViewBox(self),
            enableMenu=False,
            axisItems={
                "bottom": AxisItem(orientation="bottom", rotate_ticks=True),
                "left": AxisItem(orientation="left"),
                "right": AxisItem(orientation="right"),
            }
        )
        
        self.bar_item: Optional[pg.BarGraphItem] = None
        self.line_item: Optional[pg.PlotDataItem] = None
        self.line_viewbox: Optional[pg.ViewBox] = None
        
        # Data storage for tooltips
        self._years = None
        self._bar_values = None
        self._line_values = None
        self._bar_label = ""
        self._line_label = ""
        self._bar_color = "#4a90d9"
        
        # Setup
        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.showGrid(x=False, y=True, alpha=0.3)
        
        # Legend
        self.legend = pg.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.getPlotItem())
        self.legend.hide()
        
        self.hideAxis("right")
        
        # Enable mouse tracking for tooltips
        self.setMouseTracking(True)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
    
    def clear_plot(self):
        """Clear all plot items."""
        self.selection = []
        
        if self.bar_item is not None:
            self.removeItem(self.bar_item)
            self.bar_item = None
        
        if self.line_item is not None:
            if self.line_viewbox is not None:
                self.line_viewbox.removeItem(self.line_item)
            else:
                self.removeItem(self.line_item)
            self.line_item = None
        
        self.legend.clear()
        self.legend.hide()
        
        self._years = None
        self._bar_values = None
        self._line_values = None
    
    def _setup_dual_axis(self):
        """Setup secondary y-axis for line plot."""
        if self.line_viewbox is None:
            self.line_viewbox = pg.ViewBox()
            self.getPlotItem().scene().addItem(self.line_viewbox)
            self.getPlotItem().getAxis('right').linkToView(self.line_viewbox)
            self.line_viewbox.setXLink(self.getPlotItem().vb)
            
            def update_views():
                self.line_viewbox.setGeometry(self.getPlotItem().vb.sceneBoundingRect())
                self.line_viewbox.linkedViewChanged(self.getPlotItem().vb, self.line_viewbox.XAxis)
            
            self.getPlotItem().vb.sigResized.connect(update_views)
    
    def plot_data(self, years, bar_values, line_values, 
                  bar_color, line_color, bar_label, line_label,
                  show_legend=True, title=""):
        """Plot the production data."""
        self.clear_plot()
        
        if years is None or len(years) == 0:
            return
        
        # Store data for tooltips
        self._years = years
        self._bar_values = bar_values
        self._line_values = line_values
        self._bar_label = bar_label
        self._line_label = line_label
        self._bar_color = bar_color
        
        x = np.arange(len(years))
        use_dual_axis = (line_values is not None and bar_values is not None)
        
        if use_dual_axis:
            self._setup_dual_axis()
            self.showAxis("right")
        else:
            self.hideAxis("right")
        
        # Plot bars
        if bar_values is not None:
            bar_qcolor = QColor(bar_color)
            
            self.bar_item = pg.BarGraphItem(
                x=x,
                height=bar_values,
                width=self.BAR_WIDTH,
                brush=pg.mkBrush(bar_qcolor),
                pen=pg.mkPen(bar_qcolor.darker(110), width=1),
            )
            self.addItem(self.bar_item)
        
        # Plot line
        if line_values is not None:
            line_qcolor = QColor(line_color)
            pen = pg.mkPen(color=line_qcolor, width=2.5)
            
            self.line_item = pg.PlotDataItem(
                x=x, y=line_values,
                pen=pen,
                symbol='o',
                symbolSize=6,
                symbolBrush=pg.mkBrush(line_qcolor),
                symbolPen=pg.mkPen(line_qcolor.darker(120), width=1),
            )
            
            if use_dual_axis and self.line_viewbox is not None:
                self.line_viewbox.addItem(self.line_item)
                line_min = np.nanmin(line_values) if len(line_values) > 0 else 0
                line_max = np.nanmax(line_values) if len(line_values) > 0 else 1
                margin = (line_max - line_min) * 0.1 if line_max > line_min else 1
                self.line_viewbox.setYRange(max(0, line_min - margin), line_max + margin)
            else:
                self.addItem(self.line_item)
        
        # Set x-axis ticks with string labels
        ticks = [[(i, str(y)) for i, y in enumerate(years)]]
        self.getAxis('bottom').setTicks(ticks)
        
        # Axis labels and colors
        if bar_values is not None:
            self.setLabel('left', bar_label)
            self.getAxis('left').setPen(pg.mkPen(QColor(bar_color)))
            self.getAxis('left').setTextPen(pg.mkPen(QColor(bar_color)))
        elif line_values is not None:
            self.setLabel('left', line_label)
            self.getAxis('left').setPen(pg.mkPen(QColor(line_color)))
            self.getAxis('left').setTextPen(pg.mkPen(QColor(line_color)))
        
        if use_dual_axis:
            self.setLabel('right', line_label)
            self.getAxis('right').setPen(pg.mkPen(QColor(line_color)))
            self.getAxis('right').setTextPen(pg.mkPen(QColor(line_color)))
        
        if title:
            self.setTitle(title)
        
        # Legend
        if show_legend and (bar_values is not None or line_values is not None):
            self.legend.clear()
            if bar_values is not None:
                bar_sample = pg.BarGraphItem(x=[0], height=[1], width=0.5,
                                             brush=pg.mkBrush(QColor(bar_color)))
                self.legend.addItem(bar_sample, bar_label)
            if line_values is not None:
                line_sample = pg.PlotDataItem(pen=pg.mkPen(QColor(line_color), width=2))
                self.legend.addItem(line_sample, line_label)
            self.legend.show()
        
        self._reset_view(x, bar_values, line_values)
    
    def _reset_view(self, x, bar_values, line_values):
        """Reset view to fit all data."""
        if x is None or len(x) == 0:
            return
        
        x_min, x_max = -0.5, len(x) - 0.5
        
        if bar_values is not None:
            y_min = 0
            y_max = np.nanmax(bar_values) if len(bar_values) > 0 else 1
        elif line_values is not None:
            y_min = max(0, np.nanmin(line_values))
            y_max = np.nanmax(line_values) if len(line_values) > 0 else 1
        else:
            y_min, y_max = 0, 1
        
        margin = (y_max - y_min) * 0.15 if y_max > y_min else 1
        self.setXRange(x_min, x_max, padding=0.02)
        self.setYRange(y_min, y_max + margin, padding=0.02)
    
    # =========================================================================
    # TOOLTIPS
    # =========================================================================
    
    def _on_mouse_moved(self, pos):
        """Handle mouse move for tooltips."""
        if self.bar_item is None or self._years is None:
            QToolTip.hideText()
            return
        
        # Convert scene position to view coordinates
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()
        
        # Find bar index
        index = round(x)
        
        if 0 <= index < len(self._years):
            # Check if within bar width
            if abs(x - index) <= self.BAR_WIDTH / 2:
                # Check if within bar height
                if self._bar_values is not None and index < len(self._bar_values):
                    height = self._bar_values[index]
                    if 0 <= y <= height:
                        tooltip_text = self._get_tooltip_text(index)
                        # Get screen position
                        global_pos = self.mapToGlobal(self.mapFromScene(pos))
                        QToolTip.showText(global_pos, tooltip_text)
                        return
        
        QToolTip.hideText()
    
    def _get_tooltip_text(self, index: int) -> str:
        """Generate tooltip text for bar at index."""
        if self._years is None or index >= len(self._years):
            return ""
        
        parts = []
        
        # Year
        year = self._years[index]
        parts.append(f"<b>{year}</b>")
        
        # Bar value
        if self._bar_values is not None and index < len(self._bar_values):
            val = self._bar_values[index]
            if pd.notna(val):
                parts.append(f"{self._bar_label}: <b>{int(val):,}</b>")
        
        # Line value
        if self._line_values is not None and index < len(self._line_values):
            val = self._line_values[index]
            if pd.notna(val):
                if val == int(val):
                    parts.append(f"{self._line_label}: <b>{int(val):,}</b>")
                else:
                    parts.append(f"{self._line_label}: <b>{val:,.2f}</b>")
        
        return "<br>".join(parts)
    
    # =========================================================================
    # SELECTION
    # =========================================================================
    
    def select_by_rectangle(self, rect: QRectF):
        """Select bars within rectangle."""
        if self.bar_item is None:
            return
        
        x0, x1 = sorted((rect.topLeft().x(), rect.bottomRight().x()))
        y0, y1 = sorted((rect.topLeft().y(), rect.bottomRight().y()))
        
        x = self.bar_item.opts["x"]
        height = self.bar_item.opts["height"]
        d = self.BAR_WIDTH / 2
        
        mask = (x0 <= x + d) & (x1 >= x - d) & (y0 <= height) & (y1 >= 0)
        self.select_by_indices(list(np.flatnonzero(mask)))
    
    def select_by_click(self, p):
        """Select bar at click position."""
        if self.bar_item is None:
            return
        
        index = self._get_index_at(p)
        self.select_by_indices([index] if index is not None else [])
    
    def _get_index_at(self, p):
        """Get bar index at position."""
        if self.bar_item is None or self._bar_values is None:
            return None
        
        x = p.x()
        y = p.y()
        index = round(x)
        
        if 0 <= index < len(self._bar_values) and abs(x - index) <= self.BAR_WIDTH / 2:
            height = self._bar_values[index]
            if 0 <= y <= height or height <= y <= 0:
                return index
        return None
    
    def select_by_indices(self, indices: List[int]):
        """Update selection."""
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
        self.master.selection_changed()
    
    def _update_selection_display(self):
        """Update bar appearance based on selection."""
        if self.bar_item is None or self._bar_values is None:
            return
        
        n = len(self._bar_values)
        bar_qcolor = QColor(self._bar_color)
        
        pens = [pg.mkPen(bar_qcolor.darker(110), width=1) for _ in range(n)]
        
        select_pen = pg.mkPen(QColor(Qt.black), width=2)
        select_pen.setStyle(Qt.DashLine)
        
        for idx in self.selection:
            if 0 <= idx < n:
                pens[idx] = select_pen
        
        self.bar_item.setOpts(pens=pens)
    
    # =========================================================================
    # STATE CONTROL
    # =========================================================================
    
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
# MAIN WIDGET
# =============================================================================

class OWProductionPlot(OWWidget):
    """Visualize scientific production trends."""
    
    name = "Production Plot"
    description = "Visualize scientific production with bar and line charts"
    icon = "icons/production_plot.svg"
    priority = 65
    keywords = ["plot", "chart", "visualization", "production", "trend", "bar"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Trend analysis data")
    
    class Outputs:
        selected_data = Output("Selected Documents", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
    
    # Settings
    bar_index = settings.Setting(0)
    line_index = settings.Setting(1)
    bar_color_index = settings.Setting(0)  # Blue
    line_color_index = settings.Setting(0)  # Black
    
    show_grid = settings.Setting(True)
    show_legend = settings.Setting(True)
    
    title = settings.Setting("Scientific Production")
    
    use_cutpoint = settings.Setting(False)
    cutpoint_year = settings.Setting(2000)
    
    auto_commit = settings.Setting(True)
    
    want_main_area = True
    graph_name = "graph"
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_year = Msg("Year column not found in data")
    
    class Warning(OWWidget.Warning):
        missing_bar = Msg("Bar variable '{}' not found")
        missing_line = Msg("Line variable '{}' not found")
    
    class Information(OWWidget.Information):
        selected = Msg("{} year(s) selected")
        merged = Msg("Merged {} years before {}")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._plot_df: Optional[pd.DataFrame] = None  # Processed data for plotting
        self._index_mapping: Optional[List] = None  # Map plot indices to original indices
        
        self._setup_gui()
    
    def _setup_gui(self):
        """Setup the GUI."""
        # Variables
        box = gui.widgetBox(self.controlArea, "Variables")
        
        gui.comboBox(
            box, self, "bar_index",
            label="Bars:",
            items=[opt[0] for opt in BAR_OPTIONS],
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        gui.comboBox(
            box, self, "line_index",
            label="Line:",
            items=[opt[0] for opt in LINE_OPTIONS],
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        # Appearance
        color_box = gui.widgetBox(self.controlArea, "Appearance")
        
        gui.comboBox(
            color_box, self, "bar_color_index",
            label="Bar color:",
            items=list(BAR_COLORS.keys()),
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        gui.comboBox(
            color_box, self, "line_color_index",
            label="Line color:",
            items=list(LINE_COLORS.keys()),
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        # Display
        display_box = gui.widgetBox(self.controlArea, "Display")
        
        gui.checkBox(display_box, self, "show_grid", "Show grid",
                     callback=self._on_grid_changed)
        gui.checkBox(display_box, self, "show_legend", "Show legend",
                     callback=self._on_setting_changed)
        
        # Cutpoint - merges data before the cutpoint year
        cutpoint_box = gui.widgetBox(self.controlArea, "Cutpoint")
        
        gui.checkBox(cutpoint_box, self, "use_cutpoint", 
                     "Merge years before:",
                     callback=self._on_setting_changed)
        
        gui.spin(cutpoint_box, self, "cutpoint_year",
                 minv=1900, maxv=2100, step=1,
                 callback=self._on_setting_changed)
        
        # Title
        title_box = gui.widgetBox(self.controlArea, "Title")
        gui.lineEdit(title_box, self, "title", callback=self._on_setting_changed)
        
        # Commit
        gui.auto_commit(self.controlArea, self, "auto_commit", "Send Selection")
        
        self.controlArea.layout().addStretch(1)
        
        # Main area - plot
        self.graph = ProductionPlotGraph(master=self, parent=self.mainArea)
        self.mainArea.layout().addWidget(self.graph)
    
    def _on_setting_changed(self):
        self._update_plot()
    
    def _on_grid_changed(self):
        self.graph.showGrid(x=False, y=self.show_grid, alpha=0.3)
    
    def selection_changed(self):
        """Called when graph selection changes."""
        n = len(self.graph.selection)
        if n > 0:
            self.Information.selected(n)
        else:
            self.Information.clear()
        self.commit.deferred()
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive input data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._plot_df = None
        self._index_mapping = None
        
        if data is None:
            self.Error.no_data()
            self.graph.clear_plot()
            self.commit.now()
            return
        
        self._df = self._table_to_df(data)
        
        if not self._has_year_column():
            self.Error.no_year()
            self.graph.clear_plot()
            self.commit.now()
            return
        
        # Set cutpoint default based on data range
        year_col = self._get_year_column()
        if year_col:
            try:
                min_year = int(self._df[year_col].min())
                max_year = int(self._df[year_col].max())
                if self.cutpoint_year < min_year or self.cutpoint_year > max_year:
                    self.cutpoint_year = min_year + (max_year - min_year) // 4
            except (ValueError, TypeError):
                pass
        
        self._update_plot()
        self.commit.now()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)
    
    def _has_year_column(self) -> bool:
        if self._df is None:
            return False
        return any(col in self._df.columns for col in ["Year", "Period"])
    
    def _get_year_column(self) -> Optional[str]:
        for col in ["Year", "Period"]:
            if col in self._df.columns:
                return col
        return None
    
    def _process_data_with_cutpoint(self) -> pd.DataFrame:
        """Process data, merging years before cutpoint if enabled."""
        if self._df is None:
            return None
        
        df = self._df.copy()
        year_col = self._get_year_column()
        
        # Ensure Year is integer
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
        
        if not self.use_cutpoint:
            # No cutpoint - just ensure years are integers
            self._index_mapping = list(range(len(df)))
            return df
        
        # Split data before and after cutpoint
        before_mask = df[year_col] < self.cutpoint_year
        before = df[before_mask]
        after = df[~before_mask]
        
        if before.empty:
            self._index_mapping = list(range(len(df)))
            return df
        
        # Sum numeric columns for "before" rows
        merged_row = {year_col: f"<{self.cutpoint_year}"}
        
        for col in df.columns:
            if col == year_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                # For cumulative columns, take the last value before cutpoint
                if "Cumulative" in col:
                    merged_row[col] = before[col].iloc[-1] if len(before) > 0 else 0
                else:
                    # Sum for other numeric columns
                    merged_row[col] = before[col].sum()
        
        # Create merged dataframe
        merged_df = pd.concat([
            pd.DataFrame([merged_row]),
            after.reset_index(drop=True)
        ], ignore_index=True)
        
        # Index mapping: first index maps to all "before" indices
        before_indices = list(before.index)
        after_indices = list(after.index)
        self._index_mapping = [before_indices] + [[i] for i in after_indices]
        
        # Show info message
        n_merged = len(before)
        if n_merged > 0:
            self.Information.merged(n_merged, self.cutpoint_year)
        
        return merged_df
    
    def _update_plot(self):
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None or self._df.empty:
            self.graph.clear_plot()
            return
        
        # Process data with cutpoint
        self._plot_df = self._process_data_with_cutpoint()
        
        if self._plot_df is None or self._plot_df.empty:
            self.graph.clear_plot()
            return
        
        year_col = self._get_year_column()
        if year_col is None:
            self.graph.clear_plot()
            return
        
        # Convert years to strings (already formatted by cutpoint processing)
        years = [str(y) for y in self._plot_df[year_col].values]
        
        # Bar data
        bar_name, bar_col = BAR_OPTIONS[self.bar_index]
        bar_values = None
        bar_label = bar_name
        if bar_col is not None:
            if bar_col in self._plot_df.columns:
                bar_values = self._plot_df[bar_col].values.astype(float)
                bar_label = bar_col
            else:
                self.Warning.missing_bar(bar_col)
        
        # Line data
        line_name, line_col = LINE_OPTIONS[self.line_index]
        line_values = None
        line_label = line_name
        if line_col is not None:
            if line_col in self._plot_df.columns:
                line_values = self._plot_df[line_col].values.astype(float)
                line_label = line_col
            else:
                self.Warning.missing_line(line_col)
        
        # Colors
        bar_color_name = list(BAR_COLORS.keys())[self.bar_color_index]
        bar_color = BAR_COLORS[bar_color_name]
        
        line_color_name = list(LINE_COLORS.keys())[self.line_color_index]
        line_color = LINE_COLORS[line_color_name]
        
        self.graph.plot_data(
            years=years,
            bar_values=bar_values,
            line_values=line_values,
            bar_color=bar_color,
            line_color=line_color,
            bar_label=bar_label,
            line_label=line_label,
            show_legend=self.show_legend,
            title=self.title,
        )
    
    @gui.deferred
    def commit(self):
        """Send selected data."""
        selected = None
        annotated = None
        
        if self._data is not None and self.graph.selection and self._index_mapping:
            # Map plot selection to original data indices
            original_indices = []
            for plot_idx in self.graph.selection:
                if plot_idx < len(self._index_mapping):
                    mapping = self._index_mapping[plot_idx]
                    if isinstance(mapping, list):
                        original_indices.extend(mapping)
                    else:
                        original_indices.append(mapping)
            
            original_indices = sorted(set(original_indices))
            
            if original_indices:
                selected = self._data[original_indices]
                annotated = create_annotated_table(self._data, original_indices)
        
        if annotated is None and self._data is not None:
            annotated = create_annotated_table(self._data, [])
        
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)
    
    def send_report(self):
        self.report_plot()
        if self._plot_df is not None:
            self.report_caption(f"Scientific production: {len(self._plot_df)} time periods")


if __name__ == "__main__":
    WidgetPreview(OWProductionPlot).run()
