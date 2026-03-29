# -*- coding: utf-8 -*-
"""
Entity Over Time Widget
=======================
Orange widget for analyzing entity production over time.

Tracks how authors, keywords, sources, etc. evolve over time.
Uses pyqtgraph for Qt-native visualization with selection and tooltips.
"""

import logging
from typing import Optional, List, Dict
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QToolTip

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.visualize.utils.plotutils import AxisItem, PlotWidget
from Orange.widgets.utils.annotated_data import create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

PLOT_TYPES = [
    ("Per Year", "yearly"),
    ("Cumulative", "cumulative"),
]

COLOR_MAPS = {
    "tab10": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    "Set1": ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
             '#ffff33', '#a65628', '#f781bf', '#999999'],
    "Dark2": ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
              '#e6ab02', '#a6761d', '#666666'],
    "Paired": ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
               '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a'],
    "Pastel": ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6',
               '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2'],
}


# =============================================================================
# PLOT WIDGET
# =============================================================================

class EntityTimePlotGraph(PlotWidget):
    """Custom plot widget for entity over time visualization."""
    
    def __init__(self, master, parent=None):
        super().__init__(
            parent=parent,
            enableMenu=False,
            axisItems={
                "bottom": AxisItem(orientation="bottom"),
                "left": AxisItem(orientation="left"),
            }
        )
        
        self.master = master
        self.line_items: List[pg.PlotDataItem] = []
        self.scatter_items: List[pg.ScatterPlotItem] = []
        self.entity_names: List[str] = []
        self.entity_data: Dict[str, Dict] = {}
        self.colors: List[str] = []
        self.selected_indices: List[int] = []  # indices of selected entities
        
        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.showGrid(x=False, y=False, alpha=0.3)
        
        # Legend
        self.legend = pg.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.getPlotItem())
        self.legend.hide()
        
        # Enable mouse tracking for tooltips
        self.setMouseTracking(True)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
    
    def clear_plot(self):
        """Clear all plot items."""
        self.clear()
        self.line_items = []
        self.scatter_items = []
        self.entity_names = []
        self.entity_data = {}
        self.colors = []
        self.selected_indices = []
        self.legend.clear()
        self.legend.hide()
        self.setTitle("")
    
    def set_grid(self, show: bool):
        self.showGrid(x=show, y=show, alpha=0.3)
    
    def plot_entities(self, data: Dict[str, Dict], colors: List[str],
                      title: str, y_label: str, show_legend: bool):
        """Plot entity time series."""
        self.clear_plot()
        
        if not data:
            return
        
        self.entity_data = data
        self.entity_names = list(data.keys())
        self.colors = colors
        
        # Plot each entity
        for i, (entity, info) in enumerate(data.items()):
            color = colors[i % len(colors)]
            qcolor = QColor(color)
            
            x = np.array(info["years"])
            y = np.array(info["values"])
            
            # Line
            line = pg.PlotDataItem(
                x=x, y=y,
                pen=pg.mkPen(qcolor, width=2),
                name=entity,
            )
            self.addItem(line)
            self.line_items.append(line)
            
            # Scatter points for hover
            scatter = pg.ScatterPlotItem(
                x=x, y=y,
                pen=pg.mkPen(qcolor.darker(120), width=1),
                brush=pg.mkBrush(qcolor),
                size=8,
            )
            self.addItem(scatter)
            self.scatter_items.append(scatter)
        
        # Labels
        self.setLabel('bottom', 'Year')
        self.setLabel('left', y_label)
        
        if title:
            self.setTitle(title)
        
        # Legend
        if show_legend and self.line_items:
            self.legend.clear()
            for entity, line in zip(self.entity_names, self.line_items):
                display_name = entity[:25] + "..." if len(entity) > 25 else entity
                self.legend.addItem(line, display_name)
            self.legend.show()
        
        self.autoRange()
    
    def _find_nearest_point(self, x: float, y: float):
        """Find nearest data point to mouse position."""
        if not self.entity_data:
            return None, None, None, None, float('inf')
        
        # Get view range for scaling
        view_rect = self.getPlotItem().vb.viewRect()
        x_scale = view_rect.width()
        y_scale = view_rect.height()
        
        min_dist = float('inf')
        nearest_entity_idx = None
        nearest_year = None
        nearest_value = None
        nearest_entity = None
        
        for i, (entity, info) in enumerate(self.entity_data.items()):
            years = np.array(info["years"])
            values = np.array(info["values"])
            
            for yr, val in zip(years, values):
                # Normalized distance
                dx = (x - yr) / max(x_scale, 1)
                dy = (y - val) / max(y_scale, 1)
                dist = dx ** 2 + dy ** 2
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_entity_idx = i
                    nearest_entity = entity
                    nearest_year = yr
                    nearest_value = val
        
        return nearest_entity_idx, nearest_entity, nearest_year, nearest_value, min_dist
    
    def _on_mouse_moved(self, pos):
        """Handle mouse move for tooltips."""
        if not self.entity_data:
            QToolTip.hideText()
            return
        
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        idx, entity, year, value, dist = self._find_nearest_point(x, y)
        
        # Show tooltip if close enough (threshold based on view scale)
        if dist < 0.01 and entity is not None:
            total = self.entity_data[entity].get("total", "N/A")
            tooltip = f"<b>{entity}</b><br>Year: {int(year)}<br>Count: {int(value)}<br>Total: {total}"
            global_pos = self.mapToGlobal(self.mapFromScene(pos))
            QToolTip.showText(global_pos, tooltip)
        else:
            QToolTip.hideText()
    
    def _on_mouse_clicked(self, event):
        """Handle mouse click for selection."""
        if not self.entity_data:
            return
        
        pos = event.scenePos()
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        idx, entity, year, value, dist = self._find_nearest_point(x, y)
        
        if dist < 0.02 and idx is not None:
            modifiers = event.modifiers()
            
            if modifiers & Qt.ControlModifier:
                # Toggle selection
                if idx in self.selected_indices:
                    self.selected_indices.remove(idx)
                else:
                    self.selected_indices.append(idx)
            elif modifiers & Qt.ShiftModifier:
                # Add to selection
                if idx not in self.selected_indices:
                    self.selected_indices.append(idx)
            else:
                # Single selection
                self.selected_indices = [idx]
            
            self._update_selection_display()
            self.master.selection_changed()
    
    def _update_selection_display(self):
        """Update visual appearance based on selection."""
        for i, (line, scatter) in enumerate(zip(self.line_items, self.scatter_items)):
            color = QColor(self.colors[i % len(self.colors)])
            
            if self.selected_indices:
                if i in self.selected_indices:
                    # Highlight selected
                    line.setPen(pg.mkPen(color, width=4))
                    scatter.setSize(12)
                    scatter.setBrush(pg.mkBrush(color))
                else:
                    # Dim non-selected
                    dim_color = QColor(color)
                    dim_color.setAlpha(60)
                    line.setPen(pg.mkPen(dim_color, width=1))
                    scatter.setSize(5)
                    scatter.setBrush(pg.mkBrush(dim_color))
            else:
                # Reset to normal
                line.setPen(pg.mkPen(color, width=2))
                scatter.setSize(8)
                scatter.setBrush(pg.mkBrush(color))
    
    def get_selected_entities(self) -> List[str]:
        """Return names of selected entities."""
        return [self.entity_names[i] for i in self.selected_indices if i < len(self.entity_names)]


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWEntityOverTime(OWWidget):
    """Analyze entity production over time."""
    
    name = "Entity Over Time"
    description = "Analyze production over time of authors, keywords, sources"
    icon = "icons/entity_time.svg"
    priority = 75
    keywords = ["temporal", "trend", "time", "entity", "keyword", "author"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")
    
    class Outputs:
        selected_data = Output("Selected Documents", Table, default=True)
        time_series = Output("Time Series", Table, doc="Entity time series")
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
    
    # Settings
    column_name = settings.Setting("")
    top_n = settings.Setting(10)
    min_docs = settings.Setting(1)
    plot_type_index = settings.Setting(0)
    color_map_index = settings.Setting(0)
    show_legend = settings.Setting(True)
    show_grid = settings.Setting(False)
    
    auto_commit = settings.Setting(True)
    
    want_main_area = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_column = Msg("Please select a column")
        no_year = Msg("Year column not found")
        insufficient_data = Msg("Insufficient data for analysis")
    
    class Warning(OWWidget.Warning):
        few_entities = Msg("Only {} entities meet the criteria")
    
    class Information(OWWidget.Information):
        analyzed = Msg("Showing {} entities over {} years")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._columns: List[str] = []
        self._results: Optional[Dict] = None
        self._entity_doc_indices: Dict[str, List[int]] = {}
        
        self._setup_gui()
    
    def _setup_gui(self):
        # Item Selection
        item_box = gui.widgetBox(self.controlArea, "Item Selection")
        
        gui.label(item_box, self, "Analyze:")
        self.col_combo = gui.comboBox(
            item_box, self, "column_name",
            sendSelectedValue=True,
            callback=self._on_column_changed,
        )
        
        # Settings
        settings_box = gui.widgetBox(self.controlArea, "Settings")
        
        gui.spin(settings_box, self, "top_n", minv=1, maxv=100,
                 label="Top N:", callback=self._on_setting_changed)
        
        gui.spin(settings_box, self, "min_docs", minv=1, maxv=1000,
                 label="Min Documents:", callback=self._on_setting_changed)
        
        # Plot Type
        plot_box = gui.widgetBox(self.controlArea, "Plot Type")
        
        gui.comboBox(
            plot_box, self, "plot_type_index",
            items=[p[0] for p in PLOT_TYPES],
            label="View:",
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        # Plot Options
        opt_box = gui.widgetBox(self.controlArea, "Plot Options")
        
        gui.comboBox(
            opt_box, self, "color_map_index",
            items=list(COLOR_MAPS.keys()),
            label="Color Map:",
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        gui.checkBox(opt_box, self, "show_legend", "Show legend",
                     callback=self._on_setting_changed)
        
        gui.checkBox(opt_box, self, "show_grid", "Show grid",
                     callback=self._update_grid)
        
        # Analyze button
        self.analyze_btn = gui.button(
            self.controlArea, self, "Run Analysis",
            callback=self._run_analysis,
        )
        self.analyze_btn.setMinimumHeight(35)
        
        gui.auto_commit(self.controlArea, self, "auto_commit", "Send Selection")
        
        self.controlArea.layout().addStretch(1)
        
        # Main area - plot
        self.graph = EntityTimePlotGraph(master=self, parent=self.mainArea)
        self.mainArea.layout().addWidget(self.graph)
    
    def _on_column_changed(self):
        pass
    
    def _on_setting_changed(self):
        pass
    
    def _update_grid(self):
        self.graph.set_grid(self.show_grid)
    
    def selection_changed(self):
        """Called when graph selection changes."""
        self.commit.deferred()
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._columns = []
        self._results = None
        self._entity_doc_indices = {}
        
        self.col_combo.clear()
        self.graph.clear_plot()
        
        if data is None:
            self.Error.no_data()
            return
        
        self._df = self._table_to_df(data)
        self._columns = list(self._df.columns)
        
        self.col_combo.addItems(self._columns)
        self._suggest_column()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)
    
    def _suggest_column(self):
        """Suggest appropriate column."""
        if not self._columns:
            return
        
        patterns = ["keyword", "author", "source", "affiliation", "country"]
        
        for col in self._columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern in col_lower:
                    idx = self._columns.index(col)
                    self.col_combo.setCurrentIndex(idx)
                    self.column_name = col
                    return
    
    def _get_year_column(self) -> Optional[str]:
        for col in self._columns:
            if col.lower() in ["year", "publication year", "pub year", "pubyear"]:
                return col
        return None
    
    def _run_analysis(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self.Error.no_data()
            return
        
        if not self.column_name or self.column_name not in self._df.columns:
            self.Error.no_column()
            return
        
        year_col = self._get_year_column()
        if year_col is None:
            self.Error.no_year()
            return
        
        try:
            self._compute_entity_time_series(year_col)
        except Exception as e:
            logger.exception(f"Analysis error: {e}")
            self.Error.insufficient_data()
    
    def _compute_entity_time_series(self, year_col: str):
        """Compute entity production over time."""
        df = self._df.copy()
        
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df = df.dropna(subset=[year_col])
        df[year_col] = df[year_col].astype(int)
        
        # Get entity counts per year
        entity_year_counts = defaultdict(lambda: defaultdict(int))
        entity_total_counts = Counter()
        self._entity_doc_indices = defaultdict(list)
        
        for idx, row in df.iterrows():
            year = row[year_col]
            val = row[self.column_name]
            
            if pd.isna(val):
                continue
            
            val_str = str(val)
            
            # Split by separators
            for sep in [";", "|"]:
                if sep in val_str:
                    entities = [e.strip() for e in val_str.split(sep) if e.strip()]
                    break
            else:
                entities = [val_str.strip()] if val_str.strip() else []
            
            for entity in entities:
                entity_year_counts[entity][year] += 1
                entity_total_counts[entity] += 1
                self._entity_doc_indices[entity].append(idx)
        
        # Filter by min docs and get top N
        filtered = {e: c for e, c in entity_total_counts.items() if c >= self.min_docs}
        
        if len(filtered) < 1:
            self.Error.insufficient_data()
            return
        
        if len(filtered) < self.top_n:
            self.Warning.few_entities(len(filtered))
        
        top_entities = sorted(filtered.items(), key=lambda x: -x[1])[:self.top_n]
        top_names = [e[0] for e in top_entities]
        
        # Get year range
        all_years = sorted(df[year_col].unique())
        min_year, max_year = min(all_years), max(all_years)
        year_range = list(range(min_year, max_year + 1))
        
        # Build time series data
        plot_type = PLOT_TYPES[self.plot_type_index][1]
        entity_data = {}
        
        for entity in top_names:
            yearly_counts = entity_year_counts[entity]
            
            values = []
            cumulative = 0
            
            for year in year_range:
                count = yearly_counts.get(year, 0)
                cumulative += count
                
                if plot_type == "cumulative":
                    values.append(cumulative)
                else:
                    values.append(count)
            
            entity_data[entity] = {
                "years": year_range,
                "values": values,
                "total": entity_total_counts[entity],
            }
        
        self._results = {
            "entity_data": entity_data,
            "year_range": year_range,
            "plot_type": plot_type,
        }
        
        self._update_plot()
        self.Information.analyzed(len(top_names), len(year_range))
        self._send_time_series()
    
    def _update_plot(self):
        if self._results is None:
            return
        
        entity_data = self._results["entity_data"]
        plot_type = self._results["plot_type"]
        
        color_map_name = list(COLOR_MAPS.keys())[self.color_map_index]
        colors = COLOR_MAPS[color_map_name]
        
        y_label = "Cumulative Documents" if plot_type == "cumulative" else "Documents"
        title = f"Entity Production Per Year"
        
        self.graph.plot_entities(
            data=entity_data,
            colors=colors,
            title=title,
            y_label=y_label,
            show_legend=self.show_legend,
        )
    
    def _send_time_series(self):
        """Send time series as Orange Table."""
        if self._results is None:
            self.Outputs.time_series.send(None)
            return
        
        entity_data = self._results["entity_data"]
        
        rows = []
        for entity, info in entity_data.items():
            for year, value in zip(info["years"], info["values"]):
                rows.append({"Entity": entity, "Year": year, "Count": value})
        
        if not rows:
            self.Outputs.time_series.send(None)
            return
        
        df = pd.DataFrame(rows)
        
        domain = Domain(
            [ContinuousVariable("Year"), ContinuousVariable("Count")],
            metas=[StringVariable("Entity")]
        )
        
        table = Table.from_numpy(
            domain,
            X=df[["Year", "Count"]].values,
            metas=df[["Entity"]].values.astype(object),
        )
        
        self.Outputs.time_series.send(table)
    
    @gui.deferred
    def commit(self):
        """Send selected data."""
        selected = None
        annotated = None
        
        selected_entities = self.graph.get_selected_entities()
        
        if self._data is not None and selected_entities:
            selected_indices = set()
            for entity in selected_entities:
                if entity in self._entity_doc_indices:
                    selected_indices.update(self._entity_doc_indices[entity])
            
            selected_indices = sorted(selected_indices)
            
            if selected_indices:
                selected = self._data[selected_indices]
                annotated = create_annotated_table(self._data, selected_indices)
        
        if annotated is None and self._data is not None:
            annotated = create_annotated_table(self._data, [])
        
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)


if __name__ == "__main__":
    WidgetPreview(OWEntityOverTime).run()
