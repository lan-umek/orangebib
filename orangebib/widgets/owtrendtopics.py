# -*- coding: utf-8 -*-
"""
Trend Topics Widget
===================
Orange widget for analyzing trending topics ordered by median publication year.

Shows entities on Y-axis (ordered by median year), years on X-axis.
Bubble size = document count, bubble color = citation metric.
Emerging/trending topics appear higher on the plot.
"""

import logging
import re
from typing import Optional, List, Dict
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QToolTip

import pyqtgraph as pg
from pyqtgraph import ColorMap

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

ITEM_TYPES = [
    ("Author Keywords", "Author Keywords"),
    ("Index Keywords", "Index Keywords"),
    ("Authors", "Authors"),
    ("Sources", "Source title"),
    ("Countries", "Countries"),
    ("Affiliations", "Affiliations"),
    ("References", "References"),
]

COLOR_BY_OPTIONS = [
    ("Total Citations", "total_citations"),
    ("Citations per Document", "citations_per_doc"),
    ("Document Count", "doc_count"),
    ("Growth Rate", "growth_rate"),
]

# Color maps
COLOR_MAPS = {
    "viridis": [(0.267, 0.004, 0.329), (0.282, 0.140, 0.458), (0.253, 0.265, 0.529),
                (0.206, 0.371, 0.553), (0.163, 0.471, 0.558), (0.127, 0.566, 0.550),
                (0.134, 0.658, 0.517), (0.266, 0.744, 0.440), (0.477, 0.821, 0.318),
                (0.741, 0.873, 0.150), (0.993, 0.906, 0.144)],
    "plasma": [(0.050, 0.030, 0.528), (0.186, 0.018, 0.590), (0.287, 0.011, 0.607),
               (0.381, 0.002, 0.603), (0.471, 0.016, 0.577), (0.557, 0.047, 0.532),
               (0.640, 0.091, 0.469), (0.720, 0.149, 0.394), (0.798, 0.222, 0.314),
               (0.868, 0.316, 0.226), (0.940, 0.975, 0.131)],
    "inferno": [(0.001, 0.000, 0.014), (0.046, 0.031, 0.186), (0.142, 0.046, 0.361),
                (0.258, 0.039, 0.406), (0.366, 0.071, 0.432), (0.478, 0.106, 0.431),
                (0.590, 0.132, 0.404), (0.702, 0.166, 0.349), (0.817, 0.231, 0.263),
                (0.912, 0.335, 0.153), (0.988, 0.998, 0.645)],
    "cividis": [(0.000, 0.135, 0.304), (0.087, 0.198, 0.360), (0.195, 0.262, 0.390),
                (0.302, 0.327, 0.406), (0.404, 0.393, 0.413), (0.500, 0.458, 0.413),
                (0.592, 0.522, 0.406), (0.683, 0.588, 0.390), (0.776, 0.659, 0.360),
                (0.874, 0.738, 0.304), (0.995, 0.835, 0.161)],
    "YlGnBu": [(1.000, 1.000, 0.851), (0.929, 0.973, 0.694), (0.780, 0.914, 0.706),
               (0.498, 0.804, 0.733), (0.255, 0.714, 0.769), (0.114, 0.569, 0.753),
               (0.133, 0.369, 0.659), (0.145, 0.204, 0.580), (0.031, 0.114, 0.345)],
}


def make_colormap(name: str) -> ColorMap:
    """Create pyqtgraph ColorMap from color stops."""
    colors = COLOR_MAPS.get(name, COLOR_MAPS["viridis"])
    positions = np.linspace(0, 1, len(colors))
    colors_255 = [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
    return ColorMap(positions, colors_255)


# =============================================================================
# CUSTOM AXIS FOR ENTITY NAMES
# =============================================================================

class TrendAxisItem(AxisItem):
    """Custom axis that shows entity names instead of numbers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_names = []
    
    def setEntityNames(self, names: List[str]):
        self.entity_names = names
    
    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            idx = int(round(v))
            if 0 <= idx < len(self.entity_names):
                name = self.entity_names[idx]
                if len(name) > 30:
                    name = name[:27] + "..."
                strings.append(name)
            else:
                strings.append("")
        return strings


# =============================================================================
# PLOT WIDGET
# =============================================================================

class TrendTopicsPlotGraph(PlotWidget):
    """Custom plot widget for trend topics visualization."""
    
    def __init__(self, master, parent=None):
        self.entity_axis = TrendAxisItem(orientation="left")
        
        super().__init__(
            parent=parent,
            enableMenu=False,
            axisItems={
                "bottom": AxisItem(orientation="bottom"),
                "left": self.entity_axis,
            }
        )
        
        self.master = master
        self.scatter_item: Optional[pg.ScatterPlotItem] = None
        self.data_points: List[Dict] = []
        self.selected_indices: List[int] = []
        self.color_bar_item = None
        
        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.showGrid(x=False, y=False, alpha=0.3)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
    
    def clear_plot(self):
        """Clear all plot items."""
        self.clear()
        self.scatter_item = None
        self.data_points = []
        self.selected_indices = []
        self.color_bar_item = None
        self.setTitle("")
    
    def set_grid(self, show: bool):
        self.showGrid(x=show, y=show, alpha=0.3)
    
    def plot_trends(self, data: List[Dict], entity_names: List[str],
                    color_map_name: str, color_label: str,
                    title: str, min_size: float, max_size: float):
        """Plot the trend topics bubble chart.
        
        Args:
            data: List of {entity_idx, year, doc_count, color_value, entity_name, median_year}
            entity_names: List of entity names for y-axis (ordered by median year)
            color_map_name: Name of color map
            color_label: Label for color scale
            title: Plot title
            min_size: Minimum bubble size
            max_size: Maximum bubble size
        """
        self.clear_plot()
        
        if not data:
            return
        
        self.data_points = data
        self.entity_axis.setEntityNames(entity_names)
        
        # Extract data
        x = np.array([d["year"] for d in data])
        y = np.array([d["entity_idx"] for d in data])
        sizes = np.array([d["doc_count"] for d in data])
        color_values = np.array([d["color_value"] for d in data])
        
        # Normalize sizes
        if sizes.max() > sizes.min():
            size_normalized = min_size + (max_size - min_size) * (sizes - sizes.min()) / (sizes.max() - sizes.min())
        else:
            size_normalized = np.full_like(sizes, (min_size + max_size) / 2, dtype=float)
        
        # Get colors from colormap
        cmap = make_colormap(color_map_name)
        if color_values.max() > color_values.min():
            color_normalized = (color_values - color_values.min()) / (color_values.max() - color_values.min())
        else:
            color_normalized = np.full_like(color_values, 0.5)
        
        brushes = []
        for cn in color_normalized:
            color = cmap.map(cn, mode='qcolor')
            brushes.append(pg.mkBrush(color))
        
        # Create scatter plot
        self.scatter_item = pg.ScatterPlotItem(
            x=x, y=y,
            size=size_normalized,
            brush=brushes,
            pen=pg.mkPen(QColor(30, 30, 30, 80), width=0.5),
            hoverable=True,
        )
        self.addItem(self.scatter_item)
        
        # Labels
        self.setLabel('bottom', 'Year')
        self.setLabel('left', 'Keyword')
        
        if title:
            self.setTitle(title)
        
        # Set view range
        x_margin = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 5
        self.setXRange(x.min() - x_margin, x.max() + x_margin)
        self.setYRange(-0.5, len(entity_names) - 0.5)
        
        # Add color bar
        self._add_color_bar(cmap, color_values.min(), color_values.max(), color_label)
        
        # Add size legend
        self._add_size_legend(sizes.min(), sizes.max())
    
    def _add_color_bar(self, cmap: ColorMap, vmin: float, vmax: float, label: str):
        """Add color bar to the right side."""
        # Create color bar using gradient
        bar_width = 20
        bar_height = 200
        
        # Position in view coordinates
        view_rect = self.getPlotItem().vb.viewRect()
        
        # Create a simple text-based legend
        if vmax > 1000:
            vmin_str = f"{vmin/1000:.1f}k"
            vmid_str = f"{(vmin+vmax)/2/1000:.1f}k"
            vmax_str = f"{vmax/1000:.1f}k"
        else:
            vmin_str = f"{vmin:.0f}"
            vmid_str = f"{(vmin+vmax)/2:.0f}"
            vmax_str = f"{vmax:.0f}"
        
        # Get colors for legend
        colors = [cmap.map(v, mode='qcolor') for v in [0, 0.5, 1.0]]
        
        legend_html = f'''
        <div style="background: rgba(255,255,255,0.9); padding: 8px; border: 1px solid #ccc;">
        <div style="text-align: center; margin-bottom: 5px;"><b>{label}</b></div>
        <div><span style="color: rgb({colors[2].red()},{colors[2].green()},{colors[2].blue()});">■</span> {vmax_str}</div>
        <div><span style="color: rgb({colors[1].red()},{colors[1].green()},{colors[1].blue()});">■</span> {vmid_str}</div>
        <div><span style="color: rgb({colors[0].red()},{colors[0].green()},{colors[0].blue()});">■</span> {vmin_str}</div>
        </div>
        '''
        
        self.color_bar_item = pg.TextItem(html=legend_html, anchor=(1, 0))
        self.addItem(self.color_bar_item)
        self.color_bar_item.setPos(view_rect.right() - 5, view_rect.top() + 5)
    
    def _add_size_legend(self, min_docs: float, max_docs: float):
        """Add size legend at bottom."""
        view_rect = self.getPlotItem().vb.viewRect()
        
        # Calculate example sizes
        sizes = [min_docs, (min_docs + max_docs) / 2, max_docs]
        size_strs = [f"{int(s)}" for s in sizes]
        
        legend_html = f'''
        <div style="background: rgba(255,255,255,0.9); padding: 8px; border: 1px solid #ccc;">
        <div style="text-align: center;"><b>Number of documents</b></div>
        <div style="display: flex; align-items: center; justify-content: space-around;">
        <span>● {size_strs[0]}</span>
        <span style="font-size: 1.3em;">● {size_strs[1]}</span>
        <span style="font-size: 1.6em;">● {size_strs[2]}</span>
        </div>
        </div>
        '''
        
        size_legend = pg.TextItem(html=legend_html, anchor=(0.5, 1))
        self.addItem(size_legend)
        size_legend.setPos((view_rect.left() + view_rect.right()) / 2, view_rect.bottom() - 5)
    
    def _on_mouse_moved(self, pos):
        """Handle mouse move for tooltips."""
        if not self.data_points or self.scatter_item is None:
            QToolTip.hideText()
            return
        
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Find nearest point
        min_dist = float('inf')
        nearest_point = None
        
        for point in self.data_points:
            dx = (x - point["year"]) / 10
            dy = y - point["entity_idx"]
            dist = dx**2 + dy**2
            if dist < min_dist:
                min_dist = dist
                nearest_point = point
        
        if min_dist < 0.3 and nearest_point:
            tooltip = (f"<b>{nearest_point['entity_name']}</b><br>"
                      f"Year: {nearest_point['year']}<br>"
                      f"Documents: {nearest_point['doc_count']}<br>"
                      f"Total Citations: {nearest_point.get('total_citations', 0):,}<br>"
                      f"Median Year: {nearest_point.get('median_year', 'N/A')}")
            global_pos = self.mapToGlobal(self.mapFromScene(pos))
            QToolTip.showText(global_pos, tooltip)
        else:
            QToolTip.hideText()
    
    def _on_mouse_clicked(self, event):
        """Handle mouse click for selection."""
        if not self.data_points:
            return
        
        pos = event.scenePos()
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Find clicked point
        min_dist = float('inf')
        clicked_idx = None
        
        for i, point in enumerate(self.data_points):
            dx = (x - point["year"]) / 10
            dy = y - point["entity_idx"]
            dist = dx**2 + dy**2
            if dist < min_dist:
                min_dist = dist
                clicked_idx = i
        
        if min_dist < 0.3 and clicked_idx is not None:
            modifiers = event.modifiers()
            
            if modifiers & Qt.ControlModifier:
                if clicked_idx in self.selected_indices:
                    self.selected_indices.remove(clicked_idx)
                else:
                    self.selected_indices.append(clicked_idx)
            elif modifiers & Qt.ShiftModifier:
                if clicked_idx not in self.selected_indices:
                    self.selected_indices.append(clicked_idx)
            else:
                self.selected_indices = [clicked_idx]
            
            self._update_selection_display()
            self.master.selection_changed()
    
    def _update_selection_display(self):
        """Update visual display of selection."""
        if self.scatter_item is None:
            return
        
        pens = []
        for i in range(len(self.data_points)):
            if i in self.selected_indices:
                pens.append(pg.mkPen(QColor(0, 0, 0), width=3))
            else:
                pens.append(pg.mkPen(QColor(30, 30, 30, 80), width=0.5))
        
        self.scatter_item.setPen(pens)
    
    def get_selected_data(self) -> List[Dict]:
        """Return selected data points."""
        return [self.data_points[i] for i in self.selected_indices if i < len(self.data_points)]


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWTrendTopics(OWWidget):
    """Analyze trending topics ordered by median publication year."""
    
    name = "Trend Topics"
    description = "Analyze trending topics ordered by median publication year"
    icon = "icons/trend_topics.svg"
    priority = 82
    keywords = ["trend", "topic", "emerging", "keyword", "temporal"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")
    
    class Outputs:
        selected_data = Output("Selected Documents", Table, default=True)
        trend_data = Output("Trend Data", Table, doc="Computed trend data")
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
    
    # Settings
    item_type_index = settings.Setting(0)
    column_name = settings.Setting("")
    min_docs = settings.Setting(3)
    top_n_per_year = settings.Setting(3)
    regex_filter = settings.Setting("")
    color_by_index = settings.Setting(0)
    color_map_index = settings.Setting(0)
    show_grid = settings.Setting(False)
    
    auto_commit = settings.Setting(True)
    
    want_main_area = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_column = Msg("Selected column not found")
        no_year = Msg("Year column not found")
        insufficient_data = Msg("Insufficient data for analysis")
        invalid_regex = Msg("Invalid regex pattern: {}")
    
    class Warning(OWWidget.Warning):
        few_topics = Msg("Only {} topics found")
    
    class Information(OWWidget.Information):
        analyzed = Msg("Showing {} trending topics")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._columns: List[str] = []
        self._trend_data: List[Dict] = []
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
        
        gui.spin(settings_box, self, "min_docs", minv=1, maxv=100,
                 label="Min Documents:", callback=self._on_setting_changed)
        
        gui.spin(settings_box, self, "top_n_per_year", minv=1, maxv=20,
                 label="Top N per Year:", callback=self._on_setting_changed)
        
        gui.lineEdit(settings_box, self, "regex_filter",
                     label="Regex Filter:", callback=self._on_setting_changed)
        
        # Plot Options
        opt_box = gui.widgetBox(self.controlArea, "Plot Options")
        
        gui.comboBox(
            opt_box, self, "color_by_index",
            items=[c[0] for c in COLOR_BY_OPTIONS],
            label="Color By:",
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        gui.comboBox(
            opt_box, self, "color_map_index",
            items=list(COLOR_MAPS.keys()),
            label="Color Map:",
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        gui.checkBox(opt_box, self, "show_grid", "Show grid",
                     callback=self._update_grid)
        
        # Run button
        self.run_btn = gui.button(
            self.controlArea, self, "Run Analysis",
            callback=self._run_analysis,
        )
        self.run_btn.setMinimumHeight(35)
        
        gui.auto_commit(self.controlArea, self, "auto_commit", "Send Selection")
        
        self.controlArea.layout().addStretch(1)
        
        # Main area - plot
        self.graph = TrendTopicsPlotGraph(master=self, parent=self.mainArea)
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
        self._trend_data = []
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
    
    def _get_citations_column(self) -> Optional[str]:
        for col in self._columns:
            col_lower = col.lower()
            if "cited" in col_lower or "citation" in col_lower:
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
        
        # Validate regex
        if self.regex_filter:
            try:
                re.compile(self.regex_filter)
            except re.error as e:
                self.Error.invalid_regex(str(e))
                return
        
        try:
            self._compute_trends(year_col)
        except Exception as e:
            logger.exception(f"Analysis error: {e}")
            self.Error.insufficient_data()
    
    def _compute_trends(self, year_col: str):
        """Compute trend topics ordered by median year."""
        df = self._df.copy()
        
        # Ensure year is numeric
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df = df.dropna(subset=[year_col])
        df[year_col] = df[year_col].astype(int)
        
        # Get citations column
        cit_col = self._get_citations_column()
        if cit_col and cit_col in df.columns:
            df[cit_col] = pd.to_numeric(df[cit_col], errors='coerce').fillna(0)
        else:
            cit_col = None
        
        # Compile regex filter if provided
        regex_pattern = None
        if self.regex_filter:
            regex_pattern = re.compile(self.regex_filter, re.IGNORECASE)
        
        # Extract entity data by year
        entity_year_data = defaultdict(lambda: defaultdict(lambda: {"docs": 0, "citations": 0}))
        entity_all_years = defaultdict(list)  # For computing median
        entity_total = defaultdict(lambda: {"docs": 0, "citations": 0})
        self._entity_doc_indices = defaultdict(list)
        
        for idx, row in df.iterrows():
            year = row[year_col]
            val = row[self.column_name]
            citations = row[cit_col] if cit_col else 0
            
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
                # Apply regex filter
                if regex_pattern and not regex_pattern.search(entity):
                    continue
                
                entity_year_data[entity][year]["docs"] += 1
                entity_year_data[entity][year]["citations"] += citations
                entity_all_years[entity].append(year)
                entity_total[entity]["docs"] += 1
                entity_total[entity]["citations"] += citations
                self._entity_doc_indices[entity].append(idx)
        
        # Filter by min docs
        filtered_entities = {e: d for e, d in entity_total.items() if d["docs"] >= self.min_docs}
        
        if len(filtered_entities) < 1:
            self.Error.insufficient_data()
            return
        
        # Calculate median year for each entity
        entity_median_year = {}
        for entity in filtered_entities:
            years = entity_all_years[entity]
            entity_median_year[entity] = np.median(years)
        
        # Select top N per year based on growth/recent appearance
        # Get unique years from entity appearances
        all_years = sorted(set(y for years in entity_all_years.values() for y in years))
        
        # Select trending topics - those with highest median year (most recent)
        # Sort entities by median year (descending) to get trending topics
        sorted_by_median = sorted(entity_median_year.items(), key=lambda x: -x[1])
        
        # Take top entities (limit to reasonable number)
        max_entities = min(len(sorted_by_median), 50)
        top_trending = sorted_by_median[:max_entities]
        
        # Sort them back by median year (ascending) for display (oldest at bottom, newest at top)
        top_trending_sorted = sorted(top_trending, key=lambda x: x[1])
        entity_names = [e[0] for e in top_trending_sorted]
        
        # Build data points
        color_by = COLOR_BY_OPTIONS[self.color_by_index][1]
        trend_data = []
        
        for entity_idx, entity in enumerate(entity_names):
            year_data = entity_year_data[entity]
            median_year = entity_median_year[entity]
            total_docs = entity_total[entity]["docs"]
            total_citations = entity_total[entity]["citations"]
            
            # Calculate growth rate (docs in last 3 years / total)
            recent_years = [y for y in year_data.keys() if y >= max(year_data.keys()) - 3]
            recent_docs = sum(year_data[y]["docs"] for y in recent_years)
            growth_rate = recent_docs / total_docs if total_docs > 0 else 0
            
            for year, yd in year_data.items():
                doc_count = yd["docs"]
                citations = yd["citations"]
                
                # Calculate color value
                if color_by == "total_citations":
                    color_value = citations
                elif color_by == "citations_per_doc":
                    color_value = citations / doc_count if doc_count > 0 else 0
                elif color_by == "doc_count":
                    color_value = doc_count
                else:  # growth_rate
                    color_value = growth_rate
                
                trend_data.append({
                    "entity_idx": entity_idx,
                    "entity_name": entity,
                    "year": year,
                    "doc_count": doc_count,
                    "total_citations": citations,
                    "color_value": color_value,
                    "median_year": median_year,
                    "growth_rate": growth_rate,
                })
        
        self._trend_data = trend_data
        
        if len(entity_names) < 5:
            self.Warning.few_topics(len(entity_names))
        
        # Plot
        color_map_name = list(COLOR_MAPS.keys())[self.color_map_index]
        color_label = COLOR_BY_OPTIONS[self.color_by_index][0]
        
        self.graph.plot_trends(
            data=trend_data,
            entity_names=entity_names,
            color_map_name=color_map_name,
            color_label=color_label,
            title="",
            min_size=5,
            max_size=35,
        )
        
        self.Information.analyzed(len(entity_names))
        self._send_trend_data()
    
    def _send_trend_data(self):
        """Send trend data as Orange Table."""
        if not self._trend_data:
            self.Outputs.trend_data.send(None)
            return
        
        df = pd.DataFrame(self._trend_data)
        
        domain = Domain(
            [ContinuousVariable("Year"), ContinuousVariable("Documents"),
             ContinuousVariable("Citations"), ContinuousVariable("MedianYear"),
             ContinuousVariable("GrowthRate")],
            metas=[StringVariable("Entity")]
        )
        
        table = Table.from_numpy(
            domain,
            X=df[["year", "doc_count", "total_citations", "median_year", "growth_rate"]].values,
            metas=df[["entity_name"]].values.astype(object),
        )
        
        self.Outputs.trend_data.send(table)
    
    @gui.deferred
    def commit(self):
        """Send selected data."""
        selected = None
        annotated = None
        
        selected_points = self.graph.get_selected_data()
        
        if self._data is not None and selected_points:
            selected_indices = set()
            for point in selected_points:
                entity = point["entity_name"]
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
    WidgetPreview(OWTrendTopics).run()
