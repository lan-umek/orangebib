# -*- coding: utf-8 -*-
"""
Top Items Timeline Widget
=========================
Orange widget for visualizing top items production over time as a bubble plot.

Shows entities (authors, keywords, etc.) on Y-axis, years on X-axis.
Bubble size = document count, bubble color = citation metric.
"""

import logging
from typing import Optional, List, Dict, Tuple
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
    ("Authors", "Authors"),
    ("Author Keywords", "Author Keywords"),
    ("Index Keywords", "Index Keywords"),
    ("Sources", "Source title"),
    ("Countries", "Countries"),
    ("Affiliations", "Affiliations"),
    ("References", "References"),
]

COLOR_BY_OPTIONS = [
    ("Citations per document", "citations_per_doc"),
    ("Total Citations", "total_citations"),
    ("Document Count", "doc_count"),
    ("H-index", "h_index"),
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
    "coolwarm": [(0.230, 0.299, 0.754), (0.411, 0.491, 0.869), (0.585, 0.663, 0.941),
                 (0.748, 0.808, 0.968), (0.878, 0.892, 0.941), (0.957, 0.916, 0.880),
                 (0.970, 0.824, 0.744), (0.945, 0.685, 0.578), (0.880, 0.505, 0.413),
                 (0.769, 0.306, 0.322), (0.706, 0.016, 0.150)],
    "YlOrRd": [(1.000, 1.000, 0.800), (1.000, 0.929, 0.627), (0.996, 0.851, 0.463),
               (0.996, 0.698, 0.298), (0.992, 0.553, 0.235), (0.988, 0.306, 0.165),
               (0.890, 0.102, 0.110), (0.741, 0.000, 0.149), (0.502, 0.000, 0.149)],
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

class EntityAxisItem(AxisItem):
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
                # Truncate long names
                if len(name) > 25:
                    name = name[:22] + "..."
                strings.append(name)
            else:
                strings.append("")
        return strings


# =============================================================================
# PLOT WIDGET
# =============================================================================

class TopItemsPlotGraph(PlotWidget):
    """Custom plot widget for top items timeline visualization."""
    
    def __init__(self, master, parent=None):
        # Create custom left axis for entity names
        self.entity_axis = EntityAxisItem(orientation="left")
        
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
        self.data_points: List[Dict] = []  # Store data for tooltips
        self.selected_indices: List[int] = []
        
        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.showGrid(x=False, y=False, alpha=0.3)
        
        # Color bar for legend
        self.color_bar = None
        
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
        self.setTitle("")
    
    def set_grid(self, show: bool):
        self.showGrid(x=show, y=show, alpha=0.3)
    
    def plot_timeline(self, data: List[Dict], entity_names: List[str],
                      color_map_name: str, color_label: str,
                      title: str, show_size_legend: bool):
        """Plot the bubble timeline.
        
        Args:
            data: List of {entity_idx, year, doc_count, color_value, entity_name}
            entity_names: List of entity names for y-axis
            color_map_name: Name of color map
            color_label: Label for color scale
            title: Plot title
            show_size_legend: Whether to show size legend
        """
        self.clear_plot()
        
        if not data:
            return
        
        self.data_points = data
        self.entity_axis.setEntityNames(entity_names)
        
        # Extract coordinates and values
        x = np.array([d["year"] for d in data])
        y = np.array([d["entity_idx"] for d in data])
        sizes = np.array([d["doc_count"] for d in data])
        color_values = np.array([d["color_value"] for d in data])
        
        # Normalize sizes (min 5, max 30 pixels)
        if sizes.max() > sizes.min():
            size_normalized = 5 + 25 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
        else:
            size_normalized = np.full_like(sizes, 15.0)
        
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
            pen=pg.mkPen(QColor(50, 50, 50, 100), width=0.5),
            hoverable=True,
        )
        self.addItem(self.scatter_item)
        
        # Set axis labels
        self.setLabel('bottom', 'Year')
        
        if title:
            self.setTitle(title)
        
        # Set view range
        x_margin = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 5
        self.setXRange(x.min() - x_margin, x.max() + x_margin)
        self.setYRange(-0.5, len(entity_names) - 0.5)
        
        # Add color bar legend
        self._add_color_legend(cmap, color_values.min(), color_values.max(), color_label)
        
        # Add size legend
        if show_size_legend:
            self._add_size_legend(sizes.min(), sizes.max())
    
    def _add_color_legend(self, cmap: ColorMap, vmin: float, vmax: float, label: str):
        """Add color bar legend."""
        # Create a simple text legend for color scale
        text_item = pg.TextItem(
            html=f'<div style="background: rgba(255,255,255,0.8); padding: 5px;">'
                 f'<b>{label}</b><br>'
                 f'<span style="color: #440154;">●</span> {vmin:.1f}<br>'
                 f'<span style="color: #21918c;">●</span> {(vmin+vmax)/2:.1f}<br>'
                 f'<span style="color: #fde725;">●</span> {vmax:.1f}</div>',
            anchor=(1, 0)
        )
        text_item.setPos(self.getPlotItem().vb.viewRect().right(), 
                         self.getPlotItem().vb.viewRect().top())
        self.addItem(text_item)
    
    def _add_size_legend(self, min_docs: float, max_docs: float):
        """Add size legend."""
        pass  # TODO: Implement size legend if needed
    
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
            dx = (x - point["year"]) / 10  # Scale year
            dy = y - point["entity_idx"]
            dist = dx**2 + dy**2
            if dist < min_dist:
                min_dist = dist
                nearest_point = point
        
        if min_dist < 0.5 and nearest_point:
            tooltip = (f"<b>{nearest_point['entity_name']}</b><br>"
                      f"Year: {nearest_point['year']}<br>"
                      f"Documents: {nearest_point['doc_count']}<br>"
                      f"Citations: {nearest_point.get('total_citations', 'N/A')}<br>"
                      f"Cit/Doc: {nearest_point['color_value']:.2f}")
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
        
        if min_dist < 0.5 and clicked_idx is not None:
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
        
        # Rebuild pens based on selection
        pens = []
        for i in range(len(self.data_points)):
            if i in self.selected_indices:
                pens.append(pg.mkPen(QColor(0, 0, 0), width=3))
            else:
                pens.append(pg.mkPen(QColor(50, 50, 50, 100), width=0.5))
        
        self.scatter_item.setPen(pens)
    
    def get_selected_data(self) -> List[Dict]:
        """Return selected data points."""
        return [self.data_points[i] for i in self.selected_indices if i < len(self.data_points)]


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWTopItemsTimeline(OWWidget):
    """Bubble plot of top items production over time."""
    
    name = "Top Items Timeline"
    description = "Bubble plot showing entity production over time"
    icon = "icons/top_items_timeline.svg"
    priority = 80
    keywords = ["timeline", "bubble", "author", "keyword", "temporal"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")
    
    class Outputs:
        selected_data = Output("Selected Documents", Table, default=True)
        timeline_data = Output("Timeline Data", Table, doc="Computed timeline data")
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
    
    # Settings
    item_type_index = settings.Setting(0)
    column_name = settings.Setting("")
    top_n = settings.Setting(10)
    min_docs = settings.Setting(1)
    color_by_index = settings.Setting(0)
    color_map_index = settings.Setting(0)
    show_grid = settings.Setting(False)
    
    auto_commit = settings.Setting(True)
    
    want_main_area = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_column = Msg("Selected column not found")
        no_year = Msg("Year column not found")
        no_citations = Msg("Citations column not found (using zeros)")
        insufficient_data = Msg("Insufficient data for analysis")
    
    class Information(OWWidget.Information):
        analyzed = Msg("Showing {} entities over {} years")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._columns: List[str] = []
        self._timeline_data: List[Dict] = []
        self._entity_doc_indices: Dict[str, List[int]] = {}
        
        self._setup_gui()
    
    def _setup_gui(self):
        # Item Selection
        item_box = gui.widgetBox(self.controlArea, "Item Selection")
        
        gui.comboBox(
            item_box, self, "item_type_index",
            items=[t[0] for t in ITEM_TYPES],
            label="Item Type:",
            callback=self._on_item_type_changed,
            orientation=Qt.Horizontal,
        )
        
        # Custom column selection (shown when item type doesn't match data)
        gui.label(item_box, self, "Or select column:")
        self.col_combo = gui.comboBox(
            item_box, self, "column_name",
            sendSelectedValue=True,
            callback=self._on_column_changed,
        )
        
        # Settings
        settings_box = gui.widgetBox(self.controlArea, "Settings")
        
        gui.spin(settings_box, self, "top_n", minv=1, maxv=50,
                 label="Top N:", callback=self._on_setting_changed)
        
        gui.spin(settings_box, self, "min_docs", minv=1, maxv=100,
                 label="Min Documents:", callback=self._on_setting_changed)
        
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
        
        # Generate button
        self.generate_btn = gui.button(
            self.controlArea, self, "Generate Plot",
            callback=self._run_analysis,
        )
        self.generate_btn.setMinimumHeight(35)
        
        gui.auto_commit(self.controlArea, self, "auto_commit", "Send Selection")
        
        self.controlArea.layout().addStretch(1)
        
        # Main area - plot
        self.graph = TopItemsPlotGraph(master=self, parent=self.mainArea)
        self.mainArea.layout().addWidget(self.graph)
    
    def _on_item_type_changed(self):
        # Update column selection based on item type
        item_name, col_name = ITEM_TYPES[self.item_type_index]
        if col_name in self._columns:
            idx = self._columns.index(col_name)
            self.col_combo.setCurrentIndex(idx)
            self.column_name = col_name
    
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
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._columns = []
        self._timeline_data = []
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
        """Suggest column based on item type."""
        if not self._columns:
            return
        
        item_name, default_col = ITEM_TYPES[self.item_type_index]
        
        # Try exact match first
        if default_col in self._columns:
            idx = self._columns.index(default_col)
            self.col_combo.setCurrentIndex(idx)
            self.column_name = default_col
            return
        
        # Try pattern matching
        patterns = ["author", "keyword", "source", "affiliation", "country"]
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
            self._compute_timeline(year_col)
        except Exception as e:
            logger.exception(f"Analysis error: {e}")
            self.Error.insufficient_data()
    
    def _compute_timeline(self, year_col: str):
        """Compute timeline data."""
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
        
        # Extract entity data by year
        entity_year_data = defaultdict(lambda: defaultdict(lambda: {"docs": 0, "citations": 0}))
        entity_total = Counter()
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
                entity_year_data[entity][year]["docs"] += 1
                entity_year_data[entity][year]["citations"] += citations
                entity_total[entity] += 1
                self._entity_doc_indices[entity].append(idx)
        
        # Filter by min docs and get top N
        filtered = {e: c for e, c in entity_total.items() if c >= self.min_docs}
        
        if len(filtered) < 1:
            self.Error.insufficient_data()
            return
        
        top_entities = sorted(filtered.items(), key=lambda x: -x[1])[:self.top_n]
        entity_names = [e[0] for e in top_entities]
        
        # Build timeline data points
        color_by = COLOR_BY_OPTIONS[self.color_by_index][1]
        timeline_data = []
        
        for entity_idx, entity in enumerate(entity_names):
            year_data = entity_year_data[entity]
            
            # Calculate entity-level metrics for h-index
            total_citations = sum(yd["citations"] for yd in year_data.values())
            total_docs = sum(yd["docs"] for yd in year_data.values())
            
            for year, yd in year_data.items():
                doc_count = yd["docs"]
                citations = yd["citations"]
                
                # Calculate color value
                if color_by == "citations_per_doc":
                    color_value = citations / doc_count if doc_count > 0 else 0
                elif color_by == "total_citations":
                    color_value = citations
                elif color_by == "doc_count":
                    color_value = doc_count
                else:  # h_index approximation
                    color_value = total_citations / total_docs if total_docs > 0 else 0
                
                timeline_data.append({
                    "entity_idx": entity_idx,
                    "entity_name": entity,
                    "year": year,
                    "doc_count": doc_count,
                    "total_citations": citations,
                    "color_value": color_value,
                })
        
        self._timeline_data = timeline_data
        
        # Get year range for info
        years = set(d["year"] for d in timeline_data)
        
        # Plot
        color_map_name = list(COLOR_MAPS.keys())[self.color_map_index]
        color_label = COLOR_BY_OPTIONS[self.color_by_index][0]
        
        self.graph.plot_timeline(
            data=timeline_data,
            entity_names=entity_names,
            color_map_name=color_map_name,
            color_label=color_label,
            title="Top Items Timeline",
            show_size_legend=True,
        )
        
        self.Information.analyzed(len(entity_names), len(years))
        self._send_timeline_data()
    
    def _send_timeline_data(self):
        """Send timeline data as Orange Table."""
        if not self._timeline_data:
            self.Outputs.timeline_data.send(None)
            return
        
        df = pd.DataFrame(self._timeline_data)
        
        domain = Domain(
            [ContinuousVariable("Year"), ContinuousVariable("Documents"),
             ContinuousVariable("Citations"), ContinuousVariable("ColorValue")],
            metas=[StringVariable("Entity")]
        )
        
        table = Table.from_numpy(
            domain,
            X=df[["year", "doc_count", "total_citations", "color_value"]].values,
            metas=df[["entity_name"]].values.astype(object),
        )
        
        self.Outputs.timeline_data.send(table)
    
    @gui.deferred
    def commit(self):
        """Send selected data."""
        selected = None
        annotated = None
        
        selected_points = self.graph.get_selected_data()
        
        if self._data is not None and selected_points:
            # Get document indices for selected entities/years
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
    WidgetPreview(OWTopItemsTimeline).run()
