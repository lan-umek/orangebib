# -*- coding: utf-8 -*-
"""
Group Counts Plot Widget
========================
Interactive PyQtGraph visualization of group counts comparison.

Features:
- Native Qt plotting with PyQtGraph (no matplotlib)
- Horizontal grouped bar chart comparing entities across groups
- Hover tooltips showing item name, count, percentage
- Click selection to output selected items
- Ctrl+click multi-select, click empty to deselect
- Top-N per group, sort options, export to PNG
"""

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QToolTip, QApplication,
    QFileDialog, QSizePolicy,
)
from AnyQt.QtCore import Qt, pyqtSignal, QRectF
from AnyQt.QtGui import QFont, QColor

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.settings import Setting

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
# CONSTANTS
# =============================================================================

GROUP_PALETTE = [
    "#3b82f6", "#f97316", "#22c55e", "#ef4444", "#a855f7",
    "#06b6d4", "#ec4899", "#84cc16", "#f59e0b", "#6366f1",
    "#14b8a6", "#f43f5e", "#8b5cf6", "#10b981", "#e11d48",
]

SELECT, PANNING, ZOOMING = 0, 1, 2


# =============================================================================
# SELECTABLE VIEWBOX
# =============================================================================

if HAS_PYQTGRAPH:
    class SelectableViewBox(pg.ViewBox):
        """ViewBox that forwards click events to the chart for selection."""

        def __init__(self, chart, **kw):
            super().__init__(**kw)
            self.chart = chart

        def mouseClickEvent(self, ev):
            if ev.button() == Qt.LeftButton and self.chart.state == SELECT:
                self.chart.select_by_click(
                    self.mapSceneToView(ev.scenePos())
                )
                ev.accept()
            else:
                super().mouseClickEvent(ev)

        def mouseDragEvent(self, ev, axis=None):
            if self.chart.state == SELECT:
                ev.accept()
                if ev.isFinish():
                    p1 = self.mapSceneToView(ev.buttonDownScenePos())
                    p2 = self.mapSceneToView(ev.scenePos())
                    self.chart.select_by_rectangle(QRectF(p1, p2))
                else:
                    self.updateScaleBox(ev.buttonDownPos(), ev.pos())
            else:
                super().mouseDragEvent(ev, axis=axis)


# =============================================================================
# INTERACTIVE HORIZONTAL BAR CHART
# =============================================================================

if HAS_PYQTGRAPH:
    class GroupBarChart(pg.PlotWidget):
        """
        Horizontal grouped bar chart with hover and click selection.

        Each entity gets one sub-bar per group, positioned along the Y axis.
        Hover shows a tooltip; click selects/deselects items.
        """

        selectionChanged = pyqtSignal(list)  # emits list of selected item names

        def __init__(self, parent=None):
            self._selection: List[str] = []
            self.state = SELECT

            super().__init__(
                parent=parent,
                viewBox=SelectableViewBox(self),
                enableMenu=True,
            )

            # Styling
            self.setBackground('w')
            pi = self.getPlotItem()
            pi.hideAxis('top')
            pi.hideAxis('right')
            self.showGrid(x=False, y=False)

            # Data
            self._bars: List[Dict] = []
            self._bar_items: List[pg.BarGraphItem] = []
            self._group_colors: Dict[str, str] = {}
            self._group_names: List[str] = []
            self._item_names: List[str] = []

            # Hover tracking
            self.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # -----------------------------------------------------------------
        # DATA
        # -----------------------------------------------------------------

        def set_data(
            self,
            items: List[str],
            groups: List[str],
            values: Dict[str, List[float]],
            percentages: Optional[Dict[str, List[float]]] = None,
            colors: Optional[Dict[str, str]] = None,
        ):
            """
            Set chart data for horizontal grouped bar chart.

            Parameters
            ----------
            items : list of str
                Entity names (shown on Y axis), sorted bottom → top.
            groups : list of str
                Group names.
            values : dict[group_name → list[float]]
                Counts per group, aligned with `items`.
            percentages : dict, optional
                Percentage per group per item.
            colors : dict, optional
                group_name → hex color.
            """
            self.clear()
            self._bars = []
            self._bar_items = []
            self._selection = []
            self._item_names = list(items)
            self._group_names = list(groups)

            if not items or not groups:
                return

            n_items = len(items)
            n_groups = len(groups)

            # Colors
            if colors:
                self._group_colors = dict(colors)
            else:
                self._group_colors = {
                    g: GROUP_PALETTE[i % len(GROUP_PALETTE)]
                    for i, g in enumerate(groups)
                }

            # Bar geometry
            bar_height = 0.8 / n_groups
            gap = bar_height * 0.1  # small gap between sub-bars

            for g_idx, group in enumerate(groups):
                vals = values.get(group, [0.0] * n_items)
                pcts = (percentages or {}).get(group, [None] * n_items)
                color = self._group_colors[group]

                ys = []
                ws = []
                brushes = []
                pens = []

                for i_idx in range(n_items):
                    # Y position: item slot + group offset
                    y_center = i_idx + 0.1 + g_idx * bar_height + bar_height / 2
                    val = vals[i_idx] if i_idx < len(vals) else 0.0
                    pct = pcts[i_idx] if i_idx < len(pcts) else None

                    ys.append(y_center)
                    ws.append(max(val, 0))

                    brushes.append(pg.mkBrush(color))
                    pens.append(pg.mkPen('w', width=0.5))

                    self._bars.append({
                        'y_center': y_center,
                        'y_start': y_center - (bar_height - gap) / 2,
                        'y_end': y_center + (bar_height - gap) / 2,
                        'width': max(val, 0),
                        'group': group,
                        'item': items[i_idx],
                        'count': val,
                        'pct': pct,
                        'color': color,
                        'g_idx': g_idx,
                        'i_idx': i_idx,
                    })

                # Create the bar item — x0=0 is critical for horizontal bars
                bar_item = pg.BarGraphItem(
                    x0=0,
                    y=ys,
                    width=ws,
                    height=bar_height - gap,
                    brushes=brushes,
                    pens=pens,
                )
                self.addItem(bar_item)
                self._bar_items.append(bar_item)

            # Y axis tick labels
            y_axis = self.getAxis('left')
            ticks = [(i + 0.5, str(items[i])) for i in range(n_items)]
            y_axis.setTicks([ticks])

            # Invert so first item (highest ranked) is at top
            self.getPlotItem().invertY(True)

            # Labels
            self.setLabel('bottom', 'Number of documents')

            # Force view to fit data
            self.getPlotItem().getViewBox().autoRange()

        # -----------------------------------------------------------------
        # HOVER
        # -----------------------------------------------------------------

        def _on_mouse_moved(self, pos):
            """Show tooltip on hover over a bar."""
            if not self._bars:
                QToolTip.hideText()
                return

            vb = self.getPlotItem().vb
            mouse_point = vb.mapSceneToView(pos)
            mx, my = mouse_point.x(), mouse_point.y()

            for bar in self._bars:
                if (bar['y_start'] <= my <= bar['y_end'] and
                        0 <= mx <= bar['width'] and bar['width'] > 0):
                    parts = [
                        f"<b>{bar['item']}</b>",
                        f"Group: <b>{bar['group']}</b>",
                        f"Count: {bar['count']:.0f}",
                    ]
                    if bar['pct'] is not None:
                        parts.append(f"Percentage: {bar['pct']:.1f}%")
                    if bar['item'] in self._selection:
                        parts.append("<i>(selected)</i>")

                    tooltip = "<br>".join(parts)
                    global_pos = self.mapToGlobal(self.mapFromScene(pos))
                    QToolTip.showText(global_pos, tooltip)
                    return

            QToolTip.hideText()

        # -----------------------------------------------------------------
        # SELECTION
        # -----------------------------------------------------------------

        def select_by_click(self, point):
            """Handle click at point (in view coordinates)."""
            mx, my = point.x(), point.y()

            clicked_item = None
            for bar in self._bars:
                if (bar['y_start'] <= my <= bar['y_end'] and
                        0 <= mx <= bar['width'] and bar['width'] > 0):
                    clicked_item = bar['item']
                    break

            keys = QApplication.keyboardModifiers()

            if clicked_item is None:
                # Click on empty → clear
                if not (keys & (Qt.ControlModifier | Qt.ShiftModifier)):
                    self._selection = []
            elif keys & Qt.ControlModifier:
                # Ctrl+click → toggle
                if clicked_item in self._selection:
                    self._selection.remove(clicked_item)
                else:
                    self._selection.append(clicked_item)
            elif keys & Qt.ShiftModifier:
                # Shift+click → add
                if clicked_item not in self._selection:
                    self._selection.append(clicked_item)
            else:
                # Plain click → select only this
                self._selection = [clicked_item]

            self._update_selection_display()
            self.selectionChanged.emit(list(self._selection))

        def select_by_rectangle(self, rect: QRectF):
            """Select items whose bars overlap the selection rectangle."""
            if not self._bars:
                return

            y_min = min(rect.top(), rect.bottom())
            y_max = max(rect.top(), rect.bottom())

            selected_items = set()
            for bar in self._bars:
                if bar['y_end'] >= y_min and bar['y_start'] <= y_max:
                    selected_items.add(bar['item'])

            keys = QApplication.keyboardModifiers()
            if keys & Qt.ControlModifier:
                current = set(self._selection)
                self._selection = list(current ^ selected_items)
            elif keys & Qt.ShiftModifier:
                current = set(self._selection)
                self._selection = list(current | selected_items)
            else:
                self._selection = list(selected_items)

            self._update_selection_display()
            self.selectionChanged.emit(list(self._selection))

        def _update_selection_display(self):
            """Re-color bars based on selection."""
            if not self._bar_items or not self._bars:
                return

            has_selection = bool(self._selection)

            for g_idx, bar_item in enumerate(self._bar_items):
                group = self._group_names[g_idx]
                n_items = len(self._item_names)
                base_color = self._group_colors[group]

                brushes = []
                pens = []
                for i_idx in range(n_items):
                    item_name = self._item_names[i_idx]

                    if not has_selection or item_name in self._selection:
                        brushes.append(pg.mkBrush(base_color))
                        pens.append(pg.mkPen('w', width=0.5))
                    else:
                        # Dim unselected
                        c = QColor(base_color)
                        c.setAlpha(50)
                        brushes.append(pg.mkBrush(c))
                        pens.append(pg.mkPen(QColor(220, 220, 220), width=0.5))

                bar_item.setOpts(brushes=brushes, pens=pens)

        def get_selected_items(self) -> List[str]:
            return list(self._selection)

        def clear_selection(self):
            self._selection = []
            self._update_selection_display()
            self.selectionChanged.emit([])


# =============================================================================
# WIDGET
# =============================================================================

class OWGroupCountsPlot(OWWidget):
    """Interactive plot for group count comparisons."""

    name = "Group Counts Plot"
    description = ("Interactive horizontal bar chart comparing entity "
                    "frequencies across document groups with hover and selection")
    icon = "icons/group_counts.svg"
    priority = 32
    keywords = ["group", "plot", "bar chart", "comparison", "visualization",
                "hover", "selection", "pyqtgraph"]
    category = "Biblium"

    class Inputs:
        counts = Input("Counts", Table,
                        doc="Group counts table (from Group Counts widget)")

    class Outputs:
        selected_items = Output("Selected Items", Table,
                                 doc="Rows matching selected entities")

    # Settings
    top_n = Setting(10)
    sort_by_idx = Setting(0)  # 0=combined, 1+=group index
    auto_apply = Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No counts data")
        no_pyqtgraph = Msg("PyQtGraph not installed — cannot display plot")
        no_groups = Msg("No group columns found in counts data")
        plot_error = Msg("Plot error: {}")

    class Warning(OWWidget.Warning):
        no_item_col = Msg("Could not identify entity name column")

    class Information(OWWidget.Information):
        showing = Msg("Showing {} items across {} groups")

    def __init__(self):
        super().__init__()

        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._item_col: Optional[str] = None
        self._group_names: List[str] = []
        self._count_cols: Dict[str, str] = {}   # group_name → column name
        self._pct_cols: Dict[str, str] = {}

        self._setup_control_area()
        self._setup_main_area()

    # =========================================================================
    # UI
    # =========================================================================

    def _setup_control_area(self):
        """Build control area."""
        # --- Display Options ---
        display_box = gui.widgetBox(self.controlArea, "📊 Display Options")

        topn_form = QGridLayout()
        topn_form.setColumnStretch(1, 1)

        topn_form.addWidget(QLabel("Top N per group:"), 0, 0)
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(3, 50)
        self.top_n_spin.setValue(self.top_n)
        self.top_n_spin.valueChanged.connect(self._on_top_n_changed)
        self.top_n_spin.setToolTip("Number of top items per group to display")
        topn_form.addWidget(self.top_n_spin, 0, 1)

        topn_form.addWidget(QLabel("Sort by:"), 1, 0)
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Combined total")
        self.sort_combo.setCurrentIndex(self.sort_by_idx)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        topn_form.addWidget(self.sort_combo, 1, 1)

        display_box.layout().addLayout(topn_form)

        # --- Actions ---
        btn_layout = QVBoxLayout()

        self.refresh_btn = QPushButton("🔄 Refresh Plot")
        self.refresh_btn.clicked.connect(self._refresh_plot)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; border: none;
                border-radius: 4px; padding: 8px 16px;
                color: white; font-weight: bold;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        btn_layout.addWidget(self.refresh_btn)

        self.clear_sel_btn = QPushButton("Clear Selection")
        self.clear_sel_btn.clicked.connect(self._clear_selection)
        self.clear_sel_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e7ff; border: 1px solid #6366f1;
                border-radius: 4px; padding: 6px 12px;
                color: #4338ca; font-weight: bold;
            }
            QPushButton:hover { background-color: #c7d2fe; }
        """)
        btn_layout.addWidget(self.clear_sel_btn)

        self.export_btn = QPushButton("📷 Export PNG")
        self.export_btn.clicked.connect(self._export_png)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e7ff; border: 1px solid #6366f1;
                border-radius: 4px; padding: 6px 12px;
                color: #4338ca; font-weight: bold;
            }
            QPushButton:hover { background-color: #c7d2fe; }
        """)
        btn_layout.addWidget(self.export_btn)

        self.controlArea.layout().addLayout(btn_layout)

        # --- Status ---
        status_box = gui.widgetBox(self.controlArea, "Status")
        self.status_label = QLabel("No data")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        status_box.layout().addWidget(self.status_label)

        # --- Legend ---
        self.legend_box = gui.widgetBox(self.controlArea, "Legend")
        self.legend_layout = self.legend_box.layout()

    def _setup_main_area(self):
        """Build main area with the chart."""
        if not HAS_PYQTGRAPH:
            lbl = QLabel("PyQtGraph is required for this widget.\n"
                         "Install: pip install pyqtgraph")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #888; font-size: 14px;")
            self.mainArea.layout().addWidget(lbl)
            return

        self.chart = GroupBarChart()
        self.chart.selectionChanged.connect(self._on_selection_changed)
        self.chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mainArea.layout().addWidget(self.chart)

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def _on_top_n_changed(self, val):
        self.top_n = val
        if self._df is not None:
            self._refresh_plot()

    def _on_sort_changed(self, idx):
        self.sort_by_idx = idx
        if self._df is not None:
            self._refresh_plot()

    def _on_selection_changed(self, selected_items: List[str]):
        """Handle selection from chart."""
        if not selected_items:
            self.Outputs.selected_items.send(None)
            return

        if self._df is None or self._item_col is None:
            return

        mask = self._df[self._item_col].isin(selected_items)
        selected_df = self._df[mask]

        if selected_df.empty:
            self.Outputs.selected_items.send(None)
        else:
            self.Outputs.selected_items.send(self._df_to_table(selected_df))

        self.status_label.setText(
            f"Selected {len(selected_items)} items ({len(selected_df)} rows)"
        )

    # =========================================================================
    # INPUT
    # =========================================================================

    @Inputs.counts
    def set_counts(self, data: Optional[Table]):
        """Receive counts table from Group Counts widget."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()

        self._data = data
        self._df = None
        self._item_col = None
        self._group_names = []
        self._count_cols = {}
        self._pct_cols = {}

        if data is None:
            self._clear_chart()
            self.status_label.setText("No data")
            self.Outputs.selected_items.send(None)
            return

        if not HAS_PYQTGRAPH:
            self.Error.no_pyqtgraph()
            return

        self._df = self._table_to_df(data)
        self._parse_columns()

        if not self._group_names:
            self.Error.no_groups()
            return

        if self._item_col is None:
            self.Warning.no_item_col()
            return

        # Update sort combo
        self.sort_combo.blockSignals(True)
        self.sort_combo.clear()
        self.sort_combo.addItem("Combined total")
        for g in self._group_names:
            self.sort_combo.addItem(g)
        if self.sort_by_idx < self.sort_combo.count():
            self.sort_combo.setCurrentIndex(self.sort_by_idx)
        self.sort_combo.blockSignals(False)

        self._refresh_plot()

    def _parse_columns(self):
        """Detect item column and group count/percentage columns."""
        if self._df is None:
            return

        cols = list(self._df.columns)

        # Item column = first non-numeric column
        for c in cols:
            if not pd.api.types.is_numeric_dtype(self._df[c]):
                self._item_col = c
                break

        # Group count columns: "Number of documents (GroupName)"
        for c in cols:
            if c.startswith("Number of documents (") and c.endswith(")"):
                group_name = c[len("Number of documents ("):-1]
                if group_name.lower() == "combined":
                    continue
                self._group_names.append(group_name)
                self._count_cols[group_name] = c

        # Percentage columns
        for c in cols:
            if c.startswith("Percentage of documents (") and c.endswith(")"):
                group_name = c[len("Percentage of documents ("):-1]
                if group_name.lower() != "combined":
                    self._pct_cols[group_name] = c

        logger.info(
            f"Parsed: item_col={self._item_col}, "
            f"groups={self._group_names}, "
            f"count_cols={list(self._count_cols.keys())}"
        )

    # =========================================================================
    # PLOT
    # =========================================================================

    def _refresh_plot(self):
        """Rebuild the chart from current data and settings."""
        if self._df is None or not self._group_names or self._item_col is None:
            return

        try:
            self._do_refresh()
        except Exception as e:
            self.Error.plot_error(str(e))
            logger.exception("Plot refresh failed")

    def _do_refresh(self):
        """Internal refresh logic."""
        top_n = self.top_n
        sort_idx = self.sort_by_idx

        # ── Determine sort column ──
        if sort_idx == 0:
            combined_col = None
            for c in self._df.columns:
                if c == "Number of documents (Combined)":
                    combined_col = c
                    break
            if combined_col:
                sort_col = combined_col
            else:
                # Fallback: sum all group count columns
                sum_vals = pd.Series(0.0, index=self._df.index)
                for c in self._count_cols.values():
                    sum_vals += pd.to_numeric(self._df[c], errors='coerce').fillna(0)
                self._df["_combined_sum"] = sum_vals
                sort_col = "_combined_sum"
        else:
            g_idx = sort_idx - 1
            if g_idx < len(self._group_names):
                group = self._group_names[g_idx]
                sort_col = self._count_cols[group]
            else:
                sort_col = list(self._count_cols.values())[0]

        # ── Collect top-N items per group (union) ──
        all_top_items = set()
        for group in self._group_names:
            col = self._count_cols[group]
            numeric = pd.to_numeric(self._df[col], errors='coerce').fillna(0)
            top_idx = numeric.nlargest(top_n).index
            top_items = self._df.loc[top_idx, self._item_col].tolist()
            all_top_items.update(top_items)

        # ── Filter to top items and sort ──
        mask = self._df[self._item_col].isin(all_top_items)
        plot_df = self._df[mask].copy()

        sort_vals = pd.to_numeric(plot_df[sort_col], errors='coerce').fillna(0)
        plot_df = plot_df.iloc[sort_vals.argsort()[::-1]]  # descending (top items first)

        items = plot_df[self._item_col].tolist()
        groups = self._group_names

        # ── Build values / percentages dicts ──
        values = {}
        percentages = {}
        for g in groups:
            col = self._count_cols[g]
            values[g] = pd.to_numeric(
                plot_df[col], errors='coerce'
            ).fillna(0).tolist()

            pct_col = self._pct_cols.get(g)
            if pct_col and pct_col in plot_df.columns:
                percentages[g] = pd.to_numeric(
                    plot_df[pct_col], errors='coerce'
                ).fillna(0).tolist()

        # ── Render chart ──
        self.chart.set_data(
            items=items,
            groups=groups,
            values=values,
            percentages=percentages if percentages else None,
        )

        # ── Title ──
        entity_label = self._item_col or "Items"
        self.chart.setTitle(
            f"Top {entity_label}s by Group (N={len(items)})"
        )

        # ── Legend ──
        self._update_legend()

        # ── Info ──
        self.Information.showing(len(items), len(groups))
        self.status_label.setText(
            f"{len(groups)} groups, {len(items)} items shown "
            f"(from {len(self._df)} total)"
        )

        # ── Clean temp column ──
        if "_combined_sum" in self._df.columns:
            self._df.drop(columns=["_combined_sum"], inplace=True)

    def _update_legend(self):
        """Update legend in the control area."""
        while self.legend_layout.count():
            child = self.legend_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not HAS_PYQTGRAPH or not hasattr(self, 'chart'):
            return

        for g in self._group_names:
            color = self.chart._group_colors.get(g, GROUP_PALETTE[0])

            row = QHBoxLayout()
            swatch = QLabel("■")
            swatch.setStyleSheet(f"color: {color}; font-size: 18px;")
            swatch.setFixedWidth(22)
            row.addWidget(swatch)

            name_lbl = QLabel(g)
            name_lbl.setStyleSheet("font-size: 12px;")
            row.addWidget(name_lbl)
            row.addStretch()

            container = QWidget()
            container.setLayout(row)
            self.legend_layout.addWidget(container)

    def _clear_chart(self):
        """Clear the chart."""
        if HAS_PYQTGRAPH and hasattr(self, 'chart'):
            self.chart.clear()
            self.chart._bars = []
            self.chart._bar_items = []
            self.chart._selection = []

    def _clear_selection(self):
        """Clear chart selection."""
        if HAS_PYQTGRAPH and hasattr(self, 'chart'):
            self.chart.clear_selection()
            self.Outputs.selected_items.send(None)

    def _export_png(self):
        """Export chart to PNG."""
        if not HAS_PYQTGRAPH or not hasattr(self, 'chart'):
            return

        fname, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "",
            "PNG Files (*.png);;All Files (*)"
        )
        if fname:
            if not fname.endswith('.png'):
                fname += '.png'
            try:
                import pyqtgraph.exporters
                exporter = pg.exporters.ImageExporter(
                    self.chart.getPlotItem()
                )
                exporter.parameters()['width'] = 1600
                exporter.export(fname)
                logger.info(f"Plot exported to {fname}")
            except Exception as e:
                logger.exception(f"PNG export failed: {e}")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _table_to_df(self, table: Table) -> pd.DataFrame:
        """Convert Orange Table to pandas DataFrame."""
        data = {}
        domain = table.domain

        for var in domain.attributes:
            col_data = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)] if not np.isnan(v) else None
                    for v in col_data
                ]
            else:
                data[var.name] = col_data

        for var in domain.class_vars:
            col_data = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)] if not np.isnan(v) else None
                    for v in col_data
                ]
            else:
                data[var.name] = col_data

        for var in domain.metas:
            col_data = table[:, var].metas.flatten()
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)]
                    if not (isinstance(v, float) and np.isnan(v)) else None
                    for v in col_data
                ]
            elif isinstance(var, StringVariable):
                data[var.name] = [
                    str(v) if v is not None and str(v) != "?" else None
                    for v in col_data
                ]
            else:
                data[var.name] = col_data

        return pd.DataFrame(data)

    def _df_to_table(self, df: pd.DataFrame) -> Table:
        """Convert pandas DataFrame to Orange Table."""
        attrs = []
        metas = []
        X_cols = []
        M_cols = []

        for col in df.columns:
            if col.startswith("_"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                attrs.append(ContinuousVariable(str(col)))
                X_cols.append(col)
            else:
                metas.append(StringVariable(str(col)))
                M_cols.append(col)

        domain = Domain(attrs, metas=metas)

        X = np.empty((len(df), len(attrs)), dtype=float)
        for i, col in enumerate(X_cols):
            X[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(np.nan).values

        M = np.empty((len(df), len(metas)), dtype=object)
        for i, col in enumerate(M_cols):
            M[:, i] = df[col].astype(str).values

        return Table.from_numpy(domain, X, metas=M if metas else None)


# =============================================================================
# WIDGET PREVIEW
# =============================================================================

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWGroupCountsPlot).run()
