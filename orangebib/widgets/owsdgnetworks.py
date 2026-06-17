# -*- coding: utf-8 -*-
"""
SDG Networks Widget
==================
Co-occurrence network of Sustainable Development Goals (SDGs): per-SDG network
metrics, the normalized co-occurrence matrix (heatmap) and bridge papers that
connect distant SDGs. Wraps `biblium.addons.sdg_networks.analyze_sdg_networks`.

Binary ``SDG N`` indicator columns are auto-detected; if absent they are derived
from a multi-valued SDG column (e.g. ``oa_sdgs``).
"""

import re
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (QLabel, QPushButton, QProgressBar, QTabWidget,
                             QWidget, QVBoxLayout, QApplication)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from biblium.addons.sdg_networks import analyze_sdg_networks, get_sdg_columns
    try:
        from biblium.addons.sdg_networks import SDG_SHORT_NAMES
    except Exception:  # noqa: BLE001
        SDG_SHORT_NAMES = {}
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    analyze_sdg_networks = None
    get_sdg_columns = None
    SDG_SHORT_NAMES = {}
    HAS_BIBLIUM = False

# 5 Ps pillars and 3 sustainability dimensions for SDGs.
SDG_PILLARS = {1: "People", 2: "People", 3: "People", 4: "People", 5: "People",
               6: "Planet", 12: "Planet", 13: "Planet", 14: "Planet", 15: "Planet",
               7: "Prosperity", 8: "Prosperity", 9: "Prosperity",
               10: "Prosperity", 11: "Prosperity",
               16: "Peace", 17: "Partnership"}
SDG_DIMENSIONS = {1: "Social", 2: "Social", 3: "Social", 4: "Social", 5: "Social",
                  6: "Environmental", 13: "Environmental", 14: "Environmental",
                  15: "Environmental", 11: "Environmental",
                  7: "Economic", 8: "Economic", 9: "Economic", 10: "Economic",
                  12: "Economic", 16: "Governance", 17: "Governance"}
SDG_FULL_NAMES = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health and Well-being",
    4: "Quality Education", 5: "Gender Equality",
    6: "Clean Water and Sanitation", 7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure", 10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production", 13: "Climate Action",
    14: "Life Below Water", 15: "Life on Land",
    16: "Peace, Justice and Strong Institutions",
    17: "Partnerships for the Goals",
}
LABEL_MODES = ["Number", "Name", "Number + name", "Pillar (5 Ps)", "Dimension"]


def sdg_label(num, mode):
    try:
        n = int(num)
    except (ValueError, TypeError):
        return str(num)
    name = SDG_FULL_NAMES.get(n, SDG_SHORT_NAMES.get(n, ""))
    if mode == 1:
        return name or f"SDG {n}"
    if mode == 2:
        return f"SDG {n}: {name}" if name else f"SDG {n}"
    if mode == 3:
        return f"SDG {n} · {SDG_PILLARS.get(n, '?')}"
    if mode == 4:
        return f"SDG {n} · {SDG_DIMENSIONS.get(n, '?')}"
    return f"SDG {n}"

SDG_MULTI_CANDIDATES = ["oa_sdgs", "SDGs", "SDG", "sdgs", "sdg",
                        "Sustainable Development Goals"]
_SEPS = ["||", "|", "; ", ";", ", "]


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas):
        data[var.name] = table.get_column(var)
    return pd.DataFrame(data)


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    attrs, metas, X, M = [], [], [], []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.6:
            attrs.append(ContinuousVariable(str(c))); X.append(s.fillna(0).values)
        else:
            metas.append(StringVariable(str(c))); M.append(df[c].astype(str).values)
    n = len(df)
    Xarr = np.column_stack(X) if X else np.empty((n, 0))
    Marr = np.column_stack(M) if M else np.empty((n, 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), Xarr, metas=Marr)


def _split_multi(val) -> List[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    for sep in _SEPS:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def ensure_sdg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return df guaranteed to have binary 'SDG N' columns."""
    existing = []
    if get_sdg_columns is not None:
        try:
            existing = get_sdg_columns(df)
        except Exception:  # noqa: BLE001
            existing = []
    if existing:
        return df
    multi = next((c for c in SDG_MULTI_CANDIDATES if c in df.columns), None)
    if multi is None:
        return df
    out = df.copy()
    rows_sdgs = []
    found = set()
    for v in out[multi]:
        nums = set()
        for tok in _split_multi(v):
            m = re.search(r'(\d{1,2})', tok)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 17:
                    nums.add(n)
        rows_sdgs.append(nums)
        found |= nums
    for n in sorted(found):
        out[f"SDG {n}"] = [1 if n in s else 0 for s in rows_sdgs]
    return out


class SDGNetWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, year_col, min_cooc):
        super().__init__()
        self._df = df; self._yc = year_col; self._min = min_cooc

    def run(self):
        try:
            self.progress.emit("Building SDG network...")
            df = ensure_sdg_columns(self._df)
            analysis = analyze_sdg_networks(
                df, year_col=self._yc, min_cooccurrence=self._min,
                analyze_temporal=False, verbose=False)
            res = {
                "metrics": analysis.get_metrics_df(),
                "connections": analysis.get_top_connections(50),
                "bridges": analysis.get_bridge_papers_df(),
                "norm": analysis.normalized_cooccurrence,
            }
            self.finished.emit(res, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("sdg networks failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWSDGNetworks(OWWidget):
    """SDG co-occurrence network analysis."""

    name = "SDG Networks"
    description = "Co-occurrence network of Sustainable Development Goals + bridge papers"
    icon = "icons/sdg_networks.svg"
    priority = 530
    keywords = ["sdg", "sustainable development goals", "network", "co-occurrence",
                "bridge", "interdisciplinary"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with SDG indicators")

    class Outputs:
        metrics = Output("SDG Metrics", Table, doc="Per-SDG network metrics")
        connections = Output("Connections", Table, doc="Top SDG co-occurrences")
        bridges = Output("Bridge Papers", Table, doc="Papers connecting distant SDGs")
        selected = Output("Selected SDGs", Table, doc="Metric rows for selected SDGs")

    min_cooccurrence = settings.Setting(1)
    label_mode = settings.Setting(2)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        no_sdg = Msg("No SDG columns found (need 'SDG N' columns or an SDG list column)")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("{} SDGs in the network")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Options")
        gui.spin(box, self, "min_cooccurrence", 1, 20, label="Min co-occurrence:")
        gui.comboBox(box, self, "label_mode", label="SDG labels:",
                     orientation="horizontal", items=LABEL_MODES,
                     callback=self._relabel, sendSelectedValue=False)
        self.run_btn = QPushButton("Build network"); self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.summary_label = QLabel("No data"); self.summary_label.setWordWrap(True)
        self.mainArea.layout().addWidget(self.summary_label)
        self.view_tabs = QTabWidget()
        self.metrics_plot = pg.PlotWidget(background="w")
        self.metrics_plot.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.metrics_plot.setLabel("bottom", "Weighted degree")
        self.metrics_plot.scene().sigMouseClicked.connect(self._on_metric_clicked)
        self.view_tabs.addTab(self.metrics_plot, "SDG metrics")
        self._metrics_df = None
        self._selected_sdgs = set()
        hm_tab = QWidget(); hm_l = QVBoxLayout(hm_tab)
        self.heatmap = pg.PlotWidget(background="w")
        self.heat_img = pg.ImageItem(); self.heatmap.addItem(self.heat_img)
        self.heatmap.scene().sigMouseClicked.connect(self._on_heat_clicked)
        self._norm = None
        self._heat_sdgs = []
        self._heat_selected = set()
        hm_l.addWidget(self.heatmap)
        self.view_tabs.addTab(hm_tab, "Co-occurrence")
        self.mainArea.layout().addWidget(self.view_tabs)

        if not HAS_BIBLIUM:
            self.Error.no_biblium(); self.run_btn.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        if data is None:
            self.Error.no_data()

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py", "oa_publication_year"):
                return c
        return "Year"

    def _compute(self):
        self.Error.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        check = ensure_sdg_columns(self._df)
        if get_sdg_columns is not None and not get_sdg_columns(check):
            self.Error.no_sdg(); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = SDGNetWorker(self._df, self._year_col(), self.min_cooccurrence)
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, res, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or res is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            for o in (self.Outputs.metrics, self.Outputs.connections, self.Outputs.bridges):
                o.send(None)
            return
        metrics = res["metrics"]; norm = res["norm"]
        n = len(metrics) if metrics is not None else 0
        self.summary_label.setText(f"<b>{n}</b> SDGs in the co-occurrence network.")
        self._selected_sdgs = set()
        self._heat_selected = set()
        self._render_metrics(metrics)
        self._render_heatmap(norm)
        self.Outputs.selected.send(None)
        self.status_label.setText(f"Done — {n} SDGs")
        self.Information.done(n)
        self.Outputs.metrics.send(_df_to_table(metrics))
        self.Outputs.connections.send(_df_to_table(res["connections"]))
        self.Outputs.bridges.send(_df_to_table(res["bridges"]))

    def _render_metrics(self, metrics):
        self.metrics_plot.clear()
        if metrics is None or metrics.empty:
            return
        wcol = next((c for c in ("Weighted Degree", "Weighted degree", "Degree")
                     if c in metrics.columns), None)
        if wcol is None:
            return
        m = metrics.sort_values(wcol, ascending=False).head(17).reset_index(drop=True)
        self._metrics_df = m
        ys = list(range(len(m)))
        self._metric_sdgs = [m.iloc[i]["SDG"] if "SDG" in m.columns else m.iloc[i].get("Name", i)
                             for i in ys]
        brushes = [pg.mkBrush("#e67e22") if self._metric_sdgs[i] in self._selected_sdgs
                   else pg.mkBrush("#2e8b57") for i in ys]
        self.metrics_plot.addItem(pg.BarGraphItem(
            x0=0, y=ys, height=0.6,
            width=list(pd.to_numeric(m[wcol], errors="coerce").fillna(0)),
            brushes=brushes))
        self.metrics_plot.getAxis("left").setTicks(
            [[(i, sdg_label(self._metric_sdgs[i], self.label_mode)) for i in ys]])
        self.metrics_plot.setYRange(-1, len(m))
        self.metrics_plot.getViewBox().invertY(True)  # largest on top

    def _relabel(self):
        if self._metrics_df is not None:
            self._render_metrics(self._metrics_df)
        if self._norm is not None:
            self._render_heatmap(self._norm)

    def _on_metric_clicked(self, ev):
        if self._metrics_df is None or not hasattr(self, "_metric_sdgs"):
            return
        vb = self.metrics_plot.getPlotItem().vb
        p = vb.mapSceneToView(ev.scenePos())
        i = int(round(p.y()))
        if not (0 <= i < len(self._metric_sdgs)):
            return
        sdg = self._metric_sdgs[i]
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl:
            self._selected_sdgs.symmetric_difference_update({sdg})
        else:
            self._selected_sdgs = set() if self._selected_sdgs == {sdg} else {sdg}
        self._render_metrics(self._metrics_df)
        if self._selected_sdgs:
            sel = self._metrics_df[self._metrics_df["SDG"].isin(self._selected_sdgs)] \
                if "SDG" in self._metrics_df.columns else None
            self.Outputs.selected.send(_df_to_table(sel))
        else:
            self.Outputs.selected.send(None)

    def _render_heatmap(self, norm):
        self.heatmap.clear()
        self.heat_img = pg.ImageItem(); self.heatmap.addItem(self.heat_img)
        if norm is None or getattr(norm, "empty", True) or norm.shape[0] < 2:
            return
        self._norm = norm
        self._heat_sdgs = list(norm.index)
        arr = norm.to_numpy(dtype=float)
        self.heat_img.setImage(arr)
        try:
            self.heat_img.setColorMap(pg.colormap.get("viridis"))
        except Exception:  # noqa: BLE001
            pass
        labels = [sdg_label(i, self.label_mode) for i in norm.index]
        n = len(labels)
        # left axis: horizontal labels (there is room)
        self.heatmap.getAxis("left").setTicks([[(i + 0.5, labels[i]) for i in range(n)]])
        # bottom axis: short labels (number) stay horizontal; longer labels
        # (names / pillars / dimensions) are drawn vertically so they don't
        # overlap.
        for it in getattr(self, "_heat_xlabels", []):
            try:
                self.heatmap.removeItem(it)
            except Exception:  # noqa: BLE001
                pass
        self._heat_xlabels = []
        if self.label_mode == 0:
            self.heatmap.getAxis("bottom").setTicks(
                [[(i + 0.5, labels[i]) for i in range(n)]])
        else:
            self.heatmap.getAxis("bottom").setTicks([[]])  # hide default x ticks
            for i in range(n):
                t = pg.TextItem(labels[i], color=(60, 60, 60), anchor=(1.0, 0.5),
                                angle=90)
                t.setPos(i + 0.5, -0.2)
                self.heatmap.addItem(t)
                self._heat_xlabels.append(t)
        self._draw_heat_highlights()

    def _draw_heat_highlights(self):
        for it in getattr(self, "_heat_hl", []):
            try:
                self.heatmap.removeItem(it)
            except Exception:  # noqa: BLE001
                pass
        self._heat_hl = []
        for (r, c) in getattr(self, "_heat_selected", set()):
            xs = [c, c + 1, c + 1, c, c]; ys = [r, r, r + 1, r + 1, r]
            it = pg.PlotCurveItem(x=np.array(xs, dtype=float),
                                  y=np.array(ys, dtype=float),
                                  pen=pg.mkPen("#e67e22", width=3))
            it.setZValue(50)
            self.heatmap.addItem(it)
            self._heat_hl.append(it)

    def _on_heat_clicked(self, ev):
        if not self._heat_sdgs:
            return
        vb = self.heatmap.getPlotItem().vb
        p = vb.mapSceneToView(ev.scenePos())
        c = int(np.floor(p.x())); r = int(np.floor(p.y()))
        n = len(self._heat_sdgs)
        if not (0 <= r < n and 0 <= c < n):
            return
        from AnyQt.QtWidgets import QApplication
        from AnyQt.QtCore import Qt as _Qt
        ctrl = bool(QApplication.keyboardModifiers() & _Qt.ControlModifier)
        cell = (r, c)
        if ctrl:
            self._heat_selected ^= {cell}
        else:
            self._heat_selected = set() if self._heat_selected == {cell} else {cell}
        self._draw_heat_highlights()
        self._output_heat_docs()

    def _output_heat_docs(self):
        if self._df is None or self._data is None or not self._heat_selected:
            self.Outputs.selected.send(None)
            return
        try:
            sdf = ensure_sdg_columns(self._df)
        except Exception:  # noqa: BLE001
            self.Outputs.selected.send(None)
            return
        import numpy as _np
        mask = None
        for (r, c) in self._heat_selected:
            a = self._heat_sdgs[r]; b = self._heat_sdgs[c]
            ca, cb = f"SDG {int(a)}", f"SDG {int(b)}"
            if ca not in sdf.columns or cb not in sdf.columns:
                continue
            m = (pd.to_numeric(sdf[ca], errors="coerce").fillna(0) > 0) & \
                (pd.to_numeric(sdf[cb], errors="coerce").fillna(0) > 0)
            mask = m if mask is None else (mask | m)
        if mask is None:
            self.Outputs.selected.send(None)
            return
        idx = [i for i, v in enumerate(mask.tolist()) if v and i < len(self._data)]
        self.Outputs.selected.send(self._data[idx] if idx else None)

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWSDGNetworks).run()
