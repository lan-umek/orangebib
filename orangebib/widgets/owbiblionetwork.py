# -*- coding: utf-8 -*-
"""
Bibliometric Network Widget
==========================
Render co-occurrence networks (keywords, authors, sources, ...) directly in
Orange with an attractive force-directed layout, curved edges, community
colouring and node sizing by degree. Supports export to Pajek format
(.net / .clu / .vec) for use in Pajek, VOSviewer and similar tools.
"""

import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QPushButton, QSpinBox, QCheckBox, QGridLayout,
    QHBoxLayout, QFileDialog, QTabWidget,
)
from AnyQt.QtCore import QRectF

import pyqtgraph as pg
from AnyQt.QtGui import QFont

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    import networkx as nx
    HAS_NX = True
except Exception:  # noqa: BLE001
    HAS_NX = False

try:
    from scipy.ndimage import gaussian_filter
    HAS_NDIMAGE = True
except Exception:  # noqa: BLE001
    HAS_NDIMAGE = False

logger = logging.getLogger(__name__)

ENTITY_PATTERNS = ("keyword", "author", "source", "journal", "countr",
                   "affiliation", "subject", "field", "institution", "topic",
                   "concept", "sdg", "domain")
PALETTE = ["#4a90d9", "#e8743b", "#5aa454", "#c0504d", "#8064a2", "#1aa8a8",
           "#d9a441", "#9b59b6", "#16a085", "#e74c3c", "#2c3e50", "#f39c12"]
SHAPES = [("Circle", "o"), ("Square", "s"), ("Triangle", "t"),
          ("Diamond", "d"), ("Star", "star"), ("Plus", "+")]


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas):
        try:
            col = table.get_column(var)
        except Exception:  # noqa: BLE001
            continue
        if var.is_discrete:
            vals = var.values
            data[var.name] = [vals[int(v)] if (v == v and 0 <= int(v) < len(vals)) else "" for v in col]
        else:
            data[var.name] = col
    return pd.DataFrame(data)


def _split(val):
    s = str(val)
    for sep in ["||", "|", "; ", ";", ", "]:
        if sep in s:
            return [e.strip() for e in s.split(sep) if e.strip()]
    return [s.strip()] if s.strip() else []


class NetworkGraph(pg.PlotWidget):
    def __init__(self, master):
        super().__init__(background="w")
        self.master = master
        self.hideAxis("bottom"); self.hideAxis("left")
        self.setAspectLocked(True)
        self._scatter = pg.ScatterPlotItem(hoverable=True)
        self._scatter.sigClicked.connect(self._clicked)
        self._labels = []
        self._tip = pg.TextItem(color="k", fill=pg.mkBrush(255, 255, 220, 230), anchor=(0, 1))
        self._tip.setZValue(100); self._tip.hide()
        self._nodes = []
        self._drag_idx = None
        self.scene().sigMouseMoved.connect(self._moved)
        # allow manual node dragging by intercepting the view-box drag
        self._vb = self.getViewBox()
        self._vb.mouseDragEvent = self._node_drag

    def _seg(self, pos, i, j, curved):
        x0, y0 = pos[i]; x1, y1 = pos[j]
        if curved:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            nx_, ny_ = -dy, dx
            norm = (nx_ ** 2 + ny_ ** 2) ** 0.5 or 1
            cx, cy = mx + nx_ / norm * 0.12 * norm, my + ny_ / norm * 0.12 * norm
            ts = np.linspace(0, 1, 14)
            bx = (1 - ts) ** 2 * x0 + 2 * (1 - ts) * ts * cx + ts ** 2 * x1
            by = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * cy + ts ** 2 * y1
            return list(bx), list(by)
        return [x0, x1], [y0, y1]

    def render_graph(self, pos, nodes, edges, sizes, colors, labels, curved, show_labels):
        self.clear()
        self._nodes = nodes
        m = self.master
        ew = float(getattr(m, "edge_width", 1))
        ewmax = float(getattr(m, "edge_width_max", 6))
        by_w = bool(getattr(m, "edge_weight_scale", False))
        weights = [w for (_i, _j, w) in edges]
        wmin = min(weights) if weights else 1.0
        wmax = max(weights) if weights else 1.0

        def width_for(w):
            if not by_w or wmax <= wmin:
                return ew
            t = (w - wmin) / (wmax - wmin)
            return ew + (ewmax - ew) * t

        # group edges into a few width bins (keeps drawing fast)
        bins = defaultdict(lambda: ([], []))
        for (i, j, w) in edges:
            sx, sy = self._seg(pos, i, j, curved)
            wd = round(width_for(w), 1)
            xs_b, ys_b = bins[wd]
            xs_b.extend(sx + [np.nan]); ys_b.extend(sy + [np.nan])
        for wd, (xs_b, ys_b) in bins.items():
            self.addItem(pg.PlotCurveItem(
                x=np.array(xs_b), y=np.array(ys_b),
                pen=pg.mkPen((160, 160, 160, 120), width=wd), connect="finite"))

        sym = SHAPES[m.node_shape][1] if 0 <= getattr(m, "node_shape", 0) < len(SHAPES) else "o"
        spots = [{"pos": pos[i], "data": i, "size": sizes[i], "symbol": sym,
                  "brush": pg.mkBrush(colors[i]), "pen": pg.mkPen("w", width=1)}
                 for i in range(len(nodes))]
        self._scatter.setData(spots)
        self.addItem(self._scatter)
        self._labels = []
        if show_labels:
            fs = int(getattr(m, "label_font_size", 9))
            font = QFont(); font.setPointSize(fs)
            order = sorted(range(len(nodes)), key=lambda i: -sizes[i])
            shown = 0
            for i in order:
                txt = str(labels[i]) if i < len(labels) else ""
                if not txt:
                    continue
                t = pg.TextItem(txt[:24], color=(40, 40, 40), anchor=(0.5, 1.2))
                t.setFont(font)
                t.setPos(pos[i][0], pos[i][1])
                self.addItem(t); self._labels.append(t)
                shown += 1
                if shown >= 80:
                    break
        self.addItem(self._tip)
        self.getViewBox().autoRange()

    def _moved(self, p):
        if not self._nodes:
            return
        vb = self.getPlotItem().vb
        if not self.sceneBoundingRect().contains(p):
            self._tip.hide(); return
        pts = self._scatter.pointsAt(vb.mapSceneToView(p))
        if len(pts):
            i = pts[0].data()
            self._tip.setText(self.master.node_tooltip(i))
            mp = vb.mapSceneToView(p)
            self._tip.setPos(mp.x(), mp.y()); self._tip.show()
        else:
            self._tip.hide()

    def _clicked(self, _s, pts):
        if len(pts):
            self.master.on_node_clicked(pts[0].data())

    def _node_drag(self, ev, axis=None):
        from pyqtgraph import ViewBox
        vb = self._vb
        if ev.isStart():
            p = vb.mapToView(ev.buttonDownPos())
            pts = self._scatter.pointsAt(p)
            self._drag_idx = pts[0].data() if len(pts) else None
        if self._drag_idx is None:
            # no node grabbed -> default pan/zoom behaviour
            return ViewBox.mouseDragEvent(vb, ev, axis=axis)
        ev.accept()
        p = vb.mapToView(ev.pos())
        self.master._pos[self._drag_idx] = (float(p.x()), float(p.y()))
        self.master._redraw()
        if ev.isFinish():
            self._drag_idx = None


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=float)


def _cmap_rgba(t2d, name="jet"):
    """Map a 2-D array of 0..1 floats to an (H,W,4) uint8 RGBA image."""
    try:
        import matplotlib
        try:
            cmap = matplotlib.colormaps[name]
        except Exception:  # noqa: BLE001
            from matplotlib import cm
            cmap = cm.get_cmap(name)
        return (cmap(np.clip(t2d, 0, 1)) * 255).astype(np.uint8)
    except Exception:  # noqa: BLE001 (no matplotlib) – blue->red ramp
        t = np.clip(t2d, 0, 1)
        rgba = np.zeros(t.shape + (4,), dtype=np.uint8)
        rgba[..., 0] = (40 + 200 * t).astype(np.uint8)
        rgba[..., 1] = (60 + 80 * (1 - np.abs(0.5 - t) * 2)).astype(np.uint8)
        rgba[..., 2] = (220 - 180 * t).astype(np.uint8)
        rgba[..., 3] = 255
        return rgba


def _blur(acc, sigma):
    if HAS_NDIMAGE:
        return gaussian_filter(acc, sigma=sigma, mode="constant")
    # simple separable box-blur approximation of a Gaussian
    r = max(1, int(sigma))
    k = np.ones(2 * r + 1) / (2 * r + 1)
    out = acc.copy()
    for _ in range(3):
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
    return out


def compute_density_image(P, w, comm, mode="item", grid=256, bw_frac=0.12):
    """Return (rgba_uint8 [grid,grid,4], extent=(xmin,xmax,ymin,ymax))."""
    xmin, xmax = float(P[:, 0].min()), float(P[:, 0].max())
    ymin, ymax = float(P[:, 1].min()), float(P[:, 1].max())
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0
    xmin -= 0.08 * dx; xmax += 0.08 * dx
    ymin -= 0.08 * dy; ymax += 0.08 * dy
    extent = (xmin, xmax, ymin, ymax)
    cx = np.clip(((P[:, 0] - xmin) / (xmax - xmin) * (grid - 1)).astype(int), 0, grid - 1)
    cy = np.clip(((P[:, 1] - ymin) / (ymax - ymin) * (grid - 1)).astype(int), 0, grid - 1)
    sigma = max(2.0, bw_frac * grid)
    w = np.asarray(w, dtype=float)
    if w.max() <= 0:
        w = np.ones_like(w)

    if mode == "item":
        acc = np.zeros((grid, grid))
        for k in range(len(cx)):
            acc[cy[k], cx[k]] += w[k]
        d = _blur(acc, sigma)
        dmax = d.max() or 1.0
        return _cmap_rgba(d / dmax, "jet"), extent

    # cluster density: white background blended with each cluster's colour
    clusters = sorted(set(int(c) for c in comm))
    stack = np.zeros((len(clusters), grid, grid))
    for ci, c in enumerate(clusters):
        acc = np.zeros((grid, grid))
        for k in range(len(cx)):
            if int(comm[k]) == c:
                acc[cy[k], cx[k]] += w[k]
        stack[ci] = _blur(acc, sigma)
    gmax = stack.max() or 1.0
    winner = np.argmax(stack, axis=0)
    inten = (stack.max(axis=0) / gmax)[..., None]
    col_arr = np.array([_hex_to_rgb(PALETTE[c % len(PALETTE)]) for c in clusters])
    win_col = col_arr[winner]
    rgb = (255 * (1 - inten) + win_col * inten).astype(np.uint8)
    rgba = np.zeros((grid, grid, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    return rgba, extent


class DensityView(pg.PlotWidget):
    """VOSviewer-style density heatmap over the network layout."""

    def __init__(self):
        super().__init__(background="w")
        pi = self.getPlotItem()
        pi.showGrid(x=False, y=False)
        pi.hideAxis("left"); pi.hideAxis("bottom")
        pi.setMenuEnabled(False)
        self._img = pg.ImageItem()
        self.addItem(self._img)
        self._labels = []

    def render(self, pos, weights, comm, mode, bw_frac, show_labels, names,
               top_labels=40, min_pct=0, font_size=9):
        for t in self._labels:
            self.removeItem(t)
        self._labels = []
        n = len(names)
        if not pos or n == 0:
            self._img.clear()
            return
        P = np.array([pos[i] for i in range(n)], dtype=float)
        w = np.asarray(weights, dtype=float)
        rgba, extent = compute_density_image(P, w, comm, mode=mode, bw_frac=bw_frac)
        self._img.setImage(rgba, axisOrder="row-major")
        self._img.setRect(QRectF(extent[0], extent[2],
                                 extent[1] - extent[0], extent[3] - extent[2]))
        if show_labels:
            wmax = w.max() if w.size else 0
            thr = (min_pct / 100.0)
            cand = [i for i in np.argsort(-w)
                    if wmax and (w[i] / wmax) >= thr]
            cand = cand[:top_labels] if cand else list(range(min(n, top_labels)))
            # jitter labels vertically so they don't overlap
            min_dx = (extent[1] - extent[0]) * 0.06
            min_dy = (extent[3] - extent[2]) * 0.035
            placed = []
            for i in cand:
                x, y = float(P[i, 0]), float(P[i, 1])
                yy, up = y, True
                for step in range(1, 24):
                    if not any(abs(x - px) < min_dx and abs(yy - py) < min_dy
                               for px, py in placed):
                        break
                    # alternate up/down with growing offset
                    yy = y + (min_dy * ((step + 1) // 2)) * (1 if up else -1)
                    up = not up
                placed.append((x, yy))
                t = pg.TextItem(str(names[i])[:24], color=(25, 25, 25),
                                anchor=(0.5, 0.5))
                _f = QFont(); _f.setPointSize(int(font_size)); t.setFont(_f)
                t.setPos(x, yy)
                self.addItem(t); self._labels.append(t)
        vb = self.getViewBox()
        vb.setRange(xRange=(extent[0], extent[1]), yRange=(extent[2], extent[3]),
                    padding=0)
        vb.setAspectLocked(True)


class OWBiblioNetwork(OWWidget):
    """Render & export bibliometric co-occurrence networks."""

    name = "Plot Bibliometric Network"
    description = "Plot a bibliometric co-occurrence network: nice layout, largest components, node selection and Pajek export"
    icon = "icons/biblio_network.svg"
    priority = 410
    keywords = ["network", "co-occurrence", "graph", "pajek", "community",
                "collaboration"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")
        edges = Input("Edge Data", Table,
                      doc="Edge table (From/To/Weight) from Citation/Co-occurrence Network")

    class Outputs:
        node_data = Output("Node Data", Table, doc="Nodes with degree & community")
        selected = Output("Selected Documents", Table, doc="Docs for the selected node(s)")
        selected_nodes = Output("Selected Nodes", Table, doc="Stats of the selected node(s)")

    column_name = settings.Setting("")
    top_n = settings.Setting(60)
    min_occurrences = settings.Setting(2)
    min_edge_weight = settings.Setting(2)
    component_mode = settings.Setting(1)  # 0=all, 1=largest, 2=>=k nodes
    layout_index = settings.Setting(0)
    partition_method = settings.Setting(0)  # 0 louvain,1 greedy,2 label-prop,3 components,4 none
    node_size_by = settings.Setting(0)     # 0 degree,1 frequency,2 betweenness,3 uniform
    min_component_size = settings.Setting(3)
    curved = settings.Setting(True)
    show_labels = settings.Setting(True)
    density_bandwidth = settings.Setting(12)
    density_labels = settings.Setting(True)
    node_scale = settings.Setting(100)
    label_min_pct = settings.Setting(0)
    select_whole_cluster = settings.Setting(False)
    label_font_size = settings.Setting(9)
    node_shape = settings.Setting(0)
    edge_width = settings.Setting(1)
    edge_width_max = settings.Setting(6)
    edge_weight_scale = settings.Setting(False)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_entities = Msg("No entity column with data found")
        no_networkx = Msg("networkx is required for layout/communities")

    class Warning(OWWidget.Warning):
        export_failed = Msg("Export failed: {}")

    class Information(OWWidget.Information):
        built = Msg("{} nodes, {} edges")
        exported = Msg("Saved {}")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._nodes: List[str] = []
        self._pos: Dict[int, tuple] = {}
        self._edges = []
        self._community = []
        self._occ = {}
        self._betweenness = []
        self._selected_nodes = set()
        self._degree = []
        self._node_docs: Dict[int, list] = {}

        self._build_controls()
        self.tabs = QTabWidget()
        self.graph = NetworkGraph(self)
        self.tabs.addTab(self.graph, "Network")
        self.item_view = DensityView()
        self.tabs.addTab(self.item_view, "Item density")
        self.cluster_view = DensityView()
        self.tabs.addTab(self.cluster_view, "Cluster density")
        self.tabs.currentChanged.connect(lambda _i: self._render_density())
        self.mainArea.layout().addWidget(self.tabs)
        if not HAS_NX:
            self.Error.no_networkx()

    def _build_controls(self):
        box = gui.widgetBox(self.controlArea, "Network")
        g = QGridLayout()
        g.addWidget(QLabel("Item type:"), 0, 0)
        self.col_combo = QComboBox()
        self.col_combo.currentTextChanged.connect(self._on_col_changed)
        g.addWidget(self.col_combo, 0, 1)
        box.layout().addLayout(g)
        gui.spin(box, self, "top_n", 5, 300, label="Top N nodes:", callback=self._rebuild)
        gui.spin(box, self, "min_occurrences", 1, 100, label="Min occurrences:", callback=self._rebuild)
        gui.spin(box, self, "min_edge_weight", 1, 100, label="Min edge weight:", callback=self._rebuild)
        self.comp_combo = gui.comboBox(
            box, self, "component_mode",
            label="Components:", orientation="horizontal",
            items=["All components", "Largest only", "Size >= k nodes"],
            callback=self._rebuild, sendSelectedValue=False)
        gui.spin(box, self, "min_component_size", 2, 100, label="Min component size (k):",
                 callback=self._rebuild)
        self.layout_combo = gui.comboBox(
            box, self, "layout_index", label="Layout:", orientation="horizontal",
            items=["Spring (force)", "Circular", "Kamada-Kawai", "Shell",
                   "Spectral", "Random"],
            callback=self._relayout, sendSelectedValue=False)
        gui.comboBox(
            box, self, "partition_method", label="Partition:", orientation="horizontal",
            items=["Louvain", "Greedy modularity", "Label propagation",
                   "Connected components", "None"],
            callback=self._repartition, sendSelectedValue=False)
        gui.comboBox(
            box, self, "node_size_by", label="Node size:", orientation="horizontal",
            items=["Weighted degree", "Frequency", "Betweenness", "Uniform"],
            callback=self._redraw, sendSelectedValue=False)

        disp = gui.widgetBox(self.controlArea, "Display")
        gui.checkBox(disp, self, "curved", "Curved edges", callback=self._redraw)
        gui.checkBox(disp, self, "show_labels", "Show labels", callback=self._redraw)
        gui.spin(disp, self, "node_scale", 20, 400, label="Node size (%):",
                 callback=self._redraw)
        gui.spin(disp, self, "label_min_pct", 0, 100,
                 label="Hide labels below size (%):",
                 callback=self._after_label_thresh)
        gui.spin(disp, self, "label_font_size", 5, 24, label="Label font size:",
                 callback=self._redraw)
        self.shape_combo = gui.comboBox(
            disp, self, "node_shape", label="Node shape:", orientation="horizontal",
            items=[nm for nm, _ in SHAPES], callback=self._redraw,
            sendSelectedValue=False)
        gui.spin(disp, self, "edge_width", 1, 12, label="Edge width:",
                 callback=self._redraw)
        gui.checkBox(disp, self, "edge_weight_scale", "Edge thickness by weight",
                     callback=self._redraw)
        gui.spin(disp, self, "edge_width_max", 1, 24, label="Max edge width:",
                 callback=self._redraw)

        cl = gui.widgetBox(self.controlArea, "Clusters")
        from AnyQt.QtWidgets import QComboBox as _QCombo
        row = QHBoxLayout()
        row.addWidget(QLabel("Select:"))
        self.cluster_combo = _QCombo()
        self.cluster_combo.addItem("(choose cluster)", -1)
        self.cluster_combo.activated.connect(self._on_cluster_combo)
        row.addWidget(self.cluster_combo)
        from AnyQt.QtWidgets import QWidget as _QW
        w = _QW(); w.setLayout(row); cl.layout().addWidget(w)
        gui.checkBox(cl, self, "select_whole_cluster",
                     "Click selects whole cluster")

        dens = gui.widgetBox(self.controlArea, "Density (VOSviewer-style)")
        gui.spin(dens, self, "density_bandwidth", 2, 40, label="Bandwidth (%):",
                 callback=self._render_density)
        gui.checkBox(dens, self, "density_labels", "Labels on density",
                     callback=self._render_density)

        exp = gui.widgetBox(self.controlArea, "Export (Pajek)")
        row = QHBoxLayout()
        b1 = QPushButton(".net"); b1.clicked.connect(lambda: self._export("net"))
        b2 = QPushButton(".clu"); b2.clicked.connect(lambda: self._export("clu"))
        b3 = QPushButton(".vec"); b3.clicked.connect(lambda: self._export("vec"))
        for b in (b1, b2, b3):
            row.addWidget(b)
        exp.layout().addLayout(row)
        ball = QPushButton("Save all (.net + .clu + .vec)")
        ball.clicked.connect(lambda: self._export("all"))
        exp.layout().addWidget(ball)
        self.controlArea.layout().addStretch(1)

    # --------------------------------------------------------------- input
    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        if not HAS_NX:
            self.Error.no_networkx()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.col_combo.blockSignals(True)
        self.col_combo.clear()
        if data is None:
            self.col_combo.blockSignals(False)
            self.Error.no_data()
            return
        ent = self._entity_columns()
        if not ent:
            self.col_combo.blockSignals(False)
            self.Error.no_entities()
            return
        self.col_combo.addItems(ent)
        if self.column_name in ent:
            self.col_combo.setCurrentText(self.column_name)
        else:
            self.column_name = ent[0]
        self.col_combo.blockSignals(False)
        self._rebuild()

    @Inputs.edges
    def set_edges(self, table):
        """Draw a network that was computed elsewhere (e.g. Citation Network):
        build the graph directly from a From/To/Weight edge table."""
        self.Error.clear()
        if table is None or len(table) == 0:
            return
        if not HAS_NX:
            self.Error.no_networkx(); return
        # collect From/To from metas/attributes, Weight from a numeric column
        cols = {v.name.lower(): v for v in
                list(table.domain.metas) + list(table.domain.attributes)
                + list(table.domain.class_vars)}
        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None
        src = pick("from", "source", "source id", "node1", "citing")
        dst = pick("to", "target", "target id", "node2", "cited")
        wv = pick("weight", "weights", "value")
        if src is None or dst is None:
            self.Error.no_entities(); return
        s_col = [str(x) for x in table.get_column(src)]
        d_col = [str(x) for x in table.get_column(dst)]
        if wv is not None:
            try:
                w_col = [float(x) for x in table.get_column(wv)]
            except Exception:  # noqa: BLE001
                w_col = [1.0] * len(s_col)
        else:
            w_col = [1.0] * len(s_col)
        names = list(dict.fromkeys([n for n in s_col + d_col if n and n != "nan"]))
        idx_of = {n: i for i, n in enumerate(names)}
        edges = []
        for a, b, w in zip(s_col, d_col, w_col):
            if a in idx_of and b in idx_of and a != b:
                edges.append((idx_of[a], idx_of[b], w))
        if len(names) < 2 or not edges:
            self.Error.no_entities(); return
        self._nodes = names
        self._edges = edges
        self._occ = {i: 0 for i in range(len(names))}
        self._node_docs = defaultdict(list)   # no document linkage from edges
        G = nx.Graph()
        G.add_nodes_from(range(len(names)))
        for i, j, w in edges:
            G.add_edge(i, j, weight=w)
        self._G = G
        pos = self._compute_layout(G, len(names))
        self._pos = {i: (float(p[0]), float(p[1])) for i, p in pos.items()}
        self._degree = [G.degree(i, weight="weight") for i in range(len(names))]
        self._community = self._partition(G, len(names))
        try:
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
            self._betweenness = [float(bc.get(i, 0.0)) for i in range(len(names))]
        except Exception:  # noqa: BLE001
            self._betweenness = [0.0] * len(names)
        self._selected_nodes = set()
        self.Information.built(len(names), len(edges))
        self._redraw()
        self._send_node_data()

    def _entity_columns(self):
        if self._df is None:
            return []
        out = []
        for c in self._df.columns:
            cl = str(c).lower()
            if cl == "year" or cl.startswith("doi"):
                continue
            if any(k in cl for k in ENTITY_PATTERNS):
                out.append(c)
        return out

    # ------------------------------------------------------------- compute
    def _rebuild(self):
        self.Error.clear()
        if self._df is None or not self.column_name or not HAS_NX:
            if not HAS_NX:
                self.Error.no_networkx()
            return
        col = self.column_name
        # entity occurrences & per-doc lists
        occ = defaultdict(int)
        doc_entities = []
        for idx, val in self._df[col].items():
            if pd.isna(val):
                doc_entities.append((idx, []))
                continue
            ents = list(dict.fromkeys(_split(val)))
            doc_entities.append((idx, ents))
            for e in ents:
                occ[e] += 1
        keep = {e for e, c in occ.items() if c >= self.min_occurrences}
        top = sorted(keep, key=lambda e: -occ[e])[:self.top_n]
        node_index = {e: i for i, e in enumerate(top)}
        self._occ = {i: occ[e] for i, e in enumerate(top)}
        self._nodes = top
        if len(top) < 2:
            self.Information.built(len(top), 0)
            self.graph.clear()
            return

        # co-occurrence edges + node-doc map
        edge_w = defaultdict(int)
        self._node_docs = defaultdict(list)
        for idx, ents in doc_entities:
            present = [e for e in ents if e in node_index]
            for e in present:
                self._node_docs[node_index[e]].append(idx)
            for a, b in combinations(sorted(set(present)), 2):
                edge_w[(node_index[a], node_index[b])] += 1
        edges = [(i, j, w) for (i, j), w in edge_w.items()
                 if w >= self.min_edge_weight]
        self._edges = edges

        # graph, layout, communities, degree
        G = nx.Graph()
        G.add_nodes_from(range(len(top)))
        for i, j, w in edges:
            G.add_edge(i, j, weight=w)

        # --- keep only large connected components (#22) ---
        if self.component_mode in (1, 2) and G.number_of_nodes() > 0:
            comps = sorted(nx.connected_components(G), key=len, reverse=True)
            if self.component_mode == 1:        # largest only
                keep_ids = set(comps[0]) if comps else set()
            else:                                # size >= k nodes
                keep_ids = set()
                for c in comps:
                    if len(c) >= self.min_component_size:
                        keep_ids |= c
            if keep_ids and len(keep_ids) < len(top):
                old_ids = sorted(keep_ids)
                remap = {old: new for new, old in enumerate(old_ids)}
                top = [top[o] for o in old_ids]
                node_index = {e: i for i, e in enumerate(top)}
                edges = [(remap[i], remap[j], w) for (i, j, w) in edges
                         if i in remap and j in remap]
                self._node_docs = defaultdict(
                    list, {remap[o]: self._node_docs.get(o, []) for o in old_ids})
                self._nodes = top
                self._edges = edges
                G = nx.Graph()
                G.add_nodes_from(range(len(top)))
                for i, j, w in edges:
                    G.add_edge(i, j, weight=w)
        if len(top) < 2:
            self.Information.built(len(top), len(edges))
            self.graph.clear()
            return
        self._G = G
        # occurrences aligned to the (possibly remapped) node order
        self._occ = {i: occ.get(name, 0) for i, name in enumerate(top)}
        pos = self._compute_layout(G, len(top))
        self._pos = {i: (float(p[0]), float(p[1])) for i, p in pos.items()}

        self._degree = [G.degree(i, weight="weight") for i in range(len(top))]
        self._community = self._partition(G, len(top))
        # betweenness (used for node sizing / tooltip)
        try:
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
            self._betweenness = [float(bc.get(i, 0.0)) for i in range(len(top))]
        except Exception:  # noqa: BLE001
            self._betweenness = [0.0] * len(top)
        self._selected_nodes = set()

        self.Information.built(len(top), len(edges))
        self._redraw()
        self._send_node_data()
        self.Outputs.selected.send(None)
        self.Outputs.selected_nodes.send(None)

    def _partition(self, G, n):
        community = [0] * n
        method = self.partition_method
        try:
            from networkx.algorithms import community as nxcom
            if method == 4:                       # none
                return community
            if method == 1:                       # greedy modularity
                comms = nxcom.greedy_modularity_communities(G, weight="weight")
            elif method == 2:                     # label propagation
                comms = list(nxcom.asyn_lpa_communities(G, weight="weight", seed=42))
            elif method == 3:                     # connected components
                comms = list(nx.connected_components(G))
            else:                                  # louvain (default)
                comms = nxcom.louvain_communities(G, weight="weight", seed=42)
            for cid, comm in enumerate(comms):
                for nidx in comm:
                    community[nidx] = cid
        except Exception:  # noqa: BLE001
            pass
        return community

    def _repartition(self):
        """Recompute communities on the current graph and redraw (works for
        networks built from a data column or from an edge table)."""
        if getattr(self, "_G", None) is None or not self._nodes:
            return
        self._community = self._partition(self._G, len(self._nodes))
        self._redraw()
        self._send_node_data()

    def _compute_layout(self, G, n):
        idx = self.layout_index
        try:
            if idx == 1:
                return nx.circular_layout(G)
            if idx == 2:
                return nx.kamada_kawai_layout(G, weight="weight")
            if idx == 3:
                return nx.shell_layout(G)
            if idx == 4:
                return nx.spectral_layout(G)
            if idx == 5:
                return nx.random_layout(G, seed=42)
            return nx.spring_layout(G, weight="weight", seed=42,
                                    k=1.2 / (n ** 0.5))
        except Exception:  # noqa: BLE001
            try:
                return nx.circular_layout(G)
            except Exception:  # noqa: BLE001
                return {i: (0.0, 0.0) for i in range(n)}

    def _relayout(self):
        if getattr(self, "_G", None) is None or not self._nodes:
            return
        pos = self._compute_layout(self._G, len(self._nodes))
        self._pos = {i: (float(p[0]), float(p[1])) for i, p in pos.items()}
        self._redraw()

    def _size_metric(self):
        n = len(self._nodes)
        if self.node_size_by == 3:
            return [1.0] * n
        if self.node_size_by == 1:
            vals = [float(self._occ.get(i, 0)) for i in range(n)]
        elif self.node_size_by == 2:
            vals = [float(self._betweenness[i]) if i < len(self._betweenness) else 0.0
                    for i in range(n)]
        else:
            vals = [float(d) for d in self._degree]
        # fall back to a varying metric if the chosen one carries no information
        # (e.g. "Frequency" for a network built from a bare edge table)
        if vals and max(vals) - min(vals) <= 0:
            deg = [float(d) for d in self._degree]
            if deg and max(deg) - min(deg) > 0:
                return deg
        return vals

    def _redraw(self):
        if not self._nodes or not self._pos:
            return
        vals = self._size_metric()
        vmax = max(vals) if vals else 1
        scale = self.node_scale / 100.0
        if self.node_size_by == 3:
            sizes = [18 * scale] * len(vals)
        else:
            sizes = [(10 + 30 * (v / vmax if vmax else 0)) * scale for v in vals]
        colors = []
        for i in range(len(self._nodes)):
            if i in self._selected_nodes:
                colors.append("#e67e22")
            else:
                colors.append(PALETTE[self._community[i] % len(PALETTE)])
        # hide labels of nodes smaller than the chosen fraction of the largest
        thr = self.label_min_pct / 100.0
        if vmax > 0:
            labels = [self._nodes[i] if (vals[i] / vmax) >= thr else ""
                      for i in range(len(self._nodes))]
        else:
            # no weight information (e.g. network from an edge table) -> show all
            labels = list(self._nodes)
        self.graph.render_graph(self._pos, self._nodes, self._edges, sizes,
                                colors, labels, self.curved, self.show_labels)
        self._render_density()
        self._update_cluster_combo()

    def _after_label_thresh(self):
        self._redraw()
        self._render_density()

    def _render_density(self):
        if not getattr(self, "item_view", None) or not self._nodes or not self._pos:
            return
        w = self._size_metric()
        bw = self.density_bandwidth / 100.0
        self.item_view.render(self._pos, w, self._community, "item", bw,
                              self.density_labels, self._nodes,
                              min_pct=self.label_min_pct,
                              font_size=self.label_font_size)
        self.cluster_view.render(self._pos, w, self._community, "cluster", bw,
                                 self.density_labels, self._nodes,
                                 min_pct=self.label_min_pct,
                                 font_size=self.label_font_size)

    # ---------------------------------------------------------- interaction
    def node_tooltip(self, i):
        return (f"{self._nodes[i]}\nweighted degree: {self._degree[i]:.0f}\n"
                f"community: {self._community[i]}")

    def on_node_clicked(self, i):
        from AnyQt.QtWidgets import QApplication
        from AnyQt.QtCore import Qt as _Qt
        ctrl = bool(QApplication.keyboardModifiers() & _Qt.ControlModifier)
        if self.select_whole_cluster and i < len(self._community):
            cid = self._community[i]
            target = {j for j in range(len(self._nodes))
                      if j < len(self._community) and self._community[j] == cid}
        else:
            target = {i}
        if ctrl:
            self._selected_nodes.symmetric_difference_update(target)
        else:
            self._selected_nodes = set() if self._selected_nodes == target else set(target)
        self._redraw()
        self._emit_selection()

    def _select_cluster(self, cid):
        self._selected_nodes = {j for j in range(len(self._nodes))
                                if j < len(self._community) and self._community[j] == cid}
        self._redraw()
        self._emit_selection()

    def _on_cluster_combo(self, _idx):
        cid = self.cluster_combo.currentData()
        if cid is None or cid < 0:
            return
        self._select_cluster(int(cid))

    def _update_cluster_combo(self):
        if not getattr(self, "cluster_combo", None):
            return
        from collections import Counter as _C
        sizes = _C(self._community) if self._community else {}
        self.cluster_combo.blockSignals(True)
        self.cluster_combo.clear()
        self.cluster_combo.addItem("(choose cluster)", -1)
        for cid in sorted(sizes):
            self.cluster_combo.addItem(f"Cluster {cid} ({sizes[cid]} items)", int(cid))
        self.cluster_combo.blockSignals(False)

    def _emit_selection(self):
        # documents for the union of selected nodes
        docs = set()
        for ni in self._selected_nodes:
            docs.update(self._node_docs.get(ni, []))
        if self._data is not None and docs:
            idx = sorted(d for d in docs if 0 <= d < len(self._data))
            self.Outputs.selected.send(self._data[idx] if idx else None)
        else:
            self.Outputs.selected.send(None)
        self.Outputs.selected_nodes.send(self._node_data_subset(self._selected_nodes))

    def _node_data_subset(self, idxs):
        idxs = sorted(i for i in idxs if 0 <= i < len(self._nodes))
        if not idxs:
            return None
        rows = [{"Node": self._nodes[i],
                 "WeightedDegree": self._degree[i],
                 "Betweenness": (self._betweenness[i] if i < len(self._betweenness) else 0.0),
                 "Frequency": self._occ.get(i, 0),
                 "Community": self._community[i]} for i in idxs]
        df = pd.DataFrame(rows)
        domain = Domain([ContinuousVariable(c) for c in
                         ("WeightedDegree", "Betweenness", "Frequency", "Community")],
                        metas=[StringVariable("Node")])
        X = df[["WeightedDegree", "Betweenness", "Frequency", "Community"]].values.astype(float)
        M = df[["Node"]].astype(str).values
        return Table.from_numpy(domain, X, metas=M)

    def _on_col_changed(self, t):
        self.column_name = t
        self._rebuild()

    # -------------------------------------------------------------- outputs
    def _send_node_data(self):
        if not self._nodes:
            self.Outputs.node_data.send(None)
            return
        bet = [self._betweenness[i] if i < len(self._betweenness) else 0.0
               for i in range(len(self._nodes))]
        df = pd.DataFrame({
            "Node": self._nodes,
            "WeightedDegree": self._degree,
            "Betweenness": bet,
            "Frequency": [self._occ.get(i, 0) for i in range(len(self._nodes))],
            "Community": self._community,
        })
        cols = ["WeightedDegree", "Betweenness", "Frequency", "Community"]
        domain = Domain([ContinuousVariable(c) for c in cols],
                        metas=[StringVariable("Node")])
        X = df[cols].values.astype(float)
        M = df[["Node"]].astype(str).values
        self.Outputs.node_data.send(Table.from_numpy(domain, X, metas=M))

    # --------------------------------------------------------------- pajek
    def _export(self, kind):
        if not self._nodes:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Pajek", "network",
            "Pajek files (*.net *.clu *.vec);;All files (*)")
        if not path:
            return
        import os
        base, _ext = os.path.splitext(path)
        try:
            kinds = ["net", "clu", "vec"] if kind == "all" else [kind]
            for k in kinds:
                out = base + "." + k
                if k == "net":
                    self._write_net(out)
                elif k == "clu":
                    self._write_clu(out)
                elif k == "vec":
                    self._write_vec(out)
            self.Information.exported(
                os.path.basename(base) + " (" + ", ".join("." + k for k in kinds) + ")")
        except Exception as e:  # noqa: BLE001
            logger.exception("Pajek export failed")
            self.Warning.export_failed(str(e))

    def _norm_positions(self):
        n = len(self._nodes)
        if not self._pos:
            return {i: (0.5, 0.5) for i in range(n)}
        xs = [self._pos[i][0] for i in range(n)]
        ys = [self._pos[i][1] for i in range(n)]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        dx = (xmax - xmin) or 1.0
        dy = (ymax - ymin) or 1.0
        return {i: ((self._pos[i][0] - xmin) / dx, (self._pos[i][1] - ymin) / dy)
                for i in range(n)}

    def _write_net(self, path):
        pos = self._norm_positions()
        n = len(self._nodes)
        with open(path, "w", encoding="utf-8") as f:
            f.write("*Vertices %d\n" % n)
            for i in range(n):
                x, y = pos[i]
                label = str(self._nodes[i]).replace('"', "'")
                f.write('%d "%s" %.4f %.4f 0.0000\n' % (i + 1, label, x, y))
            f.write("*Edges\n")
            for (i, j, w) in self._edges:
                f.write("%d %d %.4f\n" % (i + 1, j + 1, float(w)))

    def _write_clu(self, path):
        n = len(self._nodes)
        with open(path, "w", encoding="utf-8") as f:
            f.write("*Vertices %d\n" % n)
            for i in range(n):
                f.write("%d\n" % (int(self._community[i]) + 1))

    def _write_vec(self, path):
        n = len(self._nodes)
        with open(path, "w", encoding="utf-8") as f:
            f.write("*Vertices %d\n" % n)
            for i in range(n):
                val = self._degree[i] if i < len(self._degree) else 0.0
                f.write("%.4f\n" % float(val))


if __name__ == "__main__":
    WidgetPreview(OWBiblioNetwork).run()
