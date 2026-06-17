# -*- coding: utf-8 -*-
"""
Field Networks Widget
=====================
Advanced co-occurrence analysis for OpenAlex fields / subfields (or any
multi-valued column): a normalised (Jaccard / association / …) field x field
heatmap, a disparity-filter "backbone" of the strongest links, Louvain
communities and bridging-node centralities. Outputs Node/Edge tables that
plug into the Plot Bibliometric Network widget.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import QRectF
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QGridLayout,
                             QDoubleSpinBox, QTabWidget)
import pyqtgraph as pg
from pyqtgraph import AxisItem

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from biblium.utilsbib_modules.network import (
        build_cooccurrence_matrix, normalize_symmetric_matrix,
        matrix_to_network, louvain_partition, disparity_filter_backbone,
        bridging_centralities)
    HAS_NET = True
except Exception:  # noqa: BLE001
    HAS_NET = False

NORMS = ["jaccard", "association", "inclusion", "salton", "none"]
PREFERRED_COLS = ["oa_fields", "oa_subfields", "oa_domains", "oa_topics",
                  "Author Keywords", "Index Keywords", "Keywords",
                  "Countries", "Affiliations"]
PALETTE = ["#4a90d9", "#e8743b", "#5aa454", "#c0504d", "#8064a2", "#1aa8a8",
           "#d9a441", "#9b59b6", "#16a085", "#e74c3c", "#2c3e50", "#f39c12"]
UNIFORM = "#34618d"


def _cmap_rgba(t2d, name="viridis"):
    try:
        import matplotlib
        try:
            cmap = matplotlib.colormaps[name]
        except Exception:  # noqa: BLE001
            from matplotlib import cm
            cmap = cm.get_cmap(name)
        return (cmap(np.clip(t2d, 0, 1)) * 255).astype(np.uint8)
    except Exception:  # noqa: BLE001
        t = np.clip(t2d, 0, 1)
        out = np.zeros(t.shape + (4,), dtype=np.uint8)
        out[..., 0] = (40 + 200 * t).astype(np.uint8)
        out[..., 1] = (80 * (1 - t)).astype(np.uint8)
        out[..., 2] = (220 - 180 * t).astype(np.uint8)
        out[..., 3] = 255
        return out


def _detect_sep(series):
    try:
        sample = " ".join(series.dropna().astype(str).head(100).tolist())
    except Exception:  # noqa: BLE001
        return "; "
    for c in ["||", "|", "; ", ";", " / ", "/"]:
        if c in sample:
            return c
    return "; "


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in (list(table.domain.attributes) + list(table.domain.class_vars)
                + list(table.domain.metas)):
        try:
            col = table.get_column(var)
        except Exception:  # noqa: BLE001
            continue
        if getattr(var, "is_discrete", False):
            vals = var.values
            data[var.name] = [vals[int(v)] if (v == v and 0 <= int(v) < len(vals))
                              else "" for v in col]
        else:
            data[var.name] = col
    return pd.DataFrame(data)


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    attrs, metas, ac, mc = [], [], [], []
    for c in df.columns:
        if (pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])):
            attrs.append(ContinuousVariable(str(c))); ac.append(c)
        else:
            metas.append(StringVariable(str(c))); mc.append(c)
    domain = Domain(attrs, metas=metas)
    n = len(df)
    X = np.empty((n, len(attrs)))
    for i, c in enumerate(ac):
        X[:, i] = pd.to_numeric(df[c], errors="coerce").values
    M = np.empty((n, len(metas)), dtype=object)
    for i, c in enumerate(mc):
        M[:, i] = [("" if (v is None or (isinstance(v, float) and v != v)) else str(v))
                   for v in df[c]]
    return Table.from_numpy(domain, X, metas=M)


class OWFieldNetworks(OWWidget):
    """Field co-occurrence: Jaccard heatmap, disparity backbone, bridging."""

    name = "Field Networks"
    description = ("Normalised field co-occurrence heatmap, disparity-filter "
                   "backbone, Louvain communities and bridging nodes")
    icon = "icons/field_networks.svg"
    priority = 415
    keywords = ["field", "network", "cooccurrence", "jaccard", "backbone",
                "disparity", "bridging", "openalex"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        node_data = Output("Node Data", Table, doc="Nodes: community, degree, bridging")
        edges = Output("Edge Data", Table, doc="Edges (Source/Target/Weight)")
        bridging = Output("Bridging Nodes", Table, doc="Top bridging nodes")

    column = settings.Setting("")
    min_count = settings.Setting(2)
    norm_method = settings.Setting(0)
    min_weight = settings.Setting(0.0)
    use_backbone = settings.Setting(False)
    alpha = settings.Setting(0.05)
    top_heatmap = settings.Setting(20)
    top_bridging = settings.Setting(20)
    autorun = settings.Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_net = Msg("biblium network module not available")
        failed = Msg("Computation failed: {}")

    class Warning(OWWidget.Warning):
        no_col = Msg("No suitable multi-valued column found")
        empty = Msg("Network is empty with the current settings")

    def __init__(self):
        super().__init__()
        self._df = None
        if not HAS_NET:
            self.Error.no_net()

        box = gui.widgetBox(self.controlArea, "Source")
        g = QGridLayout()
        g.addWidget(QLabel("Column:"), 0, 0)
        self.col_combo = QComboBox()
        self.col_combo.currentTextChanged.connect(lambda t: self._set("column", t))
        g.addWidget(self.col_combo, 0, 1)
        box.layout().addLayout(g)
        gui.spin(box, self, "min_count", 1, 100, label="Min entity count:",
                 callback=self._maybe_run)

        nb = gui.widgetBox(self.controlArea, "Network")
        gui.comboBox(nb, self, "norm_method", items=NORMS, label="Normalisation:",
                     orientation="horizontal", sendSelectedValue=False,
                     callback=self._maybe_run)
        mwrow = QGridLayout()
        mwrow.addWidget(QLabel("Min edge weight:"), 0, 0)
        self.mw_spin = QDoubleSpinBox()
        self.mw_spin.setRange(0.0, 1.0); self.mw_spin.setSingleStep(0.02)
        self.mw_spin.setDecimals(3); self.mw_spin.setValue(self.min_weight)
        self.mw_spin.valueChanged.connect(self._on_mw)
        mwrow.addWidget(self.mw_spin, 0, 1)
        nb.layout().addLayout(mwrow)
        gui.checkBox(nb, self, "use_backbone",
                     "Disparity-filter backbone", callback=self._maybe_run)
        arow = QGridLayout()
        arow.addWidget(QLabel("Backbone alpha:"), 0, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.005, 0.5); self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setDecimals(3); self.alpha_spin.setValue(self.alpha)
        self.alpha_spin.valueChanged.connect(self._on_alpha)
        arow.addWidget(self.alpha_spin, 0, 1)
        nb.layout().addLayout(arow)

        vb = gui.widgetBox(self.controlArea, "Display")
        gui.spin(vb, self, "top_heatmap", 5, 60, label="Heatmap top N:",
                 callback=self._maybe_run)
        gui.spin(vb, self, "top_bridging", 5, 60, label="Bridging top N:",
                 callback=self._maybe_run)

        gui.checkBox(self.controlArea, self, "autorun", "Run automatically")
        gui.button(self.controlArea, self, "Run", callback=self._run)
        self.status = QLabel(""); self.status.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status)
        self.controlArea.layout().addStretch(1)

        self.tabs = QTabWidget()
        self._hb = AxisItem(orientation="bottom")
        self._hl = AxisItem(orientation="left")
        self.heat = pg.PlotWidget(background="w",
                                  axisItems={"bottom": self._hb, "left": self._hl})
        self.heat.getPlotItem().showGrid(x=False, y=False)
        self._heat_img = pg.ImageItem()
        self.heat.addItem(self._heat_img)
        self._heat_xlabels = []
        self.bridge = pg.PlotWidget(background="w")
        self.bridge.getPlotItem().showGrid(x=False, y=False)
        self.tabs.addTab(self.heat, "Heatmap")
        self.tabs.addTab(self.bridge, "Bridging")
        self.mainArea.layout().addWidget(self.tabs)

    # ------------------------------------------------------------- input
    @Inputs.data
    def set_data(self, data):
        self._df = _table_to_df(data) if data is not None else None
        self._fill_columns()
        self._maybe_run()

    def _fill_columns(self):
        self.col_combo.blockSignals(True)
        self.col_combo.clear()
        cols = []
        if self._df is not None:
            present = [c for c in PREFERRED_COLS if c in self._df.columns]
            others = [c for c in self._df.columns
                      if c not in present
                      and not pd.api.types.is_numeric_dtype(self._df[c])]
            cols = present + others
        self.col_combo.addItems(cols)
        if self.column in cols:
            self.col_combo.setCurrentText(self.column)
        elif cols:
            self.column = cols[0]
        self.col_combo.blockSignals(False)

    # ------------------------------------------------------------- helpers
    def _set(self, attr, t):
        if t:
            setattr(self, attr, t)
            self._maybe_run()

    def _on_mw(self, v):
        self.min_weight = float(v); self._maybe_run()

    def _on_alpha(self, v):
        self.alpha = float(v); self._maybe_run()

    def _maybe_run(self):
        if self.autorun:
            self._run()

    # ------------------------------------------------------------- run
    def _run(self):
        self.Error.clear(); self.Warning.clear()
        self._heat_img.clear(); self.bridge.clear()
        self._clear_heat_labels()
        if not HAS_NET:
            self.Error.no_net(); return
        if self._df is None or self._df.empty or not self.column \
                or self.column not in self._df.columns:
            self.Warning.no_col()
            self._send(None, None, None); return
        try:
            sep = _detect_sep(self._df[self.column])
            mat = build_cooccurrence_matrix(self._df, self.column, sep=sep,
                                            min_count=self.min_count)
            if mat is None or mat.empty:
                self.Warning.empty(); self._send(None, None, None); return
            method = NORMS[self.norm_method]
            norm = (mat if method == "none"
                    else normalize_symmetric_matrix(mat, method=method))
            G = matrix_to_network(norm, min_weight=self.min_weight)
            if self.use_backbone and G.number_of_edges() > 0:
                G = disparity_filter_backbone(G, alpha=self.alpha)
            if G.number_of_nodes() == 0:
                self.Warning.empty(); self._send(None, None, None); return
            part = louvain_partition(G)
            brdf_all = bridging_centralities(G, top_n=G.number_of_nodes())
            brdf_top = bridging_centralities(G, top_n=self.top_bridging)

            self._draw_heatmap(norm)
            self._draw_bridging(brdf_top)
            self._send_outputs(G, part, brdf_all, brdf_top)
            self.status.setText(
                f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
                f"{len(set(part.values()))} communities"
                + ("  (backbone)" if self.use_backbone else ""))
        except Exception as exc:  # noqa: BLE001
            logger.exception("field networks failed")
            self.Error.failed(str(exc))
            self._send(None, None, None)

    def _send(self, nd, ed, br):
        self.Outputs.node_data.send(nd)
        self.Outputs.edges.send(ed)
        self.Outputs.bridging.send(br)

    def _send_outputs(self, G, part, brdf_all, brdf_top):
        deg = dict(G.degree())
        bmap = {r["node"]: r for _i, r in brdf_all.iterrows()} \
            if brdf_all is not None and not brdf_all.empty else {}
        rows = []
        for nname in G.nodes():
            b = bmap.get(nname, {})
            rows.append({
                "Node": str(nname),
                "Community": int(part.get(nname, 0)),
                "Degree": int(deg.get(nname, 0)),
                "Betweenness": float(b.get("betweenness", 0.0)),
                "Clustering": float(b.get("clustering", 0.0)),
                "Bridging": float(b.get("bridging_score", 0.0)),
            })
        node_df = pd.DataFrame(rows)
        edges = [{"Source": str(u), "Target": str(v),
                  "Weight": float(d.get("weight", 1.0))}
                 for u, v, d in G.edges(data=True)]
        edge_df = pd.DataFrame(edges) if edges else None
        self._send(_df_to_table(node_df), _df_to_table(edge_df),
                   _df_to_table(brdf_top))

    # ------------------------------------------------------------- plots
    def _draw_heatmap(self, norm):
        ents = list(norm.sum(axis=1).sort_values(ascending=False)
                    .head(self.top_heatmap).index)
        sub = norm.loc[ents, ents].values.astype(float)
        vmax = sub.max() if sub.size else 1.0
        rgba = _cmap_rgba(sub / (vmax or 1.0), "viridis")
        self._heat_img.setImage(rgba, axisOrder="row-major")
        self._heat_img.setRect(QRectF(0, 0, len(ents), len(ents)))
        labels = [str(e)[:28] for e in ents]
        n = len(ents)
        # left axis horizontal; bottom axis vertical text to avoid overlap
        self._hl.setTicks([[(i + 0.5, labels[i]) for i in range(n)]])
        self._hb.setTicks([[]])
        for i in range(n):
            t = pg.TextItem(labels[i], color=(60, 60, 60), anchor=(1.0, 0.5),
                            angle=90)
            t.setPos(i + 0.5, -0.2)
            self.heat.addItem(t)
            self._heat_xlabels.append(t)
        self.heat.getViewBox().setRange(xRange=(0, n), yRange=(0, n), padding=0)

    def _clear_heat_labels(self):
        for it in getattr(self, "_heat_xlabels", []):
            try:
                self.heat.removeItem(it)
            except Exception:  # noqa: BLE001
                pass
        self._heat_xlabels = []

    def _draw_bridging(self, brdf):
        if brdf is None or brdf.empty:
            self.bridge.addItem(pg.TextItem("No bridging nodes", color="k"))
            return
        d = brdf[brdf["bridging_score"].astype(float) > 0].head(self.top_bridging)
        if d.empty:
            self.bridge.addItem(pg.TextItem(
                "No bridging nodes (network has too few connected edges)",
                color="k"))
            return
        d = d.iloc[::-1]
        vals = d["bridging_score"].astype(float).tolist()
        labels = [str(x)[:30] for x in d["node"].tolist()]
        n = len(d)
        ypos = list(range(n))
        self.bridge.addItem(pg.BarGraphItem(
            x0=0, y=ypos, height=0.62, width=vals,
            brush=pg.mkBrush(QColor(UNIFORM)),
            pen=pg.mkPen("k", width=0.4)))
        self.bridge.getAxis("left").setTicks([[(ypos[k], labels[k])
                                               for k in range(n)]])
        self.bridge.setLabel("bottom", "Bridging score")
        self.bridge.setYRange(-0.5, n - 0.5)
        self.bridge.setXRange(0, (max(vals) if vals else 1) * 1.05)


if __name__ == "__main__":
    WidgetPreview(OWFieldNetworks).run()
