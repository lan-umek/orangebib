# -*- coding: utf-8 -*-
"""
Bibliographic Coupling Widget
============================
Build and render the paper-level bibliographic coupling network — papers
linked when they share references — using
`biblium.addons.references_analysis.build_paper_bibliographic_coupling`.
Requires OpenAlex-enriched data (oa_referenced_works / oa_openalex_id).
"""

import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import QLabel, QComboBox, QPushButton, QSpinBox, QGridLayout, QProgressBar

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.addons.references_analysis import build_paper_bibliographic_coupling
    import networkx as nx
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    build_paper_bibliographic_coupling = None
    nx = None

logger = logging.getLogger(__name__)

PALETTE = ["#4a90d9", "#e8743b", "#5aa454", "#c0504d", "#8064a2", "#1aa8a8",
           "#d9a441", "#9b59b6", "#16a085", "#e74c3c", "#2c3e50", "#f39c12"]


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


def _df_to_table(df):
    if df is None or df.empty:
        return None
    attrs, metas, ac, mc = [], [], [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
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
        M[:, i] = df[c].astype(object).where(df[c].notna(), "").values
    return Table.from_numpy(domain, X, metas=M)


class CouplingWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, df, refs_col, id_col, min_shared, top_n):
        super().__init__()
        self._df = df; self._refs = refs_col; self._id = id_col
        self._min = min_shared; self._top = top_n

    def run(self):
        try:
            G = build_paper_bibliographic_coupling(
                self._df, refs_col=self._refs, id_col=self._id,
                min_shared=self._min, top_n=self._top)
            self.finished.emit(G, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("coupling failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWBibCoupling(OWWidget):
    """Paper-level bibliographic coupling network."""

    name = "Bibliographic Coupling"
    description = "Network of papers sharing references (needs OpenAlex refs)"
    icon = "icons/bib_coupling.svg"
    priority = 440
    keywords = ["bibliographic coupling", "references", "network", "shared",
                "coupling"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="OpenAlex-enriched bibliographic data")

    class Outputs:
        node_data = Output("Node Data", Table, doc="Papers with degree & community")
        edge_data = Output("Edge Data", Table, doc="Coupling edges (shared references)")

    refs_col = settings.Setting("")
    id_col = settings.Setting("")
    min_shared = settings.Setting(5)
    top_n = settings.Setting(200)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons + networkx required.")
        no_refs = Msg("Need a references column (oa_referenced_works / References)")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        built = Msg("{} papers, {} coupling links")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None
        self._G = None
        self._pos: Dict = {}

        box = gui.widgetBox(self.controlArea, "Coupling")
        grid = QGridLayout()
        grid.addWidget(QLabel("References col:"), 0, 0)
        self.refs_combo = QComboBox()
        self.refs_combo.currentTextChanged.connect(lambda t: setattr(self, "refs_col", t))
        grid.addWidget(self.refs_combo, 0, 1)
        grid.addWidget(QLabel("Paper ID col:"), 1, 0)
        self.id_combo = QComboBox()
        self.id_combo.currentTextChanged.connect(lambda t: setattr(self, "id_col", t))
        grid.addWidget(self.id_combo, 1, 1)
        box.layout().addLayout(grid)
        gui.spin(box, self, "min_shared", 1, 100, label="Min shared refs:", callback=self._rebuild)
        gui.spin(box, self, "top_n", 20, 1000, label="Top N papers:", callback=self._rebuild)

        self.run_btn = QPushButton("Build")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._rebuild)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.graph = pg.PlotWidget(background="w")
        self.graph.hideAxis("bottom"); self.graph.hideAxis("left")
        self.graph.setAspectLocked(True)
        self._scatter = pg.ScatterPlotItem(hoverable=True)
        self._tip = pg.TextItem(color="k", fill=pg.mkBrush(255, 255, 220, 230), anchor=(0, 1))
        self._tip.setZValue(100); self._tip.hide()
        self.graph.scene().sigMouseMoved.connect(self._hover)
        self.mainArea.layout().addWidget(self.graph)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        cols = list(self._df.columns) if self._df is not None else []
        self._fill(self.refs_combo, ["oa_referenced_works", "referenced_works", "References", "Cited References", "CR"], cols, self.refs_col)
        self._fill(self.id_combo, ["oa_openalex_id", "OpenAlex ID", "DOI", "unique-id"], cols, self.id_col)
        if data is None:
            self.Error.no_data()

    @staticmethod
    def _fill(combo, prefer, cols, current):
        ordered = [c for c in prefer if c in cols] + [c for c in cols if c not in prefer]
        combo.blockSignals(True); combo.clear(); combo.addItems(ordered)
        if current in ordered:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _rebuild(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        refs = self.refs_combo.currentText()
        if not refs or refs not in self._df.columns:
            self.Error.no_refs(); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Building coupling network...")
        self._worker = CouplingWorker(self._df, refs, self.id_combo.currentText(),
                                      self.min_shared, self.top_n)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, G, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or G is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            return
        if G.number_of_nodes() < 2:
            self.status_label.setText("Too few coupled papers")
            self.graph.clear()
            return
        self._G = G
        try:
            pos = nx.spring_layout(G, weight="weight", seed=42,
                                   k=1.3 / (G.number_of_nodes() ** 0.5))
        except Exception:  # noqa: BLE001
            pos = nx.circular_layout(G)
        self._pos = {n: (float(p[0]), float(p[1])) for n, p in pos.items()}
        comm = {n: 0 for n in G.nodes()}
        try:
            from networkx.algorithms.community import louvain_communities
            for cid, c in enumerate(louvain_communities(G, weight="weight", seed=42)):
                for n in c:
                    comm[n] = cid
        except Exception:  # noqa: BLE001
            pass
        self._comm = comm
        self._render()
        self.status_label.setText(f"Done — {G.number_of_nodes()} papers")
        self.Information.built(G.number_of_nodes(), G.number_of_edges())
        self._send_outputs()

    def _render(self):
        G = self._G
        self.graph.clear()
        xs, ys = [], []
        for u, v in G.edges():
            x0, y0 = self._pos[u]; x1, y1 = self._pos[v]
            xs.extend([x0, x1, np.nan]); ys.extend([y0, y1, np.nan])
        if xs:
            self.graph.addItem(pg.PlotCurveItem(
                x=np.array(xs), y=np.array(ys),
                pen=pg.mkPen((170, 170, 170, 100), width=1), connect="finite"))
        self._node_ids = list(G.nodes())
        deg = dict(G.degree(weight="weight"))
        dmax = max(deg.values()) if deg else 1
        spots = []
        for i, n in enumerate(self._node_ids):
            spots.append({"pos": self._pos[n], "data": i,
                          "size": 7 + 22 * (deg[n] / dmax if dmax else 0),
                          "brush": pg.mkBrush(PALETTE[self._comm[n] % len(PALETTE)]),
                          "pen": pg.mkPen("w", width=0.5)})
        self._scatter.setData(spots)
        self.graph.addItem(self._scatter)
        self.graph.addItem(self._tip)
        self.graph.getViewBox().autoRange()

    def _hover(self, p):
        if self._G is None:
            return
        vb = self.graph.getPlotItem().vb
        if not self.graph.sceneBoundingRect().contains(p):
            self._tip.hide(); return
        pts = self._scatter.pointsAt(vb.mapSceneToView(p))
        if len(pts):
            n = self._node_ids[pts[0].data()]
            title = self._G.nodes[n].get("title", str(n))
            mp = vb.mapSceneToView(p)
            self._tip.setText(str(title)[:80]); self._tip.setPos(mp.x(), mp.y()); self._tip.show()
        else:
            self._tip.hide()

    def _send_outputs(self):
        G = self._G
        deg = dict(G.degree(weight="weight"))
        nd = pd.DataFrame([{
            "Paper": self._G.nodes[n].get("title", str(n)),
            "Citations": self._G.nodes[n].get("citations", 0),
            "CouplingDegree": deg[n], "Community": self._comm[n],
        } for n in G.nodes()])
        ed = pd.DataFrame([{
            "Source": str(self._G.nodes[u].get("title", u))[:60],
            "Target": str(self._G.nodes[v].get("title", v))[:60],
            "SharedRefs": d.get("weight", 1),
        } for u, v, d in G.edges(data=True)])
        self.Outputs.node_data.send(_df_to_table(nd))
        self.Outputs.edge_data.send(_df_to_table(ed))

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWBibCoupling).run()
