# -*- coding: utf-8 -*-
"""
Co-citation Network Widget
=========================
Build the co-citation network: cited references become nodes, linked when they
are cited together by the same papers. Reveals the intellectual base / schools
of thought of a field. Self-computed from the references column.
"""

import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional, List, Dict

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
    import networkx as nx
    HAS_NX = True
except Exception:  # noqa: BLE001
    HAS_NX = False
    nx = None

logger = logging.getLogger(__name__)
REF_CANDIDATES = ["References", "Cited References", "CR", "oa_referenced_works", "referenced_works"]
PALETTE = ["#4a90d9", "#e8743b", "#5aa454", "#c0504d", "#8064a2", "#1aa8a8",
           "#d9a441", "#9b59b6", "#16a085", "#e74c3c", "#2c3e50", "#f39c12"]


def _table_to_df(table):
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


def _split_refs(val):
    s = str(val)
    for sep in ["||", "|", "; ", ";"]:
        if sep in s:
            return [x.strip().replace("https://openalex.org/", "") for x in s.split(sep) if x.strip()]
    return [s.strip()] if s.strip() else []


class CoCiteWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, df, refs_col, top_n, min_cocit):
        super().__init__()
        self._df = df; self._refs = refs_col; self._top = top_n; self._min = min_cocit

    def run(self):
        try:
            cite_count = defaultdict(int)
            paper_refs = []
            for val in self._df[self._refs]:
                if pd.isna(val):
                    paper_refs.append([]); continue
                refs = list(dict.fromkeys(_split_refs(val)))
                paper_refs.append(refs)
                for r in refs:
                    cite_count[r] += 1
            top = [r for r, _ in sorted(cite_count.items(), key=lambda kv: -kv[1])[:self._top]]
            idx = {r: i for i, r in enumerate(top)}
            cocit = defaultdict(int)
            for refs in paper_refs:
                present = [r for r in refs if r in idx]
                for a, b in combinations(sorted(set(present)), 2):
                    cocit[(idx[a], idx[b])] += 1
            edges = [(i, j, w) for (i, j), w in cocit.items() if w >= self._min]
            self.finished.emit((top, [cite_count[r] for r in top], edges), "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("co-citation failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWCoCitation(OWWidget):
    """Co-citation network of references."""

    name = "Co-citation Network"
    description = "Network of references cited together (intellectual base)"
    icon = "icons/cocitation.svg"
    priority = 430
    keywords = ["co-citation", "cocitation", "references", "network",
                "intellectual base", "schools"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with references")

    class Outputs:
        node_data = Output("Node Data", Table, doc="References with citation count & community")

    refs_col = settings.Setting("")
    top_n = settings.Setting(60)
    min_cocitation = settings.Setting(2)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_networkx = Msg("networkx is required")
        no_refs = Msg("Need a references column")
        compute_error = Msg("{}")

    class Information(OWWidget.Information):
        built = Msg("{} references, {} co-citation links")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None
        self._nodes: List[str] = []
        self._pos: Dict = {}

        box = gui.widgetBox(self.controlArea, "Co-citation")
        grid = QGridLayout()
        grid.addWidget(QLabel("References col:"), 0, 0)
        self.refs_combo = QComboBox()
        self.refs_combo.currentTextChanged.connect(lambda t: setattr(self, "refs_col", t))
        grid.addWidget(self.refs_combo, 0, 1)
        box.layout().addLayout(grid)
        gui.spin(box, self, "top_n", 10, 300, label="Top N references:", callback=self._rebuild)
        gui.spin(box, self, "min_cocitation", 1, 50, label="Min co-citation:", callback=self._rebuild)

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

        if not HAS_NX:
            self.Error.no_networkx()
            self.run_btn.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_NX:
            self.Error.no_networkx()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        cols = list(self._df.columns) if self._df is not None else []
        ordered = [c for c in REF_CANDIDATES if c in cols] + [c for c in cols if c not in REF_CANDIDATES]
        self.refs_combo.blockSignals(True); self.refs_combo.clear(); self.refs_combo.addItems(ordered)
        if self.refs_col in ordered:
            self.refs_combo.setCurrentText(self.refs_col)
        self.refs_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _rebuild(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_NX:
            self.Error.no_networkx(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        refs = self.refs_combo.currentText()
        if not refs or refs not in self._df.columns:
            self.Error.no_refs(); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Computing co-citations...")
        self._worker = CoCiteWorker(self._df, refs, self.top_n, self.min_cocitation)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, result, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or result is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            return
        top, counts, edges = result
        if len(top) < 2 or not edges:
            self.status_label.setText("Too few co-cited references")
            self.graph.clear(); return
        self._nodes = top; self._counts = counts; self._edges = edges
        G = nx.Graph()
        G.add_nodes_from(range(len(top)))
        for i, j, w in edges:
            G.add_edge(i, j, weight=w)
        try:
            pos = nx.spring_layout(G, weight="weight", seed=42, k=1.3 / (len(top) ** 0.5))
        except Exception:  # noqa: BLE001
            pos = nx.circular_layout(G)
        self._pos = {i: (float(p[0]), float(p[1])) for i, p in pos.items()}
        self._comm = [0] * len(top)
        try:
            from networkx.algorithms.community import louvain_communities
            for cid, c in enumerate(louvain_communities(G, weight="weight", seed=42)):
                for nidx in c:
                    self._comm[nidx] = cid
        except Exception:  # noqa: BLE001
            pass
        self._render()
        self.status_label.setText(f"Done — {len(top)} references")
        self.Information.built(len(top), len(edges))
        nd = pd.DataFrame({"Reference": top, "Citations": counts,
                           "Community": self._comm})
        self.Outputs.node_data.send(_df_to_table(nd))

    def _render(self):
        self.graph.clear()
        xs, ys = [], []
        for i, j, w in self._edges:
            x0, y0 = self._pos[i]; x1, y1 = self._pos[j]
            xs.extend([x0, x1, np.nan]); ys.extend([y0, y1, np.nan])
        if xs:
            self.graph.addItem(pg.PlotCurveItem(x=np.array(xs), y=np.array(ys),
                                                pen=pg.mkPen((170, 170, 170, 100), width=1), connect="finite"))
        cmax = max(self._counts) if self._counts else 1
        spots = [{"pos": self._pos[i], "data": i,
                  "size": 7 + 22 * (self._counts[i] / cmax if cmax else 0),
                  "brush": pg.mkBrush(PALETTE[self._comm[i] % len(PALETTE)]),
                  "pen": pg.mkPen("w", width=0.5)} for i in range(len(self._nodes))]
        self._scatter.setData(spots)
        self.graph.addItem(self._scatter)
        self.graph.addItem(self._tip)
        self.graph.getViewBox().autoRange()

    def _hover(self, p):
        if not self._nodes:
            return
        vb = self.graph.getPlotItem().vb
        if not self.graph.sceneBoundingRect().contains(p):
            self._tip.hide(); return
        pts = self._scatter.pointsAt(vb.mapSceneToView(p))
        if len(pts):
            i = pts[0].data()
            mp = vb.mapSceneToView(p)
            self._tip.setText(f"{self._nodes[i][:70]}\ncited by {self._counts[i]}")
            self._tip.setPos(mp.x(), mp.y()); self._tip.show()
        else:
            self._tip.hide()

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWCoCitation).run()
