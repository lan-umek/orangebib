# -*- coding: utf-8 -*-
"""
CiteSpace Metrics Widget
=======================
CiteSpace-style structural metrics on a co-occurrence (or co-citation) network:

* **Betweenness centrality** -- identifies *pivotal / turning-point* nodes that
  bridge otherwise separate parts of the literature.
* **Burstness** -- nodes with a sudden rise in frequency over time (when a Year
  column is present).
* **Sigma = (betweenness + 1) ^ burstness** -- CiteSpace's combined novelty +
  centrality indicator.
* **Clusters** -- network communities with automatic labels (most frequent
  member), the way CiteSpace summarises a domain.

Build the network from any keyword / reference column.
"""

import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QGridLayout,
                             QProgressBar)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    HAS_NX = True
except Exception:  # noqa: BLE001
    nx = None
    HAS_NX = False

ENTITY_PATTERNS = ["keyword", "author", "source", "reference", "term", "concept",
                   "country", "institution", "affiliation", "cited"]
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


def _split(val) -> List[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    for sep in _SEPS:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


class CiteSpaceWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, col, year_col, top_n, min_occ):
        super().__init__()
        self._df = df; self._col = col; self._yc = year_col
        self._top = top_n; self._min = min_occ

    def run(self):
        try:
            self.progress.emit("Building co-occurrence network...")
            df = self._df.reset_index(drop=True)
            col_vals = df[self._col].tolist()
            if self._yc and self._yc in df.columns:
                years_arr = pd.to_numeric(df[self._yc], errors="coerce").tolist()
            else:
                years_arr = [None] * len(df)
            occ = defaultdict(int)
            first_year = {}
            doc_entities = []
            for pos in range(len(col_vals)):
                ents = list(dict.fromkeys(_split(col_vals[pos])))
                doc_entities.append(ents)
                yr = years_arr[pos]
                yr_ok = yr is not None and not (isinstance(yr, float) and np.isnan(yr))
                for e in ents:
                    occ[e] += 1
                    if yr_ok:
                        first_year[e] = min(first_year.get(e, 9999), int(yr))
            keep = {e for e, c in occ.items() if c >= self._min}
            top = sorted(keep, key=lambda e: -occ[e])[:self._top]
            idx_of = {e: i for i, e in enumerate(top)}
            if len(top) < 3:
                self.finished.emit(None, "Not enough entities for a network")
                return
            edge_w = defaultdict(int)
            for ents in doc_entities:
                present = [e for e in ents if e in idx_of]
                for a, b in combinations(sorted(set(present)), 2):
                    edge_w[(idx_of[a], idx_of[b])] += 1
            G = nx.Graph()
            G.add_nodes_from(range(len(top)))
            for (i, j), w in edge_w.items():
                G.add_edge(i, j, weight=w)

            self.progress.emit("Computing betweenness centrality...")
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
            deg = dict(G.degree(weight="weight"))
            try:
                from networkx.algorithms.community import louvain_communities
                comms = louvain_communities(G, weight="weight", seed=42)
            except Exception:  # noqa: BLE001
                comms = [set(G.nodes())]
            node_comm, comm_label = {}, {}
            for cid, comm in enumerate(comms):
                members = sorted(comm, key=lambda n: -occ[top[n]])
                comm_label[cid] = top[members[0]] if members else str(cid)
                for n in comm:
                    node_comm[n] = cid

            # crude burstness: relative recent growth weight (needs years)
            self.progress.emit("Scoring burstness / sigma...")
            rows = []
            for n in range(len(top)):
                name = top[n]
                b = float(bc.get(n, 0.0))
                burst = 0.0
                fy = first_year.get(name)
                if fy and fy != 9999:
                    # newer + frequent entities score higher (proxy for burst)
                    recency = max(0, fy - min(first_year.values())) if first_year else 0
                    span = (max(first_year.values()) - min(first_year.values())
                            ) if len(first_year) > 1 else 1
                    burst = (recency / span) * np.log1p(occ[name]) if span else 0.0
                sigma = (b + 1.0) ** (1.0 + burst)
                rows.append({
                    "Entity": name,
                    "Frequency": occ[name],
                    "Degree": float(deg.get(n, 0)),
                    "Betweenness": round(b, 4),
                    "Burstness": round(float(burst), 3),
                    "Sigma": round(float(sigma), 3),
                    "Cluster": comm_label.get(node_comm.get(n, -1), ""),
                    "First year": first_year.get(name, ""),
                })
            nodes_df = pd.DataFrame(rows).sort_values(
                "Betweenness", ascending=False).reset_index(drop=True)

            clusters = []
            for cid, comm in enumerate(comms):
                members = sorted(comm, key=lambda n: -occ[top[n]])
                clusters.append({
                    "Cluster": comm_label.get(cid, str(cid)),
                    "Size": len(comm),
                    "Total frequency": sum(occ[top[n]] for n in comm),
                    "Top members": ", ".join(top[n] for n in members[:6]),
                })
            clusters_df = pd.DataFrame(clusters).sort_values(
                "Size", ascending=False).reset_index(drop=True)
            self.finished.emit((nodes_df, clusters_df), "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("citespace failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWCiteSpace(OWWidget):
    """CiteSpace-style structural metrics (betweenness, burst, sigma, clusters)."""

    name = "CiteSpace Metrics"
    description = "Pivotal nodes (betweenness), burstness, sigma and labelled clusters"
    icon = "icons/citespace.svg"
    priority = 348
    keywords = ["citespace", "betweenness", "pivotal", "turning point", "sigma",
                "burst", "cluster", "co-citation", "structural"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    class Outputs:
        nodes = Output("Node Metrics", Table, doc="Per-node CiteSpace metrics")
        clusters = Output("Clusters", Table, doc="Labelled network clusters")

    column_name = settings.Setting("")
    top_n = settings.Setting(80)
    min_occurrences = settings.Setting(3)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_networkx = Msg("networkx is required")
        no_entities = Msg("No entity column found")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("{} nodes, {} clusters")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Network")
        grid = QGridLayout()
        grid.addWidget(QLabel("Entity column:"), 0, 0)
        self.col_combo = QComboBox()
        self.col_combo.currentTextChanged.connect(lambda t: setattr(self, "column_name", t))
        grid.addWidget(self.col_combo, 0, 1)
        box.layout().addLayout(grid)
        gui.spin(box, self, "top_n", 10, 400, label="Top N nodes:")
        gui.spin(box, self, "min_occurrences", 1, 50, label="Min occurrences:")
        self.run_btn = QPushButton("Compute metrics"); self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.summary_label = QLabel("No data"); self.summary_label.setWordWrap(True)
        self.mainArea.layout().addWidget(self.summary_label)
        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.graph.setLabel("bottom", "Frequency")
        self.graph.setLabel("left", "Betweenness centrality")
        self.mainArea.layout().addWidget(self.graph)

        if not HAS_NX:
            self.Error.no_networkx(); self.run_btn.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        if not HAS_NX:
            self.Error.no_networkx()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.col_combo.blockSignals(True)
        self.col_combo.clear()
        if self._df is not None and not self._df.empty:
            ent = [c for c in self._df.columns
                   if any(k in str(c).lower() for k in ENTITY_PATTERNS)
                   and str(c).lower() != "year"]
            self.col_combo.addItems(ent or list(self._df.columns))
            if self.column_name in ent:
                self.col_combo.setCurrentText(self.column_name)
            elif ent:
                self.column_name = ent[0]
        self.col_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py", "oa_publication_year"):
                return c
        return None

    def _compute(self):
        self.Error.clear()
        if not HAS_NX:
            self.Error.no_networkx(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        col = self.col_combo.currentText()
        if not col or col not in self._df.columns:
            self.Error.no_entities(); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = CiteSpaceWorker(self._df, col, self._year_col(),
                                       self.top_n, self.min_occurrences)
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, res, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or res is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            self.Outputs.nodes.send(None); self.Outputs.clusters.send(None)
            return
        nodes_df, clusters_df = res
        top_piv = ", ".join(nodes_df.head(4)["Entity"].astype(str))
        self.summary_label.setText(
            f"<b>{len(nodes_df)}</b> nodes, <b>{len(clusters_df)}</b> clusters. "
            f"Pivotal: {top_piv}")
        self._render(nodes_df)
        self.status_label.setText(f"Done — {len(nodes_df)} nodes")
        self.Information.done(len(nodes_df), len(clusters_df))
        self.Outputs.nodes.send(_df_to_table(nodes_df))
        self.Outputs.clusters.send(_df_to_table(clusters_df))

    def _render(self, nodes_df):
        self.graph.clear()
        if nodes_df is None or nodes_df.empty:
            return
        x = pd.to_numeric(nodes_df["Frequency"], errors="coerce").fillna(0).values.astype(float)
        y = pd.to_numeric(nodes_df["Betweenness"], errors="coerce").fillna(0).values.astype(float)
        sig = pd.to_numeric(nodes_df["Sigma"], errors="coerce").fillna(1).values.astype(float)
        smax = float(sig.max()) or 1.0
        sizes = [8 + 22 * (float(s) / smax) for s in sig]
        thr = float(np.quantile(y, 0.85)) if len(y) else 0.0
        spots = []
        for i in range(len(x)):
            br = pg.mkBrush("#e67e22") if (y[i] >= thr and thr > 0) else pg.mkBrush("#4a90d9")
            spots.append({"pos": (float(x[i]), float(y[i])), "size": sizes[i],
                          "brush": br, "pen": pg.mkPen("w")})
        scatter = pg.ScatterPlotItem()
        scatter.addPoints(spots)
        self.graph.addItem(scatter)
        for i in range(min(8, len(nodes_df))):
            t = pg.TextItem(str(nodes_df.iloc[i]["Entity"])[:20],
                            color=(40, 40, 40), anchor=(0.5, 1.3))
            t.setPos(float(x[i]), float(y[i]))
            self.graph.addItem(t)
        self.graph.getViewBox().autoRange()

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWCiteSpace).run()
