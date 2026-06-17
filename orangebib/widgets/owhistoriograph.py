# -*- coding: utf-8 -*-
"""
Historiograph Widget
====================
A chronological citation network of documents (à la HistCite / CiteNet
Explorer): documents are placed along the time axis (publication year) and the
within-corpus citation links are drawn between them, revealing the genealogy of
a research field. Node size encodes the Local Citation Score (how many times a
document is cited *within the dataset*).

The within-corpus citation graph is built with the same matcher as the Citation
Network widget. Includes aesthetic options and Pajek export.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (QLabel, QPushButton, QHBoxLayout,
                             QFileDialog, QApplication)

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:  # noqa: BLE001
    pg = None
    HAS_PG = False

try:
    import networkx as nx
    HAS_NX = True
except Exception:  # noqa: BLE001
    nx = None
    HAS_NX = False

try:
    from orangebib.widgets.owcitationnetwork import (
        build_openalex_citation_network, build_fuzzy_citation_network)
    HAS_BUILDERS = True
except Exception:  # noqa: BLE001
    build_openalex_citation_network = None
    build_fuzzy_citation_network = None
    HAS_BUILDERS = False


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas):
        data[var.name] = table.get_column(var)
    return pd.DataFrame(data)


class OWHistoriograph(OWWidget):
    """Chronological citation network (historiograph)."""

    name = "Historiograph"
    description = "Chronological document citation network (HistCite / CiteNet Explorer style)"
    icon = "icons/historiograph.svg"
    priority = 425
    keywords = ["historiograph", "histcite", "citenet", "citation", "chronological",
                "genealogy", "main path", "lcs"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        node_data = Output("Node Data", Table)
        selected = Output("Selected Documents", Table)

    min_lcs = settings.Setting(1)         # min local citation score to display
    top_n = settings.Setting(40)
    node_size_by = settings.Setting(0)    # 0 LCS, 1 global citations, 2 uniform
    curved_edges = settings.Setting(True)
    show_labels = settings.Setting(True)
    label_field = settings.Setting(0)     # 0 short id, 1 title

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_networkx = Msg("networkx is required")
        no_builders = Msg("Citation Network module unavailable")
        build_failed = Msg("{}")

    class Warning(OWWidget.Warning):
        no_edges = Msg("No within-corpus citation links found")

    class Information(OWWidget.Information):
        built = Msg("{} documents, {} citation links")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._G = None
        self._nodes = []
        self._pos = {}
        self._selected = set()

        box = gui.widgetBox(self.controlArea, "Historiograph")
        gui.spin(box, self, "min_lcs", 0, 100,
                 label="Min local citations:", callback=self._rebuild)
        gui.spin(box, self, "top_n", 5, 300,
                 label="Max documents:", callback=self._rebuild)

        abox = gui.widgetBox(self.controlArea, "Aesthetics")
        gui.comboBox(abox, self, "node_size_by", label="Node size:",
                     orientation="horizontal",
                     items=["Local citations (LCS)", "Global citations", "Uniform"],
                     callback=self._redraw, sendSelectedValue=False)
        gui.comboBox(abox, self, "label_field", label="Labels:",
                     orientation="horizontal", items=["Short ID", "Title"],
                     callback=self._redraw, sendSelectedValue=False)
        gui.checkBox(abox, self, "curved_edges", "Curved edges", callback=self._redraw)
        gui.checkBox(abox, self, "show_labels", "Show labels", callback=self._redraw)

        self.run_btn = QPushButton("Build historiograph")
        self.run_btn.setMinimumHeight(32)
        self.run_btn.clicked.connect(self._rebuild)
        self.controlArea.layout().addWidget(self.run_btn)

        ebox = gui.widgetBox(self.controlArea, "Export (Pajek)")
        row = QHBoxLayout()
        for kind in ("net", "clu", "vec"):
            b = QPushButton("." + kind)
            b.clicked.connect(lambda _=False, k=kind: self._export_pajek(k))
            row.addWidget(b)
        ebox.layout().addLayout(row)

        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        if HAS_PG:
            self.graph = pg.PlotWidget(background="w")
            self.graph.setLabel("bottom", "Publication year")
            self.graph.hideAxis("left")
            self.graph.scene().sigMouseClicked.connect(self._on_clicked)
            self.mainArea.layout().addWidget(self.graph)
        else:
            self.mainArea.layout().addWidget(QLabel("pyqtgraph not available"))

        if not HAS_NX:
            self.Error.no_networkx()
        elif not HAS_BUILDERS:
            self.Error.no_builders()

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        if data is None:
            self.Error.no_data()
            return
        self._rebuild()

    # ------------------------------------------------------------- build
    def _detect(self, *names):
        if self._df is None:
            return None
        low = {str(c).lower(): c for c in self._df.columns}
        for n in names:
            if n in self._df.columns:
                return n
            if n.lower() in low:
                return low[n.lower()]
        return None

    def _build_graph(self):
        df = self._df
        is_oa = any(("openalex" in str(c).lower() or c in
                     ("referenced_works", "oa_referenced_works"))
                    for c in df.columns)
        if is_oa:
            try:
                G, _ = build_openalex_citation_network(
                    df, keep_largest_component=False, verbose=False)
                if G.number_of_edges() > 0:
                    return G
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAlex historiograph build failed: %s", exc)
        title = self._detect("Title", "TI", "title", "display_name")
        ref = self._detect("References", "Cited References", "CR",
                           "oa_referenced_works", "referenced_works")
        idc = self._detect("EID", "DOI", "UT", "id", "unique-id", "Doc ID")
        if not title or not ref:
            return None
        if not idc:
            df = df.copy(); df["_doc_id"] = [f"D{i}" for i in range(len(df))]
            idc = "_doc_id"
        try:
            G, _ = build_fuzzy_citation_network(df, title, ref, idc,
                                                threshold=80, verbose=False)
            return G
        except Exception as exc:  # noqa: BLE001
            logger.exception("fuzzy historiograph build failed")
            self.Error.build_failed(str(exc))
            return None

    def _rebuild(self):
        self.Error.clear(); self.Warning.clear(); self.Information.clear()
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        if not HAS_NX or not HAS_BUILDERS:
            return
        G = self._build_graph()
        if G is None or G.number_of_nodes() == 0:
            self.Error.build_failed("Could not build a citation network "
                                    "(need references + titles/DOIs).")
            self._G = None; self._redraw(); return
        # local citation score = in-degree (cited within the corpus)
        for n in G.nodes():
            G.nodes[n]["lcs"] = G.in_degree(n)
        # filter to documents with enough local citations, keep top-N by LCS
        keep = [n for n in G.nodes() if G.in_degree(n) >= self.min_lcs
                or G.out_degree(n) > 0]
        keep = sorted(keep, key=lambda n: -G.nodes[n]["lcs"])[:self.top_n]
        G = G.subgraph(keep).copy()
        if G.number_of_nodes() == 0:
            self.Error.build_failed("No documents meet the filter.")
            self._G = None; self._redraw(); return
        if G.number_of_edges() == 0:
            self.Warning.no_edges()
        self._G = G
        self._selected = set()
        self._compute_positions()
        self.Information.built(G.number_of_nodes(), G.number_of_edges())
        self.status_label.setText(
            f"{G.number_of_nodes()} documents, {G.number_of_edges()} links")
        self._redraw()
        self._send_node_data()

    def _compute_positions(self):
        G = self._G
        nodes = list(G.nodes())
        self._nodes = nodes
        years = {n: float(G.nodes[n].get("year", 0) or 0) for n in nodes}
        valid_years = [y for y in years.values() if y]
        if not valid_years:
            # no years -> fall back to a simple layout
            pos = nx.spring_layout(G, seed=42)
            self._pos = {n: (float(p[0]), float(p[1])) for n, p in pos.items()}
            return
        ymin = min(valid_years)
        # within each year, stack by LCS so big nodes are separated
        by_year = {}
        for n in sorted(nodes, key=lambda k: -G.nodes[k]["lcs"]):
            yr = years[n] or ymin
            col = by_year.setdefault(yr, [])
            col.append(n)
        pos = {}
        for yr, col in by_year.items():
            k = len(col)
            for i, n in enumerate(col):
                pos[n] = (yr, (i - (k - 1) / 2.0))
        self._pos = pos

    # ------------------------------------------------------------- draw
    def _sizes(self):
        G = self._G
        if self.node_size_by == 2:
            return {n: 14.0 for n in self._nodes}
        if self.node_size_by == 1:
            vals = {n: float(G.nodes[n].get("citations", 0) or 0) for n in self._nodes}
        else:
            vals = {n: float(G.nodes[n].get("lcs", 0)) for n in self._nodes}
        vmax = max(vals.values()) if vals else 1
        return {n: 8 + 28 * (v / vmax if vmax else 0) for n, v in vals.items()}

    def _redraw(self):
        if not HAS_PG or not hasattr(self, "graph"):
            return
        self.graph.clear()
        G = self._G
        if G is None or not self._pos:
            return
        # edges (cited -> citing, drawn left-to-right in time)
        xs, ys = [], []
        for u, v in G.edges():
            if u not in self._pos or v not in self._pos:
                continue
            x0, y0 = self._pos[u]; x1, y1 = self._pos[v]
            if self.curved_edges:
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                norm = (dx * dx + dy * dy) ** 0.5 or 1
                cx, cy = mx - dy / norm * 0.12 * norm, my + dx / norm * 0.12 * norm
                t = np.linspace(0, 1, 12)
                bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
                by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
                xs += list(bx) + [np.nan]; ys += list(by) + [np.nan]
            else:
                xs += [x0, x1, np.nan]; ys += [y0, y1, np.nan]
        if xs:
            self.graph.addItem(pg.PlotCurveItem(
                x=np.array(xs), y=np.array(ys), connect="finite",
                pen=pg.mkPen((150, 150, 150, 120), width=1)))
        sizes = self._sizes()
        spots = []
        for n in self._nodes:
            if n not in self._pos:
                continue
            brush = (pg.mkBrush(230, 126, 34) if n in self._selected
                     else pg.mkBrush(74, 144, 217))
            spots.append({"pos": self._pos[n], "size": sizes[n], "data": n,
                          "brush": brush, "pen": pg.mkPen("w", width=0.5)})
        self._scatter = pg.ScatterPlotItem(hoverable=True, tip=None)
        self._scatter.addPoints(spots)
        self.graph.addItem(self._scatter)
        if self.show_labels:
            order = sorted(self._nodes, key=lambda n: -G.nodes[n].get("lcs", 0))
            for n in order[:30]:
                if n not in self._pos:
                    continue
                if self.label_field == 1:
                    lbl = str(G.nodes[n].get("title", n))[:24]
                else:
                    lbl = str(n)[:18]
                t = pg.TextItem(lbl, color=(40, 40, 40), anchor=(0.5, 1.3))
                t.setPos(self._pos[n][0], self._pos[n][1])
                self.graph.addItem(t)
        self.graph.getViewBox().autoRange()

    def _on_clicked(self, ev):
        if not HAS_PG or self._G is None or not hasattr(self, "_scatter"):
            return
        vb = self.graph.getPlotItem().vb
        pts = self._scatter.pointsAt(vb.mapSceneToView(ev.scenePos()))
        if not len(pts):
            return
        n = pts[0].data()
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl:
            self._selected.symmetric_difference_update({n})
        else:
            self._selected = set() if self._selected == {n} else {n}
        self._redraw()
        self._send_selected()

    # ------------------------------------------------------------- outputs
    def _send_node_data(self):
        G = self._G
        if G is None or G.number_of_nodes() == 0:
            self.Outputs.node_data.send(None); return
        rows = []
        for n in self._nodes:
            d = G.nodes[n]
            rows.append([str(d.get("title", n))[:120], str(n),
                         float(d.get("year", 0) or 0),
                         float(d.get("lcs", 0)),
                         float(d.get("citations", 0) or 0)])
        domain = Domain([ContinuousVariable("Year"),
                         ContinuousVariable("Local citations"),
                         ContinuousVariable("Global citations")],
                        metas=[StringVariable("Title"), StringVariable("ID")])
        X = np.array([[r[2], r[3], r[4]] for r in rows], dtype=float)
        M = np.array([[r[0], r[1]] for r in rows], dtype=object)
        self.Outputs.node_data.send(Table.from_numpy(domain, X, metas=M))

    def _send_selected(self):
        if self._data is None or not self._selected or self._df is None:
            self.Outputs.selected.send(None); return
        # map selected node ids back to documents via title/id match
        idc = self._detect("EID", "DOI", "UT", "id", "unique-id", "Doc ID")
        title = self._detect("Title", "TI", "title")
        sel = {str(s) for s in self._selected}
        idx = []
        for i in range(len(self._df)):
            hit = False
            if idc is not None and str(self._df[idc].iloc[i]) in sel:
                hit = True
            elif title is not None and str(self._df[title].iloc[i]) in sel:
                hit = True
            if hit and i < len(self._data):
                idx.append(i)
        self.Outputs.selected.send(self._data[idx] if idx else None)

    def _export_pajek(self, kind):
        if self._G is None or self._G.number_of_nodes() == 0:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Pajek", "historiograph", "Pajek (*." + kind + ")")
        if not path:
            return
        if not path.lower().endswith("." + kind):
            path += "." + kind
        G = self._G
        nodes = list(G.nodes())
        idx = {n: i + 1 for i, n in enumerate(nodes)}
        try:
            with open(path, "w", encoding="utf-8") as f:
                if kind == "net":
                    f.write(f"*Vertices {len(nodes)}\n")
                    for n in nodes:
                        lbl = str(G.nodes[n].get("title", n)).replace('"', "'")[:60]
                        f.write(f'{idx[n]} "{lbl}"\n')
                    f.write("*Arcs\n")
                    for u, v in G.edges():
                        f.write(f"{idx[u]} {idx[v]} 1\n")
                elif kind == "clu":
                    f.write(f"*Vertices {len(nodes)}\n")
                    for n in nodes:
                        f.write(f"{int(G.nodes[n].get('year', 0) or 0)}\n")
                else:  # vec
                    f.write(f"*Vertices {len(nodes)}\n")
                    for n in nodes:
                        f.write(f"{float(G.nodes[n].get('lcs', 0)):.4f}\n")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pajek export failed")
            self.Error.build_failed(f"Export failed: {exc}")


if __name__ == "__main__":
    WidgetPreview(OWHistoriograph).run()
