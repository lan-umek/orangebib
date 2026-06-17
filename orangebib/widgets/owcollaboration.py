# -*- coding: utf-8 -*-
"""
Collaboration Network Widget
===========================
International (country) collaboration network from co-authored affiliations,
using `biblium.addons.geographic_analysis.analyze_collaborations`. Countries
are nodes; edges weight = number of co-authored papers. Communities are
detected and the collaboration matrix is emitted.
"""

import logging
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
    from biblium.addons.geographic_analysis import analyze_collaborations
    import networkx as nx
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    analyze_collaborations = None
    nx = None

logger = logging.getLogger(__name__)
AFF_CANDIDATES = ["Affiliations", "Affiliation", "C1", "oa_institutions"]
COUNTRY_CANDIDATES = ["Countries of Authors", "Countries", "Country",
                      "oa_institution_countries", "authorships.countries"]
_AUTO = "(parse from affiliations)"
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


class CollabWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, df, aff, country, year, min_collab):
        super().__init__()
        self._df = df; self._aff = aff; self._country = country
        self._year = year; self._min = min_collab

    def run(self):
        try:
            links, matrix = analyze_collaborations(
                self._df, affiliation_col=self._aff, country_col=self._country,
                year_col=self._year, min_collaborations=self._min, verbose=False)
            self.finished.emit(matrix, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("collaboration failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWCollaboration(OWWidget):
    """International collaboration network."""

    name = "Collaboration Network"
    description = "Country collaboration network from co-authored affiliations"
    icon = "icons/collaboration.svg"
    priority = 450
    keywords = ["collaboration", "country", "co-authorship", "international",
                "network", "partnership"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with affiliations/countries")

    class Outputs:
        matrix = Output("Collaboration Matrix", Table, doc="Country × country counts")
        node_data = Output("Node Data", Table, doc="Countries with degree & community")

    aff_col = settings.Setting("")
    country_col = settings.Setting(_AUTO)
    min_collab = settings.Setting(1)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons + networkx required.")
        compute_error = Msg("{}")

    class Information(OWWidget.Information):
        built = Msg("{} countries, {} links")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None
        self._nodes: List[str] = []
        self._pos: Dict = {}

        box = gui.widgetBox(self.controlArea, "Columns")
        grid = QGridLayout()
        grid.addWidget(QLabel("Country column:"), 0, 0)
        self.country_combo = QComboBox()
        self.country_combo.currentTextChanged.connect(lambda t: setattr(self, "country_col", t))
        grid.addWidget(self.country_combo, 0, 1)
        grid.addWidget(QLabel("Affiliations:"), 1, 0)
        self.aff_combo = QComboBox()
        self.aff_combo.currentTextChanged.connect(lambda t: setattr(self, "aff_col", t))
        grid.addWidget(self.aff_combo, 1, 1)
        box.layout().addLayout(grid)
        gui.spin(box, self, "min_collab", 1, 50, label="Min collaborations:", callback=self._rebuild)

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
        self._labels = []

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py", "publication_year", "oa_publication_year"):
                return c
        return "Year"

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        cols = list(self._df.columns) if self._df is not None else []
        self._fill(self.country_combo, [_AUTO] + [c for c in COUNTRY_CANDIDATES if c in cols] + cols, self.country_col)
        self._fill(self.aff_combo, [c for c in AFF_CANDIDATES if c in cols] + cols, self.aff_col)
        if data is None:
            self.Error.no_data()

    @staticmethod
    def _fill(combo, items, current):
        seen, uniq = set(), []
        for it in items:
            if it not in seen:
                seen.add(it); uniq.append(it)
        combo.blockSignals(True); combo.clear(); combo.addItems(uniq)
        if current in uniq:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _rebuild(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        country = self.country_combo.currentText()
        country = None if country == _AUTO else country
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Analyzing collaborations...")
        self._worker = CollabWorker(self._df, self.aff_combo.currentText(),
                                    country, self._year_col(), self.min_collab)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, matrix, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or matrix is None or matrix.empty:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "no collaborations found")
            self.Outputs.matrix.send(None); self.Outputs.node_data.send(None)
            return
        countries = list(matrix.index)
        edges = []
        m = matrix.values
        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                w = m[i, j]
                if w and w > 0:
                    edges.append((i, j, int(w)))
        if len(countries) < 2 or not edges:
            self.status_label.setText("Too few collaborations")
            self.graph.clear(); return
        self._nodes = countries
        G = nx.Graph(); G.add_nodes_from(range(len(countries)))
        for i, j, w in edges:
            G.add_edge(i, j, weight=w)
        try:
            pos = nx.spring_layout(G, weight="weight", seed=42, k=1.3 / (len(countries) ** 0.5))
        except Exception:  # noqa: BLE001
            pos = nx.circular_layout(G)
        self._pos = {i: (float(p[0]), float(p[1])) for i, p in pos.items()}
        self._deg = [G.degree(i, weight="weight") for i in range(len(countries))]
        self._comm = [0] * len(countries)
        try:
            from networkx.algorithms.community import louvain_communities
            for cid, c in enumerate(louvain_communities(G, weight="weight", seed=42)):
                for n in c:
                    self._comm[n] = cid
        except Exception:  # noqa: BLE001
            pass
        self._edges = edges
        self._render()
        self.status_label.setText(f"Done — {len(countries)} countries")
        self.Information.built(len(countries), len(edges))
        mat_out = matrix.copy(); mat_out.insert(0, "Country", [str(i) for i in matrix.index])
        self.Outputs.matrix.send(_df_to_table(mat_out))
        nd = pd.DataFrame({"Country": countries, "Degree": self._deg, "Community": self._comm})
        self.Outputs.node_data.send(_df_to_table(nd))

    def _render(self):
        self.graph.clear()
        xs, ys = [], []
        for i, j, w in self._edges:
            x0, y0 = self._pos[i]; x1, y1 = self._pos[j]
            xs.extend([x0, x1, np.nan]); ys.extend([y0, y1, np.nan])
        if xs:
            self.graph.addItem(pg.PlotCurveItem(x=np.array(xs), y=np.array(ys),
                                                pen=pg.mkPen((170, 170, 170, 110), width=1), connect="finite"))
        dmax = max(self._deg) if self._deg else 1
        spots = [{"pos": self._pos[i], "data": i,
                  "size": 8 + 24 * (self._deg[i] / dmax if dmax else 0),
                  "brush": pg.mkBrush(PALETTE[self._comm[i] % len(PALETTE)]),
                  "pen": pg.mkPen("w", width=0.5)} for i in range(len(self._nodes))]
        self._scatter.setData(spots)
        self.graph.addItem(self._scatter)
        order = sorted(range(len(self._nodes)), key=lambda i: -self._deg[i])
        for i in order[:30]:
            t = pg.TextItem(str(self._nodes[i])[:18], color=(40, 40, 40), anchor=(0.5, 1.2))
            t.setPos(self._pos[i][0], self._pos[i][1]); self.graph.addItem(t)
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
            self._tip.setText(f"{self._nodes[i]}\ncollab degree {self._deg[i]:.0f}")
            self._tip.setPos(mp.x(), mp.y()); self._tip.show()
        else:
            self._tip.hide()

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWCollaboration).run()
