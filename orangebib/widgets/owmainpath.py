# -*- coding: utf-8 -*-
"""
Main Path Analysis Widget
========================
Trace the main path of knowledge flow through the intra-corpus citation
network (SPC / SPLC / SPNP traversal counting) using
`biblium.main_path.compute_main_path_analysis`. Builds the citation graph from
OpenAlex references (oa_referenced_works → oa_openalex_id).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import QLabel, QComboBox, QPushButton, QGridLayout, QProgressBar

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.main_path import compute_main_path_analysis
    import networkx as nx
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    compute_main_path_analysis = None
    nx = None

logger = logging.getLogger(__name__)
METHODS = ["SPC", "SPLC", "SPNP"]


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
        M[:, i] = [("" if (v is None or (isinstance(v, float) and v != v)) else str(v)) for v in df[c]]
    return Table.from_numpy(domain, X, metas=M)


def _short_id(x):
    return str(x).replace("https://openalex.org/", "").strip()


class MainPathWorker(QThread):
    finished = pyqtSignal(object, object, str)

    def __init__(self, df, id_col, refs_col, title_col, year_col, method):
        super().__init__()
        self._df = df; self._id = id_col; self._refs = refs_col
        self._title = title_col; self._year = year_col; self._method = method

    def run(self):
        try:
            ids = {}
            node_data = {}
            for _, r in self._df.iterrows():
                pid = _short_id(r.get(self._id, ""))
                if not pid:
                    continue
                ids[pid] = True
                node_data[pid] = {
                    "title": str(r.get(self._title, ""))[:120] if self._title else "",
                    "year": r.get(self._year) if self._year else None,
                }
            G = nx.DiGraph()
            G.add_nodes_from(ids.keys())
            for _, r in self._df.iterrows():
                pid = _short_id(r.get(self._id, ""))
                if not pid:
                    continue
                refs = str(r.get(self._refs, "") or "")
                for ref in [x.strip() for x in refs.replace(";", "|").split("|") if x.strip()]:
                    rid = _short_id(ref)
                    if rid in ids and rid != pid:
                        # knowledge flows from cited (rid) to citing (pid)
                        G.add_edge(rid, pid)
            if G.number_of_edges() == 0:
                self.finished.emit(None, None, "No intra-corpus citation links found "
                                   "(needs OpenAlex referenced_works).")
                return
            res = compute_main_path_analysis(G, method=self._method,
                                             node_data=node_data, verbose=False)
            # main path documents
            path = res.global_main_path or []
            rows = []
            for rank, nid in enumerate(path, 1):
                nd = node_data.get(nid, {})
                rows.append({"Rank": rank, "Title": nd.get("title", nid),
                             "Year": nd.get("year"), "ID": nid})
            path_df = pd.DataFrame(rows)
            stats = pd.DataFrame([{
                "Nodes": res.n_nodes, "Edges": res.n_edges,
                "Sources": res.n_sources, "Sinks": res.n_sinks,
                "Main path length": res.path_length,
            }])
            self.finished.emit(path_df, stats, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("main path failed")
            self.finished.emit(None, None, f"{type(exc).__name__}: {exc}")


class OWMainPath(OWWidget):
    """Main path of knowledge flow through the citation network."""

    name = "Main Path Analysis"
    description = "Main path of knowledge flow (SPC/SPLC/SPNP) from OpenAlex citations"
    icon = "icons/main_path.svg"
    priority = 460
    keywords = ["main path", "citation", "knowledge flow", "spc", "traversal",
                "trajectory"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="OpenAlex-enriched data (referenced works)")

    class Outputs:
        main_path = Output("Main Path", Table, doc="Documents on the main path")
        stats = Output("Stats", Table, doc="Network statistics")

    id_col = settings.Setting("")
    refs_col = settings.Setting("")
    method = settings.Setting("SPC")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium + networkx required.")
        compute_error = Msg("{}")

    class Information(OWWidget.Information):
        done = Msg("Main path: {} documents")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Citation graph")
        grid = QGridLayout()
        grid.addWidget(QLabel("Paper ID:"), 0, 0)
        self.id_combo = QComboBox()
        self.id_combo.currentTextChanged.connect(lambda t: setattr(self, "id_col", t))
        grid.addWidget(self.id_combo, 0, 1)
        grid.addWidget(QLabel("References:"), 1, 0)
        self.refs_combo = QComboBox()
        self.refs_combo.currentTextChanged.connect(lambda t: setattr(self, "refs_col", t))
        grid.addWidget(self.refs_combo, 1, 1)
        grid.addWidget(QLabel("Method:"), 2, 0)
        self.method_combo = QComboBox(); self.method_combo.addItems(METHODS)
        self.method_combo.setCurrentText(self.method)
        self.method_combo.currentTextChanged.connect(lambda t: setattr(self, "method", t))
        grid.addWidget(self.method_combo, 2, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Find Main Path")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.graph = pg.PlotWidget(background="w")
        self.graph.hideAxis("left")
        self.graph.setLabel("bottom", "Main path (knowledge flow →)")
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
        self._fill(self.id_combo, ["oa_openalex_id", "OpenAlex ID", "DOI"], cols, self.id_col)
        self._fill(self.refs_combo, ["oa_referenced_works", "referenced_works", "References", "CR"], cols, self.refs_col)
        if data is None:
            self.Error.no_data()

    @staticmethod
    def _fill(combo, prefer, cols, current):
        ordered = [c for c in prefer if c in cols] + [c for c in cols if c not in prefer]
        combo.blockSignals(True); combo.clear(); combo.addItems(ordered)
        if current in ordered:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _title_col(self):
        for c in ("Title", "TI", "Document Title"):
            if self._df is not None and c in self._df.columns:
                return c
        return ""

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py", "publication_year", "oa_publication_year"):
                return c
        return ""

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        if not self.id_combo.currentText() or not self.refs_combo.currentText():
            self.Error.compute_error("Select ID and references columns"); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Building citation graph...")
        self._worker = MainPathWorker(
            self._df, self.id_combo.currentText(), self.refs_combo.currentText(),
            self._title_col(), self._year_col(), self.method)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, path_df, stats, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or path_df is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            self.Outputs.main_path.send(None); self.Outputs.stats.send(None)
            return
        self._render(path_df)
        n = len(path_df)
        self.status_label.setText(f"Done — {n} documents on main path")
        self.Information.done(n)
        self.Outputs.main_path.send(_df_to_table(path_df))
        self.Outputs.stats.send(_df_to_table(stats))

    def _render(self, path_df):
        self.graph.clear()
        n = len(path_df)
        if n == 0:
            return
        xs = list(range(n))
        # arrows between consecutive nodes
        ax, ay = [], []
        for i in range(n - 1):
            ax.extend([i, i + 1, np.nan]); ay.extend([0, 0, np.nan])
        if ax:
            self.graph.addItem(pg.PlotCurveItem(x=np.array(ax), y=np.array(ay),
                                                pen=pg.mkPen("#888", width=2), connect="finite"))
        self.graph.addItem(pg.ScatterPlotItem(
            x=xs, y=[0] * n, size=16, brush=pg.mkBrush("#4a90d9"), pen=pg.mkPen("w")))
        for i, (_, r) in enumerate(path_df.iterrows()):
            lab = f"{r['Title']}".strip() or str(r["ID"])
            yr = f" ({int(r['Year'])})" if pd.notna(r.get("Year")) else ""
            t = pg.TextItem(str(lab)[:36] + yr, color=(40, 40, 40),
                            anchor=(0, 0.5), angle=35)
            t.setPos(i, 0.05)
            self.graph.addItem(t)
        self.graph.setXRange(-0.5, n - 0.5)
        self.graph.setYRange(-1, 2)

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWMainPath).run()
