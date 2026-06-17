# -*- coding: utf-8 -*-
"""
Thematic Evolution Widget
========================
Strategic themes across time periods with an alluvial flow diagram showing how
themes split, merge and evolve. For each period, keyword co-occurrence is
clustered into themes (communities); consecutive periods are linked by their
shared keywords (flow width = shared keywords).
"""

import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QSpinBox, QGridLayout,
                              QProgressBar, QTabWidget, QWidget, QVBoxLayout)

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
ENTITY_PATTERNS = ("keyword", "concept", "topic", "subject", "field")
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


def _split(val):
    s = str(val)
    for sep in ["||", "|", "; ", ";", ", "]:
        if sep in s:
            return [e.strip() for e in s.split(sep) if e.strip()]
    return [s.strip()] if s.strip() else []


def _period_themes(docs_kw, top_k, n_themes):
    """Cluster keyword co-occurrence into themes for one period.
    Returns list of (label, keyword_set, size)."""
    occ = defaultdict(int)
    for kws in docs_kw:
        for k in kws:
            occ[k] += 1
    top = [k for k, _ in sorted(occ.items(), key=lambda kv: -kv[1])[:top_k]]
    idx = {k: i for i, k in enumerate(top)}
    if len(top) < 2:
        return []
    G = nx.Graph()
    G.add_nodes_from(range(len(top)))
    ew = defaultdict(int)
    for kws in docs_kw:
        present = [k for k in kws if k in idx]
        for a, b in combinations(sorted(set(present)), 2):
            ew[(idx[a], idx[b])] += 1
    for (i, j), w in ew.items():
        G.add_edge(i, j, weight=w)
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, weight="weight", seed=42)
    except Exception:  # noqa: BLE001
        comms = [set(G.nodes())]
    comms = [set(c) for c in comms]
    node_comm = {}
    for cid, comm in enumerate(comms):
        for n in comm:
            node_comm[n] = cid
    # Callon centrality (external links) and density (internal links) per theme.
    internal = defaultdict(float)
    external = defaultdict(float)
    for (i, j), w in ew.items():
        ci, cj = node_comm.get(i), node_comm.get(j)
        if ci is None or cj is None:
            continue
        if ci == cj:
            internal[ci] += w
        else:
            external[ci] += w
            external[cj] += w
    themes = []
    for cid, comm in enumerate(comms):
        kws = {top[i] for i in comm}
        if not kws:
            continue
        size = sum(occ[k] for k in kws)
        label = max(kws, key=lambda k: occ[k])
        centrality = external[cid] * 10.0
        density = internal[cid] / max(1, len(comm)) * 100.0
        themes.append((label, kws, size, centrality, density))
    themes.sort(key=lambda t: -t[2])
    return themes[:n_themes]


class EvolutionWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, df, kw_col, year_col, n_periods, top_k, n_themes, cutpoints=None):
        super().__init__()
        self._df = df; self._kw = kw_col; self._yc = year_col
        self._np = n_periods; self._tk = top_k; self._nt = n_themes
        self._cuts = cutpoints or []

    def run(self):
        try:
            df = self._df.copy()
            df["_y"] = pd.to_numeric(df[self._yc], errors="coerce")
            df = df.dropna(subset=["_y"]); df["_y"] = df["_y"].astype(int)
            df = df[(df["_y"] > 1500) & (df["_y"] < 2100)]
            if df.empty:
                self.finished.emit(None, "No valid years"); return
            ymin, ymax = int(df["_y"].min()), int(df["_y"].max())
            # Build period boundaries from user cut points, else equal split.
            bounds = []  # list of (lo, hi)
            cuts = sorted({int(c) for c in self._cuts
                           if ymin < int(c) <= ymax})
            if cuts:
                lo = ymin
                for c in cuts:
                    bounds.append((lo, c - 1))
                    lo = c
                bounds.append((lo, ymax))
            else:
                nper = min(self._np, max(1, ymax - ymin + 1))
                for p in range(nper):
                    lo = ymin + int(p * (ymax - ymin + 1) / nper)
                    hi = ymin + int((p + 1) * (ymax - ymin + 1) / nper) - 1
                    bounds.append((lo, hi))
            period_themes = []
            labels = []
            for lo, hi in bounds:
                labels.append(f"{lo}–{hi}")
                sub = df[(df["_y"] >= lo) & (df["_y"] <= hi)]
                docs_kw = [list(dict.fromkeys(_split(v))) for v in sub[self._kw] if pd.notna(v)]
                period_themes.append(_period_themes(docs_kw, self._tk, self._nt))
            nper = len(bounds)
            # flows between consecutive periods (shared keywords)
            flows = []
            for p in range(nper - 1):
                for a, ta in enumerate(period_themes[p]):
                    ka = ta[1]
                    for b, tb in enumerate(period_themes[p + 1]):
                        shared = len(ka & tb[1])
                        if shared > 0:
                            flows.append((p, a, b, shared))
            self.finished.emit((labels, period_themes, flows), "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("thematic evolution failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWThematicEvolution(OWWidget):
    """Thematic evolution (alluvial flow of themes over time)."""

    name = "Thematic Evolution"
    description = "Alluvial flow of strategic themes across time periods"
    icon = "icons/thematic_evolution.svg"
    priority = 375
    keywords = ["thematic evolution", "alluvial", "themes", "evolution",
                "strategic", "sankey", "temporal"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with keywords + Year")

    class Outputs:
        flows = Output("Flows", Table, doc="Theme transitions between periods")

    kw_col = settings.Setting("")
    n_periods = settings.Setting(4)
    top_k = settings.Setting(50)
    n_themes = settings.Setting(6)
    cutpoints_str = settings.Setting("")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_networkx = Msg("networkx is required")
        no_year = Msg("Year column not found")
        compute_error = Msg("{}")

    class Information(OWWidget.Information):
        done = Msg("{} periods")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Themes over time")
        grid = QGridLayout()
        grid.addWidget(QLabel("Keywords column:"), 0, 0)
        self.kw_combo = QComboBox()
        self.kw_combo.currentTextChanged.connect(lambda t: setattr(self, "kw_col", t))
        grid.addWidget(self.kw_combo, 0, 1)
        box.layout().addLayout(grid)
        gui.spin(box, self, "n_periods", 2, 10, label="Periods:", callback=self._rebuild)
        gui.spin(box, self, "n_themes", 2, 12, label="Themes / period:", callback=self._rebuild)
        gui.spin(box, self, "top_k", 20, 200, label="Top keywords / period:", callback=self._rebuild)
        gui.lineEdit(box, self, "cutpoints_str",
                     label="Cut points (years, comma-sep; overrides Periods):",
                     callback=self._rebuild)

        self.run_btn = QPushButton("Build")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._rebuild)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.view_tabs = QTabWidget()
        self.graph = pg.PlotWidget(background="w")
        self.graph.hideAxis("left")
        self.view_tabs.addTab(self.graph, "Alluvial")

        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        self.period_combo = QComboBox()
        self.period_combo.currentIndexChanged.connect(self._render_map)
        map_layout.addWidget(self.period_combo)
        self.map_plot = pg.PlotWidget(background="w")
        self.map_plot.setLabel("bottom", "Centrality (relevance)")
        self.map_plot.setLabel("left", "Density (development)")
        map_layout.addWidget(self.map_plot)
        self.view_tabs.addTab(map_tab, "Thematic map")
        self.mainArea.layout().addWidget(self.view_tabs)
        self._labels = []
        self._period_themes = []

        if not HAS_NX:
            self.Error.no_networkx()
            self.run_btn.setEnabled(False)

    def _entity_columns(self):
        if self._df is None:
            return []
        return [c for c in self._df.columns if any(k in str(c).lower() for k in ENTITY_PATTERNS)]

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py", "publication_year", "oa_publication_year"):
                return c
        return None

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_NX:
            self.Error.no_networkx()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.kw_combo.blockSignals(True); self.kw_combo.clear()
        ent = self._entity_columns()
        self.kw_combo.addItems(ent)
        if self.kw_col in ent:
            self.kw_combo.setCurrentText(self.kw_col)
        elif ent:
            self.kw_col = ent[0]
        self.kw_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _rebuild(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_NX:
            self.Error.no_networkx(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        if not self.kw_combo.currentText():
            self.Error.compute_error("No keywords column"); return
        yc = self._year_col()
        if yc is None:
            self.Error.no_year(); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Computing themes per period...")
        cuts = []
        for tok in str(self.cutpoints_str).replace(";", ",").split(","):
            tok = tok.strip()
            if tok.isdigit():
                cuts.append(int(tok))
        self._worker = EvolutionWorker(self._df, self.kw_combo.currentText(), yc,
                                       self.n_periods, self.top_k, self.n_themes,
                                       cutpoints=cuts)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, result, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or result is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            self.Outputs.flows.send(None)
            return
        labels, period_themes, flows = result
        self._labels = labels
        self._period_themes = period_themes
        self._render(labels, period_themes, flows)
        self.period_combo.blockSignals(True)
        self.period_combo.clear()
        self.period_combo.addItems(labels)
        self.period_combo.blockSignals(False)
        if labels:
            self._render_map(0)
        self.status_label.setText(f"Done — {len(labels)} periods")
        self.Information.done(len(labels))
        rows = []
        for p, a, b, w in flows:
            rows.append({
                "From period": labels[p], "From theme": period_themes[p][a][0],
                "To period": labels[p + 1], "To theme": period_themes[p + 1][b][0],
                "Shared keywords": w,
            })
        fdf = pd.DataFrame(rows)
        if not fdf.empty:
            metas = [StringVariable(c) for c in fdf.columns]
            self.Outputs.flows.send(Table.from_numpy(
                Domain([], metas=metas), np.empty((len(fdf), 0)), metas=fdf.astype(str).values))
        else:
            self.Outputs.flows.send(None)

    def _render(self, labels, period_themes, flows):
        self.graph.clear()
        nper = len(labels)
        # y positions per theme (stacked, normalized within period)
        pos = {}
        for p, themes in enumerate(period_themes):
            m = len(themes)
            for i in range(m):
                pos[(p, i)] = (p, (m - 1 - i) - (m - 1) / 2.0)
        # flows as curves
        for p, a, b, w in flows:
            if (p, a) not in pos or (p + 1, b) not in pos:
                continue
            x0, y0 = pos[(p, a)]; x1, y1 = pos[(p + 1, b)]
            ts = np.linspace(0, 1, 16)
            bx = (1 - ts) * x0 + ts * x1
            by = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * ((y0 + y1) / 2) + ts ** 2 * y1
            self.graph.addItem(pg.PlotCurveItem(
                x=bx, y=by, pen=pg.mkPen((150, 150, 150, 130), width=min(1 + w, 8))))
        # theme nodes + labels
        for p, themes in enumerate(period_themes):
            for i, theme in enumerate(themes):
                label = theme[0]
                x, y = pos[(p, i)]
                self.graph.addItem(pg.ScatterPlotItem(
                    x=[x], y=[y], size=18, brush=pg.mkBrush("#4a90d9"),
                    pen=pg.mkPen("w")))
                t = pg.TextItem(str(label)[:22], color=(40, 40, 40), anchor=(0.5, 1.4))
                t.setPos(x, y); self.graph.addItem(t)
        # period axis labels
        self.graph.getAxis("bottom").setTicks([[(p, labels[p]) for p in range(nper)]])
        self.graph.setXRange(-0.5, nper - 0.5)

    def _render_map(self, period_idx):
        """Per-period strategic (thematic) map: centrality x density quadrants."""
        if not hasattr(self, "map_plot"):
            return
        self.map_plot.clear()
        if period_idx is None or period_idx < 0 or period_idx >= len(self._period_themes):
            return
        themes = self._period_themes[period_idx]
        if not themes:
            return
        xs = [t[3] for t in themes]   # centrality
        ys = [t[4] for t in themes]   # density
        sizes = [t[2] for t in themes]
        smax = max(sizes) or 1
        xmed = float(np.median(xs)) if xs else 0.0
        ymed = float(np.median(ys)) if ys else 0.0
        self.map_plot.addItem(pg.InfiniteLine(pos=xmed, angle=90,
                              pen=pg.mkPen("#bbb", style=Qt.DashLine)))
        self.map_plot.addItem(pg.InfiniteLine(pos=ymed, angle=0,
                              pen=pg.mkPen("#bbb", style=Qt.DashLine)))
        # numeric centrality/density values are not directly interpretable -> hide
        self.map_plot.getAxis("bottom").setStyle(showValues=False)
        self.map_plot.getAxis("left").setStyle(showValues=False)
        xspan = (max(xs) - min(xs)) or 1.0
        yspan = (max(ys) - min(ys)) or 1.0
        x_hi = max(xs) + xspan * 0.05; x_lo = min(xs) - xspan * 0.05
        y_hi = max(ys) + yspan * 0.08; y_lo = min(ys) - yspan * 0.08
        quad = [(x_hi, y_hi, (1, 1), "Motor themes"),
                (x_lo, y_hi, (0, 1), "Niche / specialised"),
                (x_hi, y_lo, (1, 0), "Basic / transversal"),
                (x_lo, y_lo, (0, 0), "Emerging or declining")]
        for qx, qy, anc, txt in quad:
            qi = pg.TextItem(txt, color=(150, 150, 150), anchor=anc)
            qi.setPos(qx, qy)
            self.map_plot.addItem(qi)
        for i, t in enumerate(themes):
            r = 12 + 28 * (sizes[i] / smax)
            self.map_plot.addItem(pg.ScatterPlotItem(
                x=[xs[i]], y=[ys[i]], size=r,
                brush=pg.mkBrush(PALETTE[i % len(PALETTE)]), pen=pg.mkPen("w")))
            lbl = pg.TextItem(str(t[0])[:22], color=(40, 40, 40), anchor=(0.5, 1.6))
            lbl.setPos(xs[i], ys[i])
            self.map_plot.addItem(lbl)
        self.map_plot.setTitle(
            self._labels[period_idx] if period_idx < len(self._labels) else "")

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWThematicEvolution).run()
