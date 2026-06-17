# -*- coding: utf-8 -*-
"""
Diachronic Network Widget
========================
Animated co-occurrence network over time: the layout is computed once on the
full network (so node positions stay stable), then edges/nodes are revealed
period by period — you literally watch the network grow as links accumulate.
"""

import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QTimer
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QPushButton, QSpinBox, QCheckBox, QGridLayout,
    QHBoxLayout, QSlider, QFileDialog, QApplication, QMessageBox,
)
from AnyQt.QtGui import QImage

import pyqtgraph as pg

from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    import networkx as nx
    HAS_NX = True
except Exception:  # noqa: BLE001
    HAS_NX = False

logger = logging.getLogger(__name__)

ENTITY_PATTERNS = ("keyword", "author", "source", "journal", "countr",
                   "affiliation", "subject", "field", "institution", "topic",
                   "concept", "sdg", "domain")
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


def _split(val):
    s = str(val)
    for sep in ["||", "|", "; ", ";", ", "]:
        if sep in s:
            return [e.strip() for e in s.split(sep) if e.strip()]
    return [s.strip()] if s.strip() else []


class OWDiachronicNetwork(OWWidget):
    """Animated co-occurrence network over time."""

    name = "Diachronic Network"
    description = "Animated co-occurrence network growing over time"
    icon = "icons/diachronic_network.svg"
    priority = 470
    keywords = ["diachronic", "temporal", "network", "evolution", "animation",
                "co-occurrence", "dynamic"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    class Outputs:
        selected = Output("Selected Documents", Table, doc="Docs up to current period")

    column_name = settings.Setting("")
    top_n = settings.Setting(40)
    n_periods = settings.Setting(8)
    min_edge_weight = settings.Setting(1)
    interval = settings.Setting(900)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_entities = Msg("No entity column with data found")
        no_year = Msg("Year column not found")
        no_networkx = Msg("networkx is required")

    class Information(OWWidget.Information):
        built = Msg("{} nodes over {} periods")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._nodes: List[str] = []
        self._pos: Dict[int, tuple] = {}
        self._period_edges = []     # list per period: [(i,j,w),...] (cumulative)
        self._period_docs = []      # list per period: set of doc indices
        self._period_labels = []
        self._frame = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        self._build_controls()
        self.graph = pg.PlotWidget(background="w")
        self.graph.hideAxis("bottom"); self.graph.hideAxis("left")
        self.graph.setAspectLocked(True)
        self.mainArea.layout().addWidget(self.graph)
        self._period_text = pg.TextItem(color=(120, 120, 120), anchor=(0, 1))
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
        gui.spin(box, self, "top_n", 5, 200, label="Top N nodes:", callback=self._rebuild)
        gui.spin(box, self, "n_periods", 2, 30, label="Periods:", callback=self._rebuild)
        gui.spin(box, self, "min_edge_weight", 1, 50, label="Min edge weight:", callback=self._rebuild)

        pbox = gui.widgetBox(self.controlArea, "Playback")
        row = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play"); self.play_btn.clicked.connect(self._toggle)
        row.addWidget(self.play_btn)
        rb = QPushButton("⟲"); rb.clicked.connect(self._restart); row.addWidget(rb)
        pbox.layout().addLayout(row)
        self.export_btn = QPushButton("⬇ Export animation (GIF)")
        self.export_btn.clicked.connect(self._export_animation)
        pbox.layout().addWidget(self.export_btn)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        pbox.layout().addWidget(self.slider)
        self.period_label = QLabel(""); pbox.layout().addWidget(self.period_label)
        sp = QHBoxLayout(); sp.addWidget(QLabel("Speed (ms):"))
        self.speed = QSpinBox(); self.speed.setRange(200, 4000); self.speed.setSingleStep(100)
        self.speed.setValue(self.interval)
        self.speed.valueChanged.connect(lambda v: setattr(self, "interval", v))
        sp.addWidget(self.speed); pbox.layout().addLayout(sp)
        self.controlArea.layout().addStretch(1)

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

    def _year_col(self):
        for c in self._df.columns:
            if str(c).lower() in ("year", "publication year", "py",
                                  "publication_year", "oa_publication_year"):
                return c
        return None

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self._timer.stop()
        if not HAS_NX:
            self.Error.no_networkx()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.col_combo.blockSignals(True); self.col_combo.clear()
        if data is None:
            self.col_combo.blockSignals(False); self.Error.no_data(); return
        ent = self._entity_columns()
        if not ent:
            self.col_combo.blockSignals(False); self.Error.no_entities(); return
        self.col_combo.addItems(ent)
        if self.column_name in ent:
            self.col_combo.setCurrentText(self.column_name)
        else:
            self.column_name = ent[0]
        self.col_combo.blockSignals(False)
        self._rebuild()

    def _rebuild(self):
        self.Error.clear()
        self._timer.stop(); self.play_btn.setText("▶ Play")
        if self._df is None or not self.column_name or not HAS_NX:
            if not HAS_NX:
                self.Error.no_networkx()
            return
        year_col = self._year_col()
        if year_col is None:
            self.Error.no_year(); return
        col = self.column_name
        df = self._df.copy()
        df["_y"] = pd.to_numeric(df[year_col], errors="coerce")
        df = df.dropna(subset=["_y"])
        df["_y"] = df["_y"].astype(int)
        df = df[(df["_y"] > 1500) & (df["_y"] < 2100)]
        if df.empty:
            self.Error.no_year(); return

        # full occurrences -> top-N nodes
        occ = defaultdict(int)
        doc_entities = {}
        for idx, row in df.iterrows():
            ents = list(dict.fromkeys(_split(row[col]))) if pd.notna(row[col]) else []
            doc_entities[idx] = (int(row["_y"]), ents)
            for e in ents:
                occ[e] += 1
        top = [e for e, _ in sorted(occ.items(), key=lambda kv: -kv[1])[:self.top_n]]
        node_index = {e: i for i, e in enumerate(top)}
        self._nodes = top
        if len(top) < 2:
            self.Information.built(len(top), 0); self.graph.clear(); return

        # period cutoffs (equal-width year bins, cumulative)
        years = sorted({y for y, _ in doc_entities.values()})
        ymin, ymax = years[0], years[-1]
        nper = min(self.n_periods, max(1, ymax - ymin + 1))
        edges_period = []
        docs_period = []
        labels = []
        for p in range(nper):
            cutoff = ymin + int((p + 1) * (ymax - ymin + 1) / nper) - 1
            labels.append(f"{ymin}–{cutoff}")
            edge_w = defaultdict(int)
            docs = set()
            for idx, (y, ents) in doc_entities.items():
                if y > cutoff:
                    continue
                present = [e for e in ents if e in node_index]
                if present:
                    docs.add(idx)
                for a, b in combinations(sorted(set(present)), 2):
                    edge_w[(node_index[a], node_index[b])] += 1
            edges = [(i, j, w) for (i, j), w in edge_w.items()
                     if w >= self.min_edge_weight]
            edges_period.append(edges)
            docs_period.append(docs)
        self._period_edges = edges_period
        self._period_docs = docs_period
        self._period_labels = labels

        # layout once on the FINAL network -> stable positions
        G = nx.Graph()
        G.add_nodes_from(range(len(top)))
        for i, j, w in edges_period[-1]:
            G.add_edge(i, j, weight=w)
        try:
            pos = nx.spring_layout(G, weight="weight", seed=42,
                                   k=1.3 / (len(top) ** 0.5))
        except Exception:  # noqa: BLE001
            pos = nx.circular_layout(G)
        self._pos = {i: (float(p[0]), float(p[1])) for i, p in pos.items()}
        self._communities = [0] * len(top)
        try:
            from networkx.algorithms.community import louvain_communities
            for cid, comm in enumerate(louvain_communities(G, weight="weight", seed=42)):
                for nidx in comm:
                    self._communities[nidx] = cid
        except Exception:  # noqa: BLE001
            pass

        self.slider.blockSignals(True)
        self.slider.setRange(0, nper - 1); self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._frame = 0
        self.Information.built(len(top), nper)
        self._render()

    def _render(self):
        if not self._nodes or not self._pos:
            return
        edges = self._period_edges[self._frame]
        self.graph.clear()
        # degree in current period
        deg = defaultdict(float)
        for i, j, w in edges:
            deg[i] += w; deg[j] += w
        # edges (curved)
        xs, ys = [], []
        for i, j, w in edges:
            x0, y0 = self._pos[i]; x1, y1 = self._pos[j]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            nrm = (dx * dx + dy * dy) ** 0.5 or 1
            cx, cy = mx - dy / nrm * 0.10, my + dx / nrm * 0.10
            ts = np.linspace(0, 1, 12)
            bx = (1 - ts) ** 2 * x0 + 2 * (1 - ts) * ts * cx + ts ** 2 * x1
            by = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * cy + ts ** 2 * y1
            xs.extend(list(bx) + [np.nan]); ys.extend(list(by) + [np.nan])
        if xs:
            self.graph.addItem(pg.PlotCurveItem(
                x=np.array(xs), y=np.array(ys),
                pen=pg.mkPen((160, 160, 160, 120), width=1), connect="finite"))
        dmax = max(deg.values()) if deg else 1
        spots = []
        for i in range(len(self._nodes)):
            active = deg.get(i, 0) > 0
            size = 6 + 26 * (deg.get(i, 0) / dmax if dmax else 0)
            color = PALETTE[self._communities[i] % len(PALETTE)] if active else "#dddddd"
            spots.append({"pos": self._pos[i], "size": size if active else 5,
                          "brush": pg.mkBrush(color), "pen": pg.mkPen("w", width=0.5)})
        self.graph.addItem(pg.ScatterPlotItem(spots=spots))
        # labels for active hubs
        order = sorted(range(len(self._nodes)), key=lambda i: -deg.get(i, 0))
        for i in order[:25]:
            if deg.get(i, 0) <= 0:
                continue
            t = pg.TextItem(str(self._nodes[i])[:22], color=(40, 40, 40), anchor=(0.5, 1.2))
            t.setPos(self._pos[i][0], self._pos[i][1]); self.graph.addItem(t)
        self.graph.addItem(self._period_text)
        self._period_text.setText(self._period_labels[self._frame])
        self.graph.getViewBox().autoRange()
        self.period_label.setText(
            f"Period {self._frame + 1}/{len(self._period_labels)}: "
            f"{self._period_labels[self._frame]}  ({len(edges)} links)")
        self._emit_docs()

    def _emit_docs(self):
        docs = self._period_docs[self._frame] if self._period_docs else set()
        if self._data is not None and docs:
            idx = [d for d in docs if 0 <= d < len(self._data)]
            self.Outputs.selected.send(self._data[idx] if idx else None)

    def _export_animation(self):
        """Render every period and save an animated GIF."""
        if not self._period_labels:
            QMessageBox.information(self, "Export", "Nothing to export yet.")
            return
        try:
            from PIL import Image
        except Exception:  # noqa: BLE001
            QMessageBox.warning(self, "Export",
                                "Pillow (PIL) is required to export the animation.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export animation", "diachronic_network.gif", "GIF (*.gif)")
        if not path:
            return
        if not path.lower().endswith(".gif"):
            path += ".gif"
        self._timer.stop(); self.play_btn.setText("▶ Play")
        saved = self._frame
        frames = []
        try:
            for f in range(len(self._period_labels)):
                self._frame = f
                self._render()
                QApplication.processEvents()
                qimg = self.graph.grab().toImage().convertToFormat(
                    QImage.Format_RGBA8888)
                w, h = qimg.width(), qimg.height()
                ptr = qimg.constBits()
                try:
                    ptr.setsize(h * w * 4)
                except Exception:  # noqa: BLE001
                    pass
                img = Image.frombytes("RGBA", (w, h), bytes(ptr))
                frames.append(img.convert("P", palette=Image.ADAPTIVE))
            if frames:
                frames[0].save(path, save_all=True, append_images=frames[1:],
                               duration=int(self.interval), loop=0,
                               optimize=False, disposal=2)
            QMessageBox.information(self, "Export",
                                    f"Saved {len(frames)} frames to:\n{path}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("diachronic export failed")
            QMessageBox.warning(self, "Export", f"Export failed: {exc}")
        finally:
            self._frame = saved
            self._render()

    def _advance(self):
        if self._frame >= len(self._period_labels) - 1:
            self._timer.stop(); self.play_btn.setText("▶ Play"); return
        self._frame += 1
        self.slider.blockSignals(True); self.slider.setValue(self._frame); self.slider.blockSignals(False)
        self._render()

    def _toggle(self):
        if self._timer.isActive():
            self._timer.stop(); self.play_btn.setText("▶ Play")
        else:
            if self._frame >= len(self._period_labels) - 1:
                self._frame = 0
            self._timer.start(int(self.interval)); self.play_btn.setText("⏸ Pause")

    def _restart(self):
        self._timer.stop(); self.play_btn.setText("▶ Play"); self._frame = 0
        self.slider.blockSignals(True); self.slider.setValue(0); self.slider.blockSignals(False)
        self._render()

    def _on_slider(self, v):
        self._frame = v; self._render()

    def _on_col_changed(self, t):
        self.column_name = t; self._rebuild()

    def onDeleteWidget(self):
        self._timer.stop()
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWDiachronicNetwork).run()
