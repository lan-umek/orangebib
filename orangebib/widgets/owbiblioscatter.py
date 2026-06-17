# -*- coding: utf-8 -*-
"""
Bibliometric Plot Widget
========================
Visualise an **entity-statistics table** (the output of the *Bibliometric
Statistics* widget: one row per author / source / keyword / … with numeric
indicators such as Number of documents, Total citations, H-index, Average year).

Modes
-----
* **Bar chart** – entities sorted by a chosen statistic, coloured by another.
* **Scatter**   – x / y / bubble size / colour / label, robust optional log axes.
* **Linear projection** – PCA of several indicators onto 2-D.

Click a point / bar to send the corresponding rows onward.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QGridLayout,
                             QApplication)

import pyqtgraph as pg
from pyqtgraph import AxisItem

from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

_NONE = "(none)"
MODES = ["Bar chart", "Scatter", "Linear projection"]
PALETTE = ["#4a90d9", "#e67e22", "#27ae60", "#9b59b6", "#e74c3c",
           "#16a085", "#f39c12", "#2c3e50", "#c0392b", "#7f8c8d"]
COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis",
             "coolwarm", "turbo", "Spectral", "RdYlBu"]


def _cmap_colors(t_values, name):
    """Map an iterable of 0..1 floats to QColors via a matplotlib colormap."""
    try:
        import matplotlib
        try:
            cmap = matplotlib.colormaps[name]
        except Exception:  # noqa: BLE001 (older matplotlib)
            from matplotlib import cm
            cmap = cm.get_cmap(name)
        out = []
        for t in t_values:
            r, g, b, _a = cmap(float(max(0.0, min(1.0, t))))
            out.append(QColor(int(r * 255), int(g * 255), int(b * 255)))
        return out
    except Exception:  # noqa: BLE001 (matplotlib missing) – blue→red ramp
        out = []
        for t in t_values:
            t = float(max(0.0, min(1.0, t)))
            out.append(QColor(int(40 + 200 * t),
                              int(60 + 80 * (1 - abs(0.5 - t) * 2)),
                              int(220 - 180 * t)))
        return out


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


class LogAxis(AxisItem):
    """Axis that, in log mode, prints 10^v labels for log10-transformed data."""
    log10_mode = False

    def tickStrings(self, values, scale, spacing):
        if not self.log10_mode:
            return super().tickStrings(values, scale, spacing)
        out = []
        for v in values:
            try:
                if abs(v - round(v)) < 1e-6:
                    out.append(f"10^{int(round(v))}")
                else:
                    out.append(f"{10 ** v:.3g}")
            except Exception:  # noqa: BLE001
                out.append("")
        return out


class PlotGraph(pg.PlotWidget):
    def __init__(self, master):
        self._bottom = LogAxis(orientation="bottom")
        self._left = LogAxis(orientation="left")
        super().__init__(background="w",
                         axisItems={"bottom": self._bottom, "left": self._left})
        self.master = master
        self.getPlotItem().showGrid(x=False, y=False)
        self._scatter = pg.ScatterPlotItem(hoverable=True, tip=None)
        self._scatter.sigClicked.connect(self._clicked)
        self._bars = None
        self._tip = pg.TextItem(color="k", anchor=(0, 1),
                                fill=pg.mkBrush(255, 255, 220, 235))
        self._tip.setZValue(100); self._tip.hide()
        self._points = []          # parallel to scatter spots
        self._selected = set()
        self.scene().sigMouseMoved.connect(self._moved)
        self.scene().sigMouseClicked.connect(self._scene_clicked)

    def clear_all(self):
        self.getPlotItem().clear()
        self._points = []; self._selected = set(); self._bars = None
        self._bottom.log10_mode = False; self._left.log10_mode = False
        self.getViewBox().invertY(False)
        self.getAxis("left").setTicks(None)      # restore automatic ticks
        self.getAxis("bottom").setTicks(None)

    def _moved(self, pos):
        if not self._points or self._bars is not None:
            return
        vb = self.getPlotItem().vb
        if not self.sceneBoundingRect().contains(pos):
            self._tip.hide(); return
        pts = self._scatter.pointsAt(vb.mapSceneToView(pos))
        if len(pts):
            i = pts[0].data()
            self._tip.setText(self._points[i].get("tip", ""))
            mp = vb.mapSceneToView(pos)
            self._tip.setPos(mp.x(), mp.y()); self._tip.show()
        else:
            self._tip.hide()

    def _select(self, i):
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl:
            self._selected ^= {i}
        else:
            self._selected = set() if self._selected == {i} else {i}
        self.master.on_selection([self._points[j]["idx"] for j in self._selected])
        self.master.highlight()

    def _clicked(self, _s, pts):           # scatter point click
        if len(pts):
            self._select(pts[0].data())

    def _scene_clicked(self, ev):          # bar click (by y row)
        if self._bars is None or not self._points:
            return
        vb = self.getPlotItem().vb
        mp = vb.mapSceneToView(ev.scenePos())
        row = int(round(mp.y()))
        for k, p in enumerate(self._points):
            if p.get("yrow") == row:
                self._select(k); return


class OWBiblioScatter(OWWidget):
    """Plot an entity-statistics table (bar / scatter / linear projection)."""

    name = "Performance Plot"
    description = "Performance plot (bar / scatter / projection) of an entity-statistics table"
    icon = "icons/biblio_scatter.svg"
    priority = 130
    keywords = ["performance", "scatter", "bar", "plot", "statistics",
                "entities", "log", "projection", "bubble", "ranking"]
    category = "Biblium"

    class Inputs:
        statistics = Input("Statistics", Table,
                           doc="Entity statistics (from Bibliometric Statistics)")

    class Outputs:
        selected = Output("Selected", Table, doc="Selected entities")

    mode = settings.Setting(0)
    x_col = settings.Setting("")
    y_col = settings.Setting("")
    size_col = settings.Setting(_NONE)
    color_col = settings.Setting(_NONE)
    label_col = settings.Setting(_NONE)
    log_x = settings.Setting(False)
    log_y = settings.Setting(False)
    show_labels = settings.Setting(True)
    base_size = settings.Setting(12)
    top_n = settings.Setting(25)
    colormap = settings.Setting("viridis")
    rank_col = settings.Setting("")
    limit_points = settings.Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input")
        need_numeric = Msg("The input has no numeric columns to plot")

    class Warning(OWWidget.Warning):
        looks_like_raw = Msg("This looks like a raw document table, not entity "
                             "statistics. Connect the Bibliometric Statistics output "
                             "for meaningful results.")

    def __init__(self):
        super().__init__()
        self._df = None
        self._data = None
        self._num_cols = []

        box = gui.widgetBox(self.controlArea, "Mode")
        gui.comboBox(box, self, "mode", items=MODES, callback=self._mode_changed,
                     sendSelectedValue=False)

        ax = gui.widgetBox(self.controlArea, "Axes / statistics")
        g = QGridLayout()
        self.x_lbl = QLabel("X:"); g.addWidget(self.x_lbl, 0, 0)
        self.x_combo = QComboBox()
        self.x_combo.currentTextChanged.connect(lambda t: self._set("x_col", t))
        g.addWidget(self.x_combo, 0, 1)
        self.y_lbl = QLabel("Y:"); g.addWidget(self.y_lbl, 1, 0)
        self.y_combo = QComboBox()
        self.y_combo.currentTextChanged.connect(lambda t: self._set("y_col", t))
        g.addWidget(self.y_combo, 1, 1)
        ax.layout().addLayout(g)
        self.logx_cb = gui.checkBox(ax, self, "log_x", "Log X", callback=self._replot)
        self.logy_cb = gui.checkBox(ax, self, "log_y", "Log Y", callback=self._replot)

        enc = gui.widgetBox(self.controlArea, "Encoding")
        g2 = QGridLayout()
        self.size_lbl = QLabel("Size:"); g2.addWidget(self.size_lbl, 0, 0)
        self.size_combo = QComboBox()
        self.size_combo.currentTextChanged.connect(lambda t: self._set("size_col", t))
        g2.addWidget(self.size_combo, 0, 1)
        g2.addWidget(QLabel("Colour:"), 1, 0)
        self.color_combo = QComboBox()
        self.color_combo.currentTextChanged.connect(lambda t: self._set("color_col", t))
        g2.addWidget(self.color_combo, 1, 1)
        g2.addWidget(QLabel("Colormap:"), 2, 0)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText(self.colormap if self.colormap in COLORMAPS
                                       else "viridis")
        self.cmap_combo.currentTextChanged.connect(lambda t: self._set("colormap", t))
        g2.addWidget(self.cmap_combo, 2, 1)
        self.label_lbl = QLabel("Label:"); g2.addWidget(self.label_lbl, 3, 0)
        self.label_combo = QComboBox()
        self.label_combo.currentTextChanged.connect(lambda t: self._set("label_col", t))
        g2.addWidget(self.label_combo, 3, 1)
        enc.layout().addLayout(g2)
        self.labels_cb = gui.checkBox(enc, self, "show_labels", "Show point labels",
                                      callback=self._replot)
        self.basesize_spin = gui.spin(enc, self, "base_size", 4, 40,
                                      label="Base size:", callback=self._replot)
        self.topn_spin = gui.spin(enc, self, "top_n", 5, 1000,
                                  label="Max entities:", callback=self._replot)
        self.limit_cb = gui.checkBox(enc, self, "limit_points",
                                     "Limit number of points", callback=self._replot)
        g3 = QGridLayout()
        self.rank_lbl = QLabel("Rank by:"); g3.addWidget(self.rank_lbl, 0, 0)
        self.rank_combo = QComboBox()
        self.rank_combo.currentTextChanged.connect(lambda t: self._set("rank_col", t))
        g3.addWidget(self.rank_combo, 0, 1)
        enc.layout().addLayout(g3)

        self.plot_btn = QPushButton("Plot")
        self.plot_btn.clicked.connect(self._replot)
        self.controlArea.layout().addWidget(self.plot_btn)
        self.status = QLabel(""); self.status.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status)
        self.controlArea.layout().addStretch(1)

        self.graph = PlotGraph(self)
        self.mainArea.layout().addWidget(self.graph)
        self._update_controls()

    # ----------------------------------------------------------------- helpers
    def _set(self, attr, t):
        if t == "":
            return
        setattr(self, attr, t)
        self._replot()

    def _mode_changed(self):
        self._update_controls()
        self._replot()

    def _update_controls(self):
        """Show only the controls that make sense for the current mode."""
        is_scatter = self.mode == 1
        is_bar = self.mode == 0
        is_proj = self.mode == 2
        for w in (self.x_lbl, self.x_combo, self.logx_cb, self.logy_cb):
            w.setVisible(is_scatter)
        self.y_lbl.setText("Y:" if is_scatter else "Sort / value:")
        for w in (self.y_lbl, self.y_combo):
            w.setVisible(not is_proj)
        for w in (self.size_lbl, self.size_combo, self.basesize_spin):
            w.setVisible(not is_bar)
        self.labels_cb.setVisible(not is_bar)
        self.topn_spin.setVisible(True)
        for w in (self.limit_cb, self.rank_lbl, self.rank_combo):
            w.setVisible(not is_bar)

    # ----------------------------------------------------------------- input
    @Inputs.statistics
    def set_data(self, data):
        self.Error.clear(); self.Warning.clear()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        if data is None:
            self.Error.no_data(); self.graph.clear_all(); return
        self._num_cols = [c for c in self._df.columns
                          if pd.api.types.is_numeric_dtype(self._df[c])]
        if not self._num_cols:
            self.Error.need_numeric(); self.graph.clear_all(); return
        agg_hint = any(k in " ".join(self._df.columns).lower()
                       for k in ("number of documents", "total citations",
                                 "h-index", "h_index", "average year"))
        if len(self._df) > 50 and not agg_hint:
            self.Warning.looks_like_raw()
        self._apply_defaults()
        self._fill_combos()
        self._update_controls()
        self._replot()

    def _apply_defaults(self):
        cols = list(self._df.columns)
        low = {c.lower(): c for c in cols}

        def pick(*cands):
            for c in cands:
                if c.lower() in low:
                    return low[c.lower()]
            for c in cols:
                if any(k in c.lower() for k in cands):
                    return c
            return None
        docs = pick("number of documents", "documents", "frequency", "n_docs")
        cites = pick("total citations", "citations")
        hidx = pick("h-index", "h_index")
        ayear = pick("average year", "mean year")
        ent = self._entity_col()
        if self.x_col not in self._num_cols:
            self.x_col = docs or self._num_cols[0]
        if self.y_col not in self._num_cols:
            self.y_col = cites or (self._num_cols[1] if len(self._num_cols) > 1
                                   else self._num_cols[0])
        if self.size_col not in ([_NONE] + self._num_cols):
            self.size_col = hidx or _NONE
        if self.color_col not in ([_NONE] + cols):
            self.color_col = ayear or _NONE
        if self.label_col not in ([_NONE] + cols):
            self.label_col = ent or _NONE
        if self.rank_col not in self._num_cols:
            self.rank_col = docs or self.y_col or self._num_cols[0]

    def _entity_col(self):
        """First non-numeric column = the entity key (author / keyword / …)."""
        for c in self._df.columns:
            if c not in self._num_cols:
                return c
        return None

    def _fill_combos(self):
        allc = list(self._df.columns)
        self._fill(self.x_combo, self._num_cols, self.x_col)
        self._fill(self.y_combo, self._num_cols, self.y_col)
        self._fill(self.size_combo, [_NONE] + self._num_cols, self.size_col)
        self._fill(self.color_combo, [_NONE] + allc, self.color_col)
        self._fill(self.label_combo, [_NONE] + allc, self.label_col)
        self._fill(self.rank_combo, self._num_cols, self.rank_col)

    @staticmethod
    def _fill(combo, items, current):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([str(i) for i in items])
        if current in items:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _num(self, col):
        return pd.to_numeric(self._df[col], errors="coerce")

    def _label_source(self):
        if self.label_col not in (_NONE, "") and self.label_col in self._df.columns:
            return self.label_col
        return self._entity_col()

    def _ent_label(self, i):
        src = self._label_source()
        if src is not None:
            return str(self._df.iloc[i][src])
        return str(i)

    def _colors_for(self, idx, color_col):
        n = len(idx)
        if color_col in (_NONE, "") or color_col not in self._df.columns:
            return [QColor("#4a90d9")] * n
        if pd.api.types.is_numeric_dtype(self._df[color_col]):
            v = self._num(color_col).iloc[idx].fillna(0).values.astype(float)
            lo, hi = (v.min(), v.max()) if len(v) else (0.0, 1.0)
            rng = (hi - lo) or 1.0
            return _cmap_colors([(x - lo) / rng for x in v], self.colormap)
        cats = self._df[color_col].iloc[idx].astype(str).tolist()
        uniq = {c: i for i, c in enumerate(dict.fromkeys(cats))}
        return [QColor(PALETTE[uniq[c] % len(PALETTE)]) for c in cats]

    def _sizes_for(self, idx, size_col):
        n = len(idx)
        if size_col in (_NONE, "") or size_col not in self._df.columns:
            return [float(self.base_size)] * n
        v = self._num(size_col).iloc[idx].fillna(0).values.astype(float)
        vmax = v.max() if len(v) else 1.0
        return [self.base_size * 0.6 + self.base_size * 2.0 * (x / vmax if vmax else 0)
                for x in v]

    def _limit_idx(self, idx):
        """Keep only the top-N entities by the chosen ranking column."""
        if not self.limit_points or len(idx) <= self.top_n:
            return idx
        rc = self.rank_col if self.rank_col in self._num_cols else None
        if rc is None:
            return idx[:self.top_n]
        vals = self._num(rc)
        return sorted(idx, key=lambda i: (float(vals.iloc[i])
                      if pd.notna(vals.iloc[i]) else -1e18),
                      reverse=True)[:self.top_n]

    # ----------------------------------------------------------------- plot
    def highlight(self):
        if self._df is not None:
            self._replot(keep_selection=True)

    def _replot(self, keep_selection=False):
        if self._df is None or self._df.empty or not self._num_cols:
            return
        sel = set(self.graph._selected) if keep_selection else set()
        self.graph.clear_all()
        self.graph._selected = sel
        if self.mode == 0:
            self._plot_bar()
        elif self.mode == 2:
            self._plot_projection()
        else:
            self._plot_scatter()

    # ---- scatter --------------------------------------------------------
    def _plot_scatter(self):
        x, y = self.x_col, self.y_col
        if x not in self._df.columns or y not in self._df.columns:
            return
        xs, ys = self._num(x), self._num(y)
        valid = xs.notna() & ys.notna()
        if self.log_x:
            valid &= xs > 0
        if self.log_y:
            valid &= ys > 0
        idx = [i for i in range(len(self._df)) if bool(valid.iloc[i])]
        if not idx:
            self.status.setText("No positive values for the chosen log axes.")
            return
        idx = self._limit_idx(idx)
        xv = xs.iloc[idx].values.astype(float)
        yv = ys.iloc[idx].values.astype(float)
        self.graph._bottom.log10_mode = self.log_x
        self.graph._left.log10_mode = self.log_y
        px = np.log10(xv) if self.log_x else xv
        py = np.log10(yv) if self.log_y else yv
        sizes = self._sizes_for(idx, self.size_col)
        colors = self._colors_for(idx, self.color_col)
        spots, points = [], []
        for k, i in enumerate(idx):
            sel = k in self.graph._selected
            spots.append({"pos": (float(px[k]), float(py[k])), "data": k,
                          "size": sizes[k], "brush": pg.mkBrush(colors[k]),
                          "pen": pg.mkPen("k", width=2 if sel else 0.4)})
            points.append({"idx": i, "tip": f"{self._ent_label(i)}\n"
                           f"{x}: {xv[k]:.4g}\n{y}: {yv[k]:.4g}"})
        self.graph._scatter.setData(spots)
        self.graph.getPlotItem().addItem(self.graph._scatter)
        self.graph.getPlotItem().addItem(self.graph._tip)
        self.graph._points = points
        if self.show_labels and self._label_source() is not None:
            for k, i in enumerate(idx):
                t = pg.TextItem(self._ent_label(i)[:24], color=(70, 70, 70),
                                anchor=(0, 0.5))
                t.setPos(float(px[k]), float(py[k]))
                self.graph.getPlotItem().addItem(t)
        self.graph.setLabel("bottom", x)
        self.graph.setLabel("left", y)
        self.graph.getViewBox().autoRange()
        self.status.setText(f"{len(idx)} entities shown")

    # ---- bar ------------------------------------------------------------
    def _plot_bar(self):
        y = self.y_col if self.y_col in self._num_cols else self._num_cols[0]
        s = self._num(y).fillna(0)
        order = list(s.sort_values(ascending=False, kind="mergesort").index)[:self.top_n]
        if not order:
            return
        vals = [float(s.loc[i]) for i in order]
        n = len(order)
        ypos = [n - 1 - k for k in range(n)]          # largest on top
        colors = self._colors_for(order, self.color_col)
        brushes = []
        for k in range(n):
            c = QColor("#ff7f0e") if k in self.graph._selected else colors[k]
            brushes.append(pg.mkBrush(c))
        self.graph._bars = pg.BarGraphItem(x0=0, y=ypos, height=0.65, width=vals,
                                           brushes=brushes,
                                           pen=pg.mkPen("k", width=0.4))
        self.graph.getPlotItem().addItem(self.graph._bars)
        labels = [self._ent_label(i)[:34] for i in order]
        self.graph.getAxis("left").setTicks([[(ypos[k], labels[k]) for k in range(n)]])
        self.graph.getAxis("left").log10_mode = False
        self.graph.setLabel("bottom", y)
        self.graph.setLabel("left", self._label_source() or "Entity")
        vmax = max(vals) if vals else 1.0
        self.graph.setYRange(-0.5, n - 0.5)
        self.graph.setXRange(0, vmax * 1.05 if vmax > 0 else 1.0)
        self.graph._points = [{"idx": order[k], "yrow": ypos[k],
                               "tip": labels[k]} for k in range(n)]
        self.status.setText(f"{n} entities sorted by {y}  (click a bar to select)")

    # ---- projection -----------------------------------------------------
    def _plot_projection(self):
        cols = self._num_cols
        if len(cols) < 2:
            self.status.setText("Need ≥2 numeric columns for a projection.")
            return
        X = self._df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        try:
            Zc = Z - Z.mean(0)
            U, S, _Vt = np.linalg.svd(Zc, full_matrices=False)
            comp = U[:, :2] * S[:2]
        except Exception:  # noqa: BLE001
            self.status.setText("Projection failed.")
            return
        idx = self._limit_idx(list(range(len(self._df))))
        sizes = self._sizes_for(idx, self.size_col)
        colors = self._colors_for(idx, self.color_col)
        spots, points = [], []
        for k in idx:
            sel = k in self.graph._selected
            spots.append({"pos": (float(comp[k, 0]), float(comp[k, 1])), "data": k,
                          "size": sizes[k], "brush": pg.mkBrush(colors[k]),
                          "pen": pg.mkPen("k", width=2 if sel else 0.4)})
            points.append({"idx": k, "tip": self._ent_label(k)})
        self.graph._scatter.setData(spots)
        self.graph.getPlotItem().addItem(self.graph._scatter)
        self.graph.getPlotItem().addItem(self.graph._tip)
        self.graph._points = points
        if self.show_labels and self._label_source() is not None:
            for k in idx:
                t = pg.TextItem(self._ent_label(k)[:20], color=(70, 70, 70),
                                anchor=(0, 0.5))
                t.setPos(float(comp[k, 0]), float(comp[k, 1]))
                self.graph.getPlotItem().addItem(t)
        self.graph.setLabel("bottom", "PC1")
        self.graph.setLabel("left", "PC2")
        self.graph.getViewBox().autoRange()
        var = (S ** 2) / (S ** 2).sum() if S.sum() else np.zeros_like(S)
        self.status.setText(f"PCA of {len(cols)} indicators — "
                            f"PC1 {var[0]*100:.0f}%, PC2 {var[1]*100:.0f}%")

    # ----------------------------------------------------------------- output
    def on_selection(self, original_indices):
        if self._data is None or not original_indices:
            self.Outputs.selected.send(None); return
        idx = [i for i in original_indices if 0 <= i < len(self._data)]
        self.Outputs.selected.send(self._data[idx] if idx else None)


if __name__ == "__main__":
    WidgetPreview(OWBiblioScatter).run()
