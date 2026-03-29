# -*- coding: utf-8 -*-
"""
Group Intersections Widget
==========================
Orange widget for analyzing document overlap between groups.

Inputs:
    Data: Bibliographic data with "Group: xxx" columns (from Setup Groups)

Outputs:
    Intersections: Table of intersection statistics
    Selected Data: Documents in the selected intersection(s)
"""

import logging
from itertools import combinations
from typing import Optional, List, Set

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton,
    QCheckBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QRadioButton, QButtonGroup,
    QAbstractItemView, QTabWidget, QFrame,
    QFileDialog, QDoubleSpinBox,
)
from AnyQt.QtCore import Qt, pyqtSignal

from Orange.data import (
    Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable,
)
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.settings import Setting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
try:
    from biblium import utilsbib
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    utilsbib = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle as MplCircle, Rectangle as MplRect, Ellipse as MplEllipse
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
GROUP_PREFIX = "Group: "
VIZ_TYPES = [
    ("venn",       "Venn Diagram"),
    ("upset",      "UpSet Plot"),
    ("heatmap",    "Heatmap"),
    ("network",    "Network"),
    ("dendrogram", "Dendrogram"),
]
PALETTE = [
    "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
]
HIGHLIGHT_COLOR = "#ef4444"


# =============================================================================
# Clickable Matplotlib canvas — passes (x, y, ctrl_held)
# =============================================================================

class ClickableCanvas(FigureCanvas):
    """Matplotlib canvas that emits click data-coordinates + modifier state."""
    # x, y in data coords, ctrl_held bool
    clicked = pyqtSignal(float, float, bool)

    def __init__(self, fig, parent=None):
        super().__init__(fig)
        self.setParent(parent)
        self.mpl_connect("button_press_event", self._on_press)

    def _on_press(self, event):
        if event.inaxes and event.button == 1:
            # Use Qt modifiers — reliable regardless of canvas focus
            from AnyQt.QtWidgets import QApplication
            mods = QApplication.queryKeyboardModifiers()
            ctrl = bool(mods & Qt.ControlModifier)
            self.clicked.emit(event.xdata, event.ydata, ctrl)


# =============================================================================
# WIDGET
# =============================================================================

class OWGroupIntersections(OWWidget):
    """Analyze document overlap between groups."""

    name = "Group Intersections"
    description = "Analyze document overlap between groups"
    icon = "icons/group_intersections.svg"
    priority = 34
    keywords = ["group", "intersection", "overlap", "venn", "upset",
                "heatmap", "network", "dendrogram"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        intersections = Output("Intersections", Table)
        selected_data = Output("Selected Data", Table)

    include_ids = Setting(False)
    id_col_name = Setting("Doc ID")
    viz_type_idx = Setting(0)
    similarity_method = Setting("jaccard")
    threshold = Setting(0.1)
    auto_apply = Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_groups = Msg("No group columns found — connect Setup Groups first")
        compute_error = Msg("Computation error: {}")

    class Warning(OWWidget.Warning):
        empty_result = Msg("No intersections found")
        viz_error = Msg("Visualization error: {}")

    class Information(OWWidget.Information):
        groups_found = Msg("Found {} groups: {}")
        results = Msg("{} intersections across {} documents (largest: {})")

    def __init__(self):
        super().__init__()
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._gm_df: Optional[pd.DataFrame] = None
        self._group_names: List[str] = []
        self._intersections_df: Optional[pd.DataFrame] = None
        self._selected_rows: Set[int] = set()

        # Visualisation bookkeeping
        self._current_fig: Optional[Figure] = None
        self._current_canvas: Optional[ClickableCanvas] = None
        self._highlights: list = []          # active highlight patches
        self._upset_patterns: list = []
        self._upset_bars = None
        self._venn_hit_tests: list = []      # (group_name, callable(x,y)->bool)
        self._venn_labels: list = []         # (text_obj, frozenset_of_groups)

        self._setup_control_area()
        self._setup_main_area()

    # =================================================================
    # UI
    # =================================================================

    def _setup_control_area(self):
        opts = gui.widgetBox(self.controlArea, "📋 Analysis Options")
        cb = QCheckBox("Include document IDs in results")
        cb.setChecked(self.include_ids)
        cb.toggled.connect(lambda v: setattr(self, "include_ids", v))
        opts.layout().addWidget(cb)

        row = QHBoxLayout()
        row.addWidget(QLabel("ID Column:"))
        self.id_col_combo = QComboBox()
        self.id_col_combo.addItem("Doc ID")
        self.id_col_combo.setEditable(True)
        row.addWidget(self.id_col_combo)
        opts.layout().addLayout(row)

        viz = gui.widgetBox(self.controlArea, "📊 Visualization Type")
        self.viz_group = QButtonGroup()
        for i, (_k, label) in enumerate(VIZ_TYPES):
            rb = QRadioButton(label)
            self.viz_group.addButton(rb, i)
            viz.layout().addWidget(rb)
            if i == self.viz_type_idx:
                rb.setChecked(True)
        self.viz_group.idClicked.connect(self._on_viz_type_changed)

        sep = QLabel("Network / Heatmap Options:")
        sep.setStyleSheet("color:#888;font-size:11px;margin-top:6px;")
        viz.layout().addWidget(sep)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Similarity:"))
        self.sim_combo = QComboBox()
        for m in ("jaccard", "count", "dice", "overlap"):
            self.sim_combo.addItem(m)
        idx = self.sim_combo.findText(self.similarity_method)
        if idx >= 0:
            self.sim_combo.setCurrentIndex(idx)
        self.sim_combo.currentTextChanged.connect(
            lambda t: setattr(self, "similarity_method", t))
        r1.addWidget(self.sim_combo)
        viz.layout().addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Threshold:"))
        sp = QDoubleSpinBox()
        sp.setRange(0.0, 1.0); sp.setSingleStep(0.05); sp.setDecimals(2)
        sp.setValue(self.threshold)
        sp.valueChanged.connect(lambda v: setattr(self, "threshold", v))
        r2.addWidget(sp)
        viz.layout().addLayout(r2)

        btn = QPushButton("📊 Analyze Intersections")
        btn.clicked.connect(self._run_analysis)
        btn.setStyleSheet(
            "QPushButton{background:#2563eb;border:none;border-radius:4px;"
            "padding:10px 20px;color:white;font-weight:bold;font-size:13px}"
            "QPushButton:hover{background:#1d4ed8}"
            "QPushButton:disabled{background:#ccc}")
        self.controlArea.layout().addWidget(btn)

        exp = QPushButton("Export Results")
        exp.clicked.connect(self._export_results)
        exp.setStyleSheet(
            "QPushButton{background:transparent;border:none;"
            "color:#4338ca;font-weight:bold;padding:4px}"
            "QPushButton:hover{color:#312e81}")
        self.controlArea.layout().addWidget(exp)
        self.controlArea.layout().addStretch()

    def _setup_main_area(self):
        lo = QVBoxLayout()

        hdr = QLabel("Results")
        hdr.setStyleSheet("font-size:16px;font-weight:bold;")
        lo.addWidget(hdr)

        cf = QFrame()
        cf.setStyleSheet("QFrame{background:transparent}")
        cl = QHBoxLayout(cf); cl.setContentsMargins(0, 0, 0, 0)
        self.card_int = self._card("📊", "0", "Intersections", "#2563eb")
        self.card_doc = self._card("📄", "0", "Documents", "#6b7280")
        self.card_lrg = self._card("📈", "0", "Largest", "#6b7280")
        cl.addWidget(self.card_int)
        cl.addWidget(self.card_doc)
        cl.addWidget(self.card_lrg)
        lo.addWidget(cf)

        self.tabs = QTabWidget()

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.tabs.addTab(self.table, "📋 Table")

        self.viz_w = QWidget()
        self.viz_lo = QVBoxLayout(self.viz_w)
        self.viz_lo.setContentsMargins(0, 0, 0, 0)
        self.tabs.addTab(self.viz_w, "🔵 Visualization")

        self.info_lbl = QLabel("Run analysis to see results.")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_lbl.setStyleSheet("padding:12px;font-size:12px;")
        self.tabs.addTab(self.info_lbl, "ℹ️ Info")

        lo.addWidget(self.tabs)
        w = QWidget(); w.setLayout(lo)
        self.mainArea.layout().addWidget(w)

    # helpers
    def _card(self, icon, val, label, color):
        f = QFrame()
        f.setStyleSheet(
            "QFrame{background:white;border:1px solid #e5e7eb;"
            "border-radius:6px;padding:8px 14px}")
        ly = QVBoxLayout(f); ly.setContentsMargins(8, 6, 8, 6)
        v = QLabel(f"{icon} {val}")
        v.setStyleSheet(f"font-size:18px;font-weight:bold;color:{color};")
        v.setObjectName("cv"); ly.addWidget(v)
        ly.addWidget(QLabel(label))
        return f

    def _set_card(self, card, val):
        lbl = card.findChild(QLabel, "cv")
        if lbl:
            lbl.setText(f"{lbl.text().split(' ',1)[0]} {val}")

    # =================================================================
    # EVENTS
    # =================================================================

    def _on_viz_type_changed(self, idx):
        self.viz_type_idx = idx
        if self._intersections_df is not None:
            self._display_viz()

    # =================================================================
    # INPUT
    # =================================================================

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear(); self.Warning.clear(); self.Information.clear()
        self._data = data
        self._df = self._gm_df = self._intersections_df = None
        self._group_names = []; self._selected_rows = set()

        if data is None:
            self._clear(); return
        self._df = self._table_to_df(data)
        self._extract_groups()

        self.id_col_combo.clear()
        for c in self._df.columns:
            if not c.startswith(GROUP_PREFIX):
                self.id_col_combo.addItem(c)
        idx = self.id_col_combo.findText(self.id_col_name)
        if idx >= 0:
            self.id_col_combo.setCurrentIndex(idx)

        if self.auto_apply and self._gm_df is not None:
            self._run_analysis()

    def _extract_groups(self):
        if self._df is None:
            return
        gcols = [c for c in self._df.columns if c.startswith(GROUP_PREFIX)]
        if not gcols:
            self.Error.no_groups(); return
        gm = pd.DataFrame(index=self._df.index)
        names = []
        for c in gcols:
            clean = c[len(GROUP_PREFIX):]
            gm[clean] = self._to_binary(self._df[c])
            names.append(clean)
        self._gm_df = gm; self._group_names = names
        prev = ", ".join(names[:5])
        if len(names) > 5:
            prev += f" (+{len(names)-5})"
        self.Information.groups_found(len(names), prev)

    # =================================================================
    # COMPUTE
    # =================================================================

    def _run_analysis(self):
        if self._df is None or self._gm_df is None:
            return
        self.Error.clear(); self.Warning.clear()
        try:
            idf = self._compute()
        except Exception as e:
            self.Error.compute_error(str(e)); return
        if idf is None or idf.empty:
            self.Warning.empty_result(); self._clear(); return

        self._intersections_df = idf
        self._fill_table(idf)
        self._display_viz()
        self._fill_info(idf)
        self._send()

        n_i = len(idf)
        n_d = int(idf["Size"].sum())
        mx = int(idf["Size"].max())
        self._set_card(self.card_int, str(n_i))
        self._set_card(self.card_doc, str(n_d))
        self._set_card(self.card_lrg, str(mx))
        self.Information.results(n_i, n_d, mx)

    def _compute(self):
        gm = self._gm_df
        id_name = self.id_col_combo.currentText()
        ids = (self._df[id_name]
               if self.include_ids and id_name in self._df.columns else None)

        if HAS_BIBLIUM:
            res = utilsbib.compute_group_intersections(
                gm, include_ids=self.include_ids, id_column=ids)
        else:
            res = self._compute_fb(gm, ids)

        if "Groups" in res.columns:
            res["_groups_tuple"] = res["Groups"]
            res["Groups"] = res["Groups"].apply(
                lambda t: " ∩ ".join(t) if isinstance(t, tuple) else str(t))
            res["N Groups"] = res["_groups_tuple"].apply(
                lambda t: len(t) if isinstance(t, tuple) else 1)
        return res

    def _compute_fb(self, gm, ids=None):
        cols = gm.columns.tolist(); rows = []
        for mask, sub in gm.groupby(cols):
            if not isinstance(mask, tuple):
                mask = (mask,)
            active = tuple(c for c, f in zip(cols, mask) if f == 1)
            if not active:
                continue
            r = {"Groups": active, "Size": len(sub)}
            if self.include_ids and ids is not None:
                r["ID"] = ids.loc[sub.index].tolist()
            rows.append(r)
        return pd.DataFrame(rows).sort_values(
            "Size", ascending=False).reset_index(drop=True)

    # =================================================================
    # TABLE
    # =================================================================

    def _fill_table(self, df):
        self.table.setSortingEnabled(False)
        self.table.clear()
        show = [c for c in df.columns if not c.startswith("_")]
        nr, nc = len(df), len(show)
        self.table.setRowCount(nr); self.table.setColumnCount(nc)
        self.table.setHorizontalHeaderLabels(show)
        for r in range(nr):
            for ci, col in enumerate(show):
                v = df.iloc[r][col]
                if isinstance(v, (list, tuple)):
                    t = "; ".join(str(x) for x in v)
                elif isinstance(v, float):
                    t = f"{v:.0f}" if v == int(v) else f"{v:.4f}"
                else:
                    t = str(v)
                it = QTableWidgetItem(t)
                if isinstance(v, (int, float, np.integer, np.floating)):
                    it.setData(Qt.UserRole, float(v))
                self.table.setItem(r, ci, it)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

        sm = self.table.selectionModel()
        if sm:
            try:
                sm.selectionChanged.disconnect()
            except Exception:
                pass
            sm.selectionChanged.connect(self._on_table_sel)

    def _on_table_sel(self):
        rows = {i.row() for i in self.table.selectionModel().selectedRows()}
        if rows == self._selected_rows:
            return
        self._selected_rows = rows
        self._emit_selection()

    # =================================================================
    # SELECTION HELPERS
    # =================================================================

    def _select_table_by_groups(self, target_set: set, *,
                                containing=False, additive=False):
        """Select table rows matching *target_set*.

        * containing=False → exact match; True → any overlap
        * additive=True    → add to existing selection (Ctrl+Click)
        * Does NOT switch tabs.
        """
        from AnyQt.QtCore import QItemSelectionModel
        sm = self.table.selectionModel()
        if sm is None:
            return
        if not additive:
            self.table.clearSelection()
        model = self.table.model()
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if not it:
                continue
            row_groups = set(it.text().split(" ∩ "))
            hit = (target_set & row_groups) if containing else (row_groups == target_set)
            if hit:
                idx = model.index(r, 0)
                sm.select(idx,
                          QItemSelectionModel.Select | QItemSelectionModel.Rows)

    def _emit_selection(self):
        if not self._selected_rows or self._intersections_df is None:
            self.Outputs.selected_data.send(None); return
        gm = self._gm_df; all_idx: Set[int] = set()
        for ri in self._selected_rows:
            if ri >= len(self._intersections_df):
                continue
            gt = self._intersections_df.iloc[ri].get("_groups_tuple")
            if gt is None:
                continue
            mask = pd.Series(True, index=gm.index)
            for g in self._group_names:
                mask &= (gm[g] == 1) if g in gt else (gm[g] == 0)
            all_idx.update(gm.index[mask].tolist())
        if all_idx and self._data is not None:
            valid = sorted(i for i in all_idx if i < len(self._data))
            if valid:
                self.Outputs.selected_data.send(self._data[valid]); return
        self.Outputs.selected_data.send(None)

    # =================================================================
    # HIGHLIGHT HELPERS
    # =================================================================

    def _clear_highlights(self):
        for h in self._highlights:
            try:
                h.remove()
            except Exception:
                pass
        self._highlights.clear()

    def _redraw(self):
        if self._current_canvas:
            self._current_canvas.draw_idle()

    # =================================================================
    # VISUALIZATION DISPATCH
    # =================================================================

    def _display_viz(self):
        while self.viz_lo.count():
            ch = self.viz_lo.takeAt(0)
            if ch.widget():
                ch.widget().deleteLater()
        self._highlights.clear()
        self._venn_hit_tests.clear()
        self._venn_labels.clear()

        if not HAS_MPL:
            self.viz_lo.addWidget(QLabel("matplotlib is required.")); return
        if self._gm_df is None:
            return

        key = VIZ_TYPES[min(self.viz_type_idx, len(VIZ_TYPES)-1)][0]
        try:
            {"venn": self._viz_venn, "upset": self._viz_upset,
             "heatmap": self._viz_heatmap, "network": self._viz_network,
             "dendrogram": self._viz_dendro}[key]()
        except Exception as e:
            logger.exception("viz")
            self.Warning.viz_error(str(e))
            self.viz_lo.addWidget(QLabel(f"⚠️ {e}"))

    def _embed(self, fig) -> ClickableCanvas:
        c = ClickableCanvas(fig, self.viz_w)
        c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viz_lo.addWidget(c)
        self._current_fig = fig; self._current_canvas = c
        return c


    # =================================================================
    # VENN  (built-in: circles for 2-3, ellipses for 4 groups)
    # =================================================================

    def _viz_venn(self):
        gm, groups = self._gm_df, self._group_names
        n = len(groups)
        if n < 2:
            self.viz_lo.addWidget(QLabel("Need ≥ 2 groups.")); return

        sets = {g: set(gm.index[gm[g].astype(bool)]) for g in groups}
        fig = Figure(figsize=(8, 7), dpi=100)
        ax = fig.add_subplot(111)

        self._venn_hit_tests = []
        self._venn_labels = []
        if n == 2:
            self._venn2(ax, groups, sets)
        elif n == 3:
            self._venn3(ax, groups, sets)
        elif n == 4:
            self._venn4(ax, groups, sets)
        elif n == 5:
            self._venn5(ax, groups, sets)
        else:
            self._venn_text(ax, groups, sets,
                            f"{n} groups — use UpSet plot for best results")

        ax.set_title("Group Overlap — Venn Diagram", fontsize=12)
        fig.tight_layout()
        canvas = self._embed(fig)

        # Click handler: detect region via hit-tests, highlight text label
        hit_tests = list(self._venn_hit_tests)
        venn_labels = list(self._venn_labels)
        selected_regions: set = set()

        BBOX_SEL = dict(boxstyle="round,pad=0.25", facecolor="#fde68a",
                        edgecolor="#f59e0b", alpha=0.85, lw=1.5)
        BBOX_NONE = dict(boxstyle="round,pad=0.0", facecolor="none",
                         edgecolor="none", alpha=0)

        def _update_label_highlights():
            for txt, fset in venn_labels:
                if fset in selected_regions:
                    txt.set_bbox(BBOX_SEL)
                    txt.set_fontsize(txt._orig_size + 2)
                else:
                    txt.set_bbox(BBOX_NONE)
                    txt.set_fontsize(txt._orig_size)
            self._redraw()

        def _click(xd, yd, ctrl):
            nonlocal selected_regions
            if not hit_tests:
                return
            hit_groups = frozenset(
                gn for gn, test in hit_tests if test(xd, yd))
            if not hit_groups:
                if not ctrl:
                    selected_regions = set()
                    self.table.clearSelection()
                    _update_label_highlights()
                return

            if ctrl:
                if hit_groups in selected_regions:
                    selected_regions.discard(hit_groups)
                else:
                    selected_regions.add(hit_groups)
            else:
                selected_regions = {hit_groups}

            _update_label_highlights()

            if not ctrl:
                self.table.clearSelection()
            self._select_table_by_groups(set(hit_groups), additive=ctrl)

        canvas.clicked.connect(_click)

    # ── helpers ──

    def _add_circle_hit(self, cx, cy, r, group_name):
        """Register a circle shape for click detection."""
        self._venn_hit_tests.append(
            (group_name,
             lambda x, y, _cx=cx, _cy=cy, _r=r:
                 (x - _cx)**2 + (y - _cy)**2 <= _r**2))

    @staticmethod
    def _point_in_ellipse(px, py, ecx, ecy, ew, eh, angle_deg):
        a_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)
        dx, dy = px - ecx, py - ecy
        rx =  cos_a * dx + sin_a * dy
        ry = -sin_a * dx + cos_a * dy
        return (rx / (ew / 2))**2 + (ry / (eh / 2))**2 <= 1.0

    def _add_ellipse_hit(self, ecx, ecy, ew, eh, angle, group_name):
        """Register an ellipse shape for click detection."""
        self._venn_hit_tests.append(
            (group_name,
             lambda x, y, _cx=ecx, _cy=ecy, _w=ew, _h=eh, _a=angle:
                 self._point_in_ellipse(x, y, _cx, _cy, _w, _h, _a)))

    def _add_venn_label(self, ax, x, y, value, group_set, fontsize=13):
        """Place a count label and register it for highlighting."""
        t = ax.text(x, y, str(value), ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", zorder=15)
        t._orig_size = fontsize
        self._venn_labels.append((t, frozenset(group_set)))

    @staticmethod
    def _repel_labels(label_data, min_sep=0.22, iterations=100,
                      spring_k=0.12):
        """Push overlapping label positions apart via force simulation.

        *label_data* – list of ``[x, y, count, group_set, fontsize]``
        Returns a new list with adjusted (x, y).
        """
        n = len(label_data)
        if n <= 1:
            return label_data

        pos = np.array([[d[0], d[1]] for d in label_data], dtype=float)
        origins = pos.copy()

        for _ in range(iterations):
            forces = np.zeros_like(pos)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.hypot(dx, dy)
                    if dist < min_sep:
                        if dist < 1e-8:
                            ang = i * 2.399  # golden angle offset
                            dx, dy = np.cos(ang), np.sin(ang)
                            dist = 1e-8
                        mag = (min_sep - dist) * 0.45
                        fx, fy = (dx / dist) * mag, (dy / dist) * mag
                        forces[i, 0] -= fx; forces[i, 1] -= fy
                        forces[j, 0] += fx; forces[j, 1] += fy
            # attract back toward true centroid
            forces += (origins - pos) * spring_k
            pos += forces

        return [[pos[k, 0], pos[k, 1], d[2], d[3], d[4]]
                for k, d in enumerate(label_data)]

    # ── 2-set Venn (circles) ──

    def _venn2(self, ax, groups, sets):
        g1, g2 = groups
        s1, s2 = sets[g1], sets[g2]
        r = 1.5
        cx = [-0.65, 0.65]; cy = [0, 0]

        for i, (x, y, g) in enumerate(zip(cx, cy, groups)):
            ax.add_patch(MplCircle((x, y), r, alpha=0.35,
                         facecolor=PALETTE[i], edgecolor="black", lw=1.5))
            self._add_circle_hit(x, y, r, g)

        self._add_venn_label(ax, cx[0]-0.55, 0, len(s1-s2), {g1}, 15)
        self._add_venn_label(ax, 0, 0, len(s1&s2), {g1, g2}, 15)
        self._add_venn_label(ax, cx[1]+0.55, 0, len(s2-s1), {g2}, 15)

        ax.text(cx[0], r+0.35, f"{g1}\n(n={len(s1)})",
                ha="center", fontsize=10, fontweight="bold")
        ax.text(cx[1], r+0.35, f"{g2}\n(n={len(s2)})",
                ha="center", fontsize=10, fontweight="bold")
        ax.set_xlim(-3, 3); ax.set_ylim(-2.3, 2.5)
        ax.set_aspect("equal"); ax.axis("off")

    # ── 3-set Venn (circles) ──

    def _venn3(self, ax, groups, sets):
        g1, g2, g3 = groups
        s1, s2, s3 = sets[g1], sets[g2], sets[g3]
        r = 1.5
        cx = [-0.70, 0.70, 0.00]
        cy = [ 0.45, 0.45, -0.55]

        for i in range(3):
            ax.add_patch(MplCircle((cx[i], cy[i]), r, alpha=0.30,
                         facecolor=PALETTE[i], edgecolor="black", lw=1.5))
            self._add_circle_hit(cx[i], cy[i], r, groups[i])

        self._add_venn_label(ax, cx[0]-0.55, cy[0]+0.45,
                             len(s1 - s2 - s3), {g1})
        self._add_venn_label(ax, cx[1]+0.55, cy[1]+0.45,
                             len(s2 - s1 - s3), {g2})
        self._add_venn_label(ax, cx[2], cy[2]-0.70,
                             len(s3 - s1 - s2), {g3})
        self._add_venn_label(ax, (cx[0]+cx[1])/2,
                             (cy[0]+cy[1])/2+0.50,
                             len((s1 & s2) - s3), {g1, g2})
        self._add_venn_label(ax, (cx[0]+cx[2])/2-0.40,
                             (cy[0]+cy[2])/2-0.20,
                             len((s1 & s3) - s2), {g1, g3})
        self._add_venn_label(ax, (cx[1]+cx[2])/2+0.40,
                             (cy[1]+cy[2])/2-0.20,
                             len((s2 & s3) - s1), {g2, g3})
        self._add_venn_label(ax, (cx[0]+cx[1]+cx[2])/3,
                             (cy[0]+cy[1]+cy[2])/3,
                             len(s1 & s2 & s3), {g1, g2, g3}, 14)

        ax.text(cx[0]-0.35, cy[0]+r+0.25,
                f"{g1} (n={len(s1)})", ha="center", fontsize=10,
                fontweight="bold")
        ax.text(cx[1]+0.35, cy[1]+r+0.25,
                f"{g2} (n={len(s2)})", ha="center", fontsize=10,
                fontweight="bold")
        ax.text(cx[2], cy[2]-r-0.25,
                f"{g3} (n={len(s3)})", ha="center", fontsize=10,
                fontweight="bold")
        ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3, 3)
        ax.set_aspect("equal"); ax.axis("off")

    # ── 4-set Venn (ellipses with grid-sampled centroids) ──

    def _venn4(self, ax, groups, sets):
        from matplotlib.patches import Ellipse as MplEllipse

        g = groups
        ss = [sets[gi] for gi in g]

        # Classic 4-set Venn layout: tilted ellipses
        eparams = [
            # (cx,    cy,   width, height, angle)
            (-0.15,   0.20,  3.6,   2.0,    50),
            ( 0.85,   0.20,  3.6,   2.0,   130),
            ( 0.10,  -0.55,  3.6,   2.0,    50),
            ( 0.60,  -0.55,  3.6,   2.0,   130),
        ]

        for i, (ecx, ecy, ew, eh, ea) in enumerate(eparams):
            ax.add_patch(MplEllipse(
                (ecx, ecy), ew, eh, angle=ea, alpha=0.22,
                facecolor=PALETTE[i], edgecolor="black", lw=1.3))
            self._add_ellipse_hit(ecx, ecy, ew, eh, ea, g[i])

        # ---- Sample a grid to find centroid of each region ----
        xs = np.linspace(-2.5, 3.5, 200)
        ys = np.linspace(-2.5, 2.5, 200)
        xx, yy = np.meshgrid(xs, ys)
        pts_x, pts_y = xx.ravel(), yy.ravel()

        # Vectorised membership test per ellipse
        masks = np.zeros((4, len(pts_x)), dtype=bool)
        for i, (ecx, ecy, ew, eh, ea) in enumerate(eparams):
            a_rad = np.radians(ea)
            cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)
            dx = pts_x - ecx; dy = pts_y - ecy
            rx =  cos_a * dx + sin_a * dy
            ry = -sin_a * dx + cos_a * dy
            masks[i] = (rx / (ew / 2))**2 + (ry / (eh / 2))**2 <= 1.0

        # Group points by membership pattern → find centroid
        # Encode membership as 4-bit integer for speed
        code = masks[0].astype(int)
        for k in range(1, 4):
            code = code | (masks[k].astype(int) << k)

        region_sums: dict = {}   # code → [sum_x, sum_y, count]
        for j in range(len(pts_x)):
            c = code[j]
            if c == 0:
                continue
            if c not in region_sums:
                region_sums[c] = [0.0, 0.0, 0]
            region_sums[c][0] += pts_x[j]
            region_sums[c][1] += pts_y[j]
            region_sums[c][2] += 1

        # Place labels at centroids
        for c, (sx, sy, cnt) in region_sums.items():
            gset = {g[i] for i in range(4) if c & (1 << i)}
            # Exact size: in these groups AND not in others
            exact = set.intersection(*(sets[gi] for gi in gset))
            for gi in g:
                if gi not in gset:
                    exact = exact - sets[gi]
            if not exact:
                continue
            cx_r, cy_r = sx / cnt, sy / cnt
            sz = 12 if len(gset) == 4 else 10
            self._add_venn_label(ax, cx_r, cy_r, len(exact), gset, sz)

        # Group name labels outside
        name_pos = [
            (eparams[0][0] - 1.2, eparams[0][1] + 1.4),
            (eparams[1][0] + 1.2, eparams[1][1] + 1.4),
            (eparams[2][0] - 1.2, eparams[2][1] - 1.4),
            (eparams[3][0] + 1.2, eparams[3][1] - 1.4),
        ]
        for i, (nx, ny) in enumerate(name_pos):
            ax.text(nx, ny, f"{g[i]}\n(n={len(ss[i])})",
                    ha="center", fontsize=10, fontweight="bold",
                    color=PALETTE[i])

        ax.set_xlim(-2.8, 3.8); ax.set_ylim(-2.8, 2.8)
        ax.set_aspect("equal"); ax.axis("off")

    # ── 5-set Venn (rotated ellipses, flower pattern) ──

    def _venn5(self, ax, groups, sets):
        g = groups
        ss = [sets[gi] for gi in g]
        n_g = 5

        # 5 identical ellipses rotated 72° apart, classic construction
        # Parameters tuned so all 31 regions are visible
        ew, eh = 2.6, 1.5          # ellipse width / height
        d = 0.55                    # center distance from origin
        base_angle = 90             # first ellipse points up

        eparams = []
        for i in range(n_g):
            theta = base_angle + i * 72            # position angle
            cx = d * np.cos(np.radians(theta))
            cy = d * np.sin(np.radians(theta))
            rot = theta                             # rotation = position angle
            eparams.append((cx, cy, ew, eh, rot))

        for i, (ecx, ecy, w, h, ea) in enumerate(eparams):
            ax.add_patch(MplEllipse(
                (ecx, ecy), w, h, angle=ea, alpha=0.20,
                facecolor=PALETTE[i], edgecolor="black", lw=1.2))
            self._add_ellipse_hit(ecx, ecy, w, h, ea, g[i])

        # ---- Grid-sample to find region centroids ----
        xs = np.linspace(-2.5, 2.5, 250)
        ys = np.linspace(-2.5, 2.5, 250)
        xx, yy = np.meshgrid(xs, ys)
        pts_x, pts_y = xx.ravel(), yy.ravel()

        masks = np.zeros((n_g, len(pts_x)), dtype=bool)
        for i, (ecx, ecy, w, h, ea) in enumerate(eparams):
            a_rad = np.radians(ea)
            cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)
            dx = pts_x - ecx; dy = pts_y - ecy
            rx =  cos_a * dx + sin_a * dy
            ry = -sin_a * dx + cos_a * dy
            masks[i] = (rx / (w / 2))**2 + (ry / (h / 2))**2 <= 1.0

        # Encode membership as 5-bit integer
        code = np.zeros(len(pts_x), dtype=int)
        for k in range(n_g):
            code |= masks[k].astype(int) << k

        # Accumulate centroids per region
        region_sums: dict = {}
        for j in range(len(pts_x)):
            c = code[j]
            if c == 0:
                continue
            if c not in region_sums:
                region_sums[c] = [0.0, 0.0, 0]
            region_sums[c][0] += pts_x[j]
            region_sums[c][1] += pts_y[j]
            region_sums[c][2] += 1

        # Collect label data for repulsion pass
        label_data = []
        for c, (sx, sy, cnt) in region_sums.items():
            gset = {g[i] for i in range(n_g) if c & (1 << i)}
            exact = set.intersection(*(sets[gi] for gi in gset))
            for gi in g:
                if gi not in gset:
                    exact = exact - sets[gi]
            if not exact:
                continue
            cx_r, cy_r = sx / cnt, sy / cnt
            n_in = len(gset)
            sz = 11 if n_in == n_g else (9 if n_in >= 3 else 8)
            label_data.append([cx_r, cy_r, len(exact), gset, sz])

        # Repel overlapping labels, then place
        label_data = self._repel_labels(label_data, min_sep=0.32,
                                        iterations=150, spring_k=0.06)
        for x, y, count, gset, sz in label_data:
            self._add_venn_label(ax, x, y, count, gset, sz)

        # Group name labels outside ellipses
        label_d = 1.75
        for i in range(n_g):
            theta = base_angle + i * 72
            nx = label_d * np.cos(np.radians(theta))
            ny = label_d * np.sin(np.radians(theta))
            ax.text(nx, ny, f"{g[i]}\n(n={len(ss[i])})",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=PALETTE[i])

        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal"); ax.axis("off")

    # ── text fallback (6+ groups) ──

    def _venn_text(self, ax, groups, sets, header=""):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        y = 0.92
        if header:
            ax.text(0.5, y, header, ha="center", fontsize=11,
                    fontstyle="italic")
            y -= 0.07
        for g in groups:
            ax.text(0.08, y, f"● {g}: {len(sets[g])} documents",
                    fontsize=10)
            y -= 0.05
        y -= 0.03
        for g1, g2 in combinations(groups, 2):
            ax.text(0.08, y,
                    f"  {g1} ∩ {g2}: {len(sets[g1] & sets[g2])}",
                    fontsize=9, color="#555")
            y -= 0.04
        if len(groups) >= 3:
            ax.text(0.08, y,
                    f"  All: {len(set.intersection(*sets.values()))}",
                    fontsize=9, color="#333")
        ax.axis("off")

    # =================================================================
    # UPSET
    # =================================================================

    def _viz_upset(self):
        gm, groups = self._gm_df, self._group_names
        n_g = len(groups)

        inter = []
        for mask, sub in gm.groupby(list(groups)):
            if not isinstance(mask, tuple):
                mask = (mask,)
            active = tuple(int(v) for v in mask)
            if sum(active) == 0:
                continue
            inter.append({"pat": active, "cnt": len(sub)})
        if not inter:
            return
        inter.sort(key=lambda d: d["cnt"], reverse=True)
        n_i = len(inter)

        fig = Figure(figsize=(max(8, n_i*0.45+2), 5), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_bar = fig.add_subplot(gs[0])
        ax_dot = fig.add_subplot(gs[1], sharex=ax_bar)

        xs = np.arange(n_i)
        cnts = [d["cnt"] for d in inter]
        pats = [d["pat"] for d in inter]
        colors = [PALETTE[min(sum(p)-1, len(PALETTE)-1)] for p in pats]

        bars = ax_bar.bar(xs, cnts, color=colors,
                          edgecolor="white", linewidth=0.5, alpha=0.85)
        for xi, c in zip(xs, cnts):
            ax_bar.text(xi, c + max(cnts)*0.02, str(c),
                        ha="center", va="bottom", fontsize=8)
        ax_bar.set_ylabel("Intersection Size")
        ax_bar.set_title("UpSet Plot — Group Intersections", fontsize=12)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.set_xlim(-0.6, n_i-0.4)
        plt.setp(ax_bar.get_xticklabels(), visible=False)

        for j in range(n_g):
            for xi, pat in enumerate(pats):
                ax_dot.plot(xi, j, "o",
                            color="#333" if pat[j] else "#ddd",
                            markersize=7 if pat[j] else 5)
        for xi, pat in enumerate(pats):
            ys = [j for j, v in enumerate(pat) if v]
            if len(ys) >= 2:
                ax_dot.plot([xi, xi], [min(ys), max(ys)], "-",
                            color="#333", lw=2)
        ax_dot.set_yticks(range(n_g))
        ax_dot.set_yticklabels(groups, fontsize=9)
        ax_dot.set_xlim(-0.6, n_i-0.4)
        ax_dot.set_ylim(-0.5, n_g-0.5); ax_dot.invert_yaxis()
        for s in ("top", "right", "bottom"):
            ax_dot.spines[s].set_visible(False)
        ax_dot.tick_params(bottom=False)
        plt.setp(ax_dot.get_xticklabels(), visible=False)

        fig.tight_layout()
        canvas = self._embed(fig)
        self._upset_bars = bars; self._upset_patterns = pats

        # Track which bar indices are currently selected
        selected_bars: set = set()

        def _click(xd, yd, ctrl):
            nonlocal selected_bars
            idx = int(round(xd))
            if idx < 0 or idx >= n_i:
                return

            # UpSet: always single selection (patterns are specific,
            # multi-select would produce confusing document sets)
            selected_bars = {idx}

            # Update highlights
            for i, rect in enumerate(bars):
                if i in selected_bars:
                    rect.set_edgecolor(HIGHLIGHT_COLOR)
                    rect.set_linewidth(3)
                else:
                    rect.set_edgecolor("white")
                    rect.set_linewidth(0.5)
            self._redraw()

            # Select table rows
            self.table.clearSelection()
            target = {g for g, v in zip(groups, pats[idx]) if v}
            self._select_table_by_groups(target, additive=False)

        canvas.clicked.connect(_click)

    # =================================================================
    # HEATMAP
    # =================================================================

    def _viz_heatmap(self):
        gm, groups = self._gm_df, self._group_names
        method = self.similarity_method
        sim = self._sim_matrix(gm, groups, method)
        n_g = len(groups)

        fig = Figure(figsize=(max(6, n_g*0.8+2), max(5, n_g*0.7+1)), dpi=100)
        ax = fig.add_subplot(111)

        is_cnt = (method == "count")
        im = ax.imshow(sim.values, cmap="YlOrRd", aspect="auto",
                       vmin=(None if is_cnt else 0),
                       vmax=(None if is_cnt else 1))
        ax.set_xticks(range(n_g))
        ax.set_yticks(range(n_g))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(groups, fontsize=9)
        ax.grid(False)

        ml = {"jaccard": "Jaccard Index", "count": "Shared Documents",
              "dice": "Dice Coefficient", "overlap": "Overlap Coefficient"}
        fig.colorbar(im, ax=ax, shrink=0.8).set_label(ml.get(method, method))

        mx = sim.values.max()
        for i in range(n_g):
            for j in range(n_g):
                v = sim.iloc[i, j]
                t = f"{int(v)}" if is_cnt else f"{v:.2f}"
                thr = mx*0.5 if is_cnt else 0.5
                ax.text(j, i, t, ha="center", va="center",
                        color="white" if v > thr else "black", fontsize=9)

        ax.set_title(f"Group Similarity — {ml.get(method, method)}", fontsize=12)
        fig.tight_layout()
        canvas = self._embed(fig)

        selected_cells: set = set()   # (row, col) pairs

        def _click(xd, yd, ctrl):
            nonlocal selected_cells
            ci, ri = int(round(xd)), int(round(yd))
            if not (0 <= ci < n_g and 0 <= ri < n_g and ci != ri):
                return
            cell = (ri, ci)

            if ctrl:
                if cell in selected_cells:
                    selected_cells.discard(cell)
                else:
                    selected_cells.add(cell)
            else:
                selected_cells = {cell}

            # Redraw highlights
            self._clear_highlights()
            for (r, c) in selected_cells:
                rect = MplRect((c-0.5, r-0.5), 1, 1, fill=False,
                               edgecolor=HIGHLIGHT_COLOR, lw=3, zorder=10)
                ax.add_patch(rect)
                self._highlights.append(rect)
            self._redraw()

            # Select table rows
            if not ctrl:
                self.table.clearSelection()
            self._select_table_by_groups(
                {groups[ri], groups[ci]}, additive=ctrl)

        canvas.clicked.connect(_click)

    def _sim_matrix(self, gm, groups, method):
        if HAS_BIBLIUM and method in ("jaccard", "count"):
            try:
                m = utilsbib.compute_group_similarity_matrices(
                    gm, methods=[method])
                if method in m:
                    return m[method]
            except Exception:
                pass
        n = len(groups)
        mat = pd.DataFrame(0.0, index=groups, columns=groups)
        for i, g1 in enumerate(groups):
            s1 = set(gm.index[gm[g1].astype(bool)])
            for j, g2 in enumerate(groups):
                if i == j:
                    mat.iloc[i, j] = len(s1) if method == "count" else 1.0
                    continue
                s2 = set(gm.index[gm[g2].astype(bool)])
                inter = len(s1 & s2); union = len(s1 | s2)
                if method == "jaccard":
                    mat.iloc[i, j] = inter/union if union else 0
                elif method == "count":
                    mat.iloc[i, j] = inter
                elif method == "dice":
                    d = len(s1)+len(s2)
                    mat.iloc[i, j] = 2*inter/d if d else 0
                elif method == "overlap":
                    d = min(len(s1), len(s2))
                    mat.iloc[i, j] = inter/d if d else 0
                else:
                    mat.iloc[i, j] = inter/union if union else 0
        return mat

    # =================================================================
    # NETWORK
    # =================================================================

    def _viz_network(self):
        if not HAS_NX:
            self.viz_lo.addWidget(
                QLabel("networkx is required.\npip install networkx")); return

        gm, groups = self._gm_df, self._group_names
        method, thr = self.similarity_method, self.threshold
        sim = self._sim_matrix(gm, groups, method)
        sizes = {g: int(gm[g].sum()) for g in groups}

        G = nx.Graph()
        for g in groups:
            G.add_node(g)
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i < j:
                    s = sim.loc[g1, g2]
                    if (method == "count" and s > 0) or s > thr:
                        G.add_edge(g1, g2, weight=s)

        fig = Figure(figsize=(8, 7), dpi=100)
        ax = fig.add_subplot(111)

        n = len(groups)
        pos = (nx.spring_layout(G, k=3/np.sqrt(max(n, 1)),
                                iterations=100, seed=42)
               if n > 1 else {g: (0.5, 0.5) for g in groups})

        ncol = [PALETTE[i % len(PALETTE)] for i in range(len(G.nodes()))]
        mx_sz = max(sizes.values()) if sizes else 1
        nsz = [1500*(sizes[g]/mx_sz+0.3) for g in G.nodes()]

        ew = [G[u][v]["weight"] for u, v in G.edges()]
        widths = [5*(w/max(ew)) for w in ew] if ew else []

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ncol,
                               node_size=nsz, alpha=0.8)
        if G.edges():
            nx.draw_networkx_edges(G, pos, ax=ax, width=widths,
                                   alpha=0.6, edge_color="gray")
        labels = {g: f"{g}\n(n={sizes[g]})" for g in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax,
                                font_size=10, font_weight="bold")
        if G.edges():
            fmt = ".0f" if method == "count" else ".2f"
            elbl = {(u, v): f"{G[u][v]['weight']:{fmt}}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, elbl, ax=ax,
                                         font_size=8, alpha=0.8)

        ml = {"jaccard": "Jaccard Similarity", "count": "Shared Documents",
              "dice": "Dice Coefficient", "overlap": "Overlap Coefficient"}
        ax.set_title(f"Group Network ({ml.get(method, method)})", fontsize=12)
        ax.axis("off"); fig.tight_layout()
        canvas = self._embed(fig)

        # Compute data-space "radius" for each node (approximate)
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        dx = xlim[1] - xlim[0]; dy = ylim[1] - ylim[0]
        data_scale = max(dx, dy) if max(dx, dy) > 0 else 1.0
        hit_r = 0.08 * data_scale  # click tolerance

        selected_nodes: set = set()

        def _click(xd, yd, ctrl):
            nonlocal selected_nodes
            best, bd = None, 1e9
            for g, (px, py) in pos.items():
                d = np.hypot(xd - px, yd - py)
                if d < bd:
                    best, bd = g, d
            if best is None or bd > hit_r:
                if not ctrl:
                    self.table.clearSelection()
                    self._clear_highlights(); self._redraw()
                return

            if ctrl:
                if best in selected_nodes:
                    selected_nodes.discard(best)
                else:
                    selected_nodes.add(best)
            else:
                selected_nodes = {best}

            # Redraw highlights
            self._clear_highlights()
            for g in selected_nodes:
                px, py = pos[g]
                hl = MplCircle((px, py), hit_r*0.8, fill=False,
                               edgecolor=HIGHLIGHT_COLOR, lw=3, zorder=10)
                ax.add_patch(hl)
                self._highlights.append(hl)
            self._redraw()

            if not ctrl:
                self.table.clearSelection()
            self._select_table_by_groups(
                {best}, containing=True, additive=ctrl)

        canvas.clicked.connect(_click)

    # =================================================================
    # DENDROGRAM
    # =================================================================

    def _viz_dendro(self):
        if not HAS_SCIPY:
            self.viz_lo.addWidget(
                QLabel("scipy is required.\npip install scipy")); return
        gm, groups = self._gm_df, self._group_names
        if len(groups) < 2:
            self.viz_lo.addWidget(QLabel("Need ≥ 2 groups.")); return

        fig = Figure(figsize=(max(7, len(groups)*0.8+2), 5), dpi=100)
        ax = fig.add_subplot(111)

        dist = pdist(gm.T.values, metric="jaccard")
        Z = linkage(dist, method="average")
        scipy_dendrogram(Z, labels=groups, leaf_rotation=45,
                         leaf_font_size=9, ax=ax, above_threshold_color="gray")

        for i, lbl in enumerate(ax.get_xmajorticklabels()):
            lbl.set_color(PALETTE[i % len(PALETTE)])
            lbl.set_fontweight("bold")

        ax.set_facecolor("white"); ax.grid(False)
        ax.set_ylabel("Jaccard Distance", fontsize=10)
        ax.set_title("Group Similarity Dendrogram", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        canvas = self._embed(fig)

        # Click near a leaf label → select that group
        tick_xs = [t.get_position()[0] for t in ax.get_xmajorticklabels()]
        tick_lbls = [t.get_text() for t in ax.get_xmajorticklabels()]
        half_gap = ((tick_xs[1]-tick_xs[0])/2
                    if len(tick_xs) > 1 else 999)

        selected_leaves: set = set()

        def _click(xd, yd, ctrl):
            nonlocal selected_leaves
            if not tick_xs:
                return
            dists = [abs(xd - tx) for tx in tick_xs]
            bi = int(np.argmin(dists))
            if dists[bi] > half_gap:
                return
            g = tick_lbls[bi]

            if ctrl:
                if g in selected_leaves:
                    selected_leaves.discard(g)
                else:
                    selected_leaves.add(g)
            else:
                selected_leaves = {g}

            # Highlight selected leaves by drawing marker
            self._clear_highlights()
            for sg in selected_leaves:
                si = tick_lbls.index(sg)
                hl = ax.plot(tick_xs[si], 0, "v", color=HIGHLIGHT_COLOR,
                             markersize=12, zorder=20)[0]
                self._highlights.append(hl)
            self._redraw()

            if not ctrl:
                self.table.clearSelection()
            self._select_table_by_groups(
                {g}, containing=True, additive=ctrl)

        canvas.clicked.connect(_click)

    # =================================================================
    # INFO TAB
    # =================================================================

    def _fill_info(self, df):
        groups, gm = self._group_names, self._gm_df
        L = [f"<h3>Group Intersections Summary</h3>",
             f"<b>Groups:</b> {len(groups)}<br>",
             f"<b>Documents:</b> {len(gm)}<br>",
             f"<b>Intersections:</b> {len(df)}<br>"]
        if "Size" in df.columns:
            L.append(f"<b>Largest:</b> {int(df['Size'].max())}<br>")
            L.append(f"<b>Smallest:</b> {int(df['Size'].min())}<br>")
        L.append("<br><b>Group sizes:</b><br>")
        for g in groups:
            L.append(f"&nbsp;&nbsp;• {g}: {int(gm[g].sum())} documents<br>")
        if len(groups) <= 10:
            L.append("<br><b>Pairwise overlaps:</b><br>")
            for g1, g2 in combinations(groups, 2):
                s1 = set(gm.index[gm[g1].astype(bool)])
                s2 = set(gm.index[gm[g2].astype(bool)])
                ov = len(s1 & s2)
                jac = ov/len(s1|s2) if len(s1|s2) else 0
                L.append(f"&nbsp;&nbsp;• {g1} ∩ {g2}: {ov} (J={jac:.3f})<br>")
        self.info_lbl.setText("".join(L))

    # =================================================================
    # OUTPUTS
    # =================================================================

    def _send(self):
        if self._intersections_df is None:
            self._clear(); return
        out = self._intersections_df[
            [c for c in self._intersections_df.columns
             if not c.startswith("_")]].copy()
        self.Outputs.intersections.send(self._df_to_table(out))
        self.Outputs.selected_data.send(None)

    def _clear(self):
        self.Outputs.intersections.send(None)
        self.Outputs.selected_data.send(None)
        self.table.setRowCount(0); self.table.setColumnCount(0)
        for c in (self.card_int, self.card_doc, self.card_lrg):
            self._set_card(c, "0")
        self.info_lbl.setText("No results.")

    def _export_results(self):
        if self._intersections_df is None:
            return
        fname, _ = QFileDialog.getSaveFileName(
            self, "Export", "group_intersections.xlsx",
            "Excel (*.xlsx);;CSV (*.csv)")
        if not fname:
            return
        out = self._intersections_df[
            [c for c in self._intersections_df.columns
             if not c.startswith("_")]]
        try:
            if fname.endswith(".csv"):
                out.to_csv(fname, index=False)
            else:
                out.to_excel(
                    fname if fname.endswith(".xlsx") else fname+".xlsx",
                    index=False, engine="openpyxl")
        except Exception as e:
            self.Error.compute_error(f"Export: {e}")

    # =================================================================
    # HELPERS
    # =================================================================

    @staticmethod
    def _to_binary(s):
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1)
        truthy = {"yes", "true", "1", "1.0", "t", "y"}
        return s.apply(
            lambda v: 1.0 if str(v).strip().lower() in truthy else 0.0)

    def _table_to_df(self, table):
        data = {}
        for var in table.domain.attributes:
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)] if not np.isnan(v) else None
                    for v in col]
            else:
                data[var.name] = col
        for var in table.domain.class_vars:
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)] if not np.isnan(v) else None
                    for v in col]
            else:
                data[var.name] = col
        for var in table.domain.metas:
            col = table[:, var].metas.flatten()
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)]
                    if not (isinstance(v, float) and np.isnan(v)) else None
                    for v in col]
            elif isinstance(var, StringVariable):
                data[var.name] = [
                    str(v) if v is not None and str(v) != "?" else None
                    for v in col]
            else:
                data[var.name] = col
        return pd.DataFrame(data)

    def _df_to_table(self, df):
        attrs, metas, Xc, Mc = [], [], [], []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                attrs.append(ContinuousVariable(str(col))); Xc.append(col)
            else:
                metas.append(StringVariable(str(col))); Mc.append(col)
        domain = Domain(attrs, metas=metas)
        X = np.empty((len(df), len(attrs)), float)
        for i, c in enumerate(Xc):
            X[:, i] = pd.to_numeric(df[c], errors="coerce").fillna(np.nan)
        M = np.empty((len(df), len(metas)), object)
        for i, c in enumerate(Mc):
            M[:, i] = df[c].fillna("").astype(str).values
        return Table.from_numpy(domain, X, metas=M if metas else None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWGroupIntersections).run()
