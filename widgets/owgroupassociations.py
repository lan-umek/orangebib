# -*- coding: utf-8 -*-
"""
Group Associations Widget  (v2)
================================
Orange widget for analysing entity–group relationships with five
interactive visualisations.

Statistical analyses
--------------------
Contingency · Diversity · Correspondence Analysis · Chi-square · SVD · Log-ratio

Visualisations (all with hover tooltips + click selection)
----------------------------------------------------------
Heatmap · CA Biplot · Balloon Plot · Bipartite Graph · Sankey Diagram

Selection model
---------------
* Heatmap / CA Biplot / Bipartite  →  **entity-level** selection
* Balloon Plot / Sankey            →  **cell-level** (entity × group pair)

Ctrl+Click extends selection; plain click replaces.
"""

import logging
import re
from typing import Optional, List, Set, Tuple
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QAbstractItemView, QTabWidget,
    QSplitter, QFrame, QLineEdit, QFileDialog, QApplication,
    QToolButton, QToolTip,
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal, QPoint
from AnyQt.QtGui import QColor, QFont

from Orange.data import (
    Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable,
)
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.settings import Setting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    sp_stats = None

try:
    from sklearn.feature_extraction.text import CountVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUP_PREFIX = "Group: "

_VIZ_HEATMAP = 0
_VIZ_BIPLOT = 1
_VIZ_BALLOON = 2
_VIZ_BIPARTITE = 3
_VIZ_SANKEY = 4
VIZ_NAMES = ["Heatmap", "CA Biplot", "Balloon Plot",
             "Bipartite Graph", "Sankey Diagram"]

_GROUP_COLOURS = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
    "#db2777", "#0891b2", "#65a30d", "#ea580c", "#6d28d9",
]

ENTITY_CONFIGS = {
    "sources": {
        "label": "Sources (Journals)",
        "entity_col_alts": ["Source title", "Source", "Journal", "source", "SO"],
        "entity_label": "Source",
        "value_type": "string",
        "description": "Associate journals with groups",
    },
    "authors": {
        "label": "Authors",
        "entity_col_alts": ["Authors", "Author", "authors", "AU"],
        "entity_label": "Author",
        "value_type": "list",
        "description": "Associate authors with groups",
    },
    "author_keywords": {
        "label": "Author Keywords",
        "entity_col_alts": [
            "Processed Author Keywords", "Author Keywords",
            "author keywords", "DE", "Keywords",
        ],
        "entity_label": "Keyword",
        "value_type": "list",
        "description": "Associate author keywords with groups",
    },
    "index_keywords": {
        "label": "Index Keywords",
        "entity_col_alts": [
            "Index Keywords", "Keywords Plus",
            "index keywords", "ID", "KeywordsPlus",
        ],
        "entity_label": "Keyword",
        "value_type": "list",
        "description": "Associate index keywords with groups",
    },
    "countries": {
        "label": "Countries",
        "entity_col_alts": [
            "Countries of Authors", "All Countries", "Country",
            "CA Country", "countries",
        ],
        "entity_label": "Country",
        "value_type": "list",
        "description": "Associate countries with groups",
    },
    "affiliations": {
        "label": "Affiliations",
        "entity_col_alts": ["Affiliations", "Affiliation", "C1", "C3"],
        "entity_label": "Affiliation",
        "value_type": "list",
        "description": "Associate affiliations with groups",
    },
    "references": {
        "label": "References",
        "entity_col_alts": [
            "References", "Cited References", "CR", "cited references",
        ],
        "entity_label": "Reference",
        "value_type": "list",
        "description": "Associate cited references with groups",
    },
    "subject_areas": {
        "label": "Subject Areas",
        "entity_col_alts": [
            "Subject Area", "Subject Areas", "Research Areas",
            "WC", "SC", "Topics",
        ],
        "entity_label": "Subject Area",
        "value_type": "list",
        "description": "Associate subject areas with groups",
    },
    "document_types": {
        "label": "Document Types",
        "entity_col_alts": [
            "Document Type", "Document type", "type", "DT", "Type",
        ],
        "entity_label": "Document Type",
        "value_type": "string",
        "description": "Associate document types with groups",
    },
    "ngrams_title": {
        "label": "N-grams (Title)",
        "entity_col_alts": ["Processed Title", "Title", "TI", "title"],
        "entity_label": "Term",
        "value_type": "text",
        "description": "Associate title n-grams with groups",
    },
    "ngrams_abstract": {
        "label": "N-grams (Abstract)",
        "entity_col_alts": [
            "Processed Abstract", "Abstract", "AB", "abstract",
        ],
        "entity_label": "Term",
        "value_type": "text",
        "description": "Associate abstract n-grams with groups",
    },
}


# =============================================================================
# WORKER THREAD
# =============================================================================

class AssocWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self._func = func

    def run(self):
        try:
            self.finished.emit((self._func(), None))
        except Exception as e:
            logger.exception("AssocWorker")
            self.finished.emit((None, str(e)))


# =============================================================================
# WIDGET
# =============================================================================

class OWGroupAssociations(OWWidget):

    name = "Group Associations"
    description = (
        "Analyse entity–group relationships: contingency, diversity, CA, "
        "chi-square, SVD, log-ratio — with heatmap, biplot, balloon, "
        "bipartite and Sankey visualisations"
    )
    icon = "icons/group_associations.svg"
    priority = 35
    keywords = [
        "group", "association", "entity", "contingency", "chi-square",
        "diversity", "correspondence", "svd", "log-ratio", "heatmap",
        "bipartite", "sankey",
    ]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table,
                     doc="Bibliographic data with group columns")

    class Outputs:
        contingency = Output("Contingency", Table)
        diversity = Output("Diversity", Table)
        correspondence = Output("Correspondence", Table)
        chi_square = Output("Chi-square", Table)
        svd_out = Output("SVD", Table)
        log_ratio = Output("Log-ratio", Table)
        selected_data = Output("Selected Documents", Table, default=True)
        filtered = Output("Filtered", Table)

    # ── settings ──
    entity_type_idx = Setting(0)
    top_n = Setting(50)
    min_freq = Setting(5)
    compute_diversity = Setting(True)
    compute_ca = Setting(True)
    compute_chi = Setting(True)
    compute_svd = Setting(True)
    compute_logratio = Setting(True)
    include_patterns_text = Setting("")
    exclude_patterns_text = Setting("")
    auto_apply = Setting(True)
    viz_type_idx = Setting(0)
    min_abs_residual = Setting(0.0)
    splitter_state = Setting(b"")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_groups = Msg("No 'Group: …' columns — connect Setup Groups")
        no_column = Msg("Column not found: {}")
        compute_error = Msg("{}")

    class Warning(OWWidget.Warning):
        few_items = Msg("Only {} entities passed filters")
        scipy_missing = Msg("scipy required for chi-square")

    class Information(OWWidget.Information):
        computing = Msg("Computing…")

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(self):
        super().__init__()

        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._gm_df: Optional[pd.DataFrame] = None
        self._group_names: List[str] = []
        self._worker: Optional[AssocWorker] = None

        # result storage
        self._contingency_df: Optional[pd.DataFrame] = None
        self._diversity_df: Optional[pd.DataFrame] = None
        self._ca_df: Optional[pd.DataFrame] = None
        self._chi_df: Optional[pd.DataFrame] = None
        self._svd_df: Optional[pd.DataFrame] = None
        self._logratio_df: Optional[pd.DataFrame] = None
        self._filtered_df: Optional[pd.DataFrame] = None

        # visualisation state
        self._selected_entities: Set[int] = set()               # row indices
        self._selected_cells: Set[Tuple[str, str]] = set()      # (entity, group)
        self._hover_idx: int = -1
        self._hover_cell: Tuple[int, int] = (-1, -1)            # (row, col)
        self._entity_label: str = "Source"
        self._entity_col_name: Optional[str] = None
        self._mpl_cids: list = []
        self._viz_meta: Optional[dict] = None

        self._setup_control_area()
        self._setup_main_area()

    # =========================================================================
    # CONTROL AREA
    # =========================================================================

    def _setup_control_area(self):
        # ── Entity Selection ──
        box = gui.widgetBox(self.controlArea, "📋 Entity Selection")
        f = QGridLayout(); f.setColumnStretch(1, 1)
        f.addWidget(QLabel("Entity Type:"), 0, 0)
        self.entity_combo = QComboBox()
        for cfg in ENTITY_CONFIGS.values():
            self.entity_combo.addItem(cfg["label"])
        self.entity_combo.setCurrentIndex(self.entity_type_idx)
        self.entity_combo.currentIndexChanged.connect(self._on_entity_changed)
        f.addWidget(self.entity_combo, 0, 1)
        self.entity_desc = QLabel("")
        self.entity_desc.setStyleSheet("color:#3b82f6;font-size:11px;")
        self.entity_desc.setWordWrap(True)
        f.addWidget(self.entity_desc, 1, 0, 1, 2)
        box.layout().addLayout(f)
        self._update_entity_description()

        # ── Statistics ──
        sbox = gui.widgetBox(self.controlArea, "📊 Statistics to Include")
        for attr, label in [
            ("compute_diversity", "Diversity Index"),
            ("compute_ca", "Correspondence Analysis"),
            ("compute_chi", "Chi-square Test"),
            ("compute_svd", "SVD Analysis"),
            ("compute_logratio", "Log-ratio"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(getattr(self, attr))
            cb.toggled.connect(lambda v, a=attr: setattr(self, a, v))
            sbox.layout().addWidget(cb)

        # ── Selection Criteria ──
        sel = gui.widgetBox(self.controlArea, "🔎 Selection Criteria")
        sf = QGridLayout(); sf.setColumnStretch(1, 1)
        sf.addWidget(QLabel("Top N Items:"), 0, 0)
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 5000); self.top_n_spin.setValue(self.top_n)
        self.top_n_spin.valueChanged.connect(lambda v: setattr(self, "top_n", v))
        sf.addWidget(self.top_n_spin, 0, 1)
        sf.addWidget(QLabel("Min Frequency:"), 1, 0)
        self.min_freq_spin = QSpinBox()
        self.min_freq_spin.setRange(1, 1000); self.min_freq_spin.setValue(self.min_freq)
        self.min_freq_spin.valueChanged.connect(lambda v: setattr(self, "min_freq", v))
        sf.addWidget(self.min_freq_spin, 1, 1)
        sel.layout().addLayout(sf)

        # ── Filtering (collapsible) ──
        _, self._filter_content = self._make_collapsible(
            self.controlArea, "🔍 Filtering", collapsed=True)
        ff = QGridLayout(); ff.setColumnStretch(1, 1)
        ff.addWidget(QLabel("Include (regex):"), 0, 0)
        self.include_edit = QLineEdit(self.include_patterns_text)
        self.include_edit.setPlaceholderText("pat1 ; pat2  (semicolon-sep)")
        self.include_edit.editingFinished.connect(
            lambda: setattr(self, "include_patterns_text",
                            self.include_edit.text()))
        ff.addWidget(self.include_edit, 0, 1)
        ff.addWidget(QLabel("Exclude (regex):"), 1, 0)
        self.exclude_edit = QLineEdit(self.exclude_patterns_text)
        self.exclude_edit.setPlaceholderText("pat1 ; pat2  (semicolon-sep)")
        self.exclude_edit.editingFinished.connect(
            lambda: setattr(self, "exclude_patterns_text",
                            self.exclude_edit.text()))
        ff.addWidget(self.exclude_edit, 1, 1)
        self._filter_content.layout().addLayout(ff)

        # ── Display Options (collapsible) ──
        _, self._disp_content = self._make_collapsible(
            self.controlArea, "🖥️ Display Options", collapsed=True)
        self.auto_apply_cb = QCheckBox("Auto apply on data change")
        self.auto_apply_cb.setChecked(self.auto_apply)
        self.auto_apply_cb.toggled.connect(
            lambda v: setattr(self, "auto_apply", v))
        self._disp_content.layout().addWidget(self.auto_apply_cb)

        # Min |residual| for balloon
        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("Min |residual|:"))
        self.min_resid_spin = QDoubleSpinBox()
        self.min_resid_spin.setRange(0.0, 20.0)
        self.min_resid_spin.setSingleStep(0.1)
        self.min_resid_spin.setDecimals(1)
        self.min_resid_spin.setValue(self.min_abs_residual)
        self.min_resid_spin.setToolTip(
            "Suppress balloon bubbles with |std. residual| below this value")
        self.min_resid_spin.valueChanged.connect(self._on_min_resid_changed)
        r_row.addWidget(self.min_resid_spin)
        r_row.addStretch()
        self._disp_content.layout().addLayout(r_row)

        # ── Buttons ──
        bl = QVBoxLayout()
        self.compute_btn = QPushButton("▶ Compute Associations")
        self.compute_btn.clicked.connect(self._run_compute)
        self.compute_btn.setStyleSheet(
            "QPushButton{background:#16a34a;border:none;border-radius:4px;"
            "padding:10px 20px;color:white;font-weight:bold;font-size:13px}"
            "QPushButton:hover{background:#15803d}"
            "QPushButton:disabled{background:#ccc}")
        bl.addWidget(self.compute_btn)
        ql = QHBoxLayout()
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setStyleSheet(
            "QPushButton{background:#e0e7ff;border:1px solid #6366f1;"
            "border-radius:4px;padding:6px 12px;color:#4338ca;font-weight:bold}"
            "QPushButton:hover{background:#c7d2fe}")
        ql.addWidget(self.export_btn); ql.addStretch()
        bl.addLayout(ql)
        self.controlArea.layout().addLayout(bl)

    @staticmethod
    def _make_collapsible(parent, title, collapsed=False):
        c = QWidget()
        lo = QVBoxLayout(c); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)
        h = QToolButton(); h.setText(f"  {title}")
        h.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        h.setArrowType(Qt.RightArrow if collapsed else Qt.DownArrow)
        h.setCheckable(True); h.setChecked(not collapsed)
        h.setStyleSheet("QToolButton{border:none;font-weight:bold;font-size:12px;"
                         "padding:4px 0;color:#374151}"
                         "QToolButton:hover{color:#1d4ed8}")
        h.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lo.addWidget(h)
        w = QWidget(); QVBoxLayout(w).setContentsMargins(8, 4, 4, 8)
        w.setVisible(not collapsed); lo.addWidget(w)
        h.toggled.connect(lambda ck: (
            w.setVisible(ck),
            h.setArrowType(Qt.DownArrow if ck else Qt.RightArrow)))
        parent.layout().addWidget(c)
        return h, w

    # =========================================================================
    # MAIN AREA
    # =========================================================================

    def _setup_main_area(self):
        layout = self.mainArea.layout()
        if layout is None:
            layout = QVBoxLayout(); self.mainArea.setLayout(layout)

        self.status_frame = QFrame()
        self.status_frame.setStyleSheet(
            "background:#f0f9ff;border:1px solid #bae6fd;border-radius:4px;")
        sl = QHBoxLayout(self.status_frame)
        self.status_label = QLabel("No data loaded")
        self.status_label.setStyleSheet("color:#0369a1;border:none;")
        sl.addWidget(self.status_label)
        layout.addWidget(self.status_frame)

        self.splitter = QSplitter(Qt.Vertical)

        # ── viz panel ──
        vp = QWidget(); vl = QVBoxLayout(vp); vl.setContentsMargins(0, 0, 0, 0)
        vh = QHBoxLayout()
        vh.addWidget(QLabel("Visualisation:"))
        self.viz_combo = QComboBox(); self.viz_combo.addItems(VIZ_NAMES)
        self.viz_combo.setCurrentIndex(self.viz_type_idx)
        self.viz_combo.currentIndexChanged.connect(self._on_viz_changed)
        vh.addWidget(self.viz_combo); vh.addStretch()
        self.sel_label = QLabel("")
        self.sel_label.setStyleSheet("color:#6366f1;font-size:11px;")
        vh.addWidget(self.sel_label)
        vl.addLayout(vh)

        if HAS_MPL:
            self.fig = Figure(figsize=(8, 5), dpi=100)
            self.fig.set_tight_layout(True)
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setMinimumHeight(200)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            vl.addWidget(self.canvas)
        else:
            self.fig = self.canvas = None
            vl.addWidget(QLabel("matplotlib not available"))
        self.splitter.addWidget(vp)

        # ── table tabs ──
        self.tabs = QTabWidget()
        self.tab_contingency = self._make_tw()
        self.tab_diversity = self._make_tw()
        self.tab_ca = self._make_tw()
        self.tab_chi = self._make_tw()
        self.tab_svd = self._make_tw()
        self.tab_lr = self._make_tw()
        for name, tw in [("Contingency", self.tab_contingency),
                         ("Diversity", self.tab_diversity),
                         ("Correspondence", self.tab_ca),
                         ("Chi-square", self.tab_chi),
                         ("SVD", self.tab_svd),
                         ("Log-ratio", self.tab_lr)]:
            self.tabs.addTab(tw, name)
        self.tab_contingency.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 3); self.splitter.setStretchFactor(1, 2)
        if self.splitter_state:
            self.splitter.restoreState(self.splitter_state)
        layout.addWidget(self.splitter)

    @staticmethod
    def _make_tw():
        tw = QTableWidget(); tw.setAlternatingRowColors(True)
        tw.setSelectionBehavior(QAbstractItemView.SelectRows)
        tw.setSortingEnabled(True)
        tw.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        tw.horizontalHeader().setStretchLastSection(True)
        return tw

    # =========================================================================
    # INPUT
    # =========================================================================

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear(); self.Warning.clear()
        self._clear_results()
        if data is None:
            self._data = self._df = self._gm_df = None
            self._group_names = []
            self.status_label.setText("No data loaded")
            self._send_outputs(); return

        self._data = data
        self._df = self._table_to_df(data)
        group_cols = [c for c in self._df.columns if c.startswith(GROUP_PREFIX)]
        if not group_cols:
            self.Error.no_groups()
            self._gm_df = None; self._group_names = []; return

        self._group_names = [c[len(GROUP_PREFIX):] for c in group_cols]
        gm = self._df[group_cols].copy(); gm.columns = self._group_names
        for c in gm.columns:
            col = gm[c]
            if col.dtype == object:
                cl = col.astype(str).str.strip().str.lower()
                gm[c] = cl.map({"yes": 1, "no": 0, "true": 1, "false": 0,
                                 "1": 1, "0": 0, "1.0": 1, "0.0": 0}
                                ).fillna(0).astype(int)
            else:
                gm[c] = pd.to_numeric(col, errors="coerce").fillna(0).astype(int)
        self._gm_df = gm
        self.status_label.setText(
            f"📊 {len(self._df):,} documents · {len(self._group_names)} groups")
        if self.auto_apply:
            self._run_compute()

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def _on_entity_changed(self, idx):
        self.entity_type_idx = idx
        self._update_entity_description()

    def _update_entity_description(self):
        cfg = list(ENTITY_CONFIGS.values())[self.entity_type_idx]
        self.entity_desc.setText(cfg["description"])

    def _on_viz_changed(self, idx):
        self.viz_type_idx = idx
        self._update_visualization()

    def _on_min_resid_changed(self, v):
        self.min_abs_residual = v
        if self.viz_type_idx == _VIZ_BALLOON:
            self._update_visualization()

    # =========================================================================
    # COMPUTE
    # =========================================================================

    def _run_compute(self):
        if self._df is None or self._gm_df is None:
            return
        self.Information.computing()
        self.compute_btn.setEnabled(False)
        self._worker = AssocWorker(self._compute_associations)
        self._worker.finished.connect(self._on_compute_done)
        self._worker.start()

    def _on_compute_done(self, result):
        self.Information.clear(); self.compute_btn.setEnabled(True)
        self._worker = None
        results, err = result
        if err:
            self.Error.compute_error(err); return
        (cont, div_df, ca_df, chi_df, svd_df, lr_df, filt_df) = results
        self._contingency_df = cont
        self._diversity_df = div_df
        self._ca_df = ca_df
        self._chi_df = chi_df
        self._svd_df = svd_df
        self._logratio_df = lr_df
        self._filtered_df = filt_df
        self._selected_entities.clear(); self._selected_cells.clear()
        self._hover_idx = -1; self._hover_cell = (-1, -1)
        cfg = list(ENTITY_CONFIGS.values())[self.entity_type_idx]
        self._entity_label = cfg["entity_label"]
        self._entity_col_name = self._find_column(self._df, cfg["entity_col_alts"])
        self._display_all(); self._update_visualization(); self._send_outputs()
        parts = []
        n_ent = len(cont) if cont is not None and not cont.empty else 0
        if n_ent: parts.append(f"{n_ent} entities")
        if div_df is not None: parts.append("diversity")
        if ca_df is not None: parts.append("CA")
        if chi_df is not None: parts.append("χ²")
        if svd_df is not None: parts.append("SVD")
        if lr_df is not None: parts.append("log-ratio")
        self.status_label.setText(f"✅ {' · '.join(parts)}")

    def _compute_associations(self):
        df, gm, groups = self._df, self._gm_df, self._group_names
        cfg = list(ENTITY_CONFIGS.values())[self.entity_type_idx]
        entity_col = self._find_column(df, cfg["entity_col_alts"])
        if entity_col is None:
            raise ValueError(f"Column not found: {cfg['entity_col_alts'][:4]}")
        label = cfg["entity_label"]
        vt = cfg["value_type"]
        sep = self._detect_separator(df)

        cont = self._build_contingency(df, gm, groups, entity_col, label, vt, sep)
        if cont.empty:
            return (pd.DataFrame(), None, None, None, None, None, None)
        cont = self._apply_filters(cont, label)
        if cont.empty:
            return (pd.DataFrame(), None, None, None, None, None, None)
        if len(cont) < 3:
            self.Warning.few_items(len(cont))
        mat = cont[groups].values.astype(float)
        items = cont[label].tolist()

        div_df = self._calc_diversity(mat, groups) if self.compute_diversity else None
        ca_df = (self._calc_ca(mat, items, groups, label)
                 if self.compute_ca and min(mat.shape) >= 2 else None)
        chi_df = (self._calc_chi_square(mat, items, groups, label)
                  if self.compute_chi and HAS_SCIPY else None)
        svd_df = self._calc_svd(mat, items, groups, label) if self.compute_svd else None
        lr_df = (self._calc_logratio(mat, items, groups, label)
                 if self.compute_logratio and len(groups) >= 2 else None)
        return (cont, div_df, ca_df, chi_df, svd_df, lr_df, cont)

    # ── contingency ──
    def _build_contingency(self, df, gm, groups, ecol, elabel, vtype, sep):
        rows = []
        for g in groups:
            mask = gm[g].astype(bool)
            sub = df.loc[mask]
            if sub.empty: continue
            for e in self._extract_entities(sub, ecol, vtype, sep):
                rows.append((e, g))
        if not rows: return pd.DataFrame()
        pf = pd.DataFrame(rows, columns=[elabel, "_g"])
        pv = pf.groupby([elabel, "_g"]).size().unstack(fill_value=0)
        for g in groups:
            if g not in pv.columns: pv[g] = 0
        pv = pv[groups]; pv["Total"] = pv.sum(axis=1)
        pv = pv.sort_values("Total", ascending=False).reset_index()
        return pv[pv["Total"] >= self.min_freq].head(self.top_n).reset_index(drop=True)

    def _extract_entities(self, df, ecol, vtype, sep):
        if ecol not in df.columns: return []
        s = df[ecol].dropna().astype(str).str.strip()
        s = s[(s != "") & (s.str.lower() != "nan")]
        if vtype == "list":
            it = s.str.split(sep).explode().str.strip()
            return it[(it != "") & (it.str.lower() != "nan")].tolist()
        if vtype == "text" and HAS_SKLEARN:
            try:
                v = CountVectorizer(ngram_range=(1, 2), max_features=2000)
                m = v.fit_transform(s); t = v.get_feature_names_out()
                return [t[c] for _, c in zip(*m.nonzero())]
            except Exception: pass
        return s.tolist()

    # ── filtering ──
    def _apply_filters(self, cont, elabel):
        for mode, text in [("include", self.include_patterns_text),
                           ("exclude", self.exclude_patterns_text)]:
            pats = [p.strip() for p in text.split(";") if p.strip()]
            if not pats: continue
            mask = pd.Series(False, index=cont.index)
            for p in pats:
                try: mask |= cont[elabel].str.contains(p, case=False, na=False, regex=True)
                except re.error: mask |= cont[elabel].str.contains(
                    re.escape(p), case=False, na=False, regex=True)
            cont = cont[mask] if mode == "include" else cont[~mask]
        return cont.reset_index(drop=True)

    # ── diversity ──
    def _calc_diversity(self, mat, groups):
        rows = []
        for j, g in enumerate(groups):
            c = mat[:, j].copy(); t = c.sum()
            if t == 0:
                rows.append({"Group": g, "N entities": 0, "N documents": 0,
                             "Shannon H": 0, "Shannon Hmax": 0,
                             "Shannon Evenness": 0, "Simpson D": 0, "Simpson 1-D": 0})
                continue
            p = c / t; p = p[p > 0]; ne = len(p)
            sh = -np.sum(p * np.log(p))
            hm = np.log(ne) if ne > 1 else 0
            ev = sh / hm if hm > 0 else 0
            sd = np.sum(p ** 2)
            rows.append({"Group": g, "N entities": ne, "N documents": int(t),
                         "Shannon H": round(sh, 4), "Shannon Hmax": round(hm, 4),
                         "Shannon Evenness": round(ev, 4),
                         "Simpson D": round(sd, 6), "Simpson 1-D": round(1 - sd, 6)})
        return pd.DataFrame(rows)

    # ── correspondence analysis ──
    def _calc_ca(self, mat, items, groups, elabel):
        g = mat.sum()
        if g == 0: return None
        P = mat / g; r = P.sum(1, keepdims=True); c = P.sum(0, keepdims=True)
        r[r == 0] = 1e-12; c[c == 0] = 1e-12
        S = (P - r @ c) / np.sqrt(r @ c)
        k = min(mat.shape[0], mat.shape[1], 10) - 1
        if k < 1: return None
        try: U, s, Vt = np.linalg.svd(S, full_matrices=False)
        except np.linalg.LinAlgError: return None
        k = min(k, len(s)); ine = s[:k] ** 2; ti = ine.sum()
        rc = (U[:, :k] * s[:k]) / np.sqrt(r)
        cc = (Vt[:k, :].T * s[:k]) / np.sqrt(c.T)
        rows = []
        for i, it in enumerate(items):
            row = {elabel: it, "Type": "Entity"}
            for d in range(min(k, 3)): row[f"Dim {d+1}"] = round(float(rc[i, d]), 6)
            rows.append(row)
        for j, gn in enumerate(groups):
            row = {elabel: gn, "Type": "Group"}
            for d in range(min(k, 3)): row[f"Dim {d+1}"] = round(float(cc[j, d]), 6)
            rows.append(row)
        df = pd.DataFrame(rows)
        for d in range(min(k, 3)):
            pct = ine[d] / ti * 100 if ti > 0 else 0
            df = pd.concat([df, pd.DataFrame([{
                elabel: f"[Inertia Dim {d+1}]", "Type": "Info",
                f"Dim {d+1}": round(pct, 2)}])], ignore_index=True)
        return df

    # ── chi-square ──
    def _calc_chi_square(self, mat, items, groups, elabel):
        if not HAS_SCIPY: return None
        try: chi2, pv, dof, exp = sp_stats.chi2_contingency(mat)
        except ValueError: return None
        contrib = (mat - exp) ** 2 / np.where(exp > 0, exp, 1)
        rchi = contrib.sum(1)
        rows = []
        for i, it in enumerate(items):
            row = {elabel: it, "Chi-square contribution": round(float(rchi[i]), 4)}
            for j, g in enumerate(groups):
                row[f"{g} Observed"] = int(mat[i, j])
                row[f"{g} Expected"] = round(float(exp[i, j]), 2)
                row[f"{g} Residual"] = round(float(mat[i, j] - exp[i, j]), 2)
                row[f"{g} Std. residual"] = round(
                    float((mat[i, j] - exp[i, j]) / max(np.sqrt(exp[i, j]), 1e-12)), 3)
            rows.append(row)
        df = pd.DataFrame(rows).sort_values(
            "Chi-square contribution", ascending=False).reset_index(drop=True)
        df = pd.concat([df, pd.DataFrame([{
            elabel: f"[Overall: χ²={chi2:.2f}, p={pv:.2e}, df={dof}]",
            "Chi-square contribution": round(chi2, 4)}])], ignore_index=True)
        return df

    # ── SVD ──
    def _calc_svd(self, mat, items, groups, elabel):
        if min(mat.shape) < 2: return None
        cs = mat.sum(0, keepdims=True); cs[cs == 0] = 1; nm = mat / cs
        try: U, s, Vt = np.linalg.svd(nm, full_matrices=False)
        except np.linalg.LinAlgError: return None
        k = min(len(s), 5); tv = np.sum(s ** 2)
        rows = []
        for d in range(k):
            rows.append({elabel: f"[Singular value {d+1}]", "Type": "Singular value",
                         "Value": round(float(s[d]), 6),
                         "% Variance": round(s[d] ** 2 / tv * 100 if tv else 0, 2)})
        for i, it in enumerate(items):
            row = {elabel: it, "Type": "Entity loading"}
            for d in range(min(k, 3)): row[f"Component {d+1}"] = round(float(U[i, d]), 6)
            rows.append(row)
        for j, g in enumerate(groups):
            row = {elabel: g, "Type": "Group loading"}
            for d in range(min(k, 3)): row[f"Component {d+1}"] = round(float(Vt[d, j]), 6)
            rows.append(row)
        return pd.DataFrame(rows)

    # ── log-ratio ──
    def _calc_logratio(self, mat, items, groups, elabel):
        cs = mat.sum(0); ni = mat.shape[0]
        props = (mat + 0.5) / (cs + 0.5 * ni)[None, :]
        rows = []
        for i, it in enumerate(items):
            row = {elabel: it}
            for a, b in combinations(range(len(groups)), 2):
                row[f"log₂({groups[a]}/{groups[b]})"] = round(
                    float(np.log2(props[i, a] / props[i, b])), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    # =========================================================================
    # VISUALISATION — redraw dispatcher
    # =========================================================================

    def _update_visualization(self):
        if not HAS_MPL or self.canvas is None: return
        if self._contingency_df is None or self._contingency_df.empty:
            self.fig.clear(); ax = self.fig.add_subplot(111)
            ax.text(.5, .5, "No data — click Compute", ha="center", va="center",
                    fontsize=14, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off(); self.canvas.draw_idle(); return
        for cid in self._mpl_cids: self.canvas.mpl_disconnect(cid)
        self._mpl_cids.clear(); self.fig.clear()

        {_VIZ_HEATMAP:   self._draw_heatmap,
         _VIZ_BIPLOT:    self._draw_biplot,
         _VIZ_BALLOON:   self._draw_balloon,
         _VIZ_BIPARTITE: self._draw_bipartite,
         _VIZ_SANKEY:    self._draw_sankey,
        }.get(self.viz_type_idx, self._draw_heatmap)()

        self._mpl_cids.append(
            self.canvas.mpl_connect("motion_notify_event", self._on_mpl_hover))
        self._mpl_cids.append(
            self.canvas.mpl_connect("button_press_event", self._on_mpl_click))
        self.canvas.draw_idle()

    # ── helpers ──
    def _cont_data(self):
        """Return (items, mat, groups) from contingency."""
        c = self._contingency_df
        g = self._group_names; l = self._entity_label
        return c[l].tolist(), c[g].values.astype(float), g

    # ── 0  Heatmap ──
    def _draw_heatmap(self):
        items, mat, groups = self._cont_data()
        ni, ng = mat.shape
        rs = mat.sum(1, keepdims=True); rs[rs == 0] = 1; normed = mat / rs
        ax = self.fig.add_subplot(111)
        im = ax.imshow(normed, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=max(normed.max() * 1.05, .01),
                        interpolation="nearest")
        for i in range(ni):
            for j in range(ng):
                c = "white" if normed[i, j] > normed.max() * .65 else "black"
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center",
                        fontsize=8, color=c, fontweight="bold")
        ax.set_xticks(range(ng)); ax.set_xticklabels(groups, fontsize=10, fontweight="bold")
        ax.set_yticks(range(ni)); ax.set_yticklabels(
            items, fontsize=max(6, min(9, 300 // max(ni, 1))))
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        for idx in self._selected_entities:
            if 0 <= idx < ni:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (-.5, idx-.5), ng, 1, boxstyle="round,pad=.02",
                    lw=2.5, ec="#6366f1", fc="none", zorder=10))
        if 0 <= self._hover_idx < ni:
            ax.axhspan(self._hover_idx-.5, self._hover_idx+.5,
                        color="#fbbf24", alpha=.15, zorder=1)
        self.fig.colorbar(im, ax=ax, label="Row proportion", shrink=.6, pad=.02)
        ax.set_title(f"Entity × Group Heatmap ({self._entity_label}s)", fontsize=12, pad=12)
        self.fig.tight_layout()
        self._viz_meta = {"type": "heatmap", "n_items": ni, "n_groups": ng,
                          "items": items, "groups": groups, "mat": mat,
                          "normed": normed, "ax": ax}

    # ── 1  CA Biplot ──
    def _draw_biplot(self):
        ca = self._ca_df; cont = self._contingency_df
        groups = self._group_names; label = self._entity_label
        ax = self.fig.add_subplot(111)
        if ca is None or ca.empty or "Dim 1" not in ca.columns or "Dim 2" not in ca.columns:
            ax.text(.5, .5, "CA not computed", ha="center", va="center",
                    fontsize=12, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off(); self._viz_meta = {"type": "biplot"}; return
        ef = ca[ca["Type"] == "Entity"].reset_index(drop=True)
        gf = ca[ca["Type"] == "Group"].reset_index(drop=True)
        inf = ca[ca["Type"] == "Info"]
        totals = cont["Total"].values.astype(float) if cont is not None else np.ones(len(ef))
        mx = max(totals.max(), 1)
        sizes = 20 + 200 * totals / mx if len(totals) == len(ef) else np.full(len(ef), 60)
        ex, ey = ef["Dim 1"].astype(float).values, ef["Dim 2"].astype(float).values
        gx, gy = gf["Dim 1"].astype(float).values, gf["Dim 2"].astype(float).values
        cols = ["#6366f1" if i in self._selected_entities
                else "#fbbf24" if i == self._hover_idx
                else "#94a3b8" for i in range(len(ex))]
        ecols = ["#312e81" if i in self._selected_entities else "#64748b"
                 for i in range(len(ex))]
        ax.scatter(ex, ey, s=sizes, c=cols, edgecolors=ecols, lw=1.2, alpha=.8, zorder=5)
        for i in np.argsort(-totals)[:15]:
            if i < len(ef):
                ax.annotate(ef.iloc[i][label], (ex[i], ey[i]), xytext=(5, 5),
                            textcoords="offset points", fontsize=7, color="#475569",
                            alpha=.85, zorder=6)
        for j, g in enumerate(groups):
            if j < len(gx):
                gc = _GROUP_COLOURS[j % len(_GROUP_COLOURS)]
                ax.scatter(gx[j], gy[j], s=250, c=gc, marker="D",
                           edgecolors="white", lw=2, zorder=10)
                ax.annotate(g, (gx[j], gy[j]), xytext=(8, 8), textcoords="offset points",
                            fontsize=11, fontweight="bold", color=gc, zorder=11,
                            bbox=dict(boxstyle="round,pad=.2", fc="white", ec=gc, alpha=.85))
        ax.axhline(0, color="#e2e8f0", lw=.8, zorder=0)
        ax.axvline(0, color="#e2e8f0", lw=.8, zorder=0)
        i1 = i2 = 0.0
        for _, r in inf.iterrows():
            if pd.notna(r.get("Dim 1")): i1 = r["Dim 1"]
            if pd.notna(r.get("Dim 2")): i2 = r["Dim 2"]
        ax.set_xlabel(f"Dim 1 ({i1:.1f}% inertia)"); ax.set_ylabel(f"Dim 2 ({i2:.1f}% inertia)")
        ax.set_title("Correspondence Analysis Biplot", fontsize=12, pad=10)
        ax.grid(True, alpha=.2, ls="--"); self.fig.tight_layout()
        self._viz_meta = {"type": "biplot", "ax": ax, "ex": ex, "ey": ey,
                          "gx": gx, "gy": gy, "ent_names": ef[label].tolist(),
                          "grp_names": gf[label].tolist()}

    # ── 2  Balloon Plot (cell-level selection, residual threshold) ──
    def _draw_balloon(self):
        chi = self._chi_df; cont = self._contingency_df
        groups = self._group_names; label = self._entity_label
        ax = self.fig.add_subplot(111)
        if chi is None or chi.empty:
            ax.text(.5, .5, "Chi-square not computed", ha="center", va="center",
                    fontsize=12, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off(); self._viz_meta = {"type": "balloon"}; return
        dr = chi[~chi[label].str.startswith("[", na=False)].reset_index(drop=True)
        items = dr[label].tolist(); ni = len(items); ng = len(groups)
        if ni == 0:
            ax.text(.5, .5, "No entities", ha="center", va="center",
                    fontsize=12, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off(); self._viz_meta = {"type": "balloon"}; return

        resid = np.zeros((ni, ng))
        counts = np.zeros((ni, ng), dtype=int)
        for j, g in enumerate(groups):
            col = f"{g} Std. residual"
            if col in dr.columns:
                resid[:, j] = pd.to_numeric(dr[col], errors="coerce").fillna(0).values
            obs_col = f"{g} Observed"
            if obs_col in dr.columns:
                counts[:, j] = pd.to_numeric(dr[obs_col], errors="coerce").fillna(0).astype(int).values
        mx = max(np.abs(resid).max(), .01)
        thresh = self.min_abs_residual
        visible = np.abs(resid) >= thresh  # which bubbles to show

        for i in range(ni):
            for j in range(ng):
                if not visible[i, j]: continue
                v = resid[i, j]
                sz = abs(v) / mx * 600 + 20
                clr = "#2563eb" if v > 0 else "#dc2626"
                alp = min(.3 + abs(v) / mx * .6, .9)
                is_sel = (items[i], groups[j]) in self._selected_cells
                ec = "#6366f1" if is_sel else "white"
                lw = 3 if is_sel else .5
                ax.scatter(j, i, s=sz, c=clr, alpha=alp,
                           edgecolors=ec, linewidths=lw, zorder=5)

        # hover highlight
        hi, hj = self._hover_cell
        if 0 <= hi < ni and 0 <= hj < ng and visible[hi, hj]:
            ax.add_patch(mpatches.Rectangle(
                (hj - .45, hi - .45), .9, .9,
                lw=2, ec="#fbbf24", fc="#fbbf24", alpha=.18, zorder=1))

        ax.set_xticks(range(ng)); ax.set_xticklabels(groups, fontsize=10, fontweight="bold")
        ax.set_yticks(range(ni))
        ax.set_yticklabels(items, fontsize=max(6, min(9, 300 // max(ni, 1))))
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xlim(-.6, ng - .4); ax.set_ylim(ni - .5, -.5)
        ax.grid(True, alpha=.15, ls="-")
        ax.legend(handles=[
            mpatches.Patch(color="#2563eb", label="Over-represented (+)"),
            mpatches.Patch(color="#dc2626", label="Under-represented (−)")],
            loc="lower right", fontsize=8, framealpha=.85)
        title = "Association Plot (Std. Residuals)"
        if thresh > 0:
            title += f"  [|r| ≥ {thresh:.1f}]"
        ax.set_title(title, fontsize=12, pad=12); self.fig.tight_layout()
        self._viz_meta = {"type": "balloon", "n_items": ni, "n_groups": ng,
                          "items": items, "groups": groups, "resid": resid,
                          "counts": counts, "visible": visible, "ax": ax}

    # ── 3  Bipartite Graph ──
    def _draw_bipartite(self):
        items, mat, groups = self._cont_data()
        ni, ng = mat.shape
        ax = self.fig.add_subplot(111)
        if ni == 0:
            ax.text(.5, .5, "No entities", ha="center", va="center",
                    fontsize=12, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off(); self._viz_meta = {"type": "bipartite"}; return

        x_left, x_right = 0.0, 1.0
        totals = mat.sum(1)
        max_t = max(totals.max(), 1)

        # Entity y positions
        y_ent = np.linspace(0, 1, ni)
        # Group y positions
        y_grp = np.linspace(0, 1, ng) if ng > 1 else [0.5]

        # Draw connections (behind nodes)
        max_count = max(mat.max(), 1)
        for i in range(ni):
            for j in range(ng):
                c = mat[i, j]
                if c == 0: continue
                lw = 0.5 + 5 * (c / max_count)
                gc = _GROUP_COLOURS[j % len(_GROUP_COLOURS)]
                is_sel = i in self._selected_entities
                alp = .7 if is_sel else .25
                # Bezier curve
                n_pts = 50; t = np.linspace(0, 1, n_pts)
                h = 3 * t ** 2 - 2 * t ** 3
                xs = x_left + (x_right - x_left) * t
                ys = y_ent[i] + (y_grp[j] - y_ent[i]) * h
                ax.plot(xs, ys, color=gc, lw=lw, alpha=alp, zorder=2,
                        solid_capstyle="round")

        # Entity nodes
        node_sizes = 30 + 250 * totals / max_t
        ent_cols = ["#6366f1" if i in self._selected_entities
                    else "#fbbf24" if i == self._hover_idx
                    else "#475569" for i in range(ni)]
        ent_ec = ["#312e81" if i in self._selected_entities else "#334155"
                  for i in range(ni)]
        ax.scatter([x_left] * ni, y_ent, s=node_sizes,
                   c=ent_cols, edgecolors=ent_ec, lw=1.5, zorder=5)
        # Entity labels
        max_labels = min(ni, 40)
        for i in range(max_labels):
            ax.text(x_left - .03, y_ent[i], items[i],
                    ha="right", va="center", fontsize=max(6, min(8, 250 // max(ni, 1))),
                    color="#334155")

        # Group nodes
        for j in range(ng):
            gc = _GROUP_COLOURS[j % len(_GROUP_COLOURS)]
            gs = mat[:, j].sum()
            sz = 80 + 300 * gs / max(mat.sum(), 1)
            ax.scatter(x_right, y_grp[j], s=sz, c=gc, marker="s",
                       edgecolors="white", lw=2, zorder=5)
            ax.text(x_right + .03, y_grp[j], groups[j],
                    ha="left", va="center", fontsize=11,
                    fontweight="bold", color=gc)

        ax.set_xlim(-.35, 1.3); ax.set_ylim(-.05, 1.05)
        ax.set_axis_off()
        ax.set_title("Bipartite Graph (Entity ↔ Group)", fontsize=12, pad=10)
        self.fig.tight_layout()
        self._viz_meta = {"type": "bipartite", "ax": ax,
                          "items": items, "groups": groups,
                          "y_ent": y_ent, "y_grp": np.array(y_grp),
                          "x_left": x_left, "x_right": x_right,
                          "n_items": ni, "n_groups": ng}

    # ── 4  Sankey Diagram ──
    def _draw_sankey(self):
        items, mat, groups = self._cont_data()
        ni, ng = mat.shape
        ax = self.fig.add_subplot(111)
        if ni == 0:
            ax.text(.5, .5, "No entities", ha="center", va="center",
                    fontsize=12, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off(); self._viz_meta = {"type": "sankey"}; return

        total = mat.sum()
        if total == 0:
            ax.set_axis_off(); self._viz_meta = {"type": "sankey"}; return

        x_left, x_right = 0.15, 0.85
        bar_w = 0.025
        ent_totals = mat.sum(1)
        grp_totals = mat.sum(0)

        # Layout: stack vertically with small gaps
        e_gap = 0.004 * ni
        g_gap = 0.012 * ng
        avail = 1.0
        e_scale = (avail - e_gap * max(ni - 1, 0)) / max(total, 1)
        g_scale = (avail - g_gap * max(ng - 1, 0)) / max(total, 1)

        # Entity y ranges
        ey = {}; y = 0
        for i in range(ni):
            h = ent_totals[i] * e_scale
            ey[i] = (y, y + h); y += h + e_gap
        # Group y ranges
        gy = {}; y = 0
        for j in range(ng):
            h = grp_totals[j] * g_scale
            gy[j] = (y, y + h); y += h + g_gap

        # Draw flow bands
        e_cursor = {i: ey[i][0] for i in range(ni)}
        g_cursor = {j: gy[j][0] for j in range(ng)}
        bands = []  # for hit-testing

        for i in range(ni):
            for j in range(ng):
                c = mat[i, j]
                if c <= 0: continue
                lt = e_cursor[i]; lb = lt + c * e_scale; e_cursor[i] = lb
                rt = g_cursor[j]; rb = rt + c * g_scale; g_cursor[j] = rb
                gc = _GROUP_COLOURS[j % len(_GROUP_COLOURS)]
                is_sel = (items[i], groups[j]) in self._selected_cells
                alp = .75 if is_sel else .35
                elw = 1.5 if is_sel else 0

                # S-curve band
                n_pts = 60; t = np.linspace(0, 1, n_pts)
                h = 3 * t ** 2 - 2 * t ** 3
                xs = x_left + (x_right - x_left) * t
                yt = lt + (rt - lt) * h
                yb = lb + (rb - lb) * h
                ax.fill_between(xs, yb, yt, color=gc, alpha=alp, zorder=3)
                if is_sel:
                    ax.plot(xs, yt, color="#6366f1", lw=1.2, zorder=4)
                    ax.plot(xs, yb, color="#6366f1", lw=1.2, zorder=4)

                bands.append({"entity": items[i], "group": groups[j],
                               "ei": i, "gj": j,
                               "lt": lt, "lb": lb, "rt": rt, "rb": rb})

        # hover highlight for band
        hi_cell = self._hover_cell
        if hi_cell != (-1, -1):
            hn, gn = hi_cell
            for b in bands:
                if b["ei"] == hn and b["gj"] == gn:
                    n_pts = 60; t = np.linspace(0, 1, n_pts)
                    h = 3 * t ** 2 - 2 * t ** 3
                    xs = x_left + (x_right - x_left) * t
                    yt = b["lt"] + (b["rt"] - b["lt"]) * h
                    yb = b["lb"] + (b["rb"] - b["lb"]) * h
                    ax.fill_between(xs, yb, yt, color="#fbbf24", alpha=.25, zorder=2)
                    break

        # Entity bars (left)
        for i in range(ni):
            ys, ye = ey[i]
            is_sel = i in self._selected_entities or \
                     any((items[i], g) in self._selected_cells for g in groups)
            ec = "#6366f1" if is_sel else "#475569"
            lw = 2 if is_sel else .8
            ax.add_patch(mpatches.FancyBboxPatch(
                (x_left - bar_w, ys), bar_w, ye - ys,
                boxstyle="round,pad=.001", fc="#475569", ec=ec, lw=lw, zorder=5))
            if ye - ys > .008:
                ax.text(x_left - bar_w - .008, (ys + ye) / 2, items[i],
                        ha="right", va="center",
                        fontsize=max(5, min(8, 250 // max(ni, 1))),
                        color="#334155", clip_on=True)

        # Group bars (right)
        for j in range(ng):
            ys, ye = gy[j]
            gc = _GROUP_COLOURS[j % len(_GROUP_COLOURS)]
            ax.add_patch(mpatches.FancyBboxPatch(
                (x_right, ys), bar_w, ye - ys,
                boxstyle="round,pad=.001", fc=gc, ec="white", lw=1.5, zorder=5))
            ax.text(x_right + bar_w + .008, (ys + ye) / 2, groups[j],
                    ha="left", va="center", fontsize=11, fontweight="bold",
                    color=gc)

        ymax = max(max(v[1] for v in ey.values()), max(v[1] for v in gy.values()))
        ax.set_xlim(-.02, 1.02); ax.set_ylim(ymax + .02, -.02)
        ax.set_axis_off()
        ax.set_title("Sankey Diagram (Entity → Group)", fontsize=12, pad=10)
        self.fig.tight_layout()
        self._viz_meta = {"type": "sankey", "ax": ax, "bands": bands,
                          "items": items, "groups": groups,
                          "ey": ey, "gy": gy,
                          "x_left": x_left, "x_right": x_right,
                          "n_items": ni, "n_groups": ng}

    # =========================================================================
    # INTERACTION — hover
    # =========================================================================

    def _on_mpl_hover(self, event):
        meta = self._viz_meta
        if meta is None or event.inaxes is None:
            QToolTip.hideText(); return
        vt = meta.get("type")
        old_hover = self._hover_idx
        old_cell = self._hover_cell

        if vt == "heatmap":
            self._hover_grid(event, meta, mode="row")
        elif vt == "balloon":
            self._hover_grid(event, meta, mode="cell")
        elif vt == "biplot":
            self._hover_scatter(event, meta)
        elif vt == "bipartite":
            self._hover_bipartite(event, meta)
        elif vt == "sankey":
            self._hover_sankey(event, meta)

        if self._hover_idx != old_hover or self._hover_cell != old_cell:
            self._update_visualization()

    def _hover_grid(self, event, meta, mode):
        ni = meta.get("n_items", 0); ng = meta.get("n_groups", 0)
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            self._hover_idx = -1; self._hover_cell = (-1, -1)
            QToolTip.hideText(); return
        j, i = int(round(x)), int(round(y))
        if not (0 <= i < ni and 0 <= j < ng):
            self._hover_idx = -1; self._hover_cell = (-1, -1)
            QToolTip.hideText(); return

        if mode == "row":
            self._hover_idx = i; self._hover_cell = (-1, -1)
        else:  # cell
            visible = meta.get("visible")
            if visible is not None and not visible[i, j]:
                self._hover_idx = -1; self._hover_cell = (-1, -1)
                QToolTip.hideText(); return
            self._hover_idx = -1; self._hover_cell = (i, j)

        text = self._tooltip_grid(meta, i, j)
        pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
        QToolTip.showText(pos, text)

    def _tooltip_grid(self, meta, i, j):
        it = meta["items"][i]; g = meta["groups"][j]
        if meta["type"] == "heatmap":
            v = int(meta["mat"][i, j]); p = meta["normed"][i, j] * 100
            return f"<b>{it}</b><br>Group: <b>{g}</b><br>Count: {v}<br>Row %: {p:.1f}%"
        else:  # balloon
            r = meta["resid"][i, j]
            c = int(meta.get("counts", np.zeros_like(meta["resid"]))[i, j])
            d = "over" if r > 0 else "under"
            return (f"<b>{it}</b><br>Group: <b>{g}</b><br>"
                    f"Count: <b>{c}</b><br>Std. residual: {r:.3f}<br>({d}-represented)")

    def _hover_scatter(self, event, meta):
        ex = meta.get("ex"); ey = meta.get("ey")
        if ex is None or event.xdata is None:
            self._hover_idx = -1; QToolTip.hideText(); return
        d = (ex - event.xdata) ** 2 + (ey - event.ydata) ** 2
        mi = int(np.argmin(d)); ax = meta["ax"]
        xl = ax.get_xlim(); th = ((xl[1] - xl[0]) * .03) ** 2
        if d[mi] < th:
            self._hover_idx = mi
            n = meta["ent_names"][mi]
            text = f"<b>{n}</b><br>Dim 1: {ex[mi]:.4f}<br>Dim 2: {ey[mi]:.4f}"
            pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
            QToolTip.showText(pos, text)
        else:
            self._hover_idx = -1; QToolTip.hideText()

    def _hover_bipartite(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None:
            self._hover_idx = -1; QToolTip.hideText(); return
        ye = meta["y_ent"]; ni = meta["n_items"]
        xl = meta["x_left"]
        if abs(x - xl) < .06:
            dists = np.abs(ye - y)
            mi = int(np.argmin(dists))
            if dists[mi] < .03 * max(1, 1 / max(ni, 1) * 10):
                self._hover_idx = mi
                it = meta["items"][mi]
                pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
                QToolTip.showText(pos, f"<b>{it}</b>")
                return
        self._hover_idx = -1; QToolTip.hideText()

    def _hover_sankey(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None:
            self._hover_cell = (-1, -1); QToolTip.hideText(); return
        xl, xr = meta["x_left"], meta["x_right"]
        if not (xl <= x <= xr):
            self._hover_cell = (-1, -1); QToolTip.hideText(); return
        t = (x - xl) / (xr - xl)
        h = 3 * t ** 2 - 2 * t ** 3
        for b in meta["bands"]:
            yt = b["lt"] + (b["rt"] - b["lt"]) * h
            yb = b["lb"] + (b["rb"] - b["lb"]) * h
            if yb <= y <= yt:
                self._hover_cell = (b["ei"], b["gj"])
                text = (f"<b>{b['entity']}</b> → <b>{b['group']}</b><br>"
                        f"Count: {int(self._contingency_df[b['group']].iloc[b['ei']])}")
                pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
                QToolTip.showText(pos, text); return
        self._hover_cell = (-1, -1); QToolTip.hideText()

    # =========================================================================
    # INTERACTION — click
    # =========================================================================

    def _on_mpl_click(self, event):
        meta = self._viz_meta
        if meta is None or event.inaxes is None or event.button != 1:
            return
        vt = meta.get("type")
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)

        if vt in ("heatmap", "bipartite"):
            idx = self._hit_entity(event, meta)
            if idx < 0: return
            self._toggle_entity(idx, ctrl)

        elif vt == "biplot":
            idx = self._hit_scatter(event, meta)
            if idx < 0: return
            self._toggle_entity(idx, ctrl)

        elif vt == "balloon":
            cell = self._hit_balloon_cell(event, meta)
            if cell is None: return
            self._toggle_cell(cell, ctrl)

        elif vt == "sankey":
            cell = self._hit_sankey_band(event, meta)
            if cell is None: return
            self._toggle_cell(cell, ctrl)

        self._update_visualization()
        self._sync_table_selection()
        self._update_sel_label()
        self._send_selected_documents()

    def _hit_entity(self, event, meta):
        """Hit-test for entity index in heatmap or bipartite."""
        vt = meta["type"]
        if vt == "heatmap":
            y = event.ydata
            if y is None: return -1
            i = int(round(y))
            return i if 0 <= i < meta["n_items"] else -1
        elif vt == "bipartite":
            x, y = event.xdata, event.ydata
            if x is None: return -1
            ye = meta["y_ent"]; xl = meta["x_left"]
            if abs(x - xl) > .08: return -1
            d = np.abs(ye - y); mi = int(np.argmin(d))
            return mi if d[mi] < .04 else -1
        return -1

    def _hit_scatter(self, event, meta):
        ex = meta.get("ex"); ey = meta.get("ey")
        if ex is None or event.xdata is None: return -1
        d = (ex - event.xdata) ** 2 + (ey - event.ydata) ** 2
        mi = int(np.argmin(d)); ax = meta["ax"]
        xl = ax.get_xlim(); th = ((xl[1] - xl[0]) * .04) ** 2
        return mi if d[mi] < th else -1

    def _hit_balloon_cell(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None or y is None: return None
        j, i = int(round(x)), int(round(y))
        ni, ng = meta["n_items"], meta["n_groups"]
        if not (0 <= i < ni and 0 <= j < ng): return None
        visible = meta.get("visible")
        if visible is not None and not visible[i, j]: return None
        return (meta["items"][i], meta["groups"][j])

    def _hit_sankey_band(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None or y is None: return None
        xl, xr = meta["x_left"], meta["x_right"]
        if not (xl <= x <= xr): return None
        t = (x - xl) / (xr - xl); h = 3 * t ** 2 - 2 * t ** 3
        for b in meta["bands"]:
            yt = b["lt"] + (b["rt"] - b["lt"]) * h
            yb = b["lb"] + (b["rb"] - b["lb"]) * h
            if yb <= y <= yt:
                return (b["entity"], b["group"])
        return None

    def _toggle_entity(self, idx, extend):
        self._selected_cells.clear()
        if extend:
            self._selected_entities.symmetric_difference_update({idx})
        else:
            self._selected_entities = set() if self._selected_entities == {idx} else {idx}

    def _toggle_cell(self, cell, extend):
        self._selected_entities.clear()
        if extend:
            self._selected_cells.symmetric_difference_update({cell})
        else:
            self._selected_cells = set() if self._selected_cells == {cell} else {cell}

    def _sync_table_selection(self):
        tw = self.tab_contingency; tw.clearSelection()
        for idx in sorted(self._selected_entities):
            if 0 <= idx < tw.rowCount():
                tw.selectRow(idx)
        # For cell selection, highlight matching entity rows
        if self._selected_cells and self._contingency_df is not None:
            label = self._entity_label
            sel_names = {c[0] for c in self._selected_cells}
            for r in range(tw.rowCount()):
                item = tw.item(r, 0)
                if item and item.text() in sel_names:
                    tw.selectRow(r)

    def _update_sel_label(self):
        if self._selected_cells:
            cont = self._contingency_df
            label = self._entity_label
            parts = []
            total_docs = 0
            for e, g in sorted(self._selected_cells):
                # Look up count from contingency
                cnt = 0
                if cont is not None and g in cont.columns:
                    row = cont[cont[label] == e]
                    if len(row) > 0:
                        cnt = int(row[g].iloc[0])
                total_docs += cnt
                parts.append(f"{e}↔{g}")
            if len(parts) <= 2:
                self.sel_label.setText(f"Selected: {', '.join(parts)} ({total_docs} docs)")
            else:
                self.sel_label.setText(f"Selected: {', '.join(parts[:2])} + {len(parts)-2} more ({total_docs} docs)")
        elif self._selected_entities:
            cont = self._contingency_df; label = self._entity_label
            names = []
            for idx in sorted(self._selected_entities):
                if cont is not None and idx < len(cont):
                    names.append(str(cont.iloc[idx][label]))
            if len(names) <= 3:
                self.sel_label.setText(f"Selected: {', '.join(names)}")
            else:
                self.sel_label.setText(f"Selected: {', '.join(names[:3])} + {len(names)-3} more")
        else:
            self.sel_label.setText("")

    # =========================================================================
    # SELECTED DOCUMENTS OUTPUT
    # =========================================================================

    def _send_selected_documents(self):
        if self._data is None:
            self.Outputs.selected_data.send(None); return

        df = self._df; gm = self._gm_df; cont = self._contingency_df
        label = self._entity_label; ecol = self._entity_col_name
        cfg_key = list(ENTITY_CONFIGS.keys())[self.entity_type_idx]
        vt = ENTITY_CONFIGS[cfg_key]["value_type"]
        sep = self._detect_separator(df)

        if self._selected_cells and ecol:
            # Cell-level: entity AND group
            mask = pd.Series(False, index=df.index)
            for ent_name, grp_name in self._selected_cells:
                ent_mask = self._entity_mask(df, ecol, ent_name, vt, sep)
                grp_mask = gm[grp_name].astype(bool) if grp_name in gm.columns else False
                mask |= (ent_mask & grp_mask)
            indices = np.where(mask.values)[0]
        elif self._selected_entities and cont is not None and ecol:
            # Entity-level
            sel_names = set()
            for idx in self._selected_entities:
                if 0 <= idx < len(cont):
                    sel_names.add(str(cont.iloc[idx][label]))
            mask = pd.Series(False, index=df.index)
            for n in sel_names:
                mask |= self._entity_mask(df, ecol, n, vt, sep)
            indices = np.where(mask.values)[0]
        else:
            self.Outputs.selected_data.send(None); return

        if len(indices) == 0:
            self.Outputs.selected_data.send(None)
        else:
            self.Outputs.selected_data.send(self._data[indices])

    def _entity_mask(self, df, ecol, name, vtype, sep):
        """Boolean mask of rows containing *name* in *ecol*."""
        if vtype == "list":
            m = df[ecol].dropna().astype(str).apply(
                lambda x: name in {s.strip() for s in x.split(sep)})
            return m.reindex(df.index, fill_value=False)
        elif vtype == "text":
            return pd.Series(True, index=df.index)  # can't map back
        return df[ecol].astype(str).str.strip() == name

    # =========================================================================
    # TABLE DISPLAY
    # =========================================================================

    def _display_all(self):
        for tw, d in [(self.tab_contingency, self._contingency_df),
                      (self.tab_diversity, self._diversity_df),
                      (self.tab_ca, self._ca_df),
                      (self.tab_chi, self._chi_df),
                      (self.tab_svd, self._svd_df),
                      (self.tab_lr, self._logratio_df)]:
            self._fill_table(tw, d)

    @staticmethod
    def _fill_table(tw, df):
        tw.setSortingEnabled(False); tw.clear()
        if df is None or df.empty:
            tw.setRowCount(0); tw.setColumnCount(0); return
        nr, nc = df.shape; tw.setRowCount(nr); tw.setColumnCount(nc)
        tw.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(nr):
            for c in range(nc):
                v = df.iloc[r, c]
                if pd.isna(v): it = QTableWidgetItem("")
                elif isinstance(v, float):
                    it = QTableWidgetItem(str(int(v)) if v == int(v) and abs(v) < 1e12
                                          else f"{v:.4f}")
                    it.setData(Qt.UserRole, float(v))
                else: it = QTableWidgetItem(str(v))
                tw.setItem(r, c, it)
        tw.resizeColumnsToContents(); tw.setSortingEnabled(True)

    def _clear_results(self):
        self._contingency_df = self._diversity_df = self._ca_df = None
        self._chi_df = self._svd_df = self._logratio_df = self._filtered_df = None
        self._selected_entities.clear(); self._selected_cells.clear()
        self._hover_idx = -1; self._hover_cell = (-1, -1); self._viz_meta = None
        self.sel_label.setText("")
        for tw in [self.tab_contingency, self.tab_diversity, self.tab_ca,
                   self.tab_chi, self.tab_svd, self.tab_lr]:
            tw.clear(); tw.setRowCount(0); tw.setColumnCount(0)
        if HAS_MPL and self.fig is not None:
            self.fig.clear(); self.canvas.draw_idle()

    # =========================================================================
    # OUTPUTS
    # =========================================================================

    def _send_outputs(self):
        self.Outputs.contingency.send(self._df_to_table(self._contingency_df))
        self.Outputs.diversity.send(self._df_to_table(self._diversity_df))
        self.Outputs.correspondence.send(self._df_to_table(self._ca_df))
        self.Outputs.chi_square.send(self._df_to_table(self._chi_df))
        self.Outputs.svd_out.send(self._df_to_table(self._svd_df))
        self.Outputs.log_ratio.send(self._df_to_table(self._logratio_df))
        self.Outputs.filtered.send(self._df_to_table(self._filtered_df))
        if not self._selected_entities and not self._selected_cells:
            self.Outputs.selected_data.send(None)

    # =========================================================================
    # EXPORT
    # =========================================================================

    def _export_results(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export", "", "Excel (*.xlsx);;CSV (*.csv)")
        if not path: return
        try:
            if path.endswith(".xlsx"):
                with pd.ExcelWriter(path, engine="openpyxl") as w:
                    for nm, d in [("Contingency", self._contingency_df),
                                  ("Diversity", self._diversity_df),
                                  ("Correspondence", self._ca_df),
                                  ("Chi-square", self._chi_df),
                                  ("SVD", self._svd_df),
                                  ("Log-ratio", self._logratio_df)]:
                        if d is not None:
                            d.to_excel(w, sheet_name=nm, index=False)
            elif self._contingency_df is not None:
                self._contingency_df.to_csv(path, index=False)
        except Exception:
            logger.exception("export")

    # =========================================================================
    # STATE
    # =========================================================================

    def onDeleteWidget(self):
        if self._worker and self._worker.isRunning():
            self._worker.quit(); self._worker.wait(2000)
        self.splitter_state = bytes(self.splitter.saveState())
        super().onDeleteWidget()

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _find_column(df, candidates):
        cl = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c in df.columns: return c
            if c.lower() in cl: return cl[c.lower()]
        return None

    @staticmethod
    def _detect_separator(df):
        cols = ["Authors", "Author Keywords", "Index Keywords",
                "Affiliations", "References"]
        sc = {"; ": 0, ";": 0, "|": 0}
        for c in cols:
            if c not in df.columns: continue
            s = df[c].dropna().head(200).astype(str)
            for sep in sc:
                if s.str.contains(sep, regex=False).mean() > .15:
                    sc[sep] += 1
        if sc["|"] >= sc["; "] and sc["|"] > 0: return "|"
        if sc["; "] > 0: return "; "
        if sc[";"] > 0: return ";"
        return "; "

    def _table_to_df(self, table):
        data = {}; domain = table.domain
        for var in list(domain.attributes) + list(domain.class_vars):
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else None for v in col]
            else: data[var.name] = col
        for var in domain.metas:
            col = table[:, var].metas.flatten()
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)]
                                  if not (isinstance(v, float) and np.isnan(v)) else None
                                  for v in col]
            elif isinstance(var, StringVariable):
                data[var.name] = [str(v) if v is not None else "" for v in col]
            else: data[var.name] = col
        return pd.DataFrame(data)

    @staticmethod
    def _df_to_table(df):
        if df is None or df.empty: return None
        attrs, metas = [], []
        for c in df.columns:
            (attrs if pd.api.types.is_numeric_dtype(df[c]) else metas).append(
                ContinuousVariable(str(c)) if pd.api.types.is_numeric_dtype(df[c])
                else StringVariable(str(c)))
        domain = Domain(attrs, metas=metas)
        X = np.zeros((len(df), len(attrs)), float)
        M = np.zeros((len(df), len(metas)), object)
        for j, v in enumerate(attrs):
            X[:, j] = pd.to_numeric(df[v.name], errors="coerce").fillna(np.nan)
        for j, v in enumerate(metas):
            M[:, j] = df[v.name].fillna("").astype(str)
        return Table.from_numpy(domain, X, metas=M)
