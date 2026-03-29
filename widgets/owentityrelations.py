# -*- coding: utf-8 -*-
"""
Entity Relationships Widget
============================
Orange widget for analysing co-occurrence relationships between two
entity types (e.g., Authors × Keywords, Journals × Countries).

Statistical analyses
--------------------
Co-occurrence matrix · Chi-square · Correspondence Analysis · Association measures

Visualisations (all with hover tooltips + click selection)
----------------------------------------------------------
Heatmap · CA Biplot · Balloon Plot · Network Graph · Sankey Diagram

Selection model
---------------
* Heatmap / Biplot / Network  →  entity-level selection
* Balloon / Sankey            →  cell-level (row entity × column entity pair)
"""

import logging
import re
from typing import Optional, List, Set, Tuple, Dict
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
    QToolButton, QToolTip, QTextEdit, QScrollArea,
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
_VIZ_HEATMAP = 0
_VIZ_BIPLOT = 1
_VIZ_BALLOON = 2
_VIZ_NETWORK = 3
_VIZ_SANKEY = 4
VIZ_NAMES = ["Heatmap", "CA Biplot", "Balloon Plot",
             "Network Graph", "Sankey Diagram"]

_COLOURS = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
    "#db2777", "#0891b2", "#65a30d", "#ea580c", "#6d28d9",
    "#475569", "#0d9488", "#be123c", "#4f46e5", "#ca8a04",
]

ENTITY_CONFIGS = {
    "sources": {
        "label": "Sources (Journals)",
        "entity_col_alts": ["Source title", "Source", "Journal", "source", "SO"],
        "entity_label": "Source",
        "value_type": "string",
    },
    "authors": {
        "label": "Authors",
        "entity_col_alts": ["Authors", "Author", "authors", "AU"],
        "entity_label": "Author",
        "value_type": "list",
    },
    "author_keywords": {
        "label": "Author Keywords",
        "entity_col_alts": [
            "Processed Author Keywords", "Author Keywords",
            "author keywords", "DE", "Keywords",
        ],
        "entity_label": "Keyword",
        "value_type": "list",
    },
    "index_keywords": {
        "label": "Index Keywords",
        "entity_col_alts": [
            "Index Keywords", "Keywords Plus",
            "index keywords", "ID", "KeywordsPlus",
        ],
        "entity_label": "Keyword",
        "value_type": "list",
    },
    "countries": {
        "label": "Countries",
        "entity_col_alts": [
            "Countries of Authors", "All Countries", "Country",
            "CA Country", "countries",
        ],
        "entity_label": "Country",
        "value_type": "list",
    },
    "affiliations": {
        "label": "Affiliations",
        "entity_col_alts": ["Affiliations", "Affiliation", "C1", "C3"],
        "entity_label": "Affiliation",
        "value_type": "list",
    },
    "references": {
        "label": "References",
        "entity_col_alts": [
            "References", "Cited References", "CR", "cited references",
        ],
        "entity_label": "Reference",
        "value_type": "list",
    },
    "subject_areas": {
        "label": "Subject Areas",
        "entity_col_alts": [
            "Subject Area", "Subject Areas", "Research Areas",
            "WC", "SC", "Topics",
        ],
        "entity_label": "Subject",
        "value_type": "list",
    },
    "document_types": {
        "label": "Document Types",
        "entity_col_alts": [
            "Document Type", "Document type", "type", "DT", "Type",
        ],
        "entity_label": "DocType",
        "value_type": "string",
    },
    "years": {
        "label": "Publication Years",
        "entity_col_alts": ["Year", "Publication Year", "PY", "year"],
        "entity_label": "Year",
        "value_type": "string",
    },
}


# =============================================================================
# WORKER THREAD
# =============================================================================

class RelWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self._func = func

    def run(self):
        try:
            self.finished.emit((self._func(), None))
        except Exception as e:
            logger.exception("RelWorker")
            self.finished.emit((None, str(e)))


# =============================================================================
# WIDGET
# =============================================================================

class OWEntityRelations(OWWidget):

    name = "Entity Relationships"
    description = (
        "Analyse co-occurrence relationships between two entity types: "
        "authors×keywords, journals×countries, etc. "
        "With heatmap, CA biplot, balloon, network, and Sankey visualisations."
    )
    icon = "icons/entity_relations.svg"
    priority = 36
    keywords = [
        "co-occurrence", "relationship", "entity", "matrix", "chi-square",
        "correspondence", "network", "heatmap", "sankey", "bipartite",
    ]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    class Outputs:
        matrix = Output("Co-occurrence Matrix", Table)
        chi_square = Output("Chi-square", Table)
        correspondence = Output("Correspondence", Table)
        associations = Output("Association Measures", Table)
        selected_data = Output("Selected Documents", Table, default=True)

    # ── settings ──
    row_entity_idx = Setting(1)  # Authors
    col_entity_idx = Setting(2)  # Author Keywords
    row_top_n = Setting(20)
    col_top_n = Setting(20)
    row_min_freq = Setting(3)
    col_min_freq = Setting(3)
    row_include_text = Setting("")
    row_exclude_text = Setting("")
    col_include_text = Setting("")
    col_exclude_text = Setting("")
    row_regex_include = Setting("")
    row_regex_exclude = Setting("")
    col_regex_include = Setting("")
    col_regex_exclude = Setting("")
    row_max_items = Setting(0)
    col_max_items = Setting(0)
    compute_chi = Setting(True)
    compute_ca = Setting(True)
    compute_assoc = Setting(True)
    auto_apply = Setting(True)
    viz_type_idx = Setting(0)
    min_abs_residual = Setting(0.0)
    min_edge_weight = Setting(1)
    splitter_state = Setting(b"")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_column = Msg("Column not found: {}")
        compute_error = Msg("{}")

    class Warning(OWWidget.Warning):
        few_rows = Msg("Only {} row entities")
        few_cols = Msg("Only {} column entities")
        same_entity = Msg("Same entity type for rows and columns")

    class Information(OWWidget.Information):
        computing = Msg("Computing…")

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(self):
        super().__init__()

        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._worker: Optional[RelWorker] = None

        # result storage
        self._matrix_df: Optional[pd.DataFrame] = None
        self._chi_df: Optional[pd.DataFrame] = None
        self._ca_df: Optional[pd.DataFrame] = None
        self._assoc_df: Optional[pd.DataFrame] = None

        # entity info
        self._row_items: List[str] = []
        self._col_items: List[str] = []
        self._row_label: str = "Row"
        self._col_label: str = "Column"
        self._row_col_name: Optional[str] = None
        self._col_col_name: Optional[str] = None

        # visualisation state
        self._selected_rows: Set[int] = set()
        self._selected_cols: Set[int] = set()
        self._selected_cells: Set[Tuple[str, str]] = set()
        self._hover_row: int = -1
        self._hover_col: int = -1
        self._hover_cell: Tuple[int, int] = (-1, -1)
        self._mpl_cids: list = []
        self._viz_meta: Optional[dict] = None

        self._setup_control_area()
        self._setup_main_area()

    # =========================================================================
    # CONTROL AREA
    # =========================================================================

    def _setup_control_area(self):
        # Style for section headers
        section_style = """
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                color: #1e40af;
                border: 1px solid #bfdbfe;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background: #eff6ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """

        # ── Row Entity ──
        row_box = QGroupBox("📋 Row Entity")
        row_box.setStyleSheet(section_style)
        row_lo = QVBoxLayout(row_box)
        rf = QGridLayout(); rf.setColumnStretch(1, 1)
        rf.addWidget(QLabel("Entity:"), 0, 0)
        self.row_combo = QComboBox()
        for cfg in ENTITY_CONFIGS.values():
            self.row_combo.addItem(cfg["label"])
        self.row_combo.setCurrentIndex(self.row_entity_idx)
        self.row_combo.currentIndexChanged.connect(self._on_row_entity_changed)
        rf.addWidget(self.row_combo, 0, 1)
        rf.addWidget(QLabel("Top N:"), 1, 0)
        self.row_topn_spin = QSpinBox()
        self.row_topn_spin.setRange(5, 1000)
        self.row_topn_spin.setValue(self.row_top_n)
        self.row_topn_spin.valueChanged.connect(lambda v: setattr(self, "row_top_n", v))
        rf.addWidget(self.row_topn_spin, 1, 1)
        row_lo.addLayout(rf)
        self.controlArea.layout().addWidget(row_box)

        # ── Row Entity Filtering (collapsible) ──
        _, self._row_filter_content = self._make_collapsible(
            self.controlArea, "🔍 Row Entity Filtering", collapsed=True)
        self._setup_filter_section(self._row_filter_content, "row")

        # ── Column Entity ──
        col_box = QGroupBox("📋 Column Entity")
        col_box.setStyleSheet(section_style)
        col_lo = QVBoxLayout(col_box)
        cf = QGridLayout(); cf.setColumnStretch(1, 1)
        cf.addWidget(QLabel("Entity:"), 0, 0)
        self.col_combo = QComboBox()
        for cfg in ENTITY_CONFIGS.values():
            self.col_combo.addItem(cfg["label"])
        self.col_combo.setCurrentIndex(self.col_entity_idx)
        self.col_combo.currentIndexChanged.connect(self._on_col_entity_changed)
        cf.addWidget(self.col_combo, 0, 1)
        cf.addWidget(QLabel("Top N:"), 1, 0)
        self.col_topn_spin = QSpinBox()
        self.col_topn_spin.setRange(5, 1000)
        self.col_topn_spin.setValue(self.col_top_n)
        self.col_topn_spin.valueChanged.connect(lambda v: setattr(self, "col_top_n", v))
        cf.addWidget(self.col_topn_spin, 1, 1)
        col_lo.addLayout(cf)
        self.controlArea.layout().addWidget(col_box)

        # ── Column Entity Filtering (collapsible) ──
        _, self._col_filter_content = self._make_collapsible(
            self.controlArea, "🔍 Column Entity Filtering", collapsed=True)
        self._setup_filter_section(self._col_filter_content, "col")

        # ── Statistics ──
        stats_box = QGroupBox("📊 Statistics")
        stats_box.setStyleSheet(section_style)
        stats_lo = QVBoxLayout(stats_box)
        for attr, label in [
            ("compute_chi", "Chi-square Test"),
            ("compute_ca", "Correspondence Analysis"),
            ("compute_assoc", "Association Measures"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(getattr(self, attr))
            cb.toggled.connect(lambda v, a=attr: setattr(self, a, v))
            stats_lo.addWidget(cb)
        self.controlArea.layout().addWidget(stats_box)

        # ── Display Options (collapsible) ──
        _, self._disp_content = self._make_collapsible(
            self.controlArea, "🖥️ Display Options", collapsed=True)
        self.auto_apply_cb = QCheckBox("Auto apply on data change")
        self.auto_apply_cb.setChecked(self.auto_apply)
        self.auto_apply_cb.toggled.connect(lambda v: setattr(self, "auto_apply", v))
        self._disp_content.layout().addWidget(self.auto_apply_cb)

        # Min |residual| for balloon
        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("Min |residual|:"))
        self.min_resid_spin = QDoubleSpinBox()
        self.min_resid_spin.setRange(0.0, 20.0)
        self.min_resid_spin.setSingleStep(0.1)
        self.min_resid_spin.setDecimals(1)
        self.min_resid_spin.setValue(self.min_abs_residual)
        self.min_resid_spin.valueChanged.connect(self._on_min_resid_changed)
        r_row.addWidget(self.min_resid_spin)
        r_row.addStretch()
        self._disp_content.layout().addLayout(r_row)

        # Min edge weight for network
        e_row = QHBoxLayout()
        e_row.addWidget(QLabel("Min edge weight:"))
        self.min_edge_spin = QSpinBox()
        self.min_edge_spin.setRange(1, 100)
        self.min_edge_spin.setValue(self.min_edge_weight)
        self.min_edge_spin.valueChanged.connect(self._on_min_edge_changed)
        e_row.addWidget(self.min_edge_spin)
        e_row.addStretch()
        self._disp_content.layout().addLayout(e_row)

        # ── Buttons ──
        bl = QVBoxLayout()
        self.compute_btn = QPushButton("▶ Compute Relationships")
        self.compute_btn.clicked.connect(self._run_compute)
        self.compute_btn.setStyleSheet(
            "QPushButton{background:#2563eb;border:none;border-radius:4px;"
            "padding:10px 20px;color:white;font-weight:bold;font-size:13px}"
            "QPushButton:hover{background:#1d4ed8}"
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

    def _setup_filter_section(self, parent, prefix):
        """Setup filtering controls for row or column entity."""
        lo = parent.layout()

        # Include list
        inc_lbl = QLabel("Include only (one per line):")
        lo.addWidget(inc_lbl)
        inc_row = QHBoxLayout()
        inc_edit = QTextEdit()
        inc_edit.setMaximumHeight(60)
        inc_edit.setPlaceholderText("Entity1\nEntity2\n...")
        inc_edit.textChanged.connect(
            lambda e=inc_edit, p=prefix: setattr(self, f"{p}_include_text", e.toPlainText()))
        setattr(self, f"{prefix}_include_edit", inc_edit)
        inc_row.addWidget(inc_edit)
        inc_btn = QPushButton("⊞ Load")
        inc_btn.setMaximumWidth(60)
        inc_btn.clicked.connect(lambda _, p=prefix: self._load_filter_file(p, "include"))
        inc_row.addWidget(inc_btn)
        lo.addLayout(inc_row)

        # Exclude list
        exc_lbl = QLabel("Exclude (one per line):")
        lo.addWidget(exc_lbl)
        exc_row = QHBoxLayout()
        exc_edit = QTextEdit()
        exc_edit.setMaximumHeight(60)
        exc_edit.setPlaceholderText("Entity1\nEntity2\n...")
        exc_edit.textChanged.connect(
            lambda e=exc_edit, p=prefix: setattr(self, f"{p}_exclude_text", e.toPlainText()))
        setattr(self, f"{prefix}_exclude_edit", exc_edit)
        exc_row.addWidget(exc_edit)
        exc_btn = QPushButton("⊞ Load")
        exc_btn.setMaximumWidth(60)
        exc_btn.clicked.connect(lambda _, p=prefix: self._load_filter_file(p, "exclude"))
        exc_row.addWidget(exc_btn)
        lo.addLayout(exc_row)

        # Regex filters
        rg = QGridLayout()
        rg.addWidget(QLabel("Regex include:"), 0, 0)
        ri_edit = QLineEdit()
        ri_edit.setPlaceholderText("e.g., ^Smith|^Jones (regex pattern)")
        ri_edit.editingFinished.connect(
            lambda e=ri_edit, p=prefix: setattr(self, f"{p}_regex_include", e.text()))
        setattr(self, f"{prefix}_regex_inc_edit", ri_edit)
        rg.addWidget(ri_edit, 0, 1)
        rg.addWidget(QLabel("Regex exclude:"), 1, 0)
        re_edit = QLineEdit()
        re_edit.setPlaceholderText("e.g., Unknown|Anonymous")
        re_edit.editingFinished.connect(
            lambda e=re_edit, p=prefix: setattr(self, f"{p}_regex_exclude", e.text()))
        setattr(self, f"{prefix}_regex_exc_edit", re_edit)
        rg.addWidget(re_edit, 1, 1)
        rg.addWidget(QLabel("Min frequency:"), 2, 0)
        mf_spin = QSpinBox()
        mf_spin.setRange(1, 1000)
        mf_spin.setValue(getattr(self, f"{prefix}_min_freq"))
        mf_spin.valueChanged.connect(lambda v, p=prefix: setattr(self, f"{p}_min_freq", v))
        setattr(self, f"{prefix}_minfreq_spin", mf_spin)
        rg.addWidget(mf_spin, 2, 1)
        rg.addWidget(QLabel("Max items:"), 3, 0)
        mx_spin = QSpinBox()
        mx_spin.setRange(0, 10000)
        mx_spin.setValue(getattr(self, f"{prefix}_max_items"))
        mx_spin.setToolTip("0 = no limit (use Top N)")
        mx_spin.valueChanged.connect(lambda v, p=prefix: setattr(self, f"{p}_max_items", v))
        setattr(self, f"{prefix}_maxitems_spin", mx_spin)
        rg.addWidget(mx_spin, 3, 1)
        lo.addLayout(rg)

        # Skip header row checkbox
        skip_cb = QCheckBox("Skip header row when loading files")
        skip_cb.setChecked(True)
        setattr(self, f"{prefix}_skip_header", skip_cb)
        lo.addWidget(skip_cb)

        # Note text
        note = QLabel(
            "<i style='color:#6b7280;font-size:10px;'>"
            "Note: 'Include only' overrides Top N. Max items=0 means no limit.<br>"
            "Load from .txt, .csv, or .xlsx files.</i>")
        note.setWordWrap(True)
        lo.addWidget(note)

    def _load_filter_file(self, prefix, mode):
        """Load filter list from file (txt, csv, xlsx)."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {mode} list", "",
            "Text files (*.txt);;CSV (*.csv);;Excel (*.xlsx);;All (*.*)")
        if not path:
            return
        try:
            skip_header = getattr(self, f"{prefix}_skip_header", None)
            skip = skip_header.isChecked() if skip_header else True

            if path.endswith(".xlsx"):
                import pandas as pd
                df = pd.read_excel(path, header=0 if skip else None)
                lines = df.iloc[:, 0].dropna().astype(str).tolist()
            elif path.endswith(".csv"):
                import pandas as pd
                df = pd.read_csv(path, header=0 if skip else None)
                lines = df.iloc[:, 0].dropna().astype(str).tolist()
            else:
                with open(path, "r", encoding="utf-8") as f:
                    all_lines = [ln.strip() for ln in f if ln.strip()]
                lines = all_lines[1:] if skip and len(all_lines) > 1 else all_lines

            edit = getattr(self, f"{prefix}_{mode}_edit")
            edit.setPlainText("\n".join(lines))
        except Exception as e:
            logger.exception("Load filter file")

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
        self.tab_matrix = self._make_tw()
        self.tab_chi = self._make_tw()
        self.tab_ca = self._make_tw()
        self.tab_assoc = self._make_tw()
        for name, tw in [("Co-occurrence", self.tab_matrix),
                         ("Chi-square", self.tab_chi),
                         ("Correspondence", self.tab_ca),
                         ("Association", self.tab_assoc)]:
            self.tabs.addTab(tw, name)
        self.tab_matrix.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
            self._data = self._df = None
            self.status_label.setText("No data loaded")
            self._send_outputs(); return

        self._data = data
        self._df = self._table_to_df(data)
        self.status_label.setText(f"📊 {len(self._df):,} documents")
        if self.auto_apply:
            self._run_compute()

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def _on_row_entity_changed(self, idx):
        self.row_entity_idx = idx

    def _on_col_entity_changed(self, idx):
        self.col_entity_idx = idx

    def _on_viz_changed(self, idx):
        self.viz_type_idx = idx
        self._update_visualization()

    def _on_min_resid_changed(self, v):
        self.min_abs_residual = v
        if self.viz_type_idx == _VIZ_BALLOON:
            self._update_visualization()

    def _on_min_edge_changed(self, v):
        self.min_edge_weight = v
        if self.viz_type_idx == _VIZ_NETWORK:
            self._update_visualization()

    # =========================================================================
    # COMPUTE
    # =========================================================================

    def _run_compute(self):
        if self._df is None:
            return
        self.Warning.clear()
        if self.row_entity_idx == self.col_entity_idx:
            self.Warning.same_entity()
        self.Information.computing()
        self.compute_btn.setEnabled(False)
        self._worker = RelWorker(self._compute_relationships)
        self._worker.finished.connect(self._on_compute_done)
        self._worker.start()

    def _on_compute_done(self, result):
        self.Information.clear(); self.compute_btn.setEnabled(True)
        self._worker = None
        results, err = result
        if err:
            self.Error.compute_error(err); return
        (mat_df, chi_df, ca_df, assoc_df, row_items, col_items) = results
        self._matrix_df = mat_df
        self._chi_df = chi_df
        self._ca_df = ca_df
        self._assoc_df = assoc_df
        self._row_items = row_items
        self._col_items = col_items

        row_cfg = list(ENTITY_CONFIGS.values())[self.row_entity_idx]
        col_cfg = list(ENTITY_CONFIGS.values())[self.col_entity_idx]
        self._row_label = row_cfg["entity_label"]
        self._col_label = col_cfg["entity_label"]
        self._row_col_name = self._find_column(self._df, row_cfg["entity_col_alts"])
        self._col_col_name = self._find_column(self._df, col_cfg["entity_col_alts"])

        self._selected_rows.clear(); self._selected_cols.clear()
        self._selected_cells.clear()
        self._hover_row = self._hover_col = -1
        self._hover_cell = (-1, -1)

        self._display_all(); self._update_visualization(); self._send_outputs()

        nr = len(row_items); nc = len(col_items)
        if nr < 3: self.Warning.few_rows(nr)
        if nc < 3: self.Warning.few_cols(nc)

        parts = [f"{nr}×{nc} matrix"]
        if chi_df is not None: parts.append("χ²")
        if ca_df is not None: parts.append("CA")
        if assoc_df is not None: parts.append("assoc")
        self.status_label.setText(f"✅ {' · '.join(parts)}")

    def _compute_relationships(self):
        df = self._df
        row_cfg = list(ENTITY_CONFIGS.values())[self.row_entity_idx]
        col_cfg = list(ENTITY_CONFIGS.values())[self.col_entity_idx]
        sep = self._detect_separator(df)

        row_col = self._find_column(df, row_cfg["entity_col_alts"])
        col_col = self._find_column(df, col_cfg["entity_col_alts"])
        if row_col is None:
            raise ValueError(f"Row column not found: {row_cfg['entity_col_alts'][:3]}")
        if col_col is None:
            raise ValueError(f"Column column not found: {col_cfg['entity_col_alts'][:3]}")

        # Extract and filter entities
        row_items = self._get_filtered_entities(
            df, row_col, row_cfg["value_type"], sep, "row")
        col_items = self._get_filtered_entities(
            df, col_col, col_cfg["value_type"], sep, "col")

        if not row_items or not col_items:
            return (pd.DataFrame(), None, None, None, [], [])

        # Build co-occurrence matrix
        mat = self._build_cooccurrence(
            df, row_col, col_col,
            row_cfg["value_type"], col_cfg["value_type"],
            row_items, col_items, sep)

        if mat.empty:
            return (pd.DataFrame(), None, None, None, [], [])

        # Analyses
        mat_vals = mat[col_items].values.astype(float)
        chi_df = (self._calc_chi_square(mat_vals, row_items, col_items)
                  if self.compute_chi and HAS_SCIPY else None)
        ca_df = (self._calc_ca(mat_vals, row_items, col_items)
                 if self.compute_ca and min(mat_vals.shape) >= 2 else None)
        assoc_df = (self._calc_associations(mat_vals, row_items, col_items)
                    if self.compute_assoc else None)

        return (mat, chi_df, ca_df, assoc_df, row_items, col_items)

    def _get_filtered_entities(self, df, col, vtype, sep, prefix):
        """Extract and filter entities for row or col."""
        # Extract all entities
        all_ents = self._extract_entities(df, col, vtype, sep)
        counts = Counter(all_ents)

        # Apply min frequency
        min_freq = getattr(self, f"{prefix}_min_freq")
        ents = [e for e, c in counts.most_common() if c >= min_freq]

        # Apply include list (exact match)
        inc_text = getattr(self, f"{prefix}_include_text", "")
        if inc_text.strip():
            inc_set = {ln.strip() for ln in inc_text.strip().split("\n") if ln.strip()}
            ents = [e for e in ents if e in inc_set]

        # Apply exclude list
        exc_text = getattr(self, f"{prefix}_exclude_text", "")
        if exc_text.strip():
            exc_set = {ln.strip() for ln in exc_text.strip().split("\n") if ln.strip()}
            ents = [e for e in ents if e not in exc_set]

        # Apply regex include
        ri = getattr(self, f"{prefix}_regex_include", "")
        if ri.strip():
            try:
                pat = re.compile(ri, re.IGNORECASE)
                ents = [e for e in ents if pat.search(e)]
            except re.error:
                pass

        # Apply regex exclude
        re_ = getattr(self, f"{prefix}_regex_exclude", "")
        if re_.strip():
            try:
                pat = re.compile(re_, re.IGNORECASE)
                ents = [e for e in ents if not pat.search(e)]
            except re.error:
                pass

        # Apply max items or top N
        max_items = getattr(self, f"{prefix}_max_items", 0)
        top_n = getattr(self, f"{prefix}_top_n", 50)
        limit = max_items if max_items > 0 else top_n
        return ents[:limit]

    def _extract_entities(self, df, ecol, vtype, sep):
        if ecol not in df.columns:
            return []
        s = df[ecol].dropna().astype(str).str.strip()
        s = s[(s != "") & (s.str.lower() != "nan")]
        if vtype == "list":
            it = s.str.split(sep).explode().str.strip()
            return it[(it != "") & (it.str.lower() != "nan")].tolist()
        return s.tolist()

    def _build_cooccurrence(self, df, row_col, col_col, row_vt, col_vt,
                             row_items, col_items, sep):
        """Build co-occurrence matrix."""
        row_set = set(row_items)
        col_set = set(col_items)
        mat = {r: {c: 0 for c in col_items} for r in row_items}

        for _, rec in df.iterrows():
            rv = rec.get(row_col)
            cv = rec.get(col_col)
            if pd.isna(rv) or pd.isna(cv):
                continue
            rv, cv = str(rv).strip(), str(cv).strip()
            if not rv or not cv:
                continue

            # Parse row entities
            if row_vt == "list":
                r_ents = [e.strip() for e in rv.split(sep) if e.strip() in row_set]
            else:
                r_ents = [rv] if rv in row_set else []

            # Parse col entities
            if col_vt == "list":
                c_ents = [e.strip() for e in cv.split(sep) if e.strip() in col_set]
            else:
                c_ents = [cv] if cv in col_set else []

            # Increment co-occurrences
            for re in r_ents:
                for ce in c_ents:
                    mat[re][ce] += 1

        # Convert to DataFrame
        rows = []
        for r in row_items:
            row_data = {"Row": r}
            row_data.update(mat[r])
            row_data["Total"] = sum(mat[r].values())
            rows.append(row_data)
        result = pd.DataFrame(rows)
        # Filter out rows with zero total
        result = result[result["Total"] > 0].reset_index(drop=True)
        return result

    # ── chi-square ──
    def _calc_chi_square(self, mat, row_items, col_items):
        if not HAS_SCIPY or mat.size == 0:
            return None
        try:
            chi2, pv, dof, exp = sp_stats.chi2_contingency(mat)
        except ValueError:
            return None
        resid = (mat - exp) / np.where(exp > 0, np.sqrt(exp), 1)
        contrib = (mat - exp) ** 2 / np.where(exp > 0, exp, 1)
        rows = []
        for i, r in enumerate(row_items):
            for j, c in enumerate(col_items):
                rows.append({
                    "Row": r, "Column": c,
                    "Observed": int(mat[i, j]),
                    "Expected": round(float(exp[i, j]), 2),
                    "Std. residual": round(float(resid[i, j]), 3),
                    "Contribution": round(float(contrib[i, j]), 4),
                })
        df = pd.DataFrame(rows)
        df = df.sort_values("Contribution", ascending=False).reset_index(drop=True)
        # Add summary row
        df = pd.concat([df, pd.DataFrame([{
            "Row": f"[Overall: χ²={chi2:.2f}, p={pv:.2e}, df={dof}]",
            "Column": "", "Observed": "", "Expected": "",
            "Std. residual": "", "Contribution": round(chi2, 2)}])],
            ignore_index=True)
        return df

    # ── correspondence analysis ──
    def _calc_ca(self, mat, row_items, col_items):
        g = mat.sum()
        if g == 0:
            return None
        P = mat / g
        r = P.sum(1, keepdims=True)
        c = P.sum(0, keepdims=True)
        r[r == 0] = 1e-12
        c[c == 0] = 1e-12
        S = (P - r @ c) / np.sqrt(r @ c)
        k = min(mat.shape[0], mat.shape[1], 10) - 1
        if k < 1:
            return None
        try:
            U, s, Vt = np.linalg.svd(S, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        k = min(k, len(s))
        ine = s[:k] ** 2
        ti = ine.sum()
        rc = (U[:, :k] * s[:k]) / np.sqrt(r)
        cc = (Vt[:k, :].T * s[:k]) / np.sqrt(c.T)
        rows = []
        for i, r_item in enumerate(row_items):
            row = {"Entity": r_item, "Type": "Row"}
            for d in range(min(k, 3)):
                row[f"Dim {d+1}"] = round(float(rc[i, d]), 6)
            rows.append(row)
        for j, c_item in enumerate(col_items):
            row = {"Entity": c_item, "Type": "Column"}
            for d in range(min(k, 3)):
                row[f"Dim {d+1}"] = round(float(cc[j, d]), 6)
            rows.append(row)
        df = pd.DataFrame(rows)
        for d in range(min(k, 3)):
            pct = ine[d] / ti * 100 if ti > 0 else 0
            df = pd.concat([df, pd.DataFrame([{
                "Entity": f"[Inertia Dim {d+1}]", "Type": "Info",
                f"Dim {d+1}": round(pct, 2)}])], ignore_index=True)
        return df

    # ── association measures ──
    def _calc_associations(self, mat, row_items, col_items):
        """Calculate association measures for each cell."""
        rows = []
        row_totals = mat.sum(1)
        col_totals = mat.sum(0)
        grand_total = mat.sum()

        for i, r in enumerate(row_items):
            for j, c in enumerate(col_items):
                obs = mat[i, j]
                if obs == 0:
                    continue
                rt, ct = row_totals[i], col_totals[j]

                # Jaccard: intersection / union
                union = rt + ct - obs
                jaccard = obs / union if union > 0 else 0

                # Dice: 2*intersection / (|A| + |B|)
                dice = 2 * obs / (rt + ct) if (rt + ct) > 0 else 0

                # PMI: log2(P(a,b) / (P(a) * P(b)))
                p_ab = obs / grand_total if grand_total > 0 else 0
                p_a = rt / grand_total if grand_total > 0 else 0
                p_b = ct / grand_total if grand_total > 0 else 0
                pmi = (np.log2(p_ab / (p_a * p_b)) if p_a > 0 and p_b > 0 and p_ab > 0
                       else 0)

                # NPMI: PMI / -log2(P(a,b))
                npmi = pmi / (-np.log2(p_ab)) if p_ab > 0 and p_ab < 1 else 0

                rows.append({
                    "Row": r, "Column": c,
                    "Co-occurrence": int(obs),
                    "Jaccard": round(jaccard, 4),
                    "Dice": round(dice, 4),
                    "PMI": round(pmi, 4),
                    "NPMI": round(npmi, 4),
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Co-occurrence", ascending=False).reset_index(drop=True)
        return df

    # =========================================================================
    # VISUALISATION — redraw dispatcher
    # =========================================================================

    def _update_visualization(self):
        if not HAS_MPL or self.canvas is None:
            return
        if self._matrix_df is None or self._matrix_df.empty:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(.5, .5, "No data — click Compute", ha="center", va="center",
                    fontsize=14, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        for cid in self._mpl_cids:
            self.canvas.mpl_disconnect(cid)
        self._mpl_cids.clear()
        self.fig.clear()

        {_VIZ_HEATMAP:  self._draw_heatmap,
         _VIZ_BIPLOT:   self._draw_biplot,
         _VIZ_BALLOON:  self._draw_balloon,
         _VIZ_NETWORK:  self._draw_network,
         _VIZ_SANKEY:   self._draw_sankey,
        }.get(self.viz_type_idx, self._draw_heatmap)()

        self._mpl_cids.append(
            self.canvas.mpl_connect("motion_notify_event", self._on_mpl_hover))
        self._mpl_cids.append(
            self.canvas.mpl_connect("button_press_event", self._on_mpl_click))
        self.canvas.draw_idle()

    def _get_matrix_data(self):
        """Return (row_items, col_items, mat) from matrix DataFrame."""
        mat_df = self._matrix_df
        row_items = mat_df["Row"].tolist()
        col_items = self._col_items
        mat = mat_df[col_items].values.astype(float)
        return row_items, col_items, mat

    # ── 0  Heatmap ──
    def _draw_heatmap(self):
        row_items, col_items, mat = self._get_matrix_data()
        nr, nc = mat.shape
        ax = self.fig.add_subplot(111)

        # Normalize by row
        rs = mat.sum(1, keepdims=True)
        rs[rs == 0] = 1
        normed = mat / rs

        im = ax.imshow(normed, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=max(normed.max() * 1.05, .01),
                       interpolation="nearest")
        # Annotations
        for i in range(nr):
            for j in range(nc):
                c = "white" if normed[i, j] > normed.max() * .65 else "black"
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center",
                        fontsize=max(5, min(8, 200 // max(nr, nc))),
                        color=c, fontweight="bold")

        ax.set_xticks(range(nc))
        ax.set_xticklabels(col_items, fontsize=max(6, min(9, 200 // max(nc, 1))),
                          rotation=45, ha="left")
        ax.set_yticks(range(nr))
        ax.set_yticklabels(row_items, fontsize=max(6, min(9, 200 // max(nr, 1))))
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

        # Selection highlights
        for idx in self._selected_rows:
            if 0 <= idx < nr:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (-.5, idx-.5), nc, 1, boxstyle="round,pad=.02",
                    lw=2.5, ec="#6366f1", fc="none", zorder=10))
        if 0 <= self._hover_row < nr:
            ax.axhspan(self._hover_row-.5, self._hover_row+.5,
                       color="#fbbf24", alpha=.15, zorder=1)

        self.fig.colorbar(im, ax=ax, label="Row proportion", shrink=.6, pad=.02)
        ax.set_title(f"{self._row_label} × {self._col_label} Heatmap", fontsize=12, pad=12)
        self.fig.tight_layout()
        self._viz_meta = {"type": "heatmap", "n_rows": nr, "n_cols": nc,
                         "row_items": row_items, "col_items": col_items,
                         "mat": mat, "normed": normed, "ax": ax}

    # ── 1  CA Biplot ──
    def _draw_biplot(self):
        ca = self._ca_df
        ax = self.fig.add_subplot(111)
        if ca is None or ca.empty or "Dim 1" not in ca.columns:
            ax.text(.5, .5, "CA not computed", ha="center", va="center",
                    fontsize=12, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off()
            self._viz_meta = {"type": "biplot"}
            return

        rf = ca[ca["Type"] == "Row"].reset_index(drop=True)
        cf = ca[ca["Type"] == "Column"].reset_index(drop=True)
        inf = ca[ca["Type"] == "Info"]

        rx, ry = rf["Dim 1"].astype(float).values, rf["Dim 2"].astype(float).values
        cx, cy = cf["Dim 1"].astype(float).values, cf["Dim 2"].astype(float).values

        # Row entities (circles)
        ax.scatter(rx, ry, s=80, c="#2563eb", alpha=.7, edgecolors="#1e40af",
                   lw=1, zorder=5, label=self._row_label)
        for i in range(min(len(rf), 15)):
            ax.annotate(rf.iloc[i]["Entity"], (rx[i], ry[i]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="#1e40af",
                        alpha=.85)

        # Column entities (squares)
        ax.scatter(cx, cy, s=100, c="#dc2626", marker="s", alpha=.7,
                   edgecolors="#991b1b", lw=1, zorder=5, label=self._col_label)
        for j in range(min(len(cf), 15)):
            ax.annotate(cf.iloc[j]["Entity"], (cx[j], cy[j]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="#991b1b",
                        alpha=.85)

        ax.axhline(0, color="#e2e8f0", lw=.8, zorder=0)
        ax.axvline(0, color="#e2e8f0", lw=.8, zorder=0)

        # Inertia labels
        i1 = i2 = 0.0
        for _, r in inf.iterrows():
            if pd.notna(r.get("Dim 1")):
                i1 = r["Dim 1"]
            if pd.notna(r.get("Dim 2")):
                i2 = r["Dim 2"]
        ax.set_xlabel(f"Dim 1 ({i1:.1f}% inertia)")
        ax.set_ylabel(f"Dim 2 ({i2:.1f}% inertia)")
        ax.set_title("Correspondence Analysis Biplot", fontsize=12, pad=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=.2, ls="--")
        self.fig.tight_layout()
        self._viz_meta = {"type": "biplot", "ax": ax,
                         "rx": rx, "ry": ry, "cx": cx, "cy": cy,
                         "row_names": rf["Entity"].tolist(),
                         "col_names": cf["Entity"].tolist()}

    # ── 2  Balloon Plot ──
    def _draw_balloon(self):
        chi = self._chi_df
        ax = self.fig.add_subplot(111)
        if chi is None or chi.empty:
            ax.text(.5, .5, "Chi-square not computed", ha="center", va="center",
                    fontsize=12, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off()
            self._viz_meta = {"type": "balloon"}
            return

        dr = chi[~chi["Row"].str.startswith("[", na=False)].reset_index(drop=True)
        row_items = self._row_items
        col_items = self._col_items
        nr, nc = len(row_items), len(col_items)

        resid = np.zeros((nr, nc))
        counts = np.zeros((nr, nc), dtype=int)
        row_idx = {r: i for i, r in enumerate(row_items)}
        col_idx = {c: j for j, c in enumerate(col_items)}

        for _, rec in dr.iterrows():
            ri = row_idx.get(rec["Row"], -1)
            ci = col_idx.get(rec["Column"], -1)
            if ri >= 0 and ci >= 0:
                resid[ri, ci] = float(rec["Std. residual"]) if pd.notna(rec["Std. residual"]) else 0
                counts[ri, ci] = int(rec["Observed"]) if pd.notna(rec["Observed"]) else 0

        thresh = self.min_abs_residual
        visible = np.abs(resid) >= thresh
        mx = max(np.abs(resid).max(), .01)

        for i in range(nr):
            for j in range(nc):
                if not visible[i, j]:
                    continue
                v = resid[i, j]
                sz = abs(v) / mx * 500 + 20
                clr = "#2563eb" if v > 0 else "#dc2626"
                alp = min(.3 + abs(v) / mx * .6, .9)
                is_sel = (row_items[i], col_items[j]) in self._selected_cells
                ec = "#6366f1" if is_sel else "white"
                lw = 3 if is_sel else .5
                ax.scatter(j, i, s=sz, c=clr, alpha=alp,
                           edgecolors=ec, linewidths=lw, zorder=5)

        # Hover highlight
        hi, hj = self._hover_cell
        if 0 <= hi < nr and 0 <= hj < nc and visible[hi, hj]:
            ax.add_patch(mpatches.Rectangle(
                (hj - .45, hi - .45), .9, .9,
                lw=2, ec="#fbbf24", fc="#fbbf24", alpha=.18, zorder=1))

        ax.set_xticks(range(nc))
        ax.set_xticklabels(col_items, fontsize=max(6, min(9, 200 // max(nc, 1))),
                          rotation=45, ha="left")
        ax.set_yticks(range(nr))
        ax.set_yticklabels(row_items, fontsize=max(6, min(9, 200 // max(nr, 1))))
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xlim(-.6, nc - .4)
        ax.set_ylim(nr - .5, -.5)
        ax.grid(True, alpha=.15, ls="-")
        ax.legend(handles=[
            mpatches.Patch(color="#2563eb", label="Over-represented (+)"),
            mpatches.Patch(color="#dc2626", label="Under-represented (−)")],
            loc="lower right", fontsize=8, framealpha=.85)
        title = f"Association Plot ({self._row_label} × {self._col_label})"
        if thresh > 0:
            title += f"  [|r| ≥ {thresh:.1f}]"
        ax.set_title(title, fontsize=12, pad=12)
        self.fig.tight_layout()
        self._viz_meta = {"type": "balloon", "n_rows": nr, "n_cols": nc,
                         "row_items": row_items, "col_items": col_items,
                         "resid": resid, "counts": counts, "visible": visible, "ax": ax}

    # ── 3  Network Graph ──
    def _draw_network(self):
        row_items, col_items, mat = self._get_matrix_data()
        ax = self.fig.add_subplot(111)
        nr, nc = len(row_items), len(col_items)

        if nr == 0 or nc == 0:
            ax.text(.5, .5, "No data", ha="center", va="center",
                    fontsize=12, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off()
            self._viz_meta = {"type": "network"}
            return

        # Position nodes in two columns
        r_y = np.linspace(.9, .1, nr) if nr > 1 else [.5]
        c_y = np.linspace(.9, .1, nc) if nc > 1 else [.5]
        r_x, c_x = 0.2, 0.8

        # Draw edges
        max_w = max(mat.max(), 1)
        min_w = self.min_edge_weight
        edges = []
        for i in range(nr):
            for j in range(nc):
                w = mat[i, j]
                if w >= min_w:
                    lw = 0.5 + 4 * (w / max_w)
                    alp = .2 + .5 * (w / max_w)
                    is_sel = (row_items[i], col_items[j]) in self._selected_cells
                    if is_sel:
                        alp = .8
                        lw *= 1.5
                    # Bezier
                    t = np.linspace(0, 1, 30)
                    h = 3 * t ** 2 - 2 * t ** 3
                    xs = r_x + (c_x - r_x) * t
                    ys = r_y[i] + (c_y[j] - r_y[i]) * h
                    ax.plot(xs, ys, color="#64748b", lw=lw, alpha=alp, zorder=2)
                    edges.append((i, j, w))

        # Row nodes
        r_totals = mat.sum(1)
        r_max = max(r_totals.max(), 1)
        r_sizes = 30 + 200 * r_totals / r_max
        r_cols = ["#6366f1" if i in self._selected_rows else "#2563eb"
                  for i in range(nr)]
        ax.scatter([r_x] * nr, r_y, s=r_sizes, c=r_cols,
                   edgecolors="#1e40af", lw=1.2, zorder=5)
        for i in range(min(nr, 30)):
            ax.text(r_x - .05, r_y[i], row_items[i], ha="right", va="center",
                    fontsize=max(6, min(8, 200 // max(nr, 1))), color="#1e3a8a")

        # Column nodes
        c_totals = mat.sum(0)
        c_max = max(c_totals.max(), 1)
        c_sizes = 30 + 200 * c_totals / c_max
        c_cols = ["#f97316" if j in self._selected_cols else "#dc2626"
                  for j in range(nc)]
        ax.scatter([c_x] * nc, c_y, s=c_sizes, c=c_cols, marker="s",
                   edgecolors="#991b1b", lw=1.2, zorder=5)
        for j in range(min(nc, 30)):
            ax.text(c_x + .05, c_y[j], col_items[j], ha="left", va="center",
                    fontsize=max(6, min(8, 200 // max(nc, 1))), color="#7f1d1d")

        ax.set_xlim(-.05, 1.05)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        title = f"Network ({self._row_label} ↔ {self._col_label})"
        if min_w > 1:
            title += f"  [weight ≥ {min_w}]"
        ax.set_title(title, fontsize=12, pad=10)
        self.fig.tight_layout()
        self._viz_meta = {"type": "network", "ax": ax,
                         "row_items": row_items, "col_items": col_items,
                         "r_x": r_x, "c_x": c_x, "r_y": np.array(r_y), "c_y": np.array(c_y),
                         "n_rows": nr, "n_cols": nc, "mat": mat}

    # ── 4  Sankey Diagram ──
    def _draw_sankey(self):
        row_items, col_items, mat = self._get_matrix_data()
        ax = self.fig.add_subplot(111)
        nr, nc = len(row_items), len(col_items)

        if nr == 0 or nc == 0:
            ax.text(.5, .5, "No data", ha="center", va="center",
                    fontsize=12, color="#9ca3af", transform=ax.transAxes)
            ax.set_axis_off()
            self._viz_meta = {"type": "sankey"}
            return

        total = mat.sum()
        if total == 0:
            ax.set_axis_off()
            self._viz_meta = {"type": "sankey"}
            return

        x_left, x_right = 0.15, 0.85
        bar_w = 0.025
        r_totals = mat.sum(1)
        c_totals = mat.sum(0)

        r_gap = 0.004 * nr
        c_gap = 0.004 * nc
        r_scale = (1.0 - r_gap * max(nr - 1, 0)) / max(total, 1)
        c_scale = (1.0 - c_gap * max(nc - 1, 0)) / max(total, 1)

        # Row y ranges
        ry = {}
        y = 0
        for i in range(nr):
            h = r_totals[i] * r_scale
            ry[i] = (y, y + h)
            y += h + r_gap

        # Column y ranges
        cy = {}
        y = 0
        for j in range(nc):
            h = c_totals[j] * c_scale
            cy[j] = (y, y + h)
            y += h + c_gap

        # Draw bands
        r_cursor = {i: ry[i][0] for i in range(nr)}
        c_cursor = {j: cy[j][0] for j in range(nc)}
        bands = []

        for i in range(nr):
            for j in range(nc):
                cnt = mat[i, j]
                if cnt <= 0:
                    continue
                lt = r_cursor[i]
                lb = lt + cnt * r_scale
                r_cursor[i] = lb
                rt = c_cursor[j]
                rb = rt + cnt * c_scale
                c_cursor[j] = rb

                gc = _COLOURS[j % len(_COLOURS)]
                is_sel = (row_items[i], col_items[j]) in self._selected_cells
                alp = .75 if is_sel else .35

                t = np.linspace(0, 1, 50)
                h = 3 * t ** 2 - 2 * t ** 3
                xs = x_left + (x_right - x_left) * t
                yt = lt + (rt - lt) * h
                yb = lb + (rb - lb) * h
                ax.fill_between(xs, yb, yt, color=gc, alpha=alp, zorder=3)
                if is_sel:
                    ax.plot(xs, yt, color="#6366f1", lw=1.2, zorder=4)
                    ax.plot(xs, yb, color="#6366f1", lw=1.2, zorder=4)

                bands.append({"row": row_items[i], "col": col_items[j],
                              "ri": i, "cj": j, "cnt": int(cnt),
                              "lt": lt, "lb": lb, "rt": rt, "rb": rb})

        # Row bars
        for i in range(nr):
            ys, ye = ry[i]
            is_sel = i in self._selected_rows or \
                     any((row_items[i], c) in self._selected_cells for c in col_items)
            ec = "#6366f1" if is_sel else "#475569"
            ax.add_patch(mpatches.FancyBboxPatch(
                (x_left - bar_w, ys), bar_w, ye - ys,
                boxstyle="round,pad=.001", fc="#2563eb", ec=ec, lw=1.5, zorder=5))
            if ye - ys > .006:
                ax.text(x_left - bar_w - .008, (ys + ye) / 2, row_items[i],
                        ha="right", va="center",
                        fontsize=max(5, min(8, 200 // max(nr, 1))),
                        color="#1e3a8a", clip_on=True)

        # Column bars
        for j in range(nc):
            ys, ye = cy[j]
            gc = _COLOURS[j % len(_COLOURS)]
            ax.add_patch(mpatches.FancyBboxPatch(
                (x_right, ys), bar_w, ye - ys,
                boxstyle="round,pad=.001", fc=gc, ec="white", lw=1.5, zorder=5))
            ax.text(x_right + bar_w + .008, (ys + ye) / 2, col_items[j],
                    ha="left", va="center", fontsize=max(6, min(9, 200 // max(nc, 1))),
                    fontweight="bold", color=gc)

        ymax = max(max(v[1] for v in ry.values()) if ry else 1,
                   max(v[1] for v in cy.values()) if cy else 1)
        ax.set_xlim(-.02, 1.02)
        ax.set_ylim(ymax + .02, -.02)
        ax.set_axis_off()
        ax.set_title(f"Sankey ({self._row_label} → {self._col_label})", fontsize=12, pad=10)
        self.fig.tight_layout()
        self._viz_meta = {"type": "sankey", "ax": ax, "bands": bands,
                         "row_items": row_items, "col_items": col_items,
                         "ry": ry, "cy": cy, "x_left": x_left, "x_right": x_right,
                         "n_rows": nr, "n_cols": nc}

    # =========================================================================
    # INTERACTION — hover
    # =========================================================================

    def _on_mpl_hover(self, event):
        meta = self._viz_meta
        if meta is None or event.inaxes is None:
            QToolTip.hideText()
            return
        vt = meta.get("type")
        old_row, old_col, old_cell = self._hover_row, self._hover_col, self._hover_cell

        if vt == "heatmap":
            self._hover_heatmap(event, meta)
        elif vt == "balloon":
            self._hover_balloon(event, meta)
        elif vt == "biplot":
            self._hover_biplot(event, meta)
        elif vt == "network":
            self._hover_network(event, meta)
        elif vt == "sankey":
            self._hover_sankey(event, meta)

        if (self._hover_row != old_row or self._hover_col != old_col or
                self._hover_cell != old_cell):
            self._update_visualization()

    def _hover_heatmap(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            self._hover_row = -1
            QToolTip.hideText()
            return
        j, i = int(round(x)), int(round(y))
        nr, nc = meta["n_rows"], meta["n_cols"]
        if not (0 <= i < nr and 0 <= j < nc):
            self._hover_row = -1
            QToolTip.hideText()
            return
        self._hover_row = i
        r, c = meta["row_items"][i], meta["col_items"][j]
        v = int(meta["mat"][i, j])
        p = meta["normed"][i, j] * 100
        text = f"<b>{r}</b> × <b>{c}</b><br>Count: {v}<br>Row %: {p:.1f}%"
        pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
        QToolTip.showText(pos, text)

    def _hover_balloon(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            self._hover_cell = (-1, -1)
            QToolTip.hideText()
            return
        j, i = int(round(x)), int(round(y))
        nr, nc = meta["n_rows"], meta["n_cols"]
        if not (0 <= i < nr and 0 <= j < nc):
            self._hover_cell = (-1, -1)
            QToolTip.hideText()
            return
        visible = meta.get("visible")
        if visible is not None and not visible[i, j]:
            self._hover_cell = (-1, -1)
            QToolTip.hideText()
            return
        self._hover_cell = (i, j)
        r, c = meta["row_items"][i], meta["col_items"][j]
        res = meta["resid"][i, j]
        cnt = int(meta["counts"][i, j])
        d = "over" if res > 0 else "under"
        text = (f"<b>{r}</b> × <b>{c}</b><br>"
                f"Count: <b>{cnt}</b><br>Std. residual: {res:.3f}<br>({d}-represented)")
        pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
        QToolTip.showText(pos, text)

    def _hover_biplot(self, event, meta):
        rx, ry = meta.get("rx"), meta.get("ry")
        cx, cy = meta.get("cx"), meta.get("cy")
        if rx is None or event.xdata is None:
            self._hover_row = self._hover_col = -1
            QToolTip.hideText()
            return
        ax = meta["ax"]
        xl = ax.get_xlim()
        th = ((xl[1] - xl[0]) * .03) ** 2

        # Check row entities
        d_r = (rx - event.xdata) ** 2 + (ry - event.ydata) ** 2
        mi_r = int(np.argmin(d_r))
        if d_r[mi_r] < th:
            self._hover_row = mi_r
            self._hover_col = -1
            n = meta["row_names"][mi_r]
            text = f"<b>{self._row_label}: {n}</b><br>Dim 1: {rx[mi_r]:.4f}<br>Dim 2: {ry[mi_r]:.4f}"
            pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
            QToolTip.showText(pos, text)
            return

        # Check col entities
        d_c = (cx - event.xdata) ** 2 + (cy - event.ydata) ** 2
        mi_c = int(np.argmin(d_c))
        if d_c[mi_c] < th:
            self._hover_col = mi_c
            self._hover_row = -1
            n = meta["col_names"][mi_c]
            text = f"<b>{self._col_label}: {n}</b><br>Dim 1: {cx[mi_c]:.4f}<br>Dim 2: {cy[mi_c]:.4f}"
            pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
            QToolTip.showText(pos, text)
            return

        self._hover_row = self._hover_col = -1
        QToolTip.hideText()

    def _hover_network(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None:
            self._hover_row = self._hover_col = -1
            QToolTip.hideText()
            return
        r_x, c_x = meta["r_x"], meta["c_x"]
        r_y, c_y = meta["r_y"], meta["c_y"]

        # Check row nodes
        if abs(x - r_x) < .06:
            d = np.abs(r_y - y)
            mi = int(np.argmin(d))
            if d[mi] < .03:
                self._hover_row = mi
                self._hover_col = -1
                n = meta["row_items"][mi]
                pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
                QToolTip.showText(pos, f"<b>{self._row_label}: {n}</b>")
                return

        # Check col nodes
        if abs(x - c_x) < .06:
            d = np.abs(c_y - y)
            mi = int(np.argmin(d))
            if d[mi] < .03:
                self._hover_col = mi
                self._hover_row = -1
                n = meta["col_items"][mi]
                pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
                QToolTip.showText(pos, f"<b>{self._col_label}: {n}</b>")
                return

        self._hover_row = self._hover_col = -1
        QToolTip.hideText()

    def _hover_sankey(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            self._hover_cell = (-1, -1)
            QToolTip.hideText()
            return
        xl, xr = meta["x_left"], meta["x_right"]
        if not (xl <= x <= xr):
            self._hover_cell = (-1, -1)
            QToolTip.hideText()
            return
        t = (x - xl) / (xr - xl)
        h = 3 * t ** 2 - 2 * t ** 3
        for b in meta["bands"]:
            yt = b["lt"] + (b["rt"] - b["lt"]) * h
            yb = b["lb"] + (b["rb"] - b["lb"]) * h
            if yb <= y <= yt:
                self._hover_cell = (b["ri"], b["cj"])
                text = f"<b>{b['row']}</b> → <b>{b['col']}</b><br>Count: {b['cnt']}"
                pos = self.canvas.mapToGlobal(QPoint(int(event.x), int(self.canvas.height() - event.y)))
                QToolTip.showText(pos, text)
                return
        self._hover_cell = (-1, -1)
        QToolTip.hideText()

    # =========================================================================
    # INTERACTION — click
    # =========================================================================

    def _on_mpl_click(self, event):
        meta = self._viz_meta
        if meta is None or event.inaxes is None or event.button != 1:
            return
        vt = meta.get("type")
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)

        if vt == "heatmap":
            idx = self._hit_row(event, meta)
            if idx >= 0:
                self._toggle_row(idx, ctrl)
        elif vt == "biplot":
            ri, ci = self._hit_biplot(event, meta)
            if ri >= 0:
                self._toggle_row(ri, ctrl)
            elif ci >= 0:
                self._toggle_col(ci, ctrl)
        elif vt == "balloon":
            cell = self._hit_balloon_cell(event, meta)
            if cell:
                self._toggle_cell(cell, ctrl)
        elif vt == "network":
            ri, ci = self._hit_network_node(event, meta)
            if ri >= 0:
                self._toggle_row(ri, ctrl)
            elif ci >= 0:
                self._toggle_col(ci, ctrl)
        elif vt == "sankey":
            cell = self._hit_sankey_band(event, meta)
            if cell:
                self._toggle_cell(cell, ctrl)

        self._update_visualization()
        self._sync_table_selection()
        self._update_sel_label()
        self._send_selected_documents()

    def _hit_row(self, event, meta):
        y = event.ydata
        if y is None:
            return -1
        i = int(round(y))
        return i if 0 <= i < meta.get("n_rows", 0) else -1

    def _hit_biplot(self, event, meta):
        rx, ry = meta.get("rx"), meta.get("ry")
        cx, cy = meta.get("cx"), meta.get("cy")
        if rx is None or event.xdata is None:
            return -1, -1
        ax = meta["ax"]
        xl = ax.get_xlim()
        th = ((xl[1] - xl[0]) * .04) ** 2
        d_r = (rx - event.xdata) ** 2 + (ry - event.ydata) ** 2
        mi_r = int(np.argmin(d_r))
        if d_r[mi_r] < th:
            return mi_r, -1
        d_c = (cx - event.xdata) ** 2 + (cy - event.ydata) ** 2
        mi_c = int(np.argmin(d_c))
        if d_c[mi_c] < th:
            return -1, mi_c
        return -1, -1

    def _hit_balloon_cell(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return None
        j, i = int(round(x)), int(round(y))
        nr, nc = meta["n_rows"], meta["n_cols"]
        if not (0 <= i < nr and 0 <= j < nc):
            return None
        visible = meta.get("visible")
        if visible is not None and not visible[i, j]:
            return None
        return (meta["row_items"][i], meta["col_items"][j])

    def _hit_network_node(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None:
            return -1, -1
        r_x, c_x = meta["r_x"], meta["c_x"]
        r_y, c_y = meta["r_y"], meta["c_y"]
        if abs(x - r_x) < .08:
            d = np.abs(r_y - y)
            mi = int(np.argmin(d))
            if d[mi] < .04:
                return mi, -1
        if abs(x - c_x) < .08:
            d = np.abs(c_y - y)
            mi = int(np.argmin(d))
            if d[mi] < .04:
                return -1, mi
        return -1, -1

    def _hit_sankey_band(self, event, meta):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return None
        xl, xr = meta["x_left"], meta["x_right"]
        if not (xl <= x <= xr):
            return None
        t = (x - xl) / (xr - xl)
        h = 3 * t ** 2 - 2 * t ** 3
        for b in meta["bands"]:
            yt = b["lt"] + (b["rt"] - b["lt"]) * h
            yb = b["lb"] + (b["rb"] - b["lb"]) * h
            if yb <= y <= yt:
                return (b["row"], b["col"])
        return None

    def _toggle_row(self, idx, extend):
        self._selected_cells.clear()
        self._selected_cols.clear()
        if extend:
            self._selected_rows.symmetric_difference_update({idx})
        else:
            self._selected_rows = set() if self._selected_rows == {idx} else {idx}

    def _toggle_col(self, idx, extend):
        self._selected_cells.clear()
        self._selected_rows.clear()
        if extend:
            self._selected_cols.symmetric_difference_update({idx})
        else:
            self._selected_cols = set() if self._selected_cols == {idx} else {idx}

    def _toggle_cell(self, cell, extend):
        self._selected_rows.clear()
        self._selected_cols.clear()
        if extend:
            self._selected_cells.symmetric_difference_update({cell})
        else:
            self._selected_cells = set() if self._selected_cells == {cell} else {cell}

    def _sync_table_selection(self):
        tw = self.tab_matrix
        tw.clearSelection()
        # For row selection, highlight matching rows in matrix table
        if self._selected_rows and self._matrix_df is not None:
            for r in range(tw.rowCount()):
                item = tw.item(r, 0)
                if item:
                    try:
                        idx = self._row_items.index(item.text())
                        if idx in self._selected_rows:
                            tw.selectRow(r)
                    except ValueError:
                        pass

    def _update_sel_label(self):
        if self._selected_cells:
            mat_df = self._matrix_df
            parts = []
            total_docs = 0
            for r, c in sorted(self._selected_cells):
                cnt = 0
                if mat_df is not None and c in mat_df.columns:
                    row = mat_df[mat_df["Row"] == r]
                    if len(row) > 0:
                        cnt = int(row[c].iloc[0])
                total_docs += cnt
                parts.append(f"{r}↔{c}")
            if len(parts) <= 2:
                self.sel_label.setText(f"Selected: {', '.join(parts)} ({total_docs} co-occ.)")
            else:
                self.sel_label.setText(f"Selected: {', '.join(parts[:2])} + {len(parts)-2} more ({total_docs} co-occ.)")
        elif self._selected_rows:
            names = [self._row_items[i] for i in sorted(self._selected_rows)
                     if i < len(self._row_items)]
            if len(names) <= 3:
                self.sel_label.setText(f"Selected {self._row_label}s: {', '.join(names)}")
            else:
                self.sel_label.setText(f"Selected {self._row_label}s: {', '.join(names[:3])} + {len(names)-3} more")
        elif self._selected_cols:
            names = [self._col_items[j] for j in sorted(self._selected_cols)
                     if j < len(self._col_items)]
            if len(names) <= 3:
                self.sel_label.setText(f"Selected {self._col_label}s: {', '.join(names)}")
            else:
                self.sel_label.setText(f"Selected {self._col_label}s: {', '.join(names[:3])} + {len(names)-3} more")
        else:
            self.sel_label.setText("")

    # =========================================================================
    # SELECTED DOCUMENTS OUTPUT
    # =========================================================================

    def _send_selected_documents(self):
        if self._data is None or self._df is None:
            self.Outputs.selected_data.send(None)
            return

        df = self._df
        row_cfg = list(ENTITY_CONFIGS.values())[self.row_entity_idx]
        col_cfg = list(ENTITY_CONFIGS.values())[self.col_entity_idx]
        row_col = self._row_col_name
        col_col = self._col_col_name
        row_vt = row_cfg["value_type"]
        col_vt = col_cfg["value_type"]
        sep = self._detect_separator(df)

        mask = pd.Series(False, index=df.index)

        if self._selected_cells and row_col and col_col:
            for r_name, c_name in self._selected_cells:
                r_mask = self._entity_mask(df, row_col, r_name, row_vt, sep)
                c_mask = self._entity_mask(df, col_col, c_name, col_vt, sep)
                mask |= (r_mask & c_mask)
        elif self._selected_rows and row_col:
            sel_names = {self._row_items[i] for i in self._selected_rows
                         if i < len(self._row_items)}
            for n in sel_names:
                mask |= self._entity_mask(df, row_col, n, row_vt, sep)
        elif self._selected_cols and col_col:
            sel_names = {self._col_items[j] for j in self._selected_cols
                         if j < len(self._col_items)}
            for n in sel_names:
                mask |= self._entity_mask(df, col_col, n, col_vt, sep)
        else:
            self.Outputs.selected_data.send(None)
            return

        indices = np.where(mask.values)[0]
        if len(indices) == 0:
            self.Outputs.selected_data.send(None)
        else:
            self.Outputs.selected_data.send(self._data[indices])

    def _entity_mask(self, df, col, name, vtype, sep):
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        if vtype == "list":
            m = df[col].dropna().astype(str).apply(
                lambda x: name in {s.strip() for s in x.split(sep)})
            return m.reindex(df.index, fill_value=False)
        return df[col].astype(str).str.strip() == name

    # =========================================================================
    # TABLE DISPLAY
    # =========================================================================

    def _display_all(self):
        for tw, d in [(self.tab_matrix, self._matrix_df),
                      (self.tab_chi, self._chi_df),
                      (self.tab_ca, self._ca_df),
                      (self.tab_assoc, self._assoc_df)]:
            self._fill_table(tw, d)

    @staticmethod
    def _fill_table(tw, df):
        tw.setSortingEnabled(False)
        tw.clear()
        if df is None or df.empty:
            tw.setRowCount(0)
            tw.setColumnCount(0)
            return
        nr, nc = df.shape
        tw.setRowCount(nr)
        tw.setColumnCount(nc)
        tw.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(nr):
            for c in range(nc):
                v = df.iloc[r, c]
                if pd.isna(v):
                    it = QTableWidgetItem("")
                elif isinstance(v, float):
                    it = QTableWidgetItem(str(int(v)) if v == int(v) and abs(v) < 1e12
                                          else f"{v:.4f}")
                    it.setData(Qt.UserRole, float(v))
                else:
                    it = QTableWidgetItem(str(v))
                tw.setItem(r, c, it)
        tw.resizeColumnsToContents()
        tw.setSortingEnabled(True)

    def _clear_results(self):
        self._matrix_df = self._chi_df = self._ca_df = self._assoc_df = None
        self._row_items = []
        self._col_items = []
        self._selected_rows.clear()
        self._selected_cols.clear()
        self._selected_cells.clear()
        self._hover_row = self._hover_col = -1
        self._hover_cell = (-1, -1)
        self._viz_meta = None
        self.sel_label.setText("")
        for tw in [self.tab_matrix, self.tab_chi, self.tab_ca, self.tab_assoc]:
            tw.clear()
            tw.setRowCount(0)
            tw.setColumnCount(0)
        if HAS_MPL and self.fig is not None:
            self.fig.clear()
            self.canvas.draw_idle()

    # =========================================================================
    # OUTPUTS
    # =========================================================================

    def _send_outputs(self):
        self.Outputs.matrix.send(self._df_to_table(self._matrix_df))
        self.Outputs.chi_square.send(self._df_to_table(self._chi_df))
        self.Outputs.correspondence.send(self._df_to_table(self._ca_df))
        self.Outputs.associations.send(self._df_to_table(self._assoc_df))
        if not self._selected_rows and not self._selected_cols and not self._selected_cells:
            self.Outputs.selected_data.send(None)

    # =========================================================================
    # EXPORT
    # =========================================================================

    def _export_results(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export", "", "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        try:
            if path.endswith(".xlsx"):
                with pd.ExcelWriter(path, engine="openpyxl") as w:
                    for nm, d in [("Co-occurrence", self._matrix_df),
                                  ("Chi-square", self._chi_df),
                                  ("Correspondence", self._ca_df),
                                  ("Association", self._assoc_df)]:
                        if d is not None:
                            d.to_excel(w, sheet_name=nm, index=False)
            elif self._matrix_df is not None:
                self._matrix_df.to_csv(path, index=False)
        except Exception:
            logger.exception("export")

    # =========================================================================
    # STATE
    # =========================================================================

    def onDeleteWidget(self):
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        self.splitter_state = bytes(self.splitter.saveState())
        super().onDeleteWidget()

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _find_column(df, candidates):
        cl = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c in df.columns:
                return c
            if c.lower() in cl:
                return cl[c.lower()]
        return None

    @staticmethod
    def _detect_separator(df):
        cols = ["Authors", "Author Keywords", "Index Keywords",
                "Affiliations", "References"]
        sc = {"; ": 0, ";": 0, "|": 0}
        for c in cols:
            if c not in df.columns:
                continue
            s = df[c].dropna().head(200).astype(str)
            for sep in sc:
                if s.str.contains(sep, regex=False).mean() > .15:
                    sc[sep] += 1
        if sc["|"] >= sc["; "] and sc["|"] > 0:
            return "|"
        if sc["; "] > 0:
            return "; "
        if sc[";"] > 0:
            return ";"
        return "; "

    def _table_to_df(self, table):
        data = {}
        domain = table.domain
        for var in list(domain.attributes) + list(domain.class_vars):
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else None for v in col]
            else:
                data[var.name] = col
        for var in domain.metas:
            col = table[:, var].metas.flatten()
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)]
                                  if not (isinstance(v, float) and np.isnan(v)) else None
                                  for v in col]
            elif isinstance(var, StringVariable):
                data[var.name] = [str(v) if v is not None else "" for v in col]
            else:
                data[var.name] = col
        return pd.DataFrame(data)

    @staticmethod
    def _df_to_table(df):
        if df is None or df.empty:
            return None
        attrs, metas = [], []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                attrs.append(ContinuousVariable(str(c)))
            else:
                metas.append(StringVariable(str(c)))
        domain = Domain(attrs, metas=metas)
        X = np.zeros((len(df), len(attrs)), float)
        M = np.zeros((len(df), len(metas)), object)
        for j, v in enumerate(attrs):
            X[:, j] = pd.to_numeric(df[v.name], errors="coerce").fillna(np.nan)
        for j, v in enumerate(metas):
            M[:, j] = df[v.name].fillna("").astype(str)
        return Table.from_numpy(domain, X, metas=M)
