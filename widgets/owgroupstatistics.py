# -*- coding: utf-8 -*-
"""
Group Statistics Widget
=======================
Orange widget for computing performance statistics across document groups.

Uses Biblium's ``utilsbib.group_entity_stats`` to compute rich performance
indicators (document count, fraction, percentage, rank, percentile-rank,
and all additional performance metrics from ``get_entity_stats``) for each
entity within each group.

Inputs:
    Data: Bibliographic data with "Group: xxx" columns (from Setup Groups)

Outputs:
    Statistics: Full performance statistics table (wide or long format)
    Filtered Statistics: Top-N filtered subset
"""

import logging
from typing import Optional, Dict, List, Any
from collections import Counter

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QRadioButton, QButtonGroup,
    QAbstractItemView, QTabWidget, QSplitter, QFrame,
    QLineEdit, QFileDialog, QApplication,
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtGui import QColor, QFont

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.settings import Setting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Biblium imports
# ---------------------------------------------------------------------------
try:
    import biblium
    from biblium import utilsbib
    from biblium.bibstats import BiblioStats
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    utilsbib = None
    BiblioStats = None

try:
    from sklearn.feature_extraction.text import CountVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUP_PREFIX = "Group: "

ENTITY_CONFIGS = {
    "sources": {
        "label": "Sources (Journals)",
        "entity_col": "Source title",
        "entity_col_alts": ["Source title", "Source", "Journal", "source", "SO"],
        "entity_label": "Source",
        "value_type": "string",
        "bib_method": "get_group_sources_stats",
        "count_method": "count_sources",
        "description": "Performance statistics for journals across groups",
    },
    "authors": {
        "label": "Authors",
        "entity_col": "Authors",
        "entity_col_alts": ["Authors", "Author", "authors", "AU"],
        "entity_label": "Author",
        "value_type": "list",
        "bib_method": "get_group_authors_stats",
        "count_method": "count_authors",
        "description": "Performance statistics for authors across groups",
    },
    "author_keywords": {
        "label": "Author Keywords",
        "entity_col": ["Processed Author Keywords", "Author Keywords"],
        "entity_col_alts": ["Processed Author Keywords", "Author Keywords",
                            "author keywords", "DE", "Keywords"],
        "entity_label": "Keyword",
        "value_type": "list",
        "bib_method": "get_group_author_keywords_stats",
        "count_method": "count_author_keywords",
        "description": "Performance statistics for author keywords across groups",
    },
    "index_keywords": {
        "label": "Index Keywords",
        "entity_col": "Index Keywords",
        "entity_col_alts": ["Index Keywords", "Keywords Plus",
                            "index keywords", "ID", "KeywordsPlus"],
        "entity_label": "Keyword",
        "value_type": "list",
        "bib_method": "get_group_index_keywords_stats",
        "count_method": "count_index_keywords",
        "description": "Performance statistics for index keywords across groups",
    },
    "ca_countries": {
        "label": "CA Countries",
        "entity_col": "CA Country",
        "entity_col_alts": ["CA Country", "Corresponding Author's Country",
                            "CA_Country"],
        "entity_label": "Country",
        "value_type": "string",
        "bib_method": "get_group_ca_countries_stats",
        "count_method": "count_ca_countries",
        "description": "Performance statistics for corresponding-author countries",
    },
    "all_countries": {
        "label": "All Countries",
        "entity_col": ["Countries of Authors", "All Countries", "Country"],
        "entity_col_alts": ["Countries of Authors", "All Countries",
                            "Country", "countries"],
        "entity_label": "Country",
        "value_type": "list",
        "bib_method": "get_group_all_countries_stats",
        "count_method": "count_all_countries",
        "description": "Performance statistics for all countries across groups",
    },
    "affiliations": {
        "label": "Affiliations",
        "entity_col": "Affiliations",
        "entity_col_alts": ["Affiliations", "Affiliation", "C1", "C3"],
        "entity_label": "Affiliation",
        "value_type": "list",
        "bib_method": "get_group_affiliations_stats",
        "count_method": "count_affiliations",
        "description": "Performance statistics for affiliations across groups",
    },
    "references": {
        "label": "References",
        "entity_col": ["References", "Cited References"],
        "entity_col_alts": ["References", "Cited References", "CR",
                            "cited references"],
        "entity_label": "Reference",
        "value_type": "list",
        "bib_method": "get_group_references_stats",
        "count_method": "count_references",
        "description": "Performance statistics for cited references across groups",
    },
    "ngrams_abstract": {
        "label": "N-grams (Abstract)",
        "entity_col": ["Processed Abstract", "Abstract"],
        "entity_col_alts": ["Processed Abstract", "Abstract", "AB",
                            "abstract"],
        "entity_label": "Term",
        "value_type": "text",
        "bib_method": "get_group_ngrams_abstract_stats",
        "count_method": "count_ngrams_abstract",
        "description": "Performance statistics for abstract n-grams across groups",
    },
    "ngrams_title": {
        "label": "N-grams (Title)",
        "entity_col": ["Processed Title", "Title"],
        "entity_col_alts": ["Processed Title", "Title", "TI", "title"],
        "entity_label": "Term",
        "value_type": "text",
        "bib_method": "get_group_ngrams_title_stats",
        "count_method": "count_ngrams_title",
        "description": "Performance statistics for title n-grams across groups",
    },
}


# =============================================================================
# WORKER THREAD
# =============================================================================

class StatsWorker(QThread):
    """Background thread for computing group statistics."""
    finished = pyqtSignal(object)   # emits (result_df, error_msg)
    progress = pyqtSignal(str)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self._func = func

    def run(self):
        try:
            result = self._func()
            self.finished.emit((result, None))
        except Exception as e:
            self.finished.emit((None, str(e)))


# =============================================================================
# WIDGET
# =============================================================================

class OWGroupStatistics(OWWidget):
    """Compute performance statistics for entities across document groups."""

    name = "Group Statistics"
    description = ("Compute performance statistics (counts, fractions, ranks, "
                    "percentile ranks) for bibliometric entities across "
                    "document groups")
    icon = "icons/group_statistics.svg"
    priority = 33
    keywords = ["group", "statistics", "performance", "indicators",
                "comparison", "entity", "counts", "ranks", "percentile"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table,
                     doc="Bibliographic data with group columns from Setup Groups")

    class Outputs:
        statistics = Output("Statistics", Table,
                            doc="Full performance statistics table")
        filtered = Output("Filtered Statistics", Table,
                          doc="Top-N filtered statistics")

    # Settings
    entity_type_idx = Setting(0)
    top_n = Setting(100)
    output_format_idx = Setting(0)    # 0 = wide, 1 = long
    include_indicators = Setting(True)
    include_items_text = Setting("")
    exclude_items_text = Setting("")
    auto_apply = Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_groups = Msg("No group columns found — connect Setup Groups first")
        stats_error = Msg("Statistics error: {}")

    class Warning(OWWidget.Warning):
        empty_result = Msg("No results — entity column may be missing")
        sklearn_missing = Msg("scikit-learn needed for n-gram analysis")
        no_biblium = Msg("Biblium not installed — using limited fallback")

    class Information(OWWidget.Information):
        groups_found = Msg("Found {} group columns: {}")
        computing = Msg("Computing statistics…")
        done = Msg("{} entities × {} groups ({} output columns)")

    def __init__(self):
        super().__init__()

        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._gm_df: Optional[pd.DataFrame] = None
        self._group_names: List[str] = []
        self._result_df: Optional[pd.DataFrame] = None
        self._worker: Optional[StatsWorker] = None

        self._setup_control_area()
        self._setup_main_area()

    # =========================================================================
    # UI CONSTRUCTION
    # =========================================================================

    def _setup_control_area(self):
        """Build control area matching the app interface."""

        # ── Entity Selection ──
        entity_box = gui.widgetBox(self.controlArea, "📋 Entity Selection")

        form = QGridLayout()
        form.setColumnStretch(1, 1)

        form.addWidget(QLabel("Entity Type:"), 0, 0)
        self.entity_combo = QComboBox()
        for cfg in ENTITY_CONFIGS.values():
            self.entity_combo.addItem(cfg["label"])
        self.entity_combo.setCurrentIndex(self.entity_type_idx)
        self.entity_combo.currentIndexChanged.connect(self._on_entity_changed)
        form.addWidget(self.entity_combo, 0, 1)

        self.entity_desc = QLabel("")
        self.entity_desc.setStyleSheet("color: #3b82f6; font-size: 11px;")
        self.entity_desc.setWordWrap(True)
        form.addWidget(self.entity_desc, 1, 0, 1, 2)

        entity_box.layout().addLayout(form)
        self._update_entity_description()

        # ── Parameters ──
        params_box = gui.widgetBox(self.controlArea, "⚙️ Parameters")

        param_form = QGridLayout()
        param_form.setColumnStretch(1, 1)

        param_form.addWidget(QLabel("Top N Items:"), 0, 0)
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 5000)
        self.top_n_spin.setValue(self.top_n)
        self.top_n_spin.valueChanged.connect(self._on_top_n_changed)
        param_form.addWidget(self.top_n_spin, 0, 1)

        param_form.addWidget(QLabel("Output Format:"), 1, 0)
        fmt_row = QHBoxLayout()
        self.fmt_group = QButtonGroup()
        self.wide_radio = QRadioButton("Wide")
        self.long_radio = QRadioButton("Long")
        self.fmt_group.addButton(self.wide_radio, 0)
        self.fmt_group.addButton(self.long_radio, 1)
        if self.output_format_idx == 0:
            self.wide_radio.setChecked(True)
        else:
            self.long_radio.setChecked(True)
        self.fmt_group.idClicked.connect(self._on_format_changed)
        fmt_row.addWidget(self.wide_radio)
        fmt_row.addWidget(self.long_radio)
        fmt_row.addStretch()
        fmt_w = QWidget()
        fmt_w.setLayout(fmt_row)
        param_form.addWidget(fmt_w, 1, 1)

        self.indicators_cb = QCheckBox("Include performance indicators")
        self.indicators_cb.setChecked(self.include_indicators)
        self.indicators_cb.toggled.connect(self._on_indicators_changed)

        params_box.layout().addLayout(param_form)
        params_box.layout().addWidget(self.indicators_cb)

        # ── Filtering ──
        filter_box = gui.widgetBox(self.controlArea, "🔍 Filtering")

        filt_form = QGridLayout()
        filt_form.setColumnStretch(1, 1)

        filt_form.addWidget(QLabel("Include Items:"), 0, 0)
        self.include_edit = QLineEdit(self.include_items_text)
        self.include_edit.setPlaceholderText("comma-separated (optional)")
        self.include_edit.setToolTip("Only include these items (comma separated)")
        self.include_edit.editingFinished.connect(self._on_include_changed)
        filt_form.addWidget(self.include_edit, 0, 1)

        filt_form.addWidget(QLabel("Exclude Items:"), 1, 0)
        self.exclude_edit = QLineEdit(self.exclude_items_text)
        self.exclude_edit.setPlaceholderText("comma-separated (optional)")
        self.exclude_edit.setToolTip("Exclude these items (comma separated)")
        self.exclude_edit.editingFinished.connect(self._on_exclude_changed)
        filt_form.addWidget(self.exclude_edit, 1, 1)

        filter_box.layout().addLayout(filt_form)

        # ── Compute Button ──
        btn_layout = QVBoxLayout()

        self.compute_btn = QPushButton("▶ Compute Statistics")
        self.compute_btn.clicked.connect(self._run_stats)
        self.compute_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a34a; border: none;
                border-radius: 4px; padding: 10px 20px;
                color: white; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #15803d; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        btn_layout.addWidget(self.compute_btn)

        quick_row = QHBoxLayout()

        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e7ff; border: 1px solid #6366f1;
                border-radius: 4px; padding: 6px 12px;
                color: #4338ca; font-weight: bold;
            }
            QPushButton:hover { background-color: #c7d2fe; }
        """)
        quick_row.addWidget(self.export_btn)

        self.compute_all_btn = QPushButton("Compute All")
        self.compute_all_btn.clicked.connect(self._compute_all_entities)
        self.compute_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e7ff; border: 1px solid #6366f1;
                border-radius: 4px; padding: 6px 12px;
                color: #4338ca; font-weight: bold;
            }
            QPushButton:hover { background-color: #c7d2fe; }
        """)
        quick_row.addWidget(self.compute_all_btn)

        btn_layout.addLayout(quick_row)
        self.controlArea.layout().addLayout(btn_layout)

        # ── Auto apply ──
        self.auto_apply_cb = QCheckBox("Auto apply on data change")
        self.auto_apply_cb.setChecked(self.auto_apply)
        self.auto_apply_cb.toggled.connect(self._on_auto_apply_changed)
        self.controlArea.layout().addWidget(self.auto_apply_cb)

    def _setup_main_area(self):
        """Build main area with results table and status."""
        layout = QVBoxLayout()

        # Status bar
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet(
            "background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 4px;"
        )
        status_layout = QHBoxLayout(self.status_frame)
        self.status_label = QLabel("No data loaded")
        self.status_label.setStyleSheet("color: #0369a1; border: none;")
        status_layout.addWidget(self.status_label)
        layout.addWidget(self.status_frame)

        # Tab widget
        self.tab_widget = QTabWidget()

        # Results table tab
        self.result_table = QTableWidget()
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setSortingEnabled(True)
        self.tab_widget.addTab(self.result_table, "📊 Results")

        # Summary tab
        self.summary_table = QTableWidget()
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tab_widget.addTab(self.summary_table, "📋 Summary")

        layout.addWidget(self.tab_widget)

        container = QWidget()
        container.setLayout(layout)
        self.mainArea.layout().addWidget(container)

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def _on_entity_changed(self, idx):
        self.entity_type_idx = idx
        self._update_entity_description()

    def _on_top_n_changed(self, val):
        self.top_n = val

    def _on_format_changed(self, idx):
        self.output_format_idx = idx

    def _on_indicators_changed(self, checked):
        self.include_indicators = checked

    def _on_include_changed(self):
        self.include_items_text = self.include_edit.text()

    def _on_exclude_changed(self):
        self.exclude_items_text = self.exclude_edit.text()

    def _on_auto_apply_changed(self, checked):
        self.auto_apply = checked

    def _update_entity_description(self):
        keys = list(ENTITY_CONFIGS.keys())
        idx = min(self.entity_type_idx, len(keys) - 1)
        config = ENTITY_CONFIGS[keys[idx]]
        self.entity_desc.setText(config["description"])

    # =========================================================================
    # INPUT
    # =========================================================================

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive bibliographic data with embedded group columns."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()

        self._data = data
        self._df = None
        self._gm_df = None
        self._group_names = []
        self._result_df = None

        if data is None:
            self._update_status("No data loaded")
            self._clear_outputs()
            return

        self._df = self._table_to_df(data)
        self._extract_groups()

        self._update_status(
            f"Loaded {len(self._df)} documents, "
            f"{len(self._group_names)} groups"
        )

        if self.auto_apply and self._gm_df is not None:
            self._run_stats()

    def _extract_groups(self):
        """Extract group matrix from 'Group: xxx' columns."""
        if self._df is None:
            return

        group_cols = [c for c in self._df.columns if c.startswith(GROUP_PREFIX)]

        if not group_cols:
            self.Error.no_groups()
            return

        gm = pd.DataFrame(index=self._df.index)
        group_names = []
        for col in group_cols:
            clean_name = col[len(GROUP_PREFIX):]
            gm[clean_name] = self._to_binary(self._df[col])
            group_names.append(clean_name)

        self._gm_df = gm
        self._group_names = group_names

        preview = ", ".join(group_names[:5])
        if len(group_names) > 5:
            preview += f", … (+{len(group_names) - 5} more)"
        self.Information.groups_found(len(group_names), preview)

    # =========================================================================
    # CORE COMPUTATION
    # =========================================================================

    def _run_stats(self):
        """Compute statistics for the selected entity type."""
        if self._df is None or self._gm_df is None:
            return

        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()

        keys = list(ENTITY_CONFIGS.keys())
        idx = min(self.entity_type_idx, len(keys) - 1)
        config = ENTITY_CONFIGS[keys[idx]]

        self.compute_btn.setEnabled(False)
        self._update_status(f"Computing {config['label']} statistics…")
        QApplication.processEvents()

        try:
            result_df = self._compute_entity_stats(config)
        except Exception as e:
            self.Error.stats_error(str(e))
            logger.exception("Statistics computation failed")
            self.compute_btn.setEnabled(True)
            return

        self.compute_btn.setEnabled(True)

        if result_df is None or result_df.empty:
            self.Warning.empty_result()
            self._clear_outputs()
            return

        self._result_df = result_df
        self._display_results(result_df)
        self._send_outputs(result_df)

    def _compute_entity_stats(self, config: Dict) -> pd.DataFrame:
        """Compute group-level entity stats."""
        # Build clean data (remove group prefix cols)
        data_cols = [c for c in self._df.columns if not c.startswith(GROUP_PREFIX)]
        df = self._df[data_cols].copy()
        gm_df = self._gm_df

        # Resolve entity column
        entity_col = config["entity_col"]
        if isinstance(entity_col, list):
            resolved = None
            for c in entity_col:
                if c in df.columns:
                    resolved = c
                    break
            if resolved is None:
                # Try alternatives
                for c in config.get("entity_col_alts", []):
                    if c in df.columns:
                        resolved = c
                        break
            if resolved is None:
                raise ValueError(
                    f"None of the columns found: {config['entity_col_alts']}"
                )
            entity_col = resolved
        else:
            if entity_col not in df.columns:
                # Try alternatives
                entity_col = self._resolve_column(
                    config.get("entity_col_alts", [entity_col]), df
                )
                if entity_col is None:
                    raise ValueError(
                        f"Column not found: {config['entity_col_alts']}"
                    )

        # Parse include/exclude items
        include_items = self._parse_item_list(self.include_items_text)
        exclude_items = self._parse_item_list(self.exclude_items_text)

        output_format = "wide" if self.output_format_idx == 0 else "long"
        entity_label = config["entity_label"]
        value_type = config["value_type"]
        top_n = self.top_n

        # Try Biblium path
        if HAS_BIBLIUM:
            return self._stats_with_biblium(
                df, gm_df, entity_col, entity_label, value_type,
                config, include_items, exclude_items,
                top_n, output_format,
            )
        else:
            self.Warning.no_biblium()
            return self._stats_fallback(
                df, gm_df, entity_col, entity_label, value_type,
                include_items, exclude_items, top_n, output_format,
            )

    def _stats_with_biblium(
        self, df, gm_df, entity_col, entity_label, value_type,
        config, include_items, exclude_items, top_n, output_format,
    ) -> pd.DataFrame:
        """Compute statistics using Biblium's utilsbib.group_entity_stats."""
        sep = self._detect_separator(df)

        # Biblium's get_performance_indicators requires a Year column;
        # if missing, add a dummy one to prevent crashes.
        needs_year_cleanup = False
        if "Year" not in df.columns:
            year_alts = [c for c in df.columns
                         if c.lower() in ("year", "publication year", "py")]
            if year_alts:
                df = df.copy()
                df["Year"] = pd.to_numeric(df[year_alts[0]], errors="coerce")
            else:
                df = df.copy()
                df["Year"] = 2024  # placeholder so indicators don't crash
                needs_year_cleanup = True

        # Ensure Cited by exists (needed for h-index etc.)
        if "Cited by" not in df.columns:
            cite_alts = [c for c in df.columns
                         if c.lower() in ("cited by", "times cited", "tc", "citation count")]
            if cite_alts:
                df["Cited by"] = pd.to_numeric(df[cite_alts[0]], errors="coerce").fillna(0)
            else:
                df["Cited by"] = 0

        try:
            stats_df, indicators = utilsbib.group_entity_stats(
                df=df,
                group_matrix=gm_df,
                entity_col=entity_col,
                entity_label=entity_label,
                items_of_interest=include_items if include_items else None,
                exclude_items=exclude_items if exclude_items else None,
                top_n=top_n,
                output_format=output_format,
                indicators=False,  # indicator matrices not needed for widget
                value_type=value_type,
                sep=sep,
                mode="core",
            )
        except Exception as e:
            logger.warning(f"Biblium group_entity_stats failed: {e}, using fallback")
            return self._stats_fallback(
                df, gm_df, entity_col, entity_label, value_type,
                include_items, exclude_items, top_n, output_format,
            )

        # Remove dummy-year columns from output if we injected them
        if needs_year_cleanup and stats_df is not None and not stats_df.empty:
            drop_cols = [c for c in stats_df.columns
                         if "year" in c.lower() and "average" not in c.lower()]
            if drop_cols:
                stats_df = stats_df.drop(
                    columns=[c for c in drop_cols if c in stats_df.columns],
                    errors="ignore",
                )

        return stats_df

    def _stats_fallback(
        self, df, gm_df, entity_col, entity_label, value_type,
        include_items, exclude_items, top_n, output_format,
    ) -> pd.DataFrame:
        """Fallback statistics without full Biblium."""
        sep = self._detect_separator(df)
        group_names = list(gm_df.columns)

        # Step 1: Generate global counts to determine items universe
        global_counts = self._generate_counts(
            df, entity_col, entity_label, value_type, sep
        )
        if global_counts.empty:
            return pd.DataFrame()

        # Apply include/exclude filters
        if include_items:
            global_counts = global_counts[
                global_counts[entity_label].isin(include_items)
            ]
        if exclude_items:
            global_counts = global_counts[
                ~global_counts[entity_label].isin(exclude_items)
            ]

        # Top-N
        global_counts = global_counts.head(top_n)
        items_list = global_counts[entity_label].tolist()

        if not items_list:
            return pd.DataFrame()

        # Step 2: Per-group stats
        group_frames = {}
        for g in group_names:
            mask = gm_df[g].astype(bool)
            subset = df[mask]

            if subset.empty:
                group_frames[g] = pd.DataFrame({
                    entity_label: items_list,
                    "Number of documents": 0,
                    "Fraction of documents": 0.0,
                    "Percentage of documents": 0.0,
                    "Rank": np.nan,
                    "Percentrank of documents": np.nan,
                })
                continue

            g_counts = self._generate_counts(
                subset, entity_col, entity_label, value_type, sep
            )
            g_counts = g_counts[g_counts[entity_label].isin(items_list)]

            # Ensure all items are present
            all_items_df = pd.DataFrame({entity_label: items_list})
            g_stats = all_items_df.merge(g_counts, on=entity_label, how="left")
            g_stats["Number of documents"] = g_stats["Number of documents"].fillna(0)

            n_subset = len(subset)
            g_stats["Fraction of documents"] = (
                g_stats["Number of documents"] / n_subset if n_subset else 0
            )
            g_stats["Percentage of documents"] = (
                g_stats["Fraction of documents"] * 100
            )
            g_stats["Rank"] = g_stats["Number of documents"].rank(
                ascending=False, method="min"
            )
            g_stats["Percentrank of documents"] = g_stats[
                "Number of documents"
            ].rank(pct=True)

            group_frames[g] = g_stats

        # Step 3: Assemble output
        metric_cols = [
            "Number of documents", "Fraction of documents",
            "Percentage of documents", "Rank", "Percentrank of documents",
        ]

        if output_format == "wide":
            pieces = []
            for g in group_names:
                gf = group_frames.get(g, pd.DataFrame())
                if gf.empty:
                    continue
                gf = gf.set_index(entity_label)
                avail = [c for c in metric_cols if c in gf.columns]
                gf = gf[avail]
                gf.columns = [f"{g} - {c}" for c in gf.columns]
                pieces.append(gf)

            if not pieces:
                return pd.DataFrame()

            wide_df = pd.concat(pieces, axis=1)
            wide_df.index.name = entity_label
            wide_df = wide_df.reset_index()
            return wide_df

        else:  # long
            rows = []
            for g in group_names:
                gf = group_frames.get(g, pd.DataFrame())
                if gf.empty:
                    continue
                tmp = gf.copy()
                tmp.insert(0, "Group", g)
                rows.append(tmp)

            if not rows:
                return pd.DataFrame()

            long_df = pd.concat(rows, ignore_index=True)
            cols = ["Group", entity_label] + [
                c for c in long_df.columns
                if c not in ("Group", entity_label)
            ]
            return long_df[cols]

    def _generate_counts(self, df, entity_col, entity_label, value_type, sep):
        """Generate entity counts from a DataFrame."""
        if entity_col not in df.columns:
            return pd.DataFrame(columns=[entity_label, "Number of documents"])

        series = df[entity_col].dropna().astype(str).str.strip()
        series = series[series != ""]
        series = series[series.str.lower() != "nan"]

        if series.empty:
            return pd.DataFrame(columns=[entity_label, "Number of documents"])

        if value_type == "list":
            exploded = series.str.split(sep).explode().str.strip()
            exploded = exploded[exploded != ""]
            exploded = exploded[exploded.str.lower() != "nan"]
            if exploded.empty:
                return pd.DataFrame(columns=[entity_label, "Number of documents"])
            # Count documents (not total occurrences)
            doc_counts = (
                exploded.reset_index()
                .drop_duplicates()
                .groupby(exploded.name)["index"]
                .count()
            )
            counts = Counter(exploded)
            result = pd.DataFrame({
                entity_label: list(counts.keys()),
                "Number of documents": list(counts.values()),
            })
        elif value_type == "text":
            if HAS_SKLEARN:
                vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
                try:
                    mat = vectorizer.fit_transform(series)
                    terms = vectorizer.get_feature_names_out()
                    doc_freq = (mat > 0).sum(axis=0).A1
                    result = pd.DataFrame({
                        entity_label: terms,
                        "Number of documents": doc_freq,
                    })
                except Exception:
                    return pd.DataFrame(columns=[entity_label, "Number of documents"])
            else:
                counts = Counter(series)
                result = pd.DataFrame({
                    entity_label: list(counts.keys()),
                    "Number of documents": list(counts.values()),
                })
        else:  # string
            counts = Counter(series)
            result = pd.DataFrame({
                entity_label: list(counts.keys()),
                "Number of documents": list(counts.values()),
            })

        result = result.sort_values(
            "Number of documents", ascending=False
        ).reset_index(drop=True)
        return result

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def _display_results(self, df: pd.DataFrame):
        """Show results in the table widget."""
        self.result_table.setSortingEnabled(False)
        self.result_table.clear()

        if df is None or df.empty:
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)
            return

        nrows, ncols = df.shape
        self.result_table.setRowCount(nrows)
        self.result_table.setColumnCount(ncols)
        self.result_table.setHorizontalHeaderLabels(
            [str(c) for c in df.columns]
        )

        for r in range(nrows):
            for c in range(ncols):
                val = df.iloc[r, c]
                if pd.isna(val):
                    item = QTableWidgetItem("")
                elif isinstance(val, float):
                    if val == int(val) and abs(val) < 1e12:
                        item = QTableWidgetItem(str(int(val)))
                    else:
                        item = QTableWidgetItem(f"{val:.4f}")
                    item.setData(Qt.UserRole, float(val))
                else:
                    item = QTableWidgetItem(str(val))
                self.result_table.setItem(r, c, item)

        self.result_table.resizeColumnsToContents()
        self.result_table.setSortingEnabled(True)

        # Summary
        self._update_summary_tab(df)

        # Info
        keys = list(ENTITY_CONFIGS.keys())
        idx = min(self.entity_type_idx, len(keys) - 1)
        config = ENTITY_CONFIGS[keys[idx]]

        # Count unique entities
        entity_label = config["entity_label"]
        n_entities = df[entity_label].nunique() if entity_label in df.columns else nrows

        self.Information.done(n_entities, len(self._group_names), ncols)
        self._update_status(
            f"✅ {n_entities} {config['label'].lower()} × "
            f"{len(self._group_names)} groups "
            f"({ncols} columns, {nrows} rows)"
        )

    def _update_summary_tab(self, df: pd.DataFrame):
        """Update the summary tab."""
        self.summary_table.clear()

        if df is None or df.empty:
            self.summary_table.setRowCount(0)
            self.summary_table.setColumnCount(0)
            return

        # Build summary: per-group document counts & unique entities
        keys = list(ENTITY_CONFIGS.keys())
        idx = min(self.entity_type_idx, len(keys) - 1)
        config = ENTITY_CONFIGS[keys[idx]]
        entity_label = config["entity_label"]

        fmt = "wide" if self.output_format_idx == 0 else "long"

        if fmt == "wide":
            # Identify count columns
            count_cols = [c for c in df.columns if "Number of documents" in c]
            summary_rows = [
                ("Total entities", str(len(df))),
                ("Groups", str(len(self._group_names))),
                ("Output format", "Wide"),
            ]
            for c in count_cols:
                group_name = c.split(" - ")[0] if " - " in c else c
                total = df[c].sum() if pd.api.types.is_numeric_dtype(df[c]) else "N/A"
                summary_rows.append((f"Total docs ({group_name})", f"{total:.0f}"))
        else:
            summary_rows = [
                ("Groups", str(len(self._group_names))),
                ("Output format", "Long"),
                ("Total rows", str(len(df))),
            ]
            if entity_label in df.columns:
                summary_rows.append(
                    ("Unique entities", str(df[entity_label].nunique()))
                )
            if "Group" in df.columns:
                for g in df["Group"].unique():
                    g_sub = df[df["Group"] == g]
                    summary_rows.append((f"Rows ({g})", str(len(g_sub))))

        self.summary_table.setRowCount(len(summary_rows))
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])

        for r, (metric, value) in enumerate(summary_rows):
            self.summary_table.setItem(r, 0, QTableWidgetItem(metric))
            self.summary_table.setItem(r, 1, QTableWidgetItem(str(value)))

        self.summary_table.resizeColumnsToContents()
        self.summary_table.horizontalHeader().setStretchLastSection(True)

    # =========================================================================
    # OUTPUTS
    # =========================================================================

    def _send_outputs(self, df: pd.DataFrame):
        """Send results to output ports."""
        if df is None or df.empty:
            self._clear_outputs()
            return

        table = self._df_to_table(df)
        self.Outputs.statistics.send(table)

        # Filtered: top N by first count column
        count_cols = [c for c in df.columns if "Number of documents" in c]
        if count_cols:
            sort_col = count_cols[0]
            numeric_vals = pd.to_numeric(df[sort_col], errors='coerce').fillna(0)
            filtered = df.iloc[numeric_vals.nlargest(self.top_n).index]
        else:
            filtered = df.head(self.top_n)

        if not filtered.empty:
            self.Outputs.filtered.send(self._df_to_table(filtered))
        else:
            self.Outputs.filtered.send(None)

    def _clear_outputs(self):
        """Clear all outputs."""
        self.Outputs.statistics.send(None)
        self.Outputs.filtered.send(None)
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(0)
        self.summary_table.setRowCount(0)
        self.summary_table.setColumnCount(0)
        self._result_df = None

    # =========================================================================
    # COMPUTE ALL / EXPORT
    # =========================================================================

    def _compute_all_entities(self):
        """Compute statistics for all entity types."""
        if self._df is None or self._gm_df is None:
            return

        self.compute_all_btn.setEnabled(False)
        all_results = {}

        for i, (key, config) in enumerate(ENTITY_CONFIGS.items()):
            self._update_status(
                f"Computing {config['label']}… ({i + 1}/{len(ENTITY_CONFIGS)})"
            )
            QApplication.processEvents()

            try:
                result = self._compute_entity_stats(config)
                if result is not None and not result.empty:
                    all_results[config["label"]] = result
            except Exception as e:
                logger.warning(f"Failed for {config['label']}: {e}")

        self.compute_all_btn.setEnabled(True)

        if not all_results:
            self.Warning.empty_result()
            return

        # Export all to Excel
        fname, _ = QFileDialog.getSaveFileName(
            self, "Export All Statistics", "group_statistics_all.xlsx",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        if fname:
            try:
                with pd.ExcelWriter(fname, engine="openpyxl") as writer:
                    for sheet_name, result_df in all_results.items():
                        safe_name = sheet_name[:31]  # Excel sheet name limit
                        result_df.to_excel(writer, sheet_name=safe_name, index=False)
                self._update_status(
                    f"✅ Exported {len(all_results)} entity types to {fname}"
                )
            except Exception as e:
                self.Error.stats_error(f"Export failed: {e}")

    def _export_results(self):
        """Export current results to file."""
        if self._result_df is None or self._result_df.empty:
            return

        keys = list(ENTITY_CONFIGS.keys())
        idx = min(self.entity_type_idx, len(keys) - 1)
        config = ENTITY_CONFIGS[keys[idx]]
        default_name = f"group_{keys[idx]}_stats"

        fname, filt = QFileDialog.getSaveFileName(
            self, "Export Statistics", f"{default_name}.xlsx",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        if not fname:
            return

        try:
            if fname.endswith(".csv"):
                self._result_df.to_csv(fname, index=False)
            else:
                if not fname.endswith(".xlsx"):
                    fname += ".xlsx"
                self._result_df.to_excel(fname, index=False, engine="openpyxl")
            self._update_status(f"✅ Exported to {fname}")
        except Exception as e:
            self.Error.stats_error(f"Export failed: {e}")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _update_status(self, text: str):
        self.status_label.setText(text)

    @staticmethod
    def _to_binary(series: pd.Series) -> pd.Series:
        """Robustly convert a Series to binary 0/1."""
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1)
        truthy = {"yes", "true", "1", "1.0", "t", "y"}
        return series.apply(
            lambda v: 1.0 if str(v).strip().lower() in truthy else 0.0
        )

    @staticmethod
    def _parse_item_list(text: str) -> Optional[List[str]]:
        """Parse comma-separated item list."""
        if not text or not text.strip():
            return None
        items = [x.strip() for x in text.split(",") if x.strip()]
        return items if items else None

    def _resolve_column(self, candidates, df):
        """Find first matching column name."""
        for c in candidates:
            if c in df.columns:
                return c
            # Case-insensitive
            for dc in df.columns:
                if dc.lower() == c.lower():
                    return dc
        return None

    def _detect_separator(self, df: pd.DataFrame) -> str:
        """Detect the multi-value separator used in the data."""
        test_cols = [
            "Authors", "Author Keywords", "Index Keywords",
            "Affiliations", "References",
        ]
        scores = {"; ": 0, ";": 0, "|": 0}
        for col_name in test_cols:
            if col_name not in df.columns:
                continue
            sample = df[col_name].dropna().head(200).astype(str)
            for sep in scores:
                if sample.str.contains(sep, regex=False).mean() > 0.15:
                    scores[sep] += 1
        if scores["|"] >= scores["; "] and scores["|"] > 0:
            return "|"
        if scores["; "] > 0:
            return "; "
        if scores[";"] > 0:
            return ";"
        return "; "

    def _table_to_df(self, table: Table) -> pd.DataFrame:
        """Convert Orange Table to pandas DataFrame."""
        data = {}
        domain = table.domain

        for var in domain.attributes:
            col_data = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)] if not np.isnan(v) else None
                    for v in col_data
                ]
            else:
                data[var.name] = col_data

        for var in domain.class_vars:
            col_data = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)] if not np.isnan(v) else None
                    for v in col_data
                ]
            else:
                data[var.name] = col_data

        for var in domain.metas:
            col_data = table[:, var].metas.flatten()
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)]
                    if not (isinstance(v, float) and np.isnan(v)) else None
                    for v in col_data
                ]
            elif isinstance(var, StringVariable):
                data[var.name] = [
                    str(v) if v is not None and str(v) != "?" else None
                    for v in col_data
                ]
            else:
                data[var.name] = col_data

        return pd.DataFrame(data)

    def _df_to_table(self, df: pd.DataFrame) -> Table:
        """Convert pandas DataFrame to Orange Table."""
        attrs = []
        metas = []
        X_cols = []
        M_cols = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                attrs.append(ContinuousVariable(str(col)))
                X_cols.append(col)
            else:
                metas.append(StringVariable(str(col)))
                M_cols.append(col)

        domain = Domain(attrs, metas=metas)

        X = np.empty((len(df), len(attrs)), dtype=float)
        for i, col in enumerate(X_cols):
            X[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(np.nan).values

        M = np.empty((len(df), len(metas)), dtype=object)
        for i, col in enumerate(M_cols):
            M[:, i] = df[col].fillna("").astype(str).values

        return Table.from_numpy(domain, X, metas=M if metas else None)


# =============================================================================
# WIDGET PREVIEW
# =============================================================================

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWGroupStatistics).run()
