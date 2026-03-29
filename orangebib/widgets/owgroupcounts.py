# -*- coding: utf-8 -*-
"""
Group Counts Widget
===================
Orange widget for counting entities across document groups.

Uses Biblium's count_occurrences_across_groups to compare entity
frequencies (authors, keywords, sources, countries, etc.) across
user-defined document groups.

Inputs:
- Data: Bibliographic data table (with "Group: xxx" columns from Setup Groups)

Outputs:
- Counts: Merged counts table with per-group and combined statistics
- Filtered Counts: Top-N filtered counts table
"""

import os
import logging
from typing import Optional, Dict, List, Any
from collections import Counter
from functools import reduce
from itertools import chain

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QRadioButton, QButtonGroup,
    QAbstractItemView, QTabWidget, QSplitter, QFrame,
)
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.settings import Setting

# Try to import biblium
try:
    import biblium
    from biblium import utilsbib
    from biblium.bibstats import BiblioStats
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    utilsbib = None
    BiblioStats = None

# Try sklearn for text n-grams
try:
    from sklearn.feature_extraction.text import CountVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


# =============================================================================
# ENTITY CONFIGURATIONS
# =============================================================================

ENTITY_CONFIGS = {
    "sources": {
        "label": "Sources (Journals)",
        "method": "count_sources",
        "result_attr": "group_sources_counts_df",
        "description": "Count journal/source occurrences per group",
        "columns": ["Source title", "Source", "Journal", "source", "SO"],
        "count_type": "single",
        "item_label": "Source",
    },
    "authors": {
        "label": "Authors",
        "method": "count_authors",
        "result_attr": "group_authors_counts_df",
        "description": "Count author occurrences per group",
        "columns": ["Authors", "Author", "authors", "AU"],
        "count_type": "list",
        "item_label": "Author",
    },
    "author_keywords": {
        "label": "Author Keywords",
        "method": "count_author_keywords",
        "result_attr": "group_author_keywords_counts_df",
        "description": "Count author keyword occurrences per group",
        "columns": ["Author Keywords", "Author keywords", "author_keywords", "DE", "Keywords"],
        "count_type": "list",
        "item_label": "Keyword",
    },
    "index_keywords": {
        "label": "Index Keywords",
        "method": "count_index_keywords",
        "result_attr": "group_index_keywords_counts_df",
        "description": "Count index keyword occurrences per group",
        "columns": ["Index Keywords", "Index keywords", "index_keywords", "ID", "Indexed Keywords"],
        "count_type": "list",
        "item_label": "Keyword",
    },
    "ca_countries": {
        "label": "Corresponding Author Countries",
        "method": "count_ca_countries",
        "result_attr": "group_ca_countries_counts_df",
        "description": "Count CA country occurrences per group",
        "columns": ["CA Country", "Correspondence Author Country", "ca_country"],
        "count_type": "single",
        "item_label": "Country",
    },
    "all_countries": {
        "label": "All Countries",
        "method": "count_all_countries",
        "result_attr": "group_all_countries_counts_df",
        "description": "Count all collaborating countries per group",
        "columns": ["Countries of Authors", "Countries", "Country", "countries",
                     "authorships.countries"],
        "count_type": "list",
        "item_label": "Country",
    },
    "affiliations": {
        "label": "Affiliations",
        "method": "count_affiliations",
        "result_attr": "group_affiliations_counts_df",
        "description": "Count affiliation occurrences per group",
        "columns": ["Affiliations", "Affiliation", "affiliations", "C1"],
        "count_type": "list",
        "item_label": "Affiliation",
    },
    "references": {
        "label": "References",
        "method": "count_references",
        "result_attr": "group_references_counts_df",
        "description": "Count cited reference occurrences per group",
        "columns": ["References", "Cited References", "references", "CR"],
        "count_type": "list",
        "item_label": "Reference",
    },
    "ngrams_abstract": {
        "label": "N-grams (Abstract)",
        "method": "count_ngrams_abstract",
        "result_attr": "group_words_abs_counts_df",
        "description": "Count n-gram occurrences in abstracts per group",
        "columns": ["Abstract", "Processed Abstract", "abstract", "AB", "Description"],
        "count_type": "text",
        "item_label": "Term",
    },
    "ngrams_title": {
        "label": "N-grams (Title)",
        "method": "count_ngrams_title",
        "result_attr": "group_words_tit_counts_df",
        "description": "Count n-gram occurrences in titles per group",
        "columns": ["Title", "Processed Title", "title", "TI", "Article Title"],
        "count_type": "text",
        "item_label": "Term",
    },
}

# Group column prefix used by Setup Groups widget
GROUP_PREFIX = "Group: "


class OWGroupCounts(OWWidget):
    """Count entities across document groups."""

    name = "Group Counts"
    description = "Count and compare entity frequencies across document groups"
    icon = "icons/group_counts.svg"
    priority = 31
    keywords = ["group", "count", "comparison", "frequency", "authors", "keywords",
                "sources", "bibliometric", "cross-group"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table,
                     doc="Bibliographic data with group columns (from Setup Groups)")

    class Outputs:
        counts = Output("Counts", Table, doc="Merged group counts table")
        filtered_counts = Output("Filtered Counts", Table,
                                  doc="Top-N filtered counts table")

    # Settings
    entity_type_idx = Setting(0)
    merge_type_idx = Setting(0)  # 0=all items, 1=shared items
    top_n_display = Setting(50)
    top_n_plot = Setting(10)
    auto_apply = Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_groups = Msg("No group columns found. Connect data from Setup Groups "
                        "(columns prefixed 'Group: ').")
        no_column = Msg("Required column not found for '{}' entity type")
        count_error = Msg("Counting error: {}")

    class Warning(OWWidget.Warning):
        empty_result = Msg("No items found for counting")
        sklearn_missing = Msg("sklearn not installed — text n-grams disabled")
        partial_results = Msg("{} entity types failed during Count All")

    class Information(OWWidget.Information):
        counted = Msg("Counted {} unique items across {} groups")
        counting_all = Msg("Counting all entity types…")
        groups_found = Msg("Found {} group columns: {}")

    def __init__(self):
        super().__init__()

        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._gm_df: Optional[pd.DataFrame] = None  # auto-extracted group matrix
        self._group_names: List[str] = []
        self._result_df: Optional[pd.DataFrame] = None
        self._all_results: Dict[str, pd.DataFrame] = {}

        self._setup_control_area()
        self._setup_main_area()

    # =========================================================================
    # UI SETUP
    # =========================================================================

    def _setup_control_area(self):
        """Build control area."""
        # --- Entity Selection ---
        entity_box = gui.widgetBox(self.controlArea, "📋 Entity Selection")

        form = QGridLayout()
        form.setColumnStretch(1, 1)

        form.addWidget(QLabel("Entity Type:"), 0, 0)
        self.entity_combo = QComboBox()
        entity_labels = [cfg["label"] for cfg in ENTITY_CONFIGS.values()]
        self.entity_combo.addItems(entity_labels)
        self.entity_combo.setCurrentIndex(self.entity_type_idx)
        self.entity_combo.currentIndexChanged.connect(self._on_entity_changed)
        form.addWidget(self.entity_combo, 0, 1)

        self.entity_desc = QLabel("")
        self.entity_desc.setStyleSheet("color: #888; font-size: 11px;")
        self.entity_desc.setWordWrap(True)
        form.addWidget(self.entity_desc, 1, 0, 1, 2)

        entity_box.layout().addLayout(form)
        self._update_entity_description()

        # --- Options ---
        options_box = gui.widgetBox(self.controlArea, "⚙️ Options")

        merge_layout = QHBoxLayout()
        merge_layout.addWidget(QLabel("Merge Type:"))
        self.merge_combo = QComboBox()
        self.merge_combo.addItems(["All Items", "Shared Only"])
        self.merge_combo.setCurrentIndex(self.merge_type_idx)
        self.merge_combo.currentIndexChanged.connect(self._on_merge_changed)
        self.merge_combo.setToolTip(
            "'All Items' includes all entities from any group (outer join);\n"
            "'Shared Only' includes only entities present in multiple groups (inner join)"
        )
        merge_layout.addWidget(self.merge_combo)
        options_box.layout().addLayout(merge_layout)

        # --- Display Options ---
        display_box = gui.widgetBox(self.controlArea, "📊 Display Options")

        topn_form = QGridLayout()
        topn_form.setColumnStretch(1, 1)

        topn_form.addWidget(QLabel("Show Top N:"), 0, 0)
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(10, 1000)
        self.top_n_spin.setValue(self.top_n_display)
        self.top_n_spin.valueChanged.connect(self._on_top_n_changed)
        self.top_n_spin.setToolTip("Number of items to show in results table")
        topn_form.addWidget(self.top_n_spin, 0, 1)

        topn_form.addWidget(QLabel("Top N for Plot:"), 1, 0)
        self.top_n_plot_spin = QSpinBox()
        self.top_n_plot_spin.setRange(3, 30)
        self.top_n_plot_spin.setValue(self.top_n_plot)
        self.top_n_plot_spin.valueChanged.connect(self._on_plot_n_changed)
        self.top_n_plot_spin.setToolTip("Top N items per group for downstream plot widget")
        topn_form.addWidget(self.top_n_plot_spin, 1, 1)

        display_box.layout().addLayout(topn_form)

        # --- Status ---
        status_box = gui.widgetBox(self.controlArea, "📊 Status")
        self.status_label = QLabel("No data loaded")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        status_box.layout().addWidget(self.status_label)

        # --- Buttons ---
        btn_layout = QHBoxLayout()

        self.count_btn = QPushButton("📊 Count Entities")
        self.count_btn.clicked.connect(self._run_count)
        self.count_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; border: none;
                border-radius: 4px; padding: 8px 16px;
                color: white; font-weight: bold;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        btn_layout.addWidget(self.count_btn)
        self.controlArea.layout().addLayout(btn_layout)

        # Quick actions
        quick_label = QLabel("Quick Actions:")
        quick_label.setStyleSheet("color: #888; font-size: 11px;")
        self.controlArea.layout().addWidget(quick_label)

        quick_layout = QHBoxLayout()

        self.count_all_btn = QPushButton("Count All")
        self.count_all_btn.clicked.connect(self._count_all_entities)
        self.count_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e7ff; border: 1px solid #6366f1;
                border-radius: 4px; padding: 6px 12px;
                color: #4338ca; font-weight: bold;
            }
            QPushButton:hover { background-color: #c7d2fe; }
        """)
        quick_layout.addWidget(self.count_all_btn)

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
        quick_layout.addWidget(self.export_btn)

        self.controlArea.layout().addLayout(quick_layout)

        # Auto apply
        self.auto_apply_cb = QCheckBox("Auto apply on data change")
        self.auto_apply_cb.setChecked(self.auto_apply)
        self.auto_apply_cb.toggled.connect(self._on_auto_apply_changed)
        self.controlArea.layout().addWidget(self.auto_apply_cb)

    def _setup_main_area(self):
        """Build main area with tabs."""
        self.tab_widget = QTabWidget()
        self.mainArea.layout().addWidget(self.tab_widget)

        # --- Results Table Tab ---
        self.table_tab = QWidget()
        table_layout = QVBoxLayout(self.table_tab)

        self.results_header = QLabel("No results yet")
        self.results_header.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px;"
        )
        table_layout.addWidget(self.results_header)

        self.results_info = QLabel("")
        self.results_info.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        table_layout.addWidget(self.results_info)

        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSortingEnabled(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        table_layout.addWidget(self.results_table)
        self.tab_widget.addTab(self.table_tab, "📋 Table")

        # --- Summary Tab (for Count All) ---
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        self.summary_header = QLabel("Count All Summary")
        self.summary_header.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px;"
        )
        summary_layout.addWidget(self.summary_header)

        self.summary_table = QTableWidget()
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        summary_layout.addWidget(self.summary_table)
        self.tab_widget.addTab(self.summary_tab, "📋 Summary")

        # --- Info Tab ---
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        info_text = QLabel(
            "<h3>Group Counts</h3>"
            "<p>Compare entity frequencies across document groups.</p>"
            "<h4>Input</h4>"
            "<p>Connect data <b>from Setup Groups</b>. The widget auto-detects "
            "columns prefixed <code>Group: </code> and extracts a binary "
            "group matrix from them.</p>"
            "<h4>Entity Types</h4>"
            "<ul>"
            "<li><b>Sources</b> – Journal/conference names</li>"
            "<li><b>Authors</b> – Individual author names</li>"
            "<li><b>Author/Index Keywords</b> – Keyword frequencies</li>"
            "<li><b>Countries</b> – CA or all collaborating countries</li>"
            "<li><b>Affiliations</b> – Institutional affiliations</li>"
            "<li><b>References</b> – Cited references</li>"
            "<li><b>N-grams</b> – Terms from titles or abstracts</li>"
            "</ul>"
            "<h4>Merge Types</h4>"
            "<ul>"
            "<li><b>All Items</b> – Union of items from all groups (outer join)</li>"
            "<li><b>Shared Only</b> – Only items present in 2+ groups (inner join)</li>"
            "</ul>"
            "<h4>Output Columns</h4>"
            "<p>For each group: count, proportion, percentage, rank.<br>"
            "Combined: total count, combined rank, percent rank.</p>"
            "<h4>Downstream</h4>"
            "<p>Connect <b>Counts</b> output to the <b>Group Counts Plot</b> "
            "widget for interactive visualization with hover and selection.</p>"
        )
        info_text.setWordWrap(True)
        info_text.setAlignment(Qt.AlignTop)
        info_layout.addWidget(info_text)
        self.tab_widget.addTab(info_tab, "ℹ️ Info")

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def _on_entity_changed(self, idx):
        self.entity_type_idx = idx
        self._update_entity_description()

    def _on_merge_changed(self, idx):
        self.merge_type_idx = idx

    def _on_top_n_changed(self, val):
        self.top_n_display = val

    def _on_plot_n_changed(self, val):
        self.top_n_plot = val

    def _on_auto_apply_changed(self, checked):
        self.auto_apply = checked
        if checked and self._df is not None and self._gm_df is not None:
            self._run_count()

    def _update_entity_description(self):
        keys = list(ENTITY_CONFIGS.keys())
        idx = min(self.entity_type_idx, len(keys) - 1)
        key = keys[idx]
        config = ENTITY_CONFIGS[key]
        self.entity_desc.setText(
            f"Count {config['item_label'].lower()} occurrences in each group.\n"
            f"Looks for: {', '.join(config['columns'][:3])}"
        )

    # =========================================================================
    # INPUT HANDLER
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

        if data is None:
            self._update_status()
            self._clear_outputs()
            return

        # Convert to DataFrame
        self._df = self._table_to_df(data)

        # Auto-extract group matrix from "Group: xxx" columns
        self._extract_groups()

        self._update_status()

        if self.auto_apply and self._gm_df is not None:
            self._run_count()

    def _extract_groups(self):
        """Extract group matrix from 'Group: xxx' columns in the data."""
        if self._df is None:
            return

        group_cols = [
            c for c in self._df.columns if c.startswith(GROUP_PREFIX)
        ]

        if not group_cols:
            self.Error.no_groups()
            return

        # Build group matrix with clean names (strip prefix)
        gm = pd.DataFrame(index=self._df.index)
        group_names = []
        for col in group_cols:
            clean_name = col[len(GROUP_PREFIX):]
            gm[clean_name] = self._to_binary(self._df[col])
            group_names.append(clean_name)

        self._gm_df = gm
        self._group_names = group_names

        names_preview = ", ".join(group_names[:5])
        if len(group_names) > 5:
            names_preview += f", … (+{len(group_names) - 5} more)"
        self.Information.groups_found(len(group_names), names_preview)

    @staticmethod
    def _to_binary(series: pd.Series) -> pd.Series:
        """
        Robustly convert a Series to binary 0/1.

        Handles all representations that Setup Groups may produce:
        - Numeric 0/1 or 0.0/1.0 (ContinuousVariable path)
        - Strings "Yes"/"No" (DiscreteVariable with values=["No","Yes"])
        - Strings "True"/"False", "1"/"0"
        - Boolean True/False
        """
        # Fast path: already numeric
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1)

        # String-based mapping (covers DiscreteVariable "Yes"/"No" path)
        truthy = {"yes", "true", "1", "1.0", "t", "y"}
        return series.apply(
            lambda v: 1.0 if str(v).strip().lower() in truthy else 0.0
        )

    # =========================================================================
    # CORE LOGIC
    # =========================================================================

    def _run_count(self):
        """Run counting for the selected entity type."""
        self.Error.no_data.clear()
        self.Error.no_groups.clear()
        self.Error.no_column.clear()
        self.Error.count_error.clear()
        self.Warning.empty_result.clear()

        if self._df is None:
            self.Error.no_data()
            return
        if self._gm_df is None:
            self._extract_groups()
            if self._gm_df is None:
                return  # error already set

        keys = list(ENTITY_CONFIGS.keys())
        entity_key = keys[min(self.entity_type_idx, len(keys) - 1)]
        config = ENTITY_CONFIGS[entity_key]

        merge_type = "all items" if self.merge_type_idx == 0 else "shared items"

        try:
            result_df = self._count_entity(config, merge_type)
        except Exception as e:
            self.Error.count_error(str(e))
            logger.exception("Counting failed")
            return

        if result_df is None or result_df.empty:
            self.Warning.empty_result()
            self._clear_outputs()
            return

        self._result_df = result_df

        n_items = len(result_df)
        n_groups = len(self._group_names)
        self.Information.counted(f"{n_items:,}", n_groups)

        # Display results
        self._display_table(result_df, config)

        # Send outputs
        self._send_outputs(result_df)

        self.tab_widget.setCurrentIndex(0)

    def _count_entity(self, config: Dict, merge_type: str) -> pd.DataFrame:
        """Count entity occurrences across groups."""
        # Build data-only DataFrame (without group columns)
        data_cols = [c for c in self._df.columns if not c.startswith(GROUP_PREFIX)]
        df = self._df[data_cols]
        gm_df = self._gm_df

        # Resolve the column name
        col = self._resolve_column(config["columns"], df)
        if col is None:
            raise ValueError(
                f"None of the expected columns found: {config['columns']}"
            )

        count_type = config["count_type"]
        item_label = config["item_label"]

        if count_type == "text" and not HAS_SKLEARN:
            self.Warning.sklearn_missing()
            raise ValueError("sklearn required for n-gram counting")

        # Try Biblium path first
        if HAS_BIBLIUM:
            return self._count_with_biblium(df, gm_df, config, merge_type)
        else:
            return self._count_fallback(
                df, gm_df, col, count_type, item_label, merge_type
            )

    def _count_with_biblium(
        self, df, gm_df, config, merge_type
    ) -> pd.DataFrame:
        """Count using Biblium's BiblioStats + count_occurrences_across_groups."""
        sep = self._detect_separator(df)
        group_names = list(gm_df.columns)

        groups = {}
        for g in group_names:
            mask = gm_df[g].astype(bool)
            group_df = df[mask].copy()
            if group_df.empty:
                continue

            bs = BiblioStats(
                df=group_df, db="", preprocess_level=0,
                label_docs=False, res_folder=None,
            )
            bs.default_separator = sep
            groups[g] = bs

        if not groups:
            raise ValueError("All groups are empty")

        active_groups = list(groups.keys())
        active_gm = gm_df[active_groups]

        result_df = utilsbib.count_occurrences_across_groups(
            groups=groups,
            group_matrix=active_gm,
            count_func_name=config["method"],
            merge_type=merge_type,
        )
        return result_df

    def _count_fallback(
        self, df, gm_df, col, count_type, item_label, merge_type
    ) -> pd.DataFrame:
        """Fallback counting without Biblium."""
        sep = self._detect_separator(df)
        how = "outer" if merge_type == "all items" else "inner"
        group_names = list(gm_df.columns)

        dfs = []
        for g in group_names:
            mask = gm_df[g].astype(bool)
            group_df = df[mask]
            if group_df.empty:
                continue

            counts_df = self._count_single_group(
                group_df, col, count_type, item_label, sep
            )
            rename_map = {
                c: f"{c} ({g})" for c in counts_df.columns if c != item_label
            }
            counts_df = counts_df.rename(columns=rename_map)
            dfs.append(counts_df)

        if not dfs:
            raise ValueError("All groups are empty")

        key_col = item_label
        merged = reduce(
            lambda l, r: pd.merge(l, r, on=key_col, how=how), dfs
        )

        for c in merged.columns:
            if c == key_col:
                continue
            if "rank" not in c.lower():
                merged[c] = merged[c].fillna(0)
            try:
                merged[c] = pd.to_numeric(merged[c])
            except (ValueError, TypeError):
                pass

        merged = self._add_combined_stats(merged, group_names)
        return merged

    def _count_single_group(
        self, group_df, col, count_type, item_label, sep
    ) -> pd.DataFrame:
        """Count entities in a single group's data."""
        data = group_df[col].dropna().astype(str).str.strip()
        data = data[data != ""]

        if count_type == "single":
            counts = Counter(data)
        elif count_type == "list":
            all_items = []
            for val in data:
                items = [x.strip() for x in val.split(sep) if x.strip()]
                all_items.extend(items)
            counts = Counter(all_items)
        elif count_type == "text":
            if not HAS_SKLEARN or data.empty:
                return pd.DataFrame(columns=[item_label, "Number of documents"])
            vectorizer = CountVectorizer(ngram_range=(1, 2))
            mat = vectorizer.fit_transform(data)
            terms = vectorizer.get_feature_names_out()
            doc_counts = (mat > 0).sum(axis=0).A1
            counts = dict(zip(terms, doc_counts))
        else:
            counts = Counter(data)

        if not counts:
            return pd.DataFrame(columns=[item_label, "Number of documents"])

        n_group = len(group_df)
        items = list(counts.keys())
        values = list(counts.values())
        result = pd.DataFrame({
            item_label: items,
            "Number of documents": values,
            "Proportion of documents": [v / n_group if n_group else 0 for v in values],
            "Percentage of documents": [v / n_group * 100 if n_group else 0 for v in values],
        })

        result = result.sort_values(
            "Number of documents", ascending=False
        ).reset_index(drop=True)
        n = len(result)
        result["Rank"] = range(1, n + 1)
        result["Percentrank"] = [(n - i) / max(n - 1, 1) for i in range(n)]
        return result

    def _add_combined_stats(self, merged, group_names):
        """Add combined columns to merged DataFrame."""
        count_cols = [
            c for c in merged.columns if c.startswith("Number of documents (")
        ]
        if count_cols:
            merged["Number of documents (Combined)"] = merged[count_cols].sum(axis=1)
            total = merged["Number of documents (Combined)"].sum()
            if total > 0:
                merged["Proportion of documents (Combined)"] = (
                    merged["Number of documents (Combined)"] / total
                )
                merged["Percentage of documents (Combined)"] = (
                    merged["Number of documents (Combined)"] / total * 100
                )
            n = len(merged)
            ranks = merged["Number of documents (Combined)"].rank(
                method="first", ascending=False
            ).astype(int)
            merged["Rank (Combined)"] = ranks
            if n > 0:
                merged["PercentRank (Combined)"] = (n - ranks + 1) / n * 100
        return merged

    def _count_all_entities(self):
        """Count all entity types at once."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()

        if self._df is None:
            self.Error.no_data()
            return
        if self._gm_df is None:
            self._extract_groups()
            if self._gm_df is None:
                return

        merge_type = "all items" if self.merge_type_idx == 0 else "shared items"
        results = {}
        errors = []

        self.Information.counting_all()

        for key, config in ENTITY_CONFIGS.items():
            try:
                result_df = self._count_entity(config, merge_type)
                if result_df is not None and not result_df.empty:
                    results[key] = result_df
            except Exception as e:
                errors.append(f"{config['label']}: {e}")

        self._all_results = results
        self.Information.clear()

        if errors:
            self.Warning.partial_results(len(errors))

        # Display summary
        self._display_count_all_summary(results, errors)

        # Send first successful result
        if results:
            first_key = list(results.keys())[0]
            first_df = results[first_key]
            self._result_df = first_df
            first_config = ENTITY_CONFIGS[first_key]
            self._display_table(first_df, first_config)
            self._send_outputs(first_df)

        self.tab_widget.setCurrentIndex(1)  # Summary tab

    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================

    def _display_table(self, df: pd.DataFrame, config: Dict):
        """Display results in the table tab."""
        top_n = self.top_n_display
        display_df = df.head(top_n) if top_n > 0 and len(df) > top_n else df

        self.results_header.setText(f"{config['label']} Counts by Group")
        self.results_info.setText(
            f"Showing {len(display_df):,} of {len(df):,} items  |  "
            f"Groups: {', '.join(self._group_names)}"
        )

        self.results_table.setSortingEnabled(False)
        self.results_table.clear()
        self.results_table.setRowCount(len(display_df))
        self.results_table.setColumnCount(len(display_df.columns))
        self.results_table.setHorizontalHeaderLabels(
            [str(c) for c in display_df.columns]
        )

        for row_idx in range(len(display_df)):
            for col_idx, col_name in enumerate(display_df.columns):
                val = display_df.iloc[row_idx, col_idx]
                item = QTableWidgetItem()

                if isinstance(val, (int, np.integer)):
                    item.setData(Qt.DisplayRole, int(val))
                elif isinstance(val, (float, np.floating)):
                    if pd.isna(val):
                        item.setText("")
                    elif val == int(val):
                        item.setData(Qt.DisplayRole, int(val))
                    else:
                        item.setData(Qt.DisplayRole, round(float(val), 4))
                else:
                    item.setText(str(val) if pd.notna(val) else "")

                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                if "(Combined)" in str(col_name):
                    item.setBackground(QColor(240, 245, 255))

                self.results_table.setItem(row_idx, col_idx, item)

        self.results_table.setSortingEnabled(True)
        self.results_table.resizeColumnsToContents()

    def _display_count_all_summary(
        self, results: Dict[str, pd.DataFrame], errors: List[str]
    ):
        """Display summary for Count All operation."""
        summary_rows = []
        for key, result_df in results.items():
            config = ENTITY_CONFIGS[key]
            summary_rows.append({
                "Entity Type": config["label"],
                "Unique Items": len(result_df),
                "Status": "✅ Success",
            })

        for err in errors:
            entity_name = err.split(":")[0].strip()
            summary_rows.append({
                "Entity Type": entity_name,
                "Unique Items": 0,
                "Status": "❌ Error",
            })

        summary_df = pd.DataFrame(summary_rows)

        self.summary_header.setText(
            f"Count All Summary — {len(results)} succeeded, {len(errors)} failed"
        )

        self.summary_table.setSortingEnabled(False)
        self.summary_table.clear()
        self.summary_table.setRowCount(len(summary_df))
        self.summary_table.setColumnCount(len(summary_df.columns))
        self.summary_table.setHorizontalHeaderLabels(list(summary_df.columns))

        for row_idx in range(len(summary_df)):
            for col_idx, col_name in enumerate(summary_df.columns):
                val = summary_df.iloc[row_idx, col_idx]
                item = QTableWidgetItem()
                if isinstance(val, (int, np.integer)):
                    item.setData(Qt.DisplayRole, int(val))
                else:
                    item.setText(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                if "Success" in str(val):
                    item.setBackground(QColor(220, 255, 220))
                elif "Error" in str(val):
                    item.setBackground(QColor(255, 220, 220))

                self.summary_table.setItem(row_idx, col_idx, item)

        self.summary_table.setSortingEnabled(True)
        self.summary_table.resizeColumnsToContents()

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _resolve_column(
        self, candidates: List[str], df: pd.DataFrame
    ) -> Optional[str]:
        """Find the first matching column name in the DataFrame."""
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    def _detect_separator(self, df: pd.DataFrame) -> str:
        """Detect the dominant separator in multi-value columns."""
        probe_cols = [
            "Author Keywords", "Index Keywords", "Authors",
            "Affiliations", "References",
        ]
        scores = {"; ": 0, "|": 0, ";": 0}

        for col_name in probe_cols:
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
            M[:, i] = df[col].astype(str).values

        return Table.from_numpy(domain, X, metas=M if metas else None)

    def _send_outputs(self, result_df: pd.DataFrame):
        """Send counts to outputs."""
        if result_df is None or result_df.empty:
            self.Outputs.counts.send(None)
            self.Outputs.filtered_counts.send(None)
            return

        full_table = self._df_to_table(result_df)
        self.Outputs.counts.send(full_table)

        top_n = self.top_n_display
        if top_n > 0 and len(result_df) > top_n:
            filtered = result_df.head(top_n)
            self.Outputs.filtered_counts.send(self._df_to_table(filtered))
        else:
            self.Outputs.filtered_counts.send(full_table)

    def _clear_outputs(self):
        """Clear all outputs."""
        self.Outputs.counts.send(None)
        self.Outputs.filtered_counts.send(None)
        self._result_df = None

        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
        self.results_header.setText("No results yet")
        self.results_info.setText("")

    def _update_status(self):
        """Update the status label."""
        parts = []
        if self._df is not None:
            parts.append(f"Data: {len(self._df):,} documents")
        else:
            parts.append("Data: not connected")

        if self._group_names:
            parts.append(f"Groups ({len(self._group_names)}): "
                         + ", ".join(self._group_names[:5]))
        else:
            parts.append("Groups: none detected")

        self.status_label.setText("\n".join(parts))

        enabled = self._df is not None and self._gm_df is not None
        self.count_btn.setEnabled(enabled)
        self.count_all_btn.setEnabled(enabled)

    def _export_results(self):
        """Export current results to file."""
        if self._result_df is None:
            return

        from AnyQt.QtWidgets import QFileDialog

        fname, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "",
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
                self._result_df.to_excel(fname, index=False)
            logger.info(f"Results exported to {fname}")
        except Exception as e:
            logger.exception(f"Export failed: {e}")


# =============================================================================
# WIDGET PREVIEW
# =============================================================================

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWGroupCounts).run()
