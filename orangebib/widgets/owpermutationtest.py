# -*- coding: utf-8 -*-
"""
Permutation Inference Widget
============================
Permutation-based association tests for bibliometric group analyses with
*overlapping* subgroups and/or multi-valued entities.

Why permutations?
-----------------
When groups are non-disjoint (a document can belong to several groups —
e.g. thematic clusters defined by keywords) or the entity is multi-valued
(authors, keywords, countries), the classical Pearson chi-squared test of
independence is biased: independence-of-observations is violated, marginals
are inflated, and asymptotic p-values become anti-conservative.

This widget runs the row-permutation framework from
``biblium.utilsbib_modules.permutation``:

1. Hold the document × entity indicator matrix fixed.
2. Permute the document × group indicator matrix row-wise B times.
3. Recompute the chosen test statistic on each permuted contingency table.
4. Report ``p = (1 + #{T_b ≥ T_0}) / (1 + B)`` with a 95% Clopper–Pearson CI.

This preserves group sizes, entity marginals, and the overlap structure;
it breaks only the document-level association, exactly the null hypothesis
of interest.

Test statistics available
-------------------------
- **chi²** — global Pearson χ² statistic on the contingency table
- **Cramér's V** — effect-size variant of χ² (scaled to [0, 1])
- **Total inertia** — Φ² = χ²/N, the sum of CA dimension inertias
- **Dimension inertias** — vector of per-CA-dimension inertias
- **Standardized residuals** — cell-level p-values with BH/Holm/Bonferroni
  multiple-testing correction

Inputs
------
Data : Table
    Bibliographic data table.
Group Matrix : Table (optional)
    A document × group 0/1 matrix (e.g. from Setup Groups).
    If absent, columns starting with ``Group: `` in *Data* are used.

Outputs
-------
Result : Table
    One-row table with observed statistic, p-value, CI, B, elapsed time.
Null Distribution : Table
    Per-permutation values of the statistic (for histogram/QQ).
Cell p-values : Table (residuals test only)
    Group × entity cells with raw and adjusted p-values.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:  # noqa: BLE001
    HAS_PG = False
import pandas as pd
from AnyQt.QtGui import QFont
from AnyQt.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)

from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output

from orangebib.base import (
    BaseBibliumWidget,
    fmt_value,
    get_biblium_submodule,
    split_list_cell,
)

logger = logging.getLogger(__name__)


GROUP_PREFIX = "Group: "

# Test statistic options shown in the UI
TEST_OPTIONS: list[tuple[str, str, str]] = [
    # (label, biblium_test_name, alternative)
    ("Chi² (global)", "chi2", "greater"),
    ("Cramér's V (effect size)", "cramers_v", "greater"),
    ("Total inertia (Φ²)", "total_inertia", "greater"),
    ("CA dimension inertias", "dimension_inertias", "greater"),
    ("Standardized residuals (cell p-values)", "residuals", "two-sided"),
]

MULTIPLE_TESTING_OPTIONS: list[tuple[str, str]] = [
    ("Benjamini–Hochberg (FDR)", "bh"),
    ("Holm", "holm"),
    ("Bonferroni", "bonferroni"),
    ("None", "none"),
]

ENTITY_PRESETS: list[tuple[str, list[str], str]] = [
    # (UI label, candidate columns, value type: "single" or "list")
    ("Author Keywords", ["Author Keywords", "Keywords", "DE",
                         "author_keywords"], "list"),
    ("Index Keywords", ["Index Keywords", "Keywords Plus", "ID",
                        "indexed_keywords"], "list"),
    ("Authors", ["Authors", "Author", "AU", "Author full names"], "list"),
    ("Sources", ["Source title", "Source", "Journal", "SO"], "single"),
    ("Document Types", ["Document Type", "Document type", "type", "DT"],
     "single"),
    ("Countries", ["Countries of Authors", "Countries", "Country",
                   "authorships.countries"], "list"),
    ("Affiliations", ["Affiliations", "Affiliation", "C1"], "list"),
    ("References", ["References", "Cited References", "CR"], "list"),
    ("Custom column…", [], "auto"),
]


class _NumericItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically when a value is supplied."""

    def __init__(self, text, value=None):
        super().__init__(text)
        self._value = value

    def __lt__(self, other):
        try:
            if self._value is not None and getattr(other, "_value", None) is not None:
                return self._value < other._value
        except Exception:  # noqa: BLE001
            pass
        return super().__lt__(other)


class OWPermutationTest(BaseBibliumWidget):
    """Permutation-based inference for overlapping subgroups."""

    name = "Permutation Inference"
    description = (
        "Permutation tests of group×entity association — valid for "
        "overlapping subgroups and multi-valued entities."
    )
    icon = "icons/permutation_test.svg"
    priority = 690
    keywords = [
        "permutation", "test", "inference", "chi-square", "chi2",
        "cramer", "overlap", "groups", "association", "p-value",
    ]

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
        group_matrix = Input(
            "Group Matrix", Table, doc="Document × group binary matrix",
        )

    class Outputs:
        result = Output("Result", Table,
                        doc="Single-row result table (observed, p, CI, B)")
        null_distribution = Output(
            "Null Distribution", Table,
            doc="Per-permutation values of the chosen statistic",
        )
        cell_pvalues = Output(
            "Cell p-values", Table,
            doc="Group × entity p-values (residuals test only)",
        )
        selected = Output(
            "Selected Documents", Table,
            doc="Documents for the clicked group×entity cell",
        )

    # =========================================================================
    # SETTINGS
    # =========================================================================

    test_index = settings.Setting(0)         # index in TEST_OPTIONS
    entity_index = settings.Setting(0)       # index in ENTITY_PRESETS
    custom_entity_column = settings.Setting("")
    custom_separator = settings.Setting("; ")
    n_permutations = settings.Setting(2000)  # 0 = adaptive
    target_seconds = settings.Setting(5)     # used when n_permutations == 0
    random_seed = settings.Setting(42)
    multiple_testing_index = settings.Setting(0)
    n_dimensions = settings.Setting(2)       # for dimension_inertias
    auto_apply = settings.Setting(True)

    want_main_area = True
    resizing_enabled = True

    def __init__(self):
        super().__init__()

        self._data: Table | None = None
        self._group_table: Table | None = None
        self._df: pd.DataFrame | None = None
        self._group_df: pd.DataFrame | None = None  # binary doc×group
        self._entity_df: pd.DataFrame | None = None  # binary doc×entity
        self._result_df: pd.DataFrame | None = None
        self._null_df: pd.DataFrame | None = None
        self._cell_df: pd.DataFrame | None = None

        self._setup_control_area()
        self._setup_main_area()

    # =========================================================================
    # GUI
    # =========================================================================

    def _setup_control_area(self):
        # --- Test statistic ---
        test_box = gui.widgetBox(self.controlArea, "Test Statistic")
        self.test_combo = QComboBox()
        for label, _, _ in TEST_OPTIONS:
            self.test_combo.addItem(label)
        self.test_combo.setCurrentIndex(self.test_index)
        self.test_combo.currentIndexChanged.connect(self._on_test_changed)
        test_box.layout().addWidget(self.test_combo)

        # Multiple-testing combo (relevant only for residuals)
        mt_row = QHBoxLayout()
        mt_row.addWidget(QLabel("Adjustment:"))
        self.mt_combo = QComboBox()
        for label, _ in MULTIPLE_TESTING_OPTIONS:
            self.mt_combo.addItem(label)
        self.mt_combo.setCurrentIndex(self.multiple_testing_index)
        self.mt_combo.currentIndexChanged.connect(self._on_mt_changed)
        mt_row.addWidget(self.mt_combo)
        test_box.layout().addLayout(mt_row)
        self._update_mt_visibility()

        # n_dimensions for dimension_inertias
        nd_row = QHBoxLayout()
        nd_row.addWidget(QLabel("Dimensions (for dim. inertias):"))
        gui.spin(test_box, self, "n_dimensions", 1, 10, step=1,
                 callback=self._on_option_changed)

        # --- Entity ---
        ent_box = gui.widgetBox(self.controlArea, "Entity")
        self.entity_combo = QComboBox()
        for label, _, _ in ENTITY_PRESETS:
            self.entity_combo.addItem(label)
        self.entity_combo.setCurrentIndex(self.entity_index)
        self.entity_combo.currentIndexChanged.connect(self._on_entity_changed)
        ent_box.layout().addWidget(self.entity_combo)

        # Custom entity column row (visible only when "Custom" is selected)
        self.custom_col_combo = QComboBox()
        self.custom_col_combo.setEditable(True)
        self.custom_col_combo.currentTextChanged.connect(
            self._on_custom_col_changed
        )
        cc_row = QHBoxLayout()
        cc_row.addWidget(QLabel("Column:"))
        cc_row.addWidget(self.custom_col_combo)
        ent_box.layout().addLayout(cc_row)

        sep_row = QHBoxLayout()
        sep_row.addWidget(QLabel("Separator (lists):"))
        self.sep_combo = QComboBox()
        self.sep_combo.setEditable(True)
        for s in ("; ", "|", ",", "||", "Auto"):
            self.sep_combo.addItem(s)
        idx = self.sep_combo.findText(self.custom_separator)
        if idx >= 0:
            self.sep_combo.setCurrentIndex(idx)
        self.sep_combo.currentTextChanged.connect(self._on_sep_changed)
        sep_row.addWidget(self.sep_combo)
        ent_box.layout().addLayout(sep_row)

        # --- Permutation parameters ---
        perm_box = gui.widgetBox(self.controlArea, "Permutation Parameters")
        gui.spin(perm_box, self, "n_permutations", 0, 100000, step=100,
                 label="Permutations B (0 = adaptive):",
                 callback=self._on_option_changed)
        gui.spin(perm_box, self, "target_seconds", 1, 60, step=1,
                 label="Adaptive time budget (s):",
                 callback=self._on_option_changed)
        gui.spin(perm_box, self, "random_seed", 0, 2**31 - 1, step=1,
                 label="Random seed:",
                 callback=self._on_option_changed)

        # --- Apply ---
        self.run_btn = gui.button(
            self.controlArea, self, "Run Permutation Test",
            callback=self.commit, autoDefault=False,
        )
        self.run_btn.setMinimumHeight(35)
        gui.checkBox(self.controlArea, self, "auto_apply",
                     "Apply automatically")
        self.controlArea.layout().addStretch(1)

    def _setup_main_area(self):
        self.tabs = QTabWidget()
        self.mainArea.layout().addWidget(self.tabs)

        # Result tab
        self.result_widget = QWidget()
        rl = QVBoxLayout(self.result_widget)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 10))
        rl.addWidget(self.result_text)
        self.tabs.addTab(self.result_widget, "Result")

        # Null distribution tab
        self.null_widget = QWidget()
        nl = QVBoxLayout(self.null_widget)
        self.null_table = QTableWidget()
        self.null_table.setSelectionBehavior(QTableWidget.SelectRows)
        nl.addWidget(self.null_table)
        self.tabs.addTab(self.null_widget, "Null Distribution")

        if HAS_PG:
            self.hist_widget = QWidget()
            hl = QVBoxLayout(self.hist_widget)
            self.hist_note = QLabel("")
            self.hist_note.setWordWrap(True)
            self.hist_note.setStyleSheet("color:#7f8c8d;")
            hl.addWidget(self.hist_note)
            self.null_plot = pg.PlotWidget(background="w")
            self.null_plot.setLabel("bottom", "Statistic")
            self.null_plot.setLabel("left", "Frequency")
            hl.addWidget(self.null_plot)
            self.tabs.addTab(self.hist_widget, "Histogram")
            self.resid_widget = QWidget()
            rh = QVBoxLayout(self.resid_widget)
            self.resid_note = QLabel(
                "Available only for the 'Standardized residuals' test. "
                "Click a cell to output its documents.")
            self.resid_note.setWordWrap(True)
            self.resid_note.setStyleSheet("color:#7f8c8d;")
            rh.addWidget(self.resid_note)
            self.resid_plot = pg.PlotWidget(background="w")
            self.resid_img = pg.ImageItem()
            self.resid_plot.addItem(self.resid_img)
            self.resid_plot.scene().sigMouseClicked.connect(self._on_resid_clicked)
            rh.addWidget(self.resid_plot)
            self.tabs.addTab(self.resid_widget, "Residuals heatmap")

        # Cell p-values tab (residuals test)
        self.cells_widget = QWidget()
        cl = QVBoxLayout(self.cells_widget)
        self.cells_note = QLabel("")
        self.cells_note.setWordWrap(True)
        self.cells_note.setStyleSheet("color:#7f8c8d;")
        cl.addWidget(self.cells_note)
        self.cells_table = QTableWidget()
        self.cells_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cells_table.itemSelectionChanged.connect(self._on_cell_row_selected)
        cl.addWidget(self.cells_table)
        self.tabs.addTab(self.cells_widget, "Cell p-values")

    # =========================================================================
    # Settings change handlers
    # =========================================================================

    def _maybe_recompute(self):
        # the permutation test is fast here, so recompute on key control changes
        # even when "Apply automatically" is off (otherwise e.g. switching to the
        # residuals test would leave the Cell p-values tab stale/empty).
        if self._df is not None and not self._df.empty:
            self.commit()

    def _on_test_changed(self, idx: int):
        self.test_index = idx
        self._update_mt_visibility()
        self._maybe_recompute()

    def _on_mt_changed(self, idx: int):
        self.multiple_testing_index = idx
        self._maybe_recompute()

    def _on_entity_changed(self, idx: int):
        self.entity_index = idx
        self._maybe_recompute()

    def _on_custom_col_changed(self, txt: str):
        self.custom_entity_column = txt
        if self.auto_apply:
            self.commit()

    def _on_sep_changed(self, txt: str):
        self.custom_separator = txt
        if self.auto_apply:
            self.commit()

    def _on_option_changed(self):
        if self.auto_apply:
            self.commit()

    def _update_mt_visibility(self):
        is_residuals = TEST_OPTIONS[self.test_index][1] == "residuals"
        self.mt_combo.setEnabled(is_residuals)

    # =========================================================================
    # Inputs
    # =========================================================================

    @Inputs.data
    def set_data(self, data: Table | None):
        self._data = data
        self._df = self._table_to_df(data) if data is not None else None
        self._refresh_custom_col_combo()
        # Always compute when new data arrives, so connecting Setup Groups →
        # Data immediately produces a result (the "Apply automatically" box only
        # governs recomputation on in-widget parameter changes).
        if data is not None:
            self.commit()

    @Inputs.group_matrix
    def set_group_matrix(self, table: Table | None):
        self._group_table = table
        # If the incoming table is really a full data table (carries entity
        # columns), use it as the data source too — so the widget works no
        # matter whether Setup Groups' 'Data' output landed on this input or
        # on the Data input.
        if table is not None:
            gdf = self._table_to_df(table)
            non_group = [c for c in gdf.columns if not c.startswith(GROUP_PREFIX)]
            text_like = [c for c in non_group
                         if not pd.api.types.is_numeric_dtype(gdf[c])]
            if (self._df is None or self._df.empty) and (text_like or len(non_group) > 3):
                self._df = gdf
                self._refresh_custom_col_combo()
        if table is not None or self._df is not None:
            self.commit()

    def _refresh_custom_col_combo(self):
        self.custom_col_combo.blockSignals(True)
        self.custom_col_combo.clear()
        if self._df is not None:
            for c in self._df.columns:
                if not c.startswith(GROUP_PREFIX):
                    self.custom_col_combo.addItem(str(c))
        if self.custom_entity_column:
            idx = self.custom_col_combo.findText(self.custom_entity_column)
            if idx >= 0:
                self.custom_col_combo.setCurrentIndex(idx)
            else:
                self.custom_col_combo.setEditText(self.custom_entity_column)
        self.custom_col_combo.blockSignals(False)

    # =========================================================================
    # Resolve group matrix and entity matrix
    # =========================================================================

    def _resolve_group_matrix(self) -> pd.DataFrame | None:
        """Return the document × group binary DataFrame or None."""
        # 1) Prefer "Group: " columns wherever they are (Data input, or a full
        #    table that arrived on the Group Matrix input).
        for src in (self._df,
                    self._table_to_df(self._group_table)
                    if self._group_table is not None else None):
            if src is None or getattr(src, "empty", True):
                continue
            gcols = [c for c in src.columns if c.startswith(GROUP_PREFIX)]
            if gcols:
                gg = src[gcols].copy()
                gg.columns = [c[len(GROUP_PREFIX):] for c in gcols]
                return (gg.apply(pd.to_numeric, errors="coerce").fillna(0) > 0).astype(int)

        # 2) A genuine binary group matrix on the Group Matrix input (keep only
        #    columns whose values are 0/1 — never Year, Cited by, …).
        if self._group_table is not None:
            gdf = self._table_to_df(self._group_table)
            keep = []
            for c in gdf.columns:
                col = pd.to_numeric(gdf[c], errors="coerce")
                vals = set(pd.unique(col.dropna()))
                if col.notna().sum() > 0 and vals and vals.issubset({0, 1, 0.0, 1.0}):
                    keep.append(c)
            gdf = gdf[keep].fillna(0)
            return (gdf.astype(float) > 0).astype(int) if not gdf.empty else None

        # 3) Fall back to "Group: " columns in main data
        if self._df is None:
            return None
        gcols = [c for c in self._df.columns if c.startswith(GROUP_PREFIX)]
        if not gcols:
            return None
        gdf = self._df[gcols].copy()
        gdf.columns = [c[len(GROUP_PREFIX):] for c in gcols]
        return (
            gdf.apply(pd.to_numeric, errors="coerce").fillna(0) > 0
        ).astype(int)

    def _resolve_entity_matrix(self) -> tuple[pd.DataFrame | None, str]:
        """Build a document × entity 0/1 matrix from selected column.

        Returns (matrix, entity_label) or (None, "").
        """
        if self._df is None or self._df.empty:
            return None, ""

        preset_label, candidates, vtype = ENTITY_PRESETS[self.entity_index]

        if vtype == "auto":
            col = self.custom_entity_column.strip()
            if not col or col not in self._df.columns:
                return None, ""
            vtype_eff = self._guess_value_type(self._df[col])
        else:
            col = None
            for cand in candidates:
                if cand in self._df.columns:
                    col = cand
                    break
            if col is None:
                return None, preset_label
            vtype_eff = vtype

        series = self._df[col]

        if vtype_eff == "single":
            # One value per cell -> dummy-encode
            mat = pd.get_dummies(series.astype(str)).astype(int)
            mat.index = self._df.index
            return mat, col

        # list
        sep = self.custom_separator
        if sep == "Auto":
            sample = series.dropna()
            sep = "|" if (len(sample) > 0
                          and "|" in str(sample.iloc[0])) else "; "
        # Build sparse-ish wide matrix
        records: list[dict[str, int]] = []
        for v in series:
            items = split_list_cell(v, sep)
            records.append({it: 1 for it in items})
        mat = pd.DataFrame(records).fillna(0).astype(int)
        # Drop empty entities (always 0)
        mat = mat.loc[:, (mat.sum(axis=0) > 0)]
        return mat, col

    @staticmethod
    def _guess_value_type(series: pd.Series) -> str:
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return "single"
        for v in sample:
            s = str(v)
            if any(sep in s for sep in ("; ", "|", ", ")):
                return "list"
        return "single"

    # =========================================================================
    # COMMIT
    # =========================================================================

    def commit(self):
        self.clear_messages()
        self._clear_outputs()

        if self._df is None or self._df.empty:
            if self._group_table is not None:
                # groups are connected but the entity data is not
                self.Error.compute_error(
                    "Connect the *Data* output of Setup Groups to this widget's "
                    "*Data* input. The Group Matrix alone has no entity columns "
                    "(keywords/authors/…) to test. The Data output already carries "
                    "the 'Group: …' columns, so you may connect only that.")
            else:
                self.Error.compute_error(
                    "No data on the *Data* input. Connect the Data output of "
                    "Setup Groups (it carries both the entity columns and the "
                    "'Group: …' group columns).")
            self._send_outputs()
            return

        gdf = self._resolve_group_matrix()
        if gdf is None or gdf.empty or gdf.shape[1] < 2:
            self.Error.compute_error(
                "Need at least 2 group columns. Connect Setup Groups output "
                "or supply 'Group: …' columns in Data."
            )
            self._send_outputs()
            return

        edf, entity_label = self._resolve_entity_matrix()
        if edf is None or edf.empty or edf.shape[1] < 2:
            self.Error.compute_error(
                f"Could not build entity matrix from '{entity_label or 'selected column'}'."
            )
            self._send_outputs()
            return

        # Align rows
        n = min(len(gdf), len(edf))
        if len(gdf) != len(edf):
            logger.warning(
                "Group matrix (%d rows) and entity matrix (%d rows) differ; "
                "trimming to %d.", len(gdf), len(edf), n
            )
            gdf = gdf.iloc[:n].reset_index(drop=True)
            edf = edf.iloc[:n].reset_index(drop=True)

        utilsbib = get_biblium_submodule("utilsbib")
        if not self.has_biblium or utilsbib is None:
            self.Error.compute_error(
                "biblium >= 2.16 is required for permutation tests. "
                "Install it with: pip install biblium>=2.16"
            )
            self._send_outputs()
            return

        try:
            # Lazy import the permutation module
            from biblium.utilsbib_modules import permutation as perm_mod
        except Exception as exc:  # noqa: BLE001
            self.Error.compute_error(
                f"biblium.utilsbib_modules.permutation unavailable: {exc}"
            )
            self._send_outputs()
            return

        test_label, test_name, alternative = TEST_OPTIONS[self.test_index]
        mt_label, mt_name = MULTIPLE_TESTING_OPTIONS[
            self.multiple_testing_index
        ]
        n_perm = self.n_permutations if self.n_permutations > 0 else None
        seed = int(self.random_seed) if self.random_seed >= 0 else None

        try:
            kwargs: dict[str, Any] = dict(
                test=test_name,
                n_permutations=n_perm,
                target_seconds=float(self.target_seconds),
                random_state=seed,
                show_progress=False,
                warn_disjoint=False,
            )
            if test_name == "residuals":
                kwargs["multiple_testing"] = mt_name
            if test_name == "dimension_inertias":
                kwargs["n_dimensions"] = int(self.n_dimensions)

            res = perm_mod.assoc_permutation_test(gdf, edf, **kwargs)
        except Exception as exc:  # noqa: BLE001
            import traceback
            logger.error("Permutation test failed:\n%s",
                         traceback.format_exc())
            self.Error.compute_error(str(exc))
            self._send_outputs()
            return

        # ---- Build result tables ----
        self._result_df = self._build_result_df(
            res, test_label, test_name, alternative, mt_label, mt_name,
            gdf, edf, entity_label,
        )
        self._obs_value = float(res.observed) if np.ndim(res.observed) == 0 else None
        self._obs_p = float(res.p_value) if np.ndim(res.observed) == 0 else None
        if res.null_distribution is not None and np.ndim(res.observed) == 0:
            self._null_df = pd.DataFrame({
                "Permutation": np.arange(1, res.n_permutations + 1),
                "Statistic": np.asarray(res.null_distribution).ravel(),
            })
        elif res.null_distribution is not None:
            null_arr = np.asarray(res.null_distribution)
            if null_arr.ndim == 2:
                # Vector statistic (e.g. CA dimension inertias): one column per
                # component.
                self._null_df = pd.DataFrame(
                    null_arr,
                    columns=[f"Component {i + 1}"
                             for i in range(null_arr.shape[1])],
                )
                self._null_df.insert(
                    0, "Permutation",
                    np.arange(1, len(self._null_df) + 1),
                )
            else:
                # Matrix statistic (e.g. standardized residuals): the per-cell
                # null is 3-D and has no single distribution table; the per-cell
                # p-values live in the Cell p-values tab instead.
                self._null_df = None

        if test_name == "residuals":
            try:
                self._cell_df = self._build_cell_pvalues(
                    res, gdf, edf,
                    adjusted=mt_name != "none",
                )
            except Exception:  # noqa: BLE001
                logger.exception("cell p-value table build failed")
                self._cell_df = None

        try:
            self._update_displays()
        except Exception as exc:  # noqa: BLE001
            import traceback
            logger.error("display update failed:\n%s", traceback.format_exc())
            self.Error.compute_error(f"display error: {exc}")
        self._send_outputs()
        self.Information.computed(len(self._df))

    # =========================================================================
    # Output construction helpers
    # =========================================================================

    def _build_result_df(self, res, test_label, test_name, alternative,
                         mt_label, mt_name, gdf, edf,
                         entity_label) -> pd.DataFrame:
        rows: list[tuple[str, Any]] = [
            ("Test", test_label),
            ("Alternative", alternative),
            ("Entity column", entity_label),
            ("Documents (rows)", len(gdf)),
            ("Groups", gdf.shape[1]),
            ("Entities (cols)", edf.shape[1]),
            ("Permutations (B)", res.n_permutations),
            ("Elapsed (s)", round(res.elapsed_seconds, 3)),
            ("Random seed", res.seed if res.seed is not None
                              else self.random_seed),
        ]
        if test_name == "residuals":
            rows.append(("Multiple-testing adjustment", mt_label))

        # Observed + p-values
        if np.ndim(res.observed) == 0:
            rows.append(("Observed statistic",
                         round(float(res.observed), 6)))
            rows.append(("p-value", round(float(res.p_value), 6)))
            if isinstance(res.p_value_ci, tuple):
                lo, hi = res.p_value_ci
                rows.append(
                    ("p-value 95% CI", f"[{lo:.4g}, {hi:.4g}]")
                )
        elif np.asarray(res.observed).ndim >= 2:
            # matrix statistic (standardized residuals) — show a compact summary
            obs_arr = np.asarray(res.observed)
            p_arr = np.asarray(res.p_value)
            rows.append(("Statistic", "matrix of standardized residuals"))
            rows.append(("Matrix shape",
                         f"{obs_arr.shape[0]} groups × {obs_arr.shape[1]} entities"))
            flat = int(np.argmax(np.abs(obs_arr)))
            gi, ej = np.unravel_index(flat, obs_arr.shape)
            rows.append(("Largest |residual|", round(float(obs_arr[gi, ej]), 4)))
            rows.append(("Significant cells (p<0.05)", int(np.sum(p_arr < 0.05))))
            rows.append(("(details)", "see the 'Cell p-values' tab"))
        else:
            obs_arr = np.asarray(res.observed)
            p_arr = np.asarray(res.p_value)
            for i in range(obs_arr.shape[0]):
                rows.append((f"Observed[dim {i + 1}]",
                             round(float(obs_arr.flat[i]), 6)))
                rows.append((f"p-value[dim {i + 1}]",
                             round(float(p_arr.flat[i]), 6)))

        return pd.DataFrame(rows, columns=["Indicator", "Value"])

    def _build_cell_pvalues(self, res, gdf, edf,
                            adjusted: bool) -> pd.DataFrame:
        """Long-format Group × Entity p-value table."""
        observed = np.asarray(res.observed)
        p_raw = np.asarray(res.p_value)
        p_adj = res.extra.get("p_value_adjusted") if adjusted else None
        if p_adj is not None:
            p_adj = np.asarray(p_adj)

        groups = list(gdf.columns)
        entities = list(edf.columns)
        observed = np.atleast_2d(np.asarray(observed, dtype=float))
        p_raw = np.atleast_2d(np.asarray(p_raw, dtype=float))
        if p_adj is not None:
            p_adj = np.atleast_2d(np.asarray(p_adj, dtype=float))
        n_g, n_e = observed.shape
        # The statistic table may be (groups x entities) or (entities x groups);
        # detect orientation so labels line up (and the tab is never empty).
        if n_g == len(entities) and n_e == len(groups) and len(groups) != len(entities):
            observed = observed.T
            p_raw = p_raw.T
            if p_adj is not None:
                p_adj = p_adj.T
            n_g, n_e = observed.shape
        rows = []
        for i in range(n_g):
            for j in range(n_e):
                row = {
                    "Group": groups[i] if i < len(groups) else f"G{i + 1}",
                    "Entity": entities[j] if j < len(entities)
                                          else f"E{j + 1}",
                    "Std. residual": round(float(observed[i, j]), 4),
                    "p (raw)": round(float(p_raw[i, j]), 6),
                }
                if p_adj is not None:
                    row["p (adjusted)"] = round(float(p_adj[i, j]), 6)
                rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # Sort by (adjusted) p ascending, residual descending
        sort_col = "p (adjusted)" if "p (adjusted)" in df.columns \
                                   else "p (raw)"
        df = df.sort_values(
            [sort_col, "Std. residual"], ascending=[True, False]
        ).reset_index(drop=True)
        return df

    # =========================================================================
    # Display
    # =========================================================================

    def _update_displays(self):
        if self._result_df is not None and not self._result_df.empty:
            lines = ["=" * 60,
                     "PERMUTATION TEST RESULT",
                     "=" * 60, ""]
            for _, r in self._result_df.iterrows():
                lines.append(f"  {r['Indicator']}: {fmt_value(r['Value'])}")
            lines += ["", "=" * 60]
            self.result_text.setPlainText("\n".join(lines))
        else:
            self.result_text.clear()  # no result -> empty tab, no leftovers

        self._fill_table(self.null_table, self._null_df)
        self._fill_table(self.cells_table, self._cell_df)
        if hasattr(self, "cells_note"):
            if self._cell_df is not None and not self._cell_df.empty:
                self.cells_note.setText(
                    "Per-cell (group × entity) permutation p-values "
                    "with multiple-testing adjustment.")
            else:
                self.cells_note.setText(
                    "Cell p-values are produced only by the "
                    "'Standardized residuals (cell p-values)' test. "
                    "Select it in Test statistic and re-run.")
        self._render_null_hist()
        self._render_resid_heatmap()

    def _render_null_hist(self):
        if not HAS_PG or not hasattr(self, "null_plot"):
            return
        self.null_plot.clear()
        df = self._null_df
        obs = getattr(self, "_obs_value", None)
        vals = (pd.to_numeric(df["Statistic"], errors="coerce").dropna().values
                if (df is not None and "Statistic" in df.columns and not df.empty)
                else np.array([]))
        if len(vals) == 0:
            # nothing to plot -> hide the empty axes, show a note instead
            self.null_plot.setVisible(False)
            self.hist_note.setText(
                "A histogram is shown only for scalar tests "
                "(Chi², Cramér's V, Total inertia). "
                "Matrix/vector tests have no single null distribution.")
            return
        self.null_plot.setVisible(True)
        self.hist_note.setText("Null distribution with the observed statistic "
                               "(red line) and p-value.")
        y, x = np.histogram(vals, bins=min(50, max(10, len(vals) // 20)))
        self.null_plot.addItem(pg.BarGraphItem(
            x0=x[:-1], x1=x[1:], height=y, brush=pg.mkBrush("#9aa7b8"),
            pen=pg.mkPen("w", width=0.5)))
        if obs is not None:
            line = pg.InfiniteLine(pos=obs, angle=90,
                                   pen=pg.mkPen("#d9534f", width=2))
            self.null_plot.addItem(line)
            p = getattr(self, "_obs_p", None)
            txt = pg.TextItem(f"observed = {obs:.3g}" +
                              (f"\np = {p:.4g}" if p is not None else ""),
                              color="#d9534f", anchor=(0, 1))
            txt.setPos(obs, float(y.max()) if len(y) else 1)
            self.null_plot.addItem(txt)

    HEATMAP_TOP_N = 30

    def _render_resid_heatmap(self):
        if not HAS_PG or not hasattr(self, "resid_plot"):
            return
        self.resid_plot.clear()
        self.resid_img = pg.ImageItem()
        self.resid_plot.addItem(self.resid_img)
        self._heat_groups = []
        self._heat_entities = []
        df = self._cell_df

        def _blank(msg):
            self.resid_plot.setVisible(False)
            self.resid_note.setText(msg)

        if df is None or df.empty or "Std. residual" not in df.columns:
            _blank("Available only for the 'Standardized residuals' test. "
                   "Select it in Test statistic and re-run.")
            return
        try:
            piv = df.pivot_table(index="Group", columns="Entity",
                                 values="Std. residual", aggfunc="first").fillna(0.0)
        except Exception:  # noqa: BLE001
            _blank("Could not build the residuals heatmap.")
            return
        if piv.empty:
            _blank("No residuals to display.")
            return
        self.resid_plot.setVisible(True)
        self.resid_note.setText("Standardized residuals (top entities). "
                                "Click a cell to output its documents.")
        # keep only the most extreme entities so the picture is readable
        order = piv.abs().max(axis=0).sort_values(ascending=False)
        top = list(order.head(self.HEATMAP_TOP_N).index)
        piv = piv[top]
        groups = list(piv.index)          # few (e.g. 3)
        entities = list(piv.columns)      # top-N
        # ImageItem shows arr[x, y]: x = group (columns, bottom), y = entity
        # (rows, left). arr[group, entity] -> shape (n_groups, n_entities).
        arr = piv.to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(arr))) or 1.0
        self.resid_img.setImage(arr, levels=(-vmax, vmax))
        try:
            self.resid_img.setColorMap(pg.colormap.get("coolwarm", source="matplotlib"))
        except Exception:  # noqa: BLE001
            try:
                self.resid_img.setColorMap(pg.colormap.get("RdBu"))
            except Exception:  # noqa: BLE001
                pass
        self._heat_groups = groups
        self._heat_entities = entities
        self.resid_plot.getAxis("bottom").setTicks(
            [[(i + 0.5, str(groups[i])[:18]) for i in range(len(groups))]])
        self.resid_plot.getAxis("left").setTicks(
            [[(j + 0.5, str(entities[j])[:30]) for j in range(len(entities))]])
        self.resid_plot.setLabel("bottom", "Group")
        self.resid_plot.setLabel("left", "Entity (top by |residual|)")
        self.resid_plot.setTitle("Standardized residuals (click a cell)")

    def _on_resid_clicked(self, ev):
        groups = getattr(self, "_heat_groups", [])
        entities = getattr(self, "_heat_entities", [])
        if not groups or not entities:
            return
        vb = self.resid_plot.getPlotItem().vb
        p = vb.mapSceneToView(ev.scenePos())
        gi = int(np.floor(p.x())); ej = int(np.floor(p.y()))
        if 0 <= gi < len(groups) and 0 <= ej < len(entities):
            self._output_cell_docs(str(groups[gi]), str(entities[ej]))

    def _on_cell_row_selected(self):
        if self._cell_df is None or self._cell_df.empty:
            return
        rows = self.cells_table.selectionModel().selectedRows() \
            if self.cells_table.selectionModel() else []
        if not rows:
            return
        r = rows[0].row()
        if r >= len(self._cell_df):
            return
        rec = self._cell_df.iloc[r]
        self._output_cell_docs(str(rec.get("Group", "")), str(rec.get("Entity", "")))

    def _output_cell_docs(self, group, entity):
        """Send the documents that belong to *group* and contain *entity*."""
        if self._data is None or self._df is None:
            self.Outputs.selected.send(None)
            return
        gcol = GROUP_PREFIX + group
        if gcol not in self._df.columns:
            # group came from a binary Group Matrix without the prefix
            gcol = group if group in self._df.columns else None
        ent_col = None
        # find which column the entity came from (the resolved entity column)
        _, lbl = self._resolve_entity_matrix()
        if lbl and lbl in self._df.columns:
            ent_col = lbl
        if gcol is None or ent_col is None:
            self.Outputs.selected.send(None)
            return
        gmask = pd.to_numeric(self._df[gcol], errors="coerce").fillna(0) > 0
        emask = self._df[ent_col].astype(str).str.contains(
            re.escape(entity), case=False, na=False)
        idx = [i for i, m in enumerate((gmask & emask).tolist()) if m
               and i < len(self._data)]
        self.Outputs.selected.send(self._data[idx] if idx else None)

    @staticmethod
    def _fill_table(table: QTableWidget, df: pd.DataFrame | None):
        # always reset first so an unavailable result leaves an EMPTY tab
        table.setSortingEnabled(False)
        table.clearContents()
        table.setRowCount(0)
        table.setColumnCount(0)
        if df is None or df.empty:
            return
        nrows = min(len(df), 5000)  # cap for UI responsiveness
        table.setRowCount(nrows)
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(nrows):
            for c in range(table.columnCount()):
                v = df.iloc[r, c]
                num = None
                if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                    num = float(v)
                item = _NumericItem(fmt_value(v), num)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)

    # =========================================================================
    # Outputs
    # =========================================================================

    def _clear_outputs(self):
        self._result_df = None
        self._null_df = None
        self._cell_df = None
        self.result_text.clear()
        for tbl in (self.null_table, self.cells_table):
            tbl.clear()
            tbl.setRowCount(0)

    def _send_outputs(self):
        self.Outputs.result.send(
            self._df_to_table(self._result_df, prefer_strings=True)
        )
        self.Outputs.null_distribution.send(
            self._df_to_table(self._null_df)
        )
        self.Outputs.cell_pvalues.send(
            self._df_to_table(self._cell_df)
        )

    def send_report(self):
        self.report_items([
            ("Test", TEST_OPTIONS[self.test_index][0]),
            ("Permutations", self.n_permutations or "adaptive"),
            ("Random seed", self.random_seed),
        ])
        if self._result_df is not None:
            self.report_table("Result", self.result_text.toPlainText())


if __name__ == "__main__":
    WidgetPreview(OWPermutationTest).run()
