# -*- coding: utf-8 -*-
"""
Crosstabs Widget
===============
Cross-tabulate two categorical variables and test their association.

Wraps :func:`biblium.crosstabs.compute_crosstab`, which builds the
contingency table and computes the chi-squared test (with Fisher's exact for
2x2 tables) plus effect sizes (Cramer's V, phi, contingency coefficient).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QCheckBox, QLabel, QComboBox, QPushButton, QDoubleSpinBox,
    QGridLayout, QTableWidget, QTableWidgetItem,
)

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.crosstabs import compute_crosstab, get_categorical_columns
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    compute_crosstab = None
    get_categorical_columns = None

logger = logging.getLogger(__name__)

DISPLAY_CHOICES = [
    ("Observed", "observed"),
    ("Expected", "expected"),
    ("Row %", "row_pct"),
    ("Column %", "col_pct"),
    ("Total %", "total_pct"),
    ("Std. residuals", "residuals"),
]


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    domain = table.domain
    for var in list(domain.attributes) + list(domain.class_vars) + list(domain.metas):
        try:
            col = table.get_column(var)
        except Exception:  # noqa: BLE001
            continue
        if var.is_discrete:
            values = var.values
            data[var.name] = [
                values[int(v)] if (v == v and 0 <= int(v) < len(values)) else ""
                for v in col
            ]
        else:
            data[var.name] = col
    return pd.DataFrame(data)


def _matrix_to_table(df: Optional[pd.DataFrame], index_name: str) -> Optional[Table]:
    """Convert a contingency matrix (with a meaningful index) to an Orange Table."""
    if df is None or len(df) == 0:
        return None
    out = df.copy()
    out.insert(0, index_name, [str(i) for i in out.index])
    attrs, acols = [], []
    for c in df.columns:
        attrs.append(ContinuousVariable(str(c)))
        acols.append(c)
    domain = Domain(attrs, metas=[StringVariable(index_name)])
    n = len(out)
    X = np.empty((n, len(attrs)), dtype=float)
    for i, c in enumerate(acols):
        X[:, i] = pd.to_numeric(df[c], errors="coerce").values
    M = np.array([[str(v)] for v in out[index_name]], dtype=object)
    return Table.from_numpy(domain, X, metas=M)


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    metas = [StringVariable(str(c)) for c in df.columns]
    domain = Domain([], metas=metas)
    M = df.astype(str).values
    return Table.from_numpy(domain, np.empty((len(df), 0)), metas=M)


class OWCrosstabs(OWWidget):
    """Cross-tabulate two categorical variables with chi-squared test."""

    name = "Crosstabs"
    description = "Contingency table with chi-squared test and effect sizes"
    icon = "icons/crosstabs.svg"
    priority = 650
    keywords = ["crosstab", "contingency", "chi-square", "chi-squared", "fisher",
                "association", "cramers v"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")

    class Outputs:
        table = Output("Contingency Table", Table, doc="Selected display matrix")
        statistics = Output("Statistics", Table, doc="Test statistics and effect sizes")

    row_var = settings.Setting("")
    col_var = settings.Setting("")
    alpha = settings.Setting(0.05)
    display = settings.Setting("observed")
    auto_apply = settings.Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium is required. Install biblium>=2.16.")
        compute_error = Msg("Computation error: {}")

    class Warning(OWWidget.Warning):
        no_categorical = Msg("Need at least two categorical columns")
        low_expected = Msg("{}")

    class Information(OWWidget.Information):
        done = Msg("{}")

    def __init__(self):
        super().__init__()
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._result = None

        self._setup_controls()
        self._setup_main_area()

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _setup_controls(self):
        box = gui.widgetBox(self.controlArea, "Variables")
        grid = QGridLayout()
        grid.addWidget(QLabel("Rows:"), 0, 0)
        self.row_combo = QComboBox()
        self.row_combo.currentTextChanged.connect(self._on_row_changed)
        grid.addWidget(self.row_combo, 0, 1)

        grid.addWidget(QLabel("Columns:"), 1, 0)
        self.col_combo = QComboBox()
        self.col_combo.currentTextChanged.connect(self._on_col_changed)
        grid.addWidget(self.col_combo, 1, 1)

        grid.addWidget(QLabel("Alpha:"), 2, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.001, 0.5)
        self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setDecimals(3)
        self.alpha_spin.setValue(self.alpha)
        self.alpha_spin.valueChanged.connect(lambda v: setattr(self, "alpha", v))
        grid.addWidget(self.alpha_spin, 2, 1)
        box.layout().addLayout(grid)

        dbox = gui.widgetBox(self.controlArea, "Display")
        self.display_combo = QComboBox()
        for label, code in DISPLAY_CHOICES:
            self.display_combo.addItem(label, code)
        idx = [c for _, c in DISPLAY_CHOICES].index(self.display) \
            if self.display in [c for _, c in DISPLAY_CHOICES] else 0
        self.display_combo.setCurrentIndex(idx)
        self.display_combo.currentIndexChanged.connect(self._on_display_changed)
        dbox.layout().addWidget(self.display_combo)

        self.run_btn = QPushButton("Compute")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.auto_cb = QCheckBox("Apply automatically")
        self.auto_cb.setChecked(self.auto_apply)
        self.auto_cb.toggled.connect(lambda c: setattr(self, "auto_apply", c))
        self.controlArea.layout().addWidget(self.auto_cb)

    def _setup_main_area(self):
        sbox = gui.widgetBox(self.mainArea, "Test result")
        self.summary_label = QLabel("No data")
        self.summary_label.setWordWrap(True)
        sbox.layout().addWidget(self.summary_label)

        tbox = gui.widgetBox(self.mainArea, "Contingency table")
        self.matrix_table = QTableWidget()
        self.matrix_table.setMinimumHeight(280)
        tbox.layout().addWidget(self.matrix_table)

    def _on_row_changed(self, t):
        self.row_var = t
        if self.auto_apply:
            self._compute()

    def _on_col_changed(self, t):
        self.col_var = t
        if self.auto_apply:
            self._compute()

    def _on_display_changed(self, i):
        self.display = self.display_combo.itemData(i)
        self._render_matrix()
        self.Outputs.table.send(self._current_matrix_table())

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self._populate_combos()
        if data is None:
            self.Error.no_data()
            return
        if self.auto_apply:
            self._compute()

    def _populate_combos(self):
        self.row_combo.blockSignals(True)
        self.col_combo.blockSignals(True)
        self.row_combo.clear()
        self.col_combo.clear()
        if self._df is not None and not self._df.empty and HAS_BIBLIUM:
            cats = get_categorical_columns(self._df)
            self.row_combo.addItems(cats)
            self.col_combo.addItems(cats)
            if len(cats) < 2:
                self.Warning.no_categorical()
            if self.row_var in cats:
                self.row_combo.setCurrentText(self.row_var)
            elif cats:
                self.row_var = cats[0]
            if self.col_var in cats:
                self.col_combo.setCurrentText(self.col_var)
            elif len(cats) > 1:
                self.col_var = cats[1]
                self.col_combo.setCurrentText(cats[1])
        self.row_combo.blockSignals(False)
        self.col_combo.blockSignals(False)

    def _compute(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            return
        if self._df is None or self._df.empty:
            self.Error.no_data()
            return
        row = self.row_combo.currentText()
        col = self.col_combo.currentText()
        if not row or not col or row == col:
            return
        try:
            res = compute_crosstab(self._df, row, col, alpha=self.alpha, verbose=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("compute_crosstab failed")
            self.Error.compute_error(str(exc))
            self.Outputs.table.send(None)
            self.Outputs.statistics.send(None)
            return
        self._result = res
        chi = res.chi_squared
        es = res.effect_size
        if chi is not None and getattr(chi, "warning", ""):
            self.Warning.low_expected(chi.warning)
        sig = "significant" if (chi and chi.is_significant) else "not significant"
        self.summary_label.setText(
            f"<b>{row}</b> × <b>{col}</b> ({res.n_rows}×{res.n_cols}, "
            f"n={res.n_total})<br>"
            f"χ² = {getattr(chi,'statistic',float('nan')):.3f}, "
            f"df = {getattr(chi,'df','?')}, "
            f"p = {getattr(chi,'p_value',float('nan')):.4f} ({sig})<br>"
            f"Cramér's V = {getattr(es,'cramers_v',float('nan')):.3f} "
            f"({getattr(es,'cramers_v_interpretation','')})<br>{res.interpretation}")
        self.Information.done(f"χ² p = {getattr(chi,'p_value',float('nan')):.4f}")
        self._render_matrix()
        self.Outputs.table.send(self._current_matrix_table())
        self.Outputs.statistics.send(self._stats_table(res))

    def _current_matrix(self) -> Optional[pd.DataFrame]:
        if self._result is None:
            return None
        return getattr(self._result, self.display, None)

    def _current_matrix_table(self) -> Optional[Table]:
        m = self._current_matrix()
        if m is None:
            return None
        return _matrix_to_table(m, index_name=self._result.row_var or "row")

    def _render_matrix(self):
        m = self._current_matrix()
        if m is None or len(m) == 0:
            self.matrix_table.setRowCount(0)
            self.matrix_table.setColumnCount(0)
            return
        cols = list(m.columns)
        self.matrix_table.setColumnCount(len(cols))
        self.matrix_table.setRowCount(len(m))
        self.matrix_table.setHorizontalHeaderLabels([str(c) for c in cols])
        self.matrix_table.setVerticalHeaderLabels([str(i) for i in m.index])
        for r in range(len(m)):
            for c in range(len(cols)):
                v = m.iloc[r, c]
                txt = f"{v:,.2f}" if isinstance(v, (float, np.floating)) else str(v)
                self.matrix_table.setItem(r, c, QTableWidgetItem(txt))
        self.matrix_table.resizeColumnsToContents()

    @staticmethod
    def _stats_table(res) -> Optional[Table]:
        chi, es = res.chi_squared, res.effect_size
        rows = [
            ("Rows", res.row_var),
            ("Columns", res.col_var),
            ("N", str(res.n_total)),
            ("Chi-squared", f"{getattr(chi,'statistic',float('nan')):.4f}"),
            ("df", str(getattr(chi, "df", ""))),
            ("p-value", f"{getattr(chi,'p_value',float('nan')):.5f}"),
            ("Significant", "yes" if getattr(chi, "is_significant", False) else "no"),
            ("Min expected", f"{getattr(chi,'min_expected',float('nan')):.2f}"),
            ("Cramer's V", f"{getattr(es,'cramers_v',float('nan')):.4f}"),
            ("Cramer's V level", getattr(es, "cramers_v_interpretation", "")),
            ("Phi", f"{getattr(es,'phi',float('nan')):.4f}"),
            ("Contingency coef.", f"{getattr(es,'contingency_coef',float('nan')):.4f}"),
        ]
        if res.fisher is not None:
            rows.append(("Fisher p-value",
                         f"{getattr(res.fisher,'p_value',float('nan')):.5f}"))
        df = pd.DataFrame(rows, columns=["Statistic", "Value"])
        return _df_to_table(df)


if __name__ == "__main__":
    WidgetPreview(OWCrosstabs).run()
