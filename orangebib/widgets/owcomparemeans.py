# -*- coding: utf-8 -*-
"""
Compare Means Widget
====================
Compare a numeric variable across groups defined by a categorical variable.

Wraps :func:`biblium.compare_means.compare_means`, which runs descriptive
statistics, normality and homogeneity checks, the appropriate omnibus test
(t-test / ANOVA or their non-parametric counterparts) and post-hoc pairwise
comparisons. Results are emitted as three tables.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QCheckBox, QLabel, QComboBox, QPushButton, QDoubleSpinBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem,
)
from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.compare_means import (
        compare_means, get_numeric_columns, get_categorical_columns,
    )
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    compare_means = None
    get_numeric_columns = None
    get_categorical_columns = None

logger = logging.getLogger(__name__)


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


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    attrs, metas, acols, mcols = [], [], [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            attrs.append(ContinuousVariable(str(c))); acols.append(c)
        else:
            metas.append(StringVariable(str(c))); mcols.append(c)
    domain = Domain(attrs, metas=metas)
    n = len(df)
    X = np.empty((n, len(attrs)), dtype=float)
    for i, c in enumerate(acols):
        X[:, i] = pd.to_numeric(df[c], errors="coerce").values
    M = np.empty((n, len(metas)), dtype=object)
    for i, c in enumerate(mcols):
        M[:, i] = df[c].astype(object).where(df[c].notna(), "").values
    return Table.from_numpy(domain, X, metas=M)


class OWCompareMeans(OWWidget):
    """Compare means of a numeric variable across groups."""

    name = "Compare Means"
    description = "Compare a numeric variable across groups (t-test/ANOVA + post-hoc)"
    icon = "icons/compare_means.svg"
    priority = 660
    keywords = ["compare", "means", "anova", "t-test", "kruskal", "post-hoc",
                "statistics", "significance"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")

    class Outputs:
        descriptives = Output("Descriptives", Table, doc="Per-group descriptive statistics")
        tests = Output("Tests", Table, doc="Omnibus test results")
        post_hoc = Output("Post-hoc", Table, doc="Pairwise post-hoc comparisons")

    dependent_var = settings.Setting("")
    grouping_var = settings.Setting("")
    alpha = settings.Setting(0.05)
    auto_apply = settings.Setting(True)
    use_multi_binary = settings.Setting(False)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium is required. Install biblium>=2.16.")
        compute_error = Msg("Computation error: {}")

    class Warning(OWWidget.Warning):
        no_numeric = Msg("No numeric columns found for the dependent variable")
        no_groups = Msg("No suitable categorical columns found for grouping")

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
        grid.addWidget(QLabel("Dependent (numeric):"), 0, 0)
        self.dep_combo = QComboBox()
        self.dep_combo.currentTextChanged.connect(self._on_dep_changed)
        grid.addWidget(self.dep_combo, 0, 1)

        grid.addWidget(QLabel("Group by:"), 1, 0)
        self.group_combo = QComboBox()
        self.group_combo.currentTextChanged.connect(self._on_group_changed)
        grid.addWidget(self.group_combo, 1, 1)

        grid.addWidget(QLabel("Alpha:"), 2, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.001, 0.5)
        self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setDecimals(3)
        self.alpha_spin.setValue(self.alpha)
        self.alpha_spin.valueChanged.connect(lambda v: setattr(self, "alpha", v))
        grid.addWidget(self.alpha_spin, 2, 1)
        box.layout().addLayout(grid)

        mbox = gui.widgetBox(self.controlArea, "Or: multiple binary variables")
        self.multi_cb = QCheckBox("Compare across several binary variables")
        self.multi_cb.setChecked(self.use_multi_binary)
        self.multi_cb.toggled.connect(self._on_multi_toggled)
        mbox.layout().addWidget(self.multi_cb)
        self.multi_list = QListWidget()
        self.multi_list.setMaximumHeight(150)
        mbox.layout().addWidget(self.multi_list)

        self.run_btn = QPushButton("Compare")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.auto_cb = QCheckBox("Apply automatically")
        self.auto_cb.setChecked(self.auto_apply)
        self.auto_cb.toggled.connect(lambda c: setattr(self, "auto_apply", c))
        self.controlArea.layout().addWidget(self.auto_cb)

    def _setup_main_area(self):
        sbox = gui.widgetBox(self.mainArea, "Summary")
        self.summary_label = QLabel("No data")
        self.summary_label.setWordWrap(True)
        sbox.layout().addWidget(self.summary_label)

        dbox = gui.widgetBox(self.mainArea, "Descriptives")
        self.desc_table = QTableWidget()
        self.desc_table.setMinimumHeight(260)
        dbox.layout().addWidget(self.desc_table)

    def _on_multi_toggled(self, c):
        self.use_multi_binary = c
        if self.auto_apply:
            self._compute()

    def _binary_columns(self):
        if self._df is None:
            return []
        out = []
        for col in self._df.columns:
            ser = self._df[col]
            vals = pd.Series(ser).dropna().unique()
            if len(vals) == 2 or str(col).startswith("Group: "):
                out.append(col)
        return out

    def _populate_multi_list(self):
        self.multi_list.clear()
        for col in self._binary_columns():
            it = QListWidgetItem(str(col))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.multi_list.addItem(it)

    def _checked_binaries(self):
        return [self.multi_list.item(i).text()
                for i in range(self.multi_list.count())
                if self.multi_list.item(i).checkState() == Qt.Checked]

    def _on_dep_changed(self, t):
        self.dependent_var = t
        if self.auto_apply:
            self._compute()

    def _on_group_changed(self, t):
        self.grouping_var = t
        if self.auto_apply:
            self._compute()

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self._inject_setup_groups()
        self._populate_combos()
        if data is None:
            self.Error.no_data()
            return
        if self.auto_apply:
            self._compute()

    def _inject_setup_groups(self):
        """If the input comes from Setup Groups (binary 'Group: X' columns),
        build a single categorical 'Groups' column so the user can compare
        means directly across the defined groups."""
        if self._df is None or self._df.empty:
            return
        gcols = [c for c in self._df.columns if str(c).startswith("Group: ")]
        if not gcols or "Groups" in self._df.columns:
            return
        names = [str(c)[len("Group: "):] for c in gcols]
        mat = self._df[gcols].fillna(0)
        try:
            mat = mat.astype(float) > 0.5
        except Exception:  # noqa: BLE001
            return

        def _label(row):
            present = [names[i] for i, v in enumerate(row) if bool(v)]
            if not present:
                return "None"
            if len(present) == 1:
                return present[0]
            return " + ".join(present)
        self._df["Groups"] = [
            _label(r) for r in mat.to_numpy()]
        # prefer the synthesized grouping unless the user already chose one
        if not self.grouping_var or self.grouping_var not in self._df.columns:
            self.grouping_var = "Groups"

    def _populate_combos(self):
        self.dep_combo.blockSignals(True)
        self.group_combo.blockSignals(True)
        self.dep_combo.clear()
        self.group_combo.clear()
        if self._df is not None and not self._df.empty and HAS_BIBLIUM:
            num = get_numeric_columns(self._df)
            cat = get_categorical_columns(self._df)
            if "Groups" in self._df.columns and "Groups" not in cat:
                cat = ["Groups"] + cat
            elif "Groups" in cat:
                cat = ["Groups"] + [c for c in cat if c != "Groups"]
            self.dep_combo.addItems(num)
            self.group_combo.addItems(cat)
            if not num:
                self.Warning.no_numeric()
            if not cat:
                self.Warning.no_groups()
            if self.dependent_var in num:
                self.dep_combo.setCurrentText(self.dependent_var)
            elif num:
                self.dependent_var = num[0]
            if self.grouping_var in cat:
                self.group_combo.setCurrentText(self.grouping_var)
            elif cat:
                self.grouping_var = cat[0]
        self.dep_combo.blockSignals(False)
        self.group_combo.blockSignals(False)
        self._populate_multi_list()

    def _compute(self):
        self.Error.clear()
        self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            return
        if self._df is None or self._df.empty:
            self.Error.no_data()
            return
        dep = self.dep_combo.currentText()
        if self.use_multi_binary and dep:
            self._compute_multi_binary(dep)
            return
        grp = self.group_combo.currentText()
        if not dep or not grp:
            return
        # Pre-clean: keep numeric dependent, drop empty groups and groups with
        # fewer than 2 observations so the test has enough data.
        if dep not in self._df.columns or grp not in self._df.columns:
            self.Error.compute_error("Selected columns not found"); self._send(None, None, None); return
        work = self._df[[dep, grp]].copy()
        work[dep] = pd.to_numeric(work[dep], errors="coerce")
        work = work.dropna(subset=[dep])
        work = work[work[grp].astype(str).str.strip().ne("")]
        work = work[~work[grp].astype(str).str.lower().isin(["none", "nan"])]
        sizes = work.groupby(grp).size()
        keep_groups = sizes[sizes >= 2].index
        work = work[work[grp].isin(keep_groups)]
        if work[grp].nunique() < 2 or len(work) < 4:
            self.Error.compute_error(
                "Insufficient data: need at least 2 groups with 2+ valid "
                f"'{dep}' values each. Check the dependent and grouping variables.")
            self._send(None, None, None)
            return
        try:
            res = compare_means(work, dep, grp, alpha=self.alpha, verbose=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("compare_means failed")
            self.Error.compute_error(str(exc))
            self._send(None, None, None)
            return
        self._result = res
        desc_df = self._descriptives_df(res)
        tests_df = self._tests_df(res)
        ph_df = self._post_hoc_df(res)
        self._fill_desc_table(desc_df)
        self.summary_label.setText(
            f"<b>{dep}</b> by <b>{grp}</b> — {res.n_groups} groups, "
            f"n={res.n_total}. Recommended: {res.recommended_test}.<br>"
            f"{res.interpretation}")
        self.Information.done(f"Recommended test: {res.recommended_test}")
        self._send(_df_to_table(desc_df), _df_to_table(tests_df), _df_to_table(ph_df))

    def _compute_multi_binary(self, dep):
        """Compare the dependent variable across several binary variables: for
        each selected binary column, show group means (0/No vs 1/Yes), n and a
        two-sample test p-value."""
        bins = self._checked_binaries()
        if not bins:
            self.Error.compute_error("Tick at least one binary variable.")
            self._send(None, None, None); return
        if dep not in self._df.columns:
            self.Error.compute_error("Dependent column not found.")
            self._send(None, None, None); return
        y_all = pd.to_numeric(self._df[dep], errors="coerce")
        try:
            from scipy import stats as _st
            has_sp = True
        except Exception:  # noqa: BLE001
            has_sp = False
        rows = []
        for col in bins:
            ser = self._df[col]
            # map to boolean: 1/yes/true/positive -> True
            def _truthy(v):
                sv = str(v).strip().lower()
                return sv in ("1", "1.0", "yes", "true", "y", "present")
            mask = ser.apply(_truthy)
            # if not obviously truthy, fall back to the larger-valued category
            if mask.sum() == 0:
                vals = pd.Series(ser).dropna().unique()
                if len(vals) == 2:
                    hi = max(vals, key=lambda x: str(x))
                    mask = ser == hi
            g1 = y_all[mask].dropna()
            g0 = y_all[~mask].dropna()
            if len(g1) < 2 or len(g0) < 2:
                continue
            p = np.nan
            if has_sp:
                try:
                    p = float(_st.ttest_ind(g1, g0, equal_var=False).pvalue)
                except Exception:  # noqa: BLE001
                    p = np.nan
            rows.append({
                "Variable": str(col),
                "n (1)": int(len(g1)), "Mean (1)": round(float(g1.mean()), 3),
                "n (0)": int(len(g0)), "Mean (0)": round(float(g0.mean()), 3),
                "Difference": round(float(g1.mean() - g0.mean()), 3),
                "p-value": round(p, 4) if p == p else None,
            })
        if not rows:
            self.Error.compute_error("No binary variable had 2+ values in each group.")
            self._send(None, None, None); return
        desc_df = pd.DataFrame(rows).sort_values(
            "p-value", na_position="last").reset_index(drop=True)
        self._fill_desc_table(desc_df)
        self.summary_label.setText(
            f"<b>{dep}</b> compared across <b>{len(rows)}</b> binary variables "
            f"(sorted by p-value).")
        self.Information.done("Multiple binary comparison")
        self._send(_df_to_table(desc_df), None, None)

    def _send(self, d, t, p):
        self.Outputs.descriptives.send(d)
        self.Outputs.tests.send(t)
        self.Outputs.post_hoc.send(p)

    @staticmethod
    def _descriptives_df(res) -> pd.DataFrame:
        rows = []
        groups = list(res.group_descriptives)
        if res.overall_descriptives is not None:
            groups = groups + [res.overall_descriptives]
        for g in groups:
            rows.append({
                "Group": g.group_name, "N": g.n, "Mean": g.mean, "SD": g.std,
                "SE": g.se, "Median": g.median, "Min": g.min_val, "Max": g.max_val,
                "Skewness": g.skewness, "Kurtosis": g.kurtosis,
                "CI lower": g.ci_lower, "CI upper": g.ci_upper,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _tests_df(res) -> pd.DataFrame:
        rows = []
        for t in (res.parametric_test, res.nonparametric_test):
            if t is None:
                continue
            rows.append({
                "Test": t.test_name, "Statistic": t.statistic, "df": t.df,
                "p-value": t.p_value, "Effect size": t.effect_size,
                "Effect": t.effect_size_name,
                "Significant": "yes" if t.is_significant else "no",
                "Notes": t.notes,
            })
        if res.homogeneity_test is not None:
            h = res.homogeneity_test
            rows.append({
                "Test": getattr(h, "test_name", "Homogeneity"),
                "Statistic": getattr(h, "statistic", float("nan")),
                "df": float("nan"),
                "p-value": getattr(h, "p_value", float("nan")),
                "Effect size": float("nan"), "Effect": "",
                "Significant": "yes" if getattr(h, "is_significant", False) else "no",
                "Notes": "equal variances" if getattr(h, "equal_variance", True) else "unequal variances",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _post_hoc_df(res) -> pd.DataFrame:
        rows = []
        for p in res.post_hoc_results:
            rows.append({
                "Group 1": p.group1, "Group 2": p.group2,
                "Mean diff": p.mean_diff, "Statistic": p.statistic,
                "p-value": p.p_value, "p-adjusted": p.p_adjusted,
                "CI lower": p.ci_lower, "CI upper": p.ci_upper,
                "Significant": "yes" if p.is_significant else "no",
            })
        return pd.DataFrame(rows)

    def _fill_desc_table(self, df: pd.DataFrame):
        if df is None or df.empty:
            self.desc_table.setRowCount(0)
            self.desc_table.setColumnCount(0)
            return
        self.desc_table.setColumnCount(len(df.columns))
        self.desc_table.setRowCount(len(df))
        self.desc_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                v = df.iloc[r, c]
                txt = f"{v:,.3f}" if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else str(v)
                self.desc_table.setItem(r, c, QTableWidgetItem(txt))
        self.desc_table.resizeColumnsToContents()


if __name__ == "__main__":
    WidgetPreview(OWCompareMeans).run()
