# -*- coding: utf-8 -*-
"""
Logistic Regression Widget
=========================
Explanatory binary logistic regression (statsmodels): predict membership in a
group / binary target from numeric predictors, reporting coefficients,
standard errors, z, p-values, odds ratios with confidence intervals and the
direction of each effect. Complements Orange's predictive models with proper
inferential statistics.
"""

import os
import re
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QPushButton, QGridLayout, QListWidget, QListWidgetItem,
    QHBoxLayout, QCheckBox, QTableWidget, QTableWidgetItem, QSpinBox, QTabWidget,
    QAbstractItemView,
)

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:  # noqa: BLE001
    HAS_SM = False
    sm = None

try:
    from biblium.utilsbib_modules.firth import firth_logit
    HAS_FIRTH = True
except Exception:  # noqa: BLE001
    HAS_FIRTH = False
    firth_logit = None

logger = logging.getLogger(__name__)

_STOPWORD_CACHE = {}


def _load_stopword_sets():
    """Load (general_stopwords, specific_category_words) from the bundled
    orangebib/data/stopwords.xlsx (sheets 'general' and 'specific'). Cached."""
    if "sets" in _STOPWORD_CACHE:
        return _STOPWORD_CACHE["sets"]
    general, specific = set(), set()
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "data", "stopwords.xlsx")
        if os.path.exists(path):
            xl = pd.ExcelFile(path)
            if "general" in xl.sheet_names:
                g = pd.read_excel(xl, sheet_name="general")
                col = "english" if "english" in g.columns else g.columns[0]
                general = {str(w).strip().lower() for w in g[col].dropna()}
            if "specific" in xl.sheet_names:
                sp = pd.read_excel(xl, sheet_name="specific")
                wc = "Word" if "Word" in sp.columns else sp.columns[-1]
                specific = {str(w).strip().lower() for w in sp[wc].dropna()}
    except Exception:  # noqa: BLE001
        pass
    _STOPWORD_CACHE["sets"] = (general, specific)
    return general, specific


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


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    attrs, metas, ac, mc = [], [], [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            attrs.append(ContinuousVariable(str(c))); ac.append(c)
        else:
            metas.append(StringVariable(str(c))); mc.append(c)
    domain = Domain(attrs, metas=metas)
    n = len(df)
    X = np.empty((n, len(attrs)))
    for i, c in enumerate(ac):
        X[:, i] = pd.to_numeric(df[c], errors="coerce").values
    M = np.empty((n, len(metas)), dtype=object)
    for i, c in enumerate(mc):
        M[:, i] = [("" if (v is None or (isinstance(v, float) and v != v)) else str(v))
                   for v in df[c]]
    return Table.from_numpy(domain, X, metas=M)


class OWLogisticRegression(OWWidget):
    """Explanatory binary logistic regression with p-values and odds ratios."""

    name = "Logistic Regression"
    description = "Binary logistic regression (coefficients, p-values, odds ratios)"
    icon = "icons/logistic_regression.svg"
    priority = 695
    keywords = ["logistic", "regression", "logit", "odds ratio", "p-value",
                "inference", "group membership", "classification"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Data with a binary target and numeric predictors")

    class Outputs:
        coefficients = Output("Coefficients", Table, doc="Model coefficients & stats")
        predictions = Output("Predictions", Table, doc="Data with predicted probability/class")

    target_col = settings.Setting("")
    positive_class = settings.Setting("")
    selected_predictors = settings.Setting([])
    add_constant = settings.Setting(True)
    standardize = settings.Setting(False)
    multi_targets = settings.Setting(False)
    extra_targets = settings.Setting([])
    summary_value = settings.Setting(2)  # 0 coef, 1 p, 2 both
    use_firth = settings.Setting(False)
    kw_top_n = settings.Setting(20)
    kw_min_occ = settings.Setting(3)
    kw_remove_stopwords = settings.Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_statsmodels = Msg("statsmodels is required for this widget.")
        bad_target = Msg("Target must have exactly two classes")
        no_predictors = Msg("Select at least one predictor")
        fit_error = Msg("Model fitting failed: {}")

    class Information(OWWidget.Information):
        no_terms = Msg("No terms met the minimum-occurrence threshold")
        done = Msg("Fitted on {} rows (pseudo-R² = {:.3f})")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._num_cols: List[str] = []
        self._built_kw_cols: List[str] = []

        box = gui.widgetBox(self.controlArea, "Target")
        grid = QGridLayout()
        grid.addWidget(QLabel("Group / target:"), 0, 0)
        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        grid.addWidget(self.target_combo, 0, 1)
        grid.addWidget(QLabel("Positive class:"), 1, 0)
        self.pos_combo = QComboBox()
        self.pos_combo.currentTextChanged.connect(lambda t: setattr(self, "positive_class", t))
        grid.addWidget(self.pos_combo, 1, 1)
        box.layout().addLayout(grid)

        mtbox = gui.widgetBox(self.controlArea, "Additional targets (one tab each)")
        mtbox.layout().addWidget(QLabel(
            "Click extra binary targets to select several (each gets its own tab, "
            "plus a Summary tab)."))
        self.target_list = QListWidget()
        self.target_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.target_list.setMaximumHeight(130)
        self.target_list.itemSelectionChanged.connect(self._on_targets_selection)
        mtbox.layout().addWidget(self.target_list)

        pbox = gui.widgetBox(self.controlArea, "Predictors (numeric)")
        self.pred_list = QListWidget()
        self.pred_list.setMaximumHeight(180)
        pbox.layout().addWidget(self.pred_list)
        brow = QHBoxLayout()
        a = QPushButton("All"); a.clicked.connect(lambda: self._set_all(True))
        n = QPushButton("None"); n.clicked.connect(lambda: self._set_all(False))
        brow.addWidget(a); brow.addWidget(n)
        pbox.layout().addLayout(brow)

        kbox = gui.widgetBox(self.controlArea, "Build predictors from text/keywords")
        kg = QGridLayout()
        kg.addWidget(QLabel("Column:"), 0, 0)
        self.kw_combo = QComboBox()
        kg.addWidget(self.kw_combo, 0, 1)
        kbox.layout().addLayout(kg)
        gui.spin(kbox, self, "kw_top_n", 2, 200, label="Top N terms:")
        gui.spin(kbox, self, "kw_min_occ", 1, 100, label="Min occurrences:")
        gui.checkBox(kbox, self, "kw_remove_stopwords",
                     "Remove stopwords (general + my list + specific categories)")
        kbtn = QPushButton("Build binary predictors")
        kbtn.clicked.connect(self._build_kw_predictors)
        kbox.layout().addWidget(kbtn)

        obox = gui.widgetBox(self.controlArea, "Options")
        self.const_cb = QCheckBox("Include intercept")
        self.const_cb.setChecked(self.add_constant)
        self.const_cb.toggled.connect(lambda c: setattr(self, "add_constant", c))
        obox.layout().addWidget(self.const_cb)
        self.std_cb = QCheckBox("Standardize predictors (z-score)")
        self.std_cb.setChecked(self.standardize)
        self.std_cb.toggled.connect(lambda c: setattr(self, "standardize", c))
        obox.layout().addWidget(self.std_cb)
        gui.comboBox(obox, self, "summary_value", label="Summary cells:",
                     orientation="horizontal",
                     items=["Coefficient", "P-value", "Coefficient (p-value)"],
                     sendSelectedValue=False)
        self.firth_cb = QCheckBox("Firth penalized (handles separation)")
        self.firth_cb.setChecked(self.use_firth)
        self.firth_cb.setEnabled(HAS_FIRTH)
        self.firth_cb.toggled.connect(lambda c: setattr(self, "use_firth", c))
        obox.layout().addWidget(self.firth_cb)

        self.run_btn = QPushButton("Fit model")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._fit)
        self.controlArea.layout().addWidget(self.run_btn)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        sbox = gui.widgetBox(self.mainArea, "Model")
        self.summary_label = QLabel("No data"); self.summary_label.setWordWrap(True)
        sbox.layout().addWidget(self.summary_label)
        cbox = gui.widgetBox(self.mainArea, "Coefficients")
        self.result_tabs = QTabWidget()
        self.coef_table = QTableWidget()
        self.result_tabs.addTab(self.coef_table, "Model")
        cbox.layout().addWidget(self.result_tabs)

        if not HAS_SM:
            self.Error.no_statsmodels()
            self.run_btn.setEnabled(False)

    def _set_all(self, state):
        for i in range(self.pred_list.count()):
            self.pred_list.item(i).setCheckState(Qt.Checked if state else Qt.Unchecked)

    def _checked_predictors(self):
        return [self.pred_list.item(i).text() for i in range(self.pred_list.count())
                if self.pred_list.item(i).checkState() == Qt.Checked]

    def _on_target_changed(self, t):
        self.target_col = t
        self.pos_combo.blockSignals(True)
        self.pos_combo.clear()
        if self._df is not None and t in self._df.columns:
            vals = [str(v) for v in pd.Series(self._df[t]).dropna().unique()]
            self.pos_combo.addItems(vals)
            if self.positive_class in vals:
                self.pos_combo.setCurrentText(self.positive_class)
            elif vals:
                self.positive_class = vals[0]
        self.pos_combo.blockSignals(False)
        self._refresh_predictor_list()

    def _text_source_options(self):
        """Return an ordered {label: (kind, value)} of allowed text sources that
        are actually present. kind is "col" (value=column name) or "combine"
        (value=list of column names joined per row)."""
        cols = list(self._df.columns)

        def find(cands):
            for c in cands:
                if c in cols:
                    return c
            return None

        title = find(["Title", "TI", "Document Title", "title"])
        abstract = find(["Abstract", "AB", "Description", "abstract"])
        ak = find(["Author Keywords", "Author keywords", "DE"])
        ik = find(["Index Keywords", "Index keywords", "ID", "Keywords Plus"])
        allkw = find(["All Keywords", "Keywords", "keywords"])
        refs = find(["References", "Cited References", "CR", "references"])

        opts = {}
        if ak:
            opts["Author Keywords"] = ("col", ak)
        if ik:
            opts["Index Keywords"] = ("col", ik)
        if allkw:
            opts["All Keywords"] = ("col", allkw)
        elif ak and ik:
            opts["All Keywords"] = ("combine", [ak, ik])
        if title:
            opts["Title"] = ("col", title)
        if abstract:
            opts["Abstract"] = ("col", abstract)
        kw_for_combo = allkw or ak or ik
        if (title or abstract) and kw_for_combo:
            parts = [c for c in [title, abstract, kw_for_combo] if c]
            opts["Title + Abstract + Keywords"] = ("combine", parts)
        if refs:
            opts["References"] = ("col", refs)
        return opts

    def _resolve_text_series(self, label):
        src = getattr(self, "_text_sources", {}).get(label)
        if src is None:
            return None
        kind, val = src
        if kind == "col":
            return self._df[val].astype(object)
        # combine: join the parts per row
        def _join(row):
            vals = [str(row[c]) for c in val
                    if c in row and pd.notna(row[c]) and str(row[c]).strip()
                    and str(row[c]).lower() != "nan"]
            return "; ".join(vals)
        return self._df.apply(_join, axis=1)

    def _build_kw_predictors(self):
        """Create binary 0/1 predictor columns from the most frequent terms in
        the chosen keyword/text column."""
        self.Error.clear(); self.Information.clear()
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        col = self.kw_combo.currentText()
        series = self._resolve_text_series(col)
        if series is None:
            return
        seps = ["||", "|", "; ", ";", ", "]

        def tokens(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return []
            sx = str(val).strip()
            if not sx or sx.lower() == "nan":
                return []
            for sep in seps:
                if sep in sx:
                    return [t.strip().lower() for t in sx.split(sep) if t.strip()]
            # free text -> word tokens (>=3 chars)
            return [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}", sx)]

        stops = set()
        if self.kw_remove_stopwords:
            general, specific = _load_stopword_sets()
            stops = general | specific

        from collections import Counter
        cnt = Counter()
        per_row = []
        for v in series:
            tk = [t for t in dict.fromkeys(tokens(v)) if t not in stops]
            per_row.append(set(tk))
            cnt.update(tk)
        terms = [t for t, c in cnt.most_common() if c >= self.kw_min_occ][:self.kw_top_n]
        if not terms:
            self.Information.no_terms()
            return
        # drop previously built columns
        for c in self._built_kw_cols:
            if c in self._df.columns:
                self._df.drop(columns=[c], inplace=True)
        self._built_kw_cols = []
        for t in terms:
            name = f"kw: {t}"
            self._df[name] = [1 if t in row else 0 for row in per_row]
            self._built_kw_cols.append(name)
        # refresh numeric columns + predictor list, pre-checking the new ones
        self._num_cols = [c for c in self._df.columns
                          if pd.api.types.is_numeric_dtype(self._df[c])]
        self.selected_predictors = list(self._built_kw_cols)
        self._refresh_predictor_list()
        self.status_label.setText(
            f"Built {len(self._built_kw_cols)} keyword predictors from '{col}'.")

    def _refresh_predictor_list(self):
        self.pred_list.clear()
        for c in self._num_cols:
            if c == self.target_combo.currentText():
                continue
            it = QListWidgetItem(c)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            # default: predictors OFF (user opts in explicitly)
            checked = c in self.selected_predictors
            it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.pred_list.addItem(it)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_SM:
            self.Error.no_statsmodels()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        if data is None:
            self.Error.no_data()
            return
        self._built_kw_cols = []
        self._num_cols = [c for c in self._df.columns
                          if pd.api.types.is_numeric_dtype(self._df[c])]
        # candidate text sources: only the bibliographic text fields that make
        # sense here (keywords, title, abstract, their combination, references)
        self._text_sources = self._text_source_options()
        self.kw_combo.blockSignals(True)
        self.kw_combo.clear()
        self.kw_combo.addItems(list(self._text_sources.keys()))
        self.kw_combo.blockSignals(False)
        # target candidates: any column with exactly 2 distinct non-null values,
        # plus all columns (user may pick).
        cands = []
        for c in self._df.columns:
            nun = pd.Series(self._df[c]).dropna().nunique()
            if nun == 2:
                cands.append(c)
        cands += [c for c in self._df.columns if c not in cands]
        self.target_combo.blockSignals(True)
        self.target_combo.clear(); self.target_combo.addItems(cands)
        if self.target_col in cands:
            self.target_combo.setCurrentText(self.target_col)
        elif cands:
            self.target_col = cands[0]
        self.target_combo.blockSignals(False)
        # binary-target candidates for the multi-target list
        bin_cands = [c for c in self._df.columns
                     if pd.Series(self._df[c]).dropna().nunique() == 2]
        self.target_list.blockSignals(True)
        self.target_list.clear()
        for c in bin_cands:
            it = QListWidgetItem(str(c))
            self.target_list.addItem(it)
            if str(c) in self.extra_targets:
                it.setSelected(True)
        self.target_list.blockSignals(False)
        self._on_target_changed(self.target_combo.currentText())

    def _on_targets_selection(self):
        self.extra_targets = [it.text() for it in self.target_list.selectedItems()]

    def _checked_targets(self):
        return [it.text() for it in self.target_list.selectedItems()]

    def _fit_one(self, target, pos=None):
        """Fit a logit for one binary target. Returns (coef_df, summary, pred_df)
        or raises ValueError with a message."""
        preds = self._checked_predictors()
        if not preds:
            raise ValueError("Select at least one predictor")
        df = self._df[[target] + preds].copy()
        for c in preds:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        if df.empty:
            raise ValueError("no complete rows after dropping missing")
        ser = df[target].astype(str)
        classes = sorted(ser.unique())
        if len(classes) != 2:
            raise ValueError(f"target '{target}' must have exactly two classes")
        if pos is None or pos not in classes:
            pos = classes[-1]
        y = (ser == pos).astype(int).values
        X = df[preds].astype(float)
        if self.standardize:
            X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
        if self.use_firth and HAS_FIRTH:
            fr = firth_logit(X.values, y, add_intercept=self.add_constant,
                             feature_names=list(X.columns))
            coef_df = self._firth_coef_table(fr)
            pr2 = (1.0 - fr.log_likelihood / fr.null_log_likelihood
                   if fr.null_log_likelihood else float("nan"))
            summary = (f"Target = <b>{target}</b> (positive = '{pos}'), n = {len(df)}. "
                       f"Firth penalized; pseudo-R² = {pr2:.3f}"
                       + ("" if fr.converged else " (did not converge)") + ".")
            beta = np.asarray(fr.coef, dtype=float)
            Xmat = (np.column_stack([np.ones(len(X)), X.values])
                    if self.add_constant else X.values)
            lin = Xmat @ beta
            prob = 1.0 / (1.0 + np.exp(-lin))
            pred_df = df.copy()
            pred_df["pred_prob"] = prob
            pred_df["pred_class"] = (pred_df["pred_prob"] >= 0.5).astype(int)
            return coef_df, summary, pred_df
        Xd = sm.add_constant(X, has_constant="add") if self.add_constant else X.copy()
        model = sm.Logit(y, Xd)
        res = model.fit(disp=False, maxiter=200)
        coef_df = self._coef_table(res)
        summary = (f"Target = <b>{target}</b> (positive = '{pos}'), n = {len(df)}. "
                   f"Pseudo-R² = {res.prsquared:.3f}, LLR p = {res.llr_pvalue:.3g}.")
        pred_df = df.copy()
        pred_df["pred_prob"] = res.predict(Xd)
        pred_df["pred_class"] = (pred_df["pred_prob"] >= 0.5).astype(int)
        return coef_df, summary, pred_df

    def _fit(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_SM:
            self.Error.no_statsmodels(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        self.selected_predictors = self._checked_predictors()
        if not self.selected_predictors:
            self.Error.no_predictors(); return

        # which targets? always the primary (combo) target plus any targets
        # checked in the list below (deduplicated, combo first).
        targets = []
        combo_t = self.target_combo.currentText()
        if combo_t:
            targets.append(combo_t)
        for t in self._checked_targets():
            if t and t not in targets:
                targets.append(t)
        if not targets:
            return
        # rebuild result tabs
        self.result_tabs.clear()
        all_coef = []
        summaries = []
        first_pred = None
        for tgt in targets:
            if not tgt:
                continue
            try:
                pos = self.pos_combo.currentText() if len(targets) == 1 else None
                coef_df, summ, pred_df = self._fit_one(tgt, pos)
            except Exception as exc:  # noqa: BLE001
                logger.exception("logit fit failed for %s", tgt)
                summaries.append(f"<b>{tgt}</b>: failed — {exc}")
                continue
            tbl = QTableWidget()
            self._fill_table(tbl, coef_df)
            self.result_tabs.addTab(tbl, str(tgt)[:20])
            cd = coef_df.copy(); cd.insert(0, "Target", tgt)
            all_coef.append(cd)
            summaries.append(summ)
            if first_pred is None:
                first_pred = pred_df
        # extra Summary tab: variables (rows) x groups/targets (cols),
        # cells = coefficient and/or p-value
        if len(all_coef) > 1:
            combined = pd.concat(all_coef, ignore_index=True)

            def _cell(r):
                c, pv = r["Coefficient"], r["P-value"]
                if self.summary_value == 0:
                    return f"{c}"
                if self.summary_value == 1:
                    return f"{pv}"
                return f"{c} (p={pv})"

            combined["_cell"] = combined.apply(_cell, axis=1)
            # keep variable order as first appearance; targets in fitted order
            var_order = list(dict.fromkeys(combined["Variable"]))
            tgt_order = list(dict.fromkeys(combined["Target"]))
            piv = combined.pivot_table(index="Variable", columns="Target",
                                       values="_cell", aggfunc="first")
            piv = piv.reindex(index=var_order, columns=tgt_order)
            piv = piv.reset_index().rename(columns={"Variable": "Variable \\ Group"})
            piv = piv.fillna("")
            stab = QTableWidget()
            self._fill_table(stab, piv)
            self.result_tabs.addTab(stab, "Summary")
        # keep the legacy single coef_table reference pointing at the first tab
        if self.result_tabs.count():
            w = self.result_tabs.widget(0)
            if isinstance(w, QTableWidget):
                self.coef_table = w
        self.summary_label.setText("<br>".join(summaries) if summaries else "No model fitted")
        self.status_label.setText("Done")
        if all_coef:
            self.Outputs.coefficients.send(_df_to_table(pd.concat(all_coef, ignore_index=True)))
        if first_pred is not None:
            self.Outputs.predictions.send(_df_to_table(first_pred))

    def _fill_table(self, table, df):
        table.clear()
        table.setColumnCount(len(df.columns)); table.setRowCount(len(df))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
        table.resizeColumnsToContents()

    @staticmethod
    def _firth_coef_table(fr) -> pd.DataFrame:
        rows = []
        names = list(fr.feature_names)
        for k, name in enumerate(names):
            coef = float(fr.coef[k]); p = float(fr.p_values_wald[k])
            lo, hi = float(fr.ci_low[k]), float(fr.ci_high[k])
            is_int = str(name).lower() in ("const", "intercept")
            direction = ("intercept" if is_int
                         else ("increases odds" if coef > 0 else "decreases odds"))
            rows.append({
                "Variable": name,
                "Coefficient": round(coef, 4),
                "Std.Error": round(float(fr.se[k]), 4),
                "z": round(float(fr.z_values[k]), 3),
                "P-value": round(p, 5),
                "Odds ratio": round(float(np.exp(coef)), 4),
                "OR CI low": round(float(np.exp(lo)), 4),
                "OR CI high": round(float(np.exp(hi)), 4),
                "Significant": "yes" if p < 0.05 else "no",
                "Direction": direction,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _coef_table(res) -> pd.DataFrame:
        params = res.params
        ci = res.conf_int()
        rows = []
        for name in params.index:
            coef = params[name]
            p = res.pvalues[name]
            lo, hi = ci.loc[name, 0], ci.loc[name, 1]
            direction = ("intercept" if name == "const"
                         else ("increases odds" if coef > 0 else "decreases odds"))
            rows.append({
                "Variable": name,
                "Coefficient": round(coef, 4),
                "Std.Error": round(res.bse[name], 4),
                "z": round(res.tvalues[name], 3),
                "P-value": round(p, 5),
                "Odds ratio": round(np.exp(coef), 4),
                "OR CI low": round(np.exp(lo), 4),
                "OR CI high": round(np.exp(hi), 4),
                "Significant": "yes" if p < 0.05 else "no",
                "Direction": direction,
            })
        return pd.DataFrame(rows)

    def _fill_coef_table(self, df):
        self.coef_table.clear()
        self.coef_table.setColumnCount(len(df.columns))
        self.coef_table.setRowCount(len(df))
        self.coef_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.coef_table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
        self.coef_table.resizeColumnsToContents()


if __name__ == "__main__":
    WidgetPreview(OWLogisticRegression).run()
