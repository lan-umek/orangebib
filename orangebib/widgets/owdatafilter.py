# -*- coding: utf-8 -*-
"""
Filter widget
=============
Filter bibliographic records with simple OR compound conditions:

* numeric criteria (>, >=, <, <=, ==, !=, between),
* text / keyword criteria (contains, not contains, equals, regex, in list),
* combined with AND / OR.

Plus a Bradford's-law filter that keeps documents published in the *core* zone
of the most productive sources.
"""

import re
import logging
from typing import Optional

import pandas as pd

from AnyQt.QtWidgets import (
    QLabel, QComboBox, QPushButton, QLineEdit, QHBoxLayout, QVBoxLayout,
    QTableWidget, QGridLayout, QSpinBox, QTabWidget, QWidget, QCheckBox,
)

from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

NUM_OPS = [">", ">=", "<", "<=", "==", "!=", "between"]
TXT_OPS = ["contains", "not contains", "equals", "regex", "in list (comma)"]
_SEPS = ["||", "|", "; ", ";", ", "]


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas):
        try:
            col = table.get_column(var)
        except Exception:  # noqa: BLE001
            continue
        if getattr(var, "is_discrete", False):
            vals = var.values
            data[var.name] = [vals[int(v)] if (v == v and 0 <= int(v) < len(vals)) else ""
                              for v in col]
        else:
            data[var.name] = col
    return pd.DataFrame(data)


class OWDataFilter(OWWidget):
    """Simple / compound record filter + Bradford core filter."""

    name = "Filter"
    description = "Filter records with simple/compound numeric, text, regex and Bradford criteria"
    icon = "icons/data_filter.svg"
    priority = 60
    keywords = ["filter", "subset", "query", "regex", "bradford", "select",
                "criteria"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        matched = Output("Matching Data", Table, default=True)
        rejected = Output("Non-matching Data", Table)

    combine_mode = settings.Setting(0)     # 0 AND, 1 OR
    rules = settings.Setting([])           # list of (col, op, value)
    use_bradford = settings.Setting(False)
    bradford_source = settings.Setting("")
    bradford_zones = settings.Setting(3)

    want_main_area = False
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")

    class Information(OWWidget.Information):
        done = Msg("{} of {} records match")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None

        tabs = QTabWidget()
        # ---- rules tab ----
        rule_tab = QWidget(); rl = QVBoxLayout(rule_tab)
        gui.comboBox(rule_tab, self, "combine_mode", label="Combine rules with:",
                     orientation="horizontal", items=["AND (all)", "OR (any)"],
                     sendSelectedValue=False)
        self.rule_table = QTableWidget(0, 3)
        self.rule_table.setHorizontalHeaderLabels(["Column", "Operator", "Value"])
        rl.addWidget(self.rule_table)
        brow = QHBoxLayout()
        addb = QPushButton("+ rule"); addb.clicked.connect(self._add_rule)
        delb = QPushButton("− rule"); delb.clicked.connect(self._del_rule)
        brow.addWidget(addb); brow.addWidget(delb)
        rl.addLayout(brow)
        tabs.addTab(rule_tab, "Rules")

        # ---- bradford tab ----
        bt = QWidget(); bl = QGridLayout(bt)
        self.bradford_cb = QCheckBox(
            "Keep only documents from the Bradford core zone")
        self.bradford_cb.setChecked(self.use_bradford)
        self.bradford_cb.toggled.connect(
            lambda c: setattr(self, "use_bradford", c))
        bl.addWidget(self.bradford_cb, 0, 0, 1, 2)
        bl.addWidget(QLabel("Source column:"), 1, 0)
        self.src_combo = QComboBox()
        self.src_combo.currentTextChanged.connect(lambda t: setattr(self, "bradford_source", t))
        bl.addWidget(self.src_combo, 1, 1)
        bl.addWidget(QLabel("Number of zones:"), 2, 0)
        zspin = QSpinBox(); zspin.setRange(2, 10); zspin.setValue(self.bradford_zones)
        zspin.valueChanged.connect(lambda v: setattr(self, "bradford_zones", v))
        bl.addWidget(zspin, 2, 1)
        bl.addWidget(QLabel("<i>Core = most productive sources whose papers sum to ~1/zones of the total.</i>"), 3, 0, 1, 2)
        tabs.addTab(bt, "Bradford")
        self.controlArea.layout().addWidget(tabs)

        self.apply_btn = QPushButton("Apply filter"); self.apply_btn.setMinimumHeight(32)
        self.apply_btn.clicked.connect(self._apply)
        self.controlArea.layout().addWidget(self.apply_btn)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

    # ---------------------------------------------------------------- rules UI
    def _columns(self):
        return list(self._df.columns) if self._df is not None else []

    def _add_rule(self, col=None, op=None, val=None):
        r = self.rule_table.rowCount()
        self.rule_table.insertRow(r)
        cc = QComboBox(); cc.addItems(self._columns())
        if col and col in self._columns():
            cc.setCurrentText(col)
        oc = QComboBox(); oc.addItems(NUM_OPS + TXT_OPS)
        if op:
            oc.setCurrentText(op)
        ve = QLineEdit(val or "")
        self.rule_table.setCellWidget(r, 0, cc)
        self.rule_table.setCellWidget(r, 1, oc)
        self.rule_table.setCellWidget(r, 2, ve)

    def _del_rule(self):
        r = self.rule_table.currentRow()
        if r < 0:
            r = self.rule_table.rowCount() - 1
        if r >= 0:
            self.rule_table.removeRow(r)

    def _collect_rules(self):
        out = []
        for r in range(self.rule_table.rowCount()):
            col = self.rule_table.cellWidget(r, 0).currentText()
            op = self.rule_table.cellWidget(r, 1).currentText()
            val = self.rule_table.cellWidget(r, 2).text()
            if col and op:
                out.append((col, op, val))
        return out

    # ---------------------------------------------------------------- input
    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        cols = self._columns()
        self.src_combo.blockSignals(True)
        self.src_combo.clear()
        src_cands = [c for c in cols if any(
            k in str(c).lower() for k in ("source", "journal", "publication name", "so"))]
        self.src_combo.addItems(src_cands or cols)
        self.src_combo.blockSignals(False)
        # restore saved rules once columns are known
        self.rule_table.setRowCount(0)
        for (c, o, v) in (self.rules or []):
            self._add_rule(c, o, v)
        if not self.rules:
            self._add_rule()
        if data is None:
            self.Error.no_data()
        else:
            self._apply()

    # ---------------------------------------------------------------- apply
    def _eval_rule(self, df, col, op, val):
        if col not in df.columns:
            return pd.Series(True, index=df.index)
        s = df[col]
        if op in NUM_OPS:
            num = pd.to_numeric(s, errors="coerce")
            try:
                if op == "between":
                    parts = [p.strip() for p in re.split(r"[,;\-]", val) if p.strip()]
                    lo, hi = float(parts[0]), float(parts[1])
                    return num.between(lo, hi)
                t = float(val)
            except Exception:  # noqa: BLE001
                return pd.Series(True, index=df.index)
            return {">": num > t, ">=": num >= t, "<": num < t, "<=": num <= t,
                    "==": num == t, "!=": num != t}[op]
        sx = s.astype(str)
        low = sx.str.lower()
        v = val.lower()
        if op == "contains":
            return low.str.contains(re.escape(v), na=False)
        if op == "not contains":
            return ~low.str.contains(re.escape(v), na=False)
        if op == "equals":
            return low == v
        if op == "regex":
            try:
                return sx.str.contains(val, case=False, na=False, regex=True)
            except re.error:
                return pd.Series(True, index=df.index)
        if op == "in list (comma)":
            wanted = {w.strip().lower() for w in val.split(",") if w.strip()}
            return low.apply(lambda x: any(w in x for w in wanted)) if wanted \
                else pd.Series(True, index=df.index)
        return pd.Series(True, index=df.index)

    def _bradford_core_mask(self, df):
        col = self.bradford_source
        if not col or col not in df.columns:
            return pd.Series(True, index=df.index)
        counts = df[col].astype(str).value_counts()
        total = int(counts.sum())
        if total == 0:
            return pd.Series(True, index=df.index)
        target = total / max(2, self.bradford_zones)
        core_sources, cum = [], 0
        for src, c in counts.items():
            core_sources.append(src)
            cum += c
            if cum >= target:
                break
        return df[col].astype(str).isin(set(core_sources))

    def _apply(self):
        self.Error.clear(); self.Information.clear()
        if self._df is None or self._df.empty:
            self.Error.no_data()
            self.Outputs.matched.send(None); self.Outputs.rejected.send(None)
            return
        df = self._df
        rules = self._collect_rules()
        self.rules = rules
        if rules:
            masks = [self._eval_rule(df, c, o, v) for (c, o, v) in rules]
            mask = masks[0]
            for m in masks[1:]:
                mask = (mask & m) if self.combine_mode == 0 else (mask | m)
        else:
            mask = pd.Series(True, index=df.index)
        if self.use_bradford:
            mask = mask & self._bradford_core_mask(df)
        idx_match = [i for i, m in enumerate(mask.tolist()) if m]
        idx_rej = [i for i, m in enumerate(mask.tolist()) if not m]
        self.Outputs.matched.send(self._data[idx_match] if idx_match else None)
        self.Outputs.rejected.send(self._data[idx_rej] if idx_rej else None)
        self.status_label.setText(f"{len(idx_match)} of {len(df)} records match.")
        self.Information.done(len(idx_match), len(df))


if __name__ == "__main__":
    WidgetPreview(OWDataFilter).run()
