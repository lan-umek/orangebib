# -*- coding: utf-8 -*-
"""
Deduplicate & Merge Widget
==========================
Detect and remove duplicate records and merge exports from several databases
(e.g. Scopus + Web of Science + OpenAlex) using biblium's deduplication engine.
Matches on DOI and (optionally) normalised title; keeps the richest record.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import QLabel, QComboBox, QPushButton, QGridLayout

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from biblium.dedup import deduplicate, detect_duplicates
    HAS_DEDUP = True
except Exception:  # noqa: BLE001
    HAS_DEDUP = False
    deduplicate = detect_duplicates = None

_AUTO = "(auto)"
ACTIONS = ["Merge & deduplicate", "Flag duplicates only (no removal)"]


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in (list(table.domain.attributes) + list(table.domain.class_vars)
                + list(table.domain.metas)):
        try:
            col = table.get_column(var)
        except Exception:  # noqa: BLE001
            continue
        if getattr(var, "is_discrete", False):
            vals = var.values
            data[var.name] = [vals[int(v)] if (v == v and 0 <= int(v) < len(vals))
                              else "" for v in col]
        else:
            data[var.name] = col
    return pd.DataFrame(data)


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    attrs, metas, ac, mc = [], [], [], []
    for c in df.columns:
        if (pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])):
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


class OWDeduplicate(OWWidget):
    """Detect/remove duplicates and merge multi-database bibliographic exports."""

    name = "Deduplicate & Merge"
    description = ("Detect and remove duplicate records and merge exports from "
                   "several databases (DOI / title matching)")
    icon = "icons/deduplicate.svg"
    priority = 50
    keywords = ["deduplicate", "duplicates", "merge", "scopus", "wos",
                "web of science", "openalex", "doi"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Primary bibliographic table")
        data2 = Input("Data 2", Table, doc="Second source (optional)")
        data3 = Input("Data 3", Table, doc="Third source (optional)")

    class Outputs:
        deduplicated = Output("Deduplicated", Table, doc="Merged, unique records")
        flagged = Output("Flagged Duplicates", Table,
                         doc="All records with duplicate flags")
        merge_log = Output("Merge Log", Table, doc="Which records were merged")

    doi_col = settings.Setting(_AUTO)
    title_col = settings.Setting(_AUTO)
    use_title_matching = settings.Setting(True)
    action = settings.Setting(0)
    autorun = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    class Error(OWWidget.Error):
        no_data = Msg("Connect at least one table")
        no_dedup = Msg("biblium deduplication module not available")
        failed = Msg("Deduplication failed: {}")

    class Information(OWWidget.Information):
        done = Msg("{}")

    def __init__(self):
        super().__init__()
        self._dfs = {1: None, 2: None, 3: None}

        if not HAS_DEDUP:
            self.Error.no_dedup()

        box = gui.widgetBox(self.controlArea, "Matching")
        g = QGridLayout()
        g.addWidget(QLabel("DOI column:"), 0, 0)
        self.doi_combo = QComboBox()
        self.doi_combo.currentTextChanged.connect(lambda t: self._set("doi_col", t))
        g.addWidget(self.doi_combo, 0, 1)
        g.addWidget(QLabel("Title column:"), 1, 0)
        self.title_combo = QComboBox()
        self.title_combo.currentTextChanged.connect(lambda t: self._set("title_col", t))
        g.addWidget(self.title_combo, 1, 1)
        box.layout().addLayout(g)
        gui.checkBox(box, self, "use_title_matching",
                     "Also match by normalised title", callback=self._maybe_run)
        gui.comboBox(box, self, "action", items=ACTIONS, label="Action:",
                     orientation="horizontal", sendSelectedValue=False,
                     callback=self._maybe_run)

        gui.checkBox(self.controlArea, self, "autorun", "Run automatically")
        gui.button(self.controlArea, self, "Run", callback=self._run)

        self.status = QLabel("Connect 1–3 tables.")
        self.status.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status)
        self.controlArea.layout().addStretch(1)

    # ------------------------------------------------------------- inputs
    @Inputs.data
    def set_data(self, t):
        self._dfs[1] = _table_to_df(t) if t is not None else None

    @Inputs.data2
    def set_data2(self, t):
        self._dfs[2] = _table_to_df(t) if t is not None else None

    @Inputs.data3
    def set_data3(self, t):
        self._dfs[3] = _table_to_df(t) if t is not None else None

    def handleNewSignals(self):
        self._refresh_columns()
        self._maybe_run()

    # ------------------------------------------------------------- helpers
    def _set(self, attr, t):
        if t:
            setattr(self, attr, t)
            self._maybe_run()

    def _active_dfs(self):
        return [(i, d) for i, d in self._dfs.items()
                if d is not None and not d.empty]

    def _all_columns(self):
        cols = []
        for _i, d in self._active_dfs():
            for c in d.columns:
                if c not in cols:
                    cols.append(c)
        return cols

    def _refresh_columns(self):
        cols = self._all_columns()

        def fill(combo, current, prefer):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([_AUTO] + cols)
            pick = current if current in cols else _AUTO
            if pick == _AUTO:
                for p in prefer:
                    if p in cols:
                        pick = p
                        break
            combo.setCurrentText(pick)
            combo.blockSignals(False)
            return pick

        self.doi_col = fill(self.doi_combo, self.doi_col, ["DOI", "doi", "DI"])
        self.title_col = fill(self.title_combo, self.title_col,
                              ["Title", "title", "TI"])

    def _resolve(self, value, prefer, cols):
        if value and value != _AUTO and value in cols:
            return value
        for p in prefer:
            if p in cols:
                return p
        return prefer[0]

    def _maybe_run(self):
        if self.autorun:
            self._run()

    # ------------------------------------------------------------- run
    def _run(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_DEDUP:
            self.Error.no_dedup(); return
        active = self._active_dfs()
        if not active:
            self.Error.no_data()
            self._send(None, None, None)
            self.status.setText("Connect 1–3 tables.")
            return
        cols = self._all_columns()
        doi = self._resolve(self.doi_col, ["DOI", "doi", "DI"], cols)
        title = self._resolve(self.title_col, ["Title", "title", "TI"], cols)
        names = [f"source {i}" for i, _d in active]
        dfs = [d for _i, d in active]
        try:
            if self.action == 1:
                # flag only: detect duplicates on the concatenation
                base = (pd.concat(dfs, ignore_index=True)
                        if len(dfs) > 1 else dfs[0].copy())
                flagged = detect_duplicates(
                    base, doi_col=doi, title_col=title,
                    use_title_matching=self.use_title_matching)
                ndup = int(flagged.get("_is_duplicate", pd.Series(dtype=bool)).sum()) \
                    if "_is_duplicate" in flagged.columns else 0
                self._send(_df_to_table(base), _df_to_table(flagged), None)
                self.status.setText(
                    f"Flagged {ndup} duplicate record(s) across {len(base)} rows "
                    f"(matching on '{doi}' / '{title}').")
            else:
                res = deduplicate(
                    *dfs, source_names=names, doi_col=doi, title_col=title,
                    use_title_matching=self.use_title_matching, verbose=False)
                base = (pd.concat(dfs, ignore_index=True)
                        if len(dfs) > 1 else dfs[0].copy())
                try:
                    flagged = detect_duplicates(
                        base, doi_col=doi, title_col=title,
                        use_title_matching=self.use_title_matching)
                except Exception:  # noqa: BLE001
                    flagged = None
                try:
                    log_df = res.get_merge_log_df()
                except Exception:  # noqa: BLE001
                    log_df = None
                self._send(_df_to_table(res.df),
                           _df_to_table(flagged) if flagged is not None else None,
                           _df_to_table(log_df) if log_df is not None else None)
                self.status.setText(
                    f"In: {res.total_input}  →  unique: {res.total_output}.  "
                    f"Duplicates removed: {res.duplicates_found} "
                    f"(DOI: {res.duplicates_by_doi}, title: {res.duplicates_by_title}).")
                self.Information.done(
                    f"{res.total_output} unique of {res.total_input}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("dedup failed")
            self.Error.failed(str(exc))
            self._send(None, None, None)

    def _send(self, dedup, flagged, log):
        self.Outputs.deduplicated.send(dedup)
        self.Outputs.flagged.send(flagged)
        self.Outputs.merge_log.send(log)


if __name__ == "__main__":
    WidgetPreview(OWDeduplicate).run()
