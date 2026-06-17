# -*- coding: utf-8 -*-
"""
SDG Drift Widget
===============
Conceptual drift of Sustainable Development Goals over time: how the vocabulary
of each SDG shifts across time windows. Wraps
`biblium.addons.sdg_drift.analyze_sdg_drift`.

Binary ``SDG N`` indicator columns are auto-detected; if absent they are derived
from a multi-valued SDG column (e.g. ``oa_sdgs``).
"""

import re
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QGridLayout,
                             QProgressBar)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from biblium.addons.sdg_drift import analyze_sdg_drift, get_sdg_columns
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    analyze_sdg_drift = None
    get_sdg_columns = None
    HAS_BIBLIUM = False

SDG_MULTI_CANDIDATES = ["oa_sdgs", "SDGs", "SDG", "sdgs", "sdg",
                        "Sustainable Development Goals"]
TEXT_CANDIDATES = ["Processed Abstract", "Processed Combined Text", "Abstract",
                   "Processed Text", "Combined Text", "Title"]
_SEPS = ["||", "|", "; ", ";", ", "]


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas):
        data[var.name] = table.get_column(var)
    return pd.DataFrame(data)


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    attrs, metas, X, M = [], [], [], []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.6:
            attrs.append(ContinuousVariable(str(c))); X.append(s.fillna(0).values)
        else:
            metas.append(StringVariable(str(c))); M.append(df[c].astype(str).values)
    n = len(df)
    Xarr = np.column_stack(X) if X else np.empty((n, 0))
    Marr = np.column_stack(M) if M else np.empty((n, 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), Xarr, metas=Marr)


def _split_multi(val) -> List[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    for sep in _SEPS:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def ensure_sdg_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = []
    if get_sdg_columns is not None:
        try:
            existing = get_sdg_columns(df)
        except Exception:  # noqa: BLE001
            existing = []
    if existing:
        return df
    multi = next((c for c in SDG_MULTI_CANDIDATES if c in df.columns), None)
    if multi is None:
        return df
    out = df.copy()
    rows_sdgs, found = [], set()
    for v in out[multi]:
        nums = set()
        for tok in _split_multi(v):
            m = re.search(r'(\d{1,2})', tok)
            if m:
                k = int(m.group(1))
                if 1 <= k <= 17:
                    nums.add(k)
        rows_sdgs.append(nums); found |= nums
    for k in sorted(found):
        out[f"SDG {k}"] = [1 if k in s else 0 for s in rows_sdgs]
    return out


class DriftWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, text_col, year_col, window):
        super().__init__()
        self._df = df; self._tc = text_col; self._yc = year_col; self._win = window

    def run(self):
        try:
            self.progress.emit("Analyzing SDG drift...")
            df = ensure_sdg_columns(self._df)
            analysis = analyze_sdg_drift(
                df, text_col=self._tc, year_col=self._yc,
                window_size=self._win, verbose=False)
            res = {
                "ranking": analysis.get_drift_ranking(),
                "summary": analysis.get_sdg_summary_df(),
            }
            self.finished.emit(res, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("sdg drift failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWSDGDrift(OWWidget):
    """Conceptual drift of SDGs over time."""

    name = "SDG Drift"
    description = "How the vocabulary of each SDG drifts across time windows"
    icon = "icons/sdg_drift.svg"
    priority = 540
    keywords = ["sdg", "drift", "conceptual", "evolution", "temporal",
                "sustainable development goals"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with SDG indicators + text")

    class Outputs:
        ranking = Output("Drift Ranking", Table, doc="SDGs ranked by average drift")
        summary = Output("SDG Summary", Table, doc="Per-window SDG summary")

    text_col = settings.Setting("")
    window_size = settings.Setting(5)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        no_sdg = Msg("No SDG columns found (need 'SDG N' columns or an SDG list column)")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("{} SDGs analyzed")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Options")
        grid = QGridLayout()
        grid.addWidget(QLabel("Text column:"), 0, 0)
        self.text_combo = QComboBox()
        self.text_combo.currentTextChanged.connect(lambda t: setattr(self, "text_col", t))
        grid.addWidget(self.text_combo, 0, 1)
        box.layout().addLayout(grid)
        gui.spin(box, self, "window_size", 2, 15, label="Window size (years):")
        self.run_btn = QPushButton("Analyze drift"); self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.summary_label = QLabel("No data"); self.summary_label.setWordWrap(True)
        self.mainArea.layout().addWidget(self.summary_label)
        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.graph.setLabel("bottom", "Average drift")
        self.mainArea.layout().addWidget(self.graph)

        if not HAS_BIBLIUM:
            self.Error.no_biblium(); self.run_btn.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.text_combo.blockSignals(True)
        self.text_combo.clear()
        if self._df is not None and not self._df.empty:
            cols = [c for c in TEXT_CANDIDATES if c in self._df.columns]
            cols += [c for c in self._df.columns if c not in cols and
                     self._df[c].dtype == object]
            self.text_combo.addItems(cols)
            if self.text_col in cols:
                self.text_combo.setCurrentText(self.text_col)
            elif cols:
                self.text_col = cols[0]
        self.text_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py", "oa_publication_year"):
                return c
        return "Year"

    def _compute(self):
        self.Error.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        if not self.text_combo.currentText():
            return
        check = ensure_sdg_columns(self._df)
        if get_sdg_columns is not None and not get_sdg_columns(check):
            self.Error.no_sdg(); return
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = DriftWorker(self._df, self.text_combo.currentText(),
                                   self._year_col(), self.window_size)
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, res, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or res is None or res["ranking"] is None or res["ranking"].empty:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "no drift computed")
            self.Outputs.ranking.send(None); self.Outputs.summary.send(None)
            return
        rk = res["ranking"]
        self.summary_label.setText(
            f"<b>{len(rk)}</b> SDGs. Highest drift: " +
            ", ".join(f"SDG {r['SDG']} ({r['Avg Drift']:.2f})"
                      for _, r in rk.head(4).iterrows()))
        self._render(rk)
        self.status_label.setText(f"Done — {len(rk)} SDGs")
        self.Information.done(len(rk))
        self.Outputs.ranking.send(_df_to_table(rk))
        self.Outputs.summary.send(_df_to_table(res["summary"]))

    def _render(self, rk):
        self.graph.clear()
        if rk is None or rk.empty or "Avg Drift" not in rk.columns:
            return
        m = rk.reset_index(drop=True)
        ys = list(range(len(m)))
        self.graph.addItem(pg.BarGraphItem(
            x0=0, y=ys, height=0.6,
            width=list(pd.to_numeric(m["Avg Drift"], errors="coerce").fillna(0)),
            brush=pg.mkBrush("#c0392b")))
        labels = []
        for i in range(len(m)):
            nm = m.iloc[i].get("Name", "") if "Name" in m.columns else ""
            labels.append(f"SDG {m.iloc[i]['SDG']} {nm}"[:26])
        self.graph.getAxis("left").setTicks([[(i, labels[i]) for i in ys]])
        self.graph.setYRange(-1, len(m))
        self.graph.getViewBox().invertY(True)  # highest drift on top

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWSDGDrift).run()
