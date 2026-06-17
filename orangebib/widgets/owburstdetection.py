# -*- coding: utf-8 -*-
"""
Burst Detection Widget
=====================
Detect bursts of activity in keywords (or other entities) over time using
Kleinberg's burst-detection algorithm, as implemented in Biblium
(`BiblioStats.compute_bursts`). Shows a burst timeline (Gantt-style) and a
table of detected bursts.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox, QGridLayout,
    QProgressBar, QApplication,
)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.bibstats import BiblioStats
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    BiblioStats = None

logger = logging.getLogger(__name__)

KW_CANDIDATES = [
    "Processed Author Keywords", "Author Keywords", "Author keywords", "DE",
    "Index Keywords", "Index keywords", "ID", "Keywords",
    "oa_topics", "oa_concepts", "oa_fields",
]


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
        if pd.api.types.is_numeric_dtype(df[c]):
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
        M[:, i] = df[c].astype(object).where(df[c].notna(), "").values
    return Table.from_numpy(domain, X, metas=M)


class BurstWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, db, kw_col, year_col, top_n, s, gamma, min_dur):
        super().__init__()
        self._df = df; self._db = db; self._kw = kw_col; self._yc = year_col
        self._top_n = top_n; self._s = s; self._gamma = gamma; self._md = min_dur

    def run(self):
        try:
            self.progress.emit("Building analysis...")
            bs = BiblioStats(df=self._df, db=self._db or "", label_docs=False,
                             res_folder=None)
            self.progress.emit("Detecting bursts...")
            out = bs.compute_bursts(keyword_col=self._kw, year_col=self._yc,
                                    top_n=self._top_n, s=self._s,
                                    gamma=self._gamma, min_duration=self._md)
            self.finished.emit(out, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("burst detection failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWBurstDetection(OWWidget):
    """Kleinberg burst detection over keywords/entities through time."""

    name = "Burst Detection"
    description = "Detect bursts of activity in keywords over time (Kleinberg)"
    icon = "icons/burst_detection.svg"
    priority = 260
    keywords = ["burst", "kleinberg", "emerging", "trend", "temporal",
                "keyword", "spike"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    class Outputs:
        bursts = Output("Bursts", Table, doc="Detected bursts (Keyword, Start, End, Weight)")
        selected = Output("Selected Documents", Table, doc="Documents for the selected bursting entities")

    keyword_col = settings.Setting("")
    top_n = settings.Setting(50)
    s = settings.Setting(2.0)
    gamma = settings.Setting(1.0)
    min_duration = settings.Setting(0)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium is required. Install biblium>=2.16.")
        compute_error = Msg("Computation error: {}")

    class Warning(OWWidget.Warning):
        no_bursts = Msg("No bursts detected with current settings")

    class Information(OWWidget.Information):
        done = Msg("Detected {} bursts")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None
        self._result = None

        self._build_controls()
        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.graph.setLabel("bottom", "Year")
        self.graph.scene().sigMouseClicked.connect(self._on_bar_clicked)
        self.mainArea.layout().addWidget(self.graph)
        self._selected_kw = set()

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _build_controls(self):
        box = gui.widgetBox(self.controlArea, "Options")
        grid = QGridLayout()
        grid.addWidget(QLabel("Keywords column:"), 0, 0)
        self.kw_combo = QComboBox()
        self.kw_combo.currentTextChanged.connect(lambda t: setattr(self, "keyword_col", t))
        grid.addWidget(self.kw_combo, 0, 1)

        grid.addWidget(QLabel("Top N:"), 1, 0)
        self.topn = QSpinBox(); self.topn.setRange(5, 1000); self.topn.setValue(self.top_n)
        self.topn.valueChanged.connect(lambda v: setattr(self, "top_n", v))
        grid.addWidget(self.topn, 1, 1)

        grid.addWidget(QLabel("s (burst scaling):"), 2, 0)
        self.s_spin = QDoubleSpinBox(); self.s_spin.setRange(1.1, 10.0)
        self.s_spin.setSingleStep(0.1); self.s_spin.setValue(self.s)
        self.s_spin.valueChanged.connect(lambda v: setattr(self, "s", v))
        grid.addWidget(self.s_spin, 2, 1)

        grid.addWidget(QLabel("gamma (transition cost):"), 3, 0)
        self.g_spin = QDoubleSpinBox(); self.g_spin.setRange(0.1, 10.0)
        self.g_spin.setSingleStep(0.1); self.g_spin.setValue(self.gamma)
        self.g_spin.valueChanged.connect(lambda v: setattr(self, "gamma", v))
        grid.addWidget(self.g_spin, 3, 1)

        grid.addWidget(QLabel("Min duration (yrs):"), 4, 0)
        self.md = QSpinBox(); self.md.setRange(0, 50); self.md.setValue(self.min_duration)
        self.md.valueChanged.connect(lambda v: setattr(self, "min_duration", v))
        grid.addWidget(self.md, 4, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Detect Bursts")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py",
                                  "publication_year", "oa_publication_year"):
                return c
        return "Year"

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Warning.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.kw_combo.blockSignals(True)
        self.kw_combo.clear()
        if data is None:
            self.kw_combo.blockSignals(False)
            self.Error.no_data()
            return
        cols = [c for c in KW_CANDIDATES if c in self._df.columns]
        cols += [c for c in self._df.columns if c not in cols and
                 "keyword" in str(c).lower()]
        if not cols:
            cols = list(self._df.columns)
        self.kw_combo.addItems(cols)
        if self.keyword_col in cols:
            self.kw_combo.setCurrentText(self.keyword_col)
        else:
            self.keyword_col = cols[0]
        self.kw_combo.blockSignals(False)

    def _compute(self):
        self.Error.clear(); self.Warning.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        db = "oa" if any(str(c).startswith("oa_") for c in self._df.columns) else ""
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = BurstWorker(
            self._df, db, self.kw_combo.currentText(), self._year_col(),
            self.top_n, self.s, self.gamma, self.min_duration)
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, out, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or out is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            self.Outputs.bursts.send(None)
            return
        if out.empty:
            self.Warning.no_bursts()
            self.graph.clear()
            self.Outputs.bursts.send(None)
            self.status_label.setText("No bursts")
            return
        # drop unnamed / empty entities
        if "Keyword" in out.columns:
            kw = out["Keyword"].astype(str).str.strip()
            out = out[(kw != "") & (kw.str.lower() != "nan")].reset_index(drop=True)
        if out.empty:
            self.Warning.no_bursts()
            self.graph.clear()
            self.Outputs.bursts.send(None)
            self.Outputs.selected.send(None)
            self.status_label.setText("No bursts")
            return
        self._result = out
        self._selected_kw = set()
        self._render(out)
        self.status_label.setText(f"Done — {len(out)} bursts")
        self.Information.done(len(out))
        self.Outputs.bursts.send(_df_to_table(out))
        self.Outputs.selected.send(None)

    def _render(self, df):
        self.graph.clear()
        df = df.sort_values("Start").reset_index(drop=True)
        wmax = float(pd.to_numeric(df["Weight"], errors="coerce").max() or 1)
        names = []
        self._row_kw = []
        for i, row in df.iterrows():
            start, end = float(row["Start"]), float(row["End"])
            w = float(row["Weight"])
            kw = str(row["Keyword"])
            self._row_kw.append(kw)
            if kw in self._selected_kw:
                color = pg.mkBrush("#e67e22")
            else:
                t = w / wmax if wmax else 0.5
                color = pg.mkBrush(int(40 + 60 * (1 - t)), int(90 + 100 * (1 - t)),
                                   int(180 + 40 * t), 200)
            bar = pg.BarGraphItem(x0=start, x1=max(end, start + 0.4), y=i,
                                  height=0.7, brush=color, pen=pg.mkPen("w"))
            self.graph.addItem(bar)
            names.append(kw[:30])
        self.graph.getAxis("left").setTicks([[(i, names[i]) for i in range(len(names))]])
        self.graph.setYRange(-1, len(names))

    def _on_bar_clicked(self, ev):
        if not getattr(self, "_row_kw", None):
            return
        vb = self.graph.getPlotItem().vb
        p = vb.mapSceneToView(ev.scenePos())
        i = int(round(p.y()))
        if not (0 <= i < len(self._row_kw)):
            return
        kw = self._row_kw[i]
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl:
            self._selected_kw.symmetric_difference_update({kw})
        else:
            self._selected_kw = set() if self._selected_kw == {kw} else {kw}
        self._render(self._result)
        self._send_selected()

    def _send_selected(self):
        col = self.kw_combo.currentText()
        if (self._data is None or self._df is None or not self._selected_kw
                or not col or col not in self._df.columns):
            self.Outputs.selected.send(None)
            return
        series = self._df[col].astype(str)
        sel_low = {k.lower() for k in self._selected_kw}
        idx = []
        for ri in range(len(series)):
            cell = series.iloc[ri].lower()
            if any(k in cell for k in sel_low) and ri < len(self._data):
                idx.append(ri)
        self.Outputs.selected.send(self._data[idx] if idx else None)

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWBurstDetection).run()
