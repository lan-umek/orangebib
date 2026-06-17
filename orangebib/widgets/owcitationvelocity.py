# -*- coding: utf-8 -*-
"""
Citation Velocity Widget
=======================
Measure how fast each paper accumulates citations and classify its trend
(accelerating, steady, decelerating, ...).

Wraps :func:`biblium.citation_velocity.analyze_citation_velocity`. Citation
histories come from the dataset's yearly counts, or from the OpenAlex API
when enabled.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QLabel, QPushButton, QSpinBox, QCheckBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QProgressBar,
)

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.citation_velocity import analyze_citation_velocity
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    analyze_citation_velocity = None

logger = logging.getLogger(__name__)

METRIC_FIELDS = [
    ("title", "Title"), ("pub_year", "Year"),
    ("total_citations", "Total cites"), ("paper_age", "Age"),
    ("current_velocity", "Current vel."), ("average_velocity", "Avg vel."),
    ("peak_velocity", "Peak vel."), ("peak_year", "Peak year"),
    ("recent_citations", "Recent cites"), ("momentum", "Momentum"),
    ("momentum_pct", "Momentum %"), ("velocity_ratio", "Vel. ratio"),
    ("years_since_peak", "Yrs since peak"), ("trend", "Trend"),
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


def _trend_str(trend) -> str:
    return getattr(trend, "value", None) or getattr(trend, "name", None) or str(trend)


class VelocityWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, use_openalex, max_papers, current_year,
                 recent_window, min_age):
        super().__init__()
        self._df = df
        self._use_openalex = use_openalex
        self._max_papers = max_papers
        self._current_year = current_year
        self._recent_window = recent_window
        self._min_age = min_age
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.progress.emit(10, "Analyzing citation velocity...")
            res = analyze_citation_velocity(
                self._df,
                use_openalex=self._use_openalex,
                max_papers=self._max_papers,
                current_year=self._current_year,
                recent_window=self._recent_window,
                min_age=self._min_age,
                verbose=False,
                stop_flag=lambda: self._stop,
            )
            self.progress.emit(100, "Done")
            self.finished.emit(res, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("citation velocity failed")
            self.finished.emit(None, str(exc))


class OWCitationVelocity(OWWidget):
    """Compute citation velocity and trend metrics per paper."""

    name = "Citation Velocity"
    description = "Citation accumulation speed and trend classification per paper"
    icon = "icons/citation_velocity.svg"
    priority = 310
    keywords = ["citation", "velocity", "momentum", "trend", "acceleration",
                "impact", "dynamics"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")

    class Outputs:
        metrics = Output("Per-paper Metrics", Table, doc="Velocity metrics per paper")
        summary = Output("Summary", Table, doc="Trend counts summary")

    use_openalex = settings.Setting(False)
    max_papers = settings.Setting(500)
    current_year = settings.Setting(datetime.now().year)
    recent_window = settings.Setting(3)
    min_age = settings.Setting(2)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium is required. Install biblium>=2.16.")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("{}")

    def __init__(self):
        super().__init__()
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._worker: Optional[VelocityWorker] = None
        self._result = None

        self._setup_controls()
        self._setup_main_area()

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _setup_controls(self):
        box = gui.widgetBox(self.controlArea, "Options")
        grid = QGridLayout()
        grid.addWidget(QLabel("Max papers:"), 0, 0)
        self.papers_spin = QSpinBox()
        self.papers_spin.setRange(1, 1000000)
        self.papers_spin.setValue(self.max_papers)
        self.papers_spin.valueChanged.connect(lambda v: setattr(self, "max_papers", v))
        grid.addWidget(self.papers_spin, 0, 1)

        grid.addWidget(QLabel("Current year:"), 1, 0)
        self.year_spin = QSpinBox()
        self.year_spin.setRange(1900, 2200)
        self.year_spin.setValue(self.current_year)
        self.year_spin.valueChanged.connect(lambda v: setattr(self, "current_year", v))
        grid.addWidget(self.year_spin, 1, 1)

        grid.addWidget(QLabel("Recent window (yrs):"), 2, 0)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 50)
        self.window_spin.setValue(self.recent_window)
        self.window_spin.valueChanged.connect(lambda v: setattr(self, "recent_window", v))
        grid.addWidget(self.window_spin, 2, 1)

        grid.addWidget(QLabel("Min paper age:"), 3, 0)
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 100)
        self.age_spin.setValue(self.min_age)
        self.age_spin.valueChanged.connect(lambda v: setattr(self, "min_age", v))
        grid.addWidget(self.age_spin, 3, 1)
        box.layout().addLayout(grid)

        self.oa_cb = QCheckBox("Fetch histories via OpenAlex (slower)")
        self.oa_cb.setChecked(self.use_openalex)
        self.oa_cb.toggled.connect(lambda c: setattr(self, "use_openalex", c))
        box.layout().addWidget(self.oa_cb)

        self.run_btn = QPushButton("Compute Velocity")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.controlArea.layout().addWidget(self.cancel_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)

    def _setup_main_area(self):
        sbox = gui.widgetBox(self.mainArea, "Summary")
        self.summary_label = QLabel("No data")
        self.summary_label.setWordWrap(True)
        sbox.layout().addWidget(self.summary_label)

        tbox = gui.widgetBox(self.mainArea, "Per-paper Velocity")
        self.metrics_table = QTableWidget()
        self.metrics_table.setMinimumHeight(280)
        tbox.layout().addWidget(self.metrics_table)

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        if data is None:
            self.Error.no_data()

    def _cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.status_label.setText("Cancelling...")

    def _compute(self):
        self.Error.clear()
        self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            return
        if self._df is None or self._df.empty:
            self.Error.no_data()
            return
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")

        self._worker = VelocityWorker(
            self._df, self.use_openalex, self.max_papers, self.current_year,
            self.recent_window, self.min_age)
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, res, error):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        if error or res is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            self.Outputs.metrics.send(None)
            self.Outputs.summary.send(None)
            return
        self._result = res
        metrics_df = self._metrics_df(res)
        summary_df = self._summary_df(res)
        self._fill_table(metrics_df)
        trend_txt = ", ".join(f"{k}: {v}" for k, v in res.trend_counts.items())
        self.summary_label.setText(
            f"Analyzed <b>{res.n_analyzed}</b>/{res.n_papers} papers "
            f"(current year {res.current_year}, recent window "
            f"{res.recent_window} yrs).<br>Trends — {trend_txt}")
        self.status_label.setText(f"Done — {res.n_analyzed} papers")
        self.Information.done(f"Analyzed {res.n_analyzed} papers")
        self.Outputs.metrics.send(_df_to_table(metrics_df))
        self.Outputs.summary.send(_df_to_table(summary_df))

    @staticmethod
    def _metrics_df(res) -> pd.DataFrame:
        rows = []
        for m in res.metrics:
            row = {}
            for attr, label in METRIC_FIELDS:
                if attr == "trend":
                    row[label] = _trend_str(getattr(m, "trend", ""))
                else:
                    row[label] = getattr(m, attr, None)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _summary_df(res) -> pd.DataFrame:
        rows = [{"Trend": k, "Count": v} for k, v in res.trend_counts.items()]
        if not rows:
            rows = [{"Trend": "(none)", "Count": 0}]
        return pd.DataFrame(rows)

    def _fill_table(self, df: pd.DataFrame):
        if df is None or df.empty:
            self.metrics_table.setRowCount(0)
            self.metrics_table.setColumnCount(0)
            return
        self.metrics_table.setColumnCount(len(df.columns))
        self.metrics_table.setRowCount(len(df))
        self.metrics_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                v = df.iloc[r, c]
                if isinstance(v, (float, np.floating)) and not isinstance(v, bool):
                    txt = f"{v:,.3f}"
                elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                    txt = f"{v:,}"
                else:
                    txt = str(v) if v is not None else ""
                self.metrics_table.setItem(r, c, QTableWidgetItem(txt))
        self.metrics_table.resizeColumnsToContents()

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWCitationVelocity).run()
