# -*- coding: utf-8 -*-
"""
Reference Diversity Widget
=========================
Measure the interdisciplinarity / diversity of each paper's reference list.

Wraps :func:`biblium.reference_diversity.analyze_reference_diversity`, which
computes Shannon and Simpson diversity plus the Rao-Stirling
interdisciplinarity index over the sources, fields and topics cited by each
paper. Optionally enriches references via the OpenAlex API.
"""

import logging
import re
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
    from biblium.reference_diversity import (
        analyze_reference_diversity, add_diversity_to_dataframe,
    )
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    analyze_reference_diversity = None
    add_diversity_to_dataframe = None

logger = logging.getLogger(__name__)

# Scalar metric fields rendered in the per-paper table.
METRIC_FIELDS = [
    ("title", "Title"), ("pub_year", "Year"),
    ("reference_count", "Refs"), ("unique_sources", "Sources"),
    ("unique_fields", "Fields"), ("unique_topics", "Topics"),
    ("source_diversity", "Source div."), ("field_diversity", "Field div."),
    ("topic_diversity", "Topic div."), ("rao_stirling_index", "Rao-Stirling"),
    ("mean_ref_age", "Mean ref age"), ("self_citation_rate", "Self-cite rate"),
    ("diversity_level", "Level"),
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


class DiversityWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, use_openalex, max_papers, max_refs, current_year):
        super().__init__()
        self._df = df
        self._use_openalex = use_openalex
        self._max_papers = max_papers
        self._max_refs = max_refs
        self._current_year = current_year
        self._stop = False

    def stop(self):
        self._stop = True

    @staticmethod
    def _prepare_refs(df):
        """biblium expects a pipe-separated 'referenced_works' column. The
        OpenAlex enrichment widget writes 'oa_referenced_works' (and may have
        normalized its separator), so alias/repair it here."""
        if "referenced_works" in df.columns:
            return df
        if "oa_referenced_works" not in df.columns:
            return df
        df = df.copy()

        def _repipe(v):
            if v is None or (isinstance(v, float) and v != v):
                return ""
            parts = [p.strip() for p in re.split(r"[|;]\s*", str(v)) if p.strip()]
            return "|".join(parts)

        df["referenced_works"] = df["oa_referenced_works"].map(_repipe)
        return df

    def run(self):
        try:
            self.progress.emit(10, "Analyzing reference diversity...")
            df = self._prepare_refs(self._df)
            res = analyze_reference_diversity(
                df,
                use_openalex=self._use_openalex,
                max_papers=self._max_papers,
                max_refs_per_paper=self._max_refs,
                current_year=self._current_year,
                verbose=False,
                stop_flag=lambda: self._stop,
            )
            self.progress.emit(100, "Done")
            self.finished.emit(res, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("reference diversity failed")
            self.finished.emit(None, str(exc))


class _NumItem(QTableWidgetItem):
    """Table item that sorts numerically when a number is stored."""

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


class OWReferenceDiversity(OWWidget):
    """Compute reference diversity / interdisciplinarity metrics."""

    name = "Reference Diversity"
    description = "Shannon/Simpson/Rao-Stirling diversity of each paper's references"
    icon = "icons/reference_diversity.svg"
    priority = 170
    keywords = ["diversity", "interdisciplinarity", "references", "shannon",
                "simpson", "rao-stirling", "interdisciplinary"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")

    class Outputs:
        metrics = Output("Per-paper Metrics", Table, doc="Diversity metrics per paper")
        annotated_data = Output("Data", Table, doc="Input data with diversity columns")
        summary = Output("Summary", Table, doc="Aggregate diversity summary")

    use_openalex = settings.Setting(False)
    max_papers = settings.Setting(100)
    max_refs = settings.Setting(50)
    current_year = settings.Setting(datetime.now().year)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium is required. Install biblium>=2.16.")
        compute_error = Msg("Computation error: {}")

    class Warning(OWWidget.Warning):
        need_openalex = Msg(
            "References are OpenAlex IDs (oa_referenced_works). Enable "
            "'Enrich references via OpenAlex' to compute source/field "
            "diversity, otherwise most metrics stay 0.")

    class Information(OWWidget.Information):
        done = Msg("{}")

    def __init__(self):
        super().__init__()
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._worker: Optional[DiversityWorker] = None
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
        self.papers_spin.setRange(1, 100000)
        self.papers_spin.setValue(self.max_papers)
        self.papers_spin.valueChanged.connect(lambda v: setattr(self, "max_papers", v))
        grid.addWidget(self.papers_spin, 0, 1)

        grid.addWidget(QLabel("Max refs / paper:"), 1, 0)
        self.refs_spin = QSpinBox()
        self.refs_spin.setRange(1, 10000)
        self.refs_spin.setValue(self.max_refs)
        self.refs_spin.valueChanged.connect(lambda v: setattr(self, "max_refs", v))
        grid.addWidget(self.refs_spin, 1, 1)

        grid.addWidget(QLabel("Current year:"), 2, 0)
        self.year_spin = QSpinBox()
        self.year_spin.setRange(1900, 2200)
        self.year_spin.setValue(self.current_year)
        self.year_spin.valueChanged.connect(lambda v: setattr(self, "current_year", v))
        grid.addWidget(self.year_spin, 2, 1)
        box.layout().addLayout(grid)

        self.oa_cb = QCheckBox("Enrich references via OpenAlex (slower)")
        self.oa_cb.setChecked(self.use_openalex)
        self.oa_cb.toggled.connect(lambda c: setattr(self, "use_openalex", c))
        box.layout().addWidget(self.oa_cb)

        self.run_btn = QPushButton("Compute Diversity")
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

        tbox = gui.widgetBox(self.mainArea, "Per-paper Diversity")
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
        self.Warning.clear()
        only_oa_refs = ("referenced_works" not in self._df.columns
                        and "oa_referenced_works" in self._df.columns)
        if only_oa_refs and not self.use_openalex:
            self.Warning.need_openalex()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")

        self._worker = DiversityWorker(
            self._df, self.use_openalex, self.max_papers,
            self.max_refs, self.current_year)
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
        self.summary_label.setText(
            f"Analyzed <b>{res.n_analyzed}</b>/{res.n_papers} papers "
            f"({res.n_with_references} with references). "
            f"Avg field diversity: {res.avg_field_diversity:.3f}, "
            f"avg source diversity: {res.avg_source_diversity:.3f}. "
            f"Source: {res.data_source}.")
        self.status_label.setText(f"Done — {res.n_analyzed} papers")
        self.Information.done(f"Analyzed {res.n_analyzed} papers ({res.data_source})")

        annotated = None
        if add_diversity_to_dataframe is not None:
            try:
                annotated = _df_to_table(add_diversity_to_dataframe(self._df, res))
            except Exception:  # noqa: BLE001
                logger.warning("add_diversity_to_dataframe failed", exc_info=True)
        self.Outputs.metrics.send(_df_to_table(metrics_df))
        self.Outputs.annotated_data.send(annotated if annotated is not None else self._data)
        self.Outputs.summary.send(_df_to_table(summary_df))

    @staticmethod
    def _metrics_df(res) -> pd.DataFrame:
        rows = []
        for m in res.metrics:
            rows.append({label: getattr(m, attr, None) for attr, label in METRIC_FIELDS})
        return pd.DataFrame(rows)

    @staticmethod
    def _summary_df(res) -> pd.DataFrame:
        row = {
            "Papers": res.n_papers, "Analyzed": res.n_analyzed,
            "With references": res.n_with_references,
            "Avg reference count": res.avg_reference_count,
            "Avg source diversity": res.avg_source_diversity,
            "Avg field diversity": res.avg_field_diversity,
            "Avg ref age": res.avg_ref_age,
            "Avg self-citation rate": res.avg_self_citation_rate,
            "Data source": res.data_source,
            "API calls": res.api_calls_made,
        }
        return pd.DataFrame([row])

    def _fill_table(self, df: pd.DataFrame):
        if df is None or df.empty:
            self.metrics_table.setRowCount(0)
            self.metrics_table.setColumnCount(0)
            return
        # Drop the per-document self-citation column when it is entirely zero:
        # self-citation is meaningful at the author/source level, not per paper
        # (the aggregate stays in the Summary output).
        if "Self-cite rate" in df.columns:
            col = pd.to_numeric(df["Self-cite rate"], errors="coerce").fillna(0)
            if float(col.abs().sum()) == 0.0:
                df = df.drop(columns=["Self-cite rate"])
        self.metrics_table.setSortingEnabled(False)
        self.metrics_table.setColumnCount(len(df.columns))
        self.metrics_table.setRowCount(len(df))
        self.metrics_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                v = df.iloc[r, c]
                num = None
                if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                    num = float(v)
                    txt = f"{v:,.3f}" if isinstance(v, (float, np.floating)) else f"{v:,}"
                else:
                    txt = str(v) if v is not None else ""
                self.metrics_table.setItem(r, c, _NumItem(txt, num))
        self.metrics_table.resizeColumnsToContents()
        self.metrics_table.setSortingEnabled(True)

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWReferenceDiversity).run()
