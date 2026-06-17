# -*- coding: utf-8 -*-
"""
Document Clustering Widget
=========================
Cluster documents by their text (abstract / title / keywords) and, optionally,
by bibliographic coupling (shared references), using Biblium's
`BiblioStats.cluster_documents`. Adds a Cluster column to the data and reports
cluster sizes.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QPushButton, QSpinBox, QCheckBox, QGridLayout,
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

TEXT_CANDIDATES = ["Processed Abstract", "Abstract", "Processed Combined Text",
                   "Processed Title", "Title", "Processed Author Keywords",
                   "Author Keywords", "Index Keywords"]
REF_CANDIDATES = ["referenced_works", "oa_referenced_works", "References",
                  "Cited References", "CR"]
METHODS = ["kmeans", "agglomerative"]


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


class ClusterWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str, str)  # df, cluster_col, error

    def __init__(self, df, db, text_field, method, n_clusters, coupling):
        super().__init__()
        self._df = df; self._db = db; self._tf = text_field
        self._method = method; self._n = n_clusters; self._coupling = coupling

    def run(self):
        try:
            self.progress.emit("Building analysis...")
            bs = BiblioStats(df=self._df, db=self._db or "", label_docs=False,
                             res_folder=None)
            self.progress.emit("Clustering documents...")
            bs.cluster_documents(
                text_field=self._tf, method=self._method,
                n_clusters=(self._n or None),
                coupling_fields=(self._coupling or None))
            col = getattr(bs, "last_cluster_column", None) or f"{self._method}_cluster"
            out = bs.df.copy()
            self.finished.emit(out, col, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("clustering failed")
            self.finished.emit(None, "", f"{type(exc).__name__}: {exc}")


class OWDocClustering(OWWidget):
    """Cluster documents by text and/or bibliographic coupling."""

    name = "Document Clustering"
    description = "Cluster documents by text (and optionally shared references)"
    icon = "icons/doc_clustering.svg"
    priority = 700
    keywords = ["cluster", "clustering", "kmeans", "coupling", "topics",
                "groups", "unsupervised"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    class Outputs:
        data = Output("Data", Table, doc="Input data with a Cluster column")
        summary = Output("Cluster Summary", Table, doc="Cluster sizes")
        selected = Output("Selected Documents", Table, doc="Documents in the selected clusters")

    text_field = settings.Setting("")
    method = settings.Setting("kmeans")
    n_clusters = settings.Setting(0)
    use_coupling = settings.Setting(False)
    ref_field = settings.Setting("")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium is required (biblium>=2.16).")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("{} clusters over {} documents")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None
        self._out_df = None
        self._summary = None
        self._selected_clusters = set()

        box = gui.widgetBox(self.controlArea, "Clustering")
        grid = QGridLayout()
        grid.addWidget(QLabel("Cluster by (text):"), 0, 0)
        self.text_combo = QComboBox()
        self.text_combo.currentTextChanged.connect(lambda t: setattr(self, "text_field", t))
        grid.addWidget(self.text_combo, 0, 1)
        grid.addWidget(QLabel("Method:"), 1, 0)
        self.method_combo = QComboBox(); self.method_combo.addItems(METHODS)
        self.method_combo.setCurrentText(self.method)
        self.method_combo.currentTextChanged.connect(lambda t: setattr(self, "method", t))
        grid.addWidget(self.method_combo, 1, 1)
        grid.addWidget(QLabel("N clusters (0=auto):"), 2, 0)
        self.n_spin = QSpinBox(); self.n_spin.setRange(0, 50); self.n_spin.setValue(self.n_clusters)
        self.n_spin.valueChanged.connect(lambda v: setattr(self, "n_clusters", v))
        grid.addWidget(self.n_spin, 2, 1)
        box.layout().addLayout(grid)

        cbox = gui.widgetBox(self.controlArea, "Bibliographic coupling (optional)")
        self.coup_cb = QCheckBox("Also use shared references")
        self.coup_cb.setChecked(self.use_coupling)
        self.coup_cb.toggled.connect(self._on_coupling_toggled)
        cbox.layout().addWidget(self.coup_cb)
        self.ref_combo = QComboBox()
        self.ref_combo.setEnabled(self.use_coupling)
        self.ref_combo.currentTextChanged.connect(lambda t: setattr(self, "ref_field", t))
        cbox.layout().addWidget(self.ref_combo)

        self.run_btn = QPushButton("Cluster")
        self.run_btn.setMinimumHeight(34)
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
        self.graph.setLabel("bottom", "Documents")
        self.graph.scene().sigMouseClicked.connect(self._on_bar_clicked)
        hint = QLabel("Click a bar to select clusters and output their documents.")
        hint.setStyleSheet("color:#7f8c8d;")
        self.mainArea.layout().addWidget(hint)
        self.mainArea.layout().addWidget(self.graph)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _on_coupling_toggled(self, c):
        self.use_coupling = c
        self.ref_combo.setEnabled(c)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        for combo, cands, attr in [(self.text_combo, TEXT_CANDIDATES, "text_field"),
                                   (self.ref_combo, REF_CANDIDATES, "ref_field")]:
            combo.blockSignals(True); combo.clear()
            if self._df is not None:
                cols = [c for c in cands if c in self._df.columns]
                if combo is self.text_combo and not cols:
                    cols = [c for c in self._df.columns if self._df[c].dtype == object]
                combo.addItems(cols)
                cur = getattr(self, attr)
                if cur in cols:
                    combo.setCurrentText(cur)
                elif cols:
                    setattr(self, attr, cols[0])
            combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        tf = self.text_combo.currentText()
        if not tf:
            self.Error.compute_error("No text column"); return
        coupling = self.ref_combo.currentText() if self.use_coupling else None
        db = "oa" if any(str(c).startswith("oa_") for c in self._df.columns) else ""
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = ClusterWorker(self._df, db, tf, self.method,
                                     self.n_clusters, coupling)
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, out, cluster_col, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or out is None or cluster_col not in out.columns:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "no cluster column produced")
            self.Outputs.data.send(None); self.Outputs.summary.send(None)
            return
        out = out.rename(columns={cluster_col: "Cluster"})
        # sort clusters by frequency, descending
        sizes = out["Cluster"].value_counts().sort_values(ascending=False)
        summary = pd.DataFrame({"Cluster": sizes.index.astype(str),
                                "Size": sizes.values})
        n_clusters = len(sizes)
        self._out_df = out
        self._summary = summary
        self._selected_clusters = set()
        self.summary_label.setText(
            f"<b>{n_clusters}</b> clusters over <b>{len(out)}</b> documents.")
        self._render(summary)
        self.status_label.setText(f"Done — {n_clusters} clusters")
        self.Information.done(n_clusters, len(out))
        self.Outputs.data.send(_df_to_table(out))
        self.Outputs.summary.send(_df_to_table(summary))
        self.Outputs.selected.send(None)

    def _render(self, summary):
        self.graph.clear()
        ys = list(range(len(summary)))
        labels = [str(summary.iloc[i]["Cluster"]) for i in ys]
        brushes = [pg.mkBrush("#e67e22") if labels[i] in self._selected_clusters
                   else pg.mkBrush("#4a90d9") for i in ys]
        bar = pg.BarGraphItem(x0=0, y=ys, height=0.6,
                              width=list(summary["Size"].astype(float)),
                              brushes=brushes)
        self.graph.addItem(bar)
        self.graph.getAxis("left").setTicks(
            [[(i, f"Cluster {labels[i]}") for i in ys]])
        self.graph.setYRange(-1, len(summary))
        self.graph.getViewBox().invertY(True)  # largest on top

    def _on_bar_clicked(self, ev):
        if self._summary is None or self._out_df is None:
            return
        vb = self.graph.getPlotItem().vb
        p = vb.mapSceneToView(ev.scenePos())
        i = int(round(p.y()))
        if not (0 <= i < len(self._summary)):
            return
        label = str(self._summary.iloc[i]["Cluster"])
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl:
            self._selected_clusters.symmetric_difference_update({label})
        else:
            self._selected_clusters = (set() if self._selected_clusters == {label}
                                       else {label})
        self._render(self._summary)
        self._send_selected()

    def _send_selected(self):
        if self._out_df is None or self._data is None or not self._selected_clusters:
            self.Outputs.selected.send(None)
            return
        mask = self._out_df["Cluster"].astype(str).isin(self._selected_clusters)
        idx = [i for i, m in enumerate(mask.tolist()) if m and i < len(self._data)]
        self.Outputs.selected.send(self._data[idx] if idx else None)

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWDocClustering).run()
