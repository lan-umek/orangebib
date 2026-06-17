# -*- coding: utf-8 -*-
"""
Embedding Landscape Widget
=========================
Project documents onto a 2-D semantic map: similar papers land near each
other. Embeddings come from `biblium.addons.embedding_landscape.embed_corpus`
(sentence-transformers if available, else TF-IDF+SVD), reduced to 2-D and
clustered. Hover shows the title; the 2-D coordinates + cluster are emitted.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import QLabel, QComboBox, QPushButton, QSpinBox, QGridLayout, QProgressBar

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.addons.embedding_landscape import embed_corpus, cluster_landscape
    from sklearn.decomposition import PCA
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    embed_corpus = None
    cluster_landscape = None
    PCA = None

logger = logging.getLogger(__name__)
TEXT_CANDIDATES = ["Processed Combined Text", "Processed Abstract", "Abstract",
                   "Processed Title", "Title"]
PALETTE = ["#4a90d9", "#e8743b", "#5aa454", "#c0504d", "#8064a2", "#1aa8a8",
           "#d9a441", "#9b59b6", "#16a085", "#e74c3c", "#2c3e50", "#f39c12"]


def _table_to_df(table):
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


class LandscapeWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, text_col, n_clusters):
        super().__init__()
        self._df = df; self._tc = text_col; self._nc = n_clusters

    def run(self):
        try:
            self.progress.emit("Embedding documents...")
            emb, method_used, _ = embed_corpus(self._df, text_col=self._tc, method="auto")
            self.progress.emit("Reducing to 2-D...")
            emb = np.asarray(emb)
            if emb.shape[1] > 2:
                coords = PCA(n_components=2, random_state=42).fit_transform(emb)
            else:
                coords = emb
            self.progress.emit("Clustering...")
            try:
                labels = cluster_landscape(coords, method="kmeans", n_clusters=self._nc)
            except Exception:  # noqa: BLE001
                labels = np.zeros(len(coords), dtype=int)
            self.finished.emit((coords, np.asarray(labels), method_used), "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("embedding landscape failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWEmbeddingLandscape(OWWidget):
    """2-D semantic map of documents."""

    name = "Embedding Landscape"
    description = "2-D semantic map of documents (embeddings + clustering)"
    icon = "icons/embedding_landscape.svg"
    priority = 480
    keywords = ["embedding", "landscape", "semantic", "map", "umap", "tsne",
                "cluster", "similarity"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with text")

    class Outputs:
        coordinates = Output("Coordinates", Table, doc="2-D coords + cluster per document")

    text_col = settings.Setting("")
    n_clusters = settings.Setting(10)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons + scikit-learn required.")
        compute_error = Msg("{}")

    class Information(OWWidget.Information):
        done = Msg("{} documents · {} embedding")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None
        self._coords = None
        self._labels = None
        self._titles = None

        box = gui.widgetBox(self.controlArea, "Options")
        grid = QGridLayout()
        grid.addWidget(QLabel("Text column:"), 0, 0)
        self.text_combo = QComboBox()
        self.text_combo.currentTextChanged.connect(lambda t: setattr(self, "text_col", t))
        grid.addWidget(self.text_combo, 0, 1)
        grid.addWidget(QLabel("Clusters:"), 1, 0)
        self.nc = QSpinBox(); self.nc.setRange(2, 40); self.nc.setValue(self.n_clusters)
        self.nc.valueChanged.connect(lambda v: setattr(self, "n_clusters", v))
        grid.addWidget(self.nc, 1, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Build Landscape")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        self._scatter = pg.ScatterPlotItem(hoverable=True)
        self._tip = pg.TextItem(color="k", fill=pg.mkBrush(255, 255, 220, 230), anchor=(0, 1))
        self._tip.setZValue(100); self._tip.hide()
        self.graph.scene().sigMouseMoved.connect(self._hover)
        self.mainArea.layout().addWidget(self.graph)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _title_col(self):
        for c in ("Title", "TI", "Document Title"):
            if self._df is not None and c in self._df.columns:
                return c
        return None

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.text_combo.blockSignals(True); self.text_combo.clear()
        if self._df is not None:
            cols = [c for c in TEXT_CANDIDATES if c in self._df.columns]
            cols += [c for c in self._df.columns if c not in cols and self._df[c].dtype == object]
            self.text_combo.addItems(cols)
            if self.text_col in cols:
                self.text_combo.setCurrentText(self.text_col)
            elif cols:
                self.text_col = cols[0]
        self.text_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        tc = self.text_combo.currentText()
        if not tc:
            self.Error.compute_error("No text column"); return
        tcol = self._title_col()
        self._titles = (self._df[tcol].astype(str).tolist() if tcol else
                        [f"doc {i}" for i in range(len(self._df))])
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = LandscapeWorker(self._df, tc, self.n_clusters)
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, result, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or result is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            self.Outputs.coordinates.send(None)
            return
        coords, labels, method_used = result
        self._coords = coords; self._labels = labels
        self._render()
        n = len(coords)
        self.status_label.setText(f"Done — {n} docs ({method_used})")
        self.Information.done(n, method_used)
        out = pd.DataFrame({
            "Title": self._titles[:len(coords)],
            "x": coords[:, 0], "y": coords[:, 1], "Cluster": labels,
        })
        domain = Domain([ContinuousVariable("x"), ContinuousVariable("y"),
                         ContinuousVariable("Cluster")],
                        metas=[StringVariable("Title")])
        X = out[["x", "y", "Cluster"]].values.astype(float)
        M = out[["Title"]].astype(str).values
        self.Outputs.coordinates.send(Table.from_numpy(domain, X, metas=M))

    def _render(self):
        self.graph.clear()
        spots = []
        for i in range(len(self._coords)):
            c = int(self._labels[i]) if self._labels is not None else 0
            spots.append({"pos": (self._coords[i, 0], self._coords[i, 1]), "data": i,
                          "size": 9, "brush": pg.mkBrush(PALETTE[c % len(PALETTE)]),
                          "pen": pg.mkPen("w", width=0.3)})
        self._scatter.setData(spots)
        self.graph.addItem(self._scatter)
        self.graph.addItem(self._tip)
        self.graph.getViewBox().autoRange()

    def _hover(self, p):
        if self._coords is None:
            return
        vb = self.graph.getPlotItem().vb
        if not self.graph.sceneBoundingRect().contains(p):
            self._tip.hide(); return
        pts = self._scatter.pointsAt(vb.mapSceneToView(p))
        if len(pts):
            i = pts[0].data()
            mp = vb.mapSceneToView(p)
            self._tip.setText(str(self._titles[i])[:80]); self._tip.setPos(mp.x(), mp.y()); self._tip.show()
        else:
            self._tip.hide()

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWEmbeddingLandscape).run()
