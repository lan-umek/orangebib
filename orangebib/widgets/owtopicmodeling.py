# -*- coding: utf-8 -*-
"""
Topic Modeling Widget
=====================
Discover latent topics in a text field (Abstract / Title / combination) with
LDA, NMF or LSA. Shows the top terms per topic and per-topic coherence, and
outputs the topic-term table and a per-document topic-weight table.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QGridLayout,
                             QTabWidget, QTableWidget, QTableWidgetItem)
import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from biblium.utilsbib_modules.topic_modeling import (
        topic_modeling, get_topic_summary, compute_topic_coherence)
    HAS_TM = True
except Exception:  # noqa: BLE001
    HAS_TM = False
    topic_modeling = get_topic_summary = compute_topic_coherence = None

MODELS = ["LDA", "NMF", "LSA"]
PALETTE = ["#4a90d9", "#e8743b", "#5aa454", "#c0504d", "#8064a2", "#1aa8a8",
           "#d9a441", "#9b59b6", "#16a085", "#e74c3c", "#2c3e50", "#f39c12"]


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


class OWTopicModeling(OWWidget):
    """LDA / NMF / LSA topic modeling of a text field."""

    name = "Topic Modeling"
    description = "Latent topics (LDA / NMF / LSA): top terms, coherence, doc-topic weights"
    icon = "icons/topic_modeling.svg"
    priority = 360
    keywords = ["topic", "lda", "nmf", "lsa", "text", "abstract", "themes"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        topics = Output("Topic Terms", Table, doc="Topic / Term / Weight")
        documents = Output("Document Topics", Table, doc="Per-document topic weights")
        summary = Output("Topic Summary", Table, doc="One row per topic")

    text_source = settings.Setting("")
    model_type = settings.Setting(0)
    n_topics = settings.Setting(0)        # 0 = auto (<= max_topics)
    max_topics = settings.Setting(10)
    top_terms = settings.Setting(10)
    sel_topic = settings.Setting(0)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_tm = Msg("biblium topic_modeling module not available")
        failed = Msg("Topic modeling failed: {}")

    class Warning(OWWidget.Warning):
        no_text = Msg("No suitable text column (Abstract / Title / Keywords)")

    def __init__(self):
        super().__init__()
        self._df = None
        self._topics_df = None
        self._text_sources = {}
        if not HAS_TM:
            self.Error.no_tm()

        box = gui.widgetBox(self.controlArea, "Text & model")
        g = QGridLayout()
        g.addWidget(QLabel("Text:"), 0, 0)
        self.src_combo = QComboBox()
        self.src_combo.currentTextChanged.connect(lambda t: self._set("text_source", t))
        g.addWidget(self.src_combo, 0, 1)
        box.layout().addLayout(g)
        gui.comboBox(box, self, "model_type", items=MODELS, label="Model:",
                     orientation="horizontal", sendSelectedValue=False)
        gui.spin(box, self, "n_topics", 0, 50, label="Topics (0 = auto):")
        gui.spin(box, self, "max_topics", 2, 50, label="Max topics (auto):")
        gui.spin(box, self, "top_terms", 3, 30, label="Top terms shown:",
                 callback=self._redraw)
        gui.button(box, self, "Run", callback=self._run)

        vb = gui.widgetBox(self.controlArea, "View")
        self.topic_combo = QComboBox()
        self.topic_combo.currentIndexChanged.connect(self._on_topic)
        vb.layout().addWidget(QLabel("Topic:"))
        vb.layout().addWidget(self.topic_combo)
        self.status = QLabel(""); self.status.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status)
        self.controlArea.layout().addStretch(1)

        self.tabs = QTabWidget()
        self.terms_plot = pg.PlotWidget(background="w")
        self.terms_plot.getPlotItem().showGrid(x=False, y=False)
        self.summary_table = QTableWidget()
        self.tabs.addTab(self.terms_plot, "Top terms")
        self.tabs.addTab(self.summary_table, "Summary")
        self.mainArea.layout().addWidget(self.tabs)

    # ------------------------------------------------------------- input
    @Inputs.data
    def set_data(self, data):
        self._df = _table_to_df(data) if data is not None else None
        self._fill_sources()

    def _fill_sources(self):
        self._text_sources = self._source_options()
        self.src_combo.blockSignals(True)
        self.src_combo.clear()
        self.src_combo.addItems(list(self._text_sources.keys()))
        if self.text_source in self._text_sources:
            self.src_combo.setCurrentText(self.text_source)
        elif self._text_sources:
            self.text_source = list(self._text_sources)[0]
        self.src_combo.blockSignals(False)

    def _source_options(self):
        if self._df is None:
            return {}
        cols = list(self._df.columns)

        def find(c):
            for x in c:
                if x in cols:
                    return x
            return None
        ab = find(["Abstract", "AB", "Description", "abstract"])
        ti = find(["Title", "TI", "Document Title", "title"])
        ak = find(["Author Keywords", "Author keywords", "DE"])
        opts = {}
        if ab:
            opts["Abstract"] = ("col", ab)
        if ti:
            opts["Title"] = ("col", ti)
        if ti and ab:
            opts["Title + Abstract"] = ("combine", [ti, ab])
        if ak:
            opts["Author Keywords"] = ("col", ak)
        return opts

    def _resolve_series(self, label):
        src = self._text_sources.get(label)
        if src is None:
            return None
        kind, val = src
        if kind == "col":
            return self._df[val].astype(str)

        def _join(row):
            return " ".join(str(row[c]) for c in val
                            if pd.notna(row[c]) and str(row[c]).lower() != "nan")
        return self._df.apply(_join, axis=1)

    def _set(self, attr, t):
        if t:
            setattr(self, attr, t)

    # ------------------------------------------------------------- run
    def _run(self):
        self.Error.clear(); self.Warning.clear()
        if not HAS_TM:
            self.Error.no_tm(); return
        if self._df is None or self._df.empty:
            return
        series = self._resolve_series(self.text_source)
        if series is None:
            self.Warning.no_text(); return
        work = self._df.copy()
        work["_tm_text"] = series.fillna("").astype(str)
        try:
            nt = self.n_topics if self.n_topics > 0 else None
            doc_topic, topics_df = topic_modeling(
                work, text_column="_tm_text", model_type=MODELS[self.model_type],
                n_topics=nt, max_topics=self.max_topics)
            self._topics_df = topics_df
            try:
                coh = compute_topic_coherence(work, "_tm_text", topics_df)
            except Exception:  # noqa: BLE001
                coh = {}
            summary = get_topic_summary(topics_df, n_terms=self.top_terms)
            srows = []
            for t in topics_df["Topic"].unique():
                srows.append({"Topic": str(t),
                              "Top terms": ", ".join(summary.get(t, [])),
                              "Coherence": round(float(coh.get(t, float("nan"))), 4)
                              if coh else float("nan")})
            summary_df = pd.DataFrame(srows)
            self._fill_topic_combo(topics_df)
            self._fill_summary(summary_df)
            self._redraw()
            # outputs
            self.Outputs.topics.send(_df_to_table(topics_df))
            doc_out = doc_topic.drop(columns=["_tm_text"], errors="ignore")
            self.Outputs.documents.send(_df_to_table(doc_out))
            self.Outputs.summary.send(_df_to_table(summary_df))
            self.status.setText(
                f"{topics_df['Topic'].nunique()} topics, "
                f"{MODELS[self.model_type]} on '{self.text_source}'.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("topic modeling failed")
            self.Error.failed(str(exc))
            self.Outputs.topics.send(None)
            self.Outputs.documents.send(None)
            self.Outputs.summary.send(None)

    def _fill_topic_combo(self, topics_df):
        self.topic_combo.blockSignals(True)
        self.topic_combo.clear()
        for t in topics_df["Topic"].unique():
            self.topic_combo.addItem(str(t), t)
        if 0 <= self.sel_topic < self.topic_combo.count():
            self.topic_combo.setCurrentIndex(self.sel_topic)
        self.topic_combo.blockSignals(False)

    def _on_topic(self, idx):
        self.sel_topic = max(0, idx)
        self._redraw()

    def _redraw(self):
        self.terms_plot.clear()
        if self._topics_df is None or self._topics_df.empty:
            return
        if self.topic_combo.count() == 0:
            return
        topic = self.topic_combo.currentData()
        d = (self._topics_df[self._topics_df["Topic"] == topic]
             .nlargest(self.top_terms, "Weight").iloc[::-1])
        if d.empty:
            return
        vals = d["Weight"].astype(float).tolist()
        labels = [str(x)[:30] for x in d["Term"].tolist()]
        n = len(d)
        ypos = list(range(n))
        ci = list(self._topics_df["Topic"].unique()).index(topic)
        self.terms_plot.addItem(pg.BarGraphItem(
            x0=0, y=ypos, height=0.62, width=vals,
            brush=pg.mkBrush(PALETTE[ci % len(PALETTE)]),
            pen=pg.mkPen("k", width=0.4)))
        self.terms_plot.getAxis("left").setTicks([[(ypos[k], labels[k])
                                                   for k in range(n)]])
        self.terms_plot.setLabel("bottom", "Weight")
        self.terms_plot.setLabel("left", f"Topic {topic}")
        self.terms_plot.setYRange(-0.5, n - 0.5)
        self.terms_plot.setXRange(0, (max(vals) if vals else 1) * 1.05)

    def _fill_summary(self, df):
        self.summary_table.clear()
        self.summary_table.setColumnCount(len(df.columns))
        self.summary_table.setRowCount(len(df))
        self.summary_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.summary_table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
        self.summary_table.resizeColumnsToContents()


if __name__ == "__main__":
    WidgetPreview(OWTopicModeling).run()
