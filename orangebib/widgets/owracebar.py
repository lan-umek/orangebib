# -*- coding: utf-8 -*-
"""
RaceBar Widget
=============
Animated bar-chart race of the top entities (keywords, authors, sources, ...)
over time. For each year the top-N entities are ranked by their cumulative (or
per-year) document count or citations, and the bars animate as the ranking
changes.
"""

import logging
from collections import defaultdict
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QTimer
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QPushButton, QSpinBox, QSlider, QHBoxLayout, QGridLayout,
    QCheckBox, QFileDialog, QApplication, QMessageBox,
)

from AnyQt.QtGui import QImage
import pyqtgraph as pg

from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

ENTITY_PATTERNS = ("keyword", "author", "countr", "affiliation",
                   "subject", "field", "institution", "topic",
                   "concept", "sdg", "domain", "source title", "journal",
                   "publication name")
PALETTE = ["#4a90d9", "#e8743b", "#5aa454", "#c0504d", "#8064a2", "#1aa8a8",
           "#d9a441", "#9b59b6", "#7f8c8d", "#2c3e50", "#27ae60", "#e74c3c"]


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


def _split(val):
    s = str(val)
    for sep in ["||", "|", "; ", ";", ", "]:
        if sep in s:
            return [e.strip() for e in s.split(sep) if e.strip()]
    return [s.strip()] if s.strip() else []


class OWRaceBar(OWWidget):
    """Animated bar-chart race of top entities over time."""

    name = "RaceBar"
    description = "Animated bar-chart race of top items over time"
    icon = "icons/racebar.svg"
    priority = 250
    keywords = ["race", "animation", "bar chart race", "temporal", "top",
                "trend", "time"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    column_name = settings.Setting("")
    top_n = settings.Setting(10)
    cumulative = settings.Setting(True)
    metric = settings.Setting("Documents")
    interval = settings.Setting(700)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_year = Msg("Year column not found")
        no_entities = Msg("No entity column with data found")

    METRICS = ["Documents", "Citations"]

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._years: List[int] = []
        self._frame = 0
        self._frames = {}        # year -> list of (entity, value)
        self._colors = {}        # entity -> color

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        self._build_controls()
        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.graph.setLabel("bottom", "Count")
        self.mainArea.layout().addWidget(self.graph)
        self._bar_item = None
        self._labels = []
        self._year_text = pg.TextItem(color=(120, 120, 120), anchor=(1, 1))
        font = self._year_text.textItem.font(); font.setPointSize(28); font.setBold(True)
        self._year_text.textItem.setFont(font)
        self.graph.addItem(self._year_text)

    def _build_controls(self):
        box = gui.widgetBox(self.controlArea, "Race")
        grid = QGridLayout()
        grid.addWidget(QLabel("Item type:"), 0, 0)
        self.col_combo = QComboBox()
        self.col_combo.currentTextChanged.connect(self._on_col_changed)
        grid.addWidget(self.col_combo, 0, 1)

        grid.addWidget(QLabel("Metric:"), 1, 0)
        self.metric_combo = QComboBox(); self.metric_combo.addItems(self.METRICS)
        self.metric_combo.setCurrentText(self.metric)
        self.metric_combo.currentTextChanged.connect(self._on_metric_changed)
        grid.addWidget(self.metric_combo, 1, 1)

        grid.addWidget(QLabel("Top N:"), 2, 0)
        self.topn_spin = QSpinBox(); self.topn_spin.setRange(3, 30)
        self.topn_spin.setValue(self.top_n)
        self.topn_spin.valueChanged.connect(self._on_topn_changed)
        grid.addWidget(self.topn_spin, 2, 1)
        box.layout().addLayout(grid)

        self.cum_cb = QCheckBox("Cumulative")
        self.cum_cb.setChecked(self.cumulative)
        self.cum_cb.toggled.connect(self._on_cum_changed)
        box.layout().addWidget(self.cum_cb)

        pbox = gui.widgetBox(self.controlArea, "Playback")
        row = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        row.addWidget(self.play_btn)
        self.restart_btn = QPushButton("⟲")
        self.restart_btn.clicked.connect(self._restart)
        row.addWidget(self.restart_btn)
        pbox.layout().addLayout(row)
        self.export_btn = QPushButton("⬇ Export animation (GIF)")
        self.export_btn.clicked.connect(self._export_animation)
        pbox.layout().addWidget(self.export_btn)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        pbox.layout().addWidget(self.slider)
        self.year_label = QLabel("")
        pbox.layout().addWidget(self.year_label)

        sp = QHBoxLayout()
        sp.addWidget(QLabel("Speed (ms):"))
        self.speed = QSpinBox(); self.speed.setRange(100, 3000); self.speed.setSingleStep(100)
        self.speed.setValue(self.interval)
        self.speed.valueChanged.connect(lambda v: setattr(self, "interval", v))
        sp.addWidget(self.speed)
        pbox.layout().addLayout(sp)
        self.controlArea.layout().addStretch(1)

    # ---------------------------------------------------------------- input
    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self._timer.stop()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.col_combo.blockSignals(True)
        self.col_combo.clear()
        if data is None:
            self.col_combo.blockSignals(False)
            self.Error.no_data()
            return
        ent = self._entity_columns()
        if not ent:
            self.col_combo.blockSignals(False)
            self.Error.no_entities()
            return
        self.col_combo.addItems(ent)
        if self.column_name in ent:
            self.col_combo.setCurrentText(self.column_name)
        else:
            self.column_name = ent[0]
        self.col_combo.blockSignals(False)
        self._prepare()

    def _entity_columns(self):
        if self._df is None:
            return []
        out = []
        for c in self._df.columns:
            cl = str(c).lower()
            if cl == "year" or cl.startswith("doi"):
                continue
            if any(k in cl for k in ENTITY_PATTERNS):
                out.append(c)
        return out

    def _year_col(self):
        for c in self._df.columns:
            if str(c).lower() in ("year", "publication year", "py",
                                  "publication_year", "oa_publication_year"):
                return c
        return None

    def _cite_col(self):
        for c in ["Cited by", "Times Cited", "cited_by_count", "oa_cited_by_count", "TC"]:
            if c in self._df.columns:
                return c
        return None

    # -------------------------------------------------------------- prepare
    def _prepare(self):
        self.Error.clear()
        if self._df is None or not self.column_name:
            return
        year_col = self._year_col()
        if year_col is None:
            self.Error.no_year()
            return
        col = self.column_name
        metric = self.metric_combo.currentText()
        cite_col = self._cite_col() if metric == "Citations" else None

        df = self._df.copy()
        df["_y"] = pd.to_numeric(df[year_col], errors="coerce")
        df = df.dropna(subset=["_y"])
        df["_y"] = df["_y"].astype(int)
        df = df[(df["_y"] > 1500) & (df["_y"] < 2100)]
        if df.empty:
            self.Error.no_year()
            return

        per_year = defaultdict(lambda: defaultdict(float))
        for _, row in df.iterrows():
            ents = _split(row[col]) if pd.notna(row[col]) else []
            if not ents:
                continue
            val = 1.0
            if cite_col is not None:
                val = pd.to_numeric(pd.Series([row[cite_col]]), errors="coerce").fillna(0).iloc[0]
            for e in ents:
                per_year[int(row["_y"])][e] += val

        years = sorted(per_year.keys())
        self._years = years
        running = defaultdict(float)
        self._frames = {}
        for y in years:
            if self.cumulative:
                for e, v in per_year[y].items():
                    running[e] += v
                snapshot = dict(running)
            else:
                snapshot = dict(per_year[y])
            top = sorted(snapshot.items(), key=lambda kv: kv[1], reverse=True)[:self.top_n]
            self._frames[y] = top

        # stable colour per entity (by overall total)
        totals = defaultdict(float)
        for y in years:
            for e, v in (self._frames[y]):
                totals[e] += v
        for i, (e, _) in enumerate(sorted(totals.items(), key=lambda kv: kv[1], reverse=True)):
            self._colors[e] = PALETTE[i % len(PALETTE)]

        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(years) - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._frame = 0
        self._render()

    # ---------------------------------------------------------- animation
    def _render(self):
        if not self._years:
            return
        year = self._years[self._frame]
        data = self._frames.get(year, [])
        self.graph.clear()
        self.graph.addItem(self._year_text)
        for t in self._labels:
            try:
                self.graph.removeItem(t)
            except Exception:  # noqa: BLE001
                pass
        self._labels = []
        if not data:
            return
        data = list(reversed(data))  # largest on top
        ys = list(range(len(data)))
        widths = [v for _, v in data]
        brushes = [pg.mkBrush(self._colors.get(e, "#888")) for e, _ in data]
        bar = pg.BarGraphItem(x0=0, y=ys, height=0.7, width=widths, brushes=brushes)
        self.graph.addItem(bar)
        maxv = max(widths) if widths else 1
        for yi, (e, v) in zip(ys, data):
            name = pg.TextItem(str(e)[:30], color=(40, 40, 40), anchor=(1, 0.5))
            name.setPos(-maxv * 0.01, yi)
            self.graph.addItem(name); self._labels.append(name)
            val = pg.TextItem(f"{v:,.0f}", color=(90, 90, 90), anchor=(0, 0.5))
            val.setPos(v + maxv * 0.01, yi)
            self.graph.addItem(val); self._labels.append(val)
        self.graph.setYRange(-1, len(data))
        # leave room on the left so entity name labels are not clipped
        self.graph.setXRange(-maxv * 0.34, maxv * 1.18)
        self._year_text.setText(str(year))
        self._year_text.setPos(maxv * 1.15, len(data) - 1)
        self.year_label.setText(f"Year: {year}  ({self._frame + 1}/{len(self._years)})")

    def _export_animation(self):
        """Render every frame and save an animated GIF."""
        if not self._years:
            QMessageBox.information(self, "Export", "Nothing to export yet.")
            return
        try:
            from PIL import Image
        except Exception:  # noqa: BLE001
            QMessageBox.warning(
                self, "Export",
                "Pillow (PIL) is required to export the animation.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export animation", "racebar.gif", "GIF (*.gif)")
        if not path:
            return
        if not path.lower().endswith(".gif"):
            path += ".gif"
        self._timer.stop()
        self.play_btn.setText("▶ Play")
        saved_frame = self._frame
        frames = []
        try:
            for f in range(len(self._years)):
                self._frame = f
                self._render()
                QApplication.processEvents()
                qimg = self.graph.grab().toImage().convertToFormat(
                    QImage.Format_RGBA8888)
                w, h = qimg.width(), qimg.height()
                ptr = qimg.constBits()
                try:
                    ptr.setsize(h * w * 4)
                except Exception:  # noqa: BLE001
                    pass
                img = Image.frombytes("RGBA", (w, h), bytes(ptr))
                frames.append(img.convert("P", palette=Image.ADAPTIVE))
            if frames:
                frames[0].save(path, save_all=True, append_images=frames[1:],
                               duration=int(self.interval), loop=0,
                               optimize=False, disposal=2)
            QMessageBox.information(self, "Export", f"Saved {len(frames)} frames to:\n{path}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("racebar export failed")
            QMessageBox.warning(self, "Export", f"Export failed: {exc}")
        finally:
            self._frame = saved_frame
            self._render()

    def _advance(self):
        if not self._years:
            return
        if self._frame >= len(self._years) - 1:
            self._timer.stop()
            self.play_btn.setText("▶ Play")
            return
        self._frame += 1
        self.slider.blockSignals(True); self.slider.setValue(self._frame); self.slider.blockSignals(False)
        self._render()

    def _toggle_play(self):
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            if self._frame >= len(self._years) - 1:
                self._frame = 0
            self._timer.start(int(self.interval))
            self.play_btn.setText("⏸ Pause")

    def _restart(self):
        self._timer.stop(); self.play_btn.setText("▶ Play")
        self._frame = 0
        self.slider.blockSignals(True); self.slider.setValue(0); self.slider.blockSignals(False)
        self._render()

    def _on_slider(self, v):
        self._frame = v
        self._render()

    def _on_col_changed(self, t):
        self.column_name = t
        self._prepare()

    def _on_metric_changed(self, t):
        self.metric = t
        self._prepare()

    def _on_topn_changed(self, v):
        self.top_n = v
        self._prepare()

    def _on_cum_changed(self, c):
        self.cumulative = c
        self._prepare()

    def onDeleteWidget(self):
        self._timer.stop()
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWRaceBar).run()
