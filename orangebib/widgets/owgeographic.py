# -*- coding: utf-8 -*-
"""
Geographic Analysis Widget
=========================
Per-country bibliometric metrics (documents, authors, citations, h-index,
international collaboration) extracted from affiliations / country fields,
using `biblium.addons.geographic_analysis.analyze_countries`.

The "Countries" output includes Latitude/Longitude columns so it can be fed
straight into the Orange3-Geo "Map" widget for a world map.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QGridLayout, QProgressBar,
                             QApplication)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.addons.geographic_analysis import analyze_countries
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    analyze_countries = None

logger = logging.getLogger(__name__)

AFF_CANDIDATES = ["Affiliations", "Affiliation", "C1",
                  "oa_institutions", "Authors with affiliations"]
COUNTRY_CANDIDATES = ["Countries of Authors", "Countries", "Country",
                      "oa_institution_countries", "CA Country",
                      "authorships.countries"]
_NONE = "(parse from affiliations)"


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


class GeoWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, aff_col, country_col, cit_col, year_col):
        super().__init__()
        self._df = df; self._aff = aff_col; self._country = country_col
        self._cit = cit_col; self._year = year_col

    def run(self):
        try:
            self.progress.emit("Extracting countries...")
            metrics, _ = analyze_countries(
                self._df, affiliation_col=self._aff,
                country_col=self._country, citations_col=self._cit,
                year_col=self._year, verbose=False)
            rows = []
            for c, m in metrics.items():
                lat, lon = (m.coordinates if getattr(m, "coordinates", None)
                            else (np.nan, np.nan))
                rows.append({
                    "Country": m.country, "ISO": m.iso_code, "Region": m.region,
                    "Documents": m.n_papers, "Authors": m.n_authors,
                    "Citations": m.total_citations,
                    "Mean citations": round(m.mean_citations, 2),
                    "H-index": m.h_index,
                    "Intl collab rate": round(m.international_collab_rate, 3),
                    "Latitude": lat, "Longitude": lon,
                })
            out = pd.DataFrame(rows).sort_values("Documents", ascending=False)
            self.finished.emit(out, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("geographic analysis failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWGeographic(OWWidget):
    """Per-country bibliometric metrics (feeds Orange3-Geo Map)."""

    name = "Geographic Analysis"
    description = "Per-country metrics (documents, citations, collaboration) with map coordinates"
    icon = "icons/geographic.svg"
    priority = 500
    keywords = ["geographic", "country", "map", "world", "collaboration",
                "choropleth", "geo"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with affiliations/countries")

    class Outputs:
        countries = Output("Countries", Table, doc="Per-country metrics (+ Latitude/Longitude)")
        selected = Output("Selected Documents", Table, doc="Documents from the selected countries")

    aff_col = settings.Setting("")
    country_col = settings.Setting(_NONE)
    citations_col = settings.Setting("")
    year_col = settings.Setting("")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("{} countries")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Columns")
        grid = QGridLayout()
        grid.addWidget(QLabel("Country column:"), 0, 0)
        self.country_combo = QComboBox()
        self.country_combo.currentTextChanged.connect(lambda t: setattr(self, "country_col", t))
        grid.addWidget(self.country_combo, 0, 1)
        grid.addWidget(QLabel("Affiliations:"), 1, 0)
        self.aff_combo = QComboBox()
        self.aff_combo.currentTextChanged.connect(lambda t: setattr(self, "aff_col", t))
        grid.addWidget(self.aff_combo, 1, 1)
        grid.addWidget(QLabel("Citations:"), 2, 0)
        self.cit_combo = QComboBox()
        self.cit_combo.currentTextChanged.connect(lambda t: setattr(self, "citations_col", t))
        grid.addWidget(self.cit_combo, 2, 1)
        box.layout().addLayout(grid)

        self.run_btn = QPushButton("Analyze")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        hint = QLabel("<small>Tip: connect the <b>Countries</b> output to the "
                      "Orange3-Geo <b>Map</b> widget for a world map.</small>")
        hint.setWordWrap(True)
        self.controlArea.layout().addWidget(hint)
        self.controlArea.layout().addStretch(1)

        self.summary_label = QLabel("No data"); self.summary_label.setWordWrap(True)
        self.mainArea.layout().addWidget(self.summary_label)
        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.graph.setLabel("bottom", "Documents")
        self.graph.scene().sigMouseClicked.connect(self._on_bar_clicked)
        self.mainArea.layout().addWidget(self.graph)
        self._out = None
        self._selected_countries = set()

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        cols = list(self._df.columns) if self._df is not None else []
        self._fill(self.country_combo, [_NONE] + [c for c in COUNTRY_CANDIDATES if c in cols] + cols, self.country_col)
        self._fill(self.aff_combo, [c for c in AFF_CANDIDATES if c in cols] + cols, self.aff_col)
        self._fill(self.cit_combo, [c for c in ["Cited by", "Times Cited", "cited_by_count", "oa_cited_by_count", "TC"] if c in cols] + cols, self.citations_col)
        if data is None:
            self.Error.no_data()

    @staticmethod
    def _fill(combo, items, current):
        seen, uniq = set(), []
        for it in items:
            if it not in seen:
                seen.add(it); uniq.append(it)
        combo.blockSignals(True); combo.clear(); combo.addItems(uniq)
        if current in uniq:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py",
                                  "publication_year", "oa_publication_year"):
                return c
        return "Year"

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        country = self.country_combo.currentText()
        country = None if country == _NONE else country
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = GeoWorker(self._df, self.aff_combo.currentText(),
                                 country, self.cit_combo.currentText() or "Cited by",
                                 self._year_col())
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, out, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or out is None or out.empty:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "no countries found")
            self.Outputs.countries.send(None)
            return
        self.summary_label.setText(
            f"<b>{len(out)}</b> countries. Top: " +
            ", ".join(f"{r['Country']} ({r['Documents']})"
                      for _, r in out.head(5).iterrows()))
        self._out = out.reset_index(drop=True)
        self._selected_countries = set()
        self._render(out)
        self.status_label.setText(f"Done — {len(out)} countries")
        self.Information.done(len(out))
        self.Outputs.countries.send(_df_to_table(out))
        self.Outputs.selected.send(None)

    def _render(self, out):
        self.graph.clear()
        top = out.head(15).reset_index(drop=True)
        self._top_countries = [str(top.iloc[i]["Country"]) for i in range(len(top))]
        ys = list(range(len(top)))
        brushes = [pg.mkBrush("#e67e22") if self._top_countries[i] in self._selected_countries
                   else pg.mkBrush("#4a90d9") for i in ys]
        bar = pg.BarGraphItem(x0=0, y=ys, height=0.6,
                              width=list(top["Documents"].astype(float)),
                              brushes=brushes)
        self.graph.addItem(bar)
        self.graph.getAxis("left").setTicks(
            [[(i, self._top_countries[i]) for i in ys]])
        self.graph.setYRange(-1, len(top))
        self.graph.getViewBox().invertY(True)  # largest on top

    def _on_bar_clicked(self, ev):
        if not getattr(self, "_top_countries", None):
            return
        vb = self.graph.getPlotItem().vb
        p = vb.mapSceneToView(ev.scenePos())
        i = int(round(p.y()))
        if not (0 <= i < len(self._top_countries)):
            return
        c = self._top_countries[i]
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl:
            self._selected_countries.symmetric_difference_update({c})
        else:
            self._selected_countries = (set() if self._selected_countries == {c}
                                        else {c})
        self._render(self._out)
        self._send_selected()

    def _send_selected(self):
        col = self.country_combo.currentText()
        if (self._data is None or self._df is None or not self._selected_countries
                or col in (None, "", _NONE) or col not in self._df.columns):
            self.Outputs.selected.send(None)
            return
        series = self._df[col].astype(str)
        idx = []
        for ri in range(len(series)):
            cell = series.iloc[ri]
            if any(c in cell for c in self._selected_countries) and ri < len(self._data):
                idx.append(ri)
        self.Outputs.selected.send(self._data[idx] if idx else None)

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWGeographic).run()
