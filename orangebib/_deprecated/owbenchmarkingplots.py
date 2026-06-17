# -*- coding: utf-8 -*-
"""
Benchmarking Plots Widget
=========================
Compare the distribution of a field (publication year, SDG, source, ...) in the
input corpus against a reference:

* **Year-over-year** -- percentage change of yearly production within the corpus.
* **Reference dataset** -- percentage-point difference between the corpus and a
  second (reference) corpus connected to the *Reference Data* input.
* **Uniform** -- difference against a flat/uniform distribution.

Results are shown as a diverging horizontal bar chart (positive = over-represented
in the corpus, negative = under-represented) and exported as a table.
"""

import logging
from typing import List, Dict

import numpy as np
import pandas as pd


from AnyQt.QtWidgets import (QComboBox, QLabel, QGridLayout, QTabWidget,
                             QWidget, QVBoxLayout)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

_SEPARATORS = ["||", "|", "; ", ";", ", "]

# Columns that look like an SDG field (after OpenAlex enrichment or manual).
_SDG_CANDIDATES = ["oa_sdgs", "SDG", "SDGs", "sdg", "sdgs",
                   "Sustainable Development Goals"]


def _split_multi(val) -> List[str]:
    """Split a multi-valued cell into individual tokens."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    for sep in _SEPARATORS:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


class OWBenchmarkingPlots(OWWidget):
    """Benchmark corpus distributions against a reference."""

    name = "Benchmarking Plots"
    description = ("Compare yearly production or SDG/field distributions against "
                   "a reference (year-over-year, reference dataset or uniform)")
    icon = "icons/benchmarking.svg"
    priority = 920
    keywords = ["benchmark", "compare", "sdg", "year", "difference", "reference"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Primary bibliographic data")
        reference = Input("Reference Data", Table, doc="Reference corpus")

    class Outputs:
        differences = Output("Differences", Table, doc="Computed benchmark differences")

    # Settings
    field_name = settings.Setting("")
    mode_index = settings.Setting(1)   # 0 year-over-year, 1 reference, 2 uniform
    top_n = settings.Setting(25)

    oa_filter = settings.Setting("")  # optional OpenAlex filter, e.g. concept/field id
    MODES = ["Year-over-year change", "vs Reference dataset", "vs Uniform",
             "vs OpenAlex global (by year)"]

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_field = Msg("Selected field not found")

    class Warning(OWWidget.Warning):
        no_reference = Msg("Reference mode selected but no Reference Data connected")

    class Information(OWWidget.Information):
        built = Msg("{} categories compared")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._ref_df = None
        self._result = None

        box = gui.widgetBox(self.controlArea, "Benchmark")
        g = QGridLayout()
        g.addWidget(QLabel("Compare by:"), 0, 0)
        self.field_combo = QComboBox()
        self.field_combo.currentTextChanged.connect(self._on_field_changed)
        g.addWidget(self.field_combo, 0, 1)
        g.addWidget(QLabel("Reference:"), 1, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.MODES)
        self.mode_combo.setCurrentIndex(self.mode_index)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        g.addWidget(self.mode_combo, 1, 1)
        box.layout().addLayout(g)
        gui.spin(box, self, "top_n", 3, 100, label="Max categories:",
                 callback=self._recompute)

        info = gui.widgetBox(self.controlArea, "Legend")
        gui.widgetLabel(
            info,
            "Positive (blue) = over-represented in the corpus.\n"
            "Negative (red) = under-represented.\n"
            "Year-over-year shows % change vs the previous year.")
        self.controlArea.layout().addStretch(1)

        self.view_tabs = QTabWidget()
        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=False, y=False, alpha=0.2)
        self.view_tabs.addTab(self.plot, "Field / SDG")
        ytab = QWidget(); yl = QVBoxLayout(ytab)
        self.year_plot = pg.PlotWidget(background="w")
        self.year_plot.showGrid(x=False, y=False, alpha=0.2)
        yl.addWidget(self.year_plot)
        self.view_tabs.addTab(ytab, "Year")
        self.mainArea.layout().addWidget(self.view_tabs)

    # ----------------------------------------------------------------- inputs
    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self._data = data
        self._df = self._table_to_df(data) if data is not None else None
        self._refresh_fields()

    @Inputs.reference
    def set_reference(self, data):
        self._ref_df = self._table_to_df(data) if data is not None else None
        self._recompute()

    @staticmethod
    def _table_to_df(table: Table) -> pd.DataFrame:
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)

    def _candidate_fields(self) -> List[str]:
        if self._df is None:
            return []
        out = []
        if any(c in self._df.columns for c in ("Year", "Period")):
            out.append("Year")
        for c in _SDG_CANDIDATES:
            if c in self._df.columns:
                out.append("SDG")
                break
        # any low-cardinality categorical / list-like column
        for c in self._df.columns:
            cl = str(c).lower()
            if cl in ("year", "period") or c in ("SDG",):
                continue
            if any(k in cl for k in ("source", "country", "type", "keyword",
                                     "author", "field", "subfield", "topic",
                                     "concept", "publisher", "language")):
                out.append(c)
        # de-duplicate preserving order
        seen, res = set(), []
        for c in out:
            if c not in seen:
                seen.add(c); res.append(c)
        return res

    def _refresh_fields(self):
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        if self._df is None:
            self.field_combo.blockSignals(False)
            self.Error.no_data()
            self.plot.clear()
            self.Outputs.differences.send(None)
            return
        fields = self._candidate_fields()
        self.field_combo.addItems(fields)
        if self.field_name in fields:
            self.field_combo.setCurrentText(self.field_name)
        elif fields:
            self.field_name = fields[0]
        self.field_combo.blockSignals(False)
        self._recompute()

    # --------------------------------------------------------------- compute
    def _distribution(self, df: pd.DataFrame, field: str) -> Dict[str, float]:
        """Return a category -> share(%) distribution for *field* in *df*."""
        counts: Dict[str, float] = {}
        if df is None or field is None:
            return counts
        if field == "Year":
            col = "Year" if "Year" in df.columns else (
                "Period" if "Period" in df.columns else None)
            if col is None:
                return counts
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            vals = vals[(vals >= 1500) & (vals <= 2100)].astype(int)
            for y in vals:
                counts[str(y)] = counts.get(str(y), 0) + 1
        else:
            if field == "SDG":
                col = next((c for c in _SDG_CANDIDATES if c in df.columns), None)
            else:
                col = field if field in df.columns else None
            if col is None:
                return counts
            for v in df[col]:
                for tok in _split_multi(v):
                    counts[tok] = counts.get(tok, 0) + 1
        total = sum(counts.values())
        if total <= 0:
            return {}
        return {k: 100.0 * v / total for k, v in counts.items()}

    def _openalex_year_distribution(self, year_lo, year_hi):
        """Fetch the global publication-count distribution per year from
        OpenAlex (optionally filtered) and return it as a share(%) dict."""
        import json
        import urllib.request
        import urllib.parse
        filt = [f"from_publication_date:{int(year_lo)}-01-01",
                f"to_publication_date:{int(year_hi)}-12-31"]
        if self.oa_filter.strip():
            filt.append(self.oa_filter.strip())
        params = {"group_by": "publication_year",
                  "filter": ",".join(filt),
                  "per_page": "1",
                  "mailto": "orange-biblium@example.org"}
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "orange-biblium"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAlex fetch failed: %s", exc)
            return None
        groups = data.get("group_by", [])
        counts = {}
        for g in groups:
            try:
                counts[str(int(g["key"]))] = float(g["count"])
            except Exception:  # noqa: BLE001
                continue
        total = sum(counts.values())
        if total <= 0:
            return None
        return {k: 100.0 * v / total for k, v in counts.items()}

    def _recompute(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        self.plot.clear(); self.year_plot.clear()
        self._result = None
        if self._df is None:
            self.Error.no_data()
            self.Outputs.differences.send(None)
            return
        field = self.field_name
        if not field:
            self.Error.no_field()
            self.Outputs.differences.send(None)
            return

        dist = self._distribution(self._df, field)
        if not dist:
            self.Error.no_field()
            self.Outputs.differences.send(None)
            return

        mode = self.mode_index
        rows = []  # (category, value, corpus%, ref%)
        if mode == 0:  # year-over-year % change (only meaningful for Year)
            keys = sorted(dist.keys(), key=lambda k: float(k) if k.replace(
                "-", "").isdigit() else k)
            prev = None
            for k in keys:
                cur = dist[k]
                if prev is not None and prev > 0:
                    rows.append((k, (cur - prev) / prev * 100.0, cur, prev))
                prev = cur
            ylabel = "% change vs previous"
        elif mode == 1:  # vs reference dataset
            if self._ref_df is None:
                self.Warning.no_reference()
                ref = {k: 0.0 for k in dist}
            else:
                ref = self._distribution(self._ref_df, field)
            allk = set(dist) | set(ref)
            for k in allk:
                rows.append((k, dist.get(k, 0) - ref.get(k, 0),
                             dist.get(k, 0), ref.get(k, 0)))
            ylabel = "percentage-point difference"
        elif mode == 3:  # vs OpenAlex global (by year)
            if field not in ("Year", "Period"):
                self.Warning.no_reference()
                ref = {k: 0.0 for k in dist}
            else:
                yrs = [int(float(k)) for k in dist if str(k).replace(".", "").isdigit()]
                ref = self._openalex_year_distribution(min(yrs), max(yrs)) if yrs else None
                if not ref:
                    self.Warning.no_reference()
                    ref = {k: 0.0 for k in dist}
            allk = set(dist) | set(ref)
            for k in allk:
                rows.append((k, dist.get(k, 0) - ref.get(k, 0),
                             dist.get(k, 0), ref.get(k, 0)))
            ylabel = "pp difference vs OpenAlex"
        else:  # vs uniform
            uni = 100.0 / len(dist)
            for k in dist:
                rows.append((k, dist[k] - uni, dist[k], uni))
            ylabel = "pp difference vs uniform"

        if not rows:
            self.Outputs.differences.send(None)
            return
        if field in ("Year", "Period"):
            # chronological order, vertical bars, own tab
            rows.sort(key=lambda r: float(r[0]) if str(r[0]).replace("-", "").isdigit()
                      else 0)
            self._result = rows
            self._draw_year(rows, ylabel)
            self.view_tabs.setCurrentIndex(1)
        else:
            rows.sort(key=lambda r: abs(r[1]), reverse=True)
            rows = rows[:self.top_n]
            rows.sort(key=lambda r: r[1])      # diverging order
            self._result = rows
            self._draw(rows, ylabel)
            self.view_tabs.setCurrentIndex(0)
        self._send_table(rows)
        self.Information.built(len(rows))

    def _draw_year(self, rows, ylabel):
        self.year_plot.clear()
        xs = list(range(len(rows)))
        vals = [r[1] for r in rows]
        cats = [str(r[0]) for r in rows]
        brushes = [pg.mkBrush("#4a90d9") if v >= 0 else pg.mkBrush("#e74c3c")
                   for v in vals]
        bar = pg.BarGraphItem(x0=xs, width=0.7, height=vals, y0=0, brushes=brushes)
        self.year_plot.addItem(bar)
        self.year_plot.addItem(pg.InfiniteLine(pos=0, angle=0,
                                               pen=pg.mkPen("#888", width=1)))
        step = max(1, len(cats) // 12)
        self.year_plot.getAxis("bottom").setTicks(
            [[(i, cats[i]) for i in xs if i % step == 0]])
        self.year_plot.setLabel("left", ylabel)
        self.year_plot.setLabel("bottom", self.field_name)

    def _draw(self, rows, ylabel):
        ys = list(range(len(rows)))
        vals = [r[1] for r in rows]
        cats = [r[0] for r in rows]
        brushes = [pg.mkBrush("#4a90d9") if v >= 0 else pg.mkBrush("#e74c3c")
                   for v in vals]
        bar = pg.BarGraphItem(x0=0, y=ys, height=0.6, width=vals, brushes=brushes)
        self.plot.addItem(bar)
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=90,
                                          pen=pg.mkPen("#888", width=1)))
        ax = self.plot.getAxis("left")
        ax.setTicks([[(i, str(cats[i])[:28]) for i in ys]])
        self.plot.setLabel("bottom", ylabel)
        self.plot.setLabel("left", self.field_name)

    def _send_table(self, rows):
        domain = Domain(
            [ContinuousVariable("Difference"),
             ContinuousVariable("Corpus %"),
             ContinuousVariable("Reference %")],
            metas=[StringVariable("Category")])
        X = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
        M = np.array([[str(r[0])] for r in rows], dtype=object)
        self.Outputs.differences.send(Table.from_numpy(domain, X, metas=M))

    # ---------------------------------------------------------------- events
    def _on_field_changed(self, t):
        self.field_name = t
        self._recompute()

    def _on_mode_changed(self, i):
        self.mode_index = i
        self._recompute()


if __name__ == "__main__":
    WidgetPreview(OWBenchmarkingPlots).run()
