# -*- coding: utf-8 -*-
"""
Discipline Analysis Widget
==========================
Profile a corpus across OpenAlex knowledge levels (domains / fields /
subfields / topics) and track how the top entities evolve over time.
Requires OpenAlex-enriched data with the columns ``oa_domains``,
``oa_fields``, ``oa_subfields`` and/or ``oa_topics``.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from collections import defaultdict

from AnyQt.QtWidgets import QLabel, QTabWidget
from AnyQt.QtGui import QColor
import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from biblium.addons.discipline_analysis import (
        analyze_corpus_disciplines, field_dynamics_over_time)
    HAS_DISC = True
except Exception:  # noqa: BLE001
    HAS_DISC = False
    analyze_corpus_disciplines = field_dynamics_over_time = None

# label -> (column, separator)
LEVELS = [
    ("Domains", ("oa_domains", "; ")),
    ("Fields", ("oa_fields", "; ")),
    ("Subfields", ("oa_subfields", "; ")),
    ("Topics", ("oa_topics", "|")),
]
PALETTE = ["#4a90d9", "#e8743b", "#5aa454", "#c0504d", "#8064a2", "#1aa8a8",
           "#d9a441", "#9b59b6", "#16a085", "#e74c3c", "#2c3e50", "#f39c12"]
COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm",
             "turbo", "Spectral", "RdYlBu"]
COLOR_BY = ["(uniform)", "Average age", "Average year", "% of corpus"]
UNIFORM = "#34618d"


def _cmap_colors(tvals, name):
    """Map 0..1 floats to QColors via a matplotlib colormap (fallback ramp)."""
    try:
        import matplotlib
        try:
            cmap = matplotlib.colormaps[name]
        except Exception:  # noqa: BLE001
            from matplotlib import cm
            cmap = cm.get_cmap(name)
        out = []
        for t in tvals:
            r, g, b, _a = cmap(float(max(0.0, min(1.0, t))))
            out.append(QColor(int(r * 255), int(g * 255), int(b * 255)))
        return out
    except Exception:  # noqa: BLE001
        return [QColor(UNIFORM) for _ in tvals]


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
    df = df.reset_index(drop=False) if df.index.name else df
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


class OWDiscipline(OWWidget):
    """Discipline / field profile and dynamics from OpenAlex levels."""

    name = "Discipline Analysis"
    description = ("Profile a corpus across OpenAlex domains / fields / subfields "
                   "/ topics and track top entities over time")
    icon = "icons/discipline.svg"
    priority = 510
    keywords = ["discipline", "field", "subfield", "domain", "topic", "openalex",
                "interdisciplinarity", "knowledge"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        profile = Output("Profile", Table, doc="Counts & % per entity")
        dynamics = Output("Dynamics", Table, doc="Per-year matrix of top entities")

    level = settings.Setting(1)          # default Fields
    top_n = settings.Setting(15)
    dyn_value = settings.Setting(0)      # 0 share %, 1 raw counts
    color_by = settings.Setting(1)       # default Average age
    colormap = settings.Setting("viridis")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_disc = Msg("biblium discipline_analysis module not available")

    class Warning(OWWidget.Warning):
        no_columns = Msg("No OpenAlex level columns (oa_domains/oa_fields/"
                         "oa_subfields/oa_topics). Run OpenAlex Enrichment first.")
        no_level = Msg("Column '{}' not found for the chosen level.")

    def __init__(self):
        super().__init__()
        self._df = None
        if not HAS_DISC:
            self.Error.no_disc()

        box = gui.widgetBox(self.controlArea, "Level")
        gui.comboBox(box, self, "level", items=[n for n, _ in LEVELS],
                     callback=self._replot, sendSelectedValue=False)
        gui.spin(box, self, "top_n", 3, 50, label="Top N:", callback=self._replot)
        cb = gui.widgetBox(self.controlArea, "Bar colour")
        gui.comboBox(cb, self, "color_by", items=COLOR_BY, label="Colour by:",
                     orientation="horizontal", sendSelectedValue=False,
                     callback=self._replot)
        gui.comboBox(cb, self, "colormap", items=COLORMAPS, label="Colormap:",
                     orientation="horizontal", sendSelectedValue=False,
                     callback=self._replot)
        gui.comboBox(self.controlArea, self, "dyn_value",
                     items=["Share (%)", "Raw counts"], label="Dynamics:",
                     orientation="horizontal", callback=self._replot,
                     sendSelectedValue=False)
        self.status = QLabel(""); self.status.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status)
        self.controlArea.layout().addStretch(1)

        self.tabs = QTabWidget()
        self.profile_plot = pg.PlotWidget(background="w")
        self.profile_plot.getPlotItem().showGrid(x=False, y=False)
        self.dyn_plot = pg.PlotWidget(background="w")
        self.dyn_plot.getPlotItem().showGrid(x=False, y=False)
        self.tabs.addTab(self.profile_plot, "Profile")
        self.tabs.addTab(self.dyn_plot, "Dynamics")
        self.mainArea.layout().addWidget(self.tabs)

    # ------------------------------------------------------------- input
    @Inputs.data
    def set_data(self, data):
        self._df = _table_to_df(data) if data is not None else None
        self._replot()

    def _year_col(self):
        for c in ("Year", "oa_publication_year", "Publication Year", "PY", "year"):
            if self._df is not None and c in self._df.columns:
                return c
        return None

    # ------------------------------------------------------------- plot
    def _replot(self):
        self.Warning.clear()
        self.profile_plot.clear(); self.dyn_plot.clear()
        if not HAS_DISC or self._df is None or self._df.empty:
            self.Outputs.profile.send(None); self.Outputs.dynamics.send(None)
            return
        present = [c for c, _ in [lv[1] for lv in LEVELS] if c in self._df.columns]
        if not present:
            self.Warning.no_columns()
            self.Outputs.profile.send(None); self.Outputs.dynamics.send(None)
            return
        col, sep = LEVELS[self.level][1]
        if col not in self._df.columns:
            self.Warning.no_level(col)
            self.Outputs.profile.send(None); self.Outputs.dynamics.send(None)
            return

        profiles = analyze_corpus_disciplines(self._df, columns_seps=((col, sep),))
        prof = profiles.get(col)
        self._draw_profile(prof, col)

        dyn_out = None
        ycol = self._year_col()
        if ycol is not None:
            mat_raw, mat_share = field_dynamics_over_time(
                self._df, year_col=ycol, multivalue_col=col, sep=sep,
                top_n=min(self.top_n, 10))
            mat = mat_share if self.dyn_value == 0 else mat_raw
            self._draw_dynamics(mat, col)
            dyn_out = (mat.reset_index().rename(columns={"index": "Year"})
                       if mat is not None and not mat.empty else None)
        else:
            self.dyn_plot.addItem(pg.TextItem("No year column found", color="k"))

        self.Outputs.profile.send(_df_to_table(prof))
        self.Outputs.dynamics.send(_df_to_table(dyn_out))
        npapers = int(prof["n_papers"].sum()) if prof is not None and not prof.empty else 0
        note = getattr(self, "_color_note", "")
        self.status.setText(
            f"{LEVELS[self.level][0]}: {0 if prof is None else len(prof)} entities "
            f"({npapers} assignments)." + (f"  {note}" if note else ""))

    def _entity_mean_year(self, col, sep):
        """Mean publication year per entity of the chosen level."""
        ycol = self._year_col()
        if ycol is None:
            return {}
        work = self._df[[col, ycol]].dropna(subset=[col])
        years = pd.to_numeric(work[ycol], errors="coerce")
        acc = defaultdict(list)
        for val, yr in zip(work[col], years):
            if pd.isna(yr):
                continue
            for e in str(val).split(sep):
                e = e.strip()
                if e:
                    acc[e].append(float(yr))
        return {e: (sum(v) / len(v)) for e, v in acc.items() if v}

    def _bar_colors(self, d, col, sep):
        """Return (brushes, note) for the profile bars based on color_by."""
        n = len(d)
        ents = [str(x) for x in d[col].tolist()]
        if self.color_by == 0:                       # uniform
            return [pg.mkBrush(QColor(UNIFORM))] * n, ""
        if self.color_by == 3:                       # % of corpus
            metric = d["pct_of_corpus"].astype(float).tolist()
            unit = "% of corpus"
        else:                                         # average age / year
            means = self._entity_mean_year(col, sep)
            if not means:                             # no year info -> uniform
                return [pg.mkBrush(QColor(UNIFORM))] * n, "no year column - uniform"
            yr = [means.get(e, float("nan")) for e in ents]
            if self.color_by == 1:                    # average age
                ref = pd.Timestamp.now().year
                metric = [ref - y if y == y else float("nan") for y in yr]
                unit = "avg age (years)"
            else:                                     # average year
                metric = yr
                unit = "avg year"
        vals = [m for m in metric if m == m]
        if not vals:
            return [pg.mkBrush(QColor(UNIFORM))] * n, "uniform"
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        tvals = [((m - lo) / rng) if m == m else 0.0 for m in metric]
        colors = _cmap_colors(tvals, self.colormap)
        note = f"colour = {unit} ({lo:.0f}–{hi:.0f})"
        return [pg.mkBrush(c) for c in colors], note

    def _draw_profile(self, prof, col):
        if prof is None or prof.empty:
            return
        sep = LEVELS[self.level][1][1]
        d = prof.head(self.top_n)
        vals = d["n_papers"].astype(float).tolist()
        labels = [str(x)[:40] for x in d[col].tolist()]
        n = len(d)
        ypos = [n - 1 - k for k in range(n)]
        brushes, note = self._bar_colors(d, col, sep)
        self._color_note = note
        self.profile_plot.addItem(pg.BarGraphItem(
            x0=0, y=ypos, height=0.62, width=vals, brushes=brushes,
            pen=pg.mkPen("k", width=0.4)))
        self.profile_plot.getAxis("left").setTicks([[(ypos[k], labels[k])
                                                     for k in range(n)]])
        self.profile_plot.setLabel("bottom", "Papers")
        self.profile_plot.setLabel("left", LEVELS[self.level][0])
        self.profile_plot.setYRange(-0.5, n - 0.5)
        self.profile_plot.setXRange(0, (max(vals) if vals else 1) * 1.05)

    def _draw_dynamics(self, mat, col):
        if mat is None or mat.empty:
            self.dyn_plot.addItem(pg.TextItem("No dynamics", color="k"))
            return
        years = list(mat.index)
        try:
            self.dyn_plot.addLegend(offset=(10, 10))
        except Exception:  # noqa: BLE001
            pass
        for k, ent in enumerate(mat.columns):
            y = mat[ent].astype(float).tolist()
            pen = pg.mkPen(PALETTE[k % len(PALETTE)], width=2)
            self.dyn_plot.plot(years, y, pen=pen, name=str(ent)[:30],
                               symbol="o", symbolSize=5,
                               symbolBrush=PALETTE[k % len(PALETTE)])
        self.dyn_plot.setLabel("bottom", "Year")
        self.dyn_plot.setLabel("left", "Share (%)" if self.dyn_value == 0 else "Papers")


if __name__ == "__main__":
    WidgetPreview(OWDiscipline).run()
