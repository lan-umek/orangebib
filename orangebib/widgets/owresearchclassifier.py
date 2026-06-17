# -*- coding: utf-8 -*-
"""
Research Classifier Widget
=========================
Keyword-based classification of documents against several built-in schemes:

* **Theories & Frameworks** (TAM, RBV, institutional theory, NPM, …)
* **Research Design** (survey, case study, experiment, mixed methods, review, …)
* **Contribution Type** (empirical, theoretical, review, methodological, …)

For each chosen scheme the widget adds binary indicator columns to the data
(e.g. ``Theory: Resource-Based View`` = 0/1), plus an overall ``Matches any``
flag. It also shows a category-frequency bar chart and a Jaccard co-occurrence
heatmap; clicking a bar or a heatmap cell sends the matching documents onward.

Classification can run on Title + Abstract + Keywords (default) or a chosen
text column.
"""

import re
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QGridLayout,
                             QTabWidget, QWidget, QVBoxLayout,
                             QApplication)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# Built-in classification schemes (category -> keyword patterns; * = wildcard)
# =============================================================================

THEORIES = {
    "Technology Acceptance Model (TAM)": ["technology acceptance", "TAM", "perceived usefulness", "perceived ease of use"],
    "UTAUT": ["UTAUT", "unified theory of acceptance"],
    "Theory of Planned Behavior": ["planned behavior", "planned behaviour", "TPB", "theory of reasoned action"],
    "Resource-Based View": ["resource-based", "resource based view", "RBV", "VRIN"],
    "Dynamic Capabilities": ["dynamic capabilit*"],
    "Institutional Theory": ["institutional theor*", "institutional logic*", "isomorphism", "legitimacy theor*"],
    "Stakeholder Theory": ["stakeholder theor*"],
    "Agency Theory": ["agency theor*", "principal-agent", "principal agent"],
    "Transaction Cost Economics": ["transaction cost*"],
    "Diffusion of Innovations": ["diffusion of innovation*", "rogers* diffusion"],
    "New Public Management": ["new public management", "NPM"],
    "New Public Governance": ["new public governance", "public value*", "collaborative governance"],
    "Social Capital Theory": ["social capital"],
    "Contingency Theory": ["contingency theor*"],
    "Complexity Theory": ["complexity theor*", "complex adaptive system*"],
    "Systems Theory": ["systems theor*", "general system* theory"],
    "Actor-Network Theory": ["actor-network", "actor network theor*", "ANT"],
    "Grounded Theory": ["grounded theor*"],
    "Self-Determination Theory": ["self-determination theor*", "SDT"],
    "Game Theory": ["game theor*", "nash equilibrium"],
    "Network Theory": ["network theor*", "social network analysis"],
}

DESIGNS = {
    "Survey": ["survey", "questionnaire", "respondent*"],
    "Experiment": ["experiment*", "randomi*ed controlled", "RCT", "treatment group", "control group"],
    "Quasi-experiment": ["quasi-experiment*", "natural experiment", "difference-in-difference*"],
    "Case Study": ["case stud*"],
    "Longitudinal": ["longitudinal", "panel data", "cohort stud*"],
    "Cross-sectional": ["cross-sectional", "cross sectional"],
    "Ethnography": ["ethnograph*", "participant observation", "field stud*"],
    "Action Research": ["action research"],
    "Mixed Methods": ["mixed method*", "mixed-method*"],
    "Systematic Review": ["systematic review", "PRISMA", "scoping review"],
    "Meta-analysis": ["meta-analysis", "meta analysis", "meta-analytic"],
    "Bibliometric": ["bibliometric*", "scientometric*", "co-citation", "co-word"],
    "Simulation/Modeling": ["simulation", "agent-based model*", "monte carlo", "computational model*"],
    "Content Analysis": ["content analysis", "thematic analysis", "discourse analysis"],
    "Comparative": ["comparative stud*", "cross-country", "cross-national"],
    "Interviews": ["interview*", "focus group*", "semi-structured"],
}

CONTRIBUTIONS = {
    "Empirical": ["empirical", "we find", "results show", "data were collected", "findings"],
    "Theoretical/Conceptual": ["conceptual", "theoretical contribution", "we propose a framework", "propositions", "we theorize"],
    "Review": ["literature review", "systematic review", "review of", "state of the art", "we review"],
    "Methodological": ["we propose a method", "new method", "methodological contribution", "novel approach to measur*", "new measure*"],
    "Case Study": ["case stud*"],
    "Commentary/Editorial": ["editorial", "commentary", "viewpoint", "opinion piece"],
}

SCHEMES = [
    ("Theory: ", "Theories & Frameworks", THEORIES),
    ("Design: ", "Research Design", DESIGNS),
    ("Contribution: ", "Contribution Type", CONTRIBUTIONS),
]

TEXT_SOURCES = ["Title + Abstract + Keywords", "Abstract", "Title",
                "Author Keywords", "Index Keywords"]


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data = {}
    for var in list(table.domain.attributes) + list(table.domain.class_vars) + list(table.domain.metas):
        try:
            col = table.get_column(var)
        except Exception:  # noqa: BLE001
            continue
        if getattr(var, "is_discrete", False):
            vals = var.values
            data[var.name] = [vals[int(v)] if (v == v and 0 <= int(v) < len(vals)) else ""
                              for v in col]
        else:
            data[var.name] = col
    return pd.DataFrame(data)


def _df_to_table(df: Optional[pd.DataFrame]) -> Optional[Table]:
    if df is None or df.empty:
        return None
    attrs, metas, X, M = [], [], [], []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.95 and df[c].dtype != object:
            attrs.append(ContinuousVariable(str(c))); X.append(s.fillna(0).values)
        else:
            metas.append(StringVariable(str(c))); M.append(df[c].astype(str).values)
    n = len(df)
    Xa = np.column_stack(X) if X else np.empty((n, 0))
    Ma = np.column_stack(M) if M else np.empty((n, 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), Xa, metas=Ma)


def _compile(keywords: List[str]):
    pats = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        esc = re.escape(kw).replace(r"\*", r"\w*")
        pats.append(re.compile(r"\b" + esc + r"\b", re.IGNORECASE))
    return pats


class OWResearchClassifier(OWWidget):
    """Classify documents against theory / design / contribution schemes."""

    name = "Research Classifier"
    description = ("Keyword classification of documents by theory/framework, "
                   "research design and contribution type (binary indicators)")
    icon = "icons/research_classifier.svg"
    priority = 395
    keywords = ["theory", "framework", "research design", "contribution",
                "classifier", "methodology", "binary", "keywords"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")

    class Outputs:
        data = Output("Data", Table, doc="Input data + binary indicator columns")
        categories = Output("Categories", Table, doc="Category frequencies")
        selected = Output("Selected Documents", Table)

    text_source = settings.Setting(0)
    use_theory = settings.Setting(True)
    use_design = settings.Setting(True)
    use_contribution = settings.Setting(False)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_scheme = Msg("Select at least one scheme")
        no_text = Msg("No suitable text column found")

    class Information(OWWidget.Information):
        done = Msg("{} documents classified into {} categories")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._cat_sets = {}      # category label -> set of row indices
        self._cat_order = []     # labels sorted desc by frequency
        self._selected = set()   # selected category labels (bar)
        self._heat_cells = set()
        self._heat_hl = []
        self._heat_names = []

        box = gui.widgetBox(self.controlArea, "Schemes")
        gui.checkBox(box, self, "use_theory", "Theories & frameworks",
                     callback=self._compute)
        gui.checkBox(box, self, "use_design", "Research design",
                     callback=self._compute)
        gui.checkBox(box, self, "use_contribution", "Contribution type",
                     callback=self._compute)

        sbox = gui.widgetBox(self.controlArea, "Text")
        grid = QGridLayout()
        grid.addWidget(QLabel("Source:"), 0, 0)
        self.src_combo = QComboBox(); self.src_combo.addItems(TEXT_SOURCES)
        self.src_combo.setCurrentIndex(self.text_source)
        self.src_combo.currentIndexChanged.connect(self._on_src_changed)
        grid.addWidget(self.src_combo, 0, 1)
        sbox.layout().addLayout(grid)

        self.run_btn = QPushButton("Classify"); self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.summary_label = QLabel("No data"); self.summary_label.setWordWrap(True)
        self.mainArea.layout().addWidget(self.summary_label)
        self.view_tabs = QTabWidget()
        self.bar = pg.PlotWidget(background="w")
        self.bar.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.bar.setLabel("bottom", "Documents")
        self.bar.scene().sigMouseClicked.connect(self._on_bar_clicked)
        self.view_tabs.addTab(self.bar, "Categories (click to select)")
        hm = QWidget(); hl = QVBoxLayout(hm)
        self.heatmap = pg.PlotWidget(background="w")
        self.heat_img = pg.ImageItem(); self.heatmap.addItem(self.heat_img)
        self.heatmap.scene().sigMouseClicked.connect(self._on_heat_clicked)
        hl.addWidget(self.heatmap)
        self.view_tabs.addTab(hm, "Overlap (Jaccard)")
        self.mainArea.layout().addWidget(self.view_tabs)

    def _on_src_changed(self, i):
        self.text_source = i
        self._compute()

    # ---------------------------------------------------------------- input
    _TEXT_GROUPS = [
        ("Title", ["Title", "TI", "Document Title"]),
        ("Abstract", ["Abstract", "Processed Abstract", "AB"]),
        ("Author Keywords", ["Author Keywords", "Keywords", "DE"]),
        ("Index Keywords", ["Index Keywords", "Keywords Plus", "ID"]),
    ]

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        if data is None:
            self.Error.no_data()
            return
        # offer only the real text fields that exist (+ the combination)
        present = [name for name, alts in self._TEXT_GROUPS
                   if any(a in self._df.columns for a in alts)]
        opts = ([("Title + Abstract + Keywords")] + present) if present else present
        self.src_combo.blockSignals(True)
        self.src_combo.clear()
        self.src_combo.addItems(opts)
        if 0 <= self.text_source < len(opts):
            self.src_combo.setCurrentIndex(self.text_source)
        else:
            self.text_source = 0
        self.src_combo.blockSignals(False)
        self._compute()

    def _build_text(self):
        df = self._df
        src = self.src_combo.currentText()

        def clean(v):
            sx = "" if v is None else str(v).strip()
            return "" if sx.lower() in ("nan", "none", "<na>") else sx

        if src == "Title + Abstract + Keywords":
            groups = [["Title", "TI", "Document Title"],
                      ["Abstract", "Processed Abstract", "AB"],
                      ["Author Keywords", "Keywords", "DE"],
                      ["Index Keywords", "Keywords Plus", "ID"]]
            cols = []
            for cands in groups:
                c = next((x for x in cands if x in df.columns), None)
                if c:
                    cols.append(c)
            if not cols:
                return None
            return df[cols].apply(lambda r: " ".join(clean(x) for x in r), axis=1)
        # single-column sources
        alts = {"Abstract": ["Abstract", "Processed Abstract", "AB"],
                "Title": ["Title", "TI", "Document Title"],
                "Author Keywords": ["Author Keywords", "Keywords", "DE"],
                "Index Keywords": ["Index Keywords", "Keywords Plus", "ID"]}.get(src, [src])
        c = next((x for x in alts if x in df.columns), None)
        if c is None:
            return None
        return df[c].map(clean)

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        active = [(pre, name, d) for (pre, name, d) in SCHEMES
                  if (self.use_theory and pre == "Theory: ")
                  or (self.use_design and pre == "Design: ")
                  or (self.use_contribution and pre == "Contribution: ")]
        if not active:
            self.Error.no_scheme(); return
        texts = self._build_text()
        if texts is None:
            self.Error.no_text(); return
        texts = texts.fillna("").astype(str).tolist()

        out = self._df.copy()
        self._cat_sets = {}
        for (pre, name, scheme) in active:
            for cat, kws in scheme.items():
                pats = _compile(kws)
                label = pre + cat
                flags = []
                rows = set()
                for ri, t in enumerate(texts):
                    hit = any(p.search(t) for p in pats) if t else False
                    flags.append(1 if hit else 0)
                    if hit:
                        rows.add(ri)
                if rows:
                    out[label] = flags
                    self._cat_sets[label] = rows
        # overall flag
        anyrows = set()
        for s in self._cat_sets.values():
            anyrows |= s
        out["Matches any"] = [1 if ri in anyrows else 0 for ri in range(len(out))]

        self._cat_order = sorted(self._cat_sets, key=lambda k: -len(self._cat_sets[k]))
        self._selected = set(); self._heat_cells = set()
        n_cat = len(self._cat_order)
        self.summary_label.setText(
            f"<b>{len(out)}</b> documents · <b>{n_cat}</b> categories matched · "
            f"{len(anyrows)} documents matched at least one.")
        self._render_bar(); self._render_heatmap()
        self.status_label.setText(f"Done — {n_cat} categories")
        self.Information.done(len(out), n_cat)
        self.Outputs.data.send(_df_to_table(out))
        self.Outputs.categories.send(self._categories_table())
        self.Outputs.selected.send(None)

    def _categories_table(self):
        if not self._cat_order:
            return None
        rows = [{"Category": c, "Documents": len(self._cat_sets[c]),
                 "Percentage": round(100.0 * len(self._cat_sets[c]) / max(1, len(self._df)), 2)}
                for c in self._cat_order]
        return _df_to_table(pd.DataFrame(rows))

    # ---------------------------------------------------------------- bar
    def _render_bar(self):
        self.bar.clear()
        cats = self._cat_order[:25]
        if not cats:
            return
        ys = list(range(len(cats)))
        widths = [len(self._cat_sets[c]) for c in cats]
        brushes = [pg.mkBrush("#e67e22") if c in self._selected
                   else pg.mkBrush("#4a90d9") for c in cats]
        self.bar.addItem(pg.BarGraphItem(x0=0, y=ys, height=0.6, width=widths,
                                         brushes=brushes))
        self.bar.getAxis("left").setTicks([[(i, cats[i]) for i in ys]])
        self.bar.setYRange(-1, len(cats))
        self.bar.getViewBox().invertY(True)
        self._bar_cats = cats

    def _on_bar_clicked(self, ev):
        if not getattr(self, "_bar_cats", None):
            return
        vb = self.bar.getPlotItem().vb
        i = int(round(vb.mapSceneToView(ev.scenePos()).y()))
        if not (0 <= i < len(self._bar_cats)):
            return
        c = self._bar_cats[i]
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl:
            self._selected ^= {c}
        else:
            self._selected = set() if self._selected == {c} else {c}
        self._render_bar()
        rows = set()
        for c in self._selected:
            rows |= self._cat_sets.get(c, set())
        self._send_rows(rows)

    # ---------------------------------------------------------------- heatmap
    def _render_heatmap(self):
        self.heatmap.clear()
        self.heat_img = pg.ImageItem(); self.heatmap.addItem(self.heat_img)
        names = self._cat_order[:20]
        self._heat_names = names
        n = len(names)
        if n < 2:
            return
        jac = np.zeros((n, n))
        for i in range(n):
            si = self._cat_sets[names[i]]
            for j in range(i, n):
                sj = self._cat_sets[names[j]]
                u = len(si | sj)
                v = (len(si & sj) / u) if u else 0.0
                jac[i, j] = jac[j, i] = v
        self.heat_img.setImage(jac, levels=(0.0, 1.0))
        try:
            self.heat_img.setColorMap(pg.colormap.get("viridis"))
        except Exception:  # noqa: BLE001
            pass
        self.heatmap.getAxis("left").setTicks([[(i + 0.5, names[i]) for i in range(n)]])
        self.heatmap.getAxis("bottom").setTicks([[]])
        for i in range(n):
            t = pg.TextItem(names[i], color=(60, 60, 60), anchor=(1.0, 0.5), angle=90)
            t.setPos(i + 0.5, -0.2)
            self.heatmap.addItem(t)
        self._draw_heat_hl()

    def _draw_heat_hl(self):
        for it in self._heat_hl:
            try:
                self.heatmap.removeItem(it)
            except Exception:  # noqa: BLE001
                pass
        self._heat_hl = []
        for (i, j) in self._heat_cells:
            xs = [i, i + 1, i + 1, i, i]; ys = [j, j, j + 1, j + 1, j]
            it = pg.PlotCurveItem(x=np.array(xs, float), y=np.array(ys, float),
                                  pen=pg.mkPen("#e67e22", width=3))
            it.setZValue(50); self.heatmap.addItem(it); self._heat_hl.append(it)

    def _on_heat_clicked(self, ev):
        names = self._heat_names
        if not names:
            return
        vb = self.heatmap.getPlotItem().vb
        p = vb.mapSceneToView(ev.scenePos())
        i = int(np.floor(p.x())); j = int(np.floor(p.y()))
        if not (0 <= i < len(names) and 0 <= j < len(names)):
            return
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        cell = (i, j)
        if ctrl:
            self._heat_cells ^= {cell}
        else:
            self._heat_cells = set() if self._heat_cells == {cell} else {cell}
        self._draw_heat_hl()
        rows = set()
        for (a, b) in self._heat_cells:
            rows |= (self._cat_sets.get(names[a], set()) & self._cat_sets.get(names[b], set()))
        self._send_rows(rows)

    def _send_rows(self, rows):
        if self._data is None or not rows:
            self.Outputs.selected.send(None); return
        idx = sorted(r for r in rows if r < len(self._data))
        self.Outputs.selected.send(self._data[idx] if idx else None)


if __name__ == "__main__":
    WidgetPreview(OWResearchClassifier).run()
