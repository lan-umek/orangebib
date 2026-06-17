# -*- coding: utf-8 -*-
"""
Methodology Classifier Widget
============================
Classify the research methodology of each paper from its abstract/title:
paradigm (quantitative / qualitative / mixed), study designs, data sources,
statistical and qualitative methods, software and sample sizes.

Wraps `biblium.addons.methodology_classifier.analyze_methodology_corpus`.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QGridLayout, QProgressBar,
                              QTabWidget, QWidget, QVBoxLayout, QApplication)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.addons.methodology_classifier import analyze_methodology_corpus
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    analyze_methodology_corpus = None

try:
    from orangebib.widgets.owresearchclassifier import (
        THEORIES, DESIGNS, CONTRIBUTIONS, _compile as _compile_scheme)
    HAS_SCHEMES = True
except Exception:  # noqa: BLE001
    THEORIES = DESIGNS = CONTRIBUTIONS = {}
    _compile_scheme = None
    HAS_SCHEMES = False

logger = logging.getLogger(__name__)

TEXT_CANDIDATES = ["Processed Abstract", "Abstract", "AB",
                   "Processed Title", "Title", "TI"]


def _nice(name) -> str:
    """Human-readable method/paradigm label (no underscores)."""
    return str(name).replace('_', ' ').strip().title()


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
    df = df.copy()
    for c in df.columns:  # flatten list-valued cells
        if df[c].apply(lambda v: isinstance(v, (list, tuple))).any():
            df[c] = df[c].apply(
                lambda v: "; ".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
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


class MethodWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, text_col, year_col):
        super().__init__()
        self._df = df; self._text_col = text_col; self._year_col = year_col

    def run(self):
        try:
            self.progress.emit("Classifying methodology...")
            analysis = analyze_methodology_corpus(
                self._df, text_col=self._text_col, year_col=self._year_col,
                verbose=False)
            res = {
                "per_doc": analysis.get_results_df(),
                "paradigms": analysis.get_paradigm_summary(),
                "methods": analysis.get_method_summary(),
            }
            self.finished.emit(res, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("methodology classification failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWMethodology(OWWidget):
    """Classify research methodology from abstracts."""

    name = "Methodology Classifier"
    description = "Classify paradigm, designs, data sources and methods per paper"
    icon = "icons/methodology.svg"
    priority = 390
    keywords = ["methodology", "method", "paradigm", "qualitative",
                "quantitative", "study design", "statistics"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data (needs Abstract/Title)")

    class Outputs:
        per_document = Output("Per-document", Table, doc="Methodology per paper")
        paradigms = Output("Paradigms", Table, doc="Paradigm distribution")
        methods = Output("Methods", Table, doc="Method distribution")
        selected = Output("Selected Documents", Table, doc="Documents using the selected methods")

    text_col = settings.Setting("")
    use_theories = settings.Setting(True)
    use_designs = settings.Setting(True)
    use_contrib = settings.Setting(False)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium addons are required (biblium>=2.16).")
        compute_error = Msg("Computation error: {}")

    class Information(OWWidget.Information):
        done = Msg("Classified {} papers")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None
        self._method_sets = {}   # nice method -> set of row indices
        self._method_order = []  # nice names sorted desc by frequency
        self._selected_methods = set()

        box = gui.widgetBox(self.controlArea, "Options")
        grid = QGridLayout()
        grid.addWidget(QLabel("Text column:"), 0, 0)
        self.text_combo = QComboBox()
        self.text_combo.currentTextChanged.connect(lambda t: setattr(self, "text_col", t))
        grid.addWidget(self.text_combo, 0, 1)
        box.layout().addLayout(grid)

        if HAS_SCHEMES:
            sbox = gui.widgetBox(self.controlArea, "Also classify (keyword-based)")
            gui.checkBox(sbox, self, "use_theories", "Theories & frameworks")
            gui.checkBox(sbox, self, "use_designs", "Research design")
            gui.checkBox(sbox, self, "use_contrib", "Contribution type")

        self.run_btn = QPushButton("Classify")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.summary_label = QLabel("No data")
        self.summary_label.setWordWrap(True)
        self.mainArea.layout().addWidget(self.summary_label)
        self.view_tabs = QTabWidget()
        self.graph = pg.PlotWidget(background="w")
        self.graph.getPlotItem().showGrid(x=False, y=False, alpha=0.2)
        self.graph.setLabel("bottom", "Documents")
        self.graph.scene().sigMouseClicked.connect(self._on_bar_clicked)
        self.view_tabs.addTab(self.graph, "Methods (click to select)")

        hm_tab = QWidget(); hm_l = QVBoxLayout(hm_tab)
        self.heatmap = pg.PlotWidget(background="w")
        self.heat_img = pg.ImageItem()
        self.heatmap.addItem(self.heat_img)
        self.heatmap.scene().sigMouseMoved.connect(self._on_heat_hover)
        self.heatmap.scene().sigMouseClicked.connect(self._on_heat_clicked)
        hm_l.addWidget(self.heatmap)
        self.view_tabs.addTab(hm_tab, "Overlap (Jaccard)")
        self.mainArea.layout().addWidget(self.view_tabs)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _year_col(self):
        for c in (self._df.columns if self._df is not None else []):
            if str(c).lower() in ("year", "publication year", "py",
                                  "publication_year", "oa_publication_year"):
                return c
        return "Year"

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.text_combo.blockSignals(True)
        self.text_combo.clear()
        if data is None:
            self.text_combo.blockSignals(False)
            self.Error.no_data()
            return
        # offer only the real text fields (+ the combination), not every column
        groups = [("Title", ["Title", "TI", "Document Title"]),
                  ("Abstract", ["Abstract", "Processed Abstract", "AB"]),
                  ("Author Keywords", ["Author Keywords", "Keywords", "DE"]),
                  ("Index Keywords", ["Index Keywords", "Keywords Plus", "ID"])]
        present = [name for name, alts in groups
                   if any(a in self._df.columns for a in alts)]
        cols = (["Title + Abstract + Keywords"] + present) if present else present
        self.text_combo.addItems(cols)
        if self.text_col in cols:
            self.text_combo.setCurrentText(self.text_col)
        elif cols:
            self.text_col = cols[0]
        self.text_combo.blockSignals(False)

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        tc = self.text_combo.currentText()
        if not tc:
            self.Error.compute_error("No text column"); return
        work = self._df
        if tc == "Title + Abstract + Keywords":
            work = self._df.copy()
            groups = [["Title", "TI", "Document Title", "title"],
                      ["Abstract", "Processed Abstract", "AB", "abstract"],
                      ["Author Keywords", "Keywords", "DE"],
                      ["Index Keywords", "Keywords Plus", "ID"]]
            def _clean(v):
                sx = "" if v is None else str(v).strip()
                return "" if sx.lower() in ("nan", "none", "<na>") else sx
            cols_found = []
            for cands in groups:
                c = next((x for x in cands if x in work.columns), None)
                if c:
                    cols_found.append(c)
            if not cols_found:
                self.Error.compute_error(
                    "None of Title/Abstract/Keywords columns found"); return
            work["_method_text"] = work[cols_found].apply(
                lambda r: " ".join(_clean(x) for x in r), axis=1)
            tc = "_method_text"
        elif tc not in self._df.columns:
            # resolve display name (Title/Abstract/...) to an actual column
            alias = {"Title": ["Title", "TI", "Document Title"],
                     "Abstract": ["Abstract", "Processed Abstract", "AB"],
                     "Author Keywords": ["Author Keywords", "Keywords", "DE"],
                     "Index Keywords": ["Index Keywords", "Keywords Plus", "ID"]}.get(tc, [tc])
            real = next((x for x in alias if x in self._df.columns), None)
            if real is None:
                self.Error.compute_error(f"Column for '{tc}' not found"); return
            tc = real
        try:
            self._scheme_texts = work[tc].fillna("").astype(str).tolist()
        except Exception:  # noqa: BLE001
            self._scheme_texts = []
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = MethodWorker(work, tc, self._year_col())
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, res, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or res is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            for o in (self.Outputs.per_document, self.Outputs.paradigms, self.Outputs.methods):
                o.send(None)
            return
        per = res["per_doc"]; para = res["paradigms"]; meth = res["methods"]
        n = len(per) if per is not None else 0
        self.status_label.setText(f"Done — {n} papers")
        self.Information.done(n)

        # Build per-document method membership from the joined method columns.
        self._method_sets = {}
        if per is not None and not per.empty:
            method_cols = [c for c in ("Statistical Methods", "Qualitative Methods")
                           if c in per.columns]
            row_methods = []
            for ri in range(len(per)):
                ms = set()
                for c in method_cols:
                    val = per.iloc[ri][c]
                    if isinstance(val, str) and val.strip():
                        for tok in val.split(";"):
                            tok = tok.strip()
                            if tok:
                                ms.add(_nice(tok))
                row_methods.append(ms)
                for m in ms:
                    self._method_sets.setdefault(m, set()).add(ri)

        # nice paradigm + method names; add binary method columns to per-doc.
        if para is not None and not para.empty and "Paradigm" in para.columns:
            para = para.copy()
            para["Paradigm"] = para["Paradigm"].map(_nice)
        if meth is not None and not meth.empty and "Method" in meth.columns:
            meth = meth.copy()
            meth["Method"] = meth["Method"].map(_nice)
            meth = meth.sort_values("Count", ascending=False).reset_index(drop=True)
        self._method_order = list(meth["Method"]) if (
            meth is not None and not meth.empty) else []

        per_out = per
        if per is not None and not per.empty:
            per_out = per.copy()
            # binary 'Method: X' indicator columns
            for m in self._method_order:
                idxs = self._method_sets.get(m, set())
                per_out[f"Method: {m}"] = [1 if ri in idxs else 0
                                           for ri in range(len(per_out))]
            # overall 'Has methodology' flag: any detected method, or a known
            # (non-unknown) paradigm / a study design
            any_method = set()
            for s2 in self._method_sets.values():
                any_method |= s2
            def _has(ri):
                if ri in any_method:
                    return 1
                par = str(per.iloc[ri].get("paradigm", "")).strip().lower()
                if par and par not in ("", "unknown", "none", "nan"):
                    return 1
                dz = str(per.iloc[ri].get("Study Designs", "")).strip()
                return 1 if dz and dz.lower() not in ("", "nan") else 0
            per_out["Has methodology"] = [_has(ri) for ri in range(len(per_out))]
            # binary indicator per paradigm (Quantitative / Qualitative / Mixed)
            if "paradigm" in per.columns:
                pars = [str(p).strip() for p in per["paradigm"].fillna("")]
                for pv in sorted(set(p for p in pars
                                     if p and p.lower() not in ("unknown", "nan", "none"))):
                    per_out[f"Paradigm: {_nice(pv)}"] = [1 if p == pv else 0 for p in pars]

        # --- keyword-based schemes: theories / research design / contribution ---
        scheme_rows = []   # (label, count) to extend the bar/heatmap
        texts = getattr(self, "_scheme_texts", []) or []
        active_schemes = []
        if HAS_SCHEMES and texts:
            if self.use_theories:
                active_schemes.append(("Theory: ", THEORIES))
            if self.use_designs:
                active_schemes.append(("Design: ", DESIGNS))
            if self.use_contrib:
                active_schemes.append(("Contribution: ", CONTRIBUTIONS))
        for (pre, scheme) in active_schemes:
            for cat, kws in scheme.items():
                pats = _compile_scheme(kws)
                rows = set()
                flags = []
                for ri, t in enumerate(texts):
                    hit = bool(t) and any(pp.search(t) for pp in pats)
                    flags.append(1 if hit else 0)
                    if hit:
                        rows.add(ri)
                if not rows:
                    continue
                label = pre + cat
                if per_out is not None and len(flags) == len(per_out):
                    per_out[label] = flags
                self._method_sets[label] = rows
                scheme_rows.append((label, len(rows)))

        # combined summary (methods + scheme categories) for the bar/heatmap
        if meth is not None and not meth.empty:
            base = meth[["Method", "Count"]].copy()
        else:
            base = pd.DataFrame(columns=["Method", "Count"])
        if scheme_rows:
            extra = pd.DataFrame(scheme_rows, columns=["Method", "Count"])
            combined = pd.concat([base, extra], ignore_index=True)
        else:
            combined = base
        combined = combined.sort_values("Count", ascending=False).reset_index(drop=True)
        self._method_order = list(combined["Method"])
        meth = combined

        self._meth_summary = meth
        self._selected_methods = set()
        if para is not None and not para.empty:
            top = "; ".join(f"{r['Paradigm']}: {r['Count']}" for _, r in para.head(4).iterrows())
            self.summary_label.setText(f"<b>Paradigms</b> — {top}")
        self._render_methods(meth)
        self._render_heatmap()
        self.Outputs.per_document.send(_df_to_table(per_out))
        self.Outputs.paradigms.send(_df_to_table(para))
        self.Outputs.methods.send(_df_to_table(meth))
        self.Outputs.selected.send(None)

    def _render_methods(self, meth):
        self.graph.clear()
        if meth is None or meth.empty:
            return
        m = meth.head(15).reset_index(drop=True)
        ys = list(range(len(m)))
        labels = [str(m.iloc[i]["Method"]) for i in ys]
        brushes = [pg.mkBrush("#e67e22") if labels[i] in self._selected_methods
                   else pg.mkBrush("#4a90d9") for i in ys]
        bar = pg.BarGraphItem(x0=0, y=ys, height=0.6,
                              width=list(pd.to_numeric(m["Count"], errors="coerce").fillna(0)),
                              brushes=brushes)
        self.graph.addItem(bar)
        self.graph.getAxis("left").setTicks(
            [[(i, labels[i]) for i in ys]])
        self.graph.setYRange(-1, len(m))
        self.graph.getViewBox().invertY(True)  # largest frequency on top

    def _render_heatmap(self):
        self.heatmap.clear()
        self.heat_img = pg.ImageItem()
        self.heatmap.addItem(self.heat_img)
        names = self._method_order[:15]
        self._heat_names = names
        n = len(names)
        if n < 2:
            return
        jac = np.zeros((n, n), dtype=float)
        for i in range(n):
            si = self._method_sets.get(names[i], set())
            for j in range(i, n):
                sj = self._method_sets.get(names[j], set())
                union = len(si | sj)
                val = (len(si & sj) / union) if union else 0.0
                jac[i, j] = jac[j, i] = val
        self._heat_jac = jac
        self.heat_img.setImage(jac, levels=(0.0, 1.0))
        try:
            self.heat_img.setColorMap(pg.colormap.get("viridis"))
        except Exception:  # noqa: BLE001
            pass
        ticks = [[(i + 0.5, names[i]) for i in range(n)]]
        self.heatmap.getAxis("bottom").setTicks(ticks)
        self.heatmap.getAxis("left").setTicks(ticks)
        self._heat_cells = set()
        self._heat_hl = []

    def _on_bar_clicked(self, ev):
        if self._meth_summary is None or self._meth_summary.empty:
            return
        m = self._meth_summary.head(15).reset_index(drop=True)
        vb = self.graph.getPlotItem().vb
        p = vb.mapSceneToView(ev.scenePos())
        i = int(round(p.y()))
        if not (0 <= i < len(m)):
            return
        label = str(m.iloc[i]["Method"])
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl:
            self._selected_methods.symmetric_difference_update({label})
        else:
            self._selected_methods = (set() if self._selected_methods == {label}
                                      else {label})
        self._render_methods(self._meth_summary)
        self._send_selected()

    def _draw_heat_highlights(self):
        for it in getattr(self, "_heat_hl", []):
            try:
                self.heatmap.removeItem(it)
            except Exception:  # noqa: BLE001
                pass
        self._heat_hl = []
        for (i, j) in getattr(self, "_heat_cells", set()):
            xs = [i, i + 1, i + 1, i, i]; ys = [j, j, j + 1, j + 1, j]
            it = pg.PlotCurveItem(x=np.array(xs, dtype=float),
                                  y=np.array(ys, dtype=float),
                                  pen=pg.mkPen("#e67e22", width=3))
            it.setZValue(50)
            self.heatmap.addItem(it)
            self._heat_hl.append(it)

    def _on_heat_clicked(self, ev):
        names = getattr(self, "_heat_names", None)
        if not names:
            return
        vb = self.heatmap.getPlotItem().vb
        p = vb.mapSceneToView(ev.scenePos())
        i = int(np.floor(p.x())); j = int(np.floor(p.y()))
        if not (0 <= i < len(names) and 0 <= j < len(names)):
            return
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if not hasattr(self, "_heat_cells"):
            self._heat_cells = set()
        cell = (i, j)
        if ctrl:
            self._heat_cells ^= {cell}
        else:
            self._heat_cells = set() if self._heat_cells == {cell} else {cell}
        self._draw_heat_highlights()
        # output documents that have BOTH methods of each selected cell (union
        # across cells)
        if self._data is None or not self._heat_cells:
            self.Outputs.selected.send(None)
            return
        rows = set()
        for (a, b) in self._heat_cells:
            sa = self._method_sets.get(names[a], set())
            sb = self._method_sets.get(names[b], set())
            rows |= (sa & sb)
        idx = sorted(r for r in rows if r < len(self._data))
        self.Outputs.selected.send(self._data[idx] if idx else None)

    def _send_selected(self):
        if self._data is None or not self._selected_methods:
            self.Outputs.selected.send(None)
            return
        rows = set()
        for m in self._selected_methods:
            rows |= self._method_sets.get(m, set())
        idx = sorted(r for r in rows if r < len(self._data))
        self.Outputs.selected.send(self._data[idx] if idx else None)

    def _on_heat_hover(self, pos):
        names = getattr(self, "_heat_names", None)
        if not names or getattr(self, "_heat_jac", None) is None:
            return
        vb = self.heatmap.getPlotItem().vb
        if not self.heatmap.sceneBoundingRect().contains(pos):
            return
        p = vb.mapSceneToView(pos)
        i = int(np.floor(p.x())); j = int(np.floor(p.y()))
        if 0 <= i < len(names) and 0 <= j < len(names):
            self.heatmap.setToolTip(
                f"{names[i]} ∩ {names[j]}\nJaccard: {self._heat_jac[i, j]:.3f}")

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWMethodology).run()
