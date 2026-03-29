# -*- coding: utf-8 -*-
"""
Bibliometric Laws Widget
========================
Orange widget for analyzing bibliometric laws (Lotka, Bradford, Zipf, Price, Pareto).

Includes both computation and Qt/pyqtgraph visualization.
"""

import logging
from typing import Optional, List, Tuple, Dict
from collections import Counter
import re

import numpy as np
import pandas as pd
from scipy import stats

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QSplitter, QTabWidget, QTextEdit

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.visualize.utils.plotutils import AxisItem, PlotWidget
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

LAWS = [
    ("Lotka's Law (Author Productivity)", "lotka"),
    ("Bradford's Law (Journal Scatter)", "bradford"),
    ("Zipf's Law (Word Frequency)", "zipf"),
    ("Price's Law (Elite Productivity)", "price"),
    ("Pareto Principle (80/20 Rule)", "pareto"),
]

# Column name patterns for auto-detection
COLUMN_PATTERNS = {
    "authors": ["authors", "author", "creator"],
    "sources": ["source", "journal", "source title", "publication"],
    "affiliations": ["affiliation", "institution", "organization", "university"],
    "countries": ["country", "countries", "author countries"],
    "keywords": ["keyword", "author keyword", "index keyword", "descriptor"],
    "title": ["title", "document title", "article title"],
    "abstract": ["abstract", "description", "summary"],
    "publisher": ["publisher", "publishing"],
}

PLOT_COLORS = {
    "observed": "#3498db",
    "theoretical": "#e74c3c",
    "zones": ["#27ae60", "#f39c12", "#e74c3c"],
}


# =============================================================================
# LAW COMPUTATIONS
# =============================================================================

def compute_lotka(counts: np.ndarray) -> Dict:
    """Compute Lotka's Law: y = C / x^n"""
    freq = Counter(counts)
    x = np.array(sorted(freq.keys()))
    y = np.array([freq[k] for k in x])
    
    total = y.sum()
    y_prop = y / total
    
    # Fit power law
    mask = (x > 0) & (y > 0)
    if mask.sum() > 1:
        log_x = np.log(x[mask])
        log_y = np.log(y[mask])
        slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
        n = -slope
        C = np.exp(intercept)
        r_squared = r_value ** 2
        y_theoretical = C / (x ** n)
        y_theo_prop = y_theoretical / y_theoretical.sum()
    else:
        n, C, r_squared = 2.0, float(y[0]), 0.0
        y_theo_prop = y_prop.copy()
    
    return {
        "x": x, "y_observed": y, "y_observed_prop": y_prop,
        "y_theoretical_prop": y_theo_prop, "n": n, "C": C,
        "r_squared": r_squared, "total_entities": len(counts),
        "total_contributions": int(counts.sum()),
    }


def compute_bradford(counts: np.ndarray, names: np.ndarray) -> Dict:
    """Compute Bradford's Law: Journal scatter across zones."""
    sorted_idx = np.argsort(-counts)
    sorted_counts = counts[sorted_idx]
    sorted_names = names[sorted_idx]
    
    cumsum = np.cumsum(sorted_counts)
    total = cumsum[-1]
    
    zone1_idx = np.searchsorted(cumsum, total / 3)
    zone2_idx = np.searchsorted(cumsum, 2 * total / 3)
    
    zones = {
        "zone1": {"journals": zone1_idx + 1, "articles": int(cumsum[zone1_idx]) if zone1_idx < len(cumsum) else 0},
        "zone2": {"journals": zone2_idx - zone1_idx, "articles": int(cumsum[zone2_idx] - cumsum[zone1_idx]) if zone2_idx < len(cumsum) else 0},
        "zone3": {"journals": len(counts) - zone2_idx - 1, "articles": int(total - cumsum[zone2_idx]) if zone2_idx < len(cumsum) else 0},
    }
    
    if zones["zone1"]["journals"] > 0 and zones["zone2"]["journals"] > 0:
        k1 = zones["zone2"]["journals"] / zones["zone1"]["journals"]
        k2 = zones["zone3"]["journals"] / zones["zone2"]["journals"] if zones["zone2"]["journals"] > 0 else 0
        bradford_multiplier = (k1 + k2) / 2
    else:
        bradford_multiplier = np.nan
    
    ranks = np.arange(1, len(counts) + 1)
    
    return {
        "ranks": ranks, "cumsum": cumsum, "log_ranks": np.log(ranks),
        "sorted_names": sorted_names, "sorted_counts": sorted_counts,
        "zones": zones, "bradford_multiplier": bradford_multiplier,
        "total_sources": len(counts), "total_articles": int(total),
    }


def compute_zipf(word_counts: Dict[str, int]) -> Dict:
    """Compute Zipf's Law: f = C / r^s"""
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    words = [w[0] for w in sorted_words]
    freqs = np.array([w[1] for w in sorted_words])
    ranks = np.arange(1, len(freqs) + 1)
    
    if len(ranks) > 1:
        log_ranks = np.log(ranks)
        log_freqs = np.log(freqs)
        slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_freqs)
        s = -slope
        C = np.exp(intercept)
        r_squared = r_value ** 2
        freqs_theoretical = C / (ranks ** s)
    else:
        s, C, r_squared = 1.0, float(freqs[0]) if len(freqs) > 0 else 1.0, 0.0
        freqs_theoretical = freqs.copy()
    
    return {
        "ranks": ranks, "words": words, "freqs_observed": freqs,
        "freqs_theoretical": freqs_theoretical, "log_ranks": np.log(ranks) if len(ranks) > 0 else np.array([]),
        "log_freqs": np.log(freqs) if len(freqs) > 0 else np.array([]),
        "s": s, "C": C, "r_squared": r_squared,
        "total_words": len(words), "total_occurrences": int(freqs.sum()) if len(freqs) > 0 else 0,
    }


def compute_price(counts: np.ndarray) -> Dict:
    """Compute Price's Law: sqrt(N) authors produce 50% of work."""
    n_total = len(counts)
    n_elite = int(np.ceil(np.sqrt(n_total)))
    
    sorted_counts = np.sort(counts)[::-1]
    total_output = sorted_counts.sum()
    
    elite_output = sorted_counts[:n_elite].sum()
    elite_proportion = elite_output / total_output if total_output > 0 else 0
    
    cumsum = np.cumsum(sorted_counts)
    n_for_50 = np.searchsorted(cumsum, total_output * 0.5) + 1
    
    return {
        "n_total": n_total, "n_elite": n_elite,
        "elite_output": int(elite_output), "total_output": int(total_output),
        "elite_proportion": elite_proportion, "n_for_50": n_for_50,
        "sorted_counts": sorted_counts, "cumsum": cumsum,
        "price_holds": elite_proportion >= 0.5,
    }


def compute_pareto(counts: np.ndarray) -> Dict:
    """Compute Pareto Principle: 20% produce 80% of output."""
    n_total = len(counts)
    n_20 = max(1, int(np.ceil(n_total * 0.2)))
    
    sorted_counts = np.sort(counts)[::-1]
    total_output = sorted_counts.sum()
    
    top_20_output = sorted_counts[:n_20].sum()
    top_20_proportion = top_20_output / total_output if total_output > 0 else 0
    
    cumsum = np.cumsum(sorted_counts)
    n_for_80 = np.searchsorted(cumsum, total_output * 0.8) + 1
    pct_for_80 = n_for_80 / n_total * 100 if n_total > 0 else 0
    
    return {
        "n_total": n_total, "n_20": n_20,
        "top_20_output": int(top_20_output), "total_output": int(total_output),
        "top_20_proportion": top_20_proportion,
        "n_for_80": n_for_80, "pct_for_80": pct_for_80,
        "sorted_counts": sorted_counts, "cumsum": cumsum,
        "pareto_holds": top_20_proportion >= 0.8,
    }


# =============================================================================
# PLOT WIDGET
# =============================================================================

class LawPlotGraph(PlotWidget):
    """Custom plot widget for bibliometric law visualization."""
    
    def __init__(self, parent=None):
        super().__init__(
            parent=parent,
            enableMenu=False,
            axisItems={
                "bottom": AxisItem(orientation="bottom"),
                "left": AxisItem(orientation="left"),
            }
        )
        
        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.showGrid(x=False, y=False, alpha=0.3)  # No grid by default
        
        self.legend = pg.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.getPlotItem())
        self.legend.hide()
        
        self.scatter_observed = None
        self.line_theoretical = None
    
    def set_grid(self, show: bool):
        """Set grid visibility."""
        self.showGrid(x=show, y=show, alpha=0.3)
    
    def clear_plot(self):
        self.clear()
        self.legend.clear()
        self.legend.hide()
        self.setTitle("")
    
    def plot_lotka(self, data: Dict, log_scale: bool, show_theoretical: bool):
        self.clear_plot()
        
        x = data["x"].astype(float)
        y_obs = data["y_observed_prop"]
        y_theo = data["y_theoretical_prop"]
        
        if log_scale:
            self.setLogMode(x=True, y=True)
            self.setLabel('bottom', 'Number of Publications (log)')
            self.setLabel('left', 'Proportion of Authors (log)')
        else:
            self.setLogMode(x=False, y=False)
            self.setLabel('bottom', 'Number of Publications')
            self.setLabel('left', 'Proportion of Authors')
        
        self.scatter_observed = pg.ScatterPlotItem(
            x=x, y=y_obs, pen=pg.mkPen(None),
            brush=pg.mkBrush(PLOT_COLORS["observed"]), size=10, name="Observed"
        )
        self.addItem(self.scatter_observed)
        
        if show_theoretical:
            self.line_theoretical = pg.PlotDataItem(
                x=x, y=y_theo, pen=pg.mkPen(PLOT_COLORS["theoretical"], width=2), name="Theoretical"
            )
            self.addItem(self.line_theoretical)
        
        self.setTitle(f"Lotka's Law (n={data['n']:.2f}, R²={data['r_squared']:.3f})")
        self._update_legend(show_theoretical)
    
    def plot_bradford(self, data: Dict, log_scale: bool, show_zones: bool):
        self.clear_plot()
        
        ranks = data["ranks"]
        cumsum = data["cumsum"]
        
        if log_scale:
            x = data["log_ranks"]
            self.setLabel('bottom', 'log(Rank)')
        else:
            x = ranks
            self.setLabel('bottom', 'Rank')
        
        self.setLogMode(x=False, y=False)
        self.setLabel('left', 'Cumulative Articles')
        
        self.scatter_observed = pg.PlotDataItem(
            x=x, y=cumsum, pen=pg.mkPen(PLOT_COLORS["observed"], width=2), name="Cumulative"
        )
        self.addItem(self.scatter_observed)
        
        k = data['bradford_multiplier']
        k_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
        self.setTitle(f"Bradford's Law (k={k_str})")
        self.legend.hide()
    
    def plot_zipf(self, data: Dict, log_scale: bool, show_theoretical: bool):
        self.clear_plot()
        
        if len(data["ranks"]) == 0:
            return
        
        if log_scale:
            x = data["log_ranks"]
            y_obs = data["log_freqs"]
            y_theo = np.log(data["freqs_theoretical"])
            self.setLabel('bottom', 'log(Rank)')
            self.setLabel('left', 'log(Frequency)')
        else:
            x = data["ranks"]
            y_obs = data["freqs_observed"]
            y_theo = data["freqs_theoretical"]
            self.setLabel('bottom', 'Rank')
            self.setLabel('left', 'Frequency')
        
        self.setLogMode(x=False, y=False)
        
        self.scatter_observed = pg.ScatterPlotItem(
            x=x, y=y_obs, pen=pg.mkPen(None),
            brush=pg.mkBrush(PLOT_COLORS["observed"]), size=5, name="Observed"
        )
        self.addItem(self.scatter_observed)
        
        if show_theoretical:
            self.line_theoretical = pg.PlotDataItem(
                x=x, y=y_theo, pen=pg.mkPen(PLOT_COLORS["theoretical"], width=2), name="Theoretical"
            )
            self.addItem(self.line_theoretical)
        
        self.setTitle(f"Zipf's Law (s={data['s']:.2f}, R²={data['r_squared']:.3f})")
        self._update_legend(show_theoretical)
    
    def plot_price(self, data: Dict):
        self.clear_plot()
        
        n = len(data["sorted_counts"])
        if n == 0:
            return
        
        x = np.arange(1, n + 1) / n * 100
        y = data["cumsum"] / data["total_output"] * 100 if data["total_output"] > 0 else np.zeros(n)
        
        self.setLogMode(x=False, y=False)
        self.setLabel('bottom', '% of Contributors')
        self.setLabel('left', '% of Output')
        
        self.scatter_observed = pg.PlotDataItem(
            x=x, y=y, pen=pg.mkPen(PLOT_COLORS["observed"], width=2), name="Actual"
        )
        self.addItem(self.scatter_observed)
        
        self.line_theoretical = pg.PlotDataItem(
            x=[0, 100], y=[0, 100], pen=pg.mkPen("#95a5a6", width=1, style=Qt.DashLine), name="Equality"
        )
        self.addItem(self.line_theoretical)
        
        elite_pct = data["n_elite"] / data["n_total"] * 100 if data["n_total"] > 0 else 0
        vline = pg.InfiniteLine(pos=elite_pct, angle=90,
                                pen=pg.mkPen(PLOT_COLORS["theoretical"], width=2, style=Qt.DashLine))
        self.addItem(vline)
        
        hline = pg.InfiniteLine(pos=50, angle=0, pen=pg.mkPen("#27ae60", width=1, style=Qt.DotLine))
        self.addItem(hline)
        
        holds = "✓" if data["price_holds"] else "✗"
        self.setTitle(f"Price's Law: √N={data['n_elite']} produce {data['elite_proportion']*100:.1f}% {holds}")
        self._update_legend(True)
    
    def plot_pareto(self, data: Dict):
        self.clear_plot()
        
        n = len(data["sorted_counts"])
        if n == 0:
            return
        
        x = np.arange(1, n + 1) / n * 100
        y = data["cumsum"] / data["total_output"] * 100 if data["total_output"] > 0 else np.zeros(n)
        
        self.setLogMode(x=False, y=False)
        self.setLabel('bottom', '% of Contributors (ranked)')
        self.setLabel('left', '% of Output')
        
        self.scatter_observed = pg.PlotDataItem(
            x=x, y=y, pen=pg.mkPen(PLOT_COLORS["observed"], width=2), name="Actual"
        )
        self.addItem(self.scatter_observed)
        
        self.line_theoretical = pg.PlotDataItem(
            x=[0, 100], y=[0, 100], pen=pg.mkPen("#95a5a6", width=1, style=Qt.DashLine), name="Equality"
        )
        self.addItem(self.line_theoretical)
        
        vline = pg.InfiniteLine(pos=20, angle=90,
                                pen=pg.mkPen(PLOT_COLORS["theoretical"], width=2, style=Qt.DashLine))
        self.addItem(vline)
        
        hline = pg.InfiniteLine(pos=80, angle=0, pen=pg.mkPen("#27ae60", width=2, style=Qt.DashLine))
        self.addItem(hline)
        
        holds = "✓" if data["pareto_holds"] else "✗"
        self.setTitle(f"Pareto: Top 20% produce {data['top_20_proportion']*100:.1f}% {holds}")
        self._update_legend(True)
    
    def _update_legend(self, show: bool):
        if show:
            self.legend.clear()
            if self.scatter_observed:
                self.legend.addItem(self.scatter_observed, "Observed")
            if self.line_theoretical:
                self.legend.addItem(self.line_theoretical, "Theoretical")
            self.legend.show()
        else:
            self.legend.hide()


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWBibliometricLaws(OWWidget):
    """Analyze bibliometric laws (Lotka, Bradford, Zipf, Price, Pareto)."""
    
    name = "Bibliometric Laws"
    description = "Analyze Lotka, Bradford, Zipf, Price, and Pareto bibliometric laws"
    icon = "icons/bibliometric_laws.svg"
    priority = 70
    keywords = ["lotka", "bradford", "zipf", "price", "pareto", "law"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")
    
    class Outputs:
        results = Output("Results", Table, doc="Law analysis results")
        distribution = Output("Distribution", Table, doc="Frequency distribution")
    
    # Settings
    law_index = settings.Setting(0)
    column_name = settings.Setting("")
    show_theoretical = settings.Setting(True)
    use_log_scale = settings.Setting(True)
    show_grid = settings.Setting(False)
    
    want_main_area = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_column = Msg("Please select a column")
        insufficient_data = Msg("Insufficient data for analysis (need at least 3 entities)")
        analysis_error = Msg("Analysis error: {}")
    
    class Warning(OWWidget.Warning):
        few_entities = Msg("Only {} entities found - results may be unreliable")
    
    class Information(OWWidget.Information):
        analyzed = Msg("Analyzed {} entities with {} total contributions")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._columns: List[str] = []
        self._results: Optional[Dict] = None
        
        self._setup_gui()
    
    def _setup_gui(self):
        # Law selection
        law_box = gui.widgetBox(self.controlArea, "Select Law")
        self.law_combo = gui.comboBox(
            law_box, self, "law_index",
            items=[law[0] for law in LAWS],
            callback=self._on_law_changed,
        )
        
        # Column selection
        col_box = gui.widgetBox(self.controlArea, "Data Column")
        self.col_combo = gui.comboBox(
            col_box, self, "column_name",
            sendSelectedValue=True,
            callback=self._on_column_changed,
        )
        
        # Options
        opt_box = gui.widgetBox(self.controlArea, "Options")
        gui.checkBox(opt_box, self, "show_theoretical", "Show theoretical fit",
                     callback=self._update_plot)
        gui.checkBox(opt_box, self, "use_log_scale", "Use log-log scale",
                     callback=self._update_plot)
        gui.checkBox(opt_box, self, "show_grid", "Show grid",
                     callback=self._update_grid)
        
        # Analyze button
        self.analyze_btn = gui.button(
            self.controlArea, self, "Analyze Law",
            callback=self._run_analysis,
        )
        self.analyze_btn.setMinimumHeight(35)
        
        self.controlArea.layout().addStretch(1)
        
        # Main area
        splitter = QSplitter(Qt.Vertical)
        self.mainArea.layout().addWidget(splitter)
        
        self.graph = LawPlotGraph()
        splitter.addWidget(self.graph)
        
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.tabs.addTab(self.results_text, "Results")
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.tabs.addTab(self.info_text, "Info")
        
        splitter.setSizes([400, 200])
        
        self._update_info_text()
    
    def _on_law_changed(self):
        self._update_info_text()
        self._suggest_column()
    
    def _on_column_changed(self):
        pass  # Wait for Analyze button
    
    def _update_grid(self):
        """Update grid visibility."""
        self.graph.set_grid(self.show_grid)
    
    def _suggest_column(self):
        """Suggest appropriate column based on selected law."""
        if not self._columns:
            return
        
        law_key = LAWS[self.law_index][1]
        
        # Patterns to look for based on law
        patterns = {
            "lotka": ["author", "creator"],
            "bradford": ["source", "journal"],
            "zipf": ["title", "abstract", "keyword"],
            "price": ["author", "creator"],
            "pareto": ["author", "source"],
        }
        
        search_patterns = patterns.get(law_key, [])
        
        for col in self._columns:
            col_lower = col.lower()
            for pattern in search_patterns:
                if pattern in col_lower:
                    idx = self._columns.index(col)
                    self.col_combo.setCurrentIndex(idx)
                    self.column_name = col
                    return
    
    def _update_info_text(self):
        law_key = LAWS[self.law_index][1]
        
        info = {
            "lotka": """<h3>Lotka's Law</h3>
<p><b>Formula:</b> y = C / x<sup>n</sup></p>
<p>Describes the frequency of publication by authors. The number of authors making 
<i>n</i> contributions is about 1/n<sup>a</sup> of those making one contribution.</p>
<p><b>Best for:</b> Authors, Institutions, Countries</p>
<p><b>Interpretation:</b> n ≈ 2 indicates classic Lotka's law.</p>""",
            
            "bradford": """<h3>Bradford's Law</h3>
<p><b>Principle:</b> Journals divide into three zones of equal productivity</p>
<p>A small core of journals produces ~1/3 of articles, with subsequent zones 
requiring geometrically more journals.</p>
<p><b>Best for:</b> Sources/Journals, Publishers</p>""",
            
            "zipf": """<h3>Zipf's Law</h3>
<p><b>Formula:</b> f = C / r<sup>s</sup></p>
<p>Word frequency is inversely proportional to rank. The most frequent word occurs 
~twice as often as the second most frequent.</p>
<p><b>Best for:</b> Title words, Abstract words, Keywords</p>
<p><b>Interpretation:</b> s ≈ 1 indicates classic Zipf's law.</p>""",
            
            "price": """<h3>Price's Law</h3>
<p><b>Principle:</b> √N authors produce 50% of publications</p>
<p>Half of scientific literature is produced by the square root of all authors.</p>
<p><b>Best for:</b> Authors, Institutions, Sources</p>""",
            
            "pareto": """<h3>Pareto Principle (80/20 Rule)</h3>
<p><b>Principle:</b> 20% of contributors produce 80% of output</p>
<p>A small proportion of entities account for most of the output.</p>
<p><b>Best for:</b> Authors, Sources, Institutions</p>""",
        }
        
        self.info_text.setHtml(info.get(law_key, ""))
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._columns = []
        self._results = None
        
        self.col_combo.clear()
        self.graph.clear_plot()
        self.results_text.clear()
        
        if data is None:
            self.Error.no_data()
            return
        
        # Convert to DataFrame and get columns
        self._df = self._table_to_df(data)
        self._columns = list(self._df.columns)
        
        # Update column combo
        self.col_combo.addItems(self._columns)
        
        # Suggest appropriate column
        self._suggest_column()
        
        self.results_text.setHtml("<p>Select a column and click <b>Analyze Law</b></p>")
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)
    
    def _run_analysis(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self.Error.no_data()
            return
        
        if not self.column_name or self.column_name not in self._df.columns:
            self.Error.no_column()
            return
        
        law_key = LAWS[self.law_index][1]
        
        try:
            if law_key == "zipf":
                self._analyze_zipf()
            else:
                self._analyze_frequency_law(law_key)
        except Exception as e:
            logger.exception(f"Analysis error: {e}")
            self.Error.analysis_error(str(e))
    
    def _get_entity_counts(self, split_values: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Extract entities and their counts from selected column.
        
        Args:
            split_values: If True, split by separators (for Authors, Keywords).
                         If False, use whole value (for Source, Publisher).
        """
        values = self._df[self.column_name].dropna()
        
        all_entities = []
        for val in values:
            val_str = str(val).strip()
            if not val_str:
                continue
                
            if split_values:
                # Try splitting by common separators
                for sep in [";", "|"]:
                    if sep in val_str:
                        entities = [e.strip() for e in val_str.split(sep) if e.strip()]
                        all_entities.extend(entities)
                        break
                else:
                    # No separator found, use whole value
                    all_entities.append(val_str)
            else:
                # Don't split - use whole value (for sources, publishers)
                all_entities.append(val_str)
        
        if not all_entities:
            return np.array([]), np.array([])
        
        counter = Counter(all_entities)
        names = np.array(list(counter.keys()))
        counts = np.array(list(counter.values()))
        
        return names, counts
    
    def _analyze_frequency_law(self, law_key: str):
        """Analyze Lotka, Bradford, Price, or Pareto."""
        # Bradford doesn't split values (each doc has one source)
        split_values = law_key not in ["bradford"]
        
        names, counts = self._get_entity_counts(split_values=split_values)
        
        if len(counts) < 3:
            self.Error.insufficient_data()
            return
        
        if len(counts) < 10:
            self.Warning.few_entities(len(counts))
        
        total_contributions = int(counts.sum())
        self.Information.analyzed(len(counts), total_contributions)
        
        if law_key == "lotka":
            self._results = compute_lotka(counts)
            self.graph.plot_lotka(self._results, self.use_log_scale, self.show_theoretical)
            self._show_lotka_results()
        
        elif law_key == "bradford":
            self._results = compute_bradford(counts, names)
            self.graph.plot_bradford(self._results, self.use_log_scale, self.show_theoretical)
            self._show_bradford_results()
        
        elif law_key == "price":
            self._results = compute_price(counts)
            self.graph.plot_price(self._results)
            self._show_price_results()
        
        elif law_key == "pareto":
            self._results = compute_pareto(counts)
            self.graph.plot_pareto(self._results)
            self._show_pareto_results()
    
    def _analyze_zipf(self):
        """Analyze Zipf's Law (word frequency)."""
        values = self._df[self.column_name].dropna()
        
        word_counts = Counter()
        for val in values:
            val_str = str(val)
            # Tokenize: extract words with 3+ characters
            words = re.findall(r'\b[a-zA-Z]{3,}\b', val_str.lower())
            word_counts.update(words)
        
        if len(word_counts) < 10:
            self.Error.insufficient_data()
            return
        
        self._results = compute_zipf(dict(word_counts))
        self.Information.analyzed(self._results['total_words'], self._results['total_occurrences'])
        
        self.graph.plot_zipf(self._results, self.use_log_scale, self.show_theoretical)
        self._show_zipf_results()
    
    def _update_plot(self):
        """Update plot with current settings."""
        if self._results is None:
            return
        
        law_key = LAWS[self.law_index][1]
        
        if law_key == "lotka":
            self.graph.plot_lotka(self._results, self.use_log_scale, self.show_theoretical)
        elif law_key == "bradford":
            self.graph.plot_bradford(self._results, self.use_log_scale, self.show_theoretical)
        elif law_key == "zipf":
            self.graph.plot_zipf(self._results, self.use_log_scale, self.show_theoretical)
        elif law_key == "price":
            self.graph.plot_price(self._results)
        elif law_key == "pareto":
            self.graph.plot_pareto(self._results)
    
    def _show_lotka_results(self):
        r = self._results
        interpretation = "Classic Lotka (n≈2)" if 1.5 <= r['n'] <= 2.5 else "Deviation from classic"
        html = f"""<h3>Lotka's Law Results</h3>
<table cellpadding="5">
<tr><td><b>Total entities:</b></td><td>{r['total_entities']:,}</td></tr>
<tr><td><b>Total contributions:</b></td><td>{r['total_contributions']:,}</td></tr>
<tr><td><b>Lotka exponent (n):</b></td><td>{r['n']:.3f}</td></tr>
<tr><td><b>Constant (C):</b></td><td>{r['C']:.3f}</td></tr>
<tr><td><b>R²:</b></td><td>{r['r_squared']:.4f}</td></tr>
</table>
<p><b>Interpretation:</b> {interpretation}</p>"""
        self.results_text.setHtml(html)
    
    def _show_bradford_results(self):
        r = self._results
        z = r["zones"]
        k = r['bradford_multiplier']
        k_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
        html = f"""<h3>Bradford's Law Results</h3>
<table cellpadding="5">
<tr><td><b>Total sources:</b></td><td>{r['total_sources']:,}</td></tr>
<tr><td><b>Total articles:</b></td><td>{r['total_articles']:,}</td></tr>
<tr><td><b>Bradford multiplier (k):</b></td><td>{k_str}</td></tr>
</table>
<h4>Zone Distribution:</h4>
<table border="1" cellpadding="5">
<tr><th>Zone</th><th>Sources</th><th>Articles</th></tr>
<tr><td>Core (Zone 1)</td><td>{z['zone1']['journals']}</td><td>{z['zone1']['articles']}</td></tr>
<tr><td>Zone 2</td><td>{z['zone2']['journals']}</td><td>{z['zone2']['articles']}</td></tr>
<tr><td>Zone 3</td><td>{z['zone3']['journals']}</td><td>{z['zone3']['articles']}</td></tr>
</table>"""
        self.results_text.setHtml(html)
    
    def _show_zipf_results(self):
        r = self._results
        top_words = list(zip(r["words"][:15], r["freqs_observed"][:15]))
        words_html = "".join([f"<tr><td>{i+1}</td><td>{w}</td><td>{int(f):,}</td></tr>" 
                             for i, (w, f) in enumerate(top_words)])
        html = f"""<h3>Zipf's Law Results</h3>
<table cellpadding="5">
<tr><td><b>Unique words:</b></td><td>{r['total_words']:,}</td></tr>
<tr><td><b>Total occurrences:</b></td><td>{r['total_occurrences']:,}</td></tr>
<tr><td><b>Zipf exponent (s):</b></td><td>{r['s']:.3f}</td></tr>
<tr><td><b>R²:</b></td><td>{r['r_squared']:.4f}</td></tr>
</table>
<h4>Top 15 Words:</h4>
<table border="1" cellpadding="3">
<tr><th>Rank</th><th>Word</th><th>Frequency</th></tr>
{words_html}
</table>"""
        self.results_text.setHtml(html)
    
    def _show_price_results(self):
        r = self._results
        holds = "✓ YES" if r["price_holds"] else "✗ NO"
        html = f"""<h3>Price's Law Results</h3>
<table cellpadding="5">
<tr><td><b>Total contributors:</b></td><td>{r['n_total']:,}</td></tr>
<tr><td><b>Elite group (√N):</b></td><td>{r['n_elite']:,}</td></tr>
<tr><td><b>Elite output:</b></td><td>{r['elite_output']:,}</td></tr>
<tr><td><b>Total output:</b></td><td>{r['total_output']:,}</td></tr>
<tr><td><b>Elite proportion:</b></td><td>{r['elite_proportion']*100:.1f}%</td></tr>
<tr><td><b>Contributors for 50%:</b></td><td>{r['n_for_50']:,}</td></tr>
</table>
<p><b>Price's Law holds:</b> {holds}</p>
<p><i>Expected: √{r['n_total']} = {r['n_elite']} produce 50%</i></p>
<p><i>Actual: {r['n_elite']} produce {r['elite_proportion']*100:.1f}%</i></p>"""
        self.results_text.setHtml(html)
    
    def _show_pareto_results(self):
        r = self._results
        holds = "✓ YES" if r["pareto_holds"] else "✗ NO"
        html = f"""<h3>Pareto Principle Results</h3>
<table cellpadding="5">
<tr><td><b>Total contributors:</b></td><td>{r['n_total']:,}</td></tr>
<tr><td><b>Top 20% (n):</b></td><td>{r['n_20']:,}</td></tr>
<tr><td><b>Top 20% output:</b></td><td>{r['top_20_output']:,}</td></tr>
<tr><td><b>Total output:</b></td><td>{r['total_output']:,}</td></tr>
<tr><td><b>Top 20% proportion:</b></td><td>{r['top_20_proportion']*100:.1f}%</td></tr>
<tr><td><b>% needed for 80%:</b></td><td>{r['pct_for_80']:.1f}%</td></tr>
</table>
<p><b>Pareto holds:</b> {holds}</p>
<p><i>Expected: 20% produce 80%</i></p>
<p><i>Actual: 20% produce {r['top_20_proportion']*100:.1f}%</i></p>"""
        self.results_text.setHtml(html)


if __name__ == "__main__":
    WidgetPreview(OWBibliometricLaws).run()
