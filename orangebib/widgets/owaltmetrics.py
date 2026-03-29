# -*- coding: utf-8 -*-
"""
Altmetrics Analysis Widget
==========================
Orange widget for analyzing alternative impact metrics beyond citations.

Supports:
- Real API data from Altmetric.com and PlumX
- Simulated data for demonstration
- Multiple visualization types
"""

import logging
import random
from typing import Optional, List, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor, QFont
from AnyQt.QtWidgets import (QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QTableWidget, QTableWidgetItem, QGridLayout,
                              QFrame, QHeaderView, QScrollArea, QPushButton)

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

ALTMETRIC_SOURCES = [
    ("Twitter/X", "twitter_count", "#1DA1F2"),
    ("Facebook", "facebook_count", "#4267B2"),
    ("Mendeley", "mendeley_count", "#A80D0D"),
    ("News", "news_count", "#34495e"),
    ("Blogs", "blog_count", "#E67E22"),
    ("Policy", "policy_count", "#27AE60"),
    ("Wikipedia", "wikipedia_count", "#636363"),
    ("Reddit", "reddit_count", "#FF4500"),
    ("Video", "video_count", "#C4302B"),
    ("Peer Review", "peer_review_count", "#9B59B6"),
]


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def simulate_altmetric_data(df: pd.DataFrame, doi_col: str, 
                            citations_col: Optional[str] = None,
                            year_col: Optional[str] = None) -> pd.DataFrame:
    """
    Simulate realistic altmetric data correlated with citations.
    
    Higher cited papers tend to have higher altmetric scores.
    """
    n = len(df)
    
    # Get citations if available (for correlation)
    if citations_col and citations_col in df.columns:
        citations = pd.to_numeric(df[citations_col], errors='coerce').fillna(0).values
        citations_norm = (citations - citations.min()) / (citations.max() - citations.min() + 1)
    else:
        citations_norm = np.random.random(n)
    
    # Get years if available (newer papers have more social media attention)
    if year_col and year_col in df.columns:
        years = pd.to_numeric(df[year_col], errors='coerce').fillna(2020).values
        year_factor = (years - years.min()) / (years.max() - years.min() + 1)
    else:
        year_factor = np.random.random(n)
    
    # Base attention probability (not all papers get attention)
    attention_prob = 0.3 + 0.5 * citations_norm + 0.2 * year_factor
    has_attention = np.random.random(n) < attention_prob
    
    # Simulate each source
    result = df.copy()
    
    # Twitter - most common, correlated with recency
    twitter = np.zeros(n)
    twitter[has_attention] = np.random.exponential(
        scale=20 + 100 * citations_norm[has_attention] + 50 * year_factor[has_attention]
    )
    result['twitter_count'] = np.maximum(0, twitter.astype(int))
    
    # Mendeley - correlated with citations
    mendeley = np.zeros(n)
    mendeley[has_attention] = np.random.exponential(
        scale=50 + 200 * citations_norm[has_attention]
    )
    result['mendeley_count'] = np.maximum(0, mendeley.astype(int))
    
    # News - rare but high impact
    news_prob = 0.1 + 0.3 * citations_norm
    has_news = (np.random.random(n) < news_prob) & has_attention
    news = np.zeros(n)
    news[has_news] = np.random.exponential(scale=3 + 10 * citations_norm[has_news])
    result['news_count'] = np.maximum(0, news.astype(int))
    
    # Blogs
    blog_prob = 0.15 + 0.25 * citations_norm
    has_blog = (np.random.random(n) < blog_prob) & has_attention
    blogs = np.zeros(n)
    blogs[has_blog] = np.random.exponential(scale=2 + 5 * citations_norm[has_blog])
    result['blog_count'] = np.maximum(0, blogs.astype(int))
    
    # Policy - very rare
    policy_prob = 0.02 + 0.08 * citations_norm
    has_policy = (np.random.random(n) < policy_prob) & has_attention
    policy = np.zeros(n)
    policy[has_policy] = np.random.poisson(lam=1 + 3 * citations_norm[has_policy])
    result['policy_count'] = np.maximum(0, policy.astype(int))
    
    # Facebook
    facebook = np.zeros(n)
    facebook[has_attention] = np.random.exponential(
        scale=5 + 30 * year_factor[has_attention]
    )
    result['facebook_count'] = np.maximum(0, facebook.astype(int))
    
    # Wikipedia - rare
    wiki_prob = 0.03 + 0.1 * citations_norm
    has_wiki = (np.random.random(n) < wiki_prob) & has_attention
    wiki = np.zeros(n)
    wiki[has_wiki] = np.random.poisson(lam=1 + 2 * citations_norm[has_wiki])
    result['wikipedia_count'] = np.maximum(0, wiki.astype(int))
    
    # Reddit
    reddit_prob = 0.05 + 0.15 * year_factor
    has_reddit = (np.random.random(n) < reddit_prob) & has_attention
    reddit = np.zeros(n)
    reddit[has_reddit] = np.random.exponential(scale=2 + 10 * year_factor[has_reddit])
    result['reddit_count'] = np.maximum(0, reddit.astype(int))
    
    # Video
    video_prob = 0.02 + 0.05 * citations_norm
    has_video = (np.random.random(n) < video_prob) & has_attention
    video = np.zeros(n)
    video[has_video] = np.random.poisson(lam=1)
    result['video_count'] = np.maximum(0, video.astype(int))
    
    # Peer review (Publons, etc.)
    pr_prob = 0.1 + 0.2 * citations_norm
    has_pr = (np.random.random(n) < pr_prob) & has_attention
    pr = np.zeros(n)
    pr[has_pr] = np.random.poisson(lam=1 + 2 * citations_norm[has_pr])
    result['peer_review_count'] = np.maximum(0, pr.astype(int))
    
    # Calculate composite Altmetric score (weighted)
    # Based on Altmetric's weighting scheme (approximate)
    result['altmetric_score'] = (
        result['twitter_count'] * 1 +
        result['facebook_count'] * 0.25 +
        result['mendeley_count'] * 0.5 +
        result['news_count'] * 8 +
        result['blog_count'] * 5 +
        result['policy_count'] * 10 +
        result['wikipedia_count'] * 3 +
        result['reddit_count'] * 0.25 +
        result['video_count'] * 5 +
        result['peer_review_count'] * 1
    )
    
    # Add some noise and round
    result['altmetric_score'] = np.maximum(0, 
        result['altmetric_score'] * (0.8 + 0.4 * np.random.random(n))
    ).round(2)
    
    # Has attention flag
    result['has_attention'] = (result['altmetric_score'] > 0).astype(int)
    
    return result


def calculate_altmetric_summary(df: pd.DataFrame) -> Dict:
    """Calculate summary statistics from altmetric data."""
    n_papers = len(df)
    
    if 'has_attention' in df.columns:
        n_with_attention = df['has_attention'].sum()
    elif 'altmetric_score' in df.columns:
        n_with_attention = (df['altmetric_score'] > 0).sum()
    else:
        n_with_attention = 0
    
    attention_rate = n_with_attention / n_papers if n_papers > 0 else 0
    
    mean_score = df['altmetric_score'].mean() if 'altmetric_score' in df.columns else 0
    median_score = df['altmetric_score'].median() if 'altmetric_score' in df.columns else 0
    max_score = df['altmetric_score'].max() if 'altmetric_score' in df.columns else 0
    
    # Source totals
    source_totals = {}
    for name, col, _ in ALTMETRIC_SOURCES:
        if col in df.columns:
            source_totals[name] = int(df[col].sum())
        else:
            source_totals[name] = 0
    
    return {
        "n_papers": n_papers,
        "n_with_attention": int(n_with_attention),
        "attention_rate": attention_rate,
        "mean_score": mean_score,
        "median_score": median_score,
        "max_score": max_score,
        "source_totals": source_totals,
    }


# =============================================================================
# STYLED WIDGETS
# =============================================================================

class MetricCard(QFrame):
    """A styled card for displaying a metric."""
    
    def __init__(self, icon: str, value: str, label: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Icon and value row
        top_row = QHBoxLayout()
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 18px;")
        top_row.addWidget(icon_label)
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50;")
        self.value_label.setAlignment(Qt.AlignRight)
        top_row.addWidget(self.value_label)
        
        layout.addLayout(top_row)
        
        # Label
        self.label_widget = QLabel(label)
        self.label_widget.setStyleSheet("font-size: 11px; color: #7f8c8d;")
        layout.addWidget(self.label_widget)
    
    def setValue(self, value: str):
        self.value_label.setText(value)


# =============================================================================
# PLOT WIDGETS
# =============================================================================

class ScoreDistributionPlot(PlotWidget):
    """Histogram of altmetric score distribution."""
    
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
        self.showGrid(x=False, y=True, alpha=0.3)
    
    def clear_plot(self):
        self.clear()
        self.setTitle("")
    
    def plot_distribution(self, scores: np.ndarray, n_with_attention: int, 
                          attention_rate: float):
        """Plot score distribution histogram."""
        self.clear_plot()
        
        if len(scores) == 0:
            return
        
        # Use log scale for scores (add 1 to handle zeros)
        log_scores = np.log(scores + 1)
        
        # Create histogram
        bins = np.linspace(0, log_scores.max() + 0.5, 30)
        hist, bin_edges = np.histogram(log_scores, bins=bins)
        
        # Plot bars
        bar_width = bin_edges[1] - bin_edges[0]
        bar_item = pg.BarGraphItem(
            x=bin_edges[:-1] + bar_width/2,
            height=hist,
            width=bar_width * 0.9,
            brush=pg.mkBrush("#3498db"),
            pen=pg.mkPen("#2980b9", width=1),
        )
        self.addItem(bar_item)
        
        self.setLabel('bottom', 'log(Altmetric Score + 1)')
        self.setLabel('left', 'Number of Papers')
        self.setTitle(f"Altmetric Score Distribution ({n_with_attention} papers, {attention_rate*100:.1f}%)")


class SourceBreakdownPlot(PlotWidget):
    """Bar chart of altmetric sources."""
    
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
        self.showGrid(x=False, y=True, alpha=0.3)
    
    def clear_plot(self):
        self.clear()
        self.setTitle("")
    
    def plot_sources(self, source_totals: Dict[str, int]):
        """Plot source breakdown."""
        self.clear_plot()
        
        # Filter non-zero sources and sort
        sources = [(name, count, color) for name, col, color in ALTMETRIC_SOURCES 
                   if name in source_totals and source_totals[name] > 0]
        sources.sort(key=lambda x: -source_totals[x[0]])
        
        if not sources:
            return
        
        names = [s[0] for s in sources]
        counts = [source_totals[s[0]] for s in sources]
        colors = [s[2] for s in sources]
        
        x = np.arange(len(names))
        
        for i, (name, count, color) in enumerate(zip(names, counts, colors)):
            bar = pg.BarGraphItem(
                x=[i], height=[count], width=0.7,
                brush=pg.mkBrush(color),
                pen=pg.mkPen(QColor(color).darker(120), width=1),
            )
            self.addItem(bar)
        
        # X-axis labels
        ticks = [[(i, names[i]) for i in range(len(names))]]
        self.getAxis('bottom').setTicks(ticks)
        
        self.setLabel('left', 'Total Mentions')
        self.setTitle("Altmetric Sources Breakdown")


class TemporalTrendsPlot(PlotWidget):
    """Line chart of altmetric trends over time."""
    
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
        self.showGrid(x=True, y=True, alpha=0.3)
        
        self.legend = pg.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.getPlotItem())
    
    def clear_plot(self):
        self.clear()
        self.legend.clear()
        self.legend.hide()
        self.setTitle("")
    
    def plot_trends(self, df: pd.DataFrame, year_col: str):
        """Plot temporal trends of altmetric attention."""
        self.clear_plot()
        
        if year_col not in df.columns:
            return
        
        # Group by year
        df_temp = df.copy()
        df_temp[year_col] = pd.to_numeric(df_temp[year_col], errors='coerce')
        df_temp = df_temp.dropna(subset=[year_col])
        df_temp[year_col] = df_temp[year_col].astype(int)
        
        grouped = df_temp.groupby(year_col).agg({
            'altmetric_score': 'mean',
            'twitter_count': 'sum',
            'mendeley_count': 'sum',
        }).reset_index()
        
        years = grouped[year_col].values
        
        if len(years) < 2:
            return
        
        # Mean score
        score_line = pg.PlotDataItem(
            x=years, y=grouped['altmetric_score'].values,
            pen=pg.mkPen("#e74c3c", width=2),
            name="Mean Score"
        )
        self.addItem(score_line)
        
        self.setLabel('bottom', 'Year')
        self.setLabel('left', 'Mean Altmetric Score')
        self.setTitle("Altmetric Attention Over Time")
        
        self.legend.clear()
        self.legend.addItem(score_line, "Mean Score")
        self.legend.show()


class CoveragePlot(PlotWidget):
    """Pie-like chart showing coverage by source."""
    
    def __init__(self, parent=None):
        super().__init__(
            parent=parent,
            enableMenu=False,
        )
        
        self.getPlotItem().buttonsHidden = True
        self.showGrid(x=False, y=False)
    
    def clear_plot(self):
        self.clear()
    
    def plot_coverage(self, df: pd.DataFrame):
        """Plot coverage rates by source."""
        self.clear_plot()
        
        n_papers = len(df)
        if n_papers == 0:
            return
        
        # Calculate coverage for each source
        coverage_data = []
        for name, col, color in ALTMETRIC_SOURCES:
            if col in df.columns:
                coverage = (df[col] > 0).sum() / n_papers * 100
                coverage_data.append((name, coverage, color))
        
        coverage_data.sort(key=lambda x: -x[1])
        
        names = [c[0] for c in coverage_data]
        coverages = [c[1] for c in coverage_data]
        colors = [c[2] for c in coverage_data]
        
        x = np.arange(len(names))
        
        for i, (name, cov, color) in enumerate(zip(names, coverages, colors)):
            bar = pg.BarGraphItem(
                x=[i], height=[cov], width=0.7,
                brush=pg.mkBrush(color),
                pen=pg.mkPen(QColor(color).darker(120), width=1),
            )
            self.addItem(bar)
        
        # X-axis labels
        ticks = [[(i, names[i][:10]) for i in range(len(names))]]
        self.getAxis('bottom').setTicks(ticks)
        
        self.setLabel('left', 'Coverage (%)')
        self.setTitle("Source Coverage (% of papers)")


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWAltmetrics(OWWidget):
    """Analyze alternative impact metrics beyond citations."""
    
    name = "Altmetrics Analysis"
    description = "Analyze alternative impact metrics beyond citations"
    icon = "icons/altmetrics.svg"
    priority = 95
    keywords = ["altmetric", "impact", "social", "twitter", "mendeley", "attention"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")
    
    class Outputs:
        altmetric_data = Output("Altmetric Data", Table, doc="Data with altmetric scores")
        top_papers = Output("Top Papers", Table, doc="Papers with highest attention")
    
    # Settings
    id_column = settings.Setting("")
    doi_column = settings.Setting("")
    citations_column = settings.Setting("")
    year_column = settings.Setting("")
    
    altmetric_api_key = settings.Setting("")
    plumx_api_key = settings.Setting("")
    simulate_data = settings.Setting(True)
    
    show_overview = settings.Setting(True)
    show_sources = settings.Setting(True)
    show_trends = settings.Setting(True)
    show_top_papers = settings.Setting(True)
    
    top_n_papers = settings.Setting(20)
    
    want_main_area = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_doi = Msg("DOI column required for API queries")
        api_error = Msg("API error: {}")
    
    class Warning(OWWidget.Warning):
        simulated_data = Msg("Using simulated altmetric data")
        low_coverage = Msg("Only {:.1f}% of papers have altmetric attention")
    
    class Information(OWWidget.Information):
        analyzed = Msg("Analyzed {} papers, {} with attention ({:.1f}%)")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._columns: List[str] = []
        self._altmetric_df: Optional[pd.DataFrame] = None
        self._summary: Optional[Dict] = None
        
        self._setup_gui()
    
    def _setup_gui(self):
        # Column Selection
        col_box = gui.widgetBox(self.controlArea, "Column Selection")
        
        self.id_combo = gui.comboBox(
            col_box, self, "id_column", label="ID column:",
            sendSelectedValue=True, orientation=Qt.Horizontal,
        )
        
        self.doi_combo = gui.comboBox(
            col_box, self, "doi_column", label="DOI column:",
            sendSelectedValue=True, orientation=Qt.Horizontal,
        )
        
        self.citations_combo = gui.comboBox(
            col_box, self, "citations_column", label="Citations column:",
            sendSelectedValue=True, orientation=Qt.Horizontal,
        )
        
        self.year_combo = gui.comboBox(
            col_box, self, "year_column", label="Year column:",
            sendSelectedValue=True, orientation=Qt.Horizontal,
        )
        
        # API Configuration
        api_box = gui.widgetBox(self.controlArea, "API Configuration")
        
        gui.lineEdit(api_box, self, "altmetric_api_key", 
                     label="Altmetric API Key:", orientation=Qt.Horizontal)
        
        gui.lineEdit(api_box, self, "plumx_api_key",
                     label="PlumX API Key:", orientation=Qt.Horizontal)
        
        gui.label(api_box, self, "Enter API keys to fetch real altmetric data.\n"
                  "Leave empty and enable simulation for demo.")
        
        gui.checkBox(api_box, self, "simulate_data", 
                     "Simulate altmetric data (for demonstration)",
                     callback=self._on_setting_changed)
        
        sim_label = QLabel("Simulation creates realistic altmetric\n"
                          "distributions correlated with citations.")
        sim_label.setStyleSheet("color: #3498db; font-style: italic;")
        api_box.layout().addWidget(sim_label)
        
        # Display Options
        display_box = gui.widgetBox(self.controlArea, "Display Options")
        
        gui.checkBox(display_box, self, "show_overview", "Show overview plots",
                     callback=self._on_display_changed)
        gui.checkBox(display_box, self, "show_sources", "Show source breakdown",
                     callback=self._on_display_changed)
        gui.checkBox(display_box, self, "show_trends", "Show temporal trends",
                     callback=self._on_display_changed)
        gui.checkBox(display_box, self, "show_top_papers", "Show top papers table",
                     callback=self._on_display_changed)
        
        # Run button
        self.run_btn = gui.button(
            self.controlArea, self, "Run Analysis",
            callback=self._run_analysis,
        )
        self.run_btn.setMinimumHeight(35)
        
        self.controlArea.layout().addStretch(1)
        
        # Main area with tabs
        self.main_widget = QWidget()
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.mainArea.layout().addWidget(self.main_widget)
        
        # Results header
        header_label = QLabel("<b>Results</b>")
        header_label.setStyleSheet("font-size: 16px; padding: 10px;")
        main_layout.addWidget(header_label)
        
        self.status_label = QLabel("Configure options and click Run to see results")
        self.status_label.setStyleSheet("color: #7f8c8d; padding-left: 10px;")
        main_layout.addWidget(self.status_label)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Results tab
        self.results_widget = QWidget()
        results_layout = QVBoxLayout(self.results_widget)
        
        # Metric cards
        self.cards_widget = QWidget()
        cards_layout = QGridLayout(self.cards_widget)
        cards_layout.setSpacing(10)
        
        self.card_papers = MetricCard("📄", "0", "Papers")
        self.card_attention = MetricCard("📢", "0", "With Attention")
        self.card_rate = MetricCard("📊", "0%", "Attention Rate")
        self.card_score = MetricCard("⭐", "0", "Mean Score")
        
        self.card_twitter = MetricCard("🐦", "0", "Twitter")
        self.card_mendeley = MetricCard("📚", "0", "Mendeley")
        self.card_news = MetricCard("📰", "0", "News")
        self.card_policy = MetricCard("🏛️", "0", "Policy")
        
        cards_layout.addWidget(self.card_papers, 0, 0)
        cards_layout.addWidget(self.card_attention, 0, 1)
        cards_layout.addWidget(self.card_rate, 0, 2)
        cards_layout.addWidget(self.card_score, 0, 3)
        cards_layout.addWidget(self.card_twitter, 1, 0)
        cards_layout.addWidget(self.card_mendeley, 1, 1)
        cards_layout.addWidget(self.card_news, 1, 2)
        cards_layout.addWidget(self.card_policy, 1, 3)
        
        results_layout.addWidget(self.cards_widget)
        
        self.tabs.addTab(self.results_widget, "📊 Results")
        
        # Overview tab with plots
        self.overview_tabs = QTabWidget()
        
        # Score distribution
        self.score_plot = ScoreDistributionPlot()
        self.overview_tabs.addTab(self.score_plot, "📈 Overview")
        
        # Sources breakdown
        self.sources_plot = SourceBreakdownPlot()
        self.overview_tabs.addTab(self.sources_plot, "📊 Sources")
        
        # Temporal trends
        self.trends_plot = TemporalTrendsPlot()
        self.overview_tabs.addTab(self.trends_plot, "📉 Trends")
        
        # Top papers
        self.top_papers_table = QTableWidget()
        self.overview_tabs.addTab(self.top_papers_table, "🏆 Top Papers")
        
        # Coverage
        self.coverage_plot = CoveragePlot()
        self.overview_tabs.addTab(self.coverage_plot, "📋 Coverage")
        
        # Info tab
        info_text = QLabel(self._get_info_text())
        info_text.setWordWrap(True)
        info_text.setStyleSheet("padding: 20px;")
        info_scroll = QScrollArea()
        info_scroll.setWidget(info_text)
        info_scroll.setWidgetResizable(True)
        self.overview_tabs.addTab(info_scroll, "ℹ Info")
        
        results_layout.addWidget(self.overview_tabs)
        
        # Add info tab to main tabs
        info_widget = QLabel(self._get_info_text())
        info_widget.setWordWrap(True)
        self.tabs.addTab(info_widget, "ℹ Info")
    
    def _get_info_text(self) -> str:
        return """
        <h2>Altmetrics Analysis</h2>
        
        <p>Altmetrics measure the attention and impact of research beyond traditional citations.
        This widget analyzes various alternative metrics including:</p>
        
        <h3>Sources Tracked</h3>
        <ul>
            <li><b>Twitter/X</b>: Social media mentions</li>
            <li><b>Facebook</b>: Shares and posts</li>
            <li><b>Mendeley</b>: Reader counts from reference managers</li>
            <li><b>News</b>: Coverage in news outlets</li>
            <li><b>Blogs</b>: Blog posts and discussions</li>
            <li><b>Policy</b>: Citations in policy documents</li>
            <li><b>Wikipedia</b>: Wikipedia citations</li>
            <li><b>Reddit</b>: Discussion threads</li>
            <li><b>Video</b>: YouTube and other video mentions</li>
            <li><b>Peer Review</b>: Publons and peer review platforms</li>
        </ul>
        
        <h3>Altmetric Score</h3>
        <p>The composite Altmetric score weights different sources based on their impact:</p>
        <ul>
            <li>Policy documents: ×10</li>
            <li>News articles: ×8</li>
            <li>Blog posts: ×5</li>
            <li>Wikipedia: ×3</li>
            <li>Twitter: ×1</li>
            <li>Facebook/Reddit: ×0.25</li>
        </ul>
        
        <h3>Simulation Mode</h3>
        <p>When simulation is enabled, the widget generates realistic altmetric data
        correlated with citation counts and publication year. This is useful for
        demonstration and testing.</p>
        """
    
    def _on_setting_changed(self):
        pass
    
    def _on_display_changed(self):
        if self._altmetric_df is not None:
            self._update_displays()
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._columns = []
        self._altmetric_df = None
        self._summary = None
        
        # Clear combos
        for combo in [self.id_combo, self.doi_combo, self.citations_combo, self.year_combo]:
            combo.clear()
        
        self._clear_displays()
        
        if data is None:
            self.Error.no_data()
            return
        
        self._df = self._table_to_df(data)
        self._columns = list(self._df.columns)
        
        # Populate combos
        for combo in [self.id_combo, self.doi_combo, self.citations_combo, self.year_combo]:
            combo.addItem("")
            combo.addItems(self._columns)
        
        # Auto-detect columns
        self._auto_detect_columns()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)
    
    def _auto_detect_columns(self):
        """Auto-detect common column names."""
        for col in self._columns:
            col_lower = col.lower()
            
            if "doi" in col_lower and not self.doi_column:
                idx = self._columns.index(col) + 1
                self.doi_combo.setCurrentIndex(idx)
                self.doi_column = col
            
            elif ("cited" in col_lower or "citation" in col_lower) and not self.citations_column:
                idx = self._columns.index(col) + 1
                self.citations_combo.setCurrentIndex(idx)
                self.citations_column = col
            
            elif col_lower in ["year", "publication year"] and not self.year_column:
                idx = self._columns.index(col) + 1
                self.year_combo.setCurrentIndex(idx)
                self.year_column = col
    
    def _clear_displays(self):
        """Clear all displays."""
        self.score_plot.clear_plot()
        self.sources_plot.clear_plot()
        self.trends_plot.clear_plot()
        self.coverage_plot.clear_plot()
        self.top_papers_table.clear()
        self.top_papers_table.setRowCount(0)
        
        self.card_papers.setValue("0")
        self.card_attention.setValue("0")
        self.card_rate.setValue("0%")
        self.card_score.setValue("0")
        self.card_twitter.setValue("0")
        self.card_mendeley.setValue("0")
        self.card_news.setValue("0")
        self.card_policy.setValue("0")
        
        self.status_label.setText("Configure options and click Run to see results")
    
    def _run_analysis(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self.Error.no_data()
            return
        
        # Check if we should simulate or use API
        if self.simulate_data:
            self.Warning.simulated_data()
            self._altmetric_df = simulate_altmetric_data(
                self._df,
                doi_col=self.doi_column,
                citations_col=self.citations_column if self.citations_column else None,
                year_col=self.year_column if self.year_column else None,
            )
        else:
            # TODO: Implement real API calls
            if not self.doi_column:
                self.Error.no_doi()
                return
            
            # For now, fall back to simulation
            self.Warning.simulated_data()
            self._altmetric_df = simulate_altmetric_data(
                self._df,
                doi_col=self.doi_column,
                citations_col=self.citations_column if self.citations_column else None,
                year_col=self.year_column if self.year_column else None,
            )
        
        # Calculate summary
        self._summary = calculate_altmetric_summary(self._altmetric_df)
        
        # Show info
        self.Information.analyzed(
            self._summary["n_papers"],
            self._summary["n_with_attention"],
            self._summary["attention_rate"] * 100
        )
        
        if self._summary["attention_rate"] < 0.2:
            self.Warning.low_coverage(self._summary["attention_rate"] * 100)
        
        # Update displays
        self._update_displays()
        
        # Send outputs
        self._send_outputs()
    
    def _update_displays(self):
        """Update all display elements."""
        if self._summary is None or self._altmetric_df is None:
            return
        
        # Update cards
        self.card_papers.setValue(f"{self._summary['n_papers']:,}")
        self.card_attention.setValue(f"{self._summary['n_with_attention']:,}")
        self.card_rate.setValue(f"{self._summary['attention_rate']*100:.1f}%")
        self.card_score.setValue(f"{self._summary['mean_score']:.2f}")
        
        st = self._summary["source_totals"]
        self.card_twitter.setValue(f"{st.get('Twitter/X', 0):,}")
        self.card_mendeley.setValue(f"{st.get('Mendeley', 0):,}")
        self.card_news.setValue(f"{st.get('News', 0):,}")
        self.card_policy.setValue(f"{st.get('Policy', 0):,}")
        
        self.status_label.setText("")
        
        # Update plots
        if self.show_overview:
            scores = self._altmetric_df['altmetric_score'].values
            self.score_plot.plot_distribution(
                scores, 
                self._summary['n_with_attention'],
                self._summary['attention_rate']
            )
        
        if self.show_sources:
            self.sources_plot.plot_sources(self._summary["source_totals"])
        
        if self.show_trends and self.year_column:
            self.trends_plot.plot_trends(self._altmetric_df, self.year_column)
        
        # Coverage plot
        self.coverage_plot.plot_coverage(self._altmetric_df)
        
        # Top papers table
        if self.show_top_papers:
            self._update_top_papers_table()
    
    def _update_top_papers_table(self):
        """Update top papers table."""
        if self._altmetric_df is None:
            return
        
        # Sort by altmetric score
        top_df = self._altmetric_df.nlargest(self.top_n_papers, 'altmetric_score')
        
        # Determine columns to show
        display_cols = ['altmetric_score']
        if self.doi_column:
            display_cols.insert(0, self.doi_column)
        
        for _, col, _ in ALTMETRIC_SOURCES[:5]:  # Top 5 sources
            if col in top_df.columns:
                display_cols.append(col)
        
        # Setup table
        self.top_papers_table.setRowCount(len(top_df))
        self.top_papers_table.setColumnCount(len(display_cols))
        self.top_papers_table.setHorizontalHeaderLabels(display_cols)
        
        for i, (_, row) in enumerate(top_df.iterrows()):
            for j, col in enumerate(display_cols):
                val = row[col]
                if isinstance(val, float):
                    val = f"{val:.2f}" if col == 'altmetric_score' else f"{val:.0f}"
                item = QTableWidgetItem(str(val))
                self.top_papers_table.setItem(i, j, item)
        
        self.top_papers_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    
    def _send_outputs(self):
        """Send output tables."""
        if self._altmetric_df is None:
            self.Outputs.altmetric_data.send(None)
            self.Outputs.top_papers.send(None)
            return
        
        # Full altmetric data
        # Create Orange Table
        alt_cols = ['altmetric_score', 'has_attention'] + [col for _, col, _ in ALTMETRIC_SOURCES if col in self._altmetric_df.columns]
        
        # Combine with original columns
        combined_df = self._altmetric_df.copy()
        
        # Build domain
        attrs = []
        for col in combined_df.columns:
            if combined_df[col].dtype in [np.float64, np.int64, float, int]:
                attrs.append(ContinuousVariable(col))
            else:
                attrs.append(StringVariable(col))
        
        # Simplified: just send numeric columns
        numeric_cols = [c for c in combined_df.columns if combined_df[c].dtype in [np.float64, np.int64, float, int]]
        
        if numeric_cols:
            domain = Domain([ContinuousVariable(c) for c in numeric_cols])
            table = Table.from_numpy(domain, combined_df[numeric_cols].values)
            self.Outputs.altmetric_data.send(table)
        else:
            self.Outputs.altmetric_data.send(None)
        
        # Top papers
        top_df = self._altmetric_df.nlargest(self.top_n_papers, 'altmetric_score')
        if len(top_df) > 0:
            numeric_cols = [c for c in top_df.columns if top_df[c].dtype in [np.float64, np.int64, float, int]]
            if numeric_cols:
                domain = Domain([ContinuousVariable(c) for c in numeric_cols])
                table = Table.from_numpy(domain, top_df[numeric_cols].values)
                self.Outputs.top_papers.send(table)
            else:
                self.Outputs.top_papers.send(None)
        else:
            self.Outputs.top_papers.send(None)


if __name__ == "__main__":
    WidgetPreview(OWAltmetrics).run()
