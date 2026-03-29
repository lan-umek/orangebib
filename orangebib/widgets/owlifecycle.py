# -*- coding: utf-8 -*-
"""
Life Cycle Analysis Widget
==========================
Orange widget for analyzing the life cycle of scientific production using
logistic growth model.

Fits a logistic curve to cumulative publication data and provides:
- Model parameters (K, Tm, r)
- Fit quality metrics (R², RMSE, AIC, BIC)
- Milestone years (10%, 50%, 90%, 99% of saturation)
- Forecasts for future years
"""

import logging
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor, QFont
from AnyQt.QtWidgets import (QSplitter, QTabWidget, QTextEdit, QVBoxLayout, 
                              QHBoxLayout, QWidget, QLabel, QFrame, QGridLayout)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.visualize.utils.plotutils import AxisItem, PlotWidget
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# LOGISTIC GROWTH MODEL
# =============================================================================

def logistic_function(t, K, r, tm):
    """
    Logistic growth function.
    
    P(t) = K / (1 + exp(-r * (t - tm)))
    
    Parameters:
        K: Carrying capacity (saturation level)
        r: Growth rate
        tm: Inflection point (time of maximum growth rate)
    """
    return K / (1 + np.exp(-r * (t - tm)))


def logistic_derivative(t, K, r, tm):
    """First derivative of logistic function (annual production rate)."""
    exp_term = np.exp(-r * (t - tm))
    return K * r * exp_term / ((1 + exp_term) ** 2)


def fit_logistic_model(years: np.ndarray, cumulative: np.ndarray) -> Dict:
    """
    Fit logistic growth model to cumulative publication data.
    
    Returns dict with model parameters and fit statistics.
    """
    # Initial parameter estimates
    K_init = cumulative[-1] * 1.5  # Saturation above current total
    tm_init = years[len(years) // 2]  # Middle year
    r_init = 0.1  # Moderate growth rate
    
    # Bounds
    bounds = (
        [cumulative[-1], 0.001, years[0]],  # Lower bounds
        [cumulative[-1] * 10, 2.0, years[-1] + 50]  # Upper bounds
    )
    
    try:
        popt, pcov = curve_fit(
            logistic_function, years, cumulative,
            p0=[K_init, r_init, tm_init],
            bounds=bounds,
            maxfev=10000
        )
        K, r, tm = popt
        
        # Calculate fitted values
        fitted = logistic_function(years, K, r, tm)
        
        # Calculate fit statistics
        residuals = cumulative - fitted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((cumulative - np.mean(cumulative)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        n = len(cumulative)
        rmse = np.sqrt(ss_res / n)
        
        # AIC and BIC (3 parameters)
        k_params = 3
        if ss_res > 0 and n > k_params:
            log_likelihood = -n/2 * np.log(ss_res / n)
            aic = 2 * k_params - 2 * log_likelihood
            bic = k_params * np.log(n) - 2 * log_likelihood
        else:
            aic = bic = np.nan
        
        # Growth duration (time from 10% to 90% of K)
        # For logistic: t_x = tm + ln(x/(1-x)) / r
        t_10 = tm + np.log(0.1 / 0.9) / r
        t_90 = tm + np.log(0.9 / 0.1) / r
        growth_duration = t_90 - t_10
        
        # Peak annual production (derivative at tm)
        peak_annual = K * r / 4  # Maximum of derivative
        
        return {
            "success": True,
            "K": K,
            "r": r,
            "tm": tm,
            "r_squared": r_squared,
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
            "growth_duration": growth_duration,
            "peak_annual": peak_annual,
            "fitted": fitted,
            "pcov": pcov,
        }
    
    except Exception as e:
        logger.exception(f"Logistic fit failed: {e}")
        return {"success": False, "error": str(e)}


def calculate_milestones(K: float, r: float, tm: float) -> Dict[str, float]:
    """Calculate milestone years (when certain % of K is reached)."""
    milestones = {}
    for pct in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
        # Solve: pct * K = K / (1 + exp(-r*(t - tm)))
        # => t = tm + ln(pct / (1 - pct)) / r
        if 0 < pct < 1:
            t = tm + np.log(pct / (1 - pct)) / r
            milestones[f"{int(pct*100)}%"] = t
    return milestones


def calculate_forecast(K: float, r: float, tm: float, years: List[int]) -> Dict[int, Dict]:
    """Calculate forecasted values for specific years."""
    forecasts = {}
    for year in years:
        cumulative = logistic_function(year, K, r, tm)
        annual = logistic_derivative(year, K, r, tm)
        forecasts[year] = {
            "cumulative": cumulative,
            "annual": annual,
        }
    return forecasts


# =============================================================================
# PLOT WIDGET
# =============================================================================

class LifeCyclePlotGraph(PlotWidget):
    """Plot widget for life cycle visualization."""
    
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
        self.legend.hide()
    
    def clear_plot(self):
        self.clear()
        self.legend.clear()
        self.legend.hide()
        self.setTitle("")
    
    def plot_lifecycle(self, years: np.ndarray, observed: np.ndarray,
                       fitted: np.ndarray, K: float, tm: float,
                       forecast_years: np.ndarray = None,
                       forecast_values: np.ndarray = None,
                       milestones: Dict = None,
                       show_forecast: bool = True,
                       show_milestones: bool = True):
        """Plot life cycle with observed data, fitted curve, and forecasts."""
        self.clear_plot()
        
        # Observed data points
        scatter = pg.ScatterPlotItem(
            x=years, y=observed,
            pen=pg.mkPen(None),
            brush=pg.mkBrush("#3498db"),
            size=8,
            name="Observed"
        )
        self.addItem(scatter)
        
        # Fitted curve
        line = pg.PlotDataItem(
            x=years, y=fitted,
            pen=pg.mkPen("#2ecc71", width=2),
            name="Fitted (Logistic)"
        )
        self.addItem(line)
        
        # Forecast
        if show_forecast and forecast_years is not None and forecast_values is not None:
            forecast_line = pg.PlotDataItem(
                x=forecast_years, y=forecast_values,
                pen=pg.mkPen("#e74c3c", width=2, style=Qt.DashLine),
                name="Forecast"
            )
            self.addItem(forecast_line)
        
        # Saturation line (K)
        k_line = pg.InfiniteLine(
            pos=K, angle=0,
            pen=pg.mkPen("#9b59b6", width=1.5, style=Qt.DotLine)
        )
        self.addItem(k_line)
        
        # Label for K
        k_label = pg.TextItem(f"K = {K:.0f}", color="#9b59b6", anchor=(0, 1))
        k_label.setPos(years[0], K)
        self.addItem(k_label)
        
        # Inflection point (Tm)
        tm_line = pg.InfiniteLine(
            pos=tm, angle=90,
            pen=pg.mkPen("#f39c12", width=1.5, style=Qt.DotLine)
        )
        self.addItem(tm_line)
        
        # Milestone markers
        if show_milestones and milestones:
            for pct, year in milestones.items():
                if years[0] - 10 <= year <= years[-1] + 50:
                    value = float(pct.replace('%', '')) / 100 * K
                    marker = pg.ScatterPlotItem(
                        x=[year], y=[value],
                        pen=pg.mkPen("#e67e22", width=2),
                        brush=pg.mkBrush(None),
                        size=15,
                        symbol='o'
                    )
                    self.addItem(marker)
        
        # Labels
        self.setLabel('bottom', 'Year')
        self.setLabel('left', 'Cumulative Publications')
        self.setTitle('Life Cycle Analysis (Logistic Growth Model)')
        
        # Legend
        self.legend.clear()
        self.legend.addItem(scatter, "Observed")
        self.legend.addItem(line, "Fitted")
        if show_forecast and forecast_years is not None:
            self.legend.addItem(forecast_line, "Forecast")
        self.legend.show()
        
        # Auto-range
        self.autoRange()


# =============================================================================
# SUMMARY WIDGET
# =============================================================================

class SummaryWidget(QWidget):
    """Widget showing model summary and statistics."""
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Placeholder text
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.layout.addWidget(self.text)
    
    def update_summary(self, results: Dict, last_year: int, last_cumulative: int,
                       projection_years: List[int]):
        """Update summary display with model results."""
        if not results.get("success"):
            self.text.setHtml(f"<h3>Model Fitting Failed</h3><p>{results.get('error', 'Unknown error')}</p>")
            return
        
        K = results["K"]
        r = results["r"]
        tm = results["tm"]
        r_squared = results["r_squared"]
        rmse = results["rmse"]
        aic = results["aic"]
        bic = results["bic"]
        growth_duration = results["growth_duration"]
        peak_annual = results["peak_annual"]
        
        # Fit quality assessment
        if r_squared >= 0.95:
            quality = "Excellent"
            quality_color = "#27ae60"
        elif r_squared >= 0.85:
            quality = "Good"
            quality_color = "#3498db"
        elif r_squared >= 0.70:
            quality = "Moderate"
            quality_color = "#f39c12"
        else:
            quality = "Poor"
            quality_color = "#e74c3c"
        
        # Progress to saturation
        progress = (last_cumulative / K) * 100 if K > 0 else 0
        
        # Life cycle phase
        if progress < 10:
            phase = "emerging phase (< 10% of saturation)"
        elif progress < 50:
            phase = "growth phase (10-50% of saturation)"
        elif progress < 90:
            phase = "maturity phase (50-90% of saturation)"
        else:
            phase = "saturation phase (> 90% of saturation)"
        
        # Milestones
        milestones = calculate_milestones(K, r, tm)
        
        # Forecasts
        forecasts = calculate_forecast(K, r, tm, projection_years)
        
        html = f"""
        <style>
            .card {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border: 1px solid #dee2e6; }}
            .metric {{ display: inline-block; text-align: center; margin: 10px 20px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
            .quality-badge {{ background: {quality_color}; color: white; padding: 5px 15px; border-radius: 4px; font-weight: bold; }}
            .progress-bar {{ background: #ecf0f1; height: 25px; border-radius: 4px; overflow: hidden; }}
            .progress-fill {{ background: #3498db; height: 100%; text-align: center; color: white; line-height: 25px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            td, th {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        </style>
        
        <h2>📊 Model Overview</h2>
        <div class="card">
            <div class="metric">
                <div class="metric-label">Saturation (K)</div>
                <div class="metric-value">{K:.0f}</div>
                <div class="metric-label">pubs</div>
            </div>
            <div class="metric">
                <div class="metric-label">Peak Year (Tm)</div>
                <div class="metric-value">{tm:.1f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Peak Annual</div>
                <div class="metric-value">{peak_annual:.0f}</div>
                <div class="metric-label">pubs/year</div>
            </div>
            <div class="metric">
                <div class="metric-label">Growth Duration (Δt)</div>
                <div class="metric-value">{growth_duration:.1f}</div>
                <div class="metric-label">years</div>
            </div>
        </div>
        
        <h2>✓ Model Fit Quality</h2>
        <div class="card">
            <span class="quality-badge">{quality}</span>
            <br><br>
            <table>
                <tr><th>R²</th><th>RMSE</th><th>AIC</th><th>BIC</th></tr>
                <tr>
                    <td><b>{r_squared:.4f}</b></td>
                    <td><b>{rmse:.2f}</b></td>
                    <td><b>{aic:.1f}</b></td>
                    <td><b>{bic:.1f}</b></td>
                </tr>
            </table>
        </div>
        
        <h2>📈 Current Status</h2>
        <div class="card">
            <table>
                <tr><td>Last Observed Year:</td><td><b>{last_year}</b></td></tr>
                <tr><td>Cumulative Total:</td><td><b>{last_cumulative:,}</b></td></tr>
                <tr><td>Progress to Saturation:</td><td><b>{progress:.1f}%</b></td></tr>
            </table>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {min(progress, 100):.1f}%">{progress:.1f}%</div>
            </div>
            <p><i>💡 The topic is in the {phase}.</i></p>
        </div>
        
        <h2>🎯 Milestone Years</h2>
        <div class="card">
            <table>
                <tr><th>Milestone</th><th>Year</th><th>Relative to Tm</th></tr>
        """
        
        for pct, year in sorted(milestones.items(), key=lambda x: float(x[0].replace('%', ''))):
            diff = year - tm
            diff_str = f"({diff:+.1f} years)" if diff != 0 else "(inflection)"
            html += f"<tr><td>{pct} of K</td><td><b>{year:.1f}</b></td><td>{diff_str}</td></tr>"
        
        html += """
            </table>
        </div>
        
        <h2>🔮 Forecast</h2>
        <div class="card">
            <table>
                <tr><th>Year</th><th>Annual</th><th>Cumulative</th></tr>
        """
        
        for year, forecast in sorted(forecasts.items()):
            html += f"<tr><td><b>{year}</b></td><td>{forecast['annual']:.0f}</td><td>{forecast['cumulative']:.0f}</td></tr>"
        
        html += """
            </table>
        </div>
        """
        
        self.text.setHtml(html)


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWLifeCycle(OWWidget):
    """Analyze life cycle of scientific production using logistic growth model."""
    
    name = "Life Cycle Analysis"
    description = "Analyze the life cycle of scientific production using logistic growth model"
    icon = "icons/life_cycle.svg"
    priority = 85
    keywords = ["life cycle", "logistic", "growth", "saturation", "forecast"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data or trend analysis data")
    
    class Outputs:
        model_results = Output("Model Results", Table, doc="Model parameters and statistics")
        forecast = Output("Forecast", Table, doc="Forecasted values")
    
    # Settings
    forecast_years = settings.Setting(50)
    projection_years_str = settings.Setting("2025, 2030, 2035")
    show_milestones = settings.Setting(True)
    show_forecast = settings.Setting(True)
    forecast_limit = settings.Setting(30)
    
    want_main_area = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_year = Msg("Year column not found")
        no_docs = Msg("Document count column not found")
        fit_failed = Msg("Model fitting failed: {}")
        insufficient_data = Msg("Need at least 5 data points for fitting")
    
    class Warning(OWWidget.Warning):
        poor_fit = Msg("Poor model fit (R² = {:.3f})")
    
    class Information(OWWidget.Information):
        model_fit = Msg("Model fit: R² = {:.4f}, K = {:.0f}")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._results: Optional[Dict] = None
        
        self._setup_gui()
    
    def _setup_gui(self):
        # Model Settings
        model_box = gui.widgetBox(self.controlArea, "Model Settings")
        
        gui.spin(model_box, self, "forecast_years", minv=10, maxv=100,
                 label="Forecast Years:", callback=self._on_setting_changed)
        
        gui.lineEdit(model_box, self, "projection_years_str",
                     label="Projection Years:", callback=self._on_setting_changed)
        
        # Display Options
        display_box = gui.widgetBox(self.controlArea, "Display Options")
        
        gui.checkBox(display_box, self, "show_milestones", "Show Milestone Years",
                     callback=self._update_plot)
        
        gui.checkBox(display_box, self, "show_forecast", "Show Forecast Period",
                     callback=self._update_plot)
        
        gui.spin(display_box, self, "forecast_limit", minv=10, maxv=100,
                 label="Plot Forecast Limit:", callback=self._update_plot)
        
        # Run button
        self.run_btn = gui.button(
            self.controlArea, self, "Run Analysis",
            callback=self._run_analysis,
        )
        self.run_btn.setMinimumHeight(35)
        
        self.controlArea.layout().addStretch(1)
        
        # Main area with tabs
        self.tabs = QTabWidget()
        self.mainArea.layout().addWidget(self.tabs)
        
        # Summary tab
        self.summary_widget = SummaryWidget()
        self.tabs.addTab(self.summary_widget, "📋 Summary")
        
        # Plot tab
        self.graph = LifeCyclePlotGraph()
        self.tabs.addTab(self.graph, "📈 Plot")
        
        # Info tab
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setHtml(self._get_info_html())
        self.tabs.addTab(self.info_text, "ℹ Info")
    
    def _get_info_html(self) -> str:
        return """
        <h2>Life Cycle Analysis</h2>
        <p>This widget fits a <b>logistic growth model</b> to cumulative publication data 
        to analyze the life cycle of a scientific topic.</p>
        
        <h3>Logistic Growth Model</h3>
        <p><b>Formula:</b> P(t) = K / (1 + exp(-r × (t - t<sub>m</sub>)))</p>
        
        <h3>Parameters</h3>
        <ul>
            <li><b>K (Saturation)</b>: Maximum cumulative publications the topic will reach</li>
            <li><b>t<sub>m</sub> (Peak Year)</b>: Year of maximum annual growth rate</li>
            <li><b>r (Growth Rate)</b>: Steepness of the growth curve</li>
        </ul>
        
        <h3>Life Cycle Phases</h3>
        <ul>
            <li><b>Emerging (< 10%)</b>: Topic is new, few publications</li>
            <li><b>Growth (10-50%)</b>: Rapid increase in publications</li>
            <li><b>Maturity (50-90%)</b>: Growth slowing, topic well-established</li>
            <li><b>Saturation (> 90%)</b>: Topic reaching maximum, declining interest</li>
        </ul>
        
        <h3>Fit Quality</h3>
        <ul>
            <li><b>R² ≥ 0.95</b>: Excellent fit</li>
            <li><b>R² ≥ 0.85</b>: Good fit</li>
            <li><b>R² ≥ 0.70</b>: Moderate fit</li>
            <li><b>R² < 0.70</b>: Poor fit</li>
        </ul>
        
        <h3>References</h3>
        <p>Bettencourt, L. M., et al. (2008). "Population modeling of the emergence 
        and development of scientific fields." <i>Scientometrics</i>, 75(3), 495-518.</p>
        """
    
    def _on_setting_changed(self):
        pass
    
    def _update_plot(self):
        if self._results and self._results.get("success"):
            self._plot_results()
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._results = None
        
        self.graph.clear_plot()
        
        if data is None:
            self.Error.no_data()
            return
        
        self._df = self._table_to_df(data)
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)
    
    def _get_year_column(self) -> Optional[str]:
        if self._df is None:
            return None
        for col in self._df.columns:
            if col.lower() in ["year", "period", "publication year"]:
                return col
        return None
    
    def _get_docs_column(self) -> Optional[str]:
        """Find document count or cumulative column."""
        if self._df is None:
            return None
        for col in self._df.columns:
            col_lower = col.lower()
            if "cumulative" in col_lower and "doc" in col_lower:
                return col
            if col_lower in ["number of documents", "documents", "doc_count", "count"]:
                return col
        # Fallback: first numeric column that's not year
        year_col = self._get_year_column()
        for col in self._df.columns:
            if col != year_col and pd.api.types.is_numeric_dtype(self._df[col]):
                return col
        return None
    
    def _run_analysis(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self.Error.no_data()
            return
        
        year_col = self._get_year_column()
        if year_col is None:
            self.Error.no_year()
            return
        
        docs_col = self._get_docs_column()
        if docs_col is None:
            self.Error.no_docs()
            return
        
        # Prepare data
        df = self._df[[year_col, docs_col]].dropna()
        df = df.sort_values(year_col)
        
        years = df[year_col].values.astype(float)
        docs = df[docs_col].values.astype(float)
        
        # Check if cumulative or annual
        if not np.all(np.diff(docs) >= 0):
            # Annual data - convert to cumulative
            cumulative = np.cumsum(docs)
        else:
            cumulative = docs
        
        if len(years) < 5:
            self.Error.insufficient_data()
            return
        
        # Fit model
        self._results = fit_logistic_model(years, cumulative)
        
        if not self._results.get("success"):
            self.Error.fit_failed(self._results.get("error", "Unknown"))
            return
        
        r_squared = self._results["r_squared"]
        K = self._results["K"]
        
        if r_squared < 0.7:
            self.Warning.poor_fit(r_squared)
        
        self.Information.model_fit(r_squared, K)
        
        # Store additional data
        self._results["years"] = years
        self._results["cumulative"] = cumulative
        
        # Update displays
        self._refresh_summary()
        self._plot_results()
        self._send_outputs()
    
    def _refresh_summary(self):
        """Update summary widget."""
        if self._results is None:
            return
        
        years = self._results["years"]
        cumulative = self._results["cumulative"]
        
        # Parse projection years
        try:
            projection_years = [int(y.strip()) for y in self.projection_years_str.split(",")]
        except:
            projection_years = [2025, 2030, 2035]
        
        self.summary_widget.update_summary(
            self._results,
            last_year=int(years[-1]),
            last_cumulative=int(cumulative[-1]),
            projection_years=projection_years
        )
    
    def _plot_results(self):
        """Update plot with results."""
        if self._results is None or not self._results.get("success"):
            return
        
        years = self._results["years"]
        cumulative = self._results["cumulative"]
        fitted = self._results["fitted"]
        K = self._results["K"]
        r = self._results["r"]
        tm = self._results["tm"]
        
        # Generate forecast
        last_year = int(years[-1])
        forecast_years = np.arange(last_year, last_year + self.forecast_limit + 1)
        forecast_values = logistic_function(forecast_years, K, r, tm)
        
        # Milestones
        milestones = calculate_milestones(K, r, tm) if self.show_milestones else None
        
        self.graph.plot_lifecycle(
            years=years,
            observed=cumulative,
            fitted=fitted,
            K=K,
            tm=tm,
            forecast_years=forecast_years if self.show_forecast else None,
            forecast_values=forecast_values if self.show_forecast else None,
            milestones=milestones,
            show_forecast=self.show_forecast,
            show_milestones=self.show_milestones,
        )
    
    def _send_outputs(self):
        """Send output tables."""
        if self._results is None or not self._results.get("success"):
            self.Outputs.model_results.send(None)
            self.Outputs.forecast.send(None)
            return
        
        # Model results table
        results_data = {
            "Parameter": ["K (Saturation)", "Tm (Peak Year)", "r (Growth Rate)",
                         "R²", "RMSE", "AIC", "BIC", "Growth Duration", "Peak Annual"],
            "Value": [
                self._results["K"],
                self._results["tm"],
                self._results["r"],
                self._results["r_squared"],
                self._results["rmse"],
                self._results["aic"],
                self._results["bic"],
                self._results["growth_duration"],
                self._results["peak_annual"],
            ]
        }
        
        domain = Domain(
            [ContinuousVariable("Value")],
            metas=[StringVariable("Parameter")]
        )
        
        results_df = pd.DataFrame(results_data)
        results_table = Table.from_numpy(
            domain,
            X=results_df[["Value"]].values,
            metas=results_df[["Parameter"]].values.astype(object),
        )
        
        self.Outputs.model_results.send(results_table)
        
        # Forecast table
        K = self._results["K"]
        r = self._results["r"]
        tm = self._results["tm"]
        last_year = int(self._results["years"][-1])
        
        forecast_years = list(range(last_year, last_year + self.forecast_years + 1))
        forecast_cum = [logistic_function(y, K, r, tm) for y in forecast_years]
        forecast_ann = [logistic_derivative(y, K, r, tm) for y in forecast_years]
        
        forecast_domain = Domain([
            ContinuousVariable("Year"),
            ContinuousVariable("Cumulative"),
            ContinuousVariable("Annual"),
        ])
        
        forecast_df = pd.DataFrame({
            "Year": forecast_years,
            "Cumulative": forecast_cum,
            "Annual": forecast_ann,
        })
        
        forecast_table = Table.from_numpy(
            forecast_domain,
            X=forecast_df.values,
        )
        
        self.Outputs.forecast.send(forecast_table)


if __name__ == "__main__":
    WidgetPreview(OWLifeCycle).run()
