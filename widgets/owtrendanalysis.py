# -*- coding: utf-8 -*-
"""
Trend Analysis Widget
=====================
Orange widget for analyzing temporal patterns and trends in bibliographic data.

Supports:
- Scientific Production analysis (documents and citations over time)
- Time period filtering and aggregation
- Trend line fitting (linear, polynomial, exponential)
- Growth rate computation
"""

import os
import logging
from typing import Optional, List, Any
import datetime

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QFrame,
)
from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

# Try to import biblium
try:
    import biblium
    from biblium import utilsbib
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    utilsbib = None

# Try scipy/numpy for trendlines
try:
    from scipy import stats
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

ANALYSIS_TYPES = {
    "Scientific Production": "production",
}

AGGREGATION_TYPES = {
    "Yearly": 1,
    "2-Year Periods": 2,
    "5-Year Periods": 5,
    "Decades": 10,
}

TRENDLINE_TYPES = {
    "None": None,
    "Linear": "linear",
    "Polynomial (2)": "poly2",
    "Polynomial (3)": "poly3",
    "Exponential": "exponential",
    "Moving Average (3)": "ma3",
    "Moving Average (5)": "ma5",
}


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWTrendAnalysis(OWWidget):
    """Analyze temporal patterns and trends in bibliographic data."""
    
    name = "Trend Analysis"
    description = "Analyze temporal patterns and trends in your data"
    icon = "icons/trend_analysis.svg"
    priority = 60
    keywords = ["trend", "production", "temporal", "time", "growth", "yearly"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        trends = Output("Trends", Table, doc="Trend analysis results")
        summary = Output("Summary", Table, doc="Summary statistics")
    
    # Settings - Analysis
    analysis_type = settings.Setting("Scientific Production")
    
    # Settings - Time Period
    year_from = settings.Setting(1900)
    year_to = settings.Setting(1900)  # 1900 means auto
    merge_before = settings.Setting(False)
    merge_year = settings.Setting(2000)
    aggregation = settings.Setting("Yearly")
    
    # Settings - Trendline
    trendline_type = settings.Setting("None")
    forecast_years = settings.Setting(0)
    
    # Settings - Advanced
    relative_counts = settings.Setting(True)
    cumulative = settings.Setting(True)
    percent_change = settings.Setting(True)
    predict_last_year = settings.Setting(False)
    
    auto_apply = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_year = Msg("Year column not found")
        compute_error = Msg("Computation error: {}")
    
    class Warning(OWWidget.Warning):
        few_years = Msg("Only {} years in dataset")
        no_citations = Msg("Citations column not found - citation metrics unavailable")
    
    class Information(OWWidget.Information):
        analyzed = Msg("Analyzed {} years: {} to {}")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._result_df: Optional[pd.DataFrame] = None
        self._summary_df: Optional[pd.DataFrame] = None
        
        self._setup_control_area()
        self._setup_main_area()
    
    # =========================================================================
    # GUI SETUP
    # =========================================================================
    
    def _setup_control_area(self):
        """Build control area."""
        # Analysis Type
        type_box = gui.widgetBox(self.controlArea, "Analysis Type")
        
        type_layout = QGridLayout()
        type_layout.addWidget(QLabel("Type:"), 0, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems(list(ANALYSIS_TYPES.keys()))
        self.type_combo.setCurrentText(self.analysis_type)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo, 0, 1)
        type_box.layout().addLayout(type_layout)
        
        # Time Period
        time_box = gui.widgetBox(self.controlArea, "Time Period")
        
        time_layout = QGridLayout()
        
        time_layout.addWidget(QLabel("From:"), 0, 0)
        self.year_from_spin = QSpinBox()
        self.year_from_spin.setRange(1900, 2100)
        self.year_from_spin.setValue(self.year_from)
        self.year_from_spin.setSpecialValueText("Auto")
        self.year_from_spin.valueChanged.connect(lambda v: setattr(self, 'year_from', v))
        time_layout.addWidget(self.year_from_spin, 0, 1)
        
        time_layout.addWidget(QLabel("To:"), 0, 2)
        self.year_to_spin = QSpinBox()
        self.year_to_spin.setRange(1900, 2100)
        self.year_to_spin.setValue(self.year_to)
        self.year_to_spin.setSpecialValueText("Auto")
        self.year_to_spin.valueChanged.connect(lambda v: setattr(self, 'year_to', v))
        time_layout.addWidget(self.year_to_spin, 0, 3)
        
        time_box.layout().addLayout(time_layout)
        
        # Merge years option
        merge_layout = QHBoxLayout()
        self.merge_cb = QCheckBox("Merge years before:")
        self.merge_cb.setChecked(self.merge_before)
        self.merge_cb.toggled.connect(lambda c: setattr(self, 'merge_before', c))
        merge_layout.addWidget(self.merge_cb)
        
        self.merge_spin = QSpinBox()
        self.merge_spin.setRange(1900, 2100)
        self.merge_spin.setValue(self.merge_year)
        self.merge_spin.valueChanged.connect(lambda v: setattr(self, 'merge_year', v))
        merge_layout.addWidget(self.merge_spin)
        merge_layout.addStretch()
        time_box.layout().addLayout(merge_layout)
        
        # Aggregation
        agg_layout = QHBoxLayout()
        agg_layout.addWidget(QLabel("Aggregation:"))
        self.agg_combo = QComboBox()
        self.agg_combo.addItems(list(AGGREGATION_TYPES.keys()))
        self.agg_combo.setCurrentText(self.aggregation)
        self.agg_combo.currentTextChanged.connect(lambda a: setattr(self, 'aggregation', a))
        agg_layout.addWidget(self.agg_combo)
        time_box.layout().addLayout(agg_layout)
        
        # Trendline Options
        trend_box = gui.widgetBox(self.controlArea, "Trendline Options")
        
        trend_layout = QGridLayout()
        trend_layout.addWidget(QLabel("Trendline:"), 0, 0)
        self.trend_combo = QComboBox()
        self.trend_combo.addItems(list(TRENDLINE_TYPES.keys()))
        self.trend_combo.setCurrentText(self.trendline_type)
        self.trend_combo.currentTextChanged.connect(lambda t: setattr(self, 'trendline_type', t))
        trend_layout.addWidget(self.trend_combo, 0, 1)
        
        trend_layout.addWidget(QLabel("Forecast years:"), 1, 0)
        self.forecast_spin = QSpinBox()
        self.forecast_spin.setRange(0, 20)
        self.forecast_spin.setValue(self.forecast_years)
        self.forecast_spin.valueChanged.connect(lambda v: setattr(self, 'forecast_years', v))
        trend_layout.addWidget(self.forecast_spin, 1, 1)
        
        trend_box.layout().addLayout(trend_layout)
        
        # Advanced Options (collapsible)
        adv_box = gui.widgetBox(self.controlArea, "Advanced Options")
        
        self.relative_cb = QCheckBox("Compute relative counts")
        self.relative_cb.setChecked(self.relative_counts)
        self.relative_cb.toggled.connect(lambda c: setattr(self, 'relative_counts', c))
        adv_box.layout().addWidget(self.relative_cb)
        
        self.cumulative_cb = QCheckBox("Compute cumulative values")
        self.cumulative_cb.setChecked(self.cumulative)
        self.cumulative_cb.toggled.connect(lambda c: setattr(self, 'cumulative', c))
        adv_box.layout().addWidget(self.cumulative_cb)
        
        self.pct_change_cb = QCheckBox("Compute percentage changes")
        self.pct_change_cb.setChecked(self.percent_change)
        self.pct_change_cb.toggled.connect(lambda c: setattr(self, 'percent_change', c))
        adv_box.layout().addWidget(self.pct_change_cb)
        
        self.predict_cb = QCheckBox("Predict last year (if incomplete)")
        self.predict_cb.setChecked(self.predict_last_year)
        self.predict_cb.toggled.connect(lambda c: setattr(self, 'predict_last_year', c))
        adv_box.layout().addWidget(self.predict_cb)
        
        # Apply button
        self.apply_btn = QPushButton("Analyze Trends")
        self.apply_btn.setMinimumHeight(35)
        self.apply_btn.clicked.connect(self.commit)
        self.controlArea.layout().addWidget(self.apply_btn)
        
        gui.auto_apply(self.controlArea, self, "auto_apply")
    
    def _setup_main_area(self):
        """Build main area with results preview."""
        # Summary
        summary_box = gui.widgetBox(self.mainArea, "Analysis Summary")
        self.summary_label = QLabel("No data")
        self.summary_label.setWordWrap(True)
        summary_box.layout().addWidget(self.summary_label)
        
        # Results table
        results_box = gui.widgetBox(self.mainArea, "Trend Data")
        self.results_table = QTableWidget()
        self.results_table.setMinimumHeight(300)
        results_box.layout().addWidget(self.results_table)
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_type_changed(self, analysis_type):
        self.analysis_type = analysis_type
        if self.auto_apply:
            self.commit()
    
    def commit(self):
        """Perform trend analysis."""
        self._compute()
    
    # =========================================================================
    # DATA HANDLING
    # =========================================================================
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive input data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._result_df = None
        self._summary_df = None
        
        if data is None:
            self.Error.no_data()
            self._update_results_display()
            return
        
        # Convert to DataFrame
        self._df = self._table_to_df(data)
        
        # Update year range defaults based on data
        year_col = self._find_column(["Year", "Publication Year", "publication_year", "PY"])
        if year_col:
            years = pd.to_numeric(self._df[year_col], errors='coerce').dropna()
            if len(years) > 0:
                # Only set if currently at default/auto
                if self.year_from == 1900:
                    self.year_from_spin.setValue(1900)  # Keep auto
                if self.year_to == 1900:
                    self.year_to_spin.setValue(1900)  # Keep auto
        
        if self.auto_apply:
            self.commit()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        """Convert Orange Table to pandas DataFrame."""
        data = {}
        
        for var in table.domain.attributes:
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else "" for v in col]
            else:
                data[var.name] = col
        
        for var in table.domain.metas:
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else "" for v in col]
            elif isinstance(var, StringVariable):
                data[var.name] = col
            else:
                data[var.name] = col
        
        if table.domain.class_var:
            var = table.domain.class_var
            col = table.get_column(var)
            if isinstance(var, DiscreteVariable):
                data[var.name] = [var.values[int(v)] if not np.isnan(v) else "" for v in col]
            else:
                data[var.name] = col
        
        return pd.DataFrame(data)
    
    def _find_column(self, candidates):
        """Find first available column from candidates."""
        if self._df is None:
            return None
        for col in candidates:
            if col in self._df.columns:
                return col
        return None
    
    # =========================================================================
    # COMPUTATION
    # =========================================================================
    
    def _compute(self):
        """Perform trend analysis."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None or self._df.empty:
            self.Error.no_data()
            self._update_results_display()
            self._send_outputs(None, None)
            return
        
        try:
            # Find year column
            year_col = self._find_column(["Year", "Publication Year", "publication_year", "PY"])
            if year_col is None:
                self.Error.no_year()
                self._update_results_display()
                self._send_outputs(None, None)
                return
            
            # Check for citations column
            cite_col = self._find_column(["Cited by", "Times Cited", "Citation Count", "cited_by_count", "TC"])
            if cite_col is None:
                self.Warning.no_citations()
            
            # Prepare data
            df = self._prepare_data(year_col, cite_col)
            if df is None or df.empty:
                self._update_results_display()
                self._send_outputs(None, None)
                return
            
            # Compute production
            result_df = self._compute_production(df)
            if result_df is None or result_df.empty:
                self._update_results_display()
                self._send_outputs(None, None)
                return
            
            # Apply aggregation
            if self.aggregation != "Yearly":
                result_df = self._apply_aggregation(result_df)
            
            # Apply merge before
            if self.merge_before:
                result_df = self._apply_merge_before(result_df)
            
            # Add trendline
            if self.trendline_type != "None":
                result_df = self._add_trendline(result_df)
            
            # Compute summary statistics
            summary_df = self._compute_summary(result_df)
            
            self._result_df = result_df
            self._summary_df = summary_df
            
            # Update display
            self._update_results_display()
            
            # Send outputs
            self._send_outputs(result_df, summary_df)
            
            # Info message
            year_min = int(result_df["Year"].min()) if "Year" in result_df.columns else 0
            year_max = int(result_df["Year"].max()) if "Year" in result_df.columns else 0
            n_years = len(result_df)
            self.Information.analyzed(n_years, year_min, year_max)
            
            if n_years < 5:
                self.Warning.few_years(n_years)
            
        except Exception as e:
            import traceback
            logger.error(f"Computation error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
            self._update_results_display()
            self._send_outputs(None, None)
    
    def _prepare_data(self, year_col, cite_col):
        """Prepare and filter data."""
        df = self._df.copy()
        
        # Standardize column names
        df["Year"] = pd.to_numeric(df[year_col], errors='coerce')
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)
        
        if cite_col and cite_col in df.columns:
            df["Cited by"] = pd.to_numeric(df[cite_col], errors='coerce').fillna(0)
        else:
            df["Cited by"] = 0
        
        # Filter by year range
        data_min = df["Year"].min()
        data_max = df["Year"].max()
        
        yr_from = self.year_from if self.year_from > 1900 else data_min
        yr_to = self.year_to if self.year_to > 1900 else data_max
        
        df = df[(df["Year"] >= yr_from) & (df["Year"] <= yr_to)]
        
        return df
    
    def _compute_production(self, df):
        """Compute scientific production statistics."""
        if HAS_BIBLIUM and utilsbib:
            return utilsbib.get_scientific_production(
                df,
                relative_counts=self.relative_counts,
                cumulative=self.cumulative,
                predict_last_year=self.predict_last_year,
                percent_change=self.percent_change,
            )
        else:
            return self._compute_production_basic(df)
    
    def _compute_production_basic(self, df):
        """Basic production computation without biblium."""
        if df.empty:
            return pd.DataFrame()
        
        # Create complete year range
        all_years = pd.Series(range(df["Year"].min(), df["Year"].max() + 1), name="Year")
        
        # Aggregate
        production = df.groupby("Year").agg(
            Documents=("Year", "count"),
            Total_Citations=("Cited by", "sum")
        ).reset_index()
        
        # Merge with full year range
        production = all_years.to_frame().merge(production, on="Year", how="left").fillna(0)
        production["Documents"] = production["Documents"].astype(int)
        production["Total_Citations"] = production["Total_Citations"].astype(int)
        
        # Relative counts
        if self.relative_counts:
            total_docs = production["Documents"].sum()
            if total_docs > 0:
                production["Proportion Documents"] = production["Documents"] / total_docs
                production["Percentage Documents"] = production["Proportion Documents"] * 100
        
        # Cumulative
        if self.cumulative:
            production["Cumulative Documents"] = production["Documents"].cumsum()
            production["Cumulative Citations"] = production["Total_Citations"].cumsum()
        
        # Percent change
        if self.percent_change:
            production["Percentage Change Documents"] = production["Documents"].pct_change() * 100
            production["Percentage Change Citations"] = production["Total_Citations"].pct_change() * 100
        
        # Rename columns
        production = production.rename(columns={
            "Documents": "Number of Documents",
            "Total_Citations": "Total Citations",
        })
        
        return production
    
    def _apply_aggregation(self, df):
        """Aggregate by time period."""
        period = AGGREGATION_TYPES.get(self.aggregation, 1)
        if period == 1:
            return df
        
        df = df.copy()
        df["Period"] = (df["Year"] // period) * period
        
        # Aggregate
        agg_dict = {}
        if "Number of Documents" in df.columns:
            agg_dict["Number of Documents"] = "sum"
        if "Total Citations" in df.columns:
            agg_dict["Total Citations"] = "sum"
        
        if not agg_dict:
            return df
        
        result = df.groupby("Period").agg(agg_dict).reset_index()
        result = result.rename(columns={"Period": "Year"})
        
        # Recompute relative/cumulative for aggregated data
        if self.relative_counts and "Number of Documents" in result.columns:
            total = result["Number of Documents"].sum()
            if total > 0:
                result["Percentage Documents"] = result["Number of Documents"] / total * 100
        
        if self.cumulative:
            if "Number of Documents" in result.columns:
                result["Cumulative Documents"] = result["Number of Documents"].cumsum()
            if "Total Citations" in result.columns:
                result["Cumulative Citations"] = result["Total Citations"].cumsum()
        
        return result
    
    def _apply_merge_before(self, df):
        """Merge years before threshold into single row."""
        if not self.merge_before:
            return df
        
        df = df.copy()
        before = df[df["Year"] < self.merge_year]
        after = df[df["Year"] >= self.merge_year]
        
        if before.empty:
            return df
        
        # Sum the "before" rows
        merged_row = {"Year": f"<{self.merge_year}"}
        for col in df.columns:
            if col == "Year":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                if "Cumulative" in col or "Percentage" in col:
                    merged_row[col] = before[col].iloc[-1] if len(before) > 0 else 0
                else:
                    merged_row[col] = before[col].sum()
        
        # Combine
        result = pd.concat([pd.DataFrame([merged_row]), after], ignore_index=True)
        
        return result
    
    def _add_trendline(self, df):
        """Add trendline to data."""
        trend_type = TRENDLINE_TYPES.get(self.trendline_type)
        if trend_type is None or "Number of Documents" not in df.columns:
            return df
        
        df = df.copy()
        
        # Get numeric years only (exclude merged periods like "<2000")
        numeric_mask = df["Year"].apply(lambda x: isinstance(x, (int, float)) or str(x).isdigit())
        if numeric_mask.sum() < 3:
            return df
        
        x = df.loc[numeric_mask, "Year"].astype(float).values
        y = df.loc[numeric_mask, "Number of Documents"].values
        
        try:
            if trend_type == "linear":
                slope, intercept, r, p, se = stats.linregress(x, y)
                df.loc[numeric_mask, "Trendline"] = slope * x + intercept
                df["R²"] = r ** 2
                
            elif trend_type == "poly2":
                coeffs = np.polyfit(x, y, 2)
                df.loc[numeric_mask, "Trendline"] = np.polyval(coeffs, x)
                
            elif trend_type == "poly3":
                coeffs = np.polyfit(x, y, 3)
                df.loc[numeric_mask, "Trendline"] = np.polyval(coeffs, x)
                
            elif trend_type == "exponential" and HAS_SCIPY:
                def exp_func(x, a, b):
                    return a * np.exp(b * (x - x.min()))
                try:
                    popt, _ = curve_fit(exp_func, x, y, p0=[y[0], 0.1], maxfev=5000)
                    df.loc[numeric_mask, "Trendline"] = exp_func(x, *popt)
                except:
                    pass
                    
            elif trend_type == "ma3":
                df.loc[numeric_mask, "Trendline"] = df.loc[numeric_mask, "Number of Documents"].rolling(3, center=True).mean()
                
            elif trend_type == "ma5":
                df.loc[numeric_mask, "Trendline"] = df.loc[numeric_mask, "Number of Documents"].rolling(5, center=True).mean()
                
        except Exception as e:
            logger.warning(f"Trendline computation failed: {e}")
        
        return df
    
    def _compute_summary(self, df):
        """Compute summary statistics."""
        summary = []
        
        if "Year" in df.columns:
            # Filter numeric years
            numeric_years = df["Year"].apply(lambda x: isinstance(x, (int, float)) or str(x).isdigit())
            if numeric_years.any():
                years = df.loc[numeric_years, "Year"].astype(int)
                summary.append(("Time Span", "Start Year", int(years.min())))
                summary.append(("Time Span", "End Year", int(years.max())))
                summary.append(("Time Span", "Number of Years", len(years)))
        
        if "Number of Documents" in df.columns:
            total = df["Number of Documents"].sum()
            avg = df["Number of Documents"].mean()
            max_val = df["Number of Documents"].max()
            summary.append(("Documents", "Total", int(total)))
            summary.append(("Documents", "Average per Year", f"{avg:.1f}"))
            summary.append(("Documents", "Maximum", int(max_val)))
        
        if "Total Citations" in df.columns:
            total = df["Total Citations"].sum()
            avg = df["Total Citations"].mean()
            summary.append(("Citations", "Total", int(total)))
            summary.append(("Citations", "Average per Year", f"{avg:.1f}"))
        
        if "Percentage Change Documents" in df.columns:
            changes = df["Percentage Change Documents"].dropna()
            if len(changes) > 0:
                avg_growth = changes.mean()
                summary.append(("Growth", "Avg Annual Growth (%)", f"{avg_growth:.1f}"))
        
        if "R²" in df.columns:
            r2 = df["R²"].iloc[0]
            summary.append(("Trendline", "R²", f"{r2:.4f}"))
        
        return pd.DataFrame(summary, columns=["Category", "Metric", "Value"])
    
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    def _send_outputs(self, result_df, summary_df):
        """Send outputs."""
        if result_df is not None and not result_df.empty:
            self.Outputs.trends.send(self._df_to_table(result_df))
        else:
            self.Outputs.trends.send(None)
        
        if summary_df is not None and not summary_df.empty:
            self.Outputs.summary.send(self._df_to_table(summary_df))
        else:
            self.Outputs.summary.send(None)
    
    def _df_to_table(self, df: pd.DataFrame) -> Table:
        """Convert DataFrame to Orange Table."""
        attrs = []
        metas = []
        
        for col in df.columns:
            col_data = df[col]
            if pd.api.types.is_numeric_dtype(col_data.dtype):
                attrs.append(ContinuousVariable(str(col)))
            else:
                metas.append(StringVariable(str(col)))
        
        domain = Domain(attrs, metas=metas)
        
        X = np.zeros((len(df), len(attrs)), dtype=float)
        M = np.zeros((len(df), len(metas)), dtype=object)
        
        for i, var in enumerate(attrs):
            X[:, i] = pd.to_numeric(df[var.name], errors='coerce').fillna(np.nan).values
        
        for i, var in enumerate(metas):
            M[:, i] = df[var.name].fillna("").astype(str).values
        
        return Table.from_numpy(domain, X, metas=M if metas else None)
    
    def _update_results_display(self):
        """Update the results preview."""
        if self._result_df is None or self._result_df.empty:
            self.summary_label.setText("No results")
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return
        
        df = self._result_df
        
        # Summary text
        summary_parts = [f"<b>Analysis:</b> {self.analysis_type}"]
        
        if "Year" in df.columns:
            numeric_years = df["Year"].apply(lambda x: isinstance(x, (int, float)) or str(x).isdigit())
            if numeric_years.any():
                years = df.loc[numeric_years, "Year"].astype(int)
                summary_parts.append(f"<b>Period:</b> {years.min()} - {years.max()}")
        
        if "Number of Documents" in df.columns:
            total_docs = df["Number of Documents"].sum()
            summary_parts.append(f"<b>Total documents:</b> {total_docs:,.0f}")
        
        if "Total Citations" in df.columns:
            total_cites = df["Total Citations"].sum()
            summary_parts.append(f"<b>Total citations:</b> {total_cites:,.0f}")
        
        if self._summary_df is not None:
            growth_rows = self._summary_df[self._summary_df["Category"] == "Growth"]
            if not growth_rows.empty:
                growth = growth_rows.iloc[0]["Value"]
                summary_parts.append(f"<b>Avg annual growth:</b> {growth}%")
        
        self.summary_label.setText("<br>".join(summary_parts))
        
        # Preview table
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i, j]
                if isinstance(val, float):
                    if pd.isna(val):
                        text = ""
                    elif "Percentage" in col or "Proportion" in col:
                        text = f"{val:.2f}"
                    elif val == int(val):
                        text = f"{int(val):,}"
                    else:
                        text = f"{val:.2f}"
                else:
                    text = str(val)
                self.results_table.setItem(i, j, QTableWidgetItem(text))
        
        self.results_table.resizeColumnsToContents()


if __name__ == "__main__":
    WidgetPreview(OWTrendAnalysis).run()
