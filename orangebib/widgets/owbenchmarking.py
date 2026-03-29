# -*- coding: utf-8 -*-
"""
Benchmarking Widget
===================
Orange widget for comparing dataset distributions against global reference data.

Compares your dataset's distribution of various categories against global 
research patterns (from OpenAlex) to identify over- and under-representation.

Supports:
- Scientific Production (Year)
- Country Distribution
- SDG Distribution
- Document Type
- Open Access Status
"""

import os
import logging
from typing import Optional, List, Any

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QRadioButton, QButtonGroup,
    QLineEdit, QFileDialog,
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
    try:
        from biblium.representation import (
            compute_relative_representation,
            SUPPORTED_REFERENCES,
            fetch_openalex_yearly_counts,
            fetch_openalex_country_counts,
            fetch_openalex_sdg_counts,
            fetch_openalex_doctype_counts,
            fetch_openalex_oa_counts,
        )
        HAS_REPRESENTATION = True
    except ImportError:
        HAS_REPRESENTATION = False
except ImportError:
    HAS_BIBLIUM = False
    HAS_REPRESENTATION = False

# Try scipy for chi-square
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

COMPARISON_TYPES = {
    "Scientific Production (Year)": {
        "category": "Year",
        "columns": ["Year", "Publication Year", "publication_year", "PY"],
        "value_type": "single",
        "description": "Compare publication year distribution",
    },
    "Country Distribution": {
        "category": "Country",
        "columns": ["Countries", "Countries of Authors", "Country", "CA Country"],
        "value_type": "list",
        "description": "Compare author country distribution",
    },
    "SDG Distribution": {
        "category": "SDG",
        "columns": ["SDG"],  # Special handling - looks for SDG01-SDG17 columns
        "value_type": "sdg",
        "description": "Compare Sustainable Development Goals distribution",
    },
    "Document Type": {
        "category": "Document Type",
        "columns": ["Document Type", "Type", "type", "DT"],
        "value_type": "single",
        "description": "Compare document type distribution",
    },
    "Open Access Status": {
        "category": "Open Access",
        "columns": ["Open Access", "is_oa", "OA Status"],
        "value_type": "single",
        "description": "Compare open access status distribution",
    },
}


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWBenchmarking(OWWidget):
    """Compare dataset against global reference data."""
    
    name = "Benchmarking"
    description = "Compare dataset distributions against global research patterns"
    icon = "icons/benchmarking.svg"
    priority = 50
    keywords = ["benchmark", "comparison", "reference", "global", "representation"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        comparison = Output("Comparison", Table, doc="Comparison results")
        over_represented = Output("Over-represented", Table, doc="Over-represented categories")
        under_represented = Output("Under-represented", Table, doc="Under-represented categories")
    
    # Settings
    comparison_type = settings.Setting("Scientific Production (Year)")
    reference_source = settings.Setting(0)  # 0=OpenAlex, 1=Custom
    custom_file = settings.Setting("")
    
    year_from = settings.Setting(0)
    year_to = settings.Setting(0)
    
    threshold = settings.Setting(1.0)
    compute_chi_square = settings.Setting(True)
    
    auto_apply = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_column = Msg("Required column not found: {}")
        fetch_error = Msg("Error fetching reference data: {}")
        compute_error = Msg("Computation error: {}")
    
    class Warning(OWWidget.Warning):
        no_matches = Msg("No matching categories between dataset and reference")
        custom_file_error = Msg("Could not load custom reference file: {}")
        no_sdg = Msg("No SDG columns found (SDG01-SDG17). Run SDG identification first.")
    
    class Information(OWWidget.Information):
        compared = Msg("Compared {:,} categories: {} over, {} under-represented")
        using_openalex = Msg("Using OpenAlex global data as reference")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._result_df: Optional[pd.DataFrame] = None
        self._reference_df: Optional[pd.DataFrame] = None
        
        self._setup_control_area()
        self._setup_main_area()
    
    # =========================================================================
    # GUI SETUP
    # =========================================================================
    
    def _setup_control_area(self):
        """Build control area."""
        # Comparison Type
        type_box = gui.widgetBox(self.controlArea, "Comparison Type")
        
        type_layout = QGridLayout()
        type_layout.addWidget(QLabel("Compare:"), 0, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems(list(COMPARISON_TYPES.keys()))
        self.type_combo.setCurrentText(self.comparison_type)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo, 0, 1)
        type_box.layout().addLayout(type_layout)
        
        self.desc_label = QLabel("")
        self.desc_label.setStyleSheet("color: gray; font-style: italic;")
        self.desc_label.setWordWrap(True)
        type_box.layout().addWidget(self.desc_label)
        self._update_description()
        
        # Reference Data
        ref_box = gui.widgetBox(self.controlArea, "Reference Data")
        
        self.ref_buttons = QButtonGroup(self)
        
        self.openalex_rb = QRadioButton("OpenAlex (Global)")
        self.openalex_rb.setChecked(self.reference_source == 0)
        self.ref_buttons.addButton(self.openalex_rb, 0)
        ref_box.layout().addWidget(self.openalex_rb)
        
        self.custom_rb = QRadioButton("Custom Reference File")
        self.custom_rb.setChecked(self.reference_source == 1)
        self.ref_buttons.addButton(self.custom_rb, 1)
        ref_box.layout().addWidget(self.custom_rb)
        
        self.ref_buttons.buttonClicked.connect(self._on_ref_source_changed)
        
        # Custom file selection
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("No file selected")
        self.file_edit.setText(self.custom_file)
        self.file_edit.setReadOnly(True)
        file_layout.addWidget(self.file_edit)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.browse_btn)
        ref_box.layout().addLayout(file_layout)
        
        # Warning about OpenAlex
        self.openalex_warn = QLabel(
            "<small>⚠️ Using OpenAlex as reference. This represents global research "
            "patterns which may differ from your database (Scopus, WoS). "
            "For database-specific comparisons, provide custom reference data.</small>"
        )
        self.openalex_warn.setStyleSheet("color: #856404; background-color: #fff3cd; padding: 5px; border-radius: 3px;")
        self.openalex_warn.setWordWrap(True)
        ref_box.layout().addWidget(self.openalex_warn)
        
        self._update_ref_ui()
        
        # Year Range
        year_box = gui.widgetBox(self.controlArea, "Year Range")
        
        year_info = QLabel("<small>Filter years for comparison (leave empty for all years in dataset)</small>")
        year_info.setStyleSheet("color: gray;")
        year_box.layout().addWidget(year_info)
        
        year_layout = QGridLayout()
        year_layout.addWidget(QLabel("From:"), 0, 0)
        self.year_from_spin = QSpinBox()
        self.year_from_spin.setRange(0, 2100)
        self.year_from_spin.setValue(self.year_from)
        self.year_from_spin.setSpecialValueText("Auto")
        self.year_from_spin.valueChanged.connect(lambda v: setattr(self, 'year_from', v))
        year_layout.addWidget(self.year_from_spin, 0, 1)
        
        year_layout.addWidget(QLabel("To:"), 0, 2)
        self.year_to_spin = QSpinBox()
        self.year_to_spin.setRange(0, 2100)
        self.year_to_spin.setValue(self.year_to)
        self.year_to_spin.setSpecialValueText("Auto")
        self.year_to_spin.valueChanged.connect(lambda v: setattr(self, 'year_to', v))
        year_layout.addWidget(self.year_to_spin, 0, 3)
        
        year_box.layout().addLayout(year_layout)
        
        # Settings
        settings_box = gui.widgetBox(self.controlArea, "Settings")
        
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold (pp):"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.1, 50.0)
        self.thresh_spin.setValue(self.threshold)
        self.thresh_spin.setSingleStep(0.5)
        self.thresh_spin.setDecimals(1)
        self.thresh_spin.valueChanged.connect(lambda v: setattr(self, 'threshold', v))
        thresh_layout.addWidget(self.thresh_spin)
        settings_box.layout().addLayout(thresh_layout)
        
        thresh_info = QLabel("<small>Categories with difference > threshold are classified as over/under-represented</small>")
        thresh_info.setStyleSheet("color: gray;")
        thresh_info.setWordWrap(True)
        settings_box.layout().addWidget(thresh_info)
        
        self.chi_cb = QCheckBox("Compute chi-square test")
        self.chi_cb.setChecked(self.compute_chi_square)
        self.chi_cb.setEnabled(HAS_SCIPY)
        self.chi_cb.toggled.connect(lambda c: setattr(self, 'compute_chi_square', c))
        if not HAS_SCIPY:
            self.chi_cb.setToolTip("scipy not installed")
        settings_box.layout().addWidget(self.chi_cb)
        
        # Apply button
        self.apply_btn = QPushButton("Compare")
        self.apply_btn.setMinimumHeight(35)
        self.apply_btn.clicked.connect(self.commit)
        self.controlArea.layout().addWidget(self.apply_btn)
        
        gui.auto_apply(self.controlArea, self, "auto_apply")
    
    def _setup_main_area(self):
        """Build main area with results preview."""
        # Summary
        summary_box = gui.widgetBox(self.mainArea, "Comparison Summary")
        self.summary_label = QLabel("No data")
        self.summary_label.setWordWrap(True)
        summary_box.layout().addWidget(self.summary_label)
        
        # Results table
        results_box = gui.widgetBox(self.mainArea, "Comparison Results")
        self.results_table = QTableWidget()
        self.results_table.setMinimumHeight(300)
        results_box.layout().addWidget(self.results_table)
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_type_changed(self, comparison_type):
        self.comparison_type = comparison_type
        self._update_description()
        if self.auto_apply:
            self.commit()
    
    def _on_ref_source_changed(self, button):
        self.reference_source = self.ref_buttons.id(button)
        self._update_ref_ui()
        if self.auto_apply:
            self.commit()
    
    def _update_description(self):
        config = COMPARISON_TYPES.get(self.comparison_type, {})
        desc = config.get("description", "")
        self.desc_label.setText(f"<small>{desc}</small>")
    
    def _update_ref_ui(self):
        is_custom = self.reference_source == 1
        self.file_edit.setEnabled(is_custom)
        self.browse_btn.setEnabled(is_custom)
        self.openalex_warn.setVisible(not is_custom)
    
    def _browse_file(self):
        filters = "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference File", "", filters)
        if path:
            self.custom_file = path
            self.file_edit.setText(path)
            if self.auto_apply:
                self.commit()
    
    def commit(self):
        """Perform comparison."""
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
        
        if data is None:
            self.Error.no_data()
            self._update_results_display()
            return
        
        # Convert to DataFrame
        self._df = self._table_to_df(data)
        
        # Update year range defaults
        year_col = self._find_column(["Year", "Publication Year", "publication_year", "PY"])
        if year_col:
            years = pd.to_numeric(self._df[year_col], errors='coerce').dropna()
            if len(years) > 0:
                if self.year_from == 0:
                    self.year_from_spin.setValue(0)
                if self.year_to == 0:
                    self.year_to_spin.setValue(0)
        
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
        """Perform benchmarking comparison."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None or self._df.empty:
            self.Error.no_data()
            self._update_results_display()
            self._send_outputs(None)
            return
        
        try:
            config = COMPARISON_TYPES.get(self.comparison_type, {})
            category = config.get("category", "Year")
            value_type = config.get("value_type", "single")
            
            # Get observed distribution
            observed_df = self._get_observed_distribution(config)
            if observed_df is None or observed_df.empty:
                self._update_results_display()
                self._send_outputs(None)
                return
            
            # Get reference distribution
            reference_df = self._get_reference_distribution(category)
            if reference_df is None or reference_df.empty:
                self._update_results_display()
                self._send_outputs(None)
                return
            
            # Compute comparison
            result_df = self._compute_comparison(observed_df, reference_df, category)
            if result_df is None or result_df.empty:
                self.Warning.no_matches()
                self._update_results_display()
                self._send_outputs(None)
                return
            
            # Add chi-square if requested
            if self.compute_chi_square and HAS_SCIPY:
                result_df = self._add_chi_square(result_df)
            
            self._result_df = result_df
            
            # Update display
            self._update_results_display()
            
            # Send outputs
            self._send_outputs(result_df)
            
            # Info message
            over = (result_df["Difference (pp)"] > self.threshold).sum()
            under = (result_df["Difference (pp)"] < -self.threshold).sum()
            self.Information.compared(len(result_df), over, under)
            
            if self.reference_source == 0:
                self.Information.using_openalex()
            
        except Exception as e:
            import traceback
            logger.error(f"Computation error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
            self._update_results_display()
            self._send_outputs(None)
    
    def _get_observed_distribution(self, config):
        """Get distribution from dataset."""
        value_type = config.get("value_type", "single")
        category = config.get("category")
        columns = config.get("columns", [])
        
        # Special handling for SDG
        if value_type == "sdg":
            return self._get_sdg_distribution()
        
        # Find column
        col = self._find_column(columns)
        if col is None:
            self.Error.no_column(", ".join(columns[:3]))
            return None
        
        df = self._df.copy()
        
        # Filter by year range if applicable
        year_col = self._find_column(["Year", "Publication Year", "publication_year", "PY"])
        if year_col and (self.year_from > 0 or self.year_to > 0):
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
            if self.year_from > 0:
                df = df[df[year_col] >= self.year_from]
            if self.year_to > 0:
                df = df[df[year_col] <= self.year_to]
        
        # Detect separator
        sep = "; "
        sample = df[col].dropna().iloc[0] if len(df) > 0 else ""
        if "|" in str(sample):
            sep = "|"
        
        # Count values
        if value_type == "list":
            # Split and explode
            exploded = df[col].dropna().astype(str).str.split(sep).explode().str.strip()
            exploded = exploded[exploded != ""]
            counts = exploded.value_counts().reset_index()
            counts.columns = [category, "Count"]
        else:
            counts = df[col].value_counts().reset_index()
            counts.columns = [category, "Count"]
        
        return counts
    
    def _get_sdg_distribution(self):
        """Get SDG distribution from binary columns."""
        # Look for SDG01-SDG17 columns
        sdg_cols = [c for c in self._df.columns 
                    if c.startswith("SDG") and len(c) <= 5 and c[3:].isdigit()]
        
        if not sdg_cols:
            self.Warning.no_sdg()
            return None
        
        # Sum each SDG column
        sdg_counts = {}
        for col in sorted(sdg_cols):
            sdg_num = int(col[3:])
            sdg_label = f"SDG {sdg_num}"
            sdg_counts[sdg_label] = self._df[col].sum()
        
        return pd.DataFrame(list(sdg_counts.items()), columns=["SDG", "Count"])
    
    def _get_reference_distribution(self, category):
        """Get reference distribution."""
        if self.reference_source == 1:  # Custom file
            return self._load_custom_reference(category)
        
        # OpenAlex
        if not HAS_REPRESENTATION:
            self.Error.fetch_error("biblium.representation not available")
            return None
        
        try:
            # Get year range
            year_range = None
            year_col = self._find_column(["Year", "Publication Year", "publication_year", "PY"])
            if year_col:
                years = pd.to_numeric(self._df[year_col], errors='coerce').dropna()
                if len(years) > 0:
                    yr_from = self.year_from if self.year_from > 0 else int(years.min())
                    yr_to = self.year_to if self.year_to > 0 else int(years.max())
                    year_range = (yr_from, yr_to)
            
            # Fetch based on category
            if category == "Year":
                if year_range:
                    return fetch_openalex_yearly_counts(year_range[0], year_range[1])
                else:
                    return fetch_openalex_yearly_counts(2000, 2024)
            elif category == "Country":
                return fetch_openalex_country_counts(year_range=year_range)
            elif category == "SDG":
                return fetch_openalex_sdg_counts(year_range=year_range)
            elif category == "Document Type":
                return fetch_openalex_doctype_counts(year_range=year_range)
            elif category == "Open Access":
                return fetch_openalex_oa_counts(year_range=year_range)
            else:
                self.Error.fetch_error(f"Unknown category: {category}")
                return None
                
        except Exception as e:
            self.Error.fetch_error(str(e))
            return None
    
    def _load_custom_reference(self, category):
        """Load custom reference file."""
        if not self.custom_file or not os.path.exists(self.custom_file):
            self.Warning.custom_file_error("File not found")
            return None
        
        try:
            ext = os.path.splitext(self.custom_file)[1].lower()
            if ext in [".xlsx", ".xls"]:
                ref_df = pd.read_excel(self.custom_file)
            else:
                ref_df = pd.read_csv(self.custom_file)
            
            # Expect first column to be category, second to be count
            if len(ref_df.columns) < 2:
                self.Warning.custom_file_error("File must have at least 2 columns")
                return None
            
            ref_df.columns = [category, "Count"] + list(ref_df.columns[2:])
            return ref_df[[category, "Count"]]
            
        except Exception as e:
            self.Warning.custom_file_error(str(e))
            return None
    
    def _compute_comparison(self, observed_df, reference_df, category):
        """Compute relative representation."""
        if HAS_REPRESENTATION:
            try:
                return compute_relative_representation(
                    observed_df, reference_df,
                    category_col=category,
                    count_col="Count",
                    threshold=self.threshold,
                )
            except Exception as e:
                logger.warning(f"Biblium comparison failed: {e}")
                return self._compute_comparison_basic(observed_df, reference_df, category)
        else:
            return self._compute_comparison_basic(observed_df, reference_df, category)
    
    def _compute_comparison_basic(self, observed_df, reference_df, category):
        """Basic comparison without biblium."""
        observed = observed_df.copy()
        reference = reference_df.copy()
        
        # Calculate percentages
        obs_total = observed["Count"].sum()
        ref_total = reference["Count"].sum()
        
        if obs_total == 0 or ref_total == 0:
            return None
        
        observed["Observed %"] = observed["Count"] / obs_total * 100
        reference["Reference %"] = reference["Count"] / ref_total * 100
        
        # Rename columns
        observed = observed.rename(columns={"Count": "Observed Count"})
        reference = reference.rename(columns={"Count": "Reference Count"})
        
        # Merge
        merged = pd.merge(
            observed[[category, "Observed Count", "Observed %"]],
            reference[[category, "Reference Count", "Reference %"]],
            on=category,
            how="inner",
        )
        
        if len(merged) == 0:
            return None
        
        # Compute differences
        merged["Difference (pp)"] = merged["Observed %"] - merged["Reference %"]
        
        # Classification
        def classify(pp_diff):
            if pp_diff > self.threshold:
                return "Over-represented"
            elif pp_diff < -self.threshold:
                return "Under-represented"
            else:
                return "Similar"
        
        merged["Classification"] = merged["Difference (pp)"].apply(classify)
        
        # Sort by absolute difference
        merged["_abs_diff"] = merged["Difference (pp)"].abs()
        merged = merged.sort_values("_abs_diff", ascending=False).drop(columns=["_abs_diff"])
        
        return merged.reset_index(drop=True)
    
    def _add_chi_square(self, result_df):
        """Add chi-square test results."""
        if not HAS_SCIPY:
            return result_df
        
        try:
            observed = result_df["Observed Count"].values
            expected_pct = result_df["Reference %"].values / 100
            total_observed = observed.sum()
            expected = expected_pct * total_observed
            
            # Avoid zero expected values
            mask = expected > 0
            if mask.sum() < 2:
                return result_df
            
            chi2, p_value = stats.chisquare(observed[mask], expected[mask])
            
            result_df["Chi-square"] = chi2
            result_df["p-value"] = p_value
            
        except Exception as e:
            logger.warning(f"Chi-square computation failed: {e}")
        
        return result_df
    
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    def _send_outputs(self, result_df):
        """Send outputs."""
        if result_df is None:
            self.Outputs.comparison.send(None)
            self.Outputs.over_represented.send(None)
            self.Outputs.under_represented.send(None)
            return
        
        # Full comparison
        self.Outputs.comparison.send(self._df_to_table(result_df))
        
        # Over-represented
        over_df = result_df[result_df["Difference (pp)"] > self.threshold].copy()
        if not over_df.empty:
            self.Outputs.over_represented.send(self._df_to_table(over_df))
        else:
            self.Outputs.over_represented.send(None)
        
        # Under-represented
        under_df = result_df[result_df["Difference (pp)"] < -self.threshold].copy()
        if not under_df.empty:
            self.Outputs.under_represented.send(self._df_to_table(under_df))
        else:
            self.Outputs.under_represented.send(None)
    
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
        
        # Summary
        over = (df["Difference (pp)"] > self.threshold).sum()
        under = (df["Difference (pp)"] < -self.threshold).sum()
        similar = len(df) - over - under
        
        summary_parts = [
            f"<b>Comparison:</b> {self.comparison_type}",
            f"<b>Categories compared:</b> {len(df)}",
            f"<b>Over-represented:</b> {over}",
            f"<b>Under-represented:</b> {under}",
            f"<b>Similar:</b> {similar}",
            f"<b>Threshold:</b> {self.threshold} pp",
        ]
        
        # Add chi-square if present
        if "Chi-square" in df.columns and "p-value" in df.columns:
            chi2 = df["Chi-square"].iloc[0]
            p_val = df["p-value"].iloc[0]
            summary_parts.append(f"<b>Chi-square:</b> {chi2:.2f}, p={p_val:.4f}")
        
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
                    elif col in ["Observed %", "Reference %", "Difference (pp)"]:
                        text = f"{val:.2f}"
                    elif col == "p-value":
                        text = f"{val:.4f}"
                    elif val == int(val):
                        text = f"{int(val):,}"
                    else:
                        text = f"{val:.2f}"
                else:
                    text = str(val)[:50]
                
                item = QTableWidgetItem(text)
                
                # Color coding for classification
                if col == "Classification":
                    if val == "Over-represented":
                        item.setBackground(Qt.green)
                    elif val == "Under-represented":
                        item.setBackground(Qt.red)
                
                self.results_table.setItem(i, j, item)
        
        self.results_table.resizeColumnsToContents()


if __name__ == "__main__":
    WidgetPreview(OWBenchmarking).run()
