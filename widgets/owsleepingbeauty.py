# -*- coding: utf-8 -*-
"""
Sleeping Beauty Detection Widget
================================
Detect papers with delayed recognition (dormant period followed by awakening).

Uses Biblium's BiblioStats.extract_sleeping_beauties() method.
Requires OpenAlex data processed through Biblium.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox,
)
from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORT BIBLIUM
# =============================================================================

try:
    from biblium.bibstats import BiblioStats
    HAS_BIBLIUM = True
    logger.info("Biblium library available for Sleeping Beauty detection")
except ImportError:
    HAS_BIBLIUM = False
    logger.warning("Biblium not available - install with: pip install biblium")


# =============================================================================
# NUMERIC TABLE ITEM
# =============================================================================

class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically."""
    
    def __init__(self, display_text: str, sort_value: float):
        super().__init__(display_text)
        self._sort_value = sort_value
    
    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWSleepingBeauty(OWWidget):
    """Detect papers with delayed recognition (Sleeping Beauties).
    
    Uses Biblium's BiblioStats.extract_sleeping_beauties() method.
    
    Requires OpenAlex data with columns:
    - counts_by_year.year (pipe-separated years)
    - Citations by Year (pipe-separated citation counts)
    - Year (publication year)
    - Cited by (total citations)
    - Title, OpenAlex ID
    """
    
    name = "Sleeping Beauty"
    description = "Detect papers with delayed recognition (dormant period followed by awakening)"
    icon = "icons/sleeping_beauty.svg"
    priority = 69
    keywords = ["sleeping beauty", "delayed recognition", "citation", "awakening", "beauty coefficient"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data (OpenAlex with yearly citations)")
    
    class Outputs:
        sleeping_beauties = Output("Sleeping Beauties", Table, doc="Detected sleeping beauties")
        selected = Output("Selected", Table, doc="Selected sleeping beauties")
    
    # Settings - match Biblium's defaults from bibstats.py
    min_beauty_coefficient = settings.Setting(30)
    min_sleep_duration = settings.Setting(3)
    min_total_citations = settings.Setting(30)
    min_awakening_intensity = settings.Setting(1.5)
    current_year = settings.Setting(2025)
    auto_apply = settings.Setting(False)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium library required - install with: pip install biblium")
        not_openalex = Msg("Data must be OpenAlex format (requires counts_by_year.year, Citations by Year columns)")
        compute_error = Msg("Computation error: {}")
    
    class Warning(OWWidget.Warning):
        no_sleeping_beauties = Msg("No sleeping beauties found with current parameters")
        few_results = Msg("Only {} sleeping beauties found")
    
    class Information(OWWidget.Information):
        found = Msg("Found {} sleeping beauties from {} papers")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._bib: Optional["BiblioStats"] = None
        self._results_df: Optional[pd.DataFrame] = None
        self._selected_indices: List[int] = []
        
        self._setup_control_area()
        self._setup_main_area()
    
    def _setup_control_area(self):
        """Build control area."""
        # Info header
        info_box = gui.widgetBox(self.controlArea, "")
        status = "✓ Biblium available" if HAS_BIBLIUM else "⚠ Biblium required"
        info_label = QLabel(
            f"<b>😴 Sleeping Beauties</b><br>"
            f"<small>Detect papers with delayed recognition<br>"
            f"(dormant period followed by awakening)<br>"
            f"<i>{status}</i></small>"
        )
        info_label.setStyleSheet("color: #7c3aed; background-color: #ede9fe; padding: 8px; border-radius: 4px;")
        info_box.layout().addWidget(info_label)
        
        # Detection Parameters (matching Biblium's parameters)
        params_box = gui.widgetBox(self.controlArea, "⚙️ Detection Parameters")
        
        # Min Beauty Coefficient
        b_layout = QHBoxLayout()
        b_layout.addWidget(QLabel("Min Beauty Coefficient:"))
        self.b_spin = QSpinBox()
        self.b_spin.setRange(1, 500)
        self.b_spin.setValue(self.min_beauty_coefficient)
        self.b_spin.setToolTip("Higher B = stronger sleeping beauty pattern")
        self.b_spin.valueChanged.connect(self._on_param_changed)
        b_layout.addWidget(self.b_spin)
        params_box.layout().addLayout(b_layout)
        
        # Min Sleep Duration
        sleep_layout = QHBoxLayout()
        sleep_layout.addWidget(QLabel("Min Sleep Duration (yrs):"))
        self.sleep_spin = QSpinBox()
        self.sleep_spin.setRange(0, 50)
        self.sleep_spin.setValue(self.min_sleep_duration)
        self.sleep_spin.setToolTip("Minimum years with low citations before awakening")
        self.sleep_spin.valueChanged.connect(self._on_param_changed)
        sleep_layout.addWidget(self.sleep_spin)
        params_box.layout().addLayout(sleep_layout)
        
        # Min Total Citations
        cite_layout = QHBoxLayout()
        cite_layout.addWidget(QLabel("Min Total Citations:"))
        self.cite_spin = QSpinBox()
        self.cite_spin.setRange(1, 10000)
        self.cite_spin.setValue(self.min_total_citations)
        self.cite_spin.setToolTip("Minimum total citations required")
        self.cite_spin.valueChanged.connect(self._on_param_changed)
        cite_layout.addWidget(self.cite_spin)
        params_box.layout().addLayout(cite_layout)
        
        # Min Awakening Intensity
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Min Awakening Intensity:"))
        self.intensity_spin = QDoubleSpinBox()
        self.intensity_spin.setRange(0.0, 100.0)
        self.intensity_spin.setValue(self.min_awakening_intensity)
        self.intensity_spin.setSingleStep(0.5)
        self.intensity_spin.setToolTip("Ratio of post-awakening to pre-awakening citations")
        self.intensity_spin.valueChanged.connect(self._on_param_changed)
        intensity_layout.addWidget(self.intensity_spin)
        params_box.layout().addLayout(intensity_layout)
        
        # Current Year
        year_layout = QHBoxLayout()
        year_layout.addWidget(QLabel("Current Year:"))
        self.year_spin = QSpinBox()
        self.year_spin.setRange(2000, 2050)
        self.year_spin.setValue(self.current_year)
        self.year_spin.valueChanged.connect(self._on_param_changed)
        year_layout.addWidget(self.year_spin)
        params_box.layout().addLayout(year_layout)
        
        # Run button
        self.run_btn = gui.button(
            self.controlArea, self, "▶ Run Analysis",
            callback=self.commit, autoDefault=False
        )
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold;")
        
        gui.checkBox(self.controlArea, self, "auto_apply", "Auto apply")
        
        # About
        about_box = gui.widgetBox(self.controlArea, "ℹ️ About")
        about_label = QLabel(
            "<small>A <b>Sleeping Beauty</b> is a paper that receives "
            "few citations for years before suddenly gaining attention.<br><br>"
            "<b>Beauty Coefficient (B)</b> measures deviation from "
            "linear citation growth (higher = stronger pattern).<br><br>"
            "<b>Awakening Intensity</b> = citations after awakening / citations before.<br><br>"
            "Based on van Raan (2004) methodology.<br><br>"
            "<b>Required columns:</b><br>"
            "• counts_by_year.year<br>"
            "• Citations by Year<br>"
            "• Year, Cited by, Title</small>"
        )
        about_label.setWordWrap(True)
        about_label.setStyleSheet("color: #666;")
        about_box.layout().addWidget(about_label)
        
        self.controlArea.layout().addStretch(1)
    
    def _setup_main_area(self):
        """Build main area."""
        main_layout = QVBoxLayout()
        self.mainArea.layout().addLayout(main_layout)
        
        # Summary
        self.summary_label = QLabel("Load OpenAlex data with yearly citations to detect sleeping beauties")
        self.summary_label.setStyleSheet("font-size: 14px; color: #6c757d;")
        main_layout.addWidget(self.summary_label)
        
        # Statistics
        stats_box = QGroupBox("📊 Summary Statistics")
        stats_layout = QGridLayout(stats_box)
        
        self.stat_labels = {}
        stats = [
            ("papers", "Papers Analyzed:", "0"),
            ("beauties", "Sleeping Beauties:", "0"),
            ("avg_b", "Avg B Coefficient:", "0"),
            ("max_b", "Max B Coefficient:", "0"),
            ("avg_sleep", "Avg Sleep (years):", "0"),
            ("avg_intensity", "Avg Intensity:", "0"),
        ]
        
        for i, (key, label, default) in enumerate(stats):
            row, col = divmod(i, 3)
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: bold;")
            stats_layout.addWidget(lbl, row * 2, col)
            
            val_lbl = QLabel(default)
            val_lbl.setStyleSheet("font-size: 18px; color: #7c3aed;")
            stats_layout.addWidget(val_lbl, row * 2 + 1, col)
            self.stat_labels[key] = val_lbl
        
        main_layout.addWidget(stats_box)
        
        # Results table
        results_box = QGroupBox("😴 Sleeping Beauties")
        results_layout = QVBoxLayout(results_box)
        
        self.results_table = QTableWidget()
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.results_table.setSortingEnabled(True)
        self.results_table.itemSelectionChanged.connect(self._on_selection_changed)
        results_layout.addWidget(self.results_table)
        
        main_layout.addWidget(results_box, 1)
        
        # Export
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_btn = QPushButton("📥 Export to Excel")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        main_layout.addLayout(export_layout)
    
    def _on_param_changed(self):
        """Handle parameter change."""
        self.min_beauty_coefficient = self.b_spin.value()
        self.min_sleep_duration = self.sleep_spin.value()
        self.min_total_citations = self.cite_spin.value()
        self.min_awakening_intensity = self.intensity_spin.value()
        self.current_year = self.year_spin.value()
        
        if self.auto_apply and self._bib is not None:
            self.commit()
    
    def _on_selection_changed(self):
        """Handle table selection change."""
        self._selected_indices = list(set(
            item.row() for item in self.results_table.selectedItems() if item.column() == 0
        ))
        self._send_selected()
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive input data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._bib = None
        self._results_df = None
        
        self._clear_results()
        
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            return
        
        if data is None:
            self.Error.no_data()
            return
        
        # Convert to DataFrame
        self._df = self._table_to_df(data)
        
        # Check for required OpenAlex columns that Biblium expects
        required_cols = ["counts_by_year.year", "Citations by Year"]
        missing = [c for c in required_cols if c not in self._df.columns]
        
        if missing:
            self.Error.not_openalex()
            return
        
        # Create Biblium BiblioStats object
        try:
            self._bib = BiblioStats(df=self._df, db="oa")
            self.summary_label.setText(f"Loaded {len(self._df)} papers - click Run Analysis")
        except Exception as e:
            self.Error.compute_error(f"Failed to initialize Biblium: {e}")
            return
        
        if self.auto_apply:
            self.commit()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        """Convert Orange Table to pandas DataFrame."""
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)
    
    def _clear_results(self):
        """Clear results display."""
        self.summary_label.setText("Load OpenAlex data with yearly citations to detect sleeping beauties")
        for lbl in self.stat_labels.values():
            lbl.setText("0")
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.export_btn.setEnabled(False)
    
    def commit(self):
        """Run sleeping beauty detection using Biblium's BiblioStats."""
        self.Error.clear()
        self.Warning.clear()
        
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self._send_outputs(None)
            return
        
        if self._bib is None:
            self.Error.no_data()
            self._send_outputs(None)
            return
        
        try:
            # Call Biblium's extract_sleeping_beauties method
            self._results_df = self._bib.extract_sleeping_beauties(
                min_beauty_coefficient=float(self.min_beauty_coefficient),
                min_sleep_years=self.min_sleep_duration,
                min_total_citations=self.min_total_citations,
                min_awakening_intensity=float(self.min_awakening_intensity),
                current_year=self.current_year
            )
            
            if self._results_df is None or self._results_df.empty:
                self.Warning.no_sleeping_beauties()
                self._clear_results()
                self._send_outputs(None)
                return
            
            n_sb = len(self._results_df)
            if n_sb < 5:
                self.Warning.few_results(n_sb)
            
            self.Information.found(n_sb, len(self._df))
            
            # Update display
            self._update_results_display()
            
            # Send outputs
            self._send_outputs_from_results()
            
        except Exception as e:
            import traceback
            logger.error(f"Sleeping Beauty error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
            self._send_outputs(None)
    
    def _update_results_display(self):
        """Update results display."""
        if self._results_df is None or self._results_df.empty:
            return
        
        n_total = len(self._df)
        n_sb = len(self._results_df)
        
        self.summary_label.setText(
            f"<b>{n_sb:,}</b> sleeping beauties detected from <b>{n_total:,}</b> papers"
        )
        
        # Update statistics
        self.stat_labels["papers"].setText(f"{n_total:,}")
        self.stat_labels["beauties"].setText(f"{n_sb:,}")
        self.stat_labels["avg_b"].setText(f"{self._results_df['beauty_coefficient'].mean():.1f}")
        self.stat_labels["max_b"].setText(f"{self._results_df['beauty_coefficient'].max():.1f}")
        
        if "sleep_duration" in self._results_df.columns:
            valid_sleep = self._results_df["sleep_duration"].dropna()
            self.stat_labels["avg_sleep"].setText(f"{valid_sleep.mean():.1f}" if len(valid_sleep) > 0 else "N/A")
        
        if "awakening_intensity" in self._results_df.columns:
            valid_int = self._results_df["awakening_intensity"].replace([np.inf, -np.inf], np.nan).dropna()
            self.stat_labels["avg_intensity"].setText(f"{valid_int.mean():.1f}" if len(valid_int) > 0 else "N/A")
        
        # Populate table
        self.results_table.setSortingEnabled(False)
        self.results_table.clear()
        self.results_table.setRowCount(len(self._results_df))
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Title", "Year", "B Coef", "Sleep", "Awakening", "Citations", "Intensity"
        ])
        
        for i, (idx, row) in enumerate(self._results_df.iterrows()):
            title = str(row.get("title", ""))
            title_display = title[:50] + "..." if len(title) > 50 else title
            
            title_item = QTableWidgetItem(title_display)
            title_item.setToolTip(title)
            self.results_table.setItem(i, 0, title_item)
            
            pub_year = row.get("publication_year", 0)
            self.results_table.setItem(i, 1, NumericTableWidgetItem(str(int(pub_year)), pub_year))
            
            b_coef = row.get("beauty_coefficient", 0)
            self.results_table.setItem(i, 2, NumericTableWidgetItem(f"{b_coef:.1f}", b_coef))
            
            sleep = row.get("sleep_duration")
            sleep_str = str(int(sleep)) if pd.notna(sleep) else "N/A"
            self.results_table.setItem(i, 3, NumericTableWidgetItem(sleep_str, sleep if pd.notna(sleep) else 0))
            
            awake = row.get("awakening_year")
            awake_str = str(int(awake)) if pd.notna(awake) else "N/A"
            self.results_table.setItem(i, 4, NumericTableWidgetItem(awake_str, awake if pd.notna(awake) else 0))
            
            citations = row.get("total_citations", 0)
            self.results_table.setItem(i, 5, NumericTableWidgetItem(f"{int(citations):,}", citations))
            
            intensity = row.get("awakening_intensity", 0)
            if pd.notna(intensity) and intensity != float('inf'):
                self.results_table.setItem(i, 6, NumericTableWidgetItem(f"{intensity:.1f}", intensity))
            else:
                self.results_table.setItem(i, 6, NumericTableWidgetItem("∞", 9999))
        
        self.results_table.resizeColumnsToContents()
        self.results_table.setSortingEnabled(True)
        self.export_btn.setEnabled(True)
    
    def _send_outputs_from_results(self):
        """Create and send output tables."""
        if self._results_df is None or self._results_df.empty:
            self._send_outputs(None)
            return
        
        # Create Orange table with Biblium's output columns
        attrs = [
            ContinuousVariable("publication_year"),
            ContinuousVariable("beauty_coefficient"),
            ContinuousVariable("sleep_duration"),
            ContinuousVariable("awakening_year"),
            ContinuousVariable("total_citations"),
            ContinuousVariable("max_citations_in_year"),
            ContinuousVariable("awakening_intensity"),
        ]
        metas = [
            StringVariable("title"),
            StringVariable("paper_id"),
        ]
        
        domain = Domain(attrs, metas=metas)
        
        # Build arrays
        X = np.zeros((len(self._results_df), len(attrs)))
        for i, (idx, row) in enumerate(self._results_df.iterrows()):
            X[i, 0] = row.get("publication_year", 0)
            X[i, 1] = row.get("beauty_coefficient", 0)
            X[i, 2] = row.get("sleep_duration", np.nan) if pd.notna(row.get("sleep_duration")) else np.nan
            X[i, 3] = row.get("awakening_year", np.nan) if pd.notna(row.get("awakening_year")) else np.nan
            X[i, 4] = row.get("total_citations", 0)
            X[i, 5] = row.get("max_citations_in_year", 0)
            intensity = row.get("awakening_intensity", 0)
            X[i, 6] = intensity if pd.notna(intensity) and intensity != float('inf') else np.nan
        
        metas_arr = np.array([
            [str(row.get("title", "")), str(row.get("paper_id", ""))]
            for idx, row in self._results_df.iterrows()
        ], dtype=object)
        
        sb_table = Table.from_numpy(domain, X, metas=metas_arr)
        self._send_outputs(sb_table)
    
    def _send_selected(self):
        """Send selected sleeping beauties."""
        if self._results_df is None or not self._selected_indices:
            self.Outputs.selected.send(None)
            return
        
        selected_df = self._results_df.iloc[self._selected_indices]
        if selected_df.empty:
            self.Outputs.selected.send(None)
            return
        
        # Create table for selected
        attrs = [
            ContinuousVariable("publication_year"),
            ContinuousVariable("beauty_coefficient"),
            ContinuousVariable("sleep_duration"),
            ContinuousVariable("awakening_year"),
            ContinuousVariable("total_citations"),
            ContinuousVariable("awakening_intensity"),
        ]
        metas = [StringVariable("title"), StringVariable("paper_id")]
        domain = Domain(attrs, metas=metas)
        
        X = np.zeros((len(selected_df), len(attrs)))
        for i, (idx, row) in enumerate(selected_df.iterrows()):
            X[i, 0] = row.get("publication_year", 0)
            X[i, 1] = row.get("beauty_coefficient", 0)
            X[i, 2] = row.get("sleep_duration", np.nan) if pd.notna(row.get("sleep_duration")) else np.nan
            X[i, 3] = row.get("awakening_year", np.nan) if pd.notna(row.get("awakening_year")) else np.nan
            X[i, 4] = row.get("total_citations", 0)
            intensity = row.get("awakening_intensity", 0)
            X[i, 5] = intensity if pd.notna(intensity) and intensity != float('inf') else np.nan
        
        metas_arr = np.array([
            [str(row.get("title", "")), str(row.get("paper_id", ""))]
            for idx, row in selected_df.iterrows()
        ], dtype=object)
        
        sel_table = Table.from_numpy(domain, X, metas=metas_arr)
        self.Outputs.selected.send(sel_table)
    
    def _send_outputs(self, sb_table: Optional[Table]):
        """Send outputs."""
        self.Outputs.sleeping_beauties.send(sb_table)
        self.Outputs.selected.send(None)
    
    def _export_results(self):
        """Export results to Excel."""
        if self._results_df is None or self._results_df.empty:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Sleeping Beauties", "sleeping_beauties.xlsx",
            "Excel files (*.xlsx);;All files (*)"
        )
        
        if not filepath:
            return
        
        try:
            export_df = self._results_df.drop(columns=["citation_history"], errors="ignore")
            export_df.to_excel(filepath, index=False)
            QMessageBox.information(self, "Exported", f"Results saved to {filepath}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not export: {e}")
    
    def get_bib(self) -> Optional["BiblioStats"]:
        """Get the Biblium BiblioStats object (for use by plot widget)."""
        return self._bib
    
    def get_results_dataframe(self) -> Optional[pd.DataFrame]:
        """Get results DataFrame (for use by plot widget)."""
        return self._results_df


if __name__ == "__main__":
    WidgetPreview(OWSleepingBeauty).run()
