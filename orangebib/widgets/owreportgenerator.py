# -*- coding: utf-8 -*-
"""
Report Generator Widget
=======================
Generate full bibliometric reports (Word, Excel, PowerPoint, LaTeX) from a
bibliographic data table, using Biblium's templated reporting engine.

The widget builds a :class:`biblium.biblium_main.BiblioAnalysis` from the
input data, runs the analyses required for the chosen report level and writes
the requested output formats to disk. Generated file paths are emitted as a
table for downstream use.
"""

import os
import logging

# biblium renders figures with matplotlib while building reports; force a
# non-interactive backend so plotting from a background QThread cannot crash Qt.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:  # noqa: BLE001
    pass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QLabel, QLineEdit, QComboBox, QPushButton, QCheckBox,
    QGridLayout, QHBoxLayout, QFileDialog, QProgressBar,
    QTableWidget, QTableWidgetItem, QSpinBox, QListWidget, QListWidgetItem,
)

from Orange.data import Table, Domain, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

# --------------------------------------------------------------------------
# Biblium import (report generation has no fallback — it requires biblium)
# --------------------------------------------------------------------------
try:
    from biblium.biblium_main import BiblioAnalysis
    try:
        from biblium.reportbib import check_report_data_availability
    except Exception:  # noqa: BLE001
        check_report_data_availability = None
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    BiblioAnalysis = None

logger = logging.getLogger(__name__)


# Report detail levels exposed by BiblioAnalysis.generate_report().
REPORT_LEVELS = ["basic", "standard", "extended", "full"]

# Output formats: label -> biblium format code / file extension.
REPORT_FORMATS = [
    ("Word (.docx)", "docx"),
    ("Excel (.xlsx)", "xlsx"),
    ("PowerPoint (.pptx)", "pptx"),
    ("LaTeX (.tex)", "tex"),
    ("HTML dashboard (.html)", "html"),
]

# Database hints accepted by BiblioAnalysis (label -> db code).
DB_CHOICES = [
    ("Auto-detect", ""),
    ("Scopus", "scopus"),
    ("Web of Science", "wos"),
    ("OpenAlex", "oa"),
    ("PubMed", "pubmed"),
    ("Dimensions", "dimensions"),
    ("Lens", "lens"),
]


def _detect_list_separator(df, default=None):
    """Detect the list separator actually used in the keyword columns so the
    report engine splits multi-valued cells correctly (OpenAlex uses '|',
    Scopus/WoS use '; '). Returns None if undetectable."""
    try:
        import pandas as _pd
        cols = [c for c in ("Author Keywords", "Index Keywords", "Keywords",
                            "Authors", "Affiliations")
                if c in df.columns]
        sample = ""
        for c in cols:
            vals = df[c].dropna().astype(str)
            if len(vals):
                sample += " ".join(vals.head(100).tolist()) + " "
        for cand in ["||", "|", "; ", ";"]:
            if cand in sample:
                return cand
    except Exception:  # noqa: BLE001
        pass
    return default


def _table_to_df(table: Optional[Table]) -> pd.DataFrame:
    """Convert an Orange Table to a pandas DataFrame (attrs + class + metas)."""
    if table is None or len(table) == 0:
        return pd.DataFrame()
    data: Dict[str, object] = {}
    domain = table.domain
    groups = list(domain.attributes) + list(domain.class_vars) + list(domain.metas)
    for var in groups:
        try:
            col = table.get_column(var)
        except Exception:  # noqa: BLE001
            continue
        if var.is_discrete:
            values = var.values
            data[var.name] = [
                values[int(v)] if (v == v and 0 <= int(v) < len(values)) else ""
                for v in col
            ]
        else:
            data[var.name] = col
    return pd.DataFrame(data)


def _build_html_fallback(res_folder, html_path, title, note=""):
    """Build a self-contained HTML report WITHOUT bokeh, by embedding the PNG
    figures biblium saved to the results folder plus a title. Returns the path
    or None if nothing was found."""
    import base64
    import glob
    pngs = []
    for root in {res_folder, os.path.join(res_folder, "reports"),
                 os.path.join(res_folder, "figures"),
                 os.path.join(res_folder, "plots")}:
        if root and os.path.isdir(root):
            pngs.extend(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
    pngs = sorted(set(pngs))
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#222}",
        "h1{color:#2c3e50}figure{margin:24px 0;border:1px solid #eee;padding:12px;",
        "border-radius:8px}figcaption{color:#666;font-size:13px;margin-top:6px}",
        "img{max-width:100%;height:auto}.note{color:#999;font-size:12px}</style>",
        f"</head><body><h1>{title}</h1>",
        "<p class='note'>Static HTML report (interactive bokeh dashboard "
        "unavailable on this install).</p>",
    ]
    if not pngs:
        parts.append("<p>No figures were found to embed. Generate the report "
                     "with Word/PowerPoint figures enabled, or install bokeh for "
                     "the interactive dashboard.</p>")
    for fp in pngs:
        try:
            with open(fp, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
        except Exception:  # noqa: BLE001
            continue
        cap = os.path.splitext(os.path.basename(fp))[0].replace("_", " ")
        parts.append(f"<figure><img src='data:image/png;base64,{b64}'/>"
                     f"<figcaption>{cap}</figcaption></figure>")
    parts.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return html_path if pngs else html_path


class ReportWorker(QThread):
    """Background worker that runs the (potentially slow) report generation."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str)  # result dict, error string

    def __init__(self, df, db, output_dir, base_name, level, formats,
                 report_title, report_subtitle, custom_mode=False,
                 simple_max=0, plot_filter=None):
        super().__init__()
        self._df = df
        self._db = db
        self._output_dir = output_dir
        self._base_name = base_name
        self._level = level
        self._formats = formats
        self._title = report_title
        self._subtitle = report_subtitle
        self._custom_mode = custom_mode
        self._simple_max = simple_max
        self._plot_filter = plot_filter

    def run(self):
        try:
            try:
                import matplotlib
                matplotlib.use("Agg", force=True)
            except Exception:  # noqa: BLE001
                pass
            self.progress.emit(10, "Building analysis object...")
            ba = BiblioAnalysis(
                df=self._df,
                db=self._db or "",
                res_folder=self._output_dir,
                verbose=False,
            )
            _sep = _detect_list_separator(self._df)
            if _sep:
                try:
                    ba.default_separator = _sep
                except Exception:  # noqa: BLE001
                    pass
            # Optional report metadata, set when supported by the engine.
            for attr, value in (("report_title", self._title),
                                ("report_subtitle", self._subtitle)):
                if value:
                    try:
                        setattr(ba, attr, value)
                    except Exception:  # noqa: BLE001
                        pass

            self.progress.emit(30, f"Running '{self._level}' analyses...")
            output = os.path.join(self._output_dir, self._base_name)

            # --- Custom report: simple mode (only the chosen figures) ---
            if self._custom_mode:
                try:
                    ba.prepare_for_report(level=self._level, verbose=False)
                except Exception:  # noqa: BLE001
                    pass
                self.progress.emit(60, "Building custom report...")
                kwargs = {"use_simple_mode": True}
                if self._simple_max and self._simple_max > 0:
                    kwargs["simple_max_plots"] = self._simple_max
                if self._plot_filter:
                    kwargs["simple_plot_filter"] = self._plot_filter
                path = ba.save_report_to_word(output + ".docx", **kwargs)
                self.progress.emit(100, "Done")
                self.finished.emit({"docx (custom)": str(path)}, "")
                return

            bib_formats = [f for f in self._formats if f != "html"]
            result = {}
            if bib_formats:
                result = ba.generate_report(
                    output=output, level=self._level, formats=bib_formats,
                    prepare=True, verbose=False) or {}

            if "html" in self._formats:
                self.progress.emit(80, "Building HTML dashboard...")
                try:
                    if not bib_formats:
                        try:
                            ba.prepare_for_report(level=self._level, verbose=False)
                        except Exception:  # noqa: BLE001
                            pass
                        from biblium.dashboard import Dashboard
                    else:
                        from biblium.dashboard import Dashboard
                    html_path = output + ".html"
                    Dashboard(ba).create(
                        html_path,
                        title=self._title or "Bibliometric Dashboard")
                    result["html"] = html_path
                except Exception as exc:  # noqa: BLE001
                    logger.exception("HTML dashboard failed (bokeh?) — fallback")
                    try:
                        fb = _build_html_fallback(
                            self._output_dir, output + ".html",
                            self._title or "Bibliometric Dashboard", str(exc))
                        if fb:
                            result["html (no-bokeh)"] = fb
                        else:
                            result["html (failed)"] = str(exc)
                    except Exception as exc2:  # noqa: BLE001
                        logger.exception("HTML fallback failed")
                        result["html (failed)"] = f"{exc} / {exc2}"

            self.progress.emit(100, "Done")
            self.finished.emit(result, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Report generation failed")
            self.finished.emit(None, str(exc))


class ScanWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, df, db, output_dir, level):
        super().__init__()
        self._df = df; self._db = db; self._dir = output_dir; self._level = level

    def run(self):
        try:
            try:
                import matplotlib
                matplotlib.use("Agg", force=True)
            except Exception:  # noqa: BLE001
                pass
            ba = BiblioAnalysis(df=self._df, db=self._db or "",
                                res_folder=self._dir, verbose=False)
            _sep = _detect_list_separator(self._df)
            if _sep:
                try:
                    ba.default_separator = _sep
                except Exception:  # noqa: BLE001
                    pass
            try:
                ba.prepare_for_report(level=self._level, verbose=False)
            except Exception:  # noqa: BLE001
                pass
            if check_report_data_availability is None:
                self.finished.emit([], ""); return
            info = check_report_data_availability(ba, template_sheet=self._level,
                                                  verbose=False)
            avail = info.get("available", []) if isinstance(info, dict) else []
            rows = [(a.get("type", ""), a.get("name", "")) for a in avail]
            self.finished.emit(rows, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("scan failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWReportGenerator(OWWidget):
    """Generate templated bibliometric reports in multiple formats."""

    name = "Report Generator"
    description = "Generate Word/Excel/PowerPoint/LaTeX bibliometric reports"
    icon = "icons/report_generator.svg"
    priority = 900
    keywords = ["report", "export", "word", "docx", "excel", "powerpoint",
                "pptx", "latex", "document"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")

    class Outputs:
        report_files = Output("Report Files", Table,
                              doc="Generated report files (format, path)")
        data = Output("Data", Table, doc="Pass-through of the input data")

    # Settings
    output_dir = settings.Setting("")
    base_name = settings.Setting("bibliometric_report")
    level = settings.Setting("basic")
    fmt_docx = settings.Setting(True)
    fmt_xlsx = settings.Setting(True)
    fmt_pptx = settings.Setting(False)
    fmt_tex = settings.Setting(False)
    fmt_html = settings.Setting(False)
    db_code = settings.Setting("")
    report_title = settings.Setting("Bibliometric Analysis Report")
    report_subtitle = settings.Setting("")
    custom_mode = settings.Setting(False)
    simple_max = settings.Setting(0)
    plot_filter = settings.Setting("")

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium is required for report generation. "
                         "Install biblium>=2.16.")
        no_format = Msg("Select at least one output format")
        no_output_dir = Msg("Choose an output folder")
        generation_error = Msg("Report generation failed: {}")

    class Information(OWWidget.Information):
        generated = Msg("Generated {} report file(s)")

    def __init__(self):
        super().__init__()
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._worker: Optional[ReportWorker] = None

        if not self.output_dir:
            self.output_dir = os.path.join(os.path.expanduser("~"),
                                           "biblium_reports")

        self._setup_controls()
        self._setup_main_area()

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.generate_btn.setEnabled(False)

    # ------------------------------------------------------------------ GUI
    def _setup_controls(self):
        # Output location
        out_box = gui.widgetBox(self.controlArea, "Output")
        row = QHBoxLayout()
        self.dir_edit = QLineEdit(self.output_dir)
        self.dir_edit.editingFinished.connect(
            lambda: setattr(self, "output_dir", self.dir_edit.text()))
        row.addWidget(self.dir_edit)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_dir)
        row.addWidget(browse)
        out_box.layout().addLayout(row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("File base name:"))
        self.name_edit = QLineEdit(self.base_name)
        self.name_edit.editingFinished.connect(
            lambda: setattr(self, "base_name", self.name_edit.text() or "bibliometric_report"))
        name_row.addWidget(self.name_edit)
        out_box.layout().addLayout(name_row)

        # Report options
        opt_box = gui.widgetBox(self.controlArea, "Report")
        grid = QGridLayout()
        grid.addWidget(QLabel("Detail level:"), 0, 0)
        self.level_combo = QComboBox()
        self.level_combo.addItems(REPORT_LEVELS)
        self.level_combo.setCurrentText(self.level)
        self.level_combo.currentTextChanged.connect(
            lambda t: setattr(self, "level", t))
        grid.addWidget(self.level_combo, 0, 1)

        grid.addWidget(QLabel("Database:"), 1, 0)
        self.db_combo = QComboBox()
        for label, code in DB_CHOICES:
            self.db_combo.addItem(label, code)
        idx = max(0, [c for _, c in DB_CHOICES].index(self.db_code)
                  if self.db_code in [c for _, c in DB_CHOICES] else 0)
        self.db_combo.setCurrentIndex(idx)
        self.db_combo.currentIndexChanged.connect(
            lambda i: setattr(self, "db_code", self.db_combo.itemData(i)))
        grid.addWidget(self.db_combo, 1, 1)
        opt_box.layout().addLayout(grid)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        self.title_edit = QLineEdit(self.report_title)
        self.title_edit.editingFinished.connect(
            lambda: setattr(self, "report_title", self.title_edit.text()))
        title_row.addWidget(self.title_edit)
        opt_box.layout().addLayout(title_row)

        # Formats
        fmt_box = gui.widgetBox(self.controlArea, "Formats")
        self._fmt_boxes = {}
        for label, code in REPORT_FORMATS:
            cb = QCheckBox(label)
            cb.setChecked(getattr(self, f"fmt_{code}"))
            cb.toggled.connect(
                lambda checked, c=code: setattr(self, f"fmt_{c}", checked))
            fmt_box.layout().addWidget(cb)
            self._fmt_boxes[code] = cb

        # Custom report (simple mode: only the figures you choose)
        cbox = gui.widgetBox(self.controlArea, "Custom report")
        self.custom_cb = QCheckBox("Custom report (Word, only chosen figures)")
        self.custom_cb.setChecked(self.custom_mode)
        self.custom_cb.toggled.connect(lambda c: setattr(self, "custom_mode", c))
        cbox.layout().addWidget(self.custom_cb)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Max figures (0=all):"))
        self.maxfig_spin = QSpinBox(); self.maxfig_spin.setRange(0, 200)
        self.maxfig_spin.setValue(self.simple_max)
        self.maxfig_spin.valueChanged.connect(lambda v: setattr(self, "simple_max", v))
        crow.addWidget(self.maxfig_spin)
        cbox.layout().addLayout(crow)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Include matching:"))
        self.filter_edit = QLineEdit(self.plot_filter)
        self.filter_edit.setPlaceholderText("e.g. production, keywords (comma-sep; empty = all)")
        self.filter_edit.textChanged.connect(lambda t: setattr(self, "plot_filter", t))
        frow.addWidget(self.filter_edit)
        cbox.layout().addLayout(frow)
        self.scan_btn = QPushButton("Scan available items")
        self.scan_btn.clicked.connect(self._scan)
        cbox.layout().addWidget(self.scan_btn)
        cbox.layout().addWidget(QLabel("Include in custom report (tick items):"))
        self.items_list = QListWidget()
        self.items_list.setMaximumHeight(170)
        cbox.layout().addWidget(self.items_list)
        brow = QHBoxLayout()
        ball = QPushButton("All"); ball.clicked.connect(lambda: self._check_all(True))
        bnone = QPushButton("None"); bnone.clicked.connect(lambda: self._check_all(False))
        brow.addWidget(ball); brow.addWidget(bnone)
        cbox.layout().addLayout(brow)

        # Generate
        self.generate_btn = QPushButton("Generate Report")
        self.generate_btn.setMinimumHeight(36)
        self.generate_btn.clicked.connect(self._generate)
        self.controlArea.layout().addWidget(self.generate_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)

    def _setup_main_area(self):
        box = gui.widgetBox(self.mainArea, "Generated Files")
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(2)
        self.files_table.setHorizontalHeaderLabels(["Format", "Path"])
        self.files_table.horizontalHeader().setStretchLastSection(True)
        self.files_table.setMinimumHeight(300)
        self.files_table.cellDoubleClicked.connect(self._open_selected_row)
        box.layout().addWidget(self.files_table)

        row = QHBoxLayout()
        self.open_file_btn = QPushButton("Open selected file")
        self.open_file_btn.clicked.connect(self._open_selected_row)
        row.addWidget(self.open_file_btn)
        self.open_folder_btn = QPushButton("Open output folder")
        self.open_folder_btn.clicked.connect(self._open_folder)
        row.addWidget(self.open_folder_btn)
        box.layout().addLayout(row)

    @staticmethod
    def _open_path(path):
        import sys
        import subprocess
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # noqa: S606 - desktop convenience
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:  # noqa: BLE001
            logger.warning("Could not open %s", path)

    def _open_selected_row(self, *args):
        r = self.files_table.currentRow()
        if r < 0:
            return
        item = self.files_table.item(r, 1)
        if item and os.path.exists(item.text()):
            self._open_path(item.text())

    def _open_folder(self):
        if self.output_dir and os.path.isdir(self.output_dir):
            self._open_path(self.output_dir)

    # -------------------------------------------------------------- helpers
    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select output folder", self.output_dir or os.path.expanduser("~"))
        if path:
            self.output_dir = path
            self.dir_edit.setText(path)

    def _selected_formats(self) -> List[str]:
        return [code for _, code in REPORT_FORMATS if getattr(self, f"fmt_{code}")]

    # ---------------------------------------------------------------- input
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.Outputs.data.send(data)
        if data is None:
            self.Error.no_data()

    # ------------------------------------------------------------ generate
    def _generate(self):
        self.Error.clear()
        self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            return
        if self._df is None or self._df.empty:
            self.Error.no_data()
            return
        formats = self._selected_formats()
        if not formats and not self.custom_mode:
            self.Error.no_format()
            return
        if not self.output_dir:
            self.Error.no_output_dir()
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            self.Error.generation_error(str(exc))
            return

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")

        checked = self._checked_items() if hasattr(self, "items_list") else []
        if self.custom_mode and checked:
            plot_filter = checked
        else:
            plot_filter = [t.strip() for t in self.plot_filter.split(",") if t.strip()] or None
        self._worker = ReportWorker(
            df=self._df, db=self.db_code, output_dir=self.output_dir,
            base_name=self.base_name or "bibliometric_report",
            level=self.level, formats=formats,
            report_title=self.report_title, report_subtitle=self.report_subtitle,
            custom_mode=self.custom_mode, simple_max=self.simple_max,
            plot_filter=plot_filter,
        )
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _scan(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM or self._df is None or self._df.empty:
            self.Error.no_data() if (self._df is None) else self.Error.no_biblium()
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass
        self.scan_btn.setEnabled(False)
        self.status_label.setText("Scanning available items...")
        self._scan_worker = ScanWorker(self._df, self.db_code, self.output_dir, self.level)
        self._scan_worker.finished.connect(self._on_scan, Qt.QueuedConnection)
        self._scan_worker.start()

    def _on_scan(self, rows, error):
        self.scan_btn.setEnabled(True)
        if error or rows is None:
            self.status_label.setText("Scan failed")
            self.Error.generation_error(error or "scan error")
            return
        self.files_table.setRowCount(len(rows))
        self.files_table.setHorizontalHeaderLabels(["Type", "Item"])
        for r, (typ, name) in enumerate(rows):
            self.files_table.setItem(r, 0, QTableWidgetItem(str(typ)))
            self.files_table.setItem(r, 1, QTableWidgetItem(str(name)))
        self.files_table.resizeColumnsToContents()
        # checkable selection list
        self.items_list.clear()
        for (typ, name) in rows:
            it = QListWidgetItem(f"{name}  [{typ}]")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setData(Qt.UserRole, str(name))
            self.items_list.addItem(it)
        self.status_label.setText(f"{len(rows)} items available — tick the ones to "
                                  "include in the custom report.")

    def _check_all(self, state):
        for i in range(self.items_list.count()):
            self.items_list.item(i).setCheckState(
                Qt.Checked if state else Qt.Unchecked)

    def _checked_items(self):
        out = []
        for i in range(self.items_list.count()):
            it = self.items_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(str(it.data(Qt.UserRole)))
        return out

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, result: Optional[dict], error: str):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        if error or result is None:
            self.status_label.setText("Failed")
            self.Error.generation_error(error or "unknown error")
            self.Outputs.report_files.send(None)
            return

        # result: {format: path}
        rows = [(fmt, str(path)) for fmt, path in result.items()]
        self._populate_table(rows)
        self.status_label.setText(f"Done — {len(rows)} file(s) written")
        self.Information.generated(len(rows))
        self.Outputs.report_files.send(self._files_to_table(rows))

    def _populate_table(self, rows):
        self.files_table.setRowCount(len(rows))
        for r, (fmt, path) in enumerate(rows):
            self.files_table.setItem(r, 0, QTableWidgetItem(str(fmt)))
            self.files_table.setItem(r, 1, QTableWidgetItem(str(path)))
        self.files_table.resizeColumnsToContents()

    @staticmethod
    def _files_to_table(rows) -> Optional[Table]:
        if not rows:
            return None
        domain = Domain([], metas=[StringVariable("Format"), StringVariable("Path")])
        metas = np.array([[str(f), str(p)] for f, p in rows], dtype=object)
        return Table.from_numpy(domain, np.empty((len(rows), 0)), metas=metas)

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWReportGenerator).run()
