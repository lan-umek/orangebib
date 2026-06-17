# -*- coding: utf-8 -*-
"""
Text Preprocessing Widget
========================
Tokenize, lowercase, lemmatize and remove stopwords from a text column, adding a
``Processed <column>`` column. Wraps `biblium.utilsbib.process_text_column`,
which supports an extended stopword list loaded from an Excel file plus extra
ad-hoc stopwords.
"""

import os
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (QLabel, QComboBox, QPushButton, QGridLayout,
                             QProgressBar, QLineEdit, QPlainTextEdit, QCheckBox,
                             QFileDialog, QHBoxLayout)

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from biblium.utilsbib import process_text_column
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    process_text_column = None
    HAS_BIBLIUM = False

TEXT_CANDIDATES = ["Abstract", "Title", "Document Title", "Combined Text",
                   "Author Keywords", "Index Keywords", "AB", "TI"]


def _default_stopwords_file():
    """Bundled extended stopword list shipped with the add-on."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "data", "stopwords.xlsx")
    return path if os.path.exists(path) else ""


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
        if s.notna().mean() > 0.8 and df[c].dtype != object:
            attrs.append(ContinuousVariable(str(c))); X.append(s.fillna(0).values)
        else:
            metas.append(StringVariable(str(c))); M.append(df[c].astype(str).values)
    n = len(df)
    Xarr = np.column_stack(X) if X else np.empty((n, 0))
    Marr = np.column_stack(M) if M else np.empty((n, 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), Xarr, metas=Marr)


class PreprocessWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, df, column, stopwords_file, extra, exclude_cats,
                 remove_numbers, remove_two_letter):
        super().__init__()
        self._df = df; self._col = column
        self._sw_file = stopwords_file or None
        self._extra = extra; self._exclude = exclude_cats
        self._rn = remove_numbers; self._rtl = remove_two_letter

    def run(self):
        try:
            self.progress.emit("Lemmatizing + removing stopwords...")
            kwargs = dict(remove_numbers=self._rn,
                          remove_two_letter_words=self._rtl)
            if self._sw_file:
                kwargs["stopwords_file"] = self._sw_file
            if self._extra:
                kwargs["extra_stopwords"] = self._extra
            if self._exclude:
                kwargs["exclude_specific_stopwords"] = self._exclude
            out = process_text_column(self._df.copy(), self._col, **kwargs)
            self.finished.emit(out, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("text preprocessing failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWTextPreprocess(OWWidget):
    """Lemmatize + remove stopwords from a text column."""

    name = "Text Preprocessing"
    description = "Lemmatize and remove stopwords (with an extended stopword file)"
    icon = "icons/text_preprocess.svg"
    priority = 80
    keywords = ["text", "preprocess", "lemmatize", "stopwords", "clean",
                "tokenize", "nlp"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data with a text column")

    class Outputs:
        data = Output("Data", Table, doc="Data with a 'Processed <column>' column")

    column_name = settings.Setting("")
    stopwords_file = settings.Setting("")  # resolved to bundled list if empty
    extra_stopwords = settings.Setting("")
    exclude_categories = settings.Setting("")
    remove_numbers = settings.Setting(True)
    remove_two_letter = settings.Setting(True)

    want_main_area = False
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium is required (biblium>=2.16).")
        compute_error = Msg("Processing error: {}")

    class Information(OWWidget.Information):
        done = Msg("Processed column 'Processed {}'")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Text column")
        grid = QGridLayout()
        grid.addWidget(QLabel("Column:"), 0, 0)
        self.col_combo = QComboBox()
        self.col_combo.currentTextChanged.connect(lambda t: setattr(self, "column_name", t))
        grid.addWidget(self.col_combo, 0, 1)
        box.layout().addLayout(grid)

        sbox = gui.widgetBox(self.controlArea, "Stopwords")
        row = QHBoxLayout()
        if not self.stopwords_file:
            self.stopwords_file = _default_stopwords_file()
        self.file_edit = QLineEdit(self.stopwords_file)
        self.file_edit.setPlaceholderText("Excel stopwords file (optional)")
        self.file_edit.textChanged.connect(lambda t: setattr(self, "stopwords_file", t))
        row.addWidget(self.file_edit)
        browse = QPushButton("…"); browse.setMaximumWidth(32)
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        sbox.layout().addLayout(row)
        sbox.layout().addWidget(QLabel("Extra stopwords (comma/space separated):"))
        self.extra_edit = QPlainTextEdit(self.extra_stopwords)
        self.extra_edit.setMaximumHeight(70)
        self.extra_edit.textChanged.connect(
            lambda: setattr(self, "extra_stopwords", self.extra_edit.toPlainText()))
        sbox.layout().addWidget(self.extra_edit)
        sbox.layout().addWidget(QLabel("Specific categories to apply (comma sep):"))
        self.cat_edit = QLineEdit(self.exclude_categories)
        self.cat_edit.setPlaceholderText("e.g. methods, generic")
        self.cat_edit.textChanged.connect(lambda t: setattr(self, "exclude_categories", t))
        sbox.layout().addWidget(self.cat_edit)

        obox = gui.widgetBox(self.controlArea, "Options")
        self.rn_cb = QCheckBox("Remove numbers"); self.rn_cb.setChecked(self.remove_numbers)
        self.rn_cb.toggled.connect(lambda c: setattr(self, "remove_numbers", c))
        obox.layout().addWidget(self.rn_cb)
        self.rtl_cb = QCheckBox("Remove two-letter words")
        self.rtl_cb.setChecked(self.remove_two_letter)
        self.rtl_cb.toggled.connect(lambda c: setattr(self, "remove_two_letter", c))
        obox.layout().addWidget(self.rtl_cb)

        self.run_btn = QPushButton("Process"); self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        if not HAS_BIBLIUM:
            self.Error.no_biblium(); self.run_btn.setEnabled(False)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Stopwords file", "", "Excel (*.xlsx *.xls);;All files (*)")
        if path:
            self.stopwords_file = path
            self.file_edit.setText(path)

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.col_combo.blockSignals(True)
        self.col_combo.clear()
        if self._df is not None and not self._df.empty:
            cols = [c for c in TEXT_CANDIDATES if c in self._df.columns]
            cols += [c for c in self._df.columns if c not in cols and
                     self._df[c].dtype == object]
            self.col_combo.addItems(cols)
            if self.column_name in cols:
                self.col_combo.setCurrentText(self.column_name)
            elif cols:
                self.column_name = cols[0]
        self.col_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _parse_list(self, txt) -> List[str]:
        out = []
        for tok in str(txt).replace(",", "\n").replace(";", "\n").split("\n"):
            tok = tok.strip()
            if tok:
                out.append(tok)
        return out

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        col = self.col_combo.currentText()
        if not col or col not in self._df.columns:
            self.Error.no_data(); return
        extra = self._parse_list(self.extra_edit.toPlainText())
        cats = self._parse_list(self.cat_edit.text())
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = PreprocessWorker(
            self._df, col, self.stopwords_file.strip(), extra, cats,
            self.remove_numbers, self.remove_two_letter)
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(lambda o, e: self._on_finished(o, e, col), Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, out, error, col):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or out is None:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "unknown error")
            self.Outputs.data.send(None)
            return
        pcol = f"Processed {col}"
        nonempty = int(out[pcol].notna().sum()) if pcol in out.columns else 0
        self.status_label.setText(f"Done — {nonempty} documents processed")
        self.Information.done(col)
        self.Outputs.data.send(_df_to_table(out))

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWTextPreprocess).run()
