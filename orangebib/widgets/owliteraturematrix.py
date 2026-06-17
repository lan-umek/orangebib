# -*- coding: utf-8 -*-
"""
Literature Matrix Widget
=======================
Extract structured information from each paper with an LLM and assemble a
"literature matrix": one row per paper, one column per question you ask
(e.g. research question, method, sample, key finding). This is an
extraction-style review aid built on `biblium.llm_utils.invoke_llm_batch`.

Requires an LLM provider + API key (HuggingFace, OpenAI or Anthropic). The key
can be typed here or read from the usual environment variables.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QLineEdit, QPushButton, QPlainTextEdit,
    QGridLayout, QTableWidget, QTableWidgetItem, QProgressBar,
)

from Orange.data import Table, Domain, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from biblium.llm_utils import invoke_llm_batch
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    invoke_llm_batch = None

PROVIDERS = ["huggingface", "openai", "anthropic"]
TEXT_CANDIDATES = ["Abstract", "Processed Abstract", "Combined Text", "Title",
                   "Document Title", "AB"]
TITLE_CANDIDATES = ["Title", "Document Title", "TI"]


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


class MatrixWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, texts, titles, fields, provider, model, api_key):
        super().__init__()
        self._texts = texts; self._titles = titles; self._fields = fields
        self._provider = provider; self._model = model or None
        self._api_key = api_key or None

    def run(self):
        try:
            prompts, index = [], []
            for ri, text in enumerate(self._texts):
                snippet = str(text)[:3000]
                for fi, field in enumerate(self._fields):
                    prompts.append(
                        "You are extracting information from a research paper.\n"
                        f"Text:\n{snippet}\n\n"
                        f"Question: {field}\n"
                        "Answer concisely (one short phrase or sentence). "
                        "If the text does not say, answer 'N/A'.\nAnswer:")
                    index.append((ri, fi))
            self.progress.emit(f"Querying LLM for {len(prompts)} cells...")
            answers = invoke_llm_batch(
                prompts, provider=self._provider, model=self._model,
                api_key=self._api_key, show_progress=False, max_tokens=120)
            grid = [["" for _ in self._fields] for _ in self._texts]
            for (ri, fi), ans in zip(index, answers):
                grid[ri][fi] = (str(ans).strip().replace("\n", " ")
                                if ans is not None else "")
            rows = []
            for ri in range(len(self._texts)):
                row = {"Title": str(self._titles[ri])[:120]}
                for fi, field in enumerate(self._fields):
                    row[field] = grid[ri][fi]
                rows.append(row)
            self.finished.emit(pd.DataFrame(rows), "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("literature matrix failed")
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OWLiteratureMatrix(OWWidget):
    """Extraction-style literature matrix built with an LLM."""

    name = "Literature Matrix"
    description = "Extract structured fields from each paper into a review matrix (LLM)"
    icon = "icons/literature_matrix.svg"
    priority = 860
    keywords = ["literature", "matrix", "extraction", "review", "llm", "ai",
                "screening", "synthesis", "questions"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data (needs Abstract/Title)")

    class Outputs:
        matrix = Output("Matrix", Table, doc="One row per paper, one column per question")

    text_col = settings.Setting("")
    fields_str = settings.Setting("Research question; Method; Sample / data; Key finding")
    provider = settings.Setting("huggingface")
    model = settings.Setting("")
    api_key = settings.Setting("")
    max_papers = settings.Setting(20)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium llm_utils are required (biblium>=2.16).")
        no_fields = Msg("Enter at least one question/column")
        compute_error = Msg("Extraction error: {}")

    class Information(OWWidget.Information):
        done = Msg("Extracted {} papers x {} fields")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "Source")
        grid = QGridLayout()
        grid.addWidget(QLabel("Text column:"), 0, 0)
        self.text_combo = QComboBox()
        self.text_combo.currentTextChanged.connect(lambda t: setattr(self, "text_col", t))
        grid.addWidget(self.text_combo, 0, 1)
        box.layout().addLayout(grid)
        gui.spin(box, self, "max_papers", 1, 200, label="Max papers:")

        fbox = gui.widgetBox(self.controlArea, "Questions / columns (one per line or ';')")
        self.fields_edit = QPlainTextEdit(self.fields_str.replace("; ", "\n"))
        self.fields_edit.setMaximumHeight(110)
        self.fields_edit.textChanged.connect(self._on_fields_changed)
        fbox.layout().addWidget(self.fields_edit)

        lbox = gui.widgetBox(self.controlArea, "LLM")
        lg = QGridLayout()
        lg.addWidget(QLabel("Provider:"), 0, 0)
        self.prov_combo = QComboBox(); self.prov_combo.addItems(PROVIDERS)
        self.prov_combo.setCurrentText(self.provider)
        self.prov_combo.currentTextChanged.connect(lambda t: setattr(self, "provider", t))
        lg.addWidget(self.prov_combo, 0, 1)
        lg.addWidget(QLabel("Model:"), 1, 0)
        self.model_edit = QLineEdit(self.model)
        self.model_edit.setPlaceholderText("leave empty for provider default")
        self.model_edit.textChanged.connect(lambda t: setattr(self, "model", t))
        lg.addWidget(self.model_edit, 1, 1)
        lg.addWidget(QLabel("API key:"), 2, 0)
        self.key_edit = QLineEdit(self.api_key)
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("or via environment variable")
        self.key_edit.textChanged.connect(lambda t: setattr(self, "api_key", t))
        lg.addWidget(self.key_edit, 2, 1)
        lbox.layout().addLayout(lg)

        self.run_btn = QPushButton("Extract"); self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        self.table = QTableWidget()
        self.mainArea.layout().addWidget(self.table)

        if not HAS_BIBLIUM:
            self.Error.no_biblium(); self.run_btn.setEnabled(False)

    def _on_fields_changed(self):
        self.fields_str = self.fields_edit.toPlainText()

    def _fields(self) -> List[str]:
        raw = self.fields_edit.toPlainText().replace(";", "\n")
        return [f.strip() for f in raw.split("\n") if f.strip()]

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.text_combo.blockSignals(True)
        self.text_combo.clear()
        if self._df is not None and not self._df.empty:
            cols = [c for c in TEXT_CANDIDATES if c in self._df.columns]
            cols += [c for c in self._df.columns if c not in cols]
            self.text_combo.addItems(cols)
            if self.text_col in cols:
                self.text_combo.setCurrentText(self.text_col)
            elif cols:
                self.text_col = cols[0]
        self.text_combo.blockSignals(False)
        if data is None:
            self.Error.no_data()

    def _compute(self):
        self.Error.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        fields = self._fields()
        if not fields:
            self.Error.no_fields(); return
        tcol = self.text_combo.currentText()
        if not tcol or tcol not in self._df.columns:
            self.Error.no_data(); return
        sub = self._df.head(self.max_papers)
        texts = sub[tcol].astype(str).tolist()
        title_col = next((c for c in TITLE_CANDIDATES if c in self._df.columns), tcol)
        titles = sub[title_col].astype(str).tolist()
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")
        self._worker = MatrixWorker(texts, titles, fields, self.provider,
                                    self.model, self.api_key)
        self._worker.progress.connect(lambda m: self.status_label.setText(m), Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, df, error):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False); self.progress_bar.setRange(0, 100)
        if error or df is None or df.empty:
            self.status_label.setText("Failed")
            self.Error.compute_error(error or "no result")
            self.Outputs.matrix.send(None)
            return
        self._fill_table(df)
        nfields = len(df.columns) - 1
        self.status_label.setText(f"Done — {len(df)} papers")
        self.Information.done(len(df), nfields)
        self.Outputs.matrix.send(self._df_to_table(df))

    def _fill_table(self, df):
        self.table.clear()
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
        self.table.resizeColumnsToContents()

    @staticmethod
    def _df_to_table(df):
        metas = [StringVariable(str(c)) for c in df.columns]
        M = df.astype(str).values
        return Table.from_numpy(Domain([], metas=metas),
                                np.empty((len(df), 0)), metas=M)

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWLiteratureMatrix).run()
