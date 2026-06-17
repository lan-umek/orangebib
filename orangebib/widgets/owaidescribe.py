# -*- coding: utf-8 -*-
"""
AI Describe Widget
=================
Generate a natural-language description / interpretation of any table using an
LLM (via `biblium.llm_utils.llm_describe_table`). Connect it after any widget
that outputs a table (counts, statistics, group results, ...) to get a written
summary you can drop into a report.

Requires an LLM provider + API key (HuggingFace, OpenAI or Anthropic). The key
can be typed here or read from the usual environment variables.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QLabel, QComboBox, QLineEdit, QPushButton, QPlainTextEdit, QTextEdit,
    QSpinBox, QGridLayout,
)

from Orange.data import Table, Domain, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

try:
    from biblium.llm_utils import llm_describe_table
    HAS_BIBLIUM = True
except Exception:  # noqa: BLE001
    HAS_BIBLIUM = False
    llm_describe_table = None

logger = logging.getLogger(__name__)

PROVIDERS = ["huggingface", "openai", "anthropic"]
ENV_KEYS = {
    "huggingface": ["HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}


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


class DescribeWorker(QThread):
    finished = pyqtSignal(str, str)  # text, error

    def __init__(self, df, provider, model, api_key, custom_prompt, max_rows):
        super().__init__()
        self._df = df; self._provider = provider; self._model = model or None
        self._api_key = api_key or None; self._custom = custom_prompt or None
        self._max_rows = max_rows

    def run(self):
        try:
            text = llm_describe_table(
                self._df, provider=self._provider, model=self._model,
                api_key=self._api_key, custom_prompt=self._custom,
                max_rows=self._max_rows)
            self.finished.emit(str(text), "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("AI describe failed")
            self.finished.emit("", f"{type(exc).__name__}: {exc}")


class OWAIDescribe(OWWidget):
    """Describe a table in natural language using an LLM."""

    name = "AI Describe"
    description = "Generate a natural-language description of a table with an LLM"
    icon = "icons/ai_describe.svg"
    priority = 850
    keywords = ["ai", "llm", "describe", "summary", "interpret", "gpt",
                "claude", "explain"]
    category = "Biblium"

    class Inputs:
        data = Input("Data", Table, doc="Any table to describe")

    class Outputs:
        description = Output("Description", Table, doc="The generated text")
        data = Output("Data", Table, doc="Pass-through of the input data")

    provider = settings.Setting("huggingface")
    model = settings.Setting("")
    api_key = settings.Setting("")
    custom_prompt = settings.Setting("")
    max_rows = settings.Setting(50)

    want_main_area = True
    resizing_enabled = True

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_biblium = Msg("Biblium is required (biblium>=2.16).")
        describe_error = Msg("{}")

    class Information(OWWidget.Information):
        done = Msg("Description generated")

    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._worker = None

        box = gui.widgetBox(self.controlArea, "LLM")
        grid = QGridLayout()
        grid.addWidget(QLabel("Provider:"), 0, 0)
        self.prov_combo = QComboBox(); self.prov_combo.addItems(PROVIDERS)
        self.prov_combo.setCurrentText(self.provider)
        self.prov_combo.currentTextChanged.connect(lambda t: setattr(self, "provider", t))
        grid.addWidget(self.prov_combo, 0, 1)
        grid.addWidget(QLabel("Model (optional):"), 1, 0)
        self.model_edit = QLineEdit(self.model)
        self.model_edit.setPlaceholderText("leave empty for provider default")
        self.model_edit.textChanged.connect(lambda t: setattr(self, "model", t))
        grid.addWidget(self.model_edit, 1, 1)
        grid.addWidget(QLabel("API key:"), 2, 0)
        self.key_edit = QLineEdit(self.api_key)
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("or set the provider env variable")
        self.key_edit.textChanged.connect(lambda t: setattr(self, "api_key", t))
        grid.addWidget(self.key_edit, 2, 1)
        grid.addWidget(QLabel("Max rows:"), 3, 0)
        self.rows_spin = QSpinBox(); self.rows_spin.setRange(5, 500)
        self.rows_spin.setValue(self.max_rows)
        self.rows_spin.valueChanged.connect(lambda v: setattr(self, "max_rows", v))
        grid.addWidget(self.rows_spin, 3, 1)
        box.layout().addLayout(grid)

        pbox = gui.widgetBox(self.controlArea, "Custom prompt (optional)")
        self.prompt_edit = QPlainTextEdit(self.custom_prompt)
        self.prompt_edit.setPlaceholderText(
            "e.g. 'Summarise the main trends in 3 sentences for a report.'")
        self.prompt_edit.setMaximumHeight(80)
        self.prompt_edit.textChanged.connect(
            lambda: setattr(self, "custom_prompt", self.prompt_edit.toPlainText()))
        pbox.layout().addWidget(self.prompt_edit)

        self.run_btn = QPushButton("Describe")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._compute)
        self.controlArea.layout().addWidget(self.run_btn)
        self.status_label = QLabel(""); self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
        self.controlArea.layout().addStretch(1)

        out = gui.widgetBox(self.mainArea, "Description")
        self.text_view = QTextEdit(); self.text_view.setReadOnly(True)
        out.layout().addWidget(self.text_view)

        if not HAS_BIBLIUM:
            self.Error.no_biblium()
            self.run_btn.setEnabled(False)

    def _resolved_key(self):
        if self.api_key.strip():
            return self.api_key.strip()
        for env in ENV_KEYS.get(self.provider, []):
            if os.environ.get(env):
                return os.environ[env]
        return ""

    @Inputs.data
    def set_data(self, data):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium()
        self._data = data
        self._df = _table_to_df(data) if data is not None else None
        self.Outputs.data.send(data)
        if data is None:
            self.Error.no_data()

    def _compute(self):
        self.Error.clear(); self.Information.clear()
        if not HAS_BIBLIUM:
            self.Error.no_biblium(); return
        if self._df is None or self._df.empty:
            self.Error.no_data(); return
        key = self._resolved_key()
        if not key and self.provider != "huggingface":
            self.Error.describe_error(
                f"An API key is required for provider '{self.provider}'.")
            return
        self.run_btn.setEnabled(False)
        self.status_label.setText("Querying the model...")
        self._worker = DescribeWorker(
            self._df, self.provider, self.model, key,
            self.custom_prompt, self.max_rows)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.start()

    def _on_finished(self, text, error):
        self.run_btn.setEnabled(True)
        if error or not text:
            self.status_label.setText("Failed")
            self.Error.describe_error(error or "empty response")
            self.Outputs.description.send(None)
            return
        self.text_view.setPlainText(text)
        self.status_label.setText("Done")
        self.Information.done()
        domain = Domain([], metas=[StringVariable("Description")])
        M = np.array([[text]], dtype=object)
        self.Outputs.description.send(
            Table.from_numpy(domain, np.empty((1, 0)), metas=M))

    def onDeleteWidget(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWAIDescribe).run()
