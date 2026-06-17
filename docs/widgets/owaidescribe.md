# AI Describe

> LLM-generated narrative description of any table.

```{figure} ../_static/img/owaidescribe.png
:alt: AI Describe
:class: widget-screenshot

The AI Describe widget.
```

## Overview

Sends a table to a large language model and returns a concise, readable
**narrative description** of it — handy for turning a statistics or counts table
into prose for a report. Works with several providers; the input data passes
through unchanged.

## Inputs
- **Data** (`Table`) — any table to describe.

## Outputs
- **Description** (`Table`) — the generated text.
- **Data** (`Table`) — pass-through of the input.

## Controls
- **Provider** — the LLM provider.
- **Model (optional)** — a specific model name.
- **API key** — provider credentials.
- **Max rows** — how many rows to send (cost/length control).
- **Custom prompt (optional)** — override the default description prompt.

**Actions:** `Describe`.

## Note
Sends data to an external LLM service — review provider terms before using on
sensitive corpora.
