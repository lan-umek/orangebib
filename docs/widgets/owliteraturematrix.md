# Literature Matrix

> LLM-extracted answer matrix: one row per paper, one column per question.

```{figure} ../_static/img/owliteraturematrix.png
:alt: Literature Matrix
:class: widget-screenshot

The Literature Matrix widget.
```

## Overview

An Elicit-style **literature review matrix**: for each paper it asks an LLM your
custom questions (e.g. "What is the method?", "What is the sample size?") against
the abstract/title, returning a tidy paper × question table to jump-start a
structured review.

## Inputs
- **Data** (`Table`) — data with Abstract / Title.

## Outputs
- **Matrix** (`Table`) — one row per paper, one column per question.

## Controls
- **Text column** — the text the questions are answered from.
- **Questions / columns** — your questions, one per line (or `;`-separated).
- **Max papers** — cap papers processed (cost control).
- **Provider / Model / API key** — the LLM backend.

**Actions:** `Extract`.

## Note
Sends paper text to an external LLM service; mind provider terms and cost.
