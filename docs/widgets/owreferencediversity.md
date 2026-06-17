# Reference Diversity

> Per-paper diversity of the cited references (disciplinary / temporal breadth).

```{figure} ../_static/img/owreferencediversity.png
:alt: Reference Diversity
:class: widget-screenshot

The Reference Diversity widget.
```

## Overview

Measures how diverse each paper's **reference list** is — a proxy for
interdisciplinarity and knowledge integration. From the cited references it
computes per-paper diversity metrics (variety, balance and disparity, plus the
age spread of references) and an aggregate summary for the corpus.

## Inputs

- **Data** (`Table`) — data with a references column.

## Outputs

- **Per-paper Metrics** (`Table`) — diversity scores per document.
- **Data** (`Table`) — the input with diversity columns appended.
- **Summary** (`Table`) — corpus-level diversity summary.

## Controls

- **Max papers** — cap the number of papers processed (speed on large corpora).
- **Max refs / paper** — cap references parsed per paper.
- **Current year** — reference year used to compute reference *age*.

**Actions:** `Compute Diversity`, `Cancel`.
