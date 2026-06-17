# Novelty Metrics

> Identify novel papers by their first-time / atypical combinations.

```{figure} ../_static/img/ownoveltymetrics.png
:alt: Novelty Metrics
:class: widget-screenshot

The Novelty Metrics widget.
```

## Overview

Scores each paper's **novelty** by how unusual its combinations of elements
(keywords, references, words) are relative to what existed before — new pairings
and atypical combinations signal novelty. Returns per-paper novelty metrics and
the most novel subset. Text columns can be cleaned (stopwords / boilerplate)
before combinations are formed.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Per-document** (`Table`) — novelty metrics per paper.
- **Novel Documents** (`Table`) — papers at/above the chosen novelty percentile.

## Controls
- **Combinations from** — the field whose element pairings define novelty (keywords, references, words).
- **Separator** — list separator for that field (hidden for free-text sources).
- **Baseline years** — the prior window against which "new" combinations are judged.
- **Novel subset percentile** — threshold for the *Novel Documents* output.
- **Remove stop words (basic + my extended list)** — clean text before forming combinations.
- **Remove general/boilerplate words (concept Excel)** — also drop generic terms.

**Actions:** `Compute`.
