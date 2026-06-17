# Network Co-occurrence

> Build a co-occurrence network (keywords, co-authorship, co-citation, …).

```{figure} ../_static/img/ownetworkcooccurrence.png
:alt: Network Co-occurrence
:class: widget-screenshot

The Network Co-occurrence widget.
```

## Overview

The **compute** half of the network pair: it builds a co-occurrence network from
the corpus — keyword co-occurrence, co-authorship, country/institution
collaboration, co-citation, or title/abstract n-grams — applies a thesaurus and
filters, and emits node/edge tables plus an Orange `Network` object. Pair its
**Edge Data** output with **Plot Bibliometric Network** to draw it.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Network** (`Network`/`object`) — for the Orange Network add-on.
- **Node Data** (`Table`) — nodes with frequency and metrics.
- **Edge Data** (`Table`) — Source/Target/Weight, for the plot widget.

## Controls
- **Network Type** — what to relate (author/index/all keywords, co-authorship, co-citation, source co-citation, country/institution collaboration, title/abstract n-grams).
- **N-gram size** — n-gram length (text modes only).
- **Top N Nodes** — keep the N most frequent entities.
- **Min Occurrences** — drop entities below this frequency.
- **Min Edge Weight** — drop weak co-occurrence links.
- **Include / Exclude** (+ **Use regex**) — keep or drop specific entities.
- **Normalize weights** — scale edge weights (association strength).
- **Thesaurus / Synonyms** — merge variant terms (inline rules or **Load from Excel…**).

**Actions:** `Build Network`, `Load from Excel…`.
