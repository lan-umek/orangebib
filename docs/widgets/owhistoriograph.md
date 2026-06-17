# Historiograph

> Chronological citation graph of the most locally-cited papers (HistCite-style).

```{figure} ../_static/img/owhistoriograph.png
:alt: Historiograph
:class: widget-screenshot

The Historiograph widget.
```

## Overview

Draws a **historiograph** — the key papers of a field arranged by year (x-axis)
with arrows for direct citations among them, node size by **local citation
score** (citations from within the corpus). It traces the historical lineage of
ideas, à la HistCite / CiteNet Explorer.

## Inputs
- **Data** (`Table`) — data with references / OpenAlex referenced works.

## Outputs
- **Node Data** (`Table`) — the papers in the graph with their local citations.
- **Selected Documents** (`Table`) — the selected node(s).

## Controls
- **Min local citations** — minimum within-corpus citations to include a paper.
- **Max documents** — cap the graph size.
- **Node size** — sizing metric (local citation score).
- **Labels** — node label format.
- **Curved edges / Show labels** — display options.

**Actions:** `Build historiograph`.
