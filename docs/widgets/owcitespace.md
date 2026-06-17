# CiteSpace Metrics

> CiteSpace-style node metrics (centrality, bursts) and labelled clusters.

```{figure} ../_static/img/owcitespace.png
:alt: CiteSpace Metrics
:class: widget-screenshot

The CiteSpace Metrics widget.
```

## Overview

Computes CiteSpace-style indicators on a co-occurrence/co-citation network:
per-node **betweenness centrality** (the "pivot" nodes that bridge clusters),
burstness, and a clustering of the network with automatic cluster labels — the
quantitative backbone of a CiteSpace-type science map.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Node Metrics** (`Table`) — per-node centrality/burst metrics.
- **Clusters** (`Table`) — labelled network clusters.

## Controls
- **Entity column** — the field whose co-occurrence network is analysed (keywords, references…).
- **Top N nodes** — keep the N most frequent nodes.
- **Min occurrences** — drop rare nodes below this frequency.

**Actions:** `Compute metrics`.
