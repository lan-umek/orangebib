# Document Clustering

> Cluster documents by text content (or bibliographic coupling).

```{figure} ../_static/img/owdocclustering.png
:alt: Document Clustering
:class: widget-screenshot

The Document Clustering widget.
```

## Overview

Groups documents into clusters from their text (TF-IDF) or, optionally, from
**bibliographic coupling** (shared references), adding a `Cluster` column.
A hard-assignment complement to **Topic Modeling** (soft topic weights).

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Data** (`Table`) — input with a `Cluster` column.
- **Cluster Summary** (`Table`) — cluster sizes.
- **Selected Documents** (`Table`) — documents in the selected cluster(s).

## Controls
- **Cluster by (text)** — the text field used for clustering.
- **Method** — clustering algorithm (k-means, agglomerative…).
- **N clusters (0 = auto)** — number of clusters, or 0 to choose automatically.
- **Bibliographic coupling (optional)** — cluster on shared references instead of/with text.

**Actions:** `Cluster`.
