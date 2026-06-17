# Embedding Landscape

> 2-D semantic map of documents from text embeddings.

```{figure} ../_static/img/owembeddinglandscape.png
:alt: Embedding Landscape
:class: widget-screenshot

The Embedding Landscape widget.
```

## Overview

Embeds documents from their text, projects them to 2-D, and clusters them into a
**semantic landscape** where nearby points are topically similar. A
content-based complement to keyword co-occurrence maps. Outputs the 2-D
coordinates and a cluster label per document.

## Inputs
- **Data** (`Table`) — data with a text column.

## Outputs
- **Coordinates** (`Table`) — 2-D coordinates + cluster per document.

## Controls
- **Text column** — the field embedded (e.g. Abstract).
- **Clusters** — number of clusters to form.

**Actions:** `Build Landscape`.
