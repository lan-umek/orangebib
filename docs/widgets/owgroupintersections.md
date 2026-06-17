# Group Intersections

> Overlap between groups (shared documents) as an UpSet / heatmap / network.

```{figure} ../_static/img/owgroupintersections.png
:alt: Group Intersections
:class: widget-screenshot

The Group Intersections widget.
```

## Overview

Shows how the groups from **Setup Groups** overlap — which documents belong to
several groups — via intersection sizes (UpSet-style), a similarity heatmap, or
a group-similarity network. Selecting an intersection outputs its documents.

## Inputs
- **Data** (`Table`) — data with group columns.

## Outputs
- **Intersections** (`Table`) — intersection sizes / similarities.
- **Selected Data** (`Table`) — documents in the selected intersection.

## Controls
- **ID Column** — the document identifier used to compute overlaps.
- **Visualization Type** — UpSet / heatmap / network.
- **Similarity** — similarity measure (e.g. Jaccard) for heatmap/network.
- **Threshold** — minimum similarity/edge to display.

**Actions:** `Analyze Intersections`, `Export Results`.
