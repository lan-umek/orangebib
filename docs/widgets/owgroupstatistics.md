# Group Statistics

> Per-group performance statistics (documents, citations, h-index…).

```{figure} ../_static/img/owgroupstatistics.png
:alt: Group Statistics
:class: widget-screenshot

The Group Statistics widget.
```

## Overview

Computes the full performance-indicator table (as in **Bibliometric
Statistics**) **separately for each group**, so groups can be benchmarked on
documents, citations, h-index and more. Requires group columns from **Setup
Groups**.

## Inputs
- **Data** (`Table`) — data with `Group:` columns.

## Outputs
- **Statistics** (`Table`) — full per-group statistics.
- **Filtered Statistics** (`Table`) — top-N filtered version.

## Controls
- **Entity Type** — the entity evaluated within each group.
- **Top N Items** — entities per group.
- **Output Format** — wide vs long.
- **Include Items / Exclude Items** — entity filters.

**Actions:** `Compute Statistics`, `Compute All`, `Export Results`.
