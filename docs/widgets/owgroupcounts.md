# Group Counts

> Entity frequencies compared across the groups defined in Setup Groups.

```{figure} ../_static/img/owgroupcounts.png
:alt: Group Counts
:class: widget-screenshot

The Group Counts widget.
```

## Overview

Counts entities (keywords, authors, sources…) **within each group** and merges
the per-group counts into one comparison table — e.g. the top keywords of group
A vs group B. Requires group columns from **Setup Groups**.

## Inputs
- **Data** (`Table`) — data with `Group:` columns.

## Outputs
- **Counts** (`Table`) — merged per-group counts.
- **Filtered Counts** (`Table`) — top-N filtered version.

## Controls
- **Entity Type** — what to count per group.
- **Merge Type** — how per-group counts are combined (union of top items, etc.).
- **Layout** — wide vs long table shape.
- **Show Top N / Top N for Plot** — how many entities to keep/plot.

**Actions:** `Count Entities`, `Count All`, `Export Results`.
