# Citation Distribution

> Histogram of citations with mean/median lines and citation classes.

```{figure} ../_static/img/owcitationdistribution.png
:alt: Citation Distribution
:class: widget-screenshot

The Citation Distribution widget.
```

## Overview

Shows how citations are distributed across the corpus — a typically very
skewed, long-tailed distribution. Plots a histogram (optionally clipped at a
high percentile so a few mega-cited papers don't flatten the rest), marks the
mean and median, and assigns each paper a **citation class** (e.g. uncited /
low / medium / highly cited). Selecting a bin outputs those papers.

## Inputs
- **Data** (`Table`) — data with a citation column.

## Outputs
- **Selected Documents** (`Table`) — papers in the selected bin.
- **Aggregated** (`Table`) — per-bin counts.
- **Metrics** (`Table`) — summary citation metrics.
- **Citation Classes** (`Table`) — each paper with its citation class.

## Controls
- **Citations** — the citation column to use.
- **Clip percentile** — cap the x-axis at this percentile to tame the tail.
- **Number of bins** — histogram resolution.
- **Show mean line / Show median line** — reference lines.
- **Bar color** — histogram colour.
- **Auto apply** — recompute on change.

**Actions:** `Analyze`, `Clear Selection`.
