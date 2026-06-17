# Citation Patterns

> Classify papers by the shape of their citation trajectory over time.

```{figure} ../_static/img/owcitationpatterns.png
:alt: Citation Patterns
:class: widget-screenshot

The Citation Patterns widget.
```

## Overview

Classifies each paper's **citation trajectory** into archetypal patterns
(early rise, late bloomer, flash-in-the-pan, sustained, sleeping beauty, …) from
its yearly citation history, and visualises the trajectories and their
distribution. Works best with OpenAlex yearly-citation data.

## Inputs
- **Data** (`Table`) — data with yearly citation counts.

## Outputs
- **Classified** (`Table`) — each paper with its assigned pattern.
- **Selected Documents** (`Table`) — papers selected in the plot.

## Controls
- **Use OpenAlex API (recommended)** — fetch yearly citations if not present.
- **Metric** / **Pattern** / **Plot Type** — what to measure, which pattern to highlight, and how to display it (trajectories, distribution, …).
- **Metric bins** — binning for the distribution view.
- **By Year from / to (0 = auto)** — restrict the time window.
- **Normalize trajectories** — scale each curve to 0–1 to compare shapes.
- **Distribution: single colour / sort by size**, **Metrics: show median line** — display options.
- **Max papers**, **Min age (years)** — sampling limits.
- **Auto apply** — recompute on change.

**Actions:** `Run Analysis`, `Export`, `Clear Selection`.
