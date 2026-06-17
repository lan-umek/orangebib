# Entity Over Time

> Per-year series for the top entities (lines, stacked area or cumulative).

```{figure} ../_static/img/owentityovertime.png
:alt: Entity Over Time
:class: widget-screenshot

The Entity Over Time widget.
```

## Overview

Tracks the top entities (keywords, authors, sources…) year by year, as lines or
a stacked area, in raw or cumulative counts. Complements **Trend Topics**
(which summarises *when* a topic peaks) by showing the full trajectory.

## Inputs
- **Data** (`Table`) — bibliographic data with Year.

## Outputs
- **Selected Documents** (`Table`).
- **Time Series** (`Table`) — year × entity matrix.

## Controls
- **Top N** — number of entities to plot.
- **Min Documents** — minimum frequency to include.
- **Year from / Year to (0 = auto)** — restrict the time window.
- **View** — lines vs stacked area.
- **Cumulative Documents** — plot running totals instead of per-year counts.
- **Color Map** — colour scheme.
- **Show legend / Show grid** — toggles.

**Actions:** `Run Analysis`.
