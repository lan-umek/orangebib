# Disruption Index

> Are papers consolidating or disrupting their field (CD index)?

```{figure} ../_static/img/owdisruptionindex.png
:alt: Disruption Index
:class: widget-screenshot

The Disruption Index widget.
```

## Overview

Computes the **disruption (CD) index** for papers from their citation/reference
structure: a disruptive paper eclipses the work it builds on (later papers cite
it but not its references), whereas a consolidating paper is co-cited with its
predecessors. Returns per-paper scores and aggregates by entity (author,
source, year). Requires references / citation linkage.

## Inputs
- **Data** (`Table`) — data with references (and citation links).

## Outputs
- **Disruption** (`Table`) — per-paper disruption metrics.
- **Aggregated** (`Table`) — disruption aggregated by entity.
- **Enhanced Data** (`Table`) — input with disruption columns.
- **Selected Documents** (`Table`) — papers selected in the plot.

## Controls
- **Analysis Type** — per-paper vs aggregated-by-entity.
- **Min Documents** / **Top N** — thresholds for the aggregated view.
- **Add disruption to dataset** — append the score columns to the data output.
- **Show visualization / Show statistics on plot** — display toggles.
- **Plot Type** — chart style.
- **Auto apply** — recompute on change.

**Actions:** `Analyze`, `View Data →`, `Export to Excel`.
