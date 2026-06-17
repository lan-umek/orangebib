# Thematic Evolution

> Sankey of how themes split/merge between successive periods.

```{figure} ../_static/img/owthematicevolution.png
:alt: Thematic Evolution
:class: widget-screenshot

The Thematic Evolution widget.
```

## Overview

Tracks how research themes **evolve** across consecutive time periods, showing
the flows (splits, merges, continuations) of keyword clusters from one period to
the next as a Sankey-style diagram. Complements the static **Thematic Map**.

## Inputs
- **Data** (`Table`) — keywords + Year.

## Outputs
- **Flows** (`Table`) — theme transitions between periods (source theme → target theme, weight).

## Controls
- **Keywords column** — the term field.
- **Periods** — number of equal time slices.
- **Cut points (years, comma-sep)** — explicit period boundaries (overrides *Periods*).
- **Themes / period** — number of themes detected per period.
- **Top keywords / period** — terms used to label each theme.

**Actions:** `Build`.
