# Conceptual Drift

> How the meaning/context of chosen terms shifts over time.

```{figure} ../_static/img/owconceptualdrift.png
:alt: Conceptual Drift
:class: widget-screenshot

The Conceptual Drift widget.
```

## Overview

Measures **semantic drift** of selected terms: how the words that co-occur with
each term change across successive time windows, quantifying how a concept's
usage/meaning evolves. Useful for tracing how a term's research context shifts.

## Inputs
- **Data** (`Table`) — text + Year.

## Outputs
- **Drift** (`Table`) — drift scores per term across windows.

## Controls
- **Terms (comma-sep)** — the terms to track.
- **Text column** — the text field providing context.
- **Window (yrs)** — width of each comparison window.

**Actions:** `Compute Drift`.
