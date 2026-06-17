# SDG Drift

> How the SDG focus of a corpus shifts over time.

```{figure} ../_static/img/owsdgdrift.png
:alt: SDG Drift
:class: widget-screenshot

The SDG Drift widget.
```

## Overview

Tracks how attention to each SDG **changes across time windows**, ranking goals
by how much their prevalence (and context) drifts. Useful for seeing which
sustainability themes are rising or fading in a field.

## Inputs
- **Data** (`Table`) — SDG-tagged data with text + Year.

## Outputs
- **Drift Ranking** (`Table`) — SDGs ranked by average drift.
- **SDG Summary** (`Table`) — per-window SDG prevalence.

## Controls
- **Window size (years)** — width of each comparison window.
- **Text column** — text field used for the contextual drift.

**Actions:** `Analyze drift`.
