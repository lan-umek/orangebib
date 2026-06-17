# Burst Detection

> Kleinberg burst detection — terms with a sudden surge in usage.

```{figure} ../_static/img/owburstdetection.png
:alt: Burst Detection
:class: widget-screenshot

The Burst Detection widget.
```

## Overview

Applies **Kleinberg's** burst-detection algorithm to keyword (or other entity)
frequencies over time to find terms that experienced a sharp, sustained surge —
the markers of emerging research fronts. Returns each burst's start/end year and
intensity (weight); selecting a burst outputs its documents.

## Inputs
- **Data** (`Table`) — keywords + Year.

## Outputs
- **Bursts** (`Table`) — Keyword, Start, End, Weight.
- **Selected Documents** (`Table`) — documents of the selected bursting term(s).

## Controls
- **Keywords column** — the entity field to scan for bursts.
- **Top N** — limit to the N strongest bursts.
- **s (burst scaling)** — the state-transition intensity factor (higher = only stronger bursts).
- **gamma (transition cost)** — cost of switching into a burst state (higher = fewer, longer bursts).
- **Min duration (yrs)** — discard bursts shorter than this.

**Actions:** `Detect Bursts`.
