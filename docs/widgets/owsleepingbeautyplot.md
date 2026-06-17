# Sleeping Beauty Plot

> Plot the citation trajectories of detected sleeping beauties.

```{figure} ../_static/img/owsleepingbeautyplot.png
:alt: Sleeping Beauty Plot
:class: widget-screenshot

The Sleeping Beauty Plot widget.
```

## Overview

Visualises the yearly citation curves of the sleeping beauties found by the
**Sleeping Beauty** widget — the long flat "sleep" followed by the "awakening"
spike — for one paper or many at once. Selecting curves outputs those papers.

## Inputs
- **Data** (`Table`) — output of the **Sleeping Beauty** widget.

## Outputs
- **Selected** (`Table`) — selected sleeping beauties.

## Controls
- **Normalize trajectories (0-1)** — scale each curve to compare shapes regardless of magnitude.
- **Single Paper** — focus on one selected paper's curve.

**Actions:** `Export Selected`, `Clear Selection`.
