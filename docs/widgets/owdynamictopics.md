# Dynamic Topic Models

> Topics and how they rise/fall across time periods.

```{figure} ../_static/img/owdynamictopics.png
:alt: Dynamic Topic Models
:class: widget-screenshot

The Dynamic Topic Models widget.
```

## Overview

Models topics across successive time periods and tracks each topic's prevalence
trajectory, so you can see which themes grow, shrink or persist. The
time-resolved counterpart of **Topic Modeling**.

## Inputs
- **Data** (`Table`) — text + Year.

## Outputs
- **Topics** (`Table`) — topic labels and their per-period trajectories.
- **Document Topics** (`Table`) — per-document topic assignment.
- **Period Metrics** (`Table`) — per-period topic prevalence.

## Controls
- **Text column** — the field modelled.
- **N topics** — number of topics.
- **Period size (yrs)** — width of each time slice.

**Actions:** `Model Topics`.
