# RaceBar

> Animated "bar-chart race" of the top entities over time.

## Overview

Animates how the ranking of the top entities (authors, keywords, sources…)
changes year by year — the familiar bar-chart-race animation — using a
cumulative or per-year metric. The animation can be played, scrubbed and
exported to an animated GIF.

```{admonition} Screenshot
:class: tip
Omitted: this widget cannot be captured head-less. See it live in Orange.
```

## Inputs
- **Data** (`Table`) — bibliographic data with Year.

## Outputs
- *(none)* — produces an on-screen animation / exported GIF.

## Controls
- **Item type** — entity to race (authors, keywords, sources…).
- **Metric** — Documents or Citations.
- **Top N** — number of bars shown each frame.
- **Cumulative** — accumulate values over years vs per-year values.
- **Speed (ms)** — frame interval of the animation.

**Actions:** `▶ Play`, `⟲` (restart), `⬇ Export animation (GIF)`.
