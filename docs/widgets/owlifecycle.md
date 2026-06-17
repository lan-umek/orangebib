# Life Cycle Analysis

> Fit a growth model (e.g. logistic/Bass) to cumulative production and forecast.

```{figure} ../_static/img/owlifecycle.png
:alt: Life Cycle Analysis
:class: widget-screenshot

The Life Cycle Analysis widget.
```

## Overview

Fits a diffusion / growth model to cumulative output over time to locate a field
or topic on its life-cycle curve (emergence → growth → maturity → saturation)
and project where it is heading. Reports the model parameters and a forecast.

## Inputs
- **Data** (`Table`) — bibliographic data (or a trend series).

## Outputs
- **Model Results** (`Table`) — fitted parameters and fit statistics.
- **Forecast** (`Table`) — projected values.

## Controls
- **Forecast horizon (years)** — how far ahead to project.
- **Fit from year / Fit to year (0 = auto)** — the window used to fit the model.
- **Projection Years** — years drawn in the projected segment.
- **Show Milestone Years** — mark inflection/saturation milestones.
- **Show Forecast Period** — shade the forecast region.

**Actions:** `Run Analysis`.
