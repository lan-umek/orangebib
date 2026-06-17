# Trend Analysis

> Time trends of production/citations with a fitted trendline and short forecast.

```{figure} ../_static/img/owtrendanalysis.png
:alt: Trend Analysis
:class: widget-screenshot

The Trend Analysis widget.
```

## Overview

Aggregates a metric (documents, citations, …) over time and fits a trendline,
optionally projecting a short forecast. Use it to quantify whether a field,
topic or venue is growing, plateauing or declining.

## Inputs
- **Data** (`Table`) — bibliographic data with a Year column.

## Outputs
- **Trends** (`Table`) — the per-period series and fitted values.
- **Summary** (`Table`) — slope/growth statistics.

## Controls
- **Type** — the metric to trend (document count, citations, …).
- **From** / **To** — restrict the time window (years).
- **Aggregation** — period granularity (e.g. yearly).
- **Trendline** — model fitted to the series (linear / polynomial / exponential).
- **Forecast years** — number of future years to project from the fit.

**Actions:** `Analyze Trends`.
