# Production Plot

> Annual scientific production with an optional secondary line (e.g. citations).

```{figure} ../_static/img/owproductionplot.png
:alt: Production Plot
:class: widget-screenshot

The Production Plot widget.
```

## Overview

Plots documents per year as bars, optionally with a second series as a line on a
right-hand axis (e.g. mean citations or cumulative output). Titles, axis labels
and colours are fully editable, and early sparse years can be merged into a
single bucket. Clicking a year sends that year's documents onward.

## Inputs

- **Data** (`Table`) — bibliographic data (needs a Year column).

## Outputs

- **Selected Documents** (`Table`) — documents of the clicked year(s).

## Controls

- **Bars** — the variable shown as bars (usually document count per year).
- **Line** — an optional second variable drawn as a line on the right axis.
- **Bar color** / **Line color** — series colours.
- **Title**, **X axis**, **Y left**, **Y right** — editable titles/labels.
- **Bar legend** / **Line legend** — series legend labels.
- **Show grid** / **Show legend** — toggles.
- **Merge years before** *(Cutpoint)* — pool all years before this into one bucket, to avoid a long sparse tail.

## Tips

- For trend topics over time use **Top Items Timeline** or **Trend Topics**
  instead; this widget is for overall production.
