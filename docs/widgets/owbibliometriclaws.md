# Bibliometric Laws

> Lotka, Bradford, Zipf, Price and Pareto laws with theoretical fits.

```{figure} ../_static/img/owbibliometriclaws.png
:alt: Bibliometric Laws
:class: widget-screenshot

The Bibliometric Laws widget.
```

## Overview

Tests the classic bibliometric distributions on your corpus:

- **Lotka** — author productivity (how many authors publish *n* papers);
- **Bradford** — journal scatter into core/zone structure;
- **Zipf** — word-frequency rank distribution;
- **Price / Pareto** — concentration of output/citations.

It plots the empirical distribution with the fitted theoretical curve and
reports the estimated parameters and goodness of fit.

## Inputs

- **Data** (`Table`) — bibliographic data.

## Outputs

- **Results** (`Table`) — fitted parameters and fit statistics.
- **Distribution** (`Table`) — the empirical frequency distribution.

## Controls

- **Select Law** — which law to test (Lotka / Bradford / Zipf / Price / Pareto).
- **Data Column** — the field the law is computed on (authors, sources, words…); defaults follow the chosen law.
- **Show theoretical fit** — overlay the fitted curve.
- **Use log-log scale** — plot on log-log axes (where the laws are linear).
- **Show grid** — toggle grid lines.

**Actions:** `Analyze Law`.
