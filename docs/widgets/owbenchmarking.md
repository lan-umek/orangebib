# Benchmarking

> Compare your corpus's category shares against a reference (over-/under-representation).

```{figure} ../_static/img/owbenchmarking.png
:alt: Benchmarking
:class: widget-screenshot

The Benchmarking widget.
```

## Overview

Benchmarks the corpus against a **reference** distribution — e.g. comparing the
share of each topic/SDG/country in your set vs a baseline — and reports which
categories are over- or under-represented (percentage-point differences). Both
computes and plots the comparison.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Comparison** (`Table`) — full comparison with differences.
- **Over-represented** / **Under-represented** (`Table`).

## Controls
- **Compare** — the categorical dimension compared (e.g. SDG, topic, country).
- **Reference Data** — the baseline to compare against (**Browse…** to load one).
- **From / To** — restrict the year range.
- **Threshold (pp)** — minimum percentage-point gap to flag a category.
- For SDG comparisons, an option shows SDGs by number only or number + name; a warning appears if SDG columns are missing (run **SDG Identifier** first).

**Actions:** `Compare`, `Browse…`.
