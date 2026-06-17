# Compare Means

> Compare a numeric variable across groups (t-test / ANOVA / non-parametric) + post-hoc.

```{figure} ../_static/img/owcomparemeans.png
:alt: Compare Means
:class: widget-screenshot

The Compare Means widget.
```

## Overview

Tests whether a numeric variable (e.g. citations) differs across the levels of a
grouping variable: group descriptives, the appropriate omnibus test (t-test /
one-way ANOVA, or non-parametric Mann-Whitney/Kruskal-Wallis) and pairwise
**post-hoc** comparisons. Can also compare across several binary group columns
at once.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Descriptives** (`Table`) — per-group descriptive statistics.
- **Tests** (`Table`) — omnibus test results.
- **Post-hoc** (`Table`) — pairwise comparisons.

## Controls
- **Dependent (numeric)** — the variable compared.
- **Group by** — the grouping variable (or use multiple binary group columns).
- **Alpha** — significance level.

**Actions:** `Compare`.
