# Crosstabs

> Contingency table between two variables with a chi-square test and effect size.

```{figure} ../_static/img/owcrosstabs.png
:alt: Crosstabs
:class: widget-screenshot

The Crosstabs widget.
```

## Overview

Cross-tabulates two categorical variables, displays the contingency table
(counts / row % / column % / expected / residuals) and runs a **chi-square**
test of independence with an effect-size measure (e.g. Cramér's V).

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Contingency Table** (`Table`) — the chosen display matrix.
- **Statistics** (`Table`) — test statistic, p-value and effect size.

## Controls
- **Rows** / **Columns** — the two variables cross-tabulated.
- **Alpha** — significance level for the test.
- **Display** — counts / proportions / expected / residuals.

**Actions:** `Compute`.
