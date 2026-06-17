# Permutation Inference

> Exact permutation p-values for group–entity association statistics.

```{figure} ../_static/img/owpermutationtest.png
:alt: Permutation Inference
:class: widget-screenshot

The Permutation Inference widget.
```

## Overview

Provides distribution-free **permutation tests** for the association between
groups and an entity: it repeatedly shuffles group labels to build the null
distribution of a chosen statistic (chi-square, Cramér's V, total inertia,
dimension inertias, or per-cell residuals) and reports the observed value, an
empirical p-value, a confidence interval and the null distribution. The residuals
test yields a **group × entity** p-value matrix; clicking a cell outputs its
documents.

## Inputs
- **Data** (`Table`) — bibliographic data, and/or
- **Group Matrix** (`Table`) — document × group binary matrix (from **Setup Groups**).

## Outputs
- **Result** (`Table`) — observed statistic, p-value, CI, B.
- **Null Distribution** (`Table`) — per-permutation statistic values.
- **Cell p-values** (`Table`) — group × entity p-values (residuals test).
- **Selected Documents** (`Table`) — documents of the clicked cell.

## Controls
- **Test Statistic** — chi-square / Cramér's V / total inertia / dimension inertias / residuals.
- **Column** + **Separator (lists)** — the entity column and its list separator.
- **Permutations B (0 = adaptive)** — number of shuffles; 0 lets it choose adaptively within a time budget.
- **Adaptive time budget (s)** — wall-clock cap when B = 0.
- **Random seed** — reproducibility.
- **Adjustment** — multiple-comparison correction for the cell p-values.
- **Dimensions** — number of CA dimensions (for dimension-inertia statistic).
- **Apply automatically** — recompute on change.

**Actions:** `Run Permutation Test`.
