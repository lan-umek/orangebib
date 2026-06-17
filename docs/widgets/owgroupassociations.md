# Group Associations

> Which entities are over-/under-represented in each group (chi-square, CA, residuals).

```{figure} ../_static/img/owgroupassociations.png
:alt: Group Associations
:class: widget-screenshot

The Group Associations widget.
```

## Overview

Cross-tabulates an entity (keywords, authors…) against the groups and tests
their association: a **chi-square** test, standardized **residuals** (which
entity is over-/under-represented in which group), **correspondence analysis**
(a 2-D associations map), diversity and log-ratio measures. Pair with
**Permutation Inference** for exact p-values. Selecting cells outputs documents.

## Inputs
- **Data** (`Table`) — data with group columns.

## Outputs
- **Contingency**, **Chi-square**, **Correspondence**, **Diversity**, **SVD**, **Log-ratio** (`Table`).
- **Selected Documents** / **Filtered** (`Table`).

## Controls
- **Entity Type** — the entity cross-tabulated against groups.
- **Top N Items / Min Frequency** — trim the entities.
- **Include (regex) / Exclude (regex)** — entity filters.
- **Statistics to Include** — which of the tables above to compute.
- **Row proportion**, **Over-/Under-represented**, **Min |residual|** — residual display options.
- **Visualisation** + **Colormap** — heatmap/CA view styling.

**Actions:** `Compute Associations`, `Export Results`.
