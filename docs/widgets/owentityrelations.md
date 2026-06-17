# Entity Relationships

> Co-occurrence, chi-square, correspondence and association measures between entities.

```{figure} ../_static/img/owentityrelations.png
:alt: Entity Relationships
:class: widget-screenshot

The Entity Relationships widget.
```

## Overview

Analyses how the values of an entity relate to each other (or to groups) via a
co-occurrence/contingency table: a **chi-square** test, **correspondence
analysis** (a 2-D map of associations), standardized **residuals**
(over-/under-represented pairs) and association measures. Selecting cells/points
outputs the underlying documents.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Co-occurrence Matrix**, **Chi-square**, **Correspondence**, **Association Measures** (`Table`).
- **Selected Documents** (`Table`).

## Controls
- **Entity** — the entity analysed; **Top N** / **Min frequency** / **Max items** trim it.
- **Include only / Exclude** (+ **Regex include/exclude**) — entity filters.
- **Visualisation** — co-occurrence heatmap, correspondence map, or residuals.
- **Row proportion** — show row-wise shares.
- **Over-represented (+) / Under-represented (−)** — highlight residual direction.
- **Min |residual|** — residual magnitude threshold.
- **Min edge weight** — for the co-occurrence view.

**Actions:** `Compute Relationships`, `Export Results`, `Load`.
