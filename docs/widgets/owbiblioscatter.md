# Performance Plot

> Bar / scatter / linear-projection of an entity-statistics table.

```{figure} ../_static/img/owbiblioscatter.png
:alt: Performance Plot
:class: widget-screenshot

The Performance Plot widget.
```

## Overview

Visualises the **Statistics** table from *Bibliometric Statistics* (one row per
author/source/keyword with numeric indicators). Three modes: a sorted **bar
chart**, a **scatter** (x/y/size/colour/label with robust optional log axes),
and a **linear projection** (PCA of several indicators onto 2-D). Clicking a
bar or point selects entities and sends them onward.

## Inputs

- **Statistics** (`Table`) — entity-statistics table (from **Bibliometric Statistics**).

## Outputs

- **Selected** (`Table`) — the entities you click.

## Controls

- **Mode** — Bar chart / Scatter / Linear projection.
- **X** / **Y** — numeric indicators for the axes (scatter); **Y** is the sort/value field for bars.
- **Log X** / **Log Y** — logarithmic axes (scatter); robust (non-positive values dropped).
- **Size** — bubble size by an indicator (scatter/projection).
- **Colour** + **Colormap** — colour by an indicator/category; default colormap *viridis*.
- **Label** — which column labels the points/bars.
- **Show point labels** — toggle point labels.
- **Base size** — base marker size.
- **Max entities** + **Limit number of points** + **Rank by** — keep only the top-N entities (by the chosen ranking metric).

**Actions:** `Plot`.

## Tips

- Default view is a bar chart ranked by citations — readable immediately;
  switch to scatter for the classic documents-vs-citations bubble map.
