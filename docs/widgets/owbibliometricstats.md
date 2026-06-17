# Bibliometric Statistics

> Per-entity performance table (documents, citations, h-index and variants, years…).

```{figure} ../_static/img/owbibliometricstats.png
:alt: Bibliometric Statistics
:class: widget-screenshot

The Bibliometric Statistics widget.
```

## Overview

For a chosen entity type (authors, sources, keywords, countries, …) computes a
full performance table: number of documents, total and average citations, the
**h-index** and its variants (g, m, and the SCI2S family — A, R, E, Q2, etc.),
first/median/last year and more. The depth setting controls how many indicators
are produced. Row selection in the preview outputs the documents of the chosen
entities. This is the standard upstream for the **Performance Plot**.

## Inputs

- **Data** (`Table`) — bibliographic data.

## Outputs

- **Statistics** (`Table`) — one row per entity with all indicators.
- **Selected Documents** (`Table`) — documents of the entities selected in the preview.

## Controls

- **Analyze** — the entity type (Authors, Sources, Author/Index/All Keywords, Countries, Affiliations…).
- **Top N** — number of top entities (by document count) to evaluate.
- **Depth** — indicator set: *core* (docs, citations, h-index, average year) → *extended* → *full* (all h-variants and year percentiles).
- **Include only / Exclude (one per line)** — restrict or drop specific entities by name.
- **Regex include / Regex exclude** — the same, by regular expression.
- **Max items** — hard cap on the number of entities (with regex selection).
- **Countries** *(Geographic filter)* — keep only documents from chosen countries before computing.
- **Apply Automatically** — recompute on change; else press **Compute Statistics**.
- **Skip header row when loading files** / **Load** — load an entity list from a file for *Include only*.

**Actions:** `Compute Statistics`, `Load`, `Select All`, `Clear Selection`.

## Tips

- Connect **Statistics → Performance Plot** to rank/scatter entities; *full*
  depth is needed for the h-index-variant columns.
