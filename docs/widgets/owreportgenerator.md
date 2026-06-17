# Report Generator

> One-click HTML/PDF bibliometric report (figures + tables).

```{figure} ../_static/img/owreportgenerator.png
:alt: Report Generator
:class: widget-screenshot

The Report Generator widget.
```

## Overview

Generates a complete bibliometric **report** from the corpus — overview,
production, top sources/authors/keywords, networks, citations and more — as HTML
and/or PDF, using biblium's reporting engine. A *custom report* mode lets you
pick exactly which figures/sections to include. The list separator is
auto-detected, so OpenAlex and Scopus data both render correctly.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Report Files** (`Table`) — the generated files (format, path).
- **Data** (`Table`) — pass-through of the input.

## Controls
- **File base name** + **Browse…** — output name and folder.
- **Detail level** — how much to include (core → full).
- **Database** — source hint (affects parsing/separators; auto-detected).
- **Title** — report title.
- **Formats** — HTML / PDF.
- **Max figures (0 = all)** — cap the number of figures.
- **Custom report** — **Scan available items** then tick exactly which to include (`All` / `None`).

**Actions:** `Generate Report`, `Scan available items`, `Open selected file`, `Open output folder`.

## Tips
- For an interactive, single-file result instead of a static report, use the
  **HTML Dashboard** widget.
