# Geographic Analysis

> Per-country metrics with coordinates, ready for the map widgets.

```{figure} ../_static/img/owgeographic.png
:alt: Geographic Analysis
:class: widget-screenshot

The Geographic Analysis widget.
```

## Overview

Aggregates the corpus **by country**: document counts, citations and
collaboration, with latitude/longitude attached so the output can be dropped
straight into Orange's **Geo** widgets (map). Selecting countries outputs their
documents.

## Inputs
- **Data** (`Table`) — data with affiliations / countries.

## Outputs
- **Countries** (`Table`) — per-country metrics + Latitude/Longitude.
- **Selected Documents** (`Table`) — documents from the selected countries.

## Controls
- **Country column** / **Affiliations** — the source columns (countries are extracted from affiliations if needed).
- **Citations** — the citation column for citation-weighted metrics.

**Actions:** `Analyze`.
