# K-Fields Plot

> Three-field (Sankey) plot of flows between, e.g., authors → keywords → sources.

```{figure} ../_static/img/owkfieldsplot.png
:alt: K-Fields Plot
:class: widget-screenshot

The K-Fields Plot widget.
```

## Overview

The `biblioshiny` **three-fields (Sankey) plot**: shows the flows linking the
top items of two or three chosen fields — e.g. which authors publish on which
keywords in which sources. Clicking a node or flow outputs its documents.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Flow Data** (`Table`) — the flow relationships between fields.
- **Selected Documents** (`Table`) — documents of the selected node/flow.

## Controls
- **K** — number of fields (columns) in the plot (2 or 3).
- **Field Selection** — which field each column shows (authors, keywords, sources, countries…).
- **Top** — number of top items per field.
- **Value** — what edge width encodes (document count…).
- **Color By** — node colouring.

**Actions:** `Generate Plot`.
