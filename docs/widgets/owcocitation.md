# Co-citation Network

> References frequently cited together (intellectual base of the field).

```{figure} ../_static/img/owcocitation.png
:alt: Co-citation Network
:class: widget-screenshot

The Co-citation Network widget.
```

## Overview

Builds the **co-citation** network: references that are repeatedly cited
together by the corpus, revealing the field's intellectual base and its schools
of thought (clusters). Nodes are cited references; edges weight how often two are
co-cited.

## Inputs
- **Data** (`Table`) — data with references.

## Outputs
- **Node Data** (`Table`) — references with citation count and community.

## Controls
- **References col** — the references column.
- **Top N references** — keep the N most-cited references.
- **Min co-citation** — minimum co-citation count for an edge.

**Actions:** `Build`. Pair the edges with **Plot Bibliometric Network** to draw.
