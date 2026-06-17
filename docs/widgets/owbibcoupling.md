# Bibliographic Coupling

> Papers linked by shared references (research-front structure).

```{figure} ../_static/img/owbibcoupling.png
:alt: Bibliographic Coupling
:class: widget-screenshot

The Bibliographic Coupling widget.
```

## Overview

Builds the **bibliographic-coupling** network: two papers are linked when they
cite the same references — the more shared references, the stronger the link.
Whereas co-citation reveals the historical base, coupling reveals the current
**research front** (clusters of papers working on related problems now).

## Inputs
- **Data** (`Table`) — OpenAlex-enriched data (referenced works).

## Outputs
- **Node Data** (`Table`) — papers with degree and community.
- **Edge Data** (`Table`) — coupling edges (shared-reference weight).

## Controls
- **Paper ID col** / **References col** — the identifier and references columns.
- **Min shared refs** — minimum shared references for an edge.
- **Top N papers** — cap the number of papers.

**Actions:** `Build`.
