# Field Networks

> Jaccard heatmap, disparity-filter backbone and bridging nodes for field co-occurrence.

```{figure} ../_static/img/owfieldnetworks.png
:alt: Field Networks
:class: widget-screenshot

The Field Networks widget.
```

## Overview

Advanced co-occurrence analysis for OpenAlex fields/subfields (or any
multi-valued column): a normalised (Jaccard / association / …) field × field
**heatmap**, a **disparity-filter backbone** that keeps only statistically
strong links, Louvain communities and **bridging** centralities (nodes linking
otherwise separate areas). Emits Node/Edge tables that plug into **Plot
Bibliometric Network**.

## Inputs
- **Data** (`Table`) — data with a multi-valued field (oa_fields/subfields, keywords, countries…).

## Outputs
- **Node Data** (`Table`) — community, degree, betweenness, clustering, bridging.
- **Edge Data** (`Table`) — Source/Target/Weight (feed the plot widget).
- **Bridging Nodes** (`Table`) — top bridging nodes.

## Controls
- **Column** — the multi-valued field analysed.
- **Min entity count** — drop rare entities.
- **Normalisation** — jaccard / association / inclusion / salton / none.
- **Min edge weight** — prune weak links.
- **Disparity-filter backbone** + **Backbone alpha** — keep only significant edges (smaller α = stricter).
- **Heatmap top N** / **Bridging top N** — sizes of the two views.
- **Run automatically** — recompute on change.

**Actions:** `Run`.
