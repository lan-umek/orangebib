# Thematic Map

> Strategic diagram of keyword clusters (centrality × density quadrants).

```{figure} ../_static/img/owthematicmap.png
:alt: Thematic Map
:class: widget-screenshot

The Thematic Map widget.
```

## Overview

Builds the `biblioshiny`-style **thematic (strategic) map**: keywords are
clustered from their co-occurrence network and each cluster is placed by
**centrality** (relevance/links to other themes, x-axis) and **density**
(internal development, y-axis), giving the four quadrants — motor, niche,
emerging/declining and basic themes. Clicking a cluster outputs its documents.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Thematic Data** (`Table`) — per-cluster centrality/density and members.
- **Selected Documents** (`Table`) — documents of the selected cluster.
- **Network** (`Table`) — the underlying co-occurrence edges.

## Controls
- **Field** — which terms to map (author/index keywords…).
- **Min Occurrences** — drop rare terms before clustering.
- **Top N Items** — keep the N most frequent terms.
- **Synonyms (Merge Items)** — thesaurus rules to merge term variants (separator `;`).
- **Auto apply** — recompute on change.

**Actions:** `Generate Thematic Map`.
