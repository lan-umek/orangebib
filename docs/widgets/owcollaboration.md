# Collaboration Network

> Country (or institution) co-authorship collaboration network.

```{figure} ../_static/img/owcollaboration.png
:alt: Collaboration Network
:class: widget-screenshot

The Collaboration Network widget.
```

## Overview

Builds the international (or inter-institutional) **collaboration** network from
co-authorship: nodes are countries/institutions, edges weight how often they
co-author papers. Emits a country × country matrix and a node table with
communities.

## Inputs
- **Data** (`Table`) — data with affiliations / countries.

## Outputs
- **Collaboration Matrix** (`Table`) — country × country co-authorship counts.
- **Node Data** (`Table`) — countries with degree and community.

## Controls
- **Country column** / **Affiliations** — the source columns.
- **Min collaborations** — minimum joint papers for an edge.

**Actions:** `Build`.
