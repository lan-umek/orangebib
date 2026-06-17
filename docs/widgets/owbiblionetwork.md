# Plot Bibliometric Network

> Draw a co-occurrence network with layout, communities, density and Pajek export.

```{figure} ../_static/img/owbiblionetwork.png
:alt: Plot Bibliometric Network
:class: widget-screenshot

The Plot Bibliometric Network widget.
```

## Overview

The **draw** half of the network pair. It renders a network — either built from
a data column, or from an **Edge Data** table produced by **Network
Co-occurrence**, **Citation Network** or **Field Networks** — with attractive
layouts, Louvain (and other) community colouring, node sizing, curved edges,
and a VOSviewer-style **density** view (item and cluster density). Nodes/clusters
are selectable (→ documents), and the network exports to **Pajek** (`.net`/`.clu`/`.vec`).

## Inputs
- **Data** (`Table`) — bibliographic data (builds the network from a column), and/or
- **Edge Data** (`Table`) — a precomputed From/To/Weight edge table.

## Outputs
- **Node Data** (`Table`) — nodes with degree and community.
- **Selected Documents** (`Table`) — documents of the selected node(s)/cluster.
- **Selected Nodes** (`Table`) — stats of the selected node(s).

## Controls
- **Item type** — entity column when building from Data.
- **Top N nodes / Min occurrences / Min edge weight** — size/trim the network.
- **Components** + **Min component size (k)** — show all, the largest only, or components ≥ k nodes.
- **Layout** — Spring, Circular, Kamada-Kawai, Shell, Spectral, Random.
- **Partition** — community detection (Louvain, greedy modularity, label propagation, connected components, or None).
- **Node size** — by weighted degree, frequency, betweenness or uniform; **Node size (%)** scales it.
- **Node shape** — circle (default), square, triangle, diamond, star, plus.
- **Label font size**, **Hide labels below size (%)**, **Show labels** — label control.
- **Edge width**, **Edge thickness by weight**, **Max edge width**, **Curved edges** — edge styling.
- **Density (VOSviewer-style)** — **Bandwidth (%)** and **Labels on density** for the Item/Cluster density tabs.
- **Clusters** — **Select** a community to forward all its items+documents; **Click selects whole cluster** toggles click behaviour.

**Actions:** Pajek export `.net` / `.clu` / `.vec` / `Save all`.
