# SDG Networks

> Co-occurrence network of SDGs and the papers bridging distant goals.

```{figure} ../_static/img/owsdgnetworks.png
:alt: SDG Networks
:class: widget-screenshot

The SDG Networks widget.
```

## Overview

From SDG-tagged data, builds the **SDG co-occurrence network** (which goals are
addressed together), reports per-SDG network metrics, the strongest goal
connections, and the **bridge papers** that link otherwise distant goals.
Requires SDG indicators from **SDG Identifier**.

## Inputs
- **Data** (`Table`) — data with SDG indicator columns.

## Outputs
- **SDG Metrics** (`Table`) — per-SDG network metrics.
- **Connections** (`Table`) — top SDG co-occurrences.
- **Bridge Papers** (`Table`) — papers connecting distant SDGs.
- **Selected SDGs** (`Table`) — metric rows for the SDGs you select.

## Controls
- **Min co-occurrence** — minimum joint occurrences for an SDG–SDG edge.
- **SDG labels** — how SDGs are labelled (number, number+name, pillar, dimension).

**Actions:** `Build network`.
