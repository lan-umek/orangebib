# Top Items Timeline

> Horizontal timeline showing the active year-span of the top entities.

```{figure} ../_static/img/owtopitemstimeline.png
:alt: Top Items Timeline
:class: widget-screenshot

The Top Items Timeline widget.
```

## Overview

Draws, for the most frequent entities (keywords, authors, sources…), a
horizontal bar spanning the years in which each was active, sized/coloured by
frequency. Good for seeing the rise and fall of terms or the active periods of
authors. Clicking an item outputs its documents.

## Inputs
- **Data** (`Table`) — bibliographic data with Year.

## Outputs
- **Selected Documents** (`Table`) — documents of the clicked item.
- **Timeline Data** (`Table`) — per-item span and counts.

## Controls
- **Item Type** — entity to track (keywords, authors, sources…).
- **Top N** — number of items to show.
- **Min Documents** — minimum frequency to include an item.
- **Color By** + **Color Map** — colour encoding, default *viridis*.
- **Title / X label / Y label**, **Show grid** — labels and grid.

**Actions:** `Generate Plot`.
