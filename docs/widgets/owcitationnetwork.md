# Citation Network

> Directed document citation network + main-path analysis.

```{figure} ../_static/img/owcitationnetwork.png
:alt: Citation Network
:class: widget-screenshot

The Citation Network widget.
```

## Overview

Builds the **directed, acyclic** document-to-document citation network (who
cites whom) from references or OpenAlex referenced-works, lays it out
chronologically, and computes the **main path** — the backbone chain of the most
structurally important citations through the field. Exports to Pajek.

## Inputs
- **Data** (`Table`) — data with references / OpenAlex referenced works.

## Outputs
- **Network** / **Main Path** (`Network`/`object`) — for the Network add-on.
- **Node Data**, **Edge Data**, **Main Path Data** (`Table`).

## Controls
- **Min Citations** — minimum local citations for a node to be included.
- **Top N Documents** — cap the number of documents.
- **Match Threshold (%)** — fuzzy reference-matching strictness.
- **Layout** — graph layout (chronological by default).
- **Node size** — sizing metric (e.g. local citations).
- **Label format** — author+year, author+year+source, DOI/EID, …
- **Curved edges / Show labels / Highlight main path** — display options.

**Actions:** `Build Network`, Pajek export `.net` / `.clu` / `.vec` / `Save all`.
