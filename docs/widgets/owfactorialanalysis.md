# Factorial Analysis

> Conceptual structure map (MCA/PCA of terms) with clustering.

```{figure} ../_static/img/owfactorialanalysis.png
:alt: Factorial Analysis
:class: widget-screenshot

The Factorial Analysis widget.
```

## Overview

The `biblioshiny` **conceptual structure** analysis: reduces a term × document
matrix to a low-dimensional factor space (MCA / CA / PCA) and clusters the terms,
producing a 2-D map of the field's conceptual structure.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Embeddings** (`Table`) — term coordinates in the factor space.
- **Clusters** (`Table`) — cluster assignment per term.

## Controls
- **Field Selection** — which terms (author/index keywords…).
- **N Terms** / **Min Doc Freq** — vocabulary size and rarity cutoff.
- **DTM Method** — how the document-term matrix is weighted.
- **DR Method** + **Components** — dimensionality-reduction method and number of axes.
- **Method** + **N Clusters** — clustering algorithm and cluster count.
- **Plot Type**, **Show term labels** — display options.

**Actions:** `Run Analysis`.
