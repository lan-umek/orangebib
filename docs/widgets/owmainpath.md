# Main Path Analysis

> The backbone citation chain through a citation network.

```{figure} ../_static/img/owmainpath.png
:alt: Main Path Analysis
:class: widget-screenshot

The Main Path Analysis widget.
```

## Overview

Computes the **main path** of a citation network — the chain of documents that
carries the greatest traversal weight (SPC/SPLC) from the field's earliest to
its most recent work, i.e. the trunk of knowledge flow. Useful for telling the
"story" of a field in a handful of pivotal papers.

## Inputs
- **Data** (`Table`) — OpenAlex-enriched data (referenced works).

## Outputs
- **Main Path** (`Table`) — the documents on the main path, in order.
- **Stats** (`Table`) — network statistics.

## Controls
- **Paper ID** / **References** — identifier and references columns.
- **Method** — traversal-count weighting (e.g. SPC / SPLC / SPNP).

**Actions:** `Find Main Path`.
