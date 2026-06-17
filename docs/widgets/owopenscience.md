# Open Science

> Detect open-science practices (data/code sharing, preprints, OA) in papers.

```{figure} ../_static/img/owopenscience.png
:alt: Open Science
:class: widget-screenshot

The Open Science widget.
```

## Overview

Scans abstracts (and DOIs) for signals of **open-science** practices — data
availability statements, code/repository links, preprints, pre-registration and
open access — flagging each paper and reporting corpus-level adoption shares.

## Inputs
- **Data** (`Table`) — data with Abstract / DOI.

## Outputs
- **Per-paper** (`Table`) — open-science flags per paper.
- **Summary** (`Table`) — corpus-level shares.

## Controls
- **Abstract / Title / DOI** — the columns scanned.

**Actions:** `Analyze`.
