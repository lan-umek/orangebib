# Research Classifier

> Tag papers with theory/framework, research-design and contribution schemes.

```{figure} ../_static/img/owresearchclassifier.png
:alt: Research Classifier
:class: widget-screenshot

The Research Classifier widget.
```

## Overview

A keyword-based classifier for three cross-cutting schemes — **theories &
frameworks**, **research design** and **contribution type** — adding a binary
indicator per category plus a category-frequency table. (The same schemes are
also available inside **Methodology Classifier**; this widget exposes them
standalone.)

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Data** (`Table`) — input plus binary indicator columns.
- **Categories** (`Table`) — category frequencies.
- **Selected Documents** (`Table`).

## Controls
- **Source** — the text field classified (Title / Abstract / Keywords / combination).
- **Theories & frameworks** / **Research design** / **Contribution type** — which schemes to apply.

**Actions:** `Classify`.
