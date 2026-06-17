# PA Concepts

> Tag documents with a built-in public-administration concept scheme.

```{figure} ../_static/img/owpaconcepts.png
:alt: PA Concepts
:class: widget-screenshot

The PA Concepts widget.
```

## Overview

A ready-made variant of **Concept Builder** carrying a curated
**public-administration** concept dictionary. Select the PA concepts of
interest (or load your own) and the widget adds a variable per concept plus a
distribution summary — a quick way to map a corpus onto established PA themes.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Data** (`Table`) — input plus PA concept variables.
- **Summary** (`Table`) — distribution of the PA concepts.
- **PA Documents** (`Table`) — documents carrying PA concepts.

## Controls
- **Search in** — text field(s) to match against.
- **Use regular expressions** — regex vs literal matching.
- **Use numeric labels (0/1)** — emit binary 0/1 columns instead of yes/no.
- **Select PA Concepts** — choose which built-in concepts to apply.
- **Custom Concepts** — browse/load a custom dictionary; reset to default.
- **Auto apply** — re-tag on change.

**Actions:** `Create PA Concept Variables`, `Select All` / `Select None`, `Keywords Preview`, `Browse`, `Reset to Default`, `Export Results`.
