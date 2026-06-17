# Trend Topics

> When topics peak — each term plotted at its median year with a Q1–Q3 spread.

```{figure} ../_static/img/owtrendtopics.png
:alt: Trend Topics
:class: widget-screenshot

The Trend Topics widget.
```

## Overview

A `biblioshiny`-style **trend topics** chart: for the most frequent terms it
plots a bubble at each term's **median publication year** with a line spanning
its interquartile (Q1–Q3) year range, so you can read at a glance which topics
are historical, current or emerging. Clicking a term outputs its documents.

## Inputs
- **Data** (`Table`) — keywords + Year.

## Outputs
- **Selected Documents** (`Table`) — documents of the clicked term.
- **Trend Data** (`Table`) — per-term median/quartile years and frequency.

## Controls
- **Min Documents** — ignore terms occurring in fewer documents.
- **Top N per Year** — how many top terms to label per year.
- **Regex Filter** — keep only terms matching a pattern.
- **Color By** + **Color Map** — colour bubbles by a property (e.g. frequency), default *viridis*.
- **Title / X label / Y label** — editable labels.
- **Show grid** — toggle grid.

**Actions:** `Run Analysis`.
