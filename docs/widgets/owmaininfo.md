# Main Information

> Headline summary, performance indicators, time series and descriptives for a corpus.

```{figure} ../_static/img/owmaininfo.png
:alt: Main Information
:class: widget-screenshot

The Main Information widget.
```

## Overview

Gives the one-look overview of a corpus, the way `biblioshiny`'s *Main
Information* does: dataset summary (documents, sources, authors, time span,
average citations, collaboration), a block of performance indicators, a yearly
time series (production and citation growth), and descriptive statistics of the
numeric fields. Each block is emitted as its own table so you can wire exactly
what you need downstream.

## Inputs

- **Data** (`Table`) — bibliographic data.

## Outputs

- **Summary** (`Table`) — headline dataset summary.
- **Performance** (`Table`) — performance indicators.
- **Time Series** (`Table`) — per-year production / citation series.
- **Descriptives** (`Table`) — descriptive statistics of numeric columns.
- **All Statistics** (`Table`) — the above combined.

## Controls

- **Dataset Summary / Performance Indicators / Time Series Analysis / Descriptive Statistics** — toggles selecting which blocks to compute (each maps to an output).
- **Exclude last year for growth rates** — drops the (usually incomplete) final year so growth rates aren't biased downward.
- **Detail level** — how much to report (core → extended → full).
- **Apply Automatically** — recompute on every change; otherwise press **Compute Statistics**.

**Actions:** `Compute Statistics`.

## Tips

- Use this as the first analytical widget after loading/cleaning, to sanity-check coverage and time span.
