# Citation Velocity

> How fast each paper accrues citations (recent vs lifetime rate).

```{figure} ../_static/img/owcitationvelocity.png
:alt: Citation Velocity
:class: widget-screenshot

The Citation Velocity widget.
```

## Overview

Measures the **rate** of citation accrual per paper rather than the raw count:
citations per year, the recent-window rate, and whether a paper is
accelerating, steady or fading. Useful for spotting momentum that totals alone
hide. Needs yearly citation data (e.g. from OpenAlex enrichment) or at least
publication year and total citations.

## Inputs
- **Data** (`Table`) — data with citations (ideally yearly counts).

## Outputs
- **Per-paper Metrics** (`Table`) — velocity metrics per document.
- **Summary** (`Table`) — counts of accelerating/steady/fading papers.

## Controls
- **Max papers** — cap papers processed (speed).
- **Current year** — reference year for age and recent-rate calculations.
- **Recent window (yrs)** — length of the recent window for the recent rate.
- **Min paper age** — ignore very new papers (too little history).

**Actions:** `Compute Velocity`, `Cancel`.
