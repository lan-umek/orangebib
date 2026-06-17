# Sleeping Beauty

> Detect long-dormant papers that suddenly awaken to heavy citation.

```{figure} ../_static/img/owsleepingbeauty.png
:alt: Sleeping Beauty
:class: widget-screenshot

The Sleeping Beauty widget.
```

## Overview

Finds **sleeping beauties** — papers that go largely uncited for years and then
experience a sharp surge — using the Beauty coefficient and related measures
from each paper's yearly citation history (OpenAlex). Returns the detected
beauties with their sleep duration and awakening intensity.

## Inputs
- **Data** (`Table`) — OpenAlex data with yearly citations.

## Outputs
- **Sleeping Beauties** (`Table`) — detected papers with their coefficients.
- **Selected** (`Table`) — the rows you select.

## Controls
- **Min Beauty Coefficient** — minimum "beauty" score to qualify.
- **Min Sleep Duration (yrs)** — minimum dormant period.
- **Min Total Citations** — exclude low-citation noise.
- **Min Awakening Intensity** — strength of the post-sleep surge.
- **Current Year** — reference year for the histories.
- **Auto apply** — recompute on change.

**Actions:** `Run Analysis`, `Export to Excel`. Feed the output to **Sleeping Beauty Plot** to see the trajectories.
