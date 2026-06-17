# Altmetrics Analysis

> Attention metrics (Altmetric / PlumX) beyond citations.

```{figure} ../_static/img/owaltmetrics.png
:alt: Altmetrics Analysis
:class: widget-screenshot

The Altmetrics Analysis widget.
```

## Overview

Adds **altmetric** attention data — news, blogs, policy documents, social media,
Mendeley readers — to papers by DOI (via Altmetric / PlumX APIs), complementing
citation-based impact with broader societal attention. Highlights the highest-
attention papers and the mix of attention sources. A simulation mode lets you
explore the widget without API keys.

## Inputs
- **Data** (`Table`) — bibliographic data (needs DOIs).

## Outputs
- **Altmetric Data** (`Table`) — input with altmetric scores.
- **Top Papers** (`Table`) — highest-attention papers.

## Controls
- **ID / DOI / Citations / Year column** — the source columns.
- **Altmetric API Key / PlumX API Key** — credentials for the live APIs.
- **Simulate altmetric data** — generate demo scores without an API key.
- **Show overview plots / source breakdown / temporal trends / top papers table** — display toggles.

**Actions:** `Run Analysis`.
