# Self-Citation Rate

> Author and journal self-citation shares.

```{figure} ../_static/img/owselfcitation.png
:alt: Self-Citation Rate
:class: widget-screenshot

The Self-Citation Rate widget.
```

## Overview

Estimates how much authors and journals cite **themselves**: the share of an
author's (or journal's) citations that come from their own later work. High
self-citation can inflate impact metrics, so this is a useful integrity check.
Requires references.

## Inputs
- **Data** (`Table`) — data with references.

## Outputs
- **Author Self-Citation** (`Table`) — per-author self-citation share.
- **Journal Self-Citation** (`Table`) — per-journal self-citation share.

## Controls
- **Min citations given (author/journal)** — ignore authors/journals below this citation volume (rates are noisy for tiny counts).

**Actions:** `Compute`.
