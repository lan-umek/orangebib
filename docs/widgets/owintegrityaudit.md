# Integrity Audit

> Flag research-integrity risk signals across the corpus.

```{figure} ../_static/img/owintegrityaudit.png
:alt: Integrity Audit
:class: widget-screenshot

The Integrity Audit widget.
```

## Overview

Runs a battery of heuristic **research-integrity** checks — e.g. excessive
self-citation, suspiciously fast turnaround, citation anomalies, retraction
hints — and flags papers for closer inspection, with a summary of how many
papers trip each check. A screening aid, not a verdict.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Papers with flags** (`Table`) — per-paper integrity flags.
- **Summary** (`Table`) — flag counts per check.

## Controls
- *(runs the standard battery; results appear in the Flag summary)*

**Actions:** `Run Audit`.
