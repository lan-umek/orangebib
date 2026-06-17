# Filter

> Keep records matching simple/compound numeric, text, regex or Bradford criteria.

```{figure} ../_static/img/owdatafilter.png
:alt: Filter
:class: widget-screenshot

The Filter widget.
```

## Overview

Subsets a corpus with one or more conditions:

- **numeric** criteria (`>`, `>=`, `<`, `<=`, `==`, `!=`, *between*) — e.g. Year ≥ 2015, Cited by ≥ 10;
- **text / keyword** criteria (*contains*, *not contains*, *equals*, *regex*, *in list*);
- combined with **AND** / **OR**.

It also offers a **Bradford's-law** filter that keeps only documents published
in the *core* zone — the few most productive sources that carry a third of the
literature. Both the matching and the non-matching records are emitted, so the
widget can also split a corpus in two.

## Inputs

- **Data** (`Table`).

## Outputs

- **Matching Data** (`Table`) — records satisfying the rules.
- **Non-matching Data** (`Table`) — the complement.

## Controls

- **Combine rules with** — **AND** (all rules must hold) or **OR** (any rule).
- Each **rule** — a column, an operator and a value; add/remove rows as needed.
- **Source column** *(Bradford)* — the journal/source field used to build Bradford zones.
- **Number of zones** *(Bradford)* — how many equal-citation zones to split sources into; the first zone is the kept *core*.

**Actions:** `+ rule` / `− rule` (add/remove a condition), `Apply filter`.

## Tips

- The status line shows how many of the total records match.
