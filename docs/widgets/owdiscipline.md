# Discipline Analysis

> Profile a corpus across OpenAlex domains / fields / subfields / topics, over time.

```{figure} ../_static/img/owdiscipline.png
:alt: Discipline Analysis
:class: widget-screenshot

The Discipline Analysis widget.
```

## Overview

Profiles the disciplinary make-up of a corpus using OpenAlex knowledge levels
(domains, fields, subfields, topics): a ranked **bar profile** of the most
frequent entities, and a **dynamics** view tracking how the top entities evolve
year by year. Bars can be coloured by a meaningful metric (e.g. average document
age). Requires OpenAlex enrichment (`oa_domains/fields/subfields/topics`).

## Inputs
- **Data** (`Table`) — OpenAlex-enriched data.

## Outputs
- **Profile** (`Table`) — counts and % per entity.
- **Dynamics** (`Table`) — year × entity matrix of the top entities.

## Controls
- **Level** — Domains / Fields / Subfields / Topics.
- **Top N** — number of entities shown.
- **Colour by** — bar colour metric: Average age (default), Average year, % of corpus, or uniform.
- **Colormap** — colour scheme (default *viridis*); used when colouring by a metric.
- **Dynamics** — show the trajectories as Share (%) or Raw counts.

## Tips
- If the `oa_*` columns are missing, run **OpenAlex Enrichment** first.
