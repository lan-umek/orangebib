# Semantic Scholar

> Enrich by DOI with Semantic Scholar citations, influential citations, fields and TLDR.

```{figure} ../_static/img/owsemanticscholar.png
:alt: Semantic Scholar
:class: widget-screenshot

The Semantic Scholar widget.
```

## Overview

Enriches papers with [Semantic Scholar](https://www.semanticscholar.org) data,
matched by DOI: total citation count, **influential** citation count, fields of
study, the AI-generated **TLDR** one-sentence summary, and an open-access PDF
link. Uses the Semantic Scholar Graph API batch endpoint; an API key is
optional but raises the rate limit for large corpora.

## Inputs

- **Data** (`Table`) — table with a DOI column.

## Outputs

- **Enriched Data** (`Table`) — the input plus `s2_` columns (citations, influential citations, fields of study, TLDR, OA PDF URL).

## Controls

- **DOI column** — the column whose DOIs are matched on Semantic Scholar.
- **API key (optional)** — a personal S2 API key; leave empty to use the public rate limit.

**Actions:** `Enrich from Semantic Scholar` (start), `Cancel` (stop).

## Tips

- Influential-citation counts are a useful complement to raw citations for
  identifying genuinely impactful work.
