# OpenAlex Enrichment

> Add OpenAlex metadata (citations, OA, topics, SDGs, references) by DOI.

```{figure} ../_static/img/owopenalexenrichment.png
:alt: OpenAlex Enrichment
:class: widget-screenshot

The OpenAlex Enrichment widget.
```

## Overview

Enriches any bibliographic table with metadata from [OpenAlex](https://openalex.org),
matched by DOI. DOIs are sent in batches (OpenAlex OR-filter, up to 50 per
request) and the matching works are merged back, adding `oa_` columns:
citation count, open-access status, topics / fields / domains, concepts,
Sustainable Development Goals, referenced works (for citation networks),
yearly citation counts and author institutions. The run is cached on disk, so
it can be stopped and resumed without re-fetching.

Run it right after **Bibliographic Data** when your export lacks citations,
references or topical metadata (e.g. a plain Scopus title list), or to attach
OpenAlex IDs needed by the citation-network widgets.

## Inputs

- **Data** (`Table`) — bibliographic table; must contain a DOI column.

## Outputs

- **Enriched Data** (`Table`) — the input table plus the selected `oa_` columns.

## Controls

- **DOI column** — the column holding DOIs; these are the keys matched against OpenAlex.
- **Email (polite pool)** — your e-mail address. OpenAlex's *polite pool* gives faster, more reliable responses; optional but recommended.
- **Batch size** — number of DOIs per API request (max 50). Larger = fewer requests.
- **Delay (s)** — pause between requests, to stay within rate limits.
- **Fields to add** — checkboxes selecting which `oa_` groups to fetch (citations, OA status, topics/fields/domains, concepts, SDGs, referenced works, counts-by-year, institutions). Fewer fields = faster.

**Actions:** `Select All` / `Deselect All` (toggle the field checkboxes), `Enrich from OpenAlex` (start), `Cancel` (stop a running fetch).

## What it produces

The same documents with extra `oa_` columns. Common follow-ups: **Discipline
Analysis** and **SDG Identifier** (use the topic/SDG columns), and **Citation
Network** / **Historiograph** (use `oa_openalex_id` + `oa_referenced_works`).

## Tips

- Provide an e-mail for the polite pool to avoid throttling on large corpora.
- Time-series widgets need the **counts-by-year** field (`oa_counts_by_year`).
