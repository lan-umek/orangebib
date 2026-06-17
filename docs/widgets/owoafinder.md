# Open Access Finder

> Find free, legal full-text PDFs for papers by DOI via OpenAlex.

```{figure} ../_static/img/owoafinder.png
:alt: Open Access Finder
:class: widget-screenshot

The Open Access Finder widget.
```

## Overview

Looks up the **best open-access location** for each paper on OpenAlex and reports
its OA status (gold / green / hybrid / bronze / closed) and a direct PDF or
landing-page URL. Double-click a row to open it in the browser. Only legal,
publisher- or repository-hosted copies are used (no SciHub or other
unauthorised sources).

## Inputs

- **Data** (`Table`) — table with a DOI column.

## Outputs

- **OA Links** (`Table`) — per-paper OA status and PDF / landing URLs.

## Controls

- **DOI column** — the column whose DOIs are resolved to OA locations.
- **Email (polite pool)** — your e-mail for OpenAlex's faster polite pool (optional, recommended).

**Actions:** `Find PDFs` (start), `Cancel` (stop). The results table is
double-clickable to open each link.

## Tips

- Pair with **OpenAlex Enrichment**: both use OpenAlex, so a single e-mail in
  the polite pool benefits the whole workflow.
