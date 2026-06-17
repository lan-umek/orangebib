# Deduplicate & Merge

> Detect/remove duplicate records and merge exports from several databases.

```{figure} ../_static/img/owdeduplicate.png
:alt: Deduplicate & Merge
:class: widget-screenshot

The Deduplicate & Merge widget.
```

## Overview

Combines one to three bibliographic tables (e.g. Scopus + Web of Science +
OpenAlex) and removes duplicate records, matching on DOI and — optionally — on
the normalised title (lower-cased, punctuation-stripped). When duplicates are
found across sources, the **richest** record (most complete fields) is kept.
Use it right after loading each export with its own **Bibliographic Data**
widget.

## Inputs

- **Data** (`Table`) — primary table.
- **Data 2** (`Table`) — second source (optional).
- **Data 3** (`Table`) — third source (optional).

## Outputs

- **Deduplicated** (`Table`) — the merged, unique corpus.
- **Flagged Duplicates** (`Table`) — every input record with duplicate flags (`_is_duplicate`, group id, match type), for inspection.
- **Merge Log** (`Table`) — which records were merged into which.

## Controls

- **Action** — *Merge & deduplicate* (combine and keep unique records) or *Flag duplicates only* (mark, but remove nothing).
- **Also match by normalised title** — in addition to DOI, treat records with the same normalised title as duplicates (catches records lacking a DOI).
- **DOI column** / **Title column** — the matching keys; *(auto)* detects `DOI`/`Title` automatically.
- **Run automatically** — recompute whenever an input or option changes (otherwise press **Run**).

**Actions:** `Run` (compute now).

## Tips

- The status line reports how many duplicates were removed by DOI vs by title.
