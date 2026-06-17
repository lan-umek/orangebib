# Bibliometric Counts

> Frequency counts of any entity (authors, sources, keywords, countries, n-grams…).

```{figure} ../_static/img/owbibliometriccounts.png
:alt: Bibliometric Counts
:class: widget-screenshot

The Bibliometric Counts widget.
```

## Overview

Counts how often each entity occurs in the corpus — authors, sources, author or
index keywords, countries, affiliations, document types, or title/abstract
n-grams — and returns a ranked table (frequency, share, rank). Selecting rows
in the preview sends the matching documents onward, so it doubles as an
entity-based document selector.

## Inputs

- **Data** (`Table`) — bibliographic data.

## Outputs

- **Counts** (`Table`) — entity, frequency, share and rank.
- **Selected Documents** (`Table`) — documents containing the entities you select in the preview.

## Controls

- **Entity Type** — what to count (Authors, Sources, Author/Index Keywords, Countries, Affiliations, Document Type, Title/Abstract n-grams…).
- **Column** — the source column (auto-chosen from the entity type, override if needed).
- **Separator** — the list separator for multi-valued cells (auto-detected; `;`, `|`, …).
- **Count as** — single value, delimited list, or free-text tokens.
- **Top N** — keep only the N most frequent entities.
- **Min Count** — drop entities occurring fewer than this many times.
- **Min n-gram / Max n-gram** — n-gram length range (for the text n-gram modes).

**Actions:** `Count Entities`, `Select All` / `Clear Selection` (preview rows → Selected Documents).

## Tips

- Counts here feed naturally into **Bibliometric Statistics** (add citations/h-index) and the network widgets.
