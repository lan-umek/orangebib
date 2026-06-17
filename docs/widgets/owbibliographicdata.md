# Bibliographic Data

> Load bibliometric data from files, the OpenAlex API, or SICRIS/COBISS.

```{figure} ../_static/img/owbibliographicdata.png
:alt: Bibliographic Data
:class: widget-screenshot

The Bibliographic Data widget.
```

## Overview

**Bibliographic Data** is the entry point of every Biblium workflow. It reads a
bibliographic corpus from one of several sources, normalises the heterogeneous
column names of different databases into a common schema (Title, Abstract,
Authors, Author Keywords, Index Keywords, Year, Cited by, DOI, …), optionally
cleans the keyword/text fields, and emits a single **Data** table that every
other Biblium widget consumes.

Because each database exports list-valued fields with its own separator
(Scopus/WoS use `; `, OpenAlex uses `|`), the widget records the source so
downstream analyses split authors, keywords and affiliations correctly.

## Inputs

- *(none)* — this is a source widget.

## Outputs

- **Data** (`Table`) — the normalised corpus, one row per document, ready for
  any downstream Biblium widget.

## Loading sources

The widget offers several input modes:

- **File** — open a database export (Scopus CSV/RIS, Web of Science, OpenAlex
  CSV, PubMed, Dimensions, Lens, and 30+ formats). The format and the database
  of origin are auto-detected where possible.
- **OpenAlex API** — query OpenAlex directly (by keywords, author, institution,
  ISSN/journal or a DOI list), optionally restricting to open-access works.
- **SICRIS / COBISS** — load Slovenian personal/institutional bibliographies.
- **Sample datasets** — small bundled corpora for learning and testing.

## Controls

**Database** — the source database hint. Leave on *Auto-detect* for files; it
sets the correct list separator and column mapping. Set it explicitly if
auto-detection guesses wrong.

**Keyword processing** — cleaning applied to keyword (and, for the selected
categories, Title/Abstract) fields:

- **Apply stopwords** — master switch for the cleaning below.
- **General stopwords** — a bundled extended English stop-word list, always
  applied when cleaning is on.
- **Specific categories** — domain word-lists (from the bundled `stopwords.xlsx`)
  that you can toggle individually; checked categories are removed from keywords
  **and** from Title/Abstract.
- **User stopwords file** — your own one-word-per-line list to remove as well.
- **Extra stopwords** — a quick comma-separated list for ad-hoc terms.
- **Synonyms / thesaurus** — rules (`old = new`) that merge keyword variants
  into a single canonical form.

## What it produces

A clean, column-normalised document table. Typical next steps: **OpenAlex
Enrichment** (add citations/topics/SDGs), **Deduplicate & Merge** (combine
several exports), **Filter**, and then **Main Information** / **Bibliometric
Statistics** and the analysis widgets.

## Messages

The widget reports when a required column is missing, when a file cannot be
parsed, or when an API query returns no records.

## Tips

- For multi-database studies, load each export separately and combine them with
  the **Deduplicate & Merge** widget.
- If keyword-based widgets show merged or odd terms, check the **Database** hint
  (it controls the list separator).
