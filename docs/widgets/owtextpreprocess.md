# Text Preprocessing

> Lemmatize and remove stopwords from a text column (with an extended stopword list).

```{figure} ../_static/img/owtextpreprocess.png
:alt: Text Preprocessing
:class: widget-screenshot

The Text Preprocessing widget.
```

## Overview

Tokenises, lower-cases, lemmatises and removes stopwords from a chosen text
column (e.g. Abstract or Title), adding a `Processed <column>` column that
text-based widgets (Topic Modeling, Concept Builder, Methodology Classifier,
Novelty Metrics) can use. Beyond the basic English stopwords it can apply an
**extended** list loaded from an Excel file plus ad-hoc extra terms, and
domain-specific category word-lists.

## Inputs

- **Data** (`Table`) — data with a text column.

## Outputs

- **Data** (`Table`) — the input plus a `Processed <column>` column.

## Controls

- **Column** — the text field to process (e.g. Abstract).
- **Extra stopwords (comma/space separated)** — additional terms to drop on top of the standard list.
- **Specific categories to apply (comma sep)** — names of domain word-list categories (from the bundled stopword Excel) to remove as well.
- **Options** — toggles for lemmatisation, lower-casing and the extended stopword list.

**Actions:** `…` (browse for a custom stopword Excel), `Process`.

## Tips

- Processing once here and reusing the `Processed` column keeps every text
  widget consistent and faster.
