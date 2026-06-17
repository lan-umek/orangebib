# Methodology Classifier

> Classify papers by research method, paradigm, theory, design and contribution.

```{figure} ../_static/img/owmethodology.png
:alt: Methodology Classifier
:class: widget-screenshot

The Methodology Classifier widget.
```

## Overview

Classifies each paper's **methodology** from its text (Title/Abstract/Keywords):
the methods used and their broad **paradigm** (qualitative / quantitative /
mixed), and — optionally — keyword-based schemes for **theories & frameworks**,
**research design** and **contribution type**. Emits binary variables per
method/paradigm/scheme and distribution tables; selecting heat-map cells outputs
the documents using a pair of methods.

## Inputs
- **Data** (`Table`) — needs Abstract / Title (and optionally keywords).

## Outputs
- **Per-document** (`Table`) — methodology flags per paper.
- **Paradigms** (`Table`) — paradigm distribution.
- **Methods** (`Table`) — method distribution.
- **Selected Documents** (`Table`) — documents using the selected method(s).

## Controls
- **Text column** — the text source for classification (Title, Abstract, both Keywords, or the combination).
- **Theories & frameworks** / **Research design** / **Contribution type** — enable these additional keyword-based schemes.

**Actions:** `Classify`.
