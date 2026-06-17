# Concept Builder

> Define custom concepts (keyword/regex sets) and tag documents with them.

```{figure} ../_static/img/owconceptbuilder.png
:alt: Concept Builder
:class: widget-screenshot

The Concept Builder widget.
```

## Overview

Lets you define your own **concepts** — named sets of keywords or regular
expressions — and flags every document that matches each concept, adding one
binary variable per concept. An overlap heat-map shows how concepts co-occur;
clicking a concept or an overlap cell outputs the matching documents. Concept
definitions can be saved to / loaded from a file for reuse.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Data** (`Table`) — input plus one binary column per concept.
- **Concept Matches** (`Table`) — documents matching any concept.
- **Selected Documents** (`Table`) — documents for the clicked concept / overlap cell.

## Controls
- **Search in** — which text field(s) concepts are matched against (Title, Abstract, Keywords, combinations).
- **Use regular expressions** — treat concept terms as regex rather than literal phrases.
- **Concept name** / **Keywords** — define a concept manually (name + its term list).
- **Auto apply** — re-tag on every change.

**Actions:** `Add Concept`, `Edit`, `Remove`, `Clear All`, `Load from File`, `Save to File`, `Apply Concepts`.

## Tips
- For a ready-made public-administration scheme, use **PA Concepts**; for the
  methodology/theory schemes, use **Methodology Classifier** / **Research Classifier**.
