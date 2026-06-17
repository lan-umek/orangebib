# HTML Dashboard

> Self-contained interactive HTML dashboard of the corpus.

```{figure} ../_static/img/owdashboard.png
:alt: HTML Dashboard
:class: widget-screenshot

The HTML Dashboard widget.
```

## Overview

Generates a single, self-contained **interactive HTML dashboard** — overview,
production, sources, authors, keywords, word cloud, networks, citations and more
— that opens in any browser with no dependencies. Built with biblium's Dashboard
engine; the list separator is auto-detected so OpenAlex/Scopus data both render.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- *(none)* — writes a `.html` file and opens it in the browser.

## Controls
- **Title** / **Subtitle** — dashboard headings.
- **Theme** — light or dark.
- **Top N per section** — how many items each section lists.

**Actions:** `Create dashboard…` (choose where to save, then build), `Open in browser`.

## Tips
- Building runs in the background; large corpora take a little time.
- For a static, citable document instead, use the **Report Generator**.
