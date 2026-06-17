# SDG Identifier

> Tag documents with UN Sustainable Development Goals from their text.

```{figure} ../_static/img/owsdgidentifier.png
:alt: SDG Identifier
:class: widget-screenshot

The SDG Identifier widget.
```

## Overview

Classifies each document against the 17 UN **Sustainable Development Goals**
using curated query dictionaries matched on the text, adding a variable per SDG
(and optionally per perspective/dimension). Produces an SDG distribution summary
and the documents addressing each goal.

## Inputs
- **Data** (`Table`) — bibliographic data.

## Outputs
- **Data** (`Table`) — input plus SDG variables.
- **SDG Summary** (`Table`) — distribution across the 17 goals.
- **SDG Documents** (`Table`) — documents with at least one SDG.

## Controls
- **Text Column** — the field matched (Title, Abstract, Keywords, or Title+Abstract+Keywords).
- **Include Perspectives / Include Dimensions** — also tag the broader SDG perspectives/pillars.
- **Queries File** — load a custom SDG query dictionary; reset to default.
- **Add results to dataset** — append the SDG columns to the data output.
- **Use numeric labels (0/1)** — emit binary columns.
- **Colour by** — colour mode for the summary (SDG number/name/pillar/dimension).
- **Auto apply** — re-tag on change.

**Actions:** `Identify SDGs`, `Browse`, `Reset to Default`, `Export Results`.
