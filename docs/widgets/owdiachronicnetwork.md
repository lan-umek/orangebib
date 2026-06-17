# Diachronic Network

> Animated growth of a co-occurrence network across time periods.

## Overview

Animates how a co-occurrence network (keywords, co-authorship, …) **grows and
restructures** over successive time periods — nodes and edges appearing as the
field develops. Playable, scrubbable and exportable to GIF; the documents up to
the current period are sent onward.

```{admonition} Screenshot
:class: tip
Animated widget — best viewed live in Orange.
```

## Inputs
- **Data** (`Table`) — bibliographic data with Year.

## Outputs
- **Selected Documents** (`Table`) — documents up to the current period.

## Controls
- **Item type** — the entity whose network is animated.
- **Top N nodes** — nodes shown.
- **Periods** — number of time slices.
- **Min edge weight** — prune weak edges.
- **Speed (ms)** — animation frame interval.

**Actions:** `▶ Play`, `⟲` (restart), `⬇ Export animation (GIF)`.
