# Topic Modeling

> Latent topics (LDA / NMF / LSA): top terms, coherence, document-topic weights.

```{figure} ../_static/img/owtopicmodeling.png
:alt: Topic Modeling
:class: widget-screenshot

The Topic Modeling widget.
```

## Overview

Discovers latent topics in a text field (Abstract, Title, or their combination)
using **LDA**, **NMF** or **LSA**. Reports the top terms per topic with a
per-topic **coherence** score, and the per-document topic-weight matrix. A
static, whole-corpus complement to **Dynamic Topic Models** (over time) and to
**Document Clustering** (hard clusters).

## Inputs
- **Data** (`Table`) — data with a text field.

## Outputs
- **Topic Terms** (`Table`) — Topic / Term / Weight.
- **Document Topics** (`Table`) — per-document topic weights.
- **Topic Summary** (`Table`) — one row per topic (top terms + coherence).

## Controls
- **Text** — the text source (Abstract / Title / Title+Abstract / Author Keywords).
- **Model** — LDA, NMF or LSA.
- **Topics (0 = auto)** — fix the number of topics, or 0 to choose automatically up to the cap.
- **Max topics (auto)** — the upper bound when auto-selecting.
- **Top terms shown** — number of terms listed/plotted per topic.
- **Topic** *(View)* — which topic's top-term bar chart to display.

**Actions:** `Run`.
