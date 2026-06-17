# Getting started

A typical workflow follows the order of the widget toolbox.

## 1. Load data

Place a **Bibliographic Data** widget and either open a database export
(Scopus, Web of Science, OpenAlex, …) or query the OpenAlex API. Sample
datasets are included for learning.

## 2. Prepare

- **OpenAlex Enrichment** — add citations, open-access status, topics, SDGs and references by DOI.
- **Deduplicate & Merge** — combine Scopus + WoS + OpenAlex and remove duplicates.
- **Filter** / **Text Preprocessing** — subset records and clean keywords/abstracts.
- **Setup Groups** — define groups for later comparison.

## 3. Describe & analyse

- **Main Information**, **Bibliometric Counts**, **Bibliometric Statistics** → **Performance Plot**.
- **Bibliometric Laws** (Lotka/Bradford/Zipf), production and trend widgets.
- Citations: **Citation Distribution**, **Citation Velocity**, **Disruption Index**, …
- Text & topics: **Topic Modeling**, **Thematic Map/Evolution**, **Methodology Classifier**.
- Networks: **Network Co-occurrence** → **Plot Bibliometric Network**, **Field Networks**,
  **Citation Network**, **Co-citation**, **Bibliographic Coupling**, **Collaboration**.
- Groups & inference: **Group Associations**, **Compare Means**, **Permutation Inference**,
  **Logistic Regression** (with Firth option).

## 4. Report

Summarise with the **Report Generator** (HTML/PDF report) or the
**HTML Dashboard** (interactive, single file).

See the {doc}`widgets/index` for details on every widget.
