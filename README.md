# Orange3-Biblium

**Bibliometric & scientometric analysis widgets for [Orange Data Mining](https://orangedatamining.com/).**

Orange3-Biblium (package `orangebib`) brings a complete bibliometric workflow into
Orange's visual programming canvas — from loading and enriching data, through
counts, statistics, citation and network analysis, SDG and methodology
classification, to interactive reports and an HTML dashboard. It wraps the
[Biblium](https://github.com/lan-umek/biblium) analysis engine.

> Status: research software, **0.2.0** (beta).

## Highlights

- **81 widgets** covering the full bibliometric pipeline.
- Load **Scopus, Web of Science, OpenAlex, PubMed, Dimensions, Lens, SICRIS/COBISS** and 30+ export formats.
- Query the **OpenAlex API** directly; enrich by DOI (citations, OA status, topics, SDGs, references).
- **Deduplicate & merge** multi-database exports.
- Co-occurrence, citation, co-citation, coupling, collaboration and **field networks** (Jaccard, disparity backbone, bridging), with a VOSviewer-style density view.
- Topic modeling, thematic maps/evolution, methodology & SDG classification.
- Group comparison, permutation inference, (Firth) logistic regression.
- One-click **HTML report** and **interactive dashboard**.

## Installation

### For users (Orange desktop)

In Orange: **Options ▸ Add-ons ▸ Add more…** and install *Orange3-Biblium* (once published to PyPI).

### With pip

```bash
pip install Orange3-Biblium
```

This pulls in `biblium` (the analysis engine) automatically.

### From source (development)

```bash
git clone https://github.com/lan-umek/orange3-biblium
cd orange3-biblium
pip install -e .
```

To register the widgets inside an existing Orange install, run the above with
**Orange's** Python, then restart Orange.

## Quick start

1. Drop a **Bibliographic Data** widget and load a Scopus/WoS/OpenAlex export (or query OpenAlex).
2. Optionally **OpenAlex Enrichment** and **Deduplicate & Merge**.
3. Explore: **Main Information**, **Bibliometric Statistics → Performance Plot**, **Network Co-occurrence → Plot Bibliometric Network**, …
4. Summarise with the **Report Generator** or **HTML Dashboard**.

See the [documentation](https://orange3-biblium.readthedocs.io) for per-widget help.

## Widget catalog

| # | Widget | Stage | Description |
|---|--------|-------|-------------|
| 1 | **Bibliographic Data** | Data input & preparation | Load bibliometric data from files, OpenAlex API, or SICRIS/COBISS |
| 2 | **OpenAlex Enrichment** | Data input & preparation | Add OpenAlex metadata (citations, OA, topics, SDGs, refs) by DOI |
| 3 | **Semantic Scholar** | Data input & preparation | Enrich by DOI with S2 citations, influential cites, fields, TLDR |
| 4 | **Open Access Finder** | Data input & preparation | Find free full-text (PDF) for papers by DOI via OpenAlex |
| 5 | **Deduplicate & Merge** | Data input & preparation | Detect and remove duplicate records and merge exports from |
| 6 | **Filter** | Data input & preparation | Filter records with simple/compound numeric, text, regex and Bradford criteria |
| 7 | **Setup Groups** | Data input & preparation | Define document groups for comparative analysis |
| 8 | **Text Preprocessing** | Data input & preparation | Lemmatize and remove stopwords (with an extended stopword file) |
| 9 | **Main Information** | Overview, counts & statistics | Compute comprehensive bibliometric statistics and dataset overview |
| 10 | **Bibliometric Counts** | Overview, counts & statistics | Count occurrences of entities (authors, keywords, sources, etc.) in bibliographic data |
| 11 | **Bibliometric Statistics** | Overview, counts & statistics | Compute performance indicators (H-index, G-index, etc.) for bibliometric entities |
| 12 | **Performance Plot** | Overview, counts & statistics | Performance plot (bar / scatter / projection) of an entity-statistics table |
| 13 | **Bibliometric Laws** | Overview, counts & statistics | Analyze Lotka, Bradford, Zipf, Price, and Pareto bibliometric laws |
| 14 | **Production Plot** | Overview, counts & statistics | Visualize scientific production with bar and line charts |
| 15 | **Top Cited** | Overview, counts & statistics | Global and local top-cited documents |
| 16 | **Reference Diversity** | Overview, counts & statistics | Shannon/Simpson/Rao-Stirling diversity of each paper |
| 17 | **Trend Analysis** | Time, trends & dynamics | Analyze temporal patterns and trends in your data |
| 18 | **Trend Topics** | Time, trends & dynamics | Analyze trending topics ordered by median publication year |
| 19 | **Top Items Timeline** | Time, trends & dynamics | Bubble plot showing entity production over time |
| 20 | **Entity Over Time** | Time, trends & dynamics | Analyze production over time of authors, keywords, sources |
| 21 | **Life Cycle Analysis** | Time, trends & dynamics | Analyze the life cycle of scientific production using logistic growth model |
| 22 | **RaceBar** | Time, trends & dynamics | Animated bar-chart race of top items over time |
| 23 | **Burst Detection** | Time, trends & dynamics | Detect bursts of activity in keywords over time (Kleinberg) |
| 24 | **Hot Topics** | Time, trends & dynamics | Emerging topics by recent activity and citation momentum |
| 25 | **Citation Distribution** | Citation analysis | Analyze citation distribution and impact metrics |
| 26 | **Citation Velocity** | Citation analysis | Citation accumulation speed and trend classification per paper |
| 27 | **Citation Patterns** | Citation analysis | Classify papers by citation trajectory |
| 28 | **Disruption Index** | Citation analysis | Measure whether papers consolidate or disrupt fields |
| 29 | **Self-Citation Rate** | Citation analysis | Author and journal self-citation rates from the within-corpus citation network |
| 30 | **Sleeping Beauty** | Citation analysis | Detect papers with delayed recognition (dormant period followed by awakening) |
| 31 | **Sleeping Beauty Plot** | Citation analysis | Interactive visualization of sleeping beauty detection results |
| 32 | **CiteSpace Metrics** | Citation analysis | Pivotal nodes (betweenness), burstness, sigma and labelled clusters |
| 33 | **Concept Builder** | Text, topics & concepts | Create binary concept variables from keywords |
| 34 | **PA Concepts** | Text, topics & concepts | Create Public Administration paradigm indicators |
| 35 | **Topic Modeling** | Text, topics & concepts | Latent topics (LDA / NMF / LSA): top terms, coherence, doc-topic weights |
| 36 | **Dynamic Topic Models** | Text, topics & concepts | Topic evolution over time with trajectories (sequential LDA) |
| 37 | **Thematic Map** | Text, topics & concepts | Strategic diagram showing research themes by centrality and density |
| 38 | **Thematic Evolution** | Text, topics & concepts | Alluvial flow of strategic themes across time periods |
| 39 | **Conceptual Drift** | Text, topics & concepts | How the meaning/context of terms shifts over time |
| 40 | **Novelty Metrics** | Text, topics & concepts | Uzzi-style combinatorial novelty and atypicality per paper |
| 41 | **Methodology Classifier** | Text, topics & concepts | Classify paradigm, designs, data sources and methods per paper |
| 42 | **Research Classifier** | Text, topics & concepts | Keyword classification of documents by theory/framework, |
| 43 | **Research Gaps** | Text, topics & concepts | Identify under-studied (SDG/geographic/methodological/temporal) gaps |
| 44 | **Network Co-occurrence** | Networks | Build co-occurrence networks from bibliographic data |
| 45 | **Plot Bibliometric Network** | Networks | Plot a bibliometric co-occurrence network: nice layout, largest components, node selection and Pajek export |
| 46 | **Field Networks** | Networks | Normalised field co-occurrence heatmap, disparity-filter |
| 47 | **Citation Network** | Networks | Document citation network with main path analysis |
| 48 | **Historiograph** | Networks | Chronological document citation network (HistCite / CiteNet Explorer style) |
| 49 | **Co-citation Network** | Networks | Network of references cited together (intellectual base) |
| 50 | **Bibliographic Coupling** | Networks | Network of papers sharing references (needs OpenAlex refs) |
| 51 | **Collaboration Network** | Networks | Country collaboration network from co-authored affiliations |
| 52 | **Main Path Analysis** | Networks | Main path of knowledge flow (SPC/SPLC/SPNP) from OpenAlex citations |
| 53 | **Diachronic Network** | Networks | Animated co-occurrence network growing over time |
| 54 | **Embedding Landscape** | Networks | 2-D semantic map of documents (embeddings + clustering) |
| 55 | **Entity Relationships** | Networks | Analyse co-occurrence relationships between two entity types: |
| 56 | **K-Fields Plot** | Networks | Visualize relationships between K bibliometric fields using Sankey diagram |
| 57 | **Geographic Analysis** | Geography, SDG & disciplines | Per-country metrics (documents, citations, collaboration) with map coordinates |
| 58 | **Discipline Analysis** | Geography, SDG & disciplines | Profile a corpus across OpenAlex domains / fields / subfields |
| 59 | **SDG Identifier** | Geography, SDG & disciplines | Identify Sustainable Development Goals in your dataset |
| 60 | **SDG Networks** | Geography, SDG & disciplines | Co-occurrence network of Sustainable Development Goals + bridge papers |
| 61 | **SDG Drift** | Geography, SDG & disciplines | How the vocabulary of each SDG drifts across time windows |
| 62 | **Group Counts** | Groups, comparison & inference | Count and compare entity frequencies across document groups |
| 63 | **Group Counts Plot** | Groups, comparison & inference | Interactive horizontal bar chart comparing entity |
| 64 | **Group Statistics** | Groups, comparison & inference | Compute performance statistics (counts, fractions, ranks, |
| 65 | **Group Intersections** | Groups, comparison & inference | Analyze document overlap between groups |
| 66 | **Group Associations** | Groups, comparison & inference | Analyse entity–group relationships: contingency, diversity, CA, |
| 67 | **Crosstabs** | Groups, comparison & inference | Contingency table with chi-squared test and effect sizes |
| 68 | **Compare Means** | Groups, comparison & inference | Compare a numeric variable across groups (t-test/ANOVA + post-hoc) |
| 69 | **Comparative Analysis** | Groups, comparison & inference | Compare groups (by a column) across bibliometric metrics |
| 70 | **Benchmarking** | Groups, comparison & inference | Compare dataset distributions against global research patterns |
| 71 | **Permutation Inference** | Groups, comparison & inference | Permutation tests of group×entity association — valid for |
| 72 | **Logistic Regression** | Groups, comparison & inference | Binary logistic regression (coefficients, p-values, odds ratios) |
| 73 | **Factorial Analysis** | Groups, comparison & inference | MCA, CA, PCA with clustering and visualization |
| 74 | **Document Clustering** | Groups, comparison & inference | Cluster documents by text (and optionally shared references) |
| 75 | **Altmetrics Analysis** | Niche, external & output | Analyze alternative impact metrics beyond citations |
| 76 | **Open Science** | Niche, external & output | Open access, data/code availability, preprints, preregistration |
| 77 | **Integrity Audit** | Niche, external & output | Screen for integrity red flags (tortured phrases, retractions, anomalies) |
| 78 | **AI Describe** | Niche, external & output | Generate a natural-language description of a table with an LLM |
| 79 | **Literature Matrix** | Niche, external & output | Extract structured fields from each paper into a review matrix (LLM) |
| 80 | **Report Generator** | Niche, external & output | Generate Word/Excel/PowerPoint/LaTeX bibliometric reports |
| 81 | **HTML Dashboard** | Niche, external & output | Generate a self-contained interactive HTML dashboard |

## Documentation

Full docs are built with Sphinx in `docs/`. Build locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Citing

If you use Orange3-Biblium in your research, please cite it (see `CITATION.cff`).

## License

Released under the **MIT License**. See [LICENSE](LICENSE).

## Author

Lan Umek, University of Ljubljana.
