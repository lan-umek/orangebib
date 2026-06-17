```{toctree}
:hidden:
:maxdepth: 1

owbibliographicdata
owopenalexenrichment
owsemanticscholar
owoafinder
owdeduplicate
owdatafilter
owsetupgroups
owtextpreprocess
owmaininfo
owbibliometriccounts
owbibliometricstats
owbiblioscatter
owbibliometriclaws
owproductionplot
owtopcited
owreferencediversity
owtrendanalysis
owtrendtopics
owtopitemstimeline
owentityovertime
owlifecycle
owracebar
owburstdetection
owhottopics
owcitationdistribution
owcitationvelocity
owcitationpatterns
owdisruptionindex
owselfcitation
owsleepingbeauty
owsleepingbeautyplot
owcitespace
owconceptbuilder
owpaconcepts
owtopicmodeling
owdynamictopics
owthematicmap
owthematicevolution
owconceptualdrift
ownoveltymetrics
owmethodology
owresearchclassifier
owresearchgaps
ownetworkcooccurrence
owbiblionetwork
owfieldnetworks
owcitationnetwork
owhistoriograph
owcocitation
owbibcoupling
owcollaboration
owmainpath
owdiachronicnetwork
owembeddinglandscape
owentityrelations
owkfieldsplot
owgeographic
owdiscipline
owsdgidentifier
owsdgnetworks
owsdgdrift
owgroupcounts
owgroupcountsplot
owgroupstatistics
owgroupintersections
owgroupassociations
owcrosstabs
owcomparemeans
owcomparative
owbenchmarking
owpermutationtest
owlogisticregression
owfactorialanalysis
owdocclustering
owaltmetrics
owopenscience
owintegrityaudit
owaidescribe
owliteraturematrix
owreportgenerator
owdashboard
```

# Widget reference

Biblium contributes 81 widgets, grouped by analysis stage. Each page documents the inputs, outputs, controls and messages.


## Data input & preparation

- [Bibliographic Data](owbibliographicdata.md) — Load bibliometric data from files, OpenAlex API, or SICRIS/COBISS
- [OpenAlex Enrichment](owopenalexenrichment.md) — Add OpenAlex metadata (citations, OA, topics, SDGs, refs) by DOI
- [Semantic Scholar](owsemanticscholar.md) — Enrich by DOI with S2 citations, influential cites, fields, TLDR
- [Open Access Finder](owoafinder.md) — Find free full-text (PDF) for papers by DOI via OpenAlex
- [Deduplicate & Merge](owdeduplicate.md) — Detect and remove duplicate records and merge exports from
- [Filter](owdatafilter.md) — Filter records with simple/compound numeric, text, regex and Bradford criteria
- [Setup Groups](owsetupgroups.md) — Define document groups for comparative analysis
- [Text Preprocessing](owtextpreprocess.md) — Lemmatize and remove stopwords (with an extended stopword file)

## Overview, counts & statistics

- [Main Information](owmaininfo.md) — Compute comprehensive bibliometric statistics and dataset overview
- [Bibliometric Counts](owbibliometriccounts.md) — Count occurrences of entities (authors, keywords, sources, etc.) in bibliographic data
- [Bibliometric Statistics](owbibliometricstats.md) — Compute performance indicators (H-index, G-index, etc.) for bibliometric entities
- [Performance Plot](owbiblioscatter.md) — Performance plot (bar / scatter / projection) of an entity-statistics table
- [Bibliometric Laws](owbibliometriclaws.md) — Analyze Lotka, Bradford, Zipf, Price, and Pareto bibliometric laws
- [Production Plot](owproductionplot.md) — Visualize scientific production with bar and line charts
- [Top Cited](owtopcited.md) — Global and local top-cited documents
- [Reference Diversity](owreferencediversity.md) — Shannon/Simpson/Rao-Stirling diversity of each paper

## Time, trends & dynamics

- [Trend Analysis](owtrendanalysis.md) — Analyze temporal patterns and trends in your data
- [Trend Topics](owtrendtopics.md) — Analyze trending topics ordered by median publication year
- [Top Items Timeline](owtopitemstimeline.md) — Bubble plot showing entity production over time
- [Entity Over Time](owentityovertime.md) — Analyze production over time of authors, keywords, sources
- [Life Cycle Analysis](owlifecycle.md) — Analyze the life cycle of scientific production using logistic growth model
- [RaceBar](owracebar.md) — Animated bar-chart race of top items over time
- [Burst Detection](owburstdetection.md) — Detect bursts of activity in keywords over time (Kleinberg)
- [Hot Topics](owhottopics.md) — Emerging topics by recent activity and citation momentum

## Citation analysis

- [Citation Distribution](owcitationdistribution.md) — Analyze citation distribution and impact metrics
- [Citation Velocity](owcitationvelocity.md) — Citation accumulation speed and trend classification per paper
- [Citation Patterns](owcitationpatterns.md) — Classify papers by citation trajectory
- [Disruption Index](owdisruptionindex.md) — Measure whether papers consolidate or disrupt fields
- [Self-Citation Rate](owselfcitation.md) — Author and journal self-citation rates from the within-corpus citation network
- [Sleeping Beauty](owsleepingbeauty.md) — Detect papers with delayed recognition (dormant period followed by awakening)
- [Sleeping Beauty Plot](owsleepingbeautyplot.md) — Interactive visualization of sleeping beauty detection results
- [CiteSpace Metrics](owcitespace.md) — Pivotal nodes (betweenness), burstness, sigma and labelled clusters

## Text, topics & concepts

- [Concept Builder](owconceptbuilder.md) — Create binary concept variables from keywords
- [PA Concepts](owpaconcepts.md) — Create Public Administration paradigm indicators
- [Topic Modeling](owtopicmodeling.md) — Latent topics (LDA / NMF / LSA): top terms, coherence, doc-topic weights
- [Dynamic Topic Models](owdynamictopics.md) — Topic evolution over time with trajectories (sequential LDA)
- [Thematic Map](owthematicmap.md) — Strategic diagram showing research themes by centrality and density
- [Thematic Evolution](owthematicevolution.md) — Alluvial flow of strategic themes across time periods
- [Conceptual Drift](owconceptualdrift.md) — How the meaning/context of terms shifts over time
- [Novelty Metrics](ownoveltymetrics.md) — Uzzi-style combinatorial novelty and atypicality per paper
- [Methodology Classifier](owmethodology.md) — Classify paradigm, designs, data sources and methods per paper
- [Research Classifier](owresearchclassifier.md) — Keyword classification of documents by theory/framework,
- [Research Gaps](owresearchgaps.md) — Identify under-studied (SDG/geographic/methodological/temporal) gaps

## Networks

- [Network Co-occurrence](ownetworkcooccurrence.md) — Build co-occurrence networks from bibliographic data
- [Plot Bibliometric Network](owbiblionetwork.md) — Plot a bibliometric co-occurrence network: nice layout, largest components, node selection and Pajek export
- [Field Networks](owfieldnetworks.md) — Normalised field co-occurrence heatmap, disparity-filter
- [Citation Network](owcitationnetwork.md) — Document citation network with main path analysis
- [Historiograph](owhistoriograph.md) — Chronological document citation network (HistCite / CiteNet Explorer style)
- [Co-citation Network](owcocitation.md) — Network of references cited together (intellectual base)
- [Bibliographic Coupling](owbibcoupling.md) — Network of papers sharing references (needs OpenAlex refs)
- [Collaboration Network](owcollaboration.md) — Country collaboration network from co-authored affiliations
- [Main Path Analysis](owmainpath.md) — Main path of knowledge flow (SPC/SPLC/SPNP) from OpenAlex citations
- [Diachronic Network](owdiachronicnetwork.md) — Animated co-occurrence network growing over time
- [Embedding Landscape](owembeddinglandscape.md) — 2-D semantic map of documents (embeddings + clustering)
- [Entity Relationships](owentityrelations.md) — Analyse co-occurrence relationships between two entity types:
- [K-Fields Plot](owkfieldsplot.md) — Visualize relationships between K bibliometric fields using Sankey diagram

## Geography, SDG & disciplines

- [Geographic Analysis](owgeographic.md) — Per-country metrics (documents, citations, collaboration) with map coordinates
- [Discipline Analysis](owdiscipline.md) — Profile a corpus across OpenAlex domains / fields / subfields
- [SDG Identifier](owsdgidentifier.md) — Identify Sustainable Development Goals in your dataset
- [SDG Networks](owsdgnetworks.md) — Co-occurrence network of Sustainable Development Goals + bridge papers
- [SDG Drift](owsdgdrift.md) — How the vocabulary of each SDG drifts across time windows

## Groups, comparison & inference

- [Group Counts](owgroupcounts.md) — Count and compare entity frequencies across document groups
- [Group Counts Plot](owgroupcountsplot.md) — Interactive horizontal bar chart comparing entity
- [Group Statistics](owgroupstatistics.md) — Compute performance statistics (counts, fractions, ranks,
- [Group Intersections](owgroupintersections.md) — Analyze document overlap between groups
- [Group Associations](owgroupassociations.md) — Analyse entity–group relationships: contingency, diversity, CA,
- [Crosstabs](owcrosstabs.md) — Contingency table with chi-squared test and effect sizes
- [Compare Means](owcomparemeans.md) — Compare a numeric variable across groups (t-test/ANOVA + post-hoc)
- [Comparative Analysis](owcomparative.md) — Compare groups (by a column) across bibliometric metrics
- [Benchmarking](owbenchmarking.md) — Compare dataset distributions against global research patterns
- [Permutation Inference](owpermutationtest.md) — Permutation tests of group×entity association — valid for
- [Logistic Regression](owlogisticregression.md) — Binary logistic regression (coefficients, p-values, odds ratios)
- [Factorial Analysis](owfactorialanalysis.md) — MCA, CA, PCA with clustering and visualization
- [Document Clustering](owdocclustering.md) — Cluster documents by text (and optionally shared references)

## Niche, external & output

- [Altmetrics Analysis](owaltmetrics.md) — Analyze alternative impact metrics beyond citations
- [Open Science](owopenscience.md) — Open access, data/code availability, preprints, preregistration
- [Integrity Audit](owintegrityaudit.md) — Screen for integrity red flags (tortured phrases, retractions, anomalies)
- [AI Describe](owaidescribe.md) — Generate a natural-language description of a table with an LLM
- [Literature Matrix](owliteraturematrix.md) — Extract structured fields from each paper into a review matrix (LLM)
- [Report Generator](owreportgenerator.md) — Generate Word/Excel/PowerPoint/LaTeX bibliometric reports
- [HTML Dashboard](owdashboard.md) — Generate a self-contained interactive HTML dashboard