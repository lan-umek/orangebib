# Setup Groups

> Define document groups for comparative analysis.

```{figure} ../_static/img/owsetupgroups.png
:alt: Setup Groups
:class: widget-screenshot

The Setup Groups widget.
```

## Overview

Defines **groups of documents** that the comparative widgets (Group Counts,
Group Statistics, Group Associations, Compare Means, Benchmarking …) then use.
Groups can be built from the values of a categorical column, from keyword /
regex rules, from concept definitions, or from time periods. It emits the data
annotated with `Group:` columns plus a document × group binary matrix.

## Inputs

- **Data** (`Table`) — bibliographic data.
- **Concepts** (`Table`) — optional concept definitions to group by.

## Outputs

- **Data** (`Table`) — the input with added `Group:` columns.
- **Group Matrix** (`Table`) — document × group binary membership matrix.
- **Group Summary** (`Table`) — per-group summary statistics.

## Controls

- **Grouping Method** — how groups are formed (by column values, keyword/regex rules, concepts, or periods).
- **Method Options** — parameters specific to the chosen method (e.g. the column, the rules, period boundaries).
- **Include** / **Exclude** — restrict which entities/values become groups.
- **Group colours** — assign stable colours used by downstream plots.
- **Group Configuration** — review and fine-tune the resulting groups.

**Actions:** `Preview Groups` (see the result before committing), `Create Groups` (emit outputs), `Browse…` (load concept/definition files).

## Tips

- A document may belong to several groups; the binary **Group Matrix** captures
  this overlap for association and intersection analyses.
