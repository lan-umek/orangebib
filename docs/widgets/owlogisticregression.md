# Logistic Regression

> Explanatory binary logistic regression with odds ratios (and a Firth option).

```{figure} ../_static/img/owlogisticregression.png
:alt: Logistic Regression
:class: widget-screenshot

The Logistic Regression widget.
```

## Overview

Fits an **explanatory** binary logistic regression (statsmodels): predict
membership in a group / binary outcome from numeric predictors, reporting
coefficients, standard errors, z, p-values, **odds ratios** with confidence
intervals and the direction of each effect. It can build binary predictors from
text/keyword columns, fit several targets at once (one tab each plus a Summary
pivot), and use **Firth** penalisation to handle separation.

## Inputs
- **Data** (`Table`) — data with a binary target and numeric predictors.

## Outputs
- **Coefficients** (`Table`) — model coefficients and statistics.
- **Predictions** (`Table`) — data with predicted probability/class.

## Controls
- **Group / target** + **Positive class** — the binary outcome and which class counts as 1.
- **Additional targets** — pick extra binary targets (each gets its own results tab + a Summary tab).
- **Predictors (numeric)** — tick the numeric predictors (`All` / `None`).
- **Build predictors from text/keywords** — turn the top terms of a text field into binary predictors: **Column** (Title/Abstract/Keywords/combination/References), **Top N terms**, **Min occurrences**, **Remove stopwords**.
- **Include intercept**, **Standardize predictors (z-score)**.
- **Firth penalized** — penalised MLE that fixes quasi/complete separation.
- **Summary cells** — what the multi-target Summary pivot shows (coefficient / p-value / both).

**Actions:** `Build binary predictors`, `Fit model`.
