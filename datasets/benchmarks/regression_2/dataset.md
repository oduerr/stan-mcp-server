# Dataset: Count Regression

## Overview

Observations of a continuous predictor and a discrete non-negative count response. The goal is to predict `response` for held-out test observations.

## Data Format

**`train.csv`** columns:
- `predictor` — continuous predictor variable (range: [0, 10])
- `response` — non-negative integer count (target)

200 training observations. 100 test observations (held out, not visible to the agent).

## Data Interface

The evaluation script passes the following to Stan:

```stan
int<lower=0> N_train;
int<lower=0> N_test;
vector[N_train] predictor_train;
vector[N_test] predictor_test;
array[N_train] int<lower=0> response_train;
array[N_test] int<lower=0> response_test;
```

## Evaluation

```bash
python datasets/regression_2/protected/evaluate.py --notes "..." --rationale "..."
```

Your model must output a `log_lik` vector of length `N_test` in the `generated quantities` block.
