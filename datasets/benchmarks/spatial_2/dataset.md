# Dataset: Counts at Irregular Locations

## Overview

Non-negative integer counts observed at 150 locations in a rectangular study
region. The goal is to predict `count` for held-out locations.

## Data Format

**`train.csv`** columns:
- `x_coord` — horizontal coordinate (continuous, 0–12)
- `y_coord` — vertical coordinate (continuous, 0–12)
- `count` — non-negative integer count (target)

150 training observations. 450 test observations (held out, not visible to the
agent). Test locations lie in the same region as the training locations.

## Data Interface

The evaluation script passes the following to Stan:

```stan
int<lower=0> N_train;
int<lower=0> N_test;
vector[N_train] x_coord_train;
vector[N_test]  x_coord_test;
vector[N_train] y_coord_train;
vector[N_test]  y_coord_test;
array[N_train] int<lower=0> count_train;
array[N_test]  int<lower=0> count_test;
```

## Evaluation

Your model must output a `log_lik` vector of length `N_test` in the
`generated quantities` block: the log predictive density of each held-out
count.
