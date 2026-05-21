# Dataset: Synthetic Continuous Units

## Overview

Observations from multiple units, each with a continuous predictor and continuous response.
The goal is to predict `response_b` for held-out test observations.

## Data Format

`train.csv` columns:
- `unit`: integer, unit identifier (1..J)
- `feature`: continuous predictor
- `response`: continuous response (target)

600 training observations.
1200 test observations (held out, not visible to the agent).

## Data Interface

The evaluation script passes the following to Stan:

```stan
data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  int<lower=0> J;
  array[N_train] int<lower=1,upper=J> unit_train;
  array[N_test] int<lower=1,upper=J> unit_test;
  vector[N_train] feature_train;
  vector[N_test] feature_test;
  vector[N_train] response_train;
  vector[N_test] response_test;
}
```

## Evaluation

Your model must output a `log_lik` vector of length `N_test` in generated quantities.
