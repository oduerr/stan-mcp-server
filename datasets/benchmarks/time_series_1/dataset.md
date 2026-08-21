# Dataset: time_series_1 — a univariate series with gaps

## Overview

A single univariate time series of 700 equally-spaced observations. Blocks of
consecutive observations have been held out; the task is to predict the values
at those held-out times.

## Data Format

`train.csv` — 500 rows:

- `t` — integer time index, 1–700, **with gaps** where test blocks were removed
- `y` — **response**: the observed value at time `t`

Test set: 200 rows, same two columns. The held-out times fall in 40 blocks of 5
consecutive steps, so every test point lies within a few steps of observed data.

response_col: y

## Data Interface

The evaluation harness passes the following to Stan:

```stan
int<lower=0> N_train;
int<lower=0> N_test;
vector[N_train] t_train;
vector[N_train] y_train;
vector[N_test] t_test;
vector[N_test] y_test;
```

`t_train` and `t_test` carry the raw integer times, so the spacing between any
two observations — and the distance from a test point to the nearest observed
one — is recoverable from the data.

## Evaluation

`fit_and_evaluate` scores the negative log predictive density of `y_test`. Your
model must declare `vector[N_test] log_lik` in generated quantities, where
`log_lik[n]` is the log predictive density of `y_test[n]`.

## Notes

`y` is continuous and real-valued. Note that the observations are ordered in
time and that `t` is not contiguous in the training set.
