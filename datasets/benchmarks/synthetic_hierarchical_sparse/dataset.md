# Dataset: Synthetic Groups

We measured a continuous outcome for items belonging to groups. Multiple items per group share a common group-level tendency.

## Data Format
`train.csv` columns:
- `group_id`: integer, group identifier (1–50)
- `value`: float, observed outcome

There are 150 training observations (3 per group). A held-out test set of 2500 observations (50 per group) is used for evaluation.

## Data Interface
Your `model.stan` must declare this data block:
```stan
data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  int<lower=0> J;
  array[N_train] int<lower=1,upper=J> group_train;
  array[N_test] int<lower=1,upper=J> group_test;
  vector[N_train] value_train;
  vector[N_test] value_test;
}
```

## Goal
Minimize NLPD on the held-out test set by improving `model.stan`.
Lower NLPD = better model.

## log_lik Contract
Your `model.stan` must output a vector `log_lik` of length `N_test` in the `generated quantities` block, containing the log-likelihood of each test observation under the model.
