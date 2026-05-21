# Dataset: Spatial Count Data

## Overview

Non-negative integer counts at locations on a **12×12 spatial grid**.
The goal is to predict `count` for held-out grid cells given the spatial structure observed in training.

## Data Format

**`train.csv`** columns:
- `row` — grid row index (0–11)
- `col` — grid column index (0–11)
- `count` — non-negative integer count (target)

108 training observations. 36 test observations (held out, not visible to the agent).

The full grid is 12×12 = 144 locations. Training and test locations are randomly sampled —
test cells are scattered among training cells, so spatial neighbours of test cells are available
in the training data.

## Data Interface

The evaluation script passes the following to Stan:

```stan
int<lower=0> N_train;
int<lower=0> N_test;
array[N_train] int<lower=0> row_train;   // 0-indexed row
array[N_train] int<lower=0> col_train;   // 0-indexed col
array[N_test]  int<lower=0> row_test;
array[N_test]  int<lower=0> col_test;
array[N_train] int<lower=0> count_train;
array[N_test]  int<lower=0> count_test;
```

Additionally, the **grid adjacency structure** is provided (standard 4-connectivity, no periodic BC):

```stan
int<lower=0> N_grid;     // total grid cells = 144
int<lower=0> N_edges;    // number of edges = 264
array[N_edges] int node1;  // 1-indexed
array[N_edges] int node2;  // 1-indexed
array[N_train] int<lower=1,upper=N_grid> train_grid_idx; // 1-indexed cell for each train obs
array[N_test]  int<lower=1,upper=N_grid> test_grid_idx;
```

## Evaluation

```bash
python datasets/spatial_1/protected/evaluate.py --notes "..." --rationale "..."
```

Your model must output a `log_lik` vector of length `N_test` in the `generated quantities` block.

> ⚠️ **Computational note:** Dense N×N covariance matrices (N=144) are feasible but slow.
> The provided sparse adjacency structure enables efficient iCAR models.
