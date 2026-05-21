#!/usr/bin/env python3
"""Generate synthetic sparse hierarchical dataset (Neal's funnel geometry).

Generative process:
    tau ~ HalfCauchy(0, 2)         # Wide hyperprior -> severe funnel geometry
    mu_j ~ Normal(0, tau)          for j = 1..J
    y_ij ~ Normal(mu_j, sigma)     for i = 1..N per group

True parameters (hidden from agent):
    tau_true = drawn from HalfCauchy(0, 2) with SEED
    sigma = 0.5
    J = 50 groups
    N_train_per_group = 3   (very sparse -> posterior funnel)
    N_test_per_group  = 50  (large, stable NLPD oracle)

Note: centered parameterization will produce ~300-600 divergences and
badly inflated NLPD. NCP is essential to fix sampling.
"""

import csv
from pathlib import Path

import numpy as np
from scipy import stats

SEED = 20260428
J = 50
N_TRAIN_PER_GROUP = 3
N_TEST_PER_GROUP = 50
SIGMA = 0.5
TAU_SCALE = 2.0

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

rng = np.random.default_rng(SEED)

# Draw tau from HalfCauchy(0, TAU_SCALE)
tau_true = float(np.abs(rng.standard_cauchy()) * TAU_SCALE)
# Clip to funnel-inducing range: large enough to spread groups, small enough
# that the joint (tau, mu_j) posterior has genuine funnel geometry
tau_true = float(np.clip(tau_true, 0.8, 3.0))

# Draw group means
mu = rng.normal(0.0, tau_true, size=J)

# Generate train observations (3 per group)
train_rows = []
for j in range(J):
    for _ in range(N_TRAIN_PER_GROUP):
        train_rows.append({
            "group_id": j + 1,
            "value": round(float(rng.normal(mu[j], SIGMA)), 4),
        })

# Generate test observations (50 per group, fresh draw from same DGP)
test_rows = []
for j in range(J):
    for _ in range(N_TEST_PER_GROUP):
        test_rows.append({
            "group_id": j + 1,
            "value": round(float(rng.normal(mu[j], SIGMA)), 4),
        })


def write_csv(path, data, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


fields = ["group_id", "value"]
write_csv(DATASET_DIR / "train.csv", train_rows, fields)
write_csv(SCRIPT_DIR / "test.csv", test_rows, fields)

# Oracle NLPD: predictive distribution given true tau, sigma, and 3 training obs
# For each group j: posterior on mu_j given 3 obs is N(mu_post_j, var_post)
# var_post = 1 / (1/tau^2 + N_train/sigma^2)
# mu_post_j = var_post * (sum_train_j / sigma^2)   [prior mean = 0]
# Predictive: N(mu_post_j, sigma^2 + var_post)
var_post = 1.0 / (1.0 / tau_true**2 + N_TRAIN_PER_GROUP / SIGMA**2)

train_arr = np.array([[r["value"] for r in train_rows if r["group_id"] == j + 1] for j in range(J)])
test_arr  = np.array([[r["value"] for r in test_rows  if r["group_id"] == j + 1] for j in range(J)])

oracle_nlpd_vals = []
for j in range(J):
    mu_post_j = var_post * np.sum(train_arr[j]) / SIGMA**2
    pred_var = SIGMA**2 + var_post
    lp = stats.norm.logpdf(test_arr[j], loc=mu_post_j, scale=np.sqrt(pred_var))
    oracle_nlpd_vals.extend((-lp).tolist())
oracle_nlpd = float(np.mean(oracle_nlpd_vals))

print(f"Generated {len(train_rows)} train, {len(test_rows)} test observations.")
print(f"Groups: {J}, Train per group: {N_TRAIN_PER_GROUP}, Test per group: {N_TEST_PER_GROUP}")
print(f"True tau: {tau_true:.4f}, sigma: {SIGMA}")
print(f"Group means (first 10): {np.round(mu[:10], 3).tolist()}")
print(f"Oracle NLPD (known tau + sigma, posterior mu_j): {oracle_nlpd:.4f}")
