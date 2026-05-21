#!/usr/bin/env python3
"""Generate overdispersed count regression dataset.

Generative process:
    log(mu(x)) = 0.5 + 0.3 * x
    y ~ NegativeBinomial(mu(x), phi=3)

True parameters (hidden from agent):
    Mean function: log-linear (exponential trend)
    Overdispersion: phi = 3 (moderate — Poisson clearly underestimates variance)
    Response: discrete non-negative counts

x density: uniform over [0, 10]
Training: 200 observations. Test: 100 observations.

The agent must discover:
1. Counts require a count likelihood (not Gaussian)
2. Overdispersion requires Negative Binomial (not Poisson)
3. Log-linear mean function (log link / Poisson regression structure)
"""

import csv
import json
from pathlib import Path

import numpy as np

SEED = 20260519

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

ALPHA_TRUE = 0.5
BETA_TRUE = 0.3
PHI_TRUE = 3.0
N_TRAIN = 200
N_TEST = 100


def true_mu(x):
    return np.exp(ALPHA_TRUE + BETA_TRUE * x)


def oracle_nlpd(x, y):
    from scipy.stats import nbinom
    mu = true_mu(x)
    p = PHI_TRUE / (PHI_TRUE + mu)
    log_probs = nbinom.logpmf(y, PHI_TRUE, p)
    return float(-np.mean(log_probs))


def main():
    rng = np.random.default_rng(SEED)

    x_train = rng.uniform(0, 10, N_TRAIN)
    mu_train = true_mu(x_train)
    p_train = PHI_TRUE / (PHI_TRUE + mu_train)
    y_train = rng.negative_binomial(PHI_TRUE, p_train).astype(int)

    x_test = rng.uniform(0, 10, N_TEST)
    mu_test = true_mu(x_test)
    p_test = PHI_TRUE / (PHI_TRUE + mu_test)
    y_test = rng.negative_binomial(PHI_TRUE, p_test).astype(int)

    x_train = np.round(x_train, 4)
    x_test = np.round(x_test, 4)

    def write_csv(path, x, y):
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["predictor", "response"])
            for xi, yi in zip(x, y):
                writer.writerow([xi, int(yi)])

    write_csv(DATASET_DIR / "train.csv", x_train, y_train)
    write_csv(SCRIPT_DIR / "test.csv", x_test, y_test)

    nlpd = oracle_nlpd(x_test, y_test)
    print(f"Generated {N_TRAIN} train + {N_TEST} test observations")
    print(f"y_train: min={y_train.min()} max={y_train.max()} mean={y_train.mean():.2f} std={y_train.std():.2f}")
    print(f"Oracle NLPD: {nlpd:.4f}")

    with open(SCRIPT_DIR / "ground_truth.json", "w") as fh:
        json.dump({
            "oracle_nlpd": round(nlpd, 4),
            "alpha_true": ALPHA_TRUE,
            "beta_true": BETA_TRUE,
            "phi_true": PHI_TRUE,
            "mean_function": "exp(0.5 + 0.3 * x)",
            "likelihood": "NegativeBinomial(mu, phi=3)",
            "n_train": N_TRAIN,
            "n_test": N_TEST,
        }, fh, indent=2)


if __name__ == "__main__":
    main()
