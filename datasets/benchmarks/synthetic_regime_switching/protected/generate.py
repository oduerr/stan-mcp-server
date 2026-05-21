#!/usr/bin/env python3
"""Generate a hierarchical regime-switching heteroscedastic regression dataset.

Hidden DGP (not revealed to the agent):
- Unit-level random intercept/slope and regime gate location/sharpness.
- Two latent regimes with different mean and variance functions.
- Mild contamination outliers to reward robust likelihoods.
"""

import csv
import json
from pathlib import Path

import numpy as np

SEED = 20260429
J = 24
N_TRAIN_PER_UNIT = 25
N_TEST_PER_UNIT = 50

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

rng = np.random.default_rng(SEED)

# Unit-level parameters
alpha = rng.normal(0.0, 1.0, size=J)
beta = rng.normal(0.35, 0.25, size=J)
phase = rng.normal(0.0, 0.35, size=J)
gate_center = rng.normal(0.0, 0.9, size=J)
gate_sharp = np.clip(rng.normal(1.1, 0.2, size=J), 0.5, None)


# Mean/variance functions by latent regime
def mu_regime1(j, x):
    return alpha[j] + beta[j] * x + 0.65 * np.sin(1.4 * x + phase[j])


def mu_regime2(j, x):
    return alpha[j] - 0.45 + (beta[j] + 0.85) * x - 0.55 * np.sin(0.75 * x - phase[j])


def sigma_regime1(x):
    return np.exp(-0.78 + 0.22 * np.abs(x) + 0.10 * (x > 0))


def sigma_regime2(x):
    return np.exp(-0.42 + 0.34 * np.abs(x))


def gate_prob(j, x):
    z = gate_sharp[j] * (x - gate_center[j])
    return 1.0 / (1.0 + np.exp(-z))


def sample_rows(n_per_unit):
    rows = []
    for j in range(J):
        xs = rng.uniform(-3.5, 3.5, size=n_per_unit)
        for x in xs:
            p2 = gate_prob(j, x)
            use_regime2 = rng.random() < p2
            if use_regime2:
                mu = mu_regime2(j, x)
                sig = sigma_regime2(x)
            else:
                mu = mu_regime1(j, x)
                sig = sigma_regime1(x)

            y = rng.normal(mu, sig)

            # 3% contamination outliers
            if rng.random() < 0.03:
                y = rng.normal(mu, 6.0 * sig)

            rows.append(
                {
                    "unit": j + 1,
                    "feature": round(float(x), 5),
                    "response": round(float(y), 5),
                }
            )
    rng.shuffle(rows)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["unit", "feature", "response"])
        writer.writeheader()
        writer.writerows(rows)


train_rows = sample_rows(N_TRAIN_PER_UNIT)
test_rows = sample_rows(N_TEST_PER_UNIT)

write_csv(DATASET_DIR / "train.csv", train_rows)
write_csv(SCRIPT_DIR / "test.csv", test_rows)

# Store hidden DGP details for reproducibility (kept in protected/)
meta = {
    "seed": SEED,
    "J": J,
    "n_train_per_unit": N_TRAIN_PER_UNIT,
    "n_test_per_unit": N_TEST_PER_UNIT,
}
with open(SCRIPT_DIR / "ground_truth.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Generated {len(train_rows)} train rows and {len(test_rows)} test rows.")
print(f"Units: {J}, train/unit: {N_TRAIN_PER_UNIT}, test/unit: {N_TEST_PER_UNIT}")
