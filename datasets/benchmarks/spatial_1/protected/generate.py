#!/usr/bin/env python3
"""Generate two-hotspot spatially correlated count dataset on a 12×12 grid.

Generative process:
    Grid: 12×12 = 144 locations, indexed by (row, col) in [0,11]²
    Covariate: none (spatial effect dominates)
    Spatial random effect: u[i] = background_iCAR(sigma=0.2) + two Gaussian bumps
        Bump 1: centre (row=2, col=2), amplitude=3.0, radius=2.8
        Bump 2: centre (row=9, col=9), amplitude=3.0, radius=2.8
    log(mu[i]) = alpha + u[i]
    y[i] ~ Poisson(mu[i])

True parameters (hidden from agent):
    alpha  = 0.5
    sigma_bg = 0.2   (background iCAR texture)
    A      = 3.0     (hotspot amplitude)
    r_bump = 2.8     (hotspot Gaussian radius in grid cells)

Split: 108 train / 36 test (random spatial locations)

NLPD reference values (computed with Stan, 4 chains × 1000 samples):
    Oracle (true mu known):          1.49
    Spatial iCAR fitted (Stan):      1.65
    Naive Poisson (no spatial):      2.35
    Gap (naive → spatial):           0.70 nats
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import poisson

SEED = 20260519
ALPHA_TRUE = 0.5
A_BUMP = 3.0
R_BUMP = 2.8
SIGMA_BG = 0.2
RHO = 0.92
N_ITER = 40
G = 12
N = G * G
N_TRAIN = 108
N_TEST = 36

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent


def simulate_icar_field(rng, G, sigma_u, rho=RHO, n_iter=N_ITER):
    N = G * G
    def idx(i, j): return i * G + j
    u = rng.normal(0, 1, N)
    for _ in range(n_iter):
        u_new = u.copy()
        for i in range(G):
            for j in range(G):
                nbrs = [idx(i-1,j) if i>0 else None, idx(i+1,j) if i<G-1 else None,
                        idx(i,j-1) if j>0 else None, idx(i,j+1) if j<G-1 else None]
                nbrs = [n for n in nbrs if n is not None]
                u_new[idx(i,j)] = rho * np.mean(u[nbrs]) + (1-rho) * rng.normal(0, 1)
        u = u_new
    return (u - u.mean()) / u.std() * sigma_u


def build_adjacency(G):
    """Return 1-indexed edge lists for a G×G grid (no periodic BC)."""
    edges_i, edges_j = [], []
    for i in range(G):
        for j in range(G):
            cell = i * G + j + 1  # 1-indexed
            if j < G - 1: edges_i.append(cell); edges_j.append(i * G + j + 2)
            if i < G - 1: edges_i.append(cell); edges_j.append((i+1) * G + j + 1)
    return edges_i, edges_j


def main():
    rng = np.random.default_rng(SEED)

    rows = np.array([i // G for i in range(N)], dtype=float)
    cols = np.array([i %  G for i in range(N)], dtype=float)

    # Two Gaussian hotspots
    c1 = np.array([2.0, 2.0])
    c2 = np.array([9.0, 9.0])
    d1 = np.sqrt((rows - c1[0])**2 + (cols - c1[1])**2)
    d2 = np.sqrt((rows - c2[0])**2 + (cols - c2[1])**2)
    bumps = A_BUMP * np.exp(-d1**2 / (2*R_BUMP**2)) + A_BUMP * np.exp(-d2**2 / (2*R_BUMP**2))

    bg = simulate_icar_field(rng, G, SIGMA_BG)
    u = bg + bumps
    u -= u.mean()

    mu = np.exp(ALPHA_TRUE + u)
    y = rng.poisson(mu).astype(int)

    idx_all = rng.permutation(N)
    train_idx = idx_all[:N_TRAIN]
    test_idx  = idx_all[N_TRAIN:N_TRAIN + N_TEST]

    oracle_nlpd = float(-np.mean(poisson.logpmf(y[test_idx], mu[test_idx])))

    def write_csv(path, indices):
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["row", "col", "count"])
            for i in indices:
                writer.writerow([int(rows[i]), int(cols[i]), int(y[i])])

    write_csv(DATASET_DIR / "train.csv", train_idx)
    write_csv(SCRIPT_DIR / "test.csv", test_idx)

    edges_i, edges_j = build_adjacency(G)

    print(f"Generated {N_TRAIN} train + {N_TEST} test observations on {G}×{G} grid")
    print(f"y: min={y.min()}  max={y.max()}  mean={y.mean():.2f}  std={y.std():.2f}")
    print(f"Oracle NLPD (true mu): {oracle_nlpd:.4f}")
    print(f"Grid: {G}×{G}, N_edges={len(edges_i)}")

    with open(SCRIPT_DIR / "ground_truth.json", "w") as fh:
        json.dump({
            "oracle_nlpd": round(oracle_nlpd, 4),
            "alpha_true": ALPHA_TRUE,
            "sigma_bg_true": SIGMA_BG,
            "hotspot_amplitude": A_BUMP,
            "hotspot_radius": R_BUMP,
            "hotspot_centres": [[2.0, 2.0], [9.0, 9.0]],
            "mean_function": "exp(0.5 + u[i])",
            "likelihood": "Poisson(mu[i])",
            "spatial_prior": "iCAR background (sigma=0.2) + two Gaussian bumps",
            "grid_size": G,
            "n_edges": len(edges_i),
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
        }, fh, indent=2)

    with open(SCRIPT_DIR / "adjacency.json", "w") as fh:
        json.dump({"node1": edges_i, "node2": edges_j, "N_edges": len(edges_i)}, fh)


if __name__ == "__main__":
    main()
