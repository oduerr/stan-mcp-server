#!/usr/bin/env python3
"""Evaluate model.stan on the Spatial Count dataset.

Handles compilation, sampling, NLPD computation, and all bookkeeping.
"""

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATASET_DIR.parent.parent
DATASET_NAME = DATASET_DIR.name

MODEL_PATH = PROJECT_ROOT / "model.stan"
MODELS_DIR = PROJECT_ROOT / "models" / DATASET_NAME
RESULTS_DIR = PROJECT_ROOT / "results" / DATASET_NAME
LOG_FILE = RESULTS_DIR / "log.jsonl"
TEST_CSV = SCRIPT_DIR / "test.csv"
TRAIN_CSV = DATASET_DIR / "train.csv"
ADJ_JSON = SCRIPT_DIR / "adjacency.json"
GT_JSON = SCRIPT_DIR / "ground_truth.json"

GRID_SIZE = 12
N_GRID = GRID_SIZE * GRID_SIZE


def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model.stan on Spatial Count dataset")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--rationale", type=str, default="")
    args = parser.parse_args()

    import cmdstanpy

    if not MODEL_PATH.exists():
        print("ERROR: model.stan not found at project root.")
        sys.exit(1)

    train_rows = load_csv(TRAIN_CSV)
    test_rows = load_csv(TEST_CSV)

    with open(ADJ_JSON) as f:
        adj = json.load(f)
    with open(GT_JSON) as f:
        gt = json.load(f)

    # Map (row, col) → 1-indexed grid cell
    def cell_idx(row, col):
        return int(row) * GRID_SIZE + int(col) + 1

    train_grid_idx = [cell_idx(r["row"], r["col"]) for r in train_rows]
    test_grid_idx  = [cell_idx(r["row"], r["col"]) for r in test_rows]

    stan_data = {
        "N_grid":  N_GRID,
        "N_edges": adj["N_edges"],
        "node1":   adj["node1"],
        "node2":   adj["node2"],
        "N_train": len(train_rows),
        "N_test":  len(test_rows),
        "row_train":  [int(r["row"]) for r in train_rows],
        "col_train":  [int(r["col"]) for r in train_rows],
        "row_test":   [int(r["row"]) for r in test_rows],
        "col_test":   [int(r["col"]) for r in test_rows],
        "count_train": [int(r["count"]) for r in train_rows],
        "count_test":  [int(r["count"]) for r in test_rows],
        "train_grid_idx": train_grid_idx,
        "test_grid_idx":  test_grid_idx,
    }
    # covariate column removed in this dataset version (12x12, no covariate)

    print("Compiling model.stan...")
    try:
        model = cmdstanpy.CmdStanModel(stan_file=str(MODEL_PATH))
    except Exception as e:
        print(f"COMPILATION ERROR:\n{e}")
        sys.exit(1)

    print("Sampling: 4 chains x 1000 iterations...")
    start_time = time.time()
    try:
        fit = model.sample(
            data=stan_data,
            chains=4,
            iter_sampling=1000,
            iter_warmup=1000,
            seed=42,
            show_console=False,
        )
    except Exception as e:
        print(f"SAMPLING ERROR:\n{e}")
        sys.exit(1)
    runtime_sec = round(time.time() - start_time, 1)

    try:
        log_lik = fit.stan_variable("log_lik")
    except Exception:
        print("ERROR: model.stan must output 'log_lik' in generated quantities block.")
        print("log_lik should have length N_test (one entry per test observation).")
        sys.exit(1)

    nlpd = -np.mean(logsumexp(log_lik, axis=0) - np.log(log_lik.shape[0]))

    diagnostics = fit.diagnose()
    summary = fit.summary()
    r_hat_max = float(summary["R_hat"].max()) if "R_hat" in summary.columns else float("nan")

    n_divergences = 0
    try:
        sampler_vars = fit.method_variables()
        if "divergent__" in sampler_vars:
            n_divergences = int(sampler_vars["divergent__"].sum())
    except Exception:
        pass

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            lines = [l for l in f if l.strip()]
        iteration = len(lines)
    else:
        iteration = 0

    best_nlpd = None
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if best_nlpd is None or entry["nlpd"] < best_nlpd:
                        best_nlpd = entry["nlpd"]

    improved = bool(nlpd < best_nlpd) if best_nlpd is not None else None

    iter_label = f"iter_{iteration:03d}"
    model_snapshot = MODELS_DIR / ("baseline.stan" if iteration == 0 else f"{iter_label}.stan")
    shutil.copy2(MODEL_PATH, model_snapshot)

    if improved or improved is None:
        shutil.copy2(MODEL_PATH, MODELS_DIR / "best.stan")

    iter_results_dir = RESULTS_DIR / iter_label
    iter_results_dir.mkdir(parents=True, exist_ok=True)
    with open(iter_results_dir / "stan.log", "w") as f:
        f.write(diagnostics)

    log_entry = {
        "iter": iteration,
        "nlpd": round(float(nlpd), 4),
        "improved": improved,
        "machine": platform.node(),
        "runtime_sec": runtime_sec,
        "n_divergences": n_divergences,
        "r_hat_max": round(float(r_hat_max), 4),
        "notes": "baseline" if iteration == 0 else args.notes,
        "rationale": "" if iteration == 0 else args.rationale,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    oracle_nlpd = gt.get("oracle_nlpd", float("nan"))
    print()
    print("=" * 50)
    print(f"DATASET: {DATASET_NAME}")
    print(f"ITERATION: {iteration}")
    print(f"NLPD: {nlpd:.4f}")
    if best_nlpd is not None:
        print(f"BEST NLPD: {best_nlpd:.4f}")
        print(f"IMPROVED: {improved}")
    print(f"ORACLE NLPD (true mu, lower bound): {oracle_nlpd:.4f}")
    print(f"DIVERGENCES: {n_divergences}")
    print(f"MAX R-HAT: {r_hat_max:.4f}")
    print(f"RUNTIME: {runtime_sec}s")
    print(f"MODEL SAVED: {model_snapshot}")
    print(f"STAN LOG: {iter_results_dir / 'stan.log'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
