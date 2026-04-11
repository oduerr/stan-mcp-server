#!/usr/bin/env python3
"""Smoke test for the Stan MCP Server (no HTTP — calls tools directly).

Requires a datasets directory with regression_1d (or any dataset that has
a single continuous predictor and a response column).

Usage:
    python test_server.py --datasets-dir /path/to/datasets --results-dir /tmp/stan_results
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent))

import stan_mcp_server.server as srv

SIMPLE_LINEAR_MODEL = """
data {
    int<lower=0> N_train;
    int<lower=0> N_test;
    vector[N_train] predictor_train;
    vector[N_test]  predictor_test;
    vector[N_train] response_train;
    vector[N_test]  response_test;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
model {
    alpha ~ normal(0, 10);
    beta  ~ normal(0, 10);
    sigma ~ exponential(1);
    response_train ~ normal(alpha + beta * predictor_train, sigma);
}
generated quantities {
    vector[N_test] log_lik;
    for (i in 1:N_test)
        log_lik[i] = normal_lpdf(response_test[i] | alpha + beta * predictor_test[i], sigma);
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--results-dir",  type=Path, required=True)
    args = parser.parse_args()

    # Inject paths into server module (mirrors what main() does at startup)
    srv._DATASETS_DIR = args.datasets_dir.resolve()
    srv._RESULTS_DIR  = args.results_dir.resolve()

    print(f"datasets : {srv._DATASETS_DIR}")
    print(f"results  : {srv._RESULTS_DIR}")

    # ── 1. check_model ─────────────────────────────────────────────────────────
    print("\n[1] check_model …")
    r = srv.check_model(SIMPLE_LINEAR_MODEL)
    assert r["status"] == "ok", f"check_model failed: {r}"
    print(f"    log_lik_length_expr = {r['log_lik_length_expr']}")

    # ── 2. get_data_summary ────────────────────────────────────────────────────
    print("\n[2] get_data_summary(regression_1d) …")
    r = srv.get_data_summary("regression_1d")
    assert r.get("status") != "error", f"get_data_summary failed: {r}"
    print(f"    n_train={r['n_train']}  n_test={r['n_test']}  columns={list(r['columns'].keys())}")

    # ── 3. fit_and_evaluate via dataset name (no inline data) ──────────────────
    print("\n[3] fit_and_evaluate(dataset='regression_1d', 2×500 draws) …")
    r = srv.fit_and_evaluate(
        stan_code=SIMPLE_LINEAR_MODEL,
        dataset="regression_1d",
        config={"chains": 2, "iter_warmup": 500, "iter_sampling": 500, "seed": 42},
    )
    assert r["status"] == "ok", f"fit_and_evaluate failed: {r}"
    nlpd   = r["nlpd"]
    n_divs = r["n_divergences"]
    r_hat  = r["r_hat_max"]
    print(f"    nlpd={nlpd}  divs={n_divs}  r_hat={r_hat}  runtime={r['runtime_sec']}s")
    assert np.isfinite(nlpd),  "NLPD must be finite"
    assert n_divs == 0,        f"expected 0 divergences, got {n_divs}"
    assert r_hat < 1.1,        f"r_hat_max={r_hat} suspiciously high"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
