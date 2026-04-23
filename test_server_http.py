#!/usr/bin/env python3
"""HTTP integration test for the Stan MCP Server.

Requires the server to already be running:
    stan-mcp-server --datasets-dir datasets --results-dir results

Usage:
    python test_server_http.py
    python test_server_http.py --url http://remote-host:8765/mcp
    python test_server_http.py --url http://127.0.0.1:8765/mcp --token <bearer-token>
"""

import argparse
import asyncio
import sys

import numpy as np
from fastmcp.client import Client

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


async def run(url: str, token: str | None) -> None:
    auth = token or None
    try:
        async with Client(url, auth=auth) as client:
            await _run_checks(client)
    except Exception as exc:
        print(f"\nCould not connect to {url}: {exc}")
        print("Is the server running?  Start it with:")
        print("  stan-mcp-server --datasets-dir datasets --results-dir results")
        sys.exit(1)


async def _run_checks(client: Client) -> None:
    # ── 0. get_capabilities ────────────────────────────────────────────────────
    print("\n[0] get_capabilities …")
    r = (await client.call_tool("get_capabilities", {})).data
    assert "tools" in r, f"get_capabilities failed: {r}"
    assert "fit_and_evaluate" in r["tools"], "fit_and_evaluate missing from capabilities"
    print(f"    tools={r['tools']}")
    print(f"    datasets_dir={r['datasets_dir']}  results_dir={r['results_dir']}")

    # ── 1. list_datasets ───────────────────────────────────────────────────────
    print("\n[1] list_datasets …")
    r = (await client.call_tool("list_datasets", {})).data
    assert "datasets" in r, f"list_datasets failed: {r}"
    print(f"    datasets={r['datasets']}  uploaded={r['uploaded']}")

    # ── 2. check_model ─────────────────────────────────────────────────────────
    print("\n[2] check_model …")
    r = (await client.call_tool("check_model", {"stan_code": SIMPLE_LINEAR_MODEL})).data
    assert r["status"] == "ok", f"check_model failed: {r}"
    print(f"    log_lik_length_expr = {r['log_lik_length_expr']}")

    # ── 3. get_data_summary ────────────────────────────────────────────────────
    print("\n[3] get_data_summary(benchmarks/regression_1d) …")
    r = (await client.call_tool("get_data_summary", {"dataset": "benchmarks/regression_1d"})).data
    assert r.get("status") != "error", f"get_data_summary failed: {r}"
    assert r["tier"] == "staged", f"expected tier=staged, got {r.get('tier')}"
    assert r["has_test"] is True, "benchmarks/regression_1d should have a test set"
    print(f"    tier={r['tier']}  n_train={r['n_train']}  n_test={r['n_test']}  columns={list(r['columns'].keys())}")

    # ── 4. fit_and_evaluate via dataset name (no inline data) ──────────────────
    print("\n[4] fit_and_evaluate(dataset='benchmarks/regression_1d', 2×500 draws) …")
    r = (await client.call_tool("fit_and_evaluate", {
        "stan_code": SIMPLE_LINEAR_MODEL,
        "dataset": "benchmarks/regression_1d",
        "config": {"chains": 2, "iter_warmup": 500, "iter_sampling": 500, "seed": 42},
    })).data
    assert r["status"] == "ok", f"fit_and_evaluate failed: {r}"
    nlpd   = r["nlpd"]
    n_divs = r["n_divergences"]
    r_hat  = r["r_hat_max"]
    print(f"    nlpd={nlpd}  divs={n_divs}  r_hat={r_hat}  runtime={r['runtime_sec']}s")
    assert np.isfinite(nlpd),  "NLPD must be finite"
    assert n_divs == 0,        f"expected 0 divergences, got {n_divs}"
    assert r_hat < 1.1,        f"r_hat_max={r_hat} suspiciously high"

    # ── 5. get_run_history ─────────────────────────────────────────────────────
    print("\n[5] get_run_history(benchmarks/regression_1d) …")
    r = (await client.call_tool("get_run_history", {"dataset": "benchmarks/regression_1d"})).data
    assert "entries" in r, f"get_run_history failed: {r}"
    assert r["n_entries"] >= 1, f"expected at least 1 entry, got {r['n_entries']}"
    print(f"    n_entries={r['n_entries']}  best_nlpd={r['best_nlpd']}")

    print("\nAll assertions passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",   default="http://127.0.0.1:8765/mcp",
                        help="MCP server URL (default: http://127.0.0.1:8765/mcp)")
    parser.add_argument("--token", default=None,
                        help="Bearer token if the server was started with --token")
    args = parser.parse_args()

    print(f"Connecting to {args.url}")
    asyncio.run(run(args.url, args.token))


if __name__ == "__main__":
    main()
