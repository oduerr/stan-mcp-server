#!/usr/bin/env python3
"""Regression test: the shadow NLPD must never reach the model.

The shadow set measures test-set selection bias by staying invisible to the
agent.  If `shadow_nlpd` ever reaches the model — under any key, through any
tool — the shadow set silently becomes a second feedback set and every number
it has produced since becomes meaningless.  Nothing about that failure is
visible from the outside: the runs keep working, the logs keep filling, and the
measurement is quietly worthless.

That is why this is a test and not a code comment.  It failed on first run:
`get_run_history` returned whole log entries verbatim, shadow_nlpd included.

The checks scan for the shadow *value* as well as shadow-named keys, so
renaming the key to hide it (`held_out_2_nlpd`) does not get past this file.

Usage:
    python test_shadow_isolation.py                 # fast paths, no CmdStan needed
    python test_shadow_isolation.py --with-fit      # + real end-to-end fit
    pytest test_shadow_isolation.py
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import stan_mcp_server.server as srv

# Distinctive enough that finding it anywhere in a tool response is proof of a
# leak, and never a rounding artefact of some unrelated float.
SENTINEL = -7.654321
DATASET = "leak_probe"


# ── helpers ────────────────────────────────────────────────────────────────────

def find_shadow_keys(obj, path="") -> list[str]:
    """Return paths of every shadow-named key anywhere in a structure."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if "shadow" in str(k).lower():
                hits.append(here)
            hits += find_shadow_keys(v, here)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_shadow_keys(v, f"{path}[{i}]")
    return hits


def find_value(obj, needle, path="") -> list[str]:
    """Return paths where `needle` appears as a value — under ANY key name."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits += find_value(v, needle, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_value(v, needle, f"{path}[{i}]")
    elif isinstance(obj, float) and abs(obj - needle) < 1e-9:
        hits.append(path)
    elif isinstance(obj, str) and str(needle) in obj:
        hits.append(path)
    return hits


def assert_clean(payload, what: str) -> None:
    """Assert a model-bound payload carries no shadow key and no shadow value."""
    keys = find_shadow_keys(payload)
    assert not keys, f"{what} leaks shadow-named key(s): {keys}"
    vals = find_value(payload, SENTINEL)
    assert not vals, f"{what} leaks the shadow VALUE at: {vals} (renamed key?)"


# ── 1. the scrubber itself ─────────────────────────────────────────────────────

def test_scrub_drops_shadow_keys_at_every_depth():
    dirty = {
        "nlpd": 1.0,
        "shadow_nlpd": SENTINEL,
        "SHADOW_NLPD": SENTINEL,          # case
        "shadow_ess": SENTINEL,           # a key that does not exist yet
        "entries": [{"nlpd": 2.0, "shadow_nlpd": SENTINEL}],   # nested in a list
        "stats": {"inner": {"shadow": {"nlpd": SENTINEL}}},    # nested block
    }
    clean = srv._scrub_shadow(dirty)

    assert_clean(clean, "_scrub_shadow output")
    # …while leaving the feedback numbers alone.
    assert clean["nlpd"] == 1.0
    assert clean["entries"][0]["nlpd"] == 2.0
    assert srv._scrub_shadow(dirty) == clean, "scrub must not mutate its input"
    assert "shadow_nlpd" in dirty, "scrub must not mutate its input"


# ── 2. get_run_history — the path that actually leaked ─────────────────────────

def test_get_run_history_hides_shadow_nlpd(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    srv._RESULTS_DIR = tmp

    # Exactly what fit_and_evaluate writes.
    srv._append_log(DATASET, {
        "iter": 0, "run_id": "probe", "nlpd": 1.234,
        "shadow_nlpd": SENTINEL,
        "diagnostics_valid": True, "improved": None,
    })

    # The value must be on disk — the measurement depends on it being recorded.
    logged = json.loads((tmp / DATASET / "log.jsonl").read_text().strip())
    assert logged["shadow_nlpd"] == SENTINEL, "shadow_nlpd must still reach the log"

    # …and must not come back out.
    history = srv.get_run_history(DATASET)
    assert_clean(history, "get_run_history")

    # The tool still has to be useful: feedback NLPD survives.
    assert history["n_entries"] == 1
    assert history["best_nlpd"] == 1.234
    assert history["entries"][0]["nlpd"] == 1.234


# ── 3. the cheap tool surface ──────────────────────────────────────────────────

def test_no_shadow_in_other_tool_responses(tmp_path=None):
    srv._RESULTS_DIR = Path(tmp_path or tempfile.mkdtemp())
    for name, payload in [
        ("get_capabilities", srv.get_capabilities()),
        ("list_datasets", srv.list_datasets()),
        ("get_run_history(empty)", srv.get_run_history("no_such_dataset")),
    ]:
        assert_clean(payload, name)


# ── 4. end-to-end: a real fit against a real shadow.csv (opt-in) ───────────────
#
# Tiny synthetic dataset rather than spatial_2: this exercises the identical
# code path in seconds instead of ~75 s, so it can run on every change.

MODEL = """
data {
    int<lower=0> N_train;
    int<lower=0> N_test;
    vector[N_train] predictor_train;
    vector[N_test]  predictor_test;
    vector[N_train] response_train;
    vector[N_test]  response_test;
}
parameters { real alpha; real beta; real<lower=0> sigma; }
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

DATASET_MD = """# Dataset: leak probe

## Data Interface

```stan
int<lower=0> N_train;
int<lower=0> N_test;
vector[N_train] predictor_train;
vector[N_test] predictor_test;
vector[N_train] response_train;
vector[N_test] response_test;
```
"""


def _write_probe_dataset(root: Path) -> None:
    """train / test / shadow from y = 2 + 3x + eps, deterministic (no seed needed)."""
    ds = root / DATASET
    (ds / "protected").mkdir(parents=True, exist_ok=True)
    (ds / "dataset.md").write_text(DATASET_MD)

    def rows(n, offset):
        out = ["predictor,response"]
        for i in range(n):
            x = (i + offset) * 0.1
            eps = 0.3 * ((i * 7 % 11) / 11 - 0.5)   # reproducible pseudo-noise
            out.append(f"{x:.4f},{2 + 3 * x + eps:.4f}")
        return "\n".join(out) + "\n"

    (ds / "train.csv").write_text(rows(60, 0))
    (ds / "protected" / "test.csv").write_text(rows(20, 60))
    (ds / "protected" / "shadow.csv").write_text(rows(20, 80))


def test_fit_and_evaluate_runs_shadow_but_does_not_return_it(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    datasets, results = tmp / "datasets", tmp / "results"
    results.mkdir(parents=True, exist_ok=True)
    _write_probe_dataset(datasets)
    srv._DATASETS_DIR, srv._RESULTS_DIR = datasets, results

    result = srv.fit_and_evaluate(
        stan_code=MODEL,
        dataset=DATASET,
        config={"chains": 2, "iter_warmup": 300, "iter_sampling": 300, "seed": 42},
    )
    assert result["status"] == "ok", f"fit failed: {result}"

    # The shadow pass must have actually run — otherwise this test would pass
    # trivially on a build where the whole feature is broken.
    entry = json.loads((results / DATASET / "log.jsonl").read_text().strip().splitlines()[-1])
    assert entry.get("shadow_nlpd") is not None, (
        "shadow pass did not run or failed silently — the leak check below proves nothing"
    )

    # The real assertion: neither the key nor the computed value comes back.
    assert_clean(result, "fit_and_evaluate result")
    leaked = find_value(result, entry["shadow_nlpd"])
    assert not leaked, f"fit_and_evaluate returns the shadow value at: {leaked}"

    # And it stays hidden on the way back out through history.
    assert_clean(srv.get_run_history(DATASET), "get_run_history after real fit")


# ── standalone runner (mirrors test_server.py's style) ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-fit", action="store_true",
                    help="also run the end-to-end fit (needs CmdStan, ~10 s)")
    args = ap.parse_args()

    fast = [
        test_scrub_drops_shadow_keys_at_every_depth,
        test_get_run_history_hides_shadow_nlpd,
        test_no_shadow_in_other_tool_responses,
    ]
    for i, fn in enumerate(fast):
        print(f"[{i}] {fn.__name__} …")
        fn()
        print("    ok")

    if args.with_fit:
        print(f"[{len(fast)}] test_fit_and_evaluate_runs_shadow_but_does_not_return_it …")
        test_fit_and_evaluate_runs_shadow_but_does_not_return_it()
        print("    ok")
    else:
        print("[skip] end-to-end fit — pass --with-fit to include it")

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
