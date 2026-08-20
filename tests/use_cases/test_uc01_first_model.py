#!/usr/bin/env python3
"""UC-1 — first model on my own data (docs/USE_CASES.md).

Scripted agent, no LLM: upload a train-only dataset → inspect it → fit a
model by dataset NAME via `sample` (G1) — the data never passes through the
"context" (here: the tool arguments), which is the point of the use case.

Needs CmdStan; skipped automatically where it is missing (e.g. CI).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

import stan_mcp_server.server as srv

try:
    import cmdstanpy
    cmdstanpy.cmdstan_path()
    HAS_CMDSTAN = True
except Exception:
    HAS_CMDSTAN = False

pytestmark = pytest.mark.skipif(not HAS_CMDSTAN, reason="CmdStan not installed")

DATASET_MD = """# My experiment

## Data Interface

```stan
int<lower=0> N_train;
vector[N_train] x_train;
vector[N_train] y_train;
```
response_col: y
"""

TRAIN_CSV = "x,y\n" + "\n".join(
    f"{i * 0.1:.1f},{1.0 + 2.0 * i * 0.1 + 0.05 * ((i * 7) % 5 - 2):.3f}"
    for i in range(40)
) + "\n"

MODEL = """
data {
  int<lower=0> N_train;
  vector[N_train] x_train;
  vector[N_train] y_train;
}
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 5);
  sigma ~ exponential(1);
  y_train ~ normal(alpha + beta * x_train, sigma);
}
"""


def test_uc01_first_model_on_own_data(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(srv, "_RESULTS_DIR", tmp_path / "results")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "results").mkdir()

    # 1. Upload (the agent does this via the HTTP endpoint; same code path).
    up = srv._save_dataset("myexp", TRAIN_CSV, DATASET_MD)
    assert up["status"] == "ok" and up["tier"] == "uploaded"
    name = up["dataset"]                       # "_uploaded/myexp"

    # 2. Inspect.
    summary = srv.get_data_summary(name)
    assert summary["tier"] == "uploaded" and summary["n_train"] == 40

    # 3. Fit BY NAME — no data dict through the tool arguments.
    r = srv.sample(
        stan_code=MODEL,
        dataset=name,
        config={"chains": 2, "iter_warmup": 300, "iter_sampling": 300},
    )
    assert r["status"] == "ok", r
    assert r["diagnostics"]["n_divergences"] == 0
    assert r["diagnostics"]["r_hat_max"] < 1.05
    assert set(r["data_keys_loaded"]) >= {"N_train", "x_train", "y_train"}
    # The draws are on disk, not in the response.
    assert "draws" not in r and Path(r["samples_path"]).is_dir()

    # 4. `data` extends/overrides the loaded dict (documented merge contract).
    r2 = srv.sample(
        stan_code=MODEL.replace(
            "model {", "model {\n  // prior_scale unused by likelihood"
        ).replace("data {", "data {\n  real prior_scale;"),
        dataset=name,
        data={"prior_scale": 2.5},
        config={"chains": 1, "iter_warmup": 200, "iter_sampling": 200},
    )
    assert r2["status"] == "ok", r2
    assert "prior_scale" in r2["data_keys_loaded"]
