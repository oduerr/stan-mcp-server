#!/usr/bin/env python3
"""UC-2 — prior predictive check (docs/USE_CASES.md).

Scripted agent, no LLM: upload data → sample from a priors-only model (no
likelihood statement) → run_python_code(run_id=…, dataset=…) plots simulated
draws against the observed data and returns the figure as an MCP image.

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

DATASET_MD = """# Reaction times

## Data Interface

```stan
int<lower=0> N_train;
vector[N_train] x_train;
vector[N_train] y_train;
```
response_col: y
"""

TRAIN_CSV = "x,y\n" + "\n".join(
    f"{i * 0.25:.2f},{5.0 + 1.5 * i * 0.25 + 0.4 * ((i * 3) % 7 - 3):.3f}"
    for i in range(30)
) + "\n"

# Priors only — the model block has no statement involving y_train, so
# sampling draws from the prior; y_rep is the prior predictive.
PRIOR_ONLY_MODEL = """
data {
  int<lower=0> N_train;
  vector[N_train] x_train;
  vector[N_train] y_train;   // declared, deliberately unused
}
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 5);
  sigma ~ exponential(0.5);
}
generated quantities {
  vector[N_train] y_rep;
  for (i in 1:N_train)
    y_rep[i] = normal_rng(alpha + beta * x_train[i], sigma);
}
"""

PLOT_CODE = """
import matplotlib.pyplot as plt
import numpy as np
y_rep = idata.posterior["y_rep"].values.reshape(-1, len(cols["y"]))
lo, hi = np.percentile(y_rep, [5, 95], axis=0)
order = np.argsort(cols["x"])
x = cols["x"][order]
plt.fill_between(x, lo[order], hi[order], alpha=0.3, label="prior predictive 90%")
plt.plot(x, cols["y"][order], "ko", ms=4, label="observed")
plt.legend(); plt.xlabel("x"); plt.ylabel("y")
plt.savefig("prior_predictive.png")
print("coverage:", float(((cols["y"] >= lo) & (cols["y"] <= hi)).mean()))
"""


def test_uc02_prior_predictive(tmp_path, monkeypatch):
    from fastmcp.utilities.types import Image

    monkeypatch.setattr(srv, "_DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(srv, "_RESULTS_DIR", tmp_path / "results")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "results").mkdir()

    up = srv._save_dataset("rt", TRAIN_CSV, DATASET_MD)
    assert up["status"] == "ok"
    name = up["dataset"]

    # 1. Simulate from the priors (no likelihood statement).
    fit = srv.sample(
        stan_code=PRIOR_ONLY_MODEL, dataset=name,
        config={"chains": 2, "iter_warmup": 200, "iter_sampling": 200},
    )
    assert fit["status"] == "ok", fit

    # 2. "Show me": plot simulated vs observed, figure returned as an image.
    r = srv.run_python_code(code=PLOT_CODE, dataset=name, run_id=fit["run_id"])
    assert isinstance(r, list), r
    head = r[0]
    assert head["status"] == "ok", head
    assert head["figures"] == ["prior_predictive.png"]
    assert "coverage:" in head["stdout"]          # aggregates, not raw rows
    assert isinstance(r[1], Image)
