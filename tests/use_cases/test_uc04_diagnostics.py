#!/usr/bin/env python3
"""UC-4 — fit and "did all go well?" (docs/USE_CASES.md).

Scripted agent, no LLM: a pathological model (Neal's funnel, centred) must
produce consultation-grade diagnostics — per-chain divergences, E-BFMI,
worst parameters, and actionable flags — while a healthy model stays quiet.

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

FUNNEL = """
data { int<lower=0> N_unused; }
parameters { real y; vector[9] x; }
model { y ~ normal(0, 3); x ~ normal(0, exp(y)); }
"""

HEALTHY = """
data { int<lower=0> N_unused; }
parameters { real mu; real<lower=0> sigma; }
model { mu ~ normal(0, 1); sigma ~ exponential(1); }
"""

CFG = {"chains": 2, "iter_warmup": 400, "iter_sampling": 400, "seed": 1}


def _fit(code, tmp_path):
    srv._RESULTS_DIR = tmp_path
    return srv.sample(stan_code=code, data={"N_unused": 0}, config=CFG)


def test_uc04_funnel_produces_consultation_grade_diagnostics(tmp_path):
    r = _fit(FUNNEL, tmp_path)
    assert r["status"] == "ok", r
    d = r["sampler_diagnostics"]
    # Structure: per-chain vectors of the right length, worst-parameter lists.
    assert len(d["divergences_per_chain"]) == 2
    assert len(d["e_bfmi_per_chain"]) == 2
    assert d["worst_r_hat"] and d["lowest_ess_bulk"]
    # The funnel is pathological: divergences are near-certain with defaults,
    # and at least one flag must point the agent at the problem.
    assert sum(d["divergences_per_chain"]) == r["diagnostics"]["n_divergences"]
    assert r["diagnostics"]["n_divergences"] > 0
    assert d["flags"], d
    assert any("divergent" in f or "E-BFMI" in f for f in d["flags"])


def test_uc04_healthy_model_stays_quiet(tmp_path):
    r = _fit(HEALTHY, tmp_path)
    assert r["status"] == "ok", r
    d = r["sampler_diagnostics"]
    assert sum(d["divergences_per_chain"]) == 0
    assert d["max_treedepth_frac"] == 0.0
    assert all(v > 0.5 for v in d["e_bfmi_per_chain"])
    assert d["flags"] == []
