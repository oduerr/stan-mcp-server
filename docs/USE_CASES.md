# Use cases

What a normal user should be able to do with an agent connected to this
server — collected **before** building features, so every feature must name
the use case it serves. Each entry is a mini-transcript: what the user says,
what the agent does with today's tools, an honest verdict, and the gap if
there is one.

Conventions: ✅ works today · ⚠️ works with friction or only in some clients ·
❌ blocked. Gaps are named in the [gap table](#gap-table) at the end; that
table is the roadmap. When a gap is closed, update the verdicts here and add
a test under `tests/use_cases/` (one scripted, LLM-free test per use case —
so this document cannot drift from reality; see Incident 2 in
[TOOL_POLICY.md](../TOOL_POLICY.md) for why that matters).

Status of this collection: **draft — extend and edit freely.**

---

## UC-1 · First model on my own data

> "Here is `sales.csv`. Fit a Bayesian model to it."

Agent: writes a `dataset.md` with a Data Interface block → uploads both via
`POST /dataset/sales` (shell, never through context) → `get_data_summary`
→ proposes a model → `check_model` → `sample` → explains the result.

**Verdict: ✅** — `sample(dataset="_uploaded/…")` loads the data by name
(train-only datasets get `N_test=0` and empty `*_test` variables); `data`
entries extend/override for extra scalars. Enforced by
`tests/use_cases/test_uc01_first_model.py`.

## UC-2 · Prior predictive check

> "Before fitting anything, show me what your priors imply about the data."

Agent: writes the model with priors only (no likelihood statement), `sample`
→ simulated `y_rep` draws land on disk → plots simulated-vs-observed and
shows the figure.

**Verdict: ✅** (server started with `--enable-code-tool`) — priors-only
model → `sample` → `run_python_code(run_id=…, dataset=…)` plots simulated vs
observed and the figure returns as an MCP image, in every client. Enforced by
`tests/use_cases/test_uc02_prior_predictive.py`.

## UC-3 · Prior choice iteration

> "Sales can't be negative and rarely exceed 10 000 — choose priors that
> respect that, and show me the prior predictive again."

Agent: adjusts priors → repeats UC-2 → shows before/after side by side and
explains the reasoning.

**Verdict: ✅** — repeat UC-2 with adjusted priors; refits are cheap
(compile cache), and before/after figures come back side by side.

## UC-4 · Fit and "did all go well?"

> "Fit it. Did the sampling work? Anything I should worry about?"

Agent: `sample` → reads `n_divergences`, `r_hat_max`, `ess_bulk_min` →
reads `logs_path` if something is off → answers in plain language.

**Verdict: ✅** — every fit returns a `sampler_diagnostics` block:
divergences and E-BFMI per chain, treedepth saturation, step sizes, the worst
parameters by R-hat / bulk ESS, and terse `flags` ("low E-BFMI in chain 2
(0.097) — often a funnel; reparameterise"). Enforced by
`tests/use_cases/test_uc04_diagnostics.py` (a funnel must light up, a healthy
model must stay quiet).

## UC-5 · Posterior predictive check

> "Does the fitted model actually reproduce my data?"

Agent: model emits `y_rep` in generated quantities → draws are on disk →
overlay density / test-statistic plots → shown and interpreted.

**Verdict: ✅** — `y_rep` in generated quantities → `run_python_code` with
`az.plot_ppc` (or a hand-rolled overlay); image in every client.

## UC-6 · Model comparison

> "Is the Student-t version actually better than the Gaussian one?"

Agent: fits both with `log_lik` on the *training* data → computes PSIS-LOO
(`az.loo`) from the on-disk draws → reports elpd difference ± SE and the
Pareto-k health check.

**Verdict: ✅** — coding agents compute LOO locally (AGENTS_SETUP.md step 7);
every other client via `run_python_code(run_id=…)` with `az.loo(idata)`.

## UC-7 · Hierarchical model on grouped data

> "The data has one row per store — fit a hierarchical model with partial
> pooling across stores."

Agent: declares a group column via the Data Interface (`J`, 1-based ids) →
per-group summaries to justify pooling → centred vs non-centred as
diagnostics demand (UC-4) → compares against complete pooling (UC-6).

**Verdict: ✅** — composes from UC-1/4/6, all closed: the
centred/non-centred decision is exactly what UC-4's flags inform.

## UC-8 · Predict for new inputs

> "What does the model predict for a store with these covariates?"

Agent: adds the new points to the data as prediction inputs and emits
`y_pred` in generated quantities → refits (cheap: compiled model is cached)
→ reports predictive mean and interval.

**Verdict: ⚠️** — workable via refit; a standalone generated-quantities pass
over an existing fit (the machinery already exists server-side for the
shadow evaluation) would avoid refitting. Low priority. **Gap G4.**

## UC-9 · Benchmark loop (AutoStan)

> "Improve the model on `benchmarks/regression_1d` until the NLPD stops
> falling."

Agent: `get_data_summary` → propose → `check_model` → `fit_and_evaluate` →
read NLPD + diagnostics → iterate. Held-out labels stay unreachable;
[TOOL_POLICY.md](../TOOL_POLICY.md) governs the offered tool surface.

**Verdict: ✅** — this is the mature path the server was built on.

---

## Gap table

The roadmap, ordered. A gap is closed only when the use-case verdicts above
flip and a test under `tests/use_cases/` enforces them.

| Gap | What | Unblocks | Status |
|---|---|---|---|
| **G1** | `sample(dataset=…)` — load uploaded/staged data by name instead of inline through context | UC-1, 2, 3, 7 | **closed** (2026-08-20) — incl. merge contract for `data` in both fit tools |
| **G2** | `run_python_code(code, dataset=…, run_id=…)` — contained server-side execution with train columns and/or the run's draws preloaded, figures returned as MCP **image content** (works in every client, incl. Claude Desktop) | UC-2, 3, 5, 6, 7 | **closed** (2026-08-20) — flag-gated (`--enable-code-tool`), TOOL_POLICY row, containment test |
| **G3** | Consultation-grade diagnostics: E-BFMI, treedepth saturation, per-parameter worst R-hat/ESS, divergences per chain | UC-4, 7 | **closed** (2026-08-20) — `sampler_diagnostics` block on every fit |
| **G4** | Standalone generated-quantities pass over an existing fit (reuse the shadow-pass machinery) | UC-8 | open — low priority |
