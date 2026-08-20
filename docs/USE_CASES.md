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

**Verdict: ⚠️** — the flow exists, but `sample` takes no `dataset=`
parameter, so the agent must pass the data dict *inline through context*,
which the whole design forbids. **Gap G1.**

## UC-2 · Prior predictive check

> "Before fitting anything, show me what your priors imply about the data."

Agent: writes the model with priors only (no likelihood statement), `sample`
→ simulated `y_rep` draws land on disk → plots simulated-vs-observed and
shows the figure.

**Verdict: ⚠️** — the simulation is a pure convention (documented in
[AGENTS.md](../AGENTS.md) once G1 lands); the *showing* works only in clients
that can execute local plotting code (Claude Code). In Claude Desktop there
is no image. **Gaps G1, G2.**

## UC-3 · Prior choice iteration

> "Sales can't be negative and rarely exceed 10 000 — choose priors that
> respect that, and show me the prior predictive again."

Agent: adjusts priors → repeats UC-2 → shows before/after side by side and
explains the reasoning.

**Verdict: ⚠️** — same dependencies as UC-2. The iteration itself is cheap
(compile cache makes refits fast). **Gaps G1, G2.**

## UC-4 · Fit and "did all go well?"

> "Fit it. Did the sampling work? Anything I should worry about?"

Agent: `sample` → reads `n_divergences`, `r_hat_max`, `ess_bulk_min` →
reads `logs_path` if something is off → answers in plain language.

**Verdict: ⚠️** — works, but the diagnostics are benchmark-grade ("is this
run valid"), not consultation-grade ("*what* went wrong *where*"). Missing:
E-BFMI, treedepth saturation, per-parameter worst offenders, divergences per
chain — the numbers behind advice like "the funnel is in `tau`, go
non-centred". **Gap G3.**

## UC-5 · Posterior predictive check

> "Does the fitted model actually reproduce my data?"

Agent: model emits `y_rep` in generated quantities → draws are on disk →
overlay density / test-statistic plots → shown and interpreted.

**Verdict: ⚠️** — same shape as UC-2: computation possible, visualization
client-dependent. **Gap G2.**

## UC-6 · Model comparison

> "Is the Student-t version actually better than the Gaussian one?"

Agent: fits both with `log_lik` on the *training* data → computes PSIS-LOO
(`az.loo`) from the on-disk draws → reports elpd difference ± SE and the
Pareto-k health check.

**Verdict: ⚠️** — works today in coding agents (documented in AGENTS.md
step 7); no path in Claude Desktop. **Gap G2** (LOO is just analysis code).

## UC-7 · Hierarchical model on grouped data

> "The data has one row per store — fit a hierarchical model with partial
> pooling across stores."

Agent: declares a group column via the Data Interface (`J`, 1-based ids) →
per-group summaries to justify pooling → centred vs non-centred as
diagnostics demand (UC-4) → compares against complete pooling (UC-6).

**Verdict: ⚠️** — everything composes from UC-1/4/6; inherits their gaps.
The `J` handling (max id, 1-based validation) is already server-side.
**Gaps G1, G2, G3.**

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
| **G1** | `sample(dataset=…)` — load uploaded/staged data by name instead of inline through context | UC-1, 2, 3, 7 | open |
| **G2** | `run_python_code(code, dataset=…, run_id=…)` — contained server-side execution with train columns and/or the run's draws preloaded, figures returned as MCP **image content** (works in every client, incl. Claude Desktop) | UC-2, 3, 5, 6, 7 | open — design discussed; must be flag-gated, TOOL_POLICY row, leak-probe test |
| **G3** | Consultation-grade diagnostics: E-BFMI, treedepth saturation, per-parameter worst R-hat/ESS, divergences per chain | UC-4, 7 | open |
| **G4** | Standalone generated-quantities pass over an existing fit (reuse the shadow-pass machinery) | UC-8 | open — low priority |
