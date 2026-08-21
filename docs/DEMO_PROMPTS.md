# Demo prompts

Copy-paste prompts that have actually been run against this server, with the
client and model that ran them. For what the server *should* support (specs
with verdicts) see [USE_CASES.md](USE_CASES.md); this file is the record of
what a real model actually did.

**Status column:** ✅ verified — a human ran it and it worked · 🧪 untested —
plausible but nobody has run it end to end yet. Do not promote a prompt to ✅
without naming the client, model and date.

| # | Prompt | Exercises | Verified with |
|---|---|---|---|
| P1 | Football attack/defence (below) | file-copy upload (stdio), `get_data_summary`, `sample`, hierarchical Poisson | ✅ Claude Desktop · Sonnet 5 · 2026-08-21 |
| P2 | Install and demo (below) | whole setup path, `fit_and_evaluate`, model improvement | ✅ scripted equivalent; 🧪 as a single prompt |
| P3 | Bring your own CSV (below) | upload, `sample`, PSIS-LOO | 🧪 |
| P4 | Prior-predictive workflow (below) | priors-only `sample`, `run_python_code` images | ✅ scripted (`tests/use_cases/test_uc02_prior_predictive.py`); 🧪 as a prompt |

---

## P1 · Football attack and defence ✅

Needs: a client that can fetch web pages, and the server reachable (Claude
Desktop → stdio, so the agent uploads by copying files into the datasets
directory — `get_upload_instructions` tells it where).

> Analyse the attack and defense capabilities of the French football league
> 2025/2026 using a Bayesian Poisson Attack Defense Model.
>
> Hint: For the data, use footballdatabase.com — specifically the
> league-scores-tables page (format:
> `footballdatabase.com/league-scores-tables/france-ligue-1-2025-2026`).

Why it is a good demo: the agent has to do every part of the job — find and
scrape the fixtures, shape them into a `train.csv` plus a `## Data Interface`
block, get them onto the server without pasting them into the conversation,
write a non-trivial hierarchical model (per-team attack and defence effects
with a shared home advantage), sample it, and then explain team-level
parameters as football rather than as numbers. Nothing about it is
Stan-specific boilerplate, and the result is interpretable by someone who has
never heard of MCMC.

Swap the league by changing the URL: `england-premier-league-2025-2026`,
`germany-bundesliga-2025-2026`, `spain-laliga-2025-2026`.

## P2 · Install and demo ✅ / 🧪

The README's onboarding prompt — hand this to an agent on a machine that does
not have the server yet.

> Read https://github.com/oduerr/stan-mcp-server/blob/main/AGENTS_SETUP.md and
> follow it: install the Stan MCP server on this machine, verify it works,
> register it, then run the demo — fit a Bayesian model to the bundled
> `benchmarks/regression_1d` dataset, improve it once, and explain the result
> to me in plain language. Plot the posterior predictive.

Every step of the runbook has been executed and verified in a clean-room
clone; the prompt as a single instruction to a model has not been recorded
here yet.

## P3 · Bring your own CSV 🧪

> I have my data in `sales.csv` and want a Bayesian model for it. Upload it to
> the Stan MCP server, look at the data summary, propose a model, fit it, and
> compare a couple of variants with PSIS-LOO. Explain your choices as you go.

## P4 · Prior-predictive workflow ✅ / 🧪

Needs `--enable-code-tool` (figures come back as images).

> Before fitting anything, show me what my priors imply about the data. Then
> pick priors that respect that sales cannot be negative and rarely exceed
> 10 000, show the prior predictive again, fit the model, and tell me whether
> the sampling went well.

The mechanics are covered by a scripted test; the phrasing above has not been
run against a model yet.

---

## Adding a prompt

1. Paste the prompt verbatim — the wording is the artifact, including hints
   that turned out to matter (P1 needed the URL format spelled out).
2. Say what it exercises, and what the setup must provide (web access, code
   tool, transport).
3. Mark ✅ only with client + model + date. Otherwise 🧪.
4. If it revealed a bug or gap, link the issue or the gap row in
   [USE_CASES.md](USE_CASES.md).
