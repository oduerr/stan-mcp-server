# Demo prompts

## P1 · Football attack and defence (Claude Desktop / Sonnet 5.0)

Needs: a client that can fetch web pages, and the server reachable (Claude
Desktop → stdio, so the agent uploads by copying files into the datasets
directory — `get_upload_instructions` tells it where). Tested 

> Analyse the attack and defense capabilities of the French football league
> 2025/2026 using a Bayesian Poisson Attack Defense Model.
>
> Hint: For the data, use footballdatabase.com — specifically the
> league-scores-tables page (format:
> `footballdatabase.com/league-scores-tables/france-ligue-1-2025-2026`).

Remarks: Tried with OpenCode (Qwee 3.5 397B Reasoning), it could not get the data so it simulated it w/o telling me.

## P2 · Install and demo ✅ / 🧪

The README's onboarding prompt — hand this to an agent on a machine that does
not have the server yet.

> Read [https://github.com/oduerr/stan-mcp-server/blob/main/AGENTS_SETUP.md](https://github.com/oduerr/stan-mcp-server/blob/main/AGENTS_SETUP.md) and
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

