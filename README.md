# Stan MCP Server

Give an LLM agent safe hands for Bayesian modelling. The agent writes
[Stan](https://mc-stan.org) code; this server compiles it, runs MCMC
(CmdStan), and answers with one honest number — the negative log predictive
density (NLPD) on **held-out data the agent can never see** — plus
convergence diagnostics. Raw data and posterior draws stay on disk and never
enter the model's context.

This is the evaluation server behind the
[AutoStan](https://github.com/tidit-ch/autostan) loop: propose a model → fit →
read the score → reason → propose a better one.

```
agent:  fit_and_evaluate(stan_code="…linear model…", dataset="benchmarks/regression_1d")
server: {"nlpd": 2.24, "r_hat_max": 1.00, "n_divergences": 0, "runtime_sec": 0.4}
agent:  "Residual spread varies with x — trying a Student-t likelihood."
agent:  fit_and_evaluate(stan_code="…student_t…", …)
server: {"nlpd": 1.56, …}          ← better predictions of unseen data
agent:  fit_and_evaluate(stan_code="…quartic mean + student_t…", …)
server: {"nlpd": 1.17, …}          ← oracle for this dataset: 0.94
```

## Choose your door

| You are… | Do this |
|---|---|
| 🙂 **Non-technical** — you use Claude Code, opencode, or a similar agent | Copy the prompt below into your agent and watch |
| 🤖 **An agent** told to set this up | Read [AGENTS.md](AGENTS.md) and follow it step by step |
| 🔧 **Technically curious** | [docs/REFERENCE.md](docs/REFERENCE.md) for the full surface, [TOOL_POLICY.md](TOOL_POLICY.md) for the leakage model |

### The copy-paste prompt

> Read https://github.com/oduerr/stan-mcp-server/blob/main/AGENTS.md and
> follow it: install the Stan MCP server on this machine, verify it works,
> register it, then run the demo — fit a Bayesian model to the bundled
> `benchmarks/regression_1d` dataset, improve it once, and explain the result
> to me in plain language.

Your agent will install everything (Python environment, CmdStan — the one
slow step, ~15 minutes of C++ compilation, once per machine), start the
server, and walk you through a model-improvement loop like the transcript
above.

## Quick start (technical humans)

```bash
git clone https://github.com/oduerr/stan-mcp-server.git && cd stan-mcp-server
uv venv && uv pip install -e .
.venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"   # once
.venv/bin/stan-mcp-server --datasets-dir datasets --results-dir results
```

MCP endpoint: `http://127.0.0.1:8765/mcp` · HTTP sidecar: port 8766 ·
verify with `.venv/bin/python test_server_http.py`. Everything else —
tools, dataset conventions, uploads, remote deployment, auth — is in
[docs/REFERENCE.md](docs/REFERENCE.md).

## What the agent gets

- `fit_and_evaluate` — the core loop: Stan code in, held-out NLPD +
  diagnostics out
- `get_data_summary` — compact per-column statistics, never raw rows
- `check_model`, `sample`, `list_datasets`, `get_capabilities`,
  `get_upload_instructions`
- `GET /train/{dataset}` on the sidecar — clients that run code locally can
  download `train.csv` **to disk** and do real EDA without the data passing
  through the model's context

## The safety model, in three sentences

Held-out test labels live in `protected/` directories that no tool and no
endpoint will serve — the agent optimises against a score it cannot fit
directly. Bulk data moves over a separate HTTP channel, so nothing large ever
enters LLM context. What each tool may leak, and which tools a benchmark
agent may be offered, is specified in [TOOL_POLICY.md](TOOL_POLICY.md) — the
single authoritative document.
