<img src="assets/logo/stan-mcp-server.svg" width="110" align="right" alt="stan-mcp-server — plug in, and the chains stream out">

# Stan MCP Server
-- 

This is an MCP server for the [Stan](https://mc-stan.org) language. It takes a [Stan model](https://mc-stan.org/docs/stan-users-guide/) as input and does the sampling (MCMC via CmdStan). It returns the diagnostics of the sampling and, when the dataset has a held-out test set, the NLPD on it. It is designed to be used together with an LLM agent, like *Claude Code*, *OpenCode*, or a similar agent.

**Ways to use it:**

- *(work in progress)* **As a Stan assistant for your own work.** Let the agent upload your data, then write the model from your verbal description. It does the heavy lifting: fixes the compile errors, runs the sampling, and reports the diagnostics. Some workflows, we are going to support are in[docs/USE_CASES.md](docs/USE_CASES.md).
- *(work in progress)* **As an improvement loop on your own data.** Starting from first model, iterate: check priors, compare variants with PSIS-LOO, and keep what predicts better.
- **As the evaluation server of the [AutoStan](https://github.com/tidit-ch/autostan) project**, where an agent
iterates against the NLPD on **held-out data** it never sees: 
  - propose a model → fit → read the score → reason → propose a better one. This could look like:

```
agent:  fit_and_evaluate(stan_code="…linear model…", dataset="benchmarks/regression_1d")
server: {"nlpd": 2.24, "r_hat_max": 1.00, "n_divergences": 0, "runtime_sec": 0.4}
agent:  "Residual spread varies with x — trying a Student-t likelihood."
agent:  fit_and_evaluate(stan_code="…student_t…", …)
server: {"nlpd": 1.56, …}          ← better predictions of unseen data
agent:  fit_and_evaluate(stan_code="…quartic mean + student_t…", …)
server: {"nlpd": 1.17, …}          ← oracle for this dataset: 0.94
```

## Next steps depending on...


| You are…                                                                 | Do this                                                                                                             |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| 🙂 **Non-technical** — you use Claude Code, opencode, or a similar agent | Copy the prompt below into your agent and watch                                                                     |
| 🤖 **An agent** told to set this up                                      | Read [AGENTS_SETUP.md](AGENTS_SETUP.md) and follow it step by step                                                        |
| 🔧 **Technically curious**                                               | [docs/REFERENCE.md](docs/REFERENCE.md) for the full surface, [TOOL_POLICY.md](TOOL_POLICY.md) for the leakage model |


### The copy-paste prompt

> Read [https://github.com/oduerr/stan-mcp-server/blob/main/AGENTS_SETUP.md](https://github.com/oduerr/stan-mcp-server/blob/main/AGENTS_SETUP.md) and follow it: install the Stan MCP server on this machine, verify it works, register it, then run the demo, fit a Bayesian model to the bundled `benchmarks/regression_1d` dataset, improve it once, and explain the result to me in plain language. Plot the posterior predictive.

Your agent will install everything (Python environment, plus a one-time CmdStan build, if not installed), start the server, and walk you through a model-improvement loop like the transcript above.

**Then bring your own data.** Once the server is installed, this works too:

> I have my data in `sales.csv` and want a Bayesian model for it. Upload it
> to the Stan MCP server, look at the data summary, propose a model, fit it,
> and compare a couple of variants with PSIS-LOO. Explain your choices as
> you go.

For further examples, see docs/USE_CASES.md

## Quick start (technical humans)

```bash
git clone https://github.com/oduerr/stan-mcp-server.git && cd stan-mcp-server
uv venv && uv pip install -e .
.venv/bin/python -c "import cmdstanpy, os; cmdstanpy.install_cmdstan(cores=os.cpu_count())"  # once
.venv/bin/stan-mcp-server --datasets-dir datasets --results-dir results
```

Connecting from Claude Desktop, or running the server on another machine? See *How to connect* in [docs/REFERENCE.md](docs/REFERENCE.md) — five topologies with recommendations.

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

Bulk data moves over a separate HTTP channel, so nothing large ever enters
LLM context — that holds for your own uploads as much as for benchmarks.
Held-out test labels (when a dataset has them) live in `protected/`
directories that no tool and no endpoint will serve, so an agent optimises
against a score it cannot fit directly. What each tool may leak, and which
tools a benchmark agent may be offered, is specified in
[TOOL_POLICY.md](TOOL_POLICY.md) — the single authoritative document.