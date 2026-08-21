# AGENTS_SETUP.md — install and operate the Stan MCP Server

You are an agent asked to install this server. Follow the steps in order.
Each step has a verification — do not continue past a failed verification.
When you are done, run the demo in step 6 and explain the result to your
human in plain language.

**What this server is:** it gives you structured Bayesian modelling tools
over MCP. You write Stan code; the server compiles it, runs MCMC, and returns
compact JSON (scores and convergence diagnostics). Raw data and posterior
draws stay on disk — never request them into your context. It serves two
jobs: helping your human model **their own data** (step 7), and benchmark
loops where a model is scored on held-out data (`fit_and_evaluate`).

**Rules while working with this repository:**

- Never read, copy, or serve anything under `datasets/*/protected/` — those
  are held-out test labels. See [TOOL_POLICY.md](TOOL_POLICY.md).
- Do not paste CSV contents into your context. Compute on files with scripts;
  read back aggregates.

## 1. Check the requirements

```bash
python3 --version        # need >= 3.10
uv --version             # preferred installer
```

- If `uv` is missing, install it: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  (or fall back to plain `pip` in step 2).
- A C++ toolchain is required for CmdStan (macOS: `xcode-select --install`;
  Debian/Ubuntu: `sudo apt install build-essential`). Verify: `make --version`
  and `g++ --version || clang++ --version`.

## 2. Install (local, recommended)

```bash
git clone https://github.com/oduerr/stan-mcp-server.git
cd stan-mcp-server
uv venv
uv pip install -e .
```

Without `uv`: `python3 -m venv .venv && .venv/bin/pip install -e .`

**Verify:** `.venv/bin/stan-mcp-server --help` prints usage.

> **Global install (alternative):** `uv tool install git+https://github.com/oduerr/stan-mcp-server`
> puts `stan-mcp-server` on PATH. You still need a datasets directory — clone
> the repo for the bundled examples, or create your own (see
> [docs/REFERENCE.md](docs/REFERENCE.md), *Dataset layout*). Prefer the local
> install for a first run.

## 3. Install CmdStan (once per machine)

```bash
.venv/bin/python -c "import cmdstanpy; print(cmdstanpy.cmdstan_path())"
```

- If this prints a path: CmdStan is already installed — continue.
- If it raises an error: install it. Pass `cores` — the default builds
  single-threaded. Measured: ~45 s on a 10-core laptop with all cores; expect
  a few minutes on older hardware.

```bash
.venv/bin/python -c "import cmdstanpy, os; cmdstanpy.install_cmdstan(cores=os.cpu_count())"
```

**Verify:** the `cmdstan_path()` check now prints a path.

## 4. Start the server

```bash
.venv/bin/stan-mcp-server --datasets-dir datasets --results-dir results \
  --enable-code-tool
```

`--enable-code-tool` exposes `run_python_code` — analysis code executed
server-side with figures returned as images. Include it for assistant use;
omit it for benchmark runs (see TOOL_POLICY.md).

Run it in the background (or a separate terminal). The startup banner shows
the version, both ports, and `history : get_run_history withheld (default)`.

**Verify (both checks must pass):**

```bash
# Sidecar serves training data:
curl -s http://127.0.0.1:8766/train/benchmarks/regression_1d | head -2
# → "predictor,response" and one data row

# Full MCP round-trip incl. a real fit (needs CmdStan; ~1 min first time):
.venv/bin/python test_server_http.py
# → "All assertions passed."
```

## 5. Register the server with the MCP client

Pick the client your human uses:

**Claude Code:**

```bash
claude mcp add --transport http stan http://127.0.0.1:8765/mcp
```

**Claude Desktop / any client using JSON config** (`.mcp.json` or
`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "stan": { "url": "http://127.0.0.1:8765/mcp" }
  }
}
```

**opencode** (`opencode.json`):

```json
{
  "mcp": {
    "stan": { "type": "remote", "url": "http://127.0.0.1:8765/mcp", "enabled": true }
  }
}
```

If the server runs with `--token <secret>`, add the header
`"Authorization": "Bearer <secret>"` (Claude Code:
`--header "Authorization: Bearer <secret>"`).

**Verify:** from the client, call the `get_capabilities` tool. It returns the
tool list and the server version.

## 6. Run the demo for your human

Using the MCP tools (not shell), on the bundled dataset:

1. `get_data_summary(dataset="benchmarks/regression_1d")` — describe the data
   in one sentence.
2. `fit_and_evaluate` with a simple linear model (the Data Interface block in
   `dataset_md` names the variables; the model must end with a
   `vector[N_test] log_lik` in `generated quantities`).
3. Report the NLPD and diagnostics. Then improve the model once — the data
   has non-Gaussian noise; a Student-t likelihood is a good second step — and
   show that the NLPD drops (linear Gaussian ≈ 2.2; Student-t lands near 1.6).
4. Explain to your human: **lower NLPD = better predictions of unseen data**,
   and the test labels never entered your context.

## 7. Working with your human's own data

The server is not only for benchmarks — it is a general Stan back end. When
your human brings a CSV:

1. Call `get_upload_instructions` for the upload URL, then upload **from the
   shell**, never by pasting the CSV into a tool argument:
   ```bash
   curl -X POST http://127.0.0.1:8766/dataset/<name> -F train=@their_data.csv
   ```
   Write a small `dataset.md` with a `## Data Interface` block first and pass
   it as `-F dataset_md=@dataset.md` — without it no CSV columns are loaded
   (see [docs/REFERENCE.md](docs/REFERENCE.md), *dataset.md convention*).
2. `get_data_summary(dataset="_uploaded/<name>")` to inspect the data.
3. Uploaded datasets are **train-only**: use `sample` (not `fit_and_evaluate`)
   and compare models with PSIS-LOO on the training `log_lik`. The draws are
   on disk at `samples_path` — compute the comparison with a script:
   ```python
   import glob, arviz as az
   csvs = sorted(glob.glob("<samples_path>/samples/*.csv"))
   try:                                    # arviz < 1
       idata = az.from_cmdstan(csvs, log_likelihood="log_lik")
   except AttributeError:                  # arviz >= 1 removed from_cmdstan
       from cmdstanpy import from_csv
       idata = az.from_cmdstanpy(from_csv("<samples_path>/samples"),
                                 log_likelihood="log_lik")
   print(az.loo(idata))
   ```
4. **Show, don't only tell** (needs `--enable-code-tool`): use
   `run_python_code(dataset=…, run_id=…)` for prior/posterior predictive
   plots, trace plots and `az.loo(idata)` — every saved .png comes back as an
   image your human sees inline. `cols` (train columns) and `idata` (the
   run's draws) are predefined.
5. Beyond scores, do what a good statistician does: read `logs_path` when a
   fit misbehaves, explain divergences and R-hat in plain language, propose
   prior and likelihood changes one at a time, and say *why*.
6. If your human wants honest held-out evaluation, tell them to place a test
   split at `<datasets-dir>/_uploaded/<name>/protected/test.csv` themselves —
   then `fit_and_evaluate` works and you must not look at that file.

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `command not found: stan-mcp-server` | venv not used → call `.venv/bin/stan-mcp-server` |
| `cmdstanpy` raises `CmdStan installataion not found` | step 3 not done → `install_cmdstan()` |
| CmdStan install fails compiling | missing C++ toolchain → step 1 |
| Port 8765/8766 already in use | another instance runs → `pkill -f stan-mcp-server`, or pass `--port`/`--upload-port` |
| `python3 -m venv` fails with ensurepip error (Debian/Ubuntu) | `sudo apt install python3-venv`, or use `uv venv` |
| `401 Unauthorized` | server started with `--token` → send the `Authorization: Bearer` header |
| `fit_and_evaluate` returns `stage: sampling_timeout` | model too slow (often a full GP) → simpler parameterisation; the 900 s ceiling cannot be raised |
| `fit_and_evaluate` refuses an uploaded dataset | uploaded datasets are train-only → use `sample` + PSIS-LOO, or have the operator place `protected/test.csv` |

## Remote server instead of local?

Start the server on the remote machine with `--token` (the server runs
arbitrary code and accepts uploads — the token keeps strangers off the ports),
tunnel both ports
(`ssh -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 user@host`), then
register `http://127.0.0.1:8765/mcp` as above. Details:
[docs/REFERENCE.md](docs/REFERENCE.md), *Remote deployment*.
