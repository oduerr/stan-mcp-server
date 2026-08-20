# Reference

The complete technical documentation for the Stan MCP Server. For a first start, read the [README](../README.md); for installation, read
[AGENTS.md](../AGENTS.md); for the leakage model and the permitted agent tool
surface, [TOOL_POLICY.md](../TOOL_POLICY.md) is authoritative.

## Architecture

```
LLM agent ──MCP (streamable-http, port 8765)──▶ stan-mcp-server ──▶ CmdStan
              compact JSON only                       │
client ─────HTTP sidecar (port 8766)─────────────────┘
              bulk data: uploads in, train.csv out
                                          datasets/…/protected/  ← never served
```

Two channels, deliberately separate:


| Channel                  | Carries                                      | Passes through LLM context?   |
| ------------------------ | -------------------------------------------- | ----------------------------- |
| MCP tools (port 8765)    | compact JSON: NLPD, diagnostics, stats, URLs | yes                           |
| HTTP sidecar (port 8766) | bulk bytes: dataset uploads, train downloads | no — client ↔ server directly |


Held-out data (`protected/test.csv`, `protected/shadow.csv`) is reachable
through neither channel.

## Running the server

```bash
stan-mcp-server \
  --datasets-dir /path/to/datasets \
  --results-dir /path/to/results \
  --host 127.0.0.1 \       # default
  --port 8765 \            # MCP endpoint (default)
  --upload-port 8766 \     # HTTP sidecar (default; 0 to disable)
  --token <secret> \       # optional bearer auth (or env STAN_MCP_TOKEN)
  --transport streamable-http   # or 'stdio' for Claude Desktop via SSH
```

`--include-run-history` additionally exposes the `get_run_history` tool
(default: withheld). It returns cross-session results and must never be
offered to benchmark agents — see [TOOL_POLICY.md](../TOOL_POLICY.md).

## MCP tools

Which of these a **benchmark agent** may be offered — and which leak — is
specified in [TOOL_POLICY.md](../TOOL_POLICY.md). The table below is the full
server surface, not the permitted agent surface.



Where the table says **run asset paths**, it means the two fields
`logs_path` (the CmdStan console log, a text file) and `samples_path` (the
run directory holding `model.stan` plus one posterior-draw CSV per chain).
Both lie under `--results-dir/_runs/<run_id>/`. The paths — not the
contents — are returned, so the agent (or you) can open them on disk; see
[Run assets](#run-assets--logs-and-posterior-draws) for the layout and how
to load the draws.




| Tool                      | Purpose                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_capabilities`        | Tool list (from the live registry), server version and configuration, upload and train-download URLs                                        |
| `list_datasets`           | List pre-staged and uploaded datasets                                                                                                       |
| `get_data_summary`        | Compact EDA for a named dataset: per-column stats (categorical columns summarized by levels), `tier`, `has_test`, `dataset_md`, `train_url` |
| `check_model`             | Compile-only check (syntax + `log_lik` presence)                                                                                            |
| `fit_and_evaluate`        | Sample + compute NLPD on the held-out test set; pre-staged datasets only                                                                    |
| `sample`                  | Sample; returns scalar diagnostics + run asset paths                                                                                        |
| `get_upload_instructions` | HTTP upload URL and field names for datasets                                                                                                |
| `get_run_history`         | Logged NLPD history for a dataset — ⚠️ cross-session; withheld unless `--include-run-history`                                               |


**Recommended call order:**
`get_capabilities` → `list_datasets` → `get_data_summary` → `check_model` →

- **Pre-staged dataset** (`tier: staged`): `fit_and_evaluate`
- **Uploaded dataset** (`tier: uploaded`): `sample` → compute PSIS-LOO yourself



## HTTP sidecar endpoints

Why a separate channel for bulk data? Three reasons. A CSV pasted into a
conversation consumes context (and money) on something the model cannot use
well — LLMs reason poorly over thousands of raw numbers, and a long context
degrades everything that follows. The aggregates the tools return carry the
information that actually drives modelling decisions at a fraction of the
tokens. And for benchmark datasets it is a hard requirement, not an
optimisation: held-out labels must never appear in the model's context, so
bulk data needs a path that bypasses it entirely.


| Endpoint               | Direction | Carries                                                             |
| ---------------------- | --------- | ------------------------------------------------------------------- |
| `POST /dataset/{name}` | in        | train CSV + optional `dataset.md` (multipart)                       |
| `GET /train/{dataset}` | out       | `train.csv`, or `dataset.md` with `?file=dataset.md` — nothing else |


`GET /train` exists so clients that execute code locally (a coding agent, an
agent loop with a local EDA tool) can compute on the raw train set without the
CSV entering LLM context: fetch to disk, print aggregates.

```bash
curl -o train.csv   http://127.0.0.1:8766/train/benchmarks/regression_1d
curl -o dataset.md "http://127.0.0.1:8766/train/benchmarks/regression_1d?file=dataset.md"
```

The filename is whitelisted, dataset names are traversal-checked, and any path
containing `protected/` is refused. The URL is advertised as
`train_download_url` in `get_capabilities` and per-dataset as `train_url` in
`get_data_summary`.

## Sampling configuration and limits

Defaults: 4 chains × 1000 warmup + 1000 sampling draws, seed 42. The caller
can change these per call via `config`, within server-side caps:


| Setting                         | Default | Cap                                   |
| ------------------------------- | ------- | ------------------------------------- |
| `chains`                        | 4       | 16                                    |
| `iter_warmup` / `iter_sampling` | 1000    | 10 000                                |
| `max_runtime_sec`               | 900     | can only be **lowered** by the caller |


The wall-clock ceiling exists because a pathological posterior samples
forever; a timeout is returned to the agent as a normal error
(`stage: sampling_timeout`) with a hint to simplify the model.

## Run assets — logs and posterior draws

Every `sample` and `fit_and_evaluate` call persists results under a short
`run_id` and returns only scalar diagnostics plus filesystem paths. Bulk data
**never enters LLM context**.

```json
{
  "run_id":        "3a7f9c1e20b4",
  "nlpd":          1.423,
  "r_hat_max":     1.003,
  "n_divergences": 0,
  "ess_bulk_min":  2841,
  "runtime_sec":   4.2,
  "logs_path":     "/path/to/results/_runs/3a7f9c1e20b4/logs.txt",
  "samples_path":  "/path/to/results/_runs/3a7f9c1e20b4"
}
```

`samples_path` is a directory containing one Stan CSV per chain. Load them
directly (requires `arviz`):

```python
import glob, arviz as az
csvs = sorted(glob.glob("/path/to/results/_runs/3a7f9c1e20b4/samples/*.csv"))
idata = az.from_cmdstan(csvs)
```

Run assets are stored under `<results-dir>/_runs/<run_id>/` and are never
automatically deleted.

## Datasets



### Layout

```
datasets/
  benchmarks/             ← pre-staged benchmark datasets (operator-managed)
    regression_1d/
      train.csv           ← training features + response
      dataset.md          ← description + ## Data Interface block
      protected/
        test.csv          ← held-out test features + response (operator-placed)
  _uploaded/              ← agent-uploaded, train-only datasets
    my_experiment/
      train.csv
      dataset.md          ← optional
```

The dataset name passed to tools is the path relative to `--datasets-dir`,
e.g. `benchmarks/regression_1d` or `_uploaded/my_experiment`.

### Two-tier system


| Tier         | How created                                        | `fit_and_evaluate`   | Suggested evaluation |
| ------------ | -------------------------------------------------- | -------------------- | -------------------- |
| **staged**   | Operator places `train.csv` + `protected/test.csv` | ✅ real held-out NLPD | `fit_and_evaluate`   |
| **uploaded** | Agent/user uploads via HTTP (train only)           | ❌ blocked            | `sample` + PSIS-LOO  |


`get_data_summary` returns `tier` and `has_test` so the agent knows which
path to follow before writing any Stan code.

The upload endpoint accepts **training data only**. Test data must be placed
manually by the server operator — it never passes through the agent or HTTP
layer. This is a deliberate security boundary: the agent cannot see held-out
labels even in principle.

### Uploading

```bash
curl -X POST http://127.0.0.1:8766/dataset/my_experiment \
     -F train=@train.csv \
     -F dataset_md=@dataset.md   # optional
```

Or from Python:

```python
import requests

with open("train.csv") as tr:
    r = requests.post(
        "http://127.0.0.1:8766/dataset/my_experiment",
        files={"train": tr},
    )
r.raise_for_status()
print(r.json())   # {"status": "ok", "tier": "uploaded", "dataset": "_uploaded/my_experiment", ...}
```

After a successful upload pass `_uploaded/my_experiment` to `sample` /
`get_data_summary`. To enable `fit_and_evaluate`, place test data at
`<datasets_dir>/_uploaded/my_experiment/protected/test.csv` manually.
Dataset names may only contain letters, digits, underscores, and hyphens.

### dataset.md convention

The `## Data Interface` section must contain a Stan-style code block
declaring all `_train` variables. Stan base names must match CSV column
names exactly (the `_train` / `_test` suffix is appended automatically):

```stan
int<lower=0> N_train;
int<lower=0> N_test;
vector[N_train] x_train;
vector[N_train] y_train;
vector[N_test]  x_test;
vector[N_test]  y_test;
```

For datasets with a grouping variable (`J`) declare it as:

```stan
int<lower=0> J;
array[N_train] int<lower=1,upper=J> group_train;
```

`J` is set to the **largest** group id seen in train + test (ids must be
1-based). The last CSV column is assumed to be the response unless
`response_col: <name>` appears anywhere in `dataset.md`.

`N_train` and `N_test` are injected automatically from the CSV row counts.
Only pass the `data` parameter of `fit_and_evaluate` when you need to
override them or supply additional scalars the CSV does not provide.

## Model contract

Every Stan model used with `fit_and_evaluate` must output a `log_lik`
vector of length `N_test` in `generated quantities`:

```stan
generated quantities {
    vector[N_test] log_lik;
    for (i in 1:N_test)
        log_lik[i] = normal_lpdf(y_test[i] | mu[i], sigma);
}
```

**No other interface contract.** Likelihood, priors, and parameterisation
are the modeller's choice.

## Compilation cache

Compiled Stan binaries are stored in a temp directory keyed by the
SHA-256 of the model source. Identical model code is never recompiled.

## Remote deployment

The recommended pattern for running the server on a remote machine (e.g. a
workstation or cloud VM accessible via VPN):

### 1. Start the server on the remote machine

```bash
stan-mcp-server \
  --host 127.0.0.1 \           # keep MCP port local; SSH tunnel handles access
  --datasets-dir /data/datasets \
  --results-dir  /data/results \
  --token $(openssl rand -hex 32)   # save this token
```



### 2. Tunnel the ports via SSH

```bash
ssh -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 user@remote-host
```

The MCP endpoint is now reachable at `http://127.0.0.1:8765/mcp` on your
local machine, and the sidecar at `http://127.0.0.1:8766`.

### 3. Mount the results directory via SSHFS

```bash
mkdir -p ~/mnt/stan-results
sshfs user@remote-host:/data/results ~/mnt/stan-results
```

Because tool responses return `logs_path` / `samples_path` as absolute paths
under `--results-dir`, and the mount makes those paths locally accessible,
the agent can read logs and samples directly.

### 4. Connect from Claude Desktop

```json
{
  "mcpServers": {
    "stan": {
      "url": "http://127.0.0.1:8765/mcp",
      "headers": { "Authorization": "Bearer <your-token>" }
    }
  }
}
```



## Security

For remote deployments (i.e. `--host 0.0.0.0`) protect the server with a
bearer token using the built-in `--token` flag, or set the environment
variable `STAN_MCP_TOKEN` (keeps the secret out of shell history):

```bash
export STAN_MCP_TOKEN=$(openssl rand -hex 32)
stan-mcp-server --host 0.0.0.0 --datasets-dir ... --results-dir ...
```

Both the MCP endpoint and the HTTP sidecar require the token. Requests
without a valid `Authorization: Bearer <token>` header receive
`401 Unauthorized`:

```bash
curl -X POST http://<server-ip>:8766/dataset/my_experiment \
     -H "Authorization: Bearer <token>" \
     -F train=@train.csv
```



## Leakage model

Three classes of leak would invalidate a benchmark measurement: held-out
labels (L1), cross-session results (L2), and the shadow-evaluation channel
(L3). What each tool and endpoint may carry, the incidents that shaped the
rules, and the checklist for adding a tool are specified in
[TOOL_POLICY.md](../TOOL_POLICY.md) — the single authoritative document;
nothing here restates it.

## Testing

```bash
pytest test_helpers.py test_shadow_isolation.py -k "not runs_shadow"  # fast, no CmdStan
python test_shadow_isolation.py --with-fit    # + real end-to-end fit (~10 s)
python test_server.py --datasets-dir datasets --results-dir /tmp/stan_results
python test_server_http.py                    # against a running server
```

CI runs the fast suite on every push and pull request.
