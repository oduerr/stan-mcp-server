# Stan MCP Server

A standalone MCP server that gives an LLM agent structured access to
CmdStan/CmdStanPy over HTTP.  The agent receives compact JSON — never raw
sampler output.

Large datasets are uploaded directly from the client to the server's HTTP
upload endpoint, so CSV content never passes through LLM context.

## Quick start

```bash
# 1. Install (requires uv — https://docs.astral.sh/uv/getting-started/installation/)
uv pip install -e .

# 2. Install CmdStan (once)
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# 3. Start the server
stan-mcp-server \
  --datasets-dir datasets \
  --results-dir  results
```

The MCP server listens at `http://127.0.0.1:8765/mcp` and the HTTP upload
endpoint is at `http://127.0.0.1:8766/dataset/{name}` by default.

## Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- CmdStan (installed via the step above)

## Installation

```bash
# Recommended — uv (fast, isolated)
uv pip install -e .

# Alternative — plain pip
pip install -e .
```

## Running the server

```bash
stan-mcp-server \
  --datasets-dir /path/to/datasets \
  --results-dir  /path/to/results \
  --host 127.0.0.1 \     # default
  --port 8765 \          # MCP endpoint (default)
  --upload-port 8766     # HTTP upload endpoint (default; 0 to disable)
```

This is could also be started with (after activating the `uv` with `source .venv/bin/activate`

```
python stan_mcp_server/server.py \
  --datasets-dir datasets \
  --results-dir results
```


## Tools

| Tool | Purpose |
|------|---------|
| `get_capabilities` | Query available tools, server configuration, and upload URL |
| `list_datasets` | List pre-staged and uploaded datasets |
| `get_data_summary` | Compact EDA for a named dataset (includes `tier` and `has_test`) |
| `check_model` | Compile-only check (syntax + `log_lik` presence) |
| `fit_and_evaluate` | Sample + compute NLPD on held-out test; pre-staged datasets only |
| `sample` | Sample; returns scalar diagnostics + run asset paths |
| `get_upload_instructions` | Return HTTP upload URL and field names for datasets |
| `get_run_history` | Return the logged NLPD history for a dataset |

**Recommended call order:**
`get_capabilities` → `list_datasets` → `get_data_summary` → `check_model` →
- **Pre-staged dataset** (`tier: staged`): `fit_and_evaluate` → `get_run_history`
- **Uploaded dataset** (`tier: uploaded`): `sample` → compute PSIS-LOO yourself

## Run assets — logs and posterior draws

Every `sample` and `fit_and_evaluate` call persists results under a short
`run_id` and returns only scalar diagnostics plus filesystem paths.  Bulk data
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

`samples_path` is a directory containing one Stan CSV per chain.  Load them
directly (requires `arviz`):

```python
import glob, arviz as az
csvs = sorted(glob.glob("/path/to/results/_runs/3a7f9c1e20b4/samples/*.csv"))
idata = az.from_cmdstan(csvs)
```

Run assets are stored under `<results-dir>/_runs/<run_id>/` and are never
automatically deleted.

## Uploading datasets at runtime

The HTTP upload endpoint accepts **training data only**.  Test data must be
placed manually by the server operator — it never passes through the agent or
HTTP layer.  This is a deliberate security boundary: the agent cannot see
held-out labels even in principle.

The LLM calls `get_upload_instructions()` to retrieve the URL and field names,
then passes them to the user or an automated client.

### Two-tier dataset system

| Tier | How created | `fit_and_evaluate` | Suggested evaluation |
|------|-------------|---------------------|----------------------|
| **staged** | Server operator places `train.csv` + `protected/test.csv` | ✅ real held-out NLPD | `fit_and_evaluate` |
| **uploaded** | Agent/user uploads via HTTP (train only) | ❌ blocked | `sample` + PSIS-LOO |

`get_data_summary` returns `tier` and `has_test` so the agent knows which
path to follow before writing any Stan code.

### HTTP upload endpoint

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
`get_data_summary`.  To enable `fit_and_evaluate`, place test data at
`<datasets_dir>/_uploaded/my_experiment/protected/test.csv` manually.

To disable the HTTP endpoint entirely:

```bash
stan-mcp-server --datasets-dir ... --results-dir ... --upload-port 0
```

Uploaded datasets are stored under `<datasets-dir>/_uploaded/` on the server.
Dataset names may only contain letters, digits, underscores, and hyphens.

## Dataset layout

Datasets live under `--datasets-dir` in two areas:

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

The `protected/test.csv` file is what makes a dataset "staged" and enables
`fit_and_evaluate`.  Uploaded datasets lack this file and are limited to
`sample` + PSIS-LOO.

The LLM only needs to pass the dataset name — the server loads data
automatically:

```python
fit_and_evaluate(stan_code=..., dataset="benchmarks/regression_1d", notes="...", rationale="...")
```

`N_train` and `N_test` are injected automatically from the CSV row counts.
Only pass the `data` parameter when you need to override them or supply
additional scalars the CSV does not provide.

### dataset.md convention

The `## Data Interface` section must contain a Stan-style code block
declaring all `_train` variables.  Stan base names must match CSV column
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

The last CSV column is assumed to be the response unless `response_col: <name>`
appears anywhere in `dataset.md`.

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

## Compilation cache

Compiled Stan binaries are stored in a temp directory keyed by the
SHA-256 of the model source.  Identical model code is never recompiled.

## Remote deployment

The recommended pattern for running the server on a remote machine (e.g. a
GPU workstation or cloud VM accessible via VPN):

### 1. Start the server on the remote machine

```bash
stan-mcp-server \
  --host 127.0.0.1 \           # keep MCP port local; SSH tunnel handles access
  --datasets-dir /data/datasets \
  --results-dir  /data/results \
  --token $(openssl rand -hex 32)   # save this token
```

### 2. Tunnel the MCP port via SSH

```bash
ssh -N -L 8765:127.0.0.1:8765 user@remote-host
```

The MCP endpoint is now reachable at `http://127.0.0.1:8765/mcp` on your
local machine.

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

> **Note:** The HTTP download endpoints (`GET /logs/{run_id}`, `GET /samples/{run_id}`)
> have been removed. Access run assets directly via the SSHFS-mounted `results_dir`
> using the `logs_path` / `samples_path` returned in tool responses.

## Security

For remote deployments (i.e. `--host 0.0.0.0`) protect the server with a
bearer token using the built-in `--token` flag.

### 1. Generate a token

```bash
openssl rand -hex 32
# e.g. a3f8c2d1e4b5...
```

### 2. Start the server with the token

```bash
stan-mcp-server \
  --host 0.0.0.0 \
  --datasets-dir /path/to/datasets \
  --results-dir  /path/to/results \
  --token a3f8c2d1e4b5...
```

Alternatively, set the environment variable `STAN_MCP_TOKEN` instead of
passing `--token` on the command line (useful for keeping secrets out of
shell history):

```bash
export STAN_MCP_TOKEN=a3f8c2d1e4b5...
stan-mcp-server --host 0.0.0.0 --datasets-dir ... --results-dir ...
```

Both the MCP endpoint (port 8765) and the HTTP upload endpoint (port 8766)
require the token.  Requests without a valid `Authorization: Bearer <token>`
header receive `401 Unauthorized`.

### 3. Connect from Claude Desktop

```json
{
  "mcpServers": {
    "stan": {
      "url": "http://<server-ip>:8765/mcp",
      "headers": { "Authorization": "Bearer a3f8c2d1e4b5..." }
    }
  }
}
```

### 4. Upload datasets with the token

```bash
curl -X POST http://<server-ip>:8766/dataset/my_experiment \
     -H "Authorization: Bearer a3f8c2d1e4b5..." \
     -F train=@train.csv
```

