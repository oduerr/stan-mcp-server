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
  --datasets-dir /path/to/datasets \
  --results-dir  /path/to/results
```

The MCP server listens at `http://127.0.0.1:8765/mcp` and the HTTP upload
endpoint at `http://127.0.0.1:8766/dataset/{name}` by default.

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

Startup output:

```
Stan MCP Server starting on http://127.0.0.1:8765/mcp
  datasets : /path/to/datasets
  results  : /path/to/results
  cache    : /tmp/stan_mcp_model_cache
  upload   : http://127.0.0.1:8766/dataset/{name}
```

## Tools

| Tool | Purpose |
|------|---------|
| `get_capabilities` | Query available tools, server configuration, and upload URL |
| `list_datasets` | List pre-staged and uploaded datasets |
| `get_data_summary` | Compact EDA for a named dataset |
| `check_model` | Compile-only check (syntax + `log_lik` presence) |
| `fit_and_evaluate` | Sample + compute NLPD; returns scalar diagnostics + run asset URLs |
| `sample` | Sample; returns scalar diagnostics + run asset URLs |
| `get_upload_instructions` | Return HTTP upload URL and field names for datasets |
| `get_run_history` | Return the logged NLPD history for a dataset |

**Recommended call order for an AutoStan loop:**
`get_capabilities` → `list_datasets` → `get_data_summary` → `check_model` → `fit_and_evaluate` → `get_run_history`

## Connecting from Claude desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "stan": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

For a remote server, add a bearer token (see **Security** below).

## Run assets — logs and posterior draws

Every `sample` and `fit_and_evaluate` call persists results server-side under
a short `run_id` and returns only scalar diagnostics plus two URLs.  Bulk data
**never enters LLM context**.

```json
{
  "run_id":      "3a7f9c1e20b4",
  "nlpd":        1.423,
  "r_hat_max":   1.003,
  "n_divergences": 0,
  "ess_bulk_min": 2841,
  "runtime_sec": 4.2,
  "logs_url":    "http://127.0.0.1:8766/logs/3a7f9c1e20b4",
  "samples_url": "http://127.0.0.1:8766/samples/3a7f9c1e20b4"
}
```

| Endpoint | Returns |
|---|---|
| `GET /logs/{run_id}` | CmdStan stdout/stderr as plain text |
| `GET /samples/{run_id}` | Per-chain Stan CSV files as a `.tar.gz` |

Download samples:

```bash
curl http://127.0.0.1:8766/samples/3a7f9c1e20b4 -o samples.tar.gz
tar xf samples.tar.gz          # expands to per-chain *.csv
```

Load in Python (requires `cmdstanpy` or `arviz`):

```python
import arviz as az
idata = az.from_cmdstan(["model-1.csv", "model-2.csv", "model-3.csv", "model-4.csv"])
```

Run assets are stored under `<results-dir>/_runs/<run_id>/` and are never
automatically deleted.

## Uploading datasets at runtime

Datasets must be uploaded via the HTTP endpoint so that CSV content —
including **test labels** — never passes through LLM context.  The LLM calls
`get_upload_instructions()` to retrieve the URL and field names, then passes
them to the user or an automated client.

### HTTP upload endpoint

```bash
curl -X POST http://127.0.0.1:8766/dataset/my_experiment \
     -F train=@train.csv \
     -F test=@test.csv \
     -F dataset_md=@dataset.md   # optional
```

Or from Python:

```python
import requests

with open("train.csv") as tr, open("test.csv") as te:
    r = requests.post(
        "http://127.0.0.1:8766/dataset/my_experiment",
        files={"train": tr, "test": te},
    )
r.raise_for_status()
print(r.json())   # {"status": "ok", "dataset": "_uploaded/my_experiment", ...}
```

After a successful upload pass `_uploaded/my_experiment` to
`fit_and_evaluate` / `get_data_summary`.

To disable the HTTP endpoint entirely:

```bash
stan-mcp-server --datasets-dir ... --results-dir ... --upload-port 0
```

Uploaded datasets are stored under `<datasets-dir>/_uploaded/` on the server.
Dataset names may only contain letters, digits, underscores, and hyphens.

## Dataset layout

Each dataset lives in its own subdirectory under `--datasets-dir`:

```
datasets/
  my_dataset/
    train.csv               ← training features + response
    dataset.md              ← description + ## Data Interface block
    protected/
      test.csv              ← held-out test features + response
```

The LLM only needs to pass the dataset name — the server loads data
automatically:

```python
fit_and_evaluate(stan_code=..., dataset="my_dataset", notes="...", rationale="...")
```

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
     -F train=@train.csv \
     -F test=@test.csv
```

### 5. Download results with the token

```bash
curl http://<server-ip>:8766/samples/<run_id> \
     -H "Authorization: Bearer a3f8c2d1e4b5..." \
     -o samples.tar.gz
```
