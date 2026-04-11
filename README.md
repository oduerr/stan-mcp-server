# PLEASE NOTE THIS IS WORK IN PROGRESS NOT READY FOR USE yet! JUST A PRIVATE PROOF OF CONCEPT FOR NOW NEEDED TO MAKE PUBLIC REPO DUE TO EASE.

# Stan MCP Server

A standalone MCP server that gives an LLM agent structured access to
CmdStan/CmdStanPy over HTTP.  The agent receives compact JSON — never raw
sampler output.

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

The server listens at `http://127.0.0.1:8765/mcp` by default.

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
  --host 127.0.0.1 \   # default
  --port 8765           # default
```

The server listens at `http://<host>:<port>/mcp` using the
[MCP streamable-http transport](https://spec.modelcontextprotocol.io/).

## Tools

| Tool | Purpose |
|------|---------|
| `get_capabilities` | Query available tools and server configuration |
| `list_datasets` | List pre-staged and uploaded datasets |
| `get_data_summary` | Compact EDA for a named dataset |
| `check_model` | Compile-only check (syntax + `log_lik` presence) |
| `fit_and_evaluate` | Sample + compute NLPD on held-out test responses |
| `sample` | Sample + return raw posterior draws |
| `upload_dataset` | Push train/test CSV content to the server at runtime |
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

## Uploading datasets at runtime

For remote servers where pre-staging files via rsync/scp is not practical,
use `upload_dataset` to push CSV content over the MCP connection:

```python
upload_dataset(
    name="my_experiment",
    train_csv="x,y\n1.0,2.1\n...",
    test_csv="x,y\n3.0,4.2\n...",
    dataset_md="..."  # optional, for type annotations
)
```

The tool returns the qualified name `_uploaded/my_experiment`, which is
then used in subsequent calls:

```python
fit_and_evaluate(stan_code=..., dataset="_uploaded/my_experiment")
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

For remote deployments add bearer-token middleware before starting the
server (auth is intentionally left out of the core to keep the server
simple for local use):

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import os

class TokenAuth(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.headers.get("Authorization") != f"Bearer {os.environ['MCP_API_TOKEN']}":
            return Response("Unauthorized", status_code=401)
        return await call_next(request)

mcp.add_middleware(TokenAuth)
```

Pass the token from the client via `headers`:

```json
{
  "mcpServers": {
    "stan": {
      "url": "http://remotehost:8765/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```
