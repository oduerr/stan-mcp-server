# HUMANS_SETUP.md — deploy and operate the server

The hands-on guide for the person running the server: install it, choose how
clients connect, put it on another machine, lock it down, and check it works.

- Want your **agent** to do the install for you? Point it at
  [AGENTS_SETUP.md](AGENTS_SETUP.md) instead — same steps, written for an agent.
- Looking for what the tools *do* (parameters, dataset conventions, contracts)?
  That is [docs/REFERENCE.md](docs/REFERENCE.md).
- What a benchmark agent may be offered, and what leaks:
  [TOOL_POLICY.md](TOOL_POLICY.md).

## Install

Requirements: Python ≥ 3.10, a C++ toolchain for CmdStan (macOS:
`xcode-select --install`; Debian/Ubuntu: `sudo apt install build-essential`),
and ideally [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/oduerr/stan-mcp-server.git
cd stan-mcp-server
uv venv && uv pip install -e .          # or: python3 -m venv .venv && .venv/bin/pip install -e .
```

CmdStan, once per machine — pass `cores`, the default builds single-threaded
(~45 s on a 10-core laptop, a few minutes on older hardware):

```bash
.venv/bin/python -c "import cmdstanpy, os; cmdstanpy.install_cmdstan(cores=os.cpu_count())"
```

Already installed? `.venv/bin/python -c "import cmdstanpy; print(cmdstanpy.cmdstan_path())"`
prints the path instead of raising.

## Simplest setup: Claude Desktop (no server to run)

If you use Claude Desktop, you never start the server yourself — Desktop
launches it for you over **stdio** and shuts it down with the app. After the
install above, open Settings → Developer → Edit Config and add:

```json
{
  "mcpServers": {
    "stan": {
      "command": "/absolute/path/to/stan-mcp-server/.venv/bin/stan-mcp-server",
      "args": ["--transport", "stdio",
               "--datasets-dir", "/absolute/path/to/stan-mcp-server/datasets",
               "--results-dir", "/absolute/path/to/stan-mcp-server/results",
               "--enable-code-tool"]
    }
  }
}
```

All paths must be **absolute** (`pwd` in the cloned directory gives you the
prefix). `--enable-code-tool` lets it return plots as images; leave it out for
benchmark work. Quit Desktop completely and reopen it, then ask *"what Stan
tools do you have?"* — it should list them.

If something is wrong, Desktop writes this server's startup banner and errors
to `~/Library/Logs/Claude/mcp-server-stan.log` (macOS) or
`%APPDATA%\Claude\logs\` (Windows).

Two consequences of stdio worth knowing: uploads happen by **copying files**
into `<datasets-dir>/_uploaded/<name>/` (the agent asks the server where), and
`--token` is refused — the client owns the process, so there is no network to
authenticate. Details in [What stdio changes](#what-stdio-changes).

## Start the server (other clients)

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

## How to connect

Two transports (`streamable-http`, `stdio`) × two locations (the server on
your machine or on a remote one) give five practical topologies. Pick by the
client you use and where the sampling should run.

| # | Topology | Client connects by | Claude Desktop? | Sidecar: uploads + `GET /train` | Auth | Recommended for |
|---|---|---|---|---|---|---|
| 1 | **Local HTTP** (default) | `http://127.0.0.1:8765/mcp` | ❌ | ✅ | none needed | **Claude Code / opencode on your own machine — start here** |
| 2 | **Remote HTTP, direct** | `http://host:8765/mcp` | ❌ | ✅ (remote port) | `--token` **required** | only inside a trusted network (VPN/Tailscale) — see the TLS caveat |
| 3 | **Remote HTTP via SSH tunnel** | `http://127.0.0.1:8765/mcp` after `ssh -L` | ❌ | ✅ through the tunnel | the tunnel | **the recommended way to use a remote sampling machine** |
| 4 | **Local stdio** | client launches the binary | ✅ | ❌ — copy files instead | rejected by design | **Claude Desktop on your own machine** |
| 5 | **Remote stdio over SSH** | client launches `ssh host stan-mcp-server --transport stdio …` | ✅ | ❌ — copy files on the remote box | SSH | **Claude Desktop + a remote sampling machine** (the only combination that does both) |

### Recommendations

- **Start with #1.** One command, everything works, nothing to secure.
- **Remote sampling: prefer #3 over #2.** The tunnel encrypts the hop, needs
  no open port, and the client config is identical to the local one. Reach for
  #2 only when a tunnel is impractical, and then only on a trusted network.
- **Claude Desktop: #4 locally, #5 for a remote machine.** Desktop's
  `claude_desktop_config.json` launches a subprocess (`command`/`args`) and has
  no `url` field for local servers, so HTTP topologies are not available to it.
- **Never expose #2 to the open internet.** This server speaks plain HTTP with
  no TLS, so a bearer token travels in clear. SSH (#3/#5) or a VPN is the
  encryption layer.
- **Run assets on remote setups**: `logs_path` / `samples_path` refer to the
  *server's* filesystem. Either mount `--results-dir` with SSHFS, or — usually
  simpler — call `run_python_code(run_id=…)` and let the server analyse the
  draws where they already are.

### Running two instances at once

An HTTP instance (for Claude Code) and a stdio instance (for Claude Desktop)
can share one `--datasets-dir`/`--results-dir`: the stdio instance binds no
ports, the compile cache and the per-dataset `log.jsonl` are protected by
inter-process locks, and `run_id`s are random. Fine for assistant use; for
**benchmark runs keep the server exclusive** — `log.jsonl` is the measurement
record and a second writer would interleave a foreign session into it. Two
HTTP instances need distinct `--port`/`--upload-port`. For full isolation,
give each instance its own `--results-dir`.

### What stdio changes

| | `streamable-http` (default) | `stdio` |
|---|---|---|
| HTTP sidecar (8766) | runs: uploads + `GET /train` | **not started** |
| Dataset upload | `POST /dataset/{name}` | copy files into `<datasets-dir>/_uploaded/<name>/` — `get_upload_instructions` returns the exact path |
| `--token` | enforced by ASGI middleware | **rejected at startup** (middleware is HTTP-only, so accepting it would promise auth that never applies) |
| Startup banner | stdout | stderr (stdout carries the protocol) |

Under stdio no tool advertises a sidecar URL: `get_capabilities` reports
`http_upload_url` / `train_download_url` as `disabled` and `get_data_summary`
omits `train_url`, so the agent is never pointed at a dead port.

`--include-run-history` additionally exposes the `get_run_history` tool
(default: withheld). It returns cross-session results and must never be
offered to benchmark agents — see [TOOL_POLICY.md](TOOL_POLICY.md).

`--enable-code-tool` additionally exposes `run_python_code` (default:
withheld) — the tool behind prior/posterior predictive plots, trace plots and
PSIS-LOO in clients that cannot execute code themselves (Claude Desktop).
Assistant use only; see [TOOL_POLICY.md](TOOL_POLICY.md).

## Connect your client

**Claude Desktop** — stdio only for a local server; see
[Simplest setup](#simplest-setup-claude-desktop-no-server-to-run) above.

**Claude Code** — with the server already running (HTTP):

```bash
claude mcp add --transport http stan http://127.0.0.1:8765/mcp
# with auth:  --header "Authorization: Bearer <token>"
```

**opencode** (`opencode.json`) and other URL-based clients:

```json
{
  "mcp": {
    "stan": { "type": "remote", "url": "http://127.0.0.1:8765/mcp", "enabled": true }
  }
}
```

Verify from the client by calling `get_capabilities` — it returns the tool
list, the server version and the purpose line.

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

### 4. Connect a client

For clients that take a URL (Claude Code, opencode, `.mcp.json`):

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

**Claude Desktop cannot use a URL for a local server** — its config launches a
subprocess over stdio. To reach a remote server from Desktop, have the stdio
command do the hop, e.g. `ssh user@host /path/to/stan-mcp-server --transport
stdio --datasets-dir … --results-dir …`, or run a local stdio↔HTTP bridge.
See *Transports* below for what stdio changes.



## Security

The server runs arbitrary Stan code and accepts dataset uploads; the bearer
token ensures only clients that know the secret can connect. Use it whenever
the ports might be reachable beyond your machine (remote host, SSH tunnel, or
`--host 0.0.0.0`). On a strictly local setup (`127.0.0.1`, no tunnel) it is
optional.

Protect the server with the built-in `--token` flag, or set the environment
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



## Testing

```bash
pytest test_helpers.py test_shadow_isolation.py -k "not runs_shadow"  # fast, no CmdStan
python test_shadow_isolation.py --with-fit    # + real end-to-end fit (~10 s)
python test_server.py --datasets-dir datasets --results-dir /tmp/stan_results
python test_server_http.py                    # against a running server
```

CI runs the fast suite on every push and pull request.
## Where things live on disk

| Path | Contents |
|---|---|
| `<datasets-dir>/benchmarks/<name>/` | staged datasets: `train.csv`, `dataset.md`, `protected/test.csv` |
| `<datasets-dir>/_uploaded/<name>/` | datasets uploaded (or copied) at runtime — train-only unless you stage a test set |
| `<results-dir>/_runs/<run_id>/` | per-run assets: `model.stan`, `logs.txt`, `samples/*.csv` — **never cleaned automatically** |
| `<results-dir>/runs.jsonl` | index over every run: id, time, tool, dataset, status, score, paths |
| `<results-dir>/<dataset>/log.jsonl` | per-dataset history of `fit_and_evaluate` calls |
| `$TMPDIR/stan_mcp_model_cache/` | compiled models, keyed by source hash |

Disk fills quietly: run assets are kept forever and posterior draws dominate
(56 GB over 528 runs on one shared machine). `runs.jsonl` is what makes
cleanup decidable — filter it by date or status, delete the matching
`samples_path` directories.
