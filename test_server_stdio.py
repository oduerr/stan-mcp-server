#!/usr/bin/env python3
"""Stdio-transport tests — the only way Claude Desktop can run this server.

Claude Desktop's `claude_desktop_config.json` launches local MCP servers with
`command`/`args` over stdio; it has no `url` field for local servers. Under
stdio the HTTP sidecar thread is never started, so anything that advertises a
sidecar URL would send the agent to a dead port, and `--token` would promise
an authentication that only exists as HTTP middleware.

Three assertions, run against a real server subprocess:

  1. no tool advertises an HTTP sidecar URL (get_capabilities, get_data_summary)
  2. get_upload_instructions returns the file-copy method instead of a URL
  3. --token is rejected outright rather than silently ignored

Usage:
    pytest test_server_stdio.py
    python test_server_stdio.py
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

TRAIN_CSV = "x,y\n1.0,2.0\n2.0,4.0\n"
DATASET_MD = """# demo

## Data Interface

```stan
int<lower=0> N_train;
vector[N_train] x_train;
vector[N_train] y_train;
```
"""


def _staged_dirs(tmp: Path) -> tuple[Path, Path]:
    ds = tmp / "datasets" / "benchmarks" / "demo"
    ds.mkdir(parents=True)
    (ds / "train.csv").write_text(TRAIN_CSV)
    (ds / "dataset.md").write_text(DATASET_MD)
    res = tmp / "results"
    res.mkdir()
    return tmp / "datasets", res


async def _ask_stdio(datasets: Path, results: Path) -> dict:
    """Start the server over stdio and call the three read-only tools."""
    from fastmcp.client import Client
    from fastmcp.client.transports import StdioTransport

    transport = StdioTransport(
        command=sys.executable,
        args=[str(Path(__file__).parent / "stan_mcp_server" / "server.py"),
              "--transport", "stdio",
              "--datasets-dir", str(datasets),
              "--results-dir", str(results)],
    )
    async with Client(transport) as c:
        tools = sorted(t.name for t in await c.list_tools())
        caps = (await c.call_tool("get_capabilities", {})).data
        summary = (await c.call_tool("get_data_summary",
                                     {"dataset": "benchmarks/demo"})).data
        upload = (await c.call_tool("get_upload_instructions", {})).data
    return {"tools": tools, "caps": caps, "summary": summary, "upload": upload}


def test_stdio_advertises_no_sidecar_url():
    with tempfile.TemporaryDirectory() as tmp:
        r = asyncio.run(_ask_stdio(*_staged_dirs(Path(tmp))))
    # 1. No dead sidecar URLs anywhere in the model-bound responses.
    assert r["caps"]["http_upload_url"] == "disabled", r["caps"]
    assert r["caps"]["train_download_url"] == "disabled", r["caps"]
    assert "train_url" not in r["summary"], r["summary"]
    assert "8766" not in json.dumps(r["caps"]) + json.dumps(r["summary"])
    # Withheld tools stay withheld on this transport too (they once did not).
    assert "get_run_history" not in r["tools"] and "run_python_code" not in r["tools"]


def test_stdio_upload_instructions_use_file_copy():
    with tempfile.TemporaryDirectory() as tmp:
        datasets, results = _staged_dirs(Path(tmp))
        r = asyncio.run(_ask_stdio(datasets, results))
    up = r["upload"]
    # 2. A usable method, not an HTTP URL the agent cannot reach.
    assert up["status"] == "ok" and up["method"] == "file_copy", up
    assert str(datasets) in up["target_dir"]
    assert "_uploaded" in up["target_dir"] and "{name}" in up["target_dir"]
    assert "train.csv" in up["files"] and "dataset.md" in up["files"]
    assert "http://" not in json.dumps(up), up


def test_stdio_rejects_token():
    # 3. --token under stdio must fail loudly, not be silently unenforced.
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "stan_mcp_server" / "server.py"),
         "--transport", "stdio", "--datasets-dir", ".", "--results-dir", ".",
         "--token", "secret123"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0, "server accepted --token under stdio"
    assert "cannot be used with --transport stdio" in proc.stderr
    assert "secret123" not in proc.stderr        # never echo the token


def main() -> None:
    for fn in (test_stdio_advertises_no_sidecar_url,
               test_stdio_upload_instructions_use_file_copy,
               test_stdio_rejects_token):
        print(f"[{fn.__name__}] …")
        fn()
        print("    ok")
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
