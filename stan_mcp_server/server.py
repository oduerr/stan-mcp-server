#!/usr/bin/env python3
"""Stan MCP Server — structured Bayesian modelling tools for LLM agents.

Serves tools over HTTP (streamable-http transport):

    fit_and_evaluate        Sample + compute NLPD; returns run_id + diagnostics + asset paths.
    sample                  Sample + persist draws to disk; returns run_id + diagnostics + asset paths.
    check_model             Compile-only model check (syntax + log_lik presence).
    get_data_summary        Compact EDA for a named dataset.
    get_upload_instructions Return HTTP upload URL and field names for datasets.
    list_datasets           List available datasets on the server.
    get_run_history         Return the logged run history for a dataset.
    get_capabilities        Describe tools and current server configuration.

Run assets (logs, posterior draws) are stored server-side under
<results-dir>/_runs/<run_id>/; tools return their filesystem paths
(`logs_path` / `samples_path`), directly accessible when --results-dir
is mounted via SSHFS on the client.  The HTTP sidecar serves:
    POST /dataset/{name}   — upload train CSV + optional dataset.md (multipart)
    GET  /train/{dataset}  — download train.csv (?file=dataset.md for the md);
                             never serves anything under protected/

Usage
-----
    stan-mcp-server --datasets-dir /path/to/datasets --results-dir /path/to/results
    stan-mcp-server --datasets-dir ./datasets --results-dir ./results --port 8765 --host 0.0.0.0

Expected datasets layout:
    datasets/
      benchmarks/          ← pre-staged benchmark datasets (have protected/test.csv)
        regression_1d/
          train.csv
          dataset.md
          protected/
            test.csv
      _uploaded/           ← agent-uploaded, train-only datasets
"""

import argparse
import asyncio
import contextlib
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import secrets
import shutil
import socket
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastmcp import FastMCP
from scipy.special import logsumexp
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# Single-sourced from pyproject.toml; "0+dev" when running uninstalled.
try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("stan-mcp-server")
except Exception:  # noqa: BLE001 - version is informational only
    _VERSION = "0+dev"

# ── Global path config (set by main() before the server starts) ────────────────
_DATASETS_DIR:  Path = Path("datasets")
_RESULTS_DIR:   Path = Path("results")
_MODEL_CACHE:   Path = Path(tempfile.gettempdir()) / "stan_mcp_model_cache"
_UPLOAD_PORT:   int  = 8766          # 0 = disabled
_UPLOAD_HOST:   str  = "127.0.0.1"
_UPLOAD_DIR:    str  = "_uploaded"
_BEARER_TOKEN:  Optional[str] = None  # set by --token; None = no auth
# Which transport main() started.  In stdio mode the HTTP sidecar thread is
# never started, so no tool may advertise an upload/download URL for it.
_TRANSPORT:     str  = "streamable-http"


class _BearerTokenMiddleware:
    """ASGI middleware — reject requests without the correct Bearer token."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            # Constant-time comparison — the server may be bound to 0.0.0.0.
            if not secrets.compare_digest(auth, f"Bearer {_BEARER_TOKEN}"):
                response = StarletteResponse("Unauthorized", status_code=401)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

# ── Default sampling config ────────────────────────────────────────────────────
_DEFAULT_CONFIG = {
    "chains": 4,
    "iter_warmup": 1000,
    "iter_sampling": 1000,
    "seed": 42,
    # Wall-clock ceiling per fit. A pathological posterior (every HMC step
    # hitting max treedepth) samples forever: a trivial NegBin GLM once ran
    # 57 min at 99.9% CPU with its output frozen. The client cannot defend
    # itself — its `requests` timeout is a READ timeout and the streamable-http
    # transport keeps the connection warm — so the limit has to live here.
    "max_runtime_sec": 900,
}

# ── Shared dataset-save logic (used by both MCP tool and HTTP endpoint) ────────

def _save_dataset(
    name: str,
    train_csv: str,
    dataset_md: Optional[str] = None,
) -> dict:
    """Validate name, write train CSV and optional dataset.md.  Returns a status dict.

    Test data is intentionally NOT accepted here — it must be placed manually
    in <datasets_dir>/<name>/protected/test.csv by the server operator.  This
    ensures that held-out labels never pass through LLM context.
    """
    if not re.fullmatch(r'[A-Za-z0-9_\-]+', name):
        return {
            "status": "error",
            "message": "Dataset name may only contain letters, digits, underscores, and hyphens.",
        }

    ds_dir = _DATASETS_DIR / _UPLOAD_DIR / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    (ds_dir / "train.csv").write_text(train_csv)
    if dataset_md is not None:
        (ds_dir / "dataset.md").write_text(dataset_md)

    try:
        train_cols = _load_csv_columns(ds_dir / "train.csv")
    except Exception as exc:
        return {"status": "error", "message": f"CSV parse error: {exc}"}

    # Validate Data Interface block against train.csv columns (fast-fail at upload).
    interface_warnings: list[str] = []
    if dataset_md is not None:
        interface = _parse_data_interface(dataset_md)
        train_col_set = set(train_cols.keys())
        for base in interface["train_vars"]:
            if base not in train_col_set and f"{base}_train" not in train_col_set:
                interface_warnings.append(
                    f"Data Interface declares '{base}_train' but column '{base}' "
                    f"(or '{base}_train') not found in train.csv. "
                    f"train.csv columns: {sorted(train_col_set)}"
                )

    result = {
        "status": "ok",
        "dataset": f"{_UPLOAD_DIR}/{name}",
        "tier": "uploaded",
        "n_train": len(next(iter(train_cols.values()))),
        "train_columns": list(train_cols.keys()),
        "note": (
            "Uploaded datasets have no held-out test set. "
            f"Use sample(dataset='{_UPLOAD_DIR}/{name}') — the server loads the "
            "train data by name — and compute PSIS-LOO on the training log_lik "
            "yourself. fit_and_evaluate requires a pre-staged dataset with "
            "protected/test.csv."
        ),
    }
    if interface_warnings:
        result["interface_warnings"] = interface_warnings
    return result


# ── HTTP sidecar app (runs on --upload-port in a daemon thread) ────────────────

_upload_app = FastAPI(title="Stan dataset sidecar", docs_url=None, redoc_url=None)


# Bulk data bypasses LLM context in BOTH directions: uploads come in via
# POST /dataset, and train data goes out via this endpoint.  Clients (a coding
# agent's curl, or the benchmark loop at run start) download train.csv to disk;
# only aggregates of it ever enter the model's context.
_SERVABLE_FILES = ("train.csv", "dataset.md")


@_upload_app.get("/train/{dataset:path}")
async def _http_get_train(dataset: str, file: str = "train.csv"):
    """Serve a dataset's train.csv or dataset.md — never anything else.

    This is deliberately NOT a generic file server: the filename is chosen
    from a two-entry whitelist, the dataset name is traversal-checked, and
    a final guard refuses any path that contains 'protected'.  Held-out
    labels (protected/test.csv, protected/shadow.csv) must stay unreachable
    over HTTP — see TOOL_POLICY.md, leak class L1.
    """
    from fastapi.responses import FileResponse  # noqa: PLC0415

    if file not in _SERVABLE_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Only {list(_SERVABLE_FILES)} are served by this endpoint.",
        )
    try:
        ds_dir = _resolve_under(_DATASETS_DIR, dataset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    path = ds_dir / file
    if "protected" in path.parts:  # belt and braces — see docstring
        raise HTTPException(status_code=403, detail="protected/ is never served.")
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail=f"{file} not found for dataset '{dataset}'."
        )
    media_type = "text/csv" if file.endswith(".csv") else "text/markdown"
    return FileResponse(path, media_type=media_type, filename=file)


@_upload_app.post("/dataset/{name}")
async def _http_upload_dataset(
    name: str,
    train: UploadFile = File(..., description="Training CSV (including header row)"),
    dataset_md: Optional[UploadFile] = File(None, description="Optional dataset.md file"),
) -> dict:
    """Upload a training CSV (and optional dataset.md) for a dataset via multipart POST.

    Test data is intentionally not accepted here — it must be placed manually
    in <datasets_dir>/_uploaded/<name>/protected/test.csv by the server operator.
    This ensures held-out labels never pass through the agent or HTTP layer.
    """
    train_csv = (await train.read()).decode()
    dataset_md_str = (await dataset_md.read()).decode() if dataset_md else None
    return _save_dataset(name, train_csv, dataset_md_str)


mcp = FastMCP("stan")

# ── Run helpers ────────────────────────────────────────────────────────────────

def _make_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _run_base_url() -> Optional[str]:
    """Return the HTTP sidecar base URL, or None when it is not reachable.

    None in stdio mode: the client launches the server as a subprocess and no
    sidecar thread runs, so advertising http://…:8766 would send the agent to
    a dead port.
    """
    if _TRANSPORT == "stdio" or not _UPLOAD_PORT:
        return None
    host = _UPLOAD_HOST if _UPLOAD_HOST != "0.0.0.0" else "127.0.0.1"
    return f"http://{host}:{_UPLOAD_PORT}"


@contextlib.contextmanager
def _capture_logs():
    """Capture all cmdstanpy log records into a StringIO buffer."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
    logger = logging.getLogger("cmdstanpy")
    prev_level = logger.level
    logger.addHandler(handler)
    if prev_level == logging.NOTSET or prev_level > logging.DEBUG:
        logger.setLevel(logging.DEBUG)
    try:
        yield buf
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)



def _compute_nlpd(log_lik: np.ndarray) -> float:
    log_mean = logsumexp(log_lik, axis=0) - np.log(log_lik.shape[0])
    return float(-np.mean(log_mean))


def _get_model(stan_code: str):
    import cmdstanpy  # noqa: PLC0415
    _MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    code_hash = hashlib.sha256(stan_code.encode()).hexdigest()[:16]
    model_file = _MODEL_CACHE / f"model_{code_hash}.stan"
    if not model_file.exists() or model_file.read_text() != stan_code:
        model_file.write_text(stan_code)
    return cmdstanpy.CmdStanModel(stan_file=str(model_file))


# Upper bounds on agent-supplied sampling settings.  The wall-clock guard
# eventually stops a runaway fit, but e.g. chains=500 would still spawn 500
# CmdStan processes before it fires — clamp before that can happen.
_CONFIG_CAPS = {"chains": 16, "iter_warmup": 10_000, "iter_sampling": 10_000}


def _merge_config(config: Optional[dict]) -> dict:
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        for k in ("chains", "iter_warmup", "iter_sampling", "seed"):
            if k in config:
                try:
                    v = int(config[k])
                except (TypeError, ValueError):
                    continue  # ignore malformed values, keep the default
                if k in _CONFIG_CAPS:
                    if v < 1:
                        continue
                    v = min(v, _CONFIG_CAPS[k])
                cfg[k] = v
        # The runtime ceiling may only be LOWERED by the caller. Letting an
        # agent raise it would defeat the guard — the models that need
        # stopping are exactly the ones that would ask for more time.
        if "max_runtime_sec" in config:
            try:
                asked = float(config["max_runtime_sec"])
                if asked > 0:
                    cfg["max_runtime_sec"] = min(asked, _DEFAULT_CONFIG["max_runtime_sec"])
            except (TypeError, ValueError):
                pass
    return cfg


def _resolve_under(base: Path, name: str) -> Path:
    """Resolve base/name, refusing names that escape base (path traversal).

    Dataset names are agent-supplied; without this check a name like
    '../../x' would read (or, via the run log, write) outside the
    configured directories.
    """
    root = base.resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError(
            f"Invalid dataset name '{name}': escapes the {base.name} directory."
        )
    return target


def _find_col(columns, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def _make_diagnostics(fit) -> dict:
    n_divergences = int(np.sum(fit.method_variables()["divergent__"]))
    summary = fit.summary()
    mask = ~summary.index.str.startswith("log_lik")
    filtered = summary[mask]
    r_hat_col = _find_col(filtered.columns, "R_hat", "R-hat", "Rhat")
    ess_col = _find_col(filtered.columns, "N_Eff", "ESS_bulk", "ess_bulk")
    r_hat_max = round(float(filtered[r_hat_col].max()), 4) if r_hat_col else float("nan")
    ess_bulk_min = int(filtered[ess_col].min()) if ess_col else -1
    return {"n_divergences": n_divergences, "r_hat_max": r_hat_max, "ess_bulk_min": ess_bulk_min}


def _make_diagnostics_detail(fit) -> dict:
    """Consultation-grade sampler diagnostics (G3, docs/USE_CASES.md UC-4).

    The three core scalars in _make_diagnostics answer "is this run valid";
    this block answers "what went wrong where": per-chain divergences and
    E-BFMI, treedepth saturation, step sizes, and the worst parameters by
    R-hat / bulk ESS — the numbers behind advice like "the funnel is in tau,
    go non-centred".
    """
    mv = fit.method_variables()
    divergent = np.asarray(mv["divergent__"])          # (draws, chains)
    treedepth = np.asarray(mv["treedepth__"])
    energy    = np.asarray(mv["energy__"])
    stepsize  = np.asarray(mv["stepsize__"])
    n_draws, n_chains = divergent.shape

    try:
        max_depth = int(fit.metadata.cmdstan_config.get("max_depth", 10))
    except Exception:
        max_depth = 10
    n_max_td = int(np.sum(treedepth >= max_depth))

    e_bfmi = []
    for c in range(n_chains):
        e = energy[:, c]
        var = float(np.var(e))
        e_bfmi.append(round(float(np.mean(np.diff(e) ** 2)) / var, 3)
                      if var > 0 else float("nan"))

    detail: dict = {
        "divergences_per_chain": [int(v) for v in divergent.sum(axis=0)],
        "n_max_treedepth": n_max_td,
        "max_treedepth_frac": round(n_max_td / divergent.size, 4),
        "max_treedepth_limit": max_depth,
        "e_bfmi_per_chain": e_bfmi,
        "stepsize_per_chain": [round(float(s), 5) for s in stepsize.mean(axis=0)],
    }

    # Worst parameters by R-hat and bulk ESS (log_lik excluded, as everywhere).
    summary = fit.summary()
    filtered = summary[~summary.index.str.startswith("log_lik")]
    r_hat_col = _find_col(filtered.columns, "R_hat", "R-hat", "Rhat")
    ess_col   = _find_col(filtered.columns, "N_Eff", "ESS_bulk", "ess_bulk")
    if r_hat_col:
        worst = filtered[r_hat_col].dropna().sort_values(ascending=False)[:3]
        detail["worst_r_hat"] = [
            {"param": str(k), "r_hat": round(float(v), 4)} for k, v in worst.items()
        ]
    if ess_col:
        low = filtered[ess_col].dropna().sort_values()[:3]
        detail["lowest_ess_bulk"] = [
            {"param": str(k), "ess_bulk": int(v)} for k, v in low.items()
        ]

    detail["flags"] = _diagnostic_flags(detail, n_chains)
    return detail


def _diagnostic_flags(detail: dict, n_chains: int) -> list[str]:
    """Terse, factual warnings derived from the detail block (pure function)."""
    flags: list[str] = []
    div = detail.get("divergences_per_chain", [])
    if sum(div) > 0:
        flags.append(
            f"{sum(div)} divergent transition(s) (per chain: {div}) — biased "
            "exploration; consider a non-centred parameterisation or tighter priors"
        )
    frac = detail.get("max_treedepth_frac", 0)
    if frac > 0.01:
        flags.append(
            f"{frac:.0%} of draws hit max treedepth "
            f"({detail.get('max_treedepth_limit')}) — strong posterior "
            "correlations or an unidentified scale; consider reparameterising"
        )
    low_bfmi = [
        f"chain {i + 1} ({v})"
        for i, v in enumerate(detail.get("e_bfmi_per_chain", []))
        if math.isfinite(v) and v < 0.2
    ]
    if low_bfmi:
        flags.append(
            "low E-BFMI in " + ", ".join(low_bfmi) + " — energy transitions "
            "too small (often a funnel or heavy tails); reparameterise"
        )
    bad_rhat = [
        f"{d['param']} ({d['r_hat']})"
        for d in detail.get("worst_r_hat", [])
        if d["r_hat"] > 1.01
    ]
    if bad_rhat:
        flags.append(
            "R-hat above 1.01 for " + ", ".join(bad_rhat) + " — chains "
            "disagree; run longer or reparameterise"
        )
    low_ess = [
        f"{d['param']} ({d['ess_bulk']})"
        for d in detail.get("lowest_ess_bulk", [])
        if d["ess_bulk"] < 100 * n_chains
    ]
    if low_ess:
        flags.append(
            "bulk ESS below 100 per chain for " + ", ".join(low_ess) +
            " — estimates are noisy; run longer or improve geometry"
        )
    return flags


def _make_param_summary(fit) -> dict:
    result: dict = {}
    for name, draws in fit.stan_variables().items():
        if name == "log_lik":
            continue
        draws = np.asarray(draws)
        if draws.ndim == 1:
            result[name] = {"mean": round(float(np.mean(draws)), 4), "sd": round(float(np.std(draws)), 4)}
        elif draws.ndim == 2:
            total = draws.shape[1]
            for i in range(min(total, 20)):
                result[f"{name}[{i + 1}]"] = {"mean": round(float(np.mean(draws[:, i])), 4), "sd": round(float(np.std(draws[:, i])), 4)}
            if total > 20:
                result[f"{name}[21+]"] = f"{total - 20} more dimensions not shown (total: {total})"
    return result


def _extract_compile_error(exc: Exception) -> str:
    msg = str(exc)
    m = re.search(r'(line \d+[^\n]{0,200})', msg)
    return m.group(1) if m else msg[:500]


def _load_csv_columns(path: Path) -> dict[str, np.ndarray]:
    """Load CSV columns.  Numeric columns become float arrays; columns with any
    non-numeric cell (strings, empty values) become object arrays of strings,
    reported as categorical by _col_stats and rejected with a clear message if
    used as Stan data."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    cols: dict[str, list] = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            cols[k].append(v)
    out: dict[str, np.ndarray] = {}
    for k, vals in cols.items():
        try:
            out[k] = np.array([float(v) for v in vals])
        except (TypeError, ValueError):
            out[k] = np.array([str(v) for v in vals], dtype=object)
    return out


def _col_stats(arr: np.ndarray) -> dict:
    if arr.dtype == object:  # non-numeric column → categorical summary
        levels = sorted({str(v) for v in arr})
        stats: dict = {"type": "categorical", "n_levels": len(levels), "levels": levels[:10]}
        if len(levels) > 10:
            stats["levels_note"] = f"first 10 of {len(levels)} levels shown"
        return stats
    return {
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "sd": round(float(np.std(arr)), 4),
    }


def _find_response_cols(dataset_md: str, columns: list[str]) -> list[str]:
    """Return response column name(s).

    Checks for 'response_cols: col1,col2' (multi-outcome datasets like Bundesliga)
    then falls back to single 'response_col: col', then to the last CSV column.
    """
    m = re.search(r'response_cols:\s*([^\s<][^\n<]*)', dataset_md)
    if m:
        return [c.strip() for c in m.group(1).split(',') if c.strip()]
    m = re.search(r'response_col:\s*(\S+)', dataset_md)
    if m:
        return [m.group(1)]
    return [columns[-1]]


def _find_response_col(dataset_md: str, columns: list[str]) -> str:
    """Return the primary response column name (first of response_cols)."""
    return _find_response_cols(dataset_md, columns)[0]


def _parse_data_interface(dataset_md: str) -> dict:
    """Parse the ## Data Interface Stan block from dataset.md.

    Stan variable base names must match CSV column names exactly
    (base = variable name without the _train/_test suffix).

    Returns dict with keys: train_vars {base: dtype}, has_J, j_var_bases.
    """
    m = re.search(r'## Data Interface.*?```(?:stan)?\n(.*?)```', dataset_md, re.DOTALL)
    if not m:
        return {"train_vars": {}, "has_J": False, "j_var_bases": []}
    block = m.group(1)

    train_vars: dict[str, str] = {}
    j_var_bases: list[str] = []
    has_J = bool(re.search(r'\bint[^;]*\bJ\s*;', block))

    for line in block.splitlines():
        line = line.strip()
        m2 = re.match(
            r'(array\s*\[[^\]]+\]\s+int[^;]*?|vector\s*\[[^\]]+\])\s+(\w+_train)\s*;',
            line,
        )
        if not m2:
            continue
        type_str, var_name = m2.group(1), m2.group(2)
        dtype = "int" if "int" in type_str else "float"
        base = var_name[:-6]  # strip '_train'
        train_vars[base] = dtype
        if re.search(r'upper\s*=\s*J', type_str):
            j_var_bases.append(base)

    return {"train_vars": train_vars, "has_J": has_J, "j_var_bases": j_var_bases}


def _load_dataset(
    dataset: str,
    test_file: str = "test.csv",
    require_test: bool = True,
) -> tuple[dict, list]:
    """Load train (+ test) CSVs for a named dataset into a Stan data dict.

    Reads <datasets_dir>/<dataset>/train.csv and
          <datasets_dir>/<dataset>/protected/<test_file>.
    Variable names are derived from the ## Data Interface block in dataset.md;
    Stan base names must match the CSV column names exactly.

    ``require_test=False`` supports train-only (uploaded) datasets for the
    `sample` tool: when the test file is absent, N_test is 0 and every
    ``*_test`` variable is an empty list — valid Stan data for models that
    declare the test variables, ignored by models that do not.

    ``test_file`` exists so the same loader can build the SHADOW evaluation
    data (protected/shadow.csv) — see the shadow block in fit_and_evaluate.
    """
    ds_dir = _resolve_under(_DATASETS_DIR, dataset)
    train_path = ds_dir / "train.csv"
    test_path  = ds_dir / "protected" / test_file
    md_path    = ds_dir / "dataset.md"

    if not train_path.exists():
        candidates = [
            str(p.parent.relative_to(_DATASETS_DIR))
            for p in _DATASETS_DIR.glob("**/train.csv")
            if _UPLOAD_DIR not in p.parts
        ]
        raise ValueError(f"Dataset '{dataset}' not found. Available: {candidates}")
    has_test = test_path.exists()
    if require_test and not has_test:
        raise ValueError(f"Test file not found at {test_path}")

    train_cols   = _load_csv_columns(train_path)
    test_cols    = _load_csv_columns(test_path) if has_test else {}
    csv_col_names = list(train_cols.keys())

    dataset_md   = md_path.read_text() if md_path.exists() else ""
    response_cols = _find_response_cols(dataset_md, csv_col_names)
    response_col  = response_cols[0]  # primary; used in csv_to_base fallback

    interface   = _parse_data_interface(dataset_md)
    train_vars  = interface["train_vars"]
    has_J       = interface["has_J"]
    j_var_bases = interface["j_var_bases"]

    _RESPONSE_ALIASES = {"y", "response", "target", "outcome", "effect"}
    csv_to_base: dict[str, str] = {}

    for csv_col in csv_col_names:
        if csv_col in train_vars:
            csv_to_base[csv_col] = csv_col
        elif csv_col.endswith("_id") and csv_col[:-3] in train_vars:
            csv_to_base[csv_col] = csv_col[:-3]
        else:
            response_candidates = [b for b in train_vars if b in _RESPONSE_ALIASES]
            if csv_col == response_col and response_candidates:
                csv_to_base[csv_col] = response_candidates[0]

    n_train = len(next(iter(train_cols.values())))
    n_test  = len(next(iter(test_cols.values()))) if test_cols else 0
    data: dict = {"N_train": n_train, "N_test": n_test}

    test_csv_cols: dict[str, str] = {}   # train CSV column -> test CSV column
    for csv_col, stan_base in csv_to_base.items():
        if train_cols[csv_col].dtype == object:
            raise ValueError(
                f"Column '{csv_col}' contains non-numeric values and cannot be "
                "passed to Stan. Encode it as integers (e.g. 1-based group ids) "
                "in the CSV first."
            )
        dtype = train_vars.get(stan_base, "float")
        conv = int if dtype == "int" else float
        data[f"{stan_base}_train"] = [conv(v) for v in train_cols[csv_col]]

        if not has_test:
            data[f"{stan_base}_test"] = []
            continue
        # Resolve test column: standard convention uses same name as train CSV;
        # intuitive convention uses {base}_test in test.csv.
        if csv_col in test_cols:
            test_csv_col = csv_col
        elif f"{stan_base}_test" in test_cols:
            test_csv_col = f"{stan_base}_test"
        elif stan_base in test_cols:
            test_csv_col = stan_base
        else:
            raise ValueError(
                f"Column '{csv_col}' (or '{stan_base}_test') missing from test.csv. "
                f"test.csv columns: {list(test_cols.keys())}"
            )
        if test_cols[test_csv_col].dtype == object:
            raise ValueError(
                f"Column '{test_csv_col}' contains non-numeric values and cannot "
                "be passed to Stan. Encode it as integers (e.g. 1-based group "
                "ids) in the CSV first."
            )
        test_csv_cols[csv_col] = test_csv_col
        data[f"{stan_base}_test"] = [conv(v) for v in test_cols[test_csv_col]]

    if has_J and j_var_bases:
        j_csv_cols = [c for c, b in csv_to_base.items() if b in j_var_bases]
        all_ids: set[int] = set()
        for c in j_csv_cols:
            all_ids.update(int(v) for v in train_cols[c])
            if has_test:
                all_ids.update(int(v) for v in test_cols[test_csv_cols[c]])
        if all_ids and min(all_ids) < 1:
            raise ValueError(
                f"Group id columns {j_csv_cols} must be 1-based for Stan "
                f"indexing (smallest id found: {min(all_ids)})."
            )
        # max, not len: models declare int<lower=1, upper=J> and index up to
        # the largest id, so non-contiguous ids (e.g. {1, 5, 9}) would make
        # len(all_ids) too small and crash the fit with an index error.
        data["J"] = max(all_ids) if all_ids else 0

    # Resolve response columns in test.csv using same fallback as above.
    resolved_test_response: list[np.ndarray] = []
    for c in response_cols:
        if c in test_cols:
            resolved_test_response.append(test_cols[c])
        elif f"{c}_test" in test_cols:
            resolved_test_response.append(test_cols[f"{c}_test"])
    y_test = [v for arr in resolved_test_response for v in arr.tolist()]
    return data, y_test

def _read_log(dataset: str) -> list[dict]:
    log_path = _resolve_under(_RESULTS_DIR, dataset) / "log.jsonl"
    if not log_path.exists():
        return []
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _append_log(dataset: str, entry: dict) -> None:
    log_path = _resolve_under(_RESULTS_DIR, dataset) / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _append_run_index(entry: dict) -> None:
    """Append one line to <results-dir>/runs.jsonl — the index over _runs/.

    OPERATOR-FACING ONLY.  No MCP tool reads this file: it spans datasets and
    sessions, which is the L2 leak class get_run_history is withheld for (see
    TOOL_POLICY.md).  Exposing it would need its own policy row.

    Written by every tool that creates a run directory, including `sample`,
    which logs nowhere else — without this, assistant-mode runs leave opaque
    hex directories with no record of what they were.  JSONL, like the
    per-dataset logs: free-text fields, tolerant of a torn line, and new keys
    do not break old readers.

    Never raises: an index failure must not fail a fit that already succeeded.
    """
    try:
        path = _RESULTS_DIR / "runs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:          # noqa: BLE001
        logging.warning("could not write run index: %s", exc)


def _run_index_entry(tool: str, run_id: str, run_dir: Path, *, dataset=None,
                     status: str, runtime_sec=None, nlpd=None, n_draws=None) -> dict:
    """Build the index line.  Timestamps are UTC ISO-8601 (sortable)."""
    from datetime import datetime, timezone  # noqa: PLC0415
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": tool,
        "dataset": dataset,
        "status": status,
        "runtime_sec": runtime_sec,
        "nlpd": nlpd,                       # None for `sample` — no held-out score
        "n_draws": n_draws,
        "machine": socket.gethostname(),
        "logs_path": str(run_dir / "logs.txt"),
        "samples_path": str(run_dir),
    }


# ── Shadow containment ─────────────────────────────────────────────────────────
# `shadow_nlpd` is written into log.jsonl on purpose — the shadow measurement is
# useless if it isn't recorded — but it must never reach the model.  The moment
# the agent can see it, the shadow set becomes a second feedback set and stops
# measuring anything.
#
# So every tool that surfaces log entries has to scrub them first.  get_run_history
# returns whole entries verbatim and did leak shadow_nlpd for the entire life of
# the shadow branch; that is the same tool as the 2026-07-30 incident, and the
# same failure mode: a path nobody thought of as an output.  Scrub at the exit,
# not at each call site, so a new tool that reads the log is safe by default.
#
# Matched on the key *name* — anything containing "shadow", at any depth — rather
# than on the one key known today.  An exact-match filter silently stops covering
# shadow_ess, shadow_n, or a nested shadow block the day someone adds one.
_SHADOW_KEY_RE = re.compile(r"shadow", re.IGNORECASE)


def _scrub_shadow(obj):
    """Recursively drop every shadow-named key from a structure bound for the model."""
    if isinstance(obj, dict):
        return {
            k: _scrub_shadow(v)
            for k, v in obj.items()
            if not _SHADOW_KEY_RE.search(str(k))
        }
    if isinstance(obj, list):
        return [_scrub_shadow(v) for v in obj]
    return obj


# ── Shared compile → sample → persist path ─────────────────────────────────────
# Used by both fit_and_evaluate and sample so the timeout handling, log capture
# and asset layout cannot drift apart (they were copy-pasted once and had to be
# patched in sync).

_TIMEOUT_MESSAGE = (
    "Sampling exceeded {limit}s and was stopped. "
    "This usually means the posterior geometry is pathological "
    "(e.g. a latent GP over many points, or an unidentified "
    "scale parameter), not that the model is merely large. "
    "Try a lower-dimensional parameterisation — a basis "
    "expansion instead of a full GP, tighter priors on scales, "
    "or non-centred parameterisation."
)


def _compile_and_sample(stan_code: str, data: dict, config: Optional[dict],
                        tool: str = "sample", dataset: Optional[str] = None) -> dict:
    """Compile, sample with the wall-clock guard, and persist run assets.

    Returns {"error": <result dict ready to return to the model>} on failure, or
    {"model", "fit", "run_id", "run_dir", "runtime_sec", "cfg"} on success.

    ``tool`` / ``dataset`` are recorded in the run index (runs.jsonl) — failures
    that got as far as creating a run directory are indexed too, so no directory
    under _runs/ is ever anonymous.
    """
    try:
        model = _get_model(stan_code)
    except Exception as exc:
        return {"error": {"status": "error", "stage": "compilation",
                          "message": _extract_compile_error(exc)}}

    run_id = _make_run_id()
    run_dir = _RESULTS_DIR / "_runs" / run_id
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.stan").write_text(stan_code)

    cfg = _merge_config(config)
    t0 = time.time()
    with _capture_logs() as log_buf:
        try:
            fit = model.sample(
                data=data,
                chains=cfg["chains"],
                iter_warmup=cfg["iter_warmup"],
                iter_sampling=cfg["iter_sampling"],
                seed=cfg["seed"],
                show_progress=False,
                show_console=False,
                output_dir=str(samples_dir),
                timeout=cfg["max_runtime_sec"],
            )
        except TimeoutError:
            (run_dir / "logs.txt").write_text(log_buf.getvalue())
            # Returned as a normal error, not raised: the agent should learn
            # "too slow, simplify" and keep iterating.
            _append_run_index(_run_index_entry(
                tool, run_id, run_dir, dataset=dataset, status="sampling_timeout",
                runtime_sec=cfg["max_runtime_sec"]))
            return {"error": {
                "status": "error",
                "stage": "sampling_timeout",
                "message": _TIMEOUT_MESSAGE.format(limit=cfg["max_runtime_sec"]),
                "run_id": run_id,
                "runtime_sec": cfg["max_runtime_sec"],
            }}
        except Exception as exc:
            (run_dir / "logs.txt").write_text(log_buf.getvalue())
            _append_run_index(_run_index_entry(
                tool, run_id, run_dir, dataset=dataset, status="sampling_error",
                runtime_sec=round(time.time() - t0, 1)))
            return {"error": {"status": "error", "stage": "sampling",
                              "message": str(exc)[:500]}}
    (run_dir / "logs.txt").write_text(log_buf.getvalue())

    return {
        "model": model,
        "fit": fit,
        "run_id": run_id,
        "run_dir": run_dir,
        "runtime_sec": round(time.time() - t0, 1),
        "cfg": cfg,
    }


def _safe_fit_summaries(fit) -> tuple[dict, dict, dict]:
    """Diagnostics + parameter summary + detail; degrade rather than fail a run."""
    try:
        diag = _make_diagnostics(fit)
    except Exception:
        diag = {"n_divergences": -1, "r_hat_max": float("nan"), "ess_bulk_min": -1}
    try:
        param_summary = _make_param_summary(fit)
    except Exception:
        param_summary = {}
    try:
        detail = _make_diagnostics_detail(fit)
    except Exception:
        detail = {}
    return diag, param_summary, detail


# ── Tool: check_model ──────────────────────────────────────────────────────────

@mcp.tool()
def check_model(stan_code: str) -> dict:
    """Compile a Stan model and verify it declares a log_lik output vector.

    Returns the declared length expression so the agent can confirm it uses
    N_test (not N_train), catching a silent but common bug.
    """
    ll_match = re.search(
        r'(?:vector|array)\[([^\]]+)\]\s+(?:real\s+)?log_lik\b',
        stan_code,
    )
    if not ll_match:
        return {
            "status": "error",
            "stage": "missing_log_lik",
            "message": "no 'log_lik' vector found in generated quantities — required for NLPD",
        }
    log_lik_length_expr = ll_match.group(1).strip()

    try:
        _get_model(stan_code)
    except Exception as exc:
        return {"status": "error", "stage": "compilation", "message": _extract_compile_error(exc)}

    return {"status": "ok", "log_lik_length_expr": log_lik_length_expr}


# ── Tool: fit_and_evaluate ─────────────────────────────────────────────────────

@mcp.tool()
def fit_and_evaluate(
    stan_code: str,
    data: Optional[dict] = None,
    y_test: Optional[list] = None,
    config: Optional[dict] = None,
    notes: Optional[str] = None,
    rationale: Optional[str] = None,
    dataset: Optional[str] = None,
) -> dict:
    """Sample from a Stan model and compute NLPD on held-out test responses.

    The Stan model must output a `log_lik` vector of length N_test in
    generated quantities.

    When `dataset` is provided and `data`/`y_test` are omitted, data are
    loaded automatically from <datasets_dir>/<dataset>/train.csv and
    <datasets_dir>/<dataset>/protected/test.csv.

    NOTE — N_train and N_test are injected automatically from the CSV row
    counts.  Only pass `data` when you need to override them or supply
    additional scalars the CSV does not provide.

    This tool only works for pre-staged datasets that have a protected test
    set (protected/test.csv placed by the server operator).  For uploaded
    (train-only) datasets use `sample` instead and compute PSIS-LOO yourself
    on the training log_lik.

    When `dataset` is provided the result is appended to
    <results_dir>/<dataset>/log.jsonl (with or without notes/rationale).

    Returns scalar diagnostics and NLPD inline.  Posterior draws and CmdStan
    logs are stored under a `run_id` and their filesystem paths are returned
    as `logs_path` / `samples_path`.  When --results-dir is mounted via SSHFS
    on the client these paths are directly accessible.  Bulk data never enters
    LLM context.
    """
    # Treat empty dict/list sent by LLM tool-callers the same as None
    if not data:
        data = None
    if not y_test:
        y_test = None

    # Validate the agent-supplied dataset name once, up front — it is used for
    # reads (train/test CSVs) and writes (the run log) further down.
    if dataset is not None:
        try:
            ds_dir = _resolve_under(_DATASETS_DIR, dataset)
        except ValueError as exc:
            return {"status": "error", "stage": "input", "message": str(exc)}

    if dataset is not None:
        # Reject train-only (uploaded) datasets — no held-out test set exists.
        test_path = ds_dir / "protected" / "test.csv"
        if not test_path.exists():
            return {
                "status": "error",
                "stage": "input",
                "message": (
                    f"Dataset '{dataset}' has no held-out test set (protected/test.csv). "
                    "Uploaded datasets are train-only. Use 'sample' instead and compute "
                    "PSIS-LOO on the training log_lik yourself."
                ),
            }
        try:
            loaded, loaded_y_test = _load_dataset(dataset)
        except ValueError as exc:
            return {"status": "error", "stage": "data_loading", "message": str(exc)}
        # `data` entries override / extend the loaded dict — the documented
        # contract ("only pass data for extra scalars"); previously passing
        # data made the loader silently skip the dataset entirely.
        if data:
            loaded.update(data)
        data = loaded
        if y_test is None:
            y_test = loaded_y_test
    elif data is None:
        return {"status": "error", "stage": "input", "message": "Either 'data' (with 'y_test') or 'dataset' must be provided."}

    # y_test is optional in explicit-data mode: it is only used for the
    # log_lik shape check. NLPD is computed from log_lik alone. When omitted,
    # the shape check is skipped and the user is responsible for correctness.

    if not re.search(r'\blog_lik\b', stan_code):
        return {"status": "error", "stage": "missing_log_lik", "message": "no 'log_lik' found in stan_code — required for NLPD computation"}

    run = _compile_and_sample(stan_code, data, config,
                              tool="fit_and_evaluate", dataset=dataset)
    if "error" in run:
        return run["error"]
    model, fit = run["model"], run["fit"]
    run_id, run_dir, runtime_sec = run["run_id"], run["run_dir"], run["runtime_sec"]

    all_vars = fit.stan_variables()
    if "log_lik" not in all_vars:
        _append_run_index(_run_index_entry(
            "fit_and_evaluate", run_id, run_dir, dataset=dataset,
            status="missing_log_lik", runtime_sec=runtime_sec))
        return {"status": "error", "stage": "missing_log_lik", "message": "'log_lik' not found in generated quantities output"}

    log_lik = np.asarray(all_vars["log_lik"])
    if log_lik.ndim == 1:
        log_lik = log_lik[:, np.newaxis]

    n_test = len(y_test) if y_test is not None else log_lik.shape[1]
    if y_test is not None and log_lik.shape[1] != n_test:
        _append_run_index(_run_index_entry(
            "fit_and_evaluate", run_id, run_dir, dataset=dataset,
            status="log_lik_shape_mismatch", runtime_sec=runtime_sec))
        return {"status": "error", "stage": "missing_log_lik", "message": f"log_lik has {log_lik.shape[1]} columns but y_test has {n_test} elements"}

    nlpd = _compute_nlpd(log_lik)

    diag, param_summary, sampler_detail = _safe_fit_summaries(fit)

    # ── Diagnostics validity gate ──────────────────────────────────────────────
    # A result is only valid when all diagnostics are finite/non-sentinel and
    # NLPD itself is finite.  An invalid result must never be accepted as a
    # best-model improvement by the calling loop.
    _diag_reasons: list[str] = []
    if not math.isfinite(nlpd):
        _diag_reasons.append(f"nlpd is not finite ({nlpd})")
    if diag["n_divergences"] < 0:
        _diag_reasons.append("n_divergences sentinel (-1): diagnostic extraction failed")
    if not math.isfinite(diag["r_hat_max"]):
        _diag_reasons.append(f"r_hat_max is not finite ({diag['r_hat_max']})")
    if diag["ess_bulk_min"] < 0:
        _diag_reasons.append(f"ess_bulk_min sentinel ({diag['ess_bulk_min']}): diagnostic extraction failed")
    diagnostics_valid = len(_diag_reasons) == 0
    result_status = "ok" if diagnostics_valid else "invalid"

    result: dict = {
        "status": result_status,
        "diagnostics_valid": diagnostics_valid,
        "run_id": run_id,
        "nlpd": round(nlpd, 4) if math.isfinite(nlpd) else None,
        "n_divergences": diag["n_divergences"],
        "r_hat_max": diag["r_hat_max"],
        "ess_bulk_min": diag["ess_bulk_min"],
        "runtime_sec": runtime_sec,
        "param_summary": param_summary,
        "sampler_diagnostics": sampler_detail,
        "data_keys_loaded": sorted(data.keys()),
        "logs_path":    str(run_dir / "logs.txt"),
        "samples_path": str(run_dir),
    }
    if not diagnostics_valid:
        result["invalid_reasons"] = _diag_reasons

    # ── SHADOW evaluation ─────────────────────────────────────────────────────
    # A second held-out set that the agent never receives feedback on.  The gap
    # between `nlpd` (which the agent optimises against, iteration after
    # iteration) and `shadow_nlpd` measures test-set selection bias — how much
    # of a reported improvement is real generalisation versus fitting the noise
    # in the one fixed feedback sample.
    #
    # Cost is negligible: the posterior draws already exist, so this is a
    # standalone generated-quantities pass (~4 s for 5000 points vs ~75 s for
    # the fit).
    #
    # !! shadow_nlpd goes into the SERVER-SIDE LOG ONLY.  It must never enter
    # !! `result`, because `result` is returned to the model.  Leaking it would
    # !! turn the shadow set into a second feedback set and silently destroy the
    # !! measurement — see the get_run_history incident (2026-07-30), where an
    # !! unlisted tool call exposed cross-session history for months' worth of
    # !! runs before anyone noticed.
    #
    # The GQ output is written to a scratch directory and deleted immediately:
    # log_lik for a 5000-point shadow set across 4000 draws is a ~150 MB CSV,
    # and cmdstanpy keeps its temp files for the lifetime of the process — in a
    # long-lived server that fills the disk within hours.
    #
    # Free-space guard: a model whose generated-quantities block emits more
    # than log_lik (y_rep, mu, …) produces N_shadow x draws numbers PER
    # variable.  One such model wrote a 14 GB CSV on 2026-07-31 and filled the
    # disk, which blocked every in-flight run.  Skip the shadow pass rather
    # than risk the machine; a missing shadow_nlpd is recoverable, a jammed
    # server is not.
    shadow_nlpd = None
    # Threshold is sized against the worst case observed: ~14 GB at 5000
    # shadow points, hence the shadow sets were cut to 2000 (worst case ~6 GB).
    _free_gb = shutil.disk_usage(str(_RESULTS_DIR)).free / 2**30
    if _free_gb < 10:
        logging.warning("skipping shadow pass: only %.1f GB free", _free_gb)
    elif dataset is not None and (ds_dir / "protected" / "shadow.csv").exists():
        try:
            shadow_data, _ = _load_dataset(dataset, test_file="shadow.csv")
            with tempfile.TemporaryDirectory(prefix="shadow_gq_") as gq_dir:
                gq = model.generate_quantities(
                    data=shadow_data, previous_fit=fit, gq_output_dir=gq_dir,
                    # 4 sig figs instead of CmdStan's 6. Measured on iter_012
                    # against the 2000-point shadow set: identical NLPD to six
                    # decimals (2.059894 either way), output 68.8 -> 53.5 MB.
                    # Worth having because the PEAK during the pass is what
                    # filled the disk, not the leftovers — those are already
                    # dropped with the TemporaryDirectory below.
                    sig_figs=4,
                )
                shadow_ll = np.asarray(gq.stan_variable("log_lik"))
                if shadow_ll.ndim == 1:
                    shadow_ll = shadow_ll[:, np.newaxis]
                s_nlpd = _compute_nlpd(shadow_ll)
                shadow_nlpd = round(s_nlpd, 4) if math.isfinite(s_nlpd) else None
        except Exception:
            shadow_nlpd = None      # never fail an evaluation over the shadow pass

    _append_run_index(_run_index_entry(
        "fit_and_evaluate", run_id, run_dir, dataset=dataset, status=result_status,
        runtime_sec=runtime_sec, nlpd=result["nlpd"],
        n_draws=run["cfg"]["chains"] * run["cfg"]["iter_sampling"]))

    if dataset is not None:
        existing = _read_log(dataset)
        iter_num = len(existing)
        # Only consider an iteration as improved when diagnostics are fully valid.
        prior_valid_nlpds = [e["nlpd"] for e in existing if e.get("nlpd") is not None and e.get("diagnostics_valid", True)]
        if not diagnostics_valid:
            improved = False
        elif iter_num == 0 or not prior_valid_nlpds:
            improved = None
        else:
            improved = bool(nlpd < min(prior_valid_nlpds))
        _append_log(dataset, {
            "iter": iter_num,
            "run_id": run_id,
            "nlpd": round(nlpd, 4) if math.isfinite(nlpd) else None,
            "shadow_nlpd": shadow_nlpd,     # server-side only — never returned to the model
            "diagnostics_valid": diagnostics_valid,
            "improved": improved,
            "machine": socket.gethostname(),
            "runtime_sec": runtime_sec,
            "n_divergences": diag["n_divergences"],
            "r_hat_max": diag["r_hat_max"],
            "notes": notes or "",
            "rationale": rationale or "",
        })

    return result


# ── Tool: sample ───────────────────────────────────────────────────────────────

@mcp.tool()
def sample(
    stan_code: str,
    data: Optional[dict] = None,
    config: Optional[dict] = None,
    dataset: Optional[str] = None,
) -> dict:
    """Sample from a Stan model and persist posterior draws to disk.

    Provide `dataset` (preferred) and/or `data`:

    - `dataset` loads the data by name from the server — this works for
      train-only (uploaded) datasets too: N_train and the `*_train` variables
      come from train.csv; when no test set exists, N_test is 0 and the
      `*_test` variables are empty.  Never paste CSV contents into `data` —
      use the upload endpoint and pass the dataset name instead.
    - `data` entries override / extend the loaded dict (extra scalars the CSV
      does not provide).  Passing `data` alone (no `dataset`) uses it as-is.

    Returns scalar diagnostics and a `run_id` only — raw draws are never
    returned inline.  Retrieve them via `samples_path` (directory of per-chain
    Stan CSVs).  CmdStan logs are available at `logs_path`.  Both paths are
    under --results-dir and are directly accessible when that directory is
    mounted via SSHFS on the client.
    """
    if not data:
        data = None
    if dataset is not None:
        try:
            loaded, _ = _load_dataset(dataset, require_test=False)
        except ValueError as exc:
            return {"status": "error", "stage": "data_loading", "message": str(exc)}
        if data:
            loaded.update(data)
        data = loaded
    if data is None:
        return {"status": "error", "stage": "input",
                "message": "Either 'dataset' or 'data' must be provided."}

    run = _compile_and_sample(stan_code, data, config,
                              tool="sample", dataset=dataset)
    if "error" in run:
        return run["error"]
    fit, cfg, run_dir = run["fit"], run["cfg"], run["run_dir"]

    diag, param_summary, sampler_detail = _safe_fit_summaries(fit)

    _append_run_index(_run_index_entry(
        "sample", run["run_id"], run_dir, dataset=dataset, status="ok",
        runtime_sec=run["runtime_sec"], n_draws=cfg["chains"] * cfg["iter_sampling"]))

    return {
        "status": "ok",
        "run_id": run["run_id"],
        "n_samples": cfg["chains"] * cfg["iter_sampling"],
        "runtime_sec": run["runtime_sec"],
        "diagnostics": diag,
        "sampler_diagnostics": sampler_detail,
        "param_summary": param_summary,
        "data_keys_loaded": sorted(data.keys()),
        "logs_path":    str(run_dir / "logs.txt"),
        "samples_path": str(run_dir),
    }


# ── Tool: get_data_summary ─────────────────────────────────────────────────────

@mcp.tool()
def get_data_summary(dataset: str) -> dict:
    """Return a compact EDA summary for a named dataset.

    Reads <datasets_dir>/<dataset>/train.csv and dataset.md.
    The response column of the test set is not exposed (held-out integrity).

    The `tier` field signals whether fit_and_evaluate is available:
      - "staged"   : has a protected test set; fit_and_evaluate works.
      - "uploaded" : train-only; use sample + PSIS-LOO instead.

    N_train and N_test are injected automatically from the CSV row counts
    when fit_and_evaluate loads the dataset.  Only pass the `data` parameter
    when you need to override them or supply additional scalars.
    Also check that dataset_md contains a ## Data Interface Stan block;
    without it no CSV columns will be loaded during sampling.
    """
    try:
        ds_dir = _resolve_under(_DATASETS_DIR, dataset)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    train_path = ds_dir / "train.csv"
    md_path    = ds_dir / "dataset.md"
    test_path  = ds_dir / "protected" / "test.csv"

    if not train_path.exists():
        candidates = [
            str(p.parent.relative_to(_DATASETS_DIR))
            for p in _DATASETS_DIR.glob("**/train.csv")
            if _UPLOAD_DIR not in p.parts
        ]
        return {"status": "error", "message": f"Dataset '{dataset}' not found. Available: {candidates}"}

    train_cols = _load_csv_columns(train_path)
    dataset_md = md_path.read_text() if md_path.exists() else ""
    response_col = _find_response_col(dataset_md, list(train_cols.keys()))

    has_test = test_path.exists()
    tier = "staged" if has_test else "uploaded"

    n_train = len(next(iter(train_cols.values())))
    n_test: Optional[int] = None
    if has_test:
        test_cols = _load_csv_columns(test_path)
        n_test = len(next(iter(test_cols.values())))

    result = {
        "dataset": dataset,
        "tier": tier,
        "has_test": has_test,
        "n_train": n_train,
        "n_test": n_test,
        "columns": {col: _col_stats(arr) for col, arr in train_cols.items()},
        "dataset_md": dataset_md,
    }
    # Direct download URL for the train CSV (never test/shadow data).  Meant
    # for clients that run code locally: fetch to disk, compute aggregates
    # there, and keep raw rows out of LLM context.
    base_url = _run_base_url()
    if base_url:
        result["train_url"] = f"{base_url}/train/{dataset}"
    return result


# ── Tool: get_upload_instructions ─────────────────────────────────────────────

@mcp.tool()
def get_upload_instructions() -> dict:
    """Return instructions for uploading datasets directly to the server via HTTP.

    Datasets must be uploaded via the HTTP endpoint (POST /dataset/{name}) so
    that CSV content — including test labels — never passes through LLM context.
    Call this tool to get the URL and field names to pass to the user or client.

    After uploading, the `dataset_md` field (or a separate `dataset.md` file)
    MUST contain a `## Data Interface` Stan block declaring the `_train`
    variables (e.g. `vector[N_train] x_train;`).  Without this block no CSV
    columns will be loaded and sampling will silently use an empty data dict.

    N_train and N_test ARE injected automatically from the CSV row counts when
    a dataset is loaded by name (same as for pre-staged datasets).  Only pass
    the `data` parameter to override them or to supply additional scalars the
    CSV does not provide.
    """
    if _TRANSPORT == "stdio":
        # No sidecar runs under stdio — but the server is a child process of
        # the client, i.e. on the same machine, so copying the files in is the
        # natural equivalent and keeps CSV content out of LLM context just as
        # the HTTP upload does.
        target = _DATASETS_DIR / _UPLOAD_DIR / "{name}"
        return {
            "status": "ok",
            "method": "file_copy",
            "reason": (
                "This server runs over stdio (no HTTP endpoint). Copy the files "
                "into the datasets directory instead — never paste CSV contents "
                "into a tool argument."
            ),
            "target_dir": str(target),
            "files": {
                "train.csv": "required — training data, including the header row",
                "dataset.md": (
                    "required in practice — must contain a '## Data Interface' "
                    "Stan block declaring the _train variables, else no CSV "
                    "columns are loaded"
                ),
            },
            "example_shell": (
                f"mkdir -p {target} && cp train.csv dataset.md {target}/"
            ),
            "note": (
                "Afterwards the dataset name is '{upload}/<name>' — pass it to "
                "get_data_summary / sample. Held-out test data must be placed by "
                "the operator at <target_dir>/protected/test.csv to enable "
                "fit_and_evaluate; the agent must never read that file."
            ).replace("{upload}", _UPLOAD_DIR),
        }
    if not _UPLOAD_PORT:
        return {
            "status": "disabled",
            "message": "The HTTP upload endpoint is disabled on this server (--upload-port 0).",
        }
    host_display = _UPLOAD_HOST if _UPLOAD_HOST != "0.0.0.0" else "<server-address>"
    base_url = f"http://{host_display}:{_UPLOAD_PORT}"
    return {
        "status": "ok",
        "upload_url_template": f"{base_url}/dataset/{{name}}",
        "method": "POST",
        "content_type": "multipart/form-data",
        "fields": {
            "train":      "required — CSV file (training data, must include header row)",
            "dataset_md": "optional — dataset.md file with ## Data Interface block for variable annotations",
        },
        "example_curl": (
            f"curl -X POST {base_url}/dataset/my_experiment "
            "-F train=@train.csv -F dataset_md=@dataset.md"
        ),
        "note": (
            "Test data is NOT accepted here — place it manually at "
            "<datasets_dir>/_uploaded/<name>/protected/test.csv to enable fit_and_evaluate. "
            "Train-only (uploaded) datasets support only the 'sample' tool; "
            "use PSIS-LOO on the training log_lik for model comparison. "
            "After a successful upload the qualified dataset name is '_uploaded/<name>', "
            "e.g. '_uploaded/my_experiment'.  Pass this to sample / get_data_summary."
        ),
    }


# ── Tool: list_datasets ──────────────────────────────────────────────────────

@mcp.tool()
def list_datasets() -> dict:
    """List all available datasets on the server.

    Returns two lists:
      - datasets : benchmark datasets under --datasets-dir/benchmarks/
                   (these have a protected test set and support fit_and_evaluate)
      - uploaded : datasets pushed via the HTTP upload endpoint (train-only;
                   use sample + PSIS-LOO, not fit_and_evaluate)

    Dataset names for benchmarks are relative paths from --datasets-dir,
    e.g. 'benchmarks/regression_1d'. Pass this full name to fit_and_evaluate.
    """
    top_level = sorted(
        str(p.parent.relative_to(_DATASETS_DIR))
        for p in _DATASETS_DIR.glob("**/train.csv")
        if _UPLOAD_DIR not in p.parts
    )
    uploaded_dir = _DATASETS_DIR / _UPLOAD_DIR
    uploaded = sorted(
        f"{_UPLOAD_DIR}/{p.parent.name}"
        for p in uploaded_dir.glob("*/train.csv")
    ) if uploaded_dir.exists() else []
    # Annotate each uploaded dataset with its current tier (may be "staged" if
    # the user has since manually placed protected/test.csv).
    uploaded_tiers = {
        name: ("staged" if (_DATASETS_DIR / name / "protected" / "test.csv").exists() else "uploaded")
        for name in uploaded
    }
    return {"datasets": top_level, "uploaded": uploaded, "uploaded_tiers": uploaded_tiers}


# ── Tool: get_run_history ─────────────────────────────────────────────────────

@mcp.tool()
def get_run_history(dataset: str) -> dict:
    """Return the full logged run history for a dataset.

    Reads <results_dir>/<dataset>/log.jsonl and returns all entries in
    chronological order.  Also surfaces the best NLPD seen so far, making
    it easy for the agent to decide whether a new model improved.
    """
    # Scrubbed, not raw: log entries carry shadow_nlpd, which the agent must
    # never see (see _scrub_shadow).  The feedback `nlpd` survives untouched.
    try:
        entries = [_scrub_shadow(e) for e in _read_log(dataset)]
    except ValueError as exc:  # path traversal in the dataset name
        return {"status": "error", "message": str(exc)}
    if not entries:
        return {"dataset": dataset, "n_entries": 0, "best_nlpd": None, "entries": []}
    nlpds = [e["nlpd"] for e in entries if "nlpd" in e]
    return {
        "dataset": dataset,
        "n_entries": len(entries),
        "best_nlpd": round(min(nlpds), 4) if nlpds else None,
        "entries": entries,
    }


# ── Tool: get_capabilities ────────────────────────────────────────────────────

def _registered_tool_names() -> list[str]:
    """Tool names currently registered with FastMCP.

    Derived from the live registry — never a hand-maintained list — so a tool
    withheld at startup (get_run_history without --include-run-history) is not
    advertised here.  Incident 1's lesson (TOOL_POLICY.md) applies to tool
    listings too: naming a withheld tool teaches the model it exists.
    """
    names: list[str] = []

    def _collect() -> None:  # list_tools is async; run it on a private loop
        names.extend(t.name for t in asyncio.run(mcp.list_tools(run_middleware=False)))

    t = threading.Thread(target=_collect)
    t.start()
    t.join()
    return sorted(names)


@mcp.tool()
def get_capabilities() -> dict:
    """Return server capabilities, available tools, and current configuration.

    Call this first to understand what the server can do and how it is
    configured before issuing other tool calls.
    """
    base_url = _run_base_url()
    upload_url = f"{base_url}/dataset/{{name}}" if base_url else "disabled"
    train_url  = f"{base_url}/train/{{dataset}}" if base_url else "disabled"
    return {
        "server": "stan-mcp-server",
        "version": _VERSION,
        "tools": _registered_tool_names(),
        "default_sampling_config": _DEFAULT_CONFIG,
        "log_lik_contract": (
            "Every model used with fit_and_evaluate must declare "
            "'vector[N_test] log_lik' in generated quantities."
        ),
        "bulk_data_policy": (
            "fit_and_evaluate and sample return only scalar diagnostics inline. "
            "Posterior draws are at <samples_path>; logs at <logs_path>. "
            "Both paths are under results_dir and accessible via SSHFS mount."
        ),
        "datasets_dir": str(_DATASETS_DIR),
        "results_dir": str(_RESULTS_DIR),
        "model_cache_dir": str(_MODEL_CACHE),
        "http_upload_url": upload_url,
        # Bulk train-data download — clients fetch this to DISK, never into
        # LLM context.  Serves only train.csv and dataset.md; protected/ is
        # unreachable by construction (whitelist + traversal check).
        "train_download_url": train_url,
    }


# ── Tool: run_python_code (assistant tier — withheld unless --enable-code-tool)

_CODE_TIMEOUT_DEFAULT = 60
_CODE_TIMEOUT_CAP     = 120
_CODE_STDOUT_CAP      = 8000
_CODE_STDERR_CAP      = 2000
_CODE_MAX_FIGURES     = 4
_RUN_ID_RE = re.compile(r"[0-9a-f]{12}")

# The subprocess runs this preamble, then the agent's code at module level.
# `cols` (train columns as numpy arrays) and/or `idata` (the run's draws as
# an arviz InferenceData) are predefined depending on what was requested.
_CODE_RUNNER_PREAMBLE = '''\
import glob
import os
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")   # headless: savefig works, no display needed
except ImportError:
    pass


def _load_cols(path):
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) for v in vals])
        except ValueError:
            out[k] = np.array(vals, dtype=object)
    return out


def _load_idata(samples_dir):
    """Build an InferenceData from CmdStan CSVs, across arviz versions.

    arviz < 1 has from_cmdstan(paths); arviz >= 1 removed it (it requires
    Python >= 3.12, so environments differ by interpreter version) and takes a
    cmdstanpy fit through from_cmdstanpy.  Passing log_likelihood explicitly
    puts log_lik in its own group, which az.loo needs on both.
    """
    import arviz as az
    if hasattr(az, "from_cmdstan"):                       # arviz < 1
        csvs = sorted(glob.glob(os.path.join(samples_dir, "*.csv")))
        try:
            return az.from_cmdstan(csvs, log_likelihood="log_lik")
        except Exception:
            return az.from_cmdstan(csvs)                  # no log_lik in this model
    from cmdstanpy import from_csv                        # arviz >= 1
    fit = from_csv(samples_dir)
    try:
        return az.from_cmdstanpy(fit, log_likelihood="log_lik")
    except Exception:
        return az.from_cmdstanpy(fit)


cols = _load_cols("train.csv") if os.path.exists("train.csv") else None
idata = _load_idata("samples") if os.path.isdir("samples") else None

# ── agent-written code below ──────────────────────────────────────────────────
'''


@mcp.tool()
def run_python_code(
    code: str,
    dataset: Optional[str] = None,
    run_id: Optional[str] = None,
    timeout_sec: int = _CODE_TIMEOUT_DEFAULT,
):
    """Execute Python analysis code on the server; figures come back as images.

    Preloaded names, depending on what you request:

    - `dataset` → `cols`: dict mapping each train.csv column to a numpy 1-D
      array (TRAIN data only — never test data).
    - `run_id` → `idata`: the run's posterior draws as an arviz
      InferenceData (from a previous `sample` / `fit_and_evaluate` call).

    numpy, matplotlib (Agg) and arviz are available.  print() the aggregates
    you want to see — stdout is capped at 8 kB.  Every figure you save as
    .png in the working directory (or via plt.savefig) is returned as an
    image, up to 4 per call.  Never print raw data rows.

    Typical uses: EDA on the train columns, prior/posterior predictive plots,
    trace plots (az.plot_trace), PSIS-LOO (az.loo(idata)).  The arviz PPC API
    differs by major version (az.plot_ppc on 0.x; az.plot_ppc_dist and friends
    on 1.x) — check az.__version__, or plot the overlay yourself.

    The code runs in an isolated working directory containing only the
    requested files, with a wall-clock limit (default 60 s, max 120 s).
    Errors return the traceback so you can fix the code and retry.
    """
    import shutil  # noqa: PLC0415

    if dataset is None and run_id is None:
        return {"status": "error", "stage": "input",
                "message": "Provide 'dataset' (for train columns) and/or 'run_id' (for posterior draws)."}
    try:
        timeout = max(1, min(int(timeout_sec), _CODE_TIMEOUT_CAP))
    except (TypeError, ValueError):
        timeout = _CODE_TIMEOUT_DEFAULT

    with tempfile.TemporaryDirectory(prefix="stan_mcp_code_") as tmp:
        tmpdir = Path(tmp)

        if dataset is not None:
            try:
                ds_dir = _resolve_under(_DATASETS_DIR, dataset)
            except ValueError as exc:
                return {"status": "error", "stage": "input", "message": str(exc)}
            train = ds_dir / "train.csv"
            if not train.exists():
                return {"status": "error", "stage": "input",
                        "message": f"Dataset '{dataset}' not found (no train.csv)."}
            shutil.copy(train, tmpdir / "train.csv")

        if run_id is not None:
            if not _RUN_ID_RE.fullmatch(str(run_id)):
                return {"status": "error", "stage": "input",
                        "message": f"Invalid run_id '{run_id}'."}
            samples_src = _RESULTS_DIR / "_runs" / run_id / "samples"
            csvs = sorted(samples_src.glob("*.csv")) if samples_src.is_dir() else []
            if not csvs:
                return {"status": "error", "stage": "input",
                        "message": f"No samples found for run_id '{run_id}'."}
            (tmpdir / "samples").mkdir()
            for f in csvs:
                shutil.copy(f, tmpdir / "samples" / f.name)
            model_stan = _RESULTS_DIR / "_runs" / run_id / "model.stan"
            if model_stan.exists():
                shutil.copy(model_stan, tmpdir / "model.stan")

        (tmpdir / "runner.py").write_text(_CODE_RUNNER_PREAMBLE + code + "\n")
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "runner.py"],
                cwd=tmpdir, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "stage": "timeout",
                    "message": (f"Code exceeded {timeout}s and was stopped. "
                                "Use cheaper computations or raise timeout_sec "
                                f"(max {_CODE_TIMEOUT_CAP}).")}

        result: dict = {
            "status": "ok" if proc.returncode == 0 else "error",
            "stdout": proc.stdout[:_CODE_STDOUT_CAP],
        }
        if len(proc.stdout) > _CODE_STDOUT_CAP:
            result["note"] = (f"stdout truncated to {_CODE_STDOUT_CAP} of "
                              f"{len(proc.stdout)} chars — print less.")
        if proc.returncode != 0:
            result["stage"] = "execution"
            result["stderr"] = proc.stderr[-_CODE_STDERR_CAP:]

        pngs = sorted(tmpdir.glob("*.png"))
        if len(pngs) > _CODE_MAX_FIGURES:
            result["figures_note"] = (f"{len(pngs)} figures produced; returning "
                                      f"the first {_CODE_MAX_FIGURES}.")
            pngs = pngs[:_CODE_MAX_FIGURES]
        from fastmcp.utilities.types import Image  # noqa: PLC0415
        images = [Image(data=p.read_bytes(), format="png") for p in pngs]
        if images:
            result["figures"] = [p.name for p in pngs]
        return [result, *images]


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stan MCP Server — serves Bayesian modelling tools over HTTP.",
    )
    parser.add_argument(
        "--datasets-dir", required=True, type=Path,
        help="Path to directory containing dataset subdirectories.",
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Path to directory where per-dataset log.jsonl files are written.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8765, type=int, help="MCP bind port (default: 8765)")
    parser.add_argument(
        "--upload-port", default=8766, type=int,
        help="HTTP upload endpoint port (default: 8766).  Pass 0 to disable.",
    )
    parser.add_argument(
        "--token", default=None,
        help=(
            "Bearer token for authentication.  If set, all requests to both "
            "the MCP endpoint and the HTTP upload endpoint must include "
            "'Authorization: Bearer <token>'.  "
            "Generate one with: openssl rand -hex 32"
        ),
    )
    parser.add_argument(
        "--transport", default="streamable-http",
        choices=["streamable-http", "stdio"],
        help="MCP transport (default: streamable-http).  Use 'stdio' for Claude Desktop via SSH.",
    )
    parser.add_argument(
        "--include-run-history",
        action="store_true",
        help=(
            "Expose the get_run_history tool (default: OFF). It returns the "
            "dataset-wide log across ALL runs, sessions and agents, so a "
            "benchmark agent could read another run's results. Off by default "
            "so the guarantee does not depend on the client remembering to "
            "exclude it. See TOOL_POLICY.md."
        ),
    )
    parser.add_argument(
        "--enable-code-tool",
        action="store_true",
        help=(
            "Expose the run_python_code tool (default: OFF). It executes "
            "agent-written Python on the SERVER HOST — contained for honest "
            "agents (isolated subprocess, only the requested files), but not "
            "an OS-level sandbox, so it must never be offered to benchmark "
            "agents. Intended for assistant use. See TOOL_POLICY.md."
        ),
    )
    args = parser.parse_args()

    # A token is enforced by ASGI middleware, which only exists on the HTTP
    # transports. Accepting it under stdio would promise an authentication
    # that is never applied — fail loudly instead.
    if args.transport == "stdio" and (args.token or os.environ.get("STAN_MCP_TOKEN")):
        parser.error(
            "--token (or STAN_MCP_TOKEN) cannot be used with --transport stdio: "
            "the bearer check is HTTP middleware and would be silently ignored. "
            "Under stdio the client launches the server as its own subprocess, "
            "so access is already limited to that client."
        )

    global _DATASETS_DIR, _RESULTS_DIR, _UPLOAD_PORT, _UPLOAD_HOST, _BEARER_TOKEN, _TRANSPORT
    _DATASETS_DIR  = args.datasets_dir.resolve()
    _RESULTS_DIR   = args.results_dir.resolve()
    _UPLOAD_PORT   = args.upload_port
    _UPLOAD_HOST   = args.host
    _BEARER_TOKEN  = args.token or os.environ.get("STAN_MCP_TOKEN")
    _TRANSPORT     = args.transport

    # ── Tool-surface gating — BEFORE any transport starts serving.  Excluding
    # a tool client-side is not enough: any other client connecting here would
    # still see it.  Unregister so no client can call it.  (This block used to
    # sit after the stdio early-return, so stdio servers exposed
    # get_run_history unconditionally.)
    withheld: list[str] = []
    if not args.include_run_history:
        withheld.append("get_run_history")
    if not args.enable_code_tool:
        withheld.append("run_python_code")
    for tool_name in withheld:
        try:
            mcp.remove_tool(tool_name)
        except Exception as exc:          # noqa: BLE001 - never fail startup
            print(f"  WARNING: could not withhold {tool_name}: {exc}")

    if args.transport == "stdio":
        import sys
        # stderr, not stdout: stdout carries the MCP protocol here. Claude
        # Desktop captures this into ~/Library/Logs/Claude/mcp-server-*.log.
        print(f"Stan MCP Server {_VERSION} (stdio) — datasets: {_DATASETS_DIR}  "
              f"results: {_RESULTS_DIR}", file=sys.stderr)
        print(f"  tools withheld : {', '.join(withheld) if withheld else 'none'}"
              f"  (--include-run-history / --enable-code-tool to expose)",
              file=sys.stderr)
        print("  http sidecar   : not started under stdio — get_upload_instructions "
              "returns the file-copy method; no train-download URL",
              file=sys.stderr)
        mcp.run(transport="stdio")
        return

    print(f"Stan MCP Server {_VERSION} starting on http://{args.host}:{args.port}/mcp")
    print(f"  datasets : {_DATASETS_DIR}")
    print(f"  results  : {_RESULTS_DIR}")
    print(f"  cache    : {_MODEL_CACHE}")
    print(f"  auth     : {'Bearer token required' if _BEARER_TOKEN else 'none (use --token to enable)'}")

    token_middleware = [Middleware(_BearerTokenMiddleware)] if _BEARER_TOKEN else []

    if _UPLOAD_PORT:
        upload_url = f"http://{args.host}:{_UPLOAD_PORT}/dataset/{{name}}"
        print(f"  upload   : {upload_url}")
        if _BEARER_TOKEN:
            _upload_app.add_middleware(_BearerTokenMiddleware)
        t = threading.Thread(
            target=uvicorn.run,
            kwargs={
                "app": _upload_app,
                "host": args.host,
                "port": _UPLOAD_PORT,
                "log_level": "error",
            },
            daemon=True,
        )
        t.start()
    else:
        print("  upload   : disabled")

    if args.include_run_history:
        print("  history  : get_run_history EXPOSED (--include-run-history) — "
              "returns results across all runs on a dataset; not suitable for "
              "benchmark agents, see TOOL_POLICY.md")
    else:
        print("  history  : get_run_history withheld (default)")
    if args.enable_code_tool:
        print("  code     : run_python_code EXPOSED (--enable-code-tool) — "
              "executes agent code on this host; assistant use only, "
              "never for benchmark agents, see TOOL_POLICY.md")
    else:
        print("  code     : run_python_code withheld (default)")

    mcp.run(transport="streamable-http", host=args.host, port=args.port,
            middleware=token_middleware)


if __name__ == "__main__":
    main()
