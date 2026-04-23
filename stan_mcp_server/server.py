#!/usr/bin/env python3
"""Stan MCP Server — structured Bayesian modelling tools for LLM agents.

Serves tools over HTTP (streamable-http transport):

    fit_and_evaluate        Sample + compute NLPD; returns run_id + diagnostics + URLs.
    sample                  Sample + persist draws to disk; returns run_id + diagnostics + URLs.
    check_model             Compile-only model check (syntax + log_lik presence).
    get_data_summary        Compact EDA for a named dataset.
    get_upload_instructions Return HTTP upload URL and field names for datasets.
    list_datasets           List available datasets on the server.
    get_run_history         Return the logged run history for a dataset.
    get_capabilities        Describe tools and current server configuration.

Run assets (logs, posterior draws) are stored server-side under
<results-dir>/_runs/<run_id>/ and served by the HTTP sidecar:
    GET /logs/{run_id}     — CmdStan log output (plain text)
    GET /samples/{run_id}  — posterior draw CSVs (tar.gz)
    POST /dataset/{name}   — upload train/test CSVs (multipart)

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
import contextlib
import csv
import hashlib
import io
import json
import logging
import os
import re
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

# ── Global path config (set by main() before the server starts) ────────────────
_DATASETS_DIR:  Path = Path("datasets")
_RESULTS_DIR:   Path = Path("results")
_MODEL_CACHE:   Path = Path(tempfile.gettempdir()) / "stan_mcp_model_cache"
_UPLOAD_PORT:   int  = 8766          # 0 = disabled
_UPLOAD_HOST:   str  = "127.0.0.1"
_UPLOAD_DIR:    str  = "_uploaded"
_BEARER_TOKEN:  Optional[str] = None  # set by --token; None = no auth


class _BearerTokenMiddleware:
    """ASGI middleware — reject requests without the correct Bearer token."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            if auth != f"Bearer {_BEARER_TOKEN}":
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
            "Use the 'sample' tool and compute PSIS-LOO on the training log_lik yourself. "
            "fit_and_evaluate requires a pre-staged dataset with protected/test.csv."
        ),
    }
    if interface_warnings:
        result["interface_warnings"] = interface_warnings
    return result


# ── HTTP upload app (runs on --upload-port in a daemon thread) ─────────────────

_upload_app = FastAPI(title="Stan dataset upload", docs_url=None, redoc_url=None)


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
    """Return the HTTP sidecar base URL, or None when the port is disabled."""
    if not _UPLOAD_PORT:
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


def _merge_config(config: Optional[dict]) -> dict:
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        for k in ("chains", "iter_warmup", "iter_sampling", "seed"):
            if k in config:
                cfg[k] = config[k]
    return cfg


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
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    cols: dict[str, list] = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            cols[k].append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def _col_stats(arr: np.ndarray) -> dict:
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


def _load_dataset(dataset: str) -> tuple[dict, list]:
    """Load train + test CSVs for a named dataset into a Stan data dict.

    Reads <datasets_dir>/<dataset>/train.csv and
          <datasets_dir>/<dataset>/protected/test.csv.
    Variable names are derived from the ## Data Interface block in dataset.md;
    Stan base names must match the CSV column names exactly.
    """
    ds_dir = _DATASETS_DIR / dataset
    train_path = ds_dir / "train.csv"
    test_path  = ds_dir / "protected" / "test.csv"
    md_path    = ds_dir / "dataset.md"

    if not train_path.exists():
        candidates = [
            str(p.parent.relative_to(_DATASETS_DIR))
            for p in _DATASETS_DIR.glob("**/train.csv")
            if _UPLOAD_DIR not in p.parts
        ]
        raise ValueError(f"Dataset '{dataset}' not found. Available: {candidates}")
    if not test_path.exists():
        raise ValueError(f"Test file not found at {test_path}")

    train_cols   = _load_csv_columns(train_path)
    test_cols    = _load_csv_columns(test_path)
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
    n_test  = len(next(iter(test_cols.values())))
    data: dict = {"N_train": n_train, "N_test": n_test}

    for csv_col, stan_base in csv_to_base.items():
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
        dtype = train_vars.get(stan_base, "float")
        if dtype == "int":
            data[f"{stan_base}_train"] = [int(v) for v in train_cols[csv_col]]
            data[f"{stan_base}_test"]  = [int(v) for v in test_cols[test_csv_col]]
        else:
            data[f"{stan_base}_train"] = train_cols[csv_col].tolist()
            data[f"{stan_base}_test"]  = test_cols[test_csv_col].tolist()

    if has_J and j_var_bases:
        j_csv_cols = [c for c, b in csv_to_base.items() if b in j_var_bases]
        all_ids: set[int] = set()
        for c in j_csv_cols:
            all_ids.update(int(v) for v in train_cols[c])
            all_ids.update(int(v) for v in test_cols[c])
        data["J"] = len(all_ids)

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
    log_path = _RESULTS_DIR / dataset / "log.jsonl"
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
    log_path = _RESULTS_DIR / dataset / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


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

    if data is None:
        if dataset is None:
            return {"status": "error", "stage": "input", "message": "Either 'data' (with 'y_test') or 'dataset' must be provided."}
        # Reject train-only (uploaded) datasets — no held-out test set exists.
        test_path = _DATASETS_DIR / dataset / "protected" / "test.csv"
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
            data, y_test = _load_dataset(dataset)
        except ValueError as exc:
            return {"status": "error", "stage": "data_loading", "message": str(exc)}

    # y_test is optional in explicit-data mode: it is only used for the
    # log_lik shape check. NLPD is computed from log_lik alone. When omitted,
    # the shape check is skipped and the user is responsible for correctness.

    if not re.search(r'\blog_lik\b', stan_code):
        return {"status": "error", "stage": "missing_log_lik", "message": "no 'log_lik' found in stan_code — required for NLPD computation"}

    try:
        model = _get_model(stan_code)
    except Exception as exc:
        return {"status": "error", "stage": "compilation", "message": _extract_compile_error(exc)}

    run_id = _make_run_id()
    run_dir = _RESULTS_DIR / "_runs" / run_id
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

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
            )
        except Exception as exc:
            (run_dir / "logs.txt").write_text(log_buf.getvalue())
            return {"status": "error", "stage": "sampling", "message": str(exc)[:500]}
    (run_dir / "logs.txt").write_text(log_buf.getvalue())
    runtime_sec = round(time.time() - t0, 1)

    all_vars = fit.stan_variables()
    if "log_lik" not in all_vars:
        return {"status": "error", "stage": "missing_log_lik", "message": "'log_lik' not found in generated quantities output"}

    log_lik = np.asarray(all_vars["log_lik"])
    if log_lik.ndim == 1:
        log_lik = log_lik[:, np.newaxis]

    n_test = len(y_test) if y_test is not None else log_lik.shape[1]
    if y_test is not None and log_lik.shape[1] != n_test:
        return {"status": "error", "stage": "missing_log_lik", "message": f"log_lik has {log_lik.shape[1]} columns but y_test has {n_test} elements"}

    nlpd = _compute_nlpd(log_lik)

    try:
        diag = _make_diagnostics(fit)
    except Exception:
        diag = {"n_divergences": -1, "r_hat_max": float("nan"), "ess_bulk_min": -1}

    try:
        param_summary = _make_param_summary(fit)
    except Exception:
        param_summary = {}

    result: dict = {
        "status": "ok",
        "run_id": run_id,
        "nlpd": round(nlpd, 4),
        "n_divergences": diag["n_divergences"],
        "r_hat_max": diag["r_hat_max"],
        "ess_bulk_min": diag["ess_bulk_min"],
        "runtime_sec": runtime_sec,
        "param_summary": param_summary,
        "data_keys_loaded": sorted(data.keys()),
        "logs_path":    str(run_dir / "logs.txt"),
        "samples_path": str(run_dir),
    }

    if dataset is not None:
        existing = _read_log(dataset)
        iter_num = len(existing)
        improved = None if iter_num == 0 else bool(nlpd < min(e["nlpd"] for e in existing if "nlpd" in e))
        _append_log(dataset, {
            "iter": iter_num,
            "run_id": run_id,
            "nlpd": round(nlpd, 4),
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
    data: dict,
    config: Optional[dict] = None,
) -> dict:
    """Sample from a Stan model and persist posterior draws to disk.

    Returns scalar diagnostics and a `run_id` only — raw draws are never
    returned inline.  Retrieve them via `samples_path` (directory of per-chain
    Stan CSVs).  CmdStan logs are available at `logs_path`.  Both paths are
    under --results-dir and are directly accessible when that directory is
    mounted via SSHFS on the client.
    """
    try:
        model = _get_model(stan_code)
    except Exception as exc:
        return {"status": "error", "stage": "compilation", "message": _extract_compile_error(exc)}

    run_id = _make_run_id()
    run_dir = _RESULTS_DIR / "_runs" / run_id
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

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
            )
        except Exception as exc:
            (run_dir / "logs.txt").write_text(log_buf.getvalue())
            return {"status": "error", "stage": "sampling", "message": str(exc)[:500]}
    (run_dir / "logs.txt").write_text(log_buf.getvalue())
    runtime_sec = round(time.time() - t0, 1)

    try:
        diag = _make_diagnostics(fit)
    except Exception:
        diag = {"n_divergences": -1, "r_hat_max": float("nan"), "ess_bulk_min": -1}

    try:
        param_summary = _make_param_summary(fit)
    except Exception:
        param_summary = {}

    result: dict = {
        "status": "ok",
        "run_id": run_id,
        "n_samples": cfg["chains"] * cfg["iter_sampling"],
        "runtime_sec": runtime_sec,
        "diagnostics": diag,
        "param_summary": param_summary,
        "data_keys_loaded": sorted(data.keys()),
        "logs_path":    str(run_dir / "logs.txt"),
        "samples_path": str(run_dir),
    }

    return result


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
    ds_dir     = _DATASETS_DIR / dataset
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

    return {
        "dataset": dataset,
        "tier": tier,
        "has_test": has_test,
        "n_train": n_train,
        "n_test": n_test,
        "columns": {col: _col_stats(arr) for col, arr in train_cols.items()},
        "dataset_md": dataset_md,
    }


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

    IMPORTANT — N and N_test are never injected automatically from the CSV.
    Declare them in the Stan `data` block and supply their values explicitly
    via the `data` parameter of `fit_and_evaluate`
    (e.g. `data={"N": 80, "N_test": 20}`).
    """
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
    benchmarks_dir = _DATASETS_DIR / "benchmarks"
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
    entries = _read_log(dataset)
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

@mcp.tool()
def get_capabilities() -> dict:
    """Return server capabilities, available tools, and current configuration.

    Call this first to understand what the server can do and how it is
    configured before issuing other tool calls.
    """
    base_url = _run_base_url()
    upload_url = f"{base_url}/dataset/{{name}}" if base_url else "disabled"
    return {
        "server": "stan-mcp-server",
        "tools": [
            "get_capabilities",
            "list_datasets",
            "get_data_summary",
            "check_model",
            "fit_and_evaluate",
            "sample",
            "get_upload_instructions",
            "get_run_history",
        ],
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
    }


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
    args = parser.parse_args()

    global _DATASETS_DIR, _RESULTS_DIR, _UPLOAD_PORT, _UPLOAD_HOST, _BEARER_TOKEN
    _DATASETS_DIR  = args.datasets_dir.resolve()
    _RESULTS_DIR   = args.results_dir.resolve()
    _UPLOAD_PORT   = args.upload_port
    _UPLOAD_HOST   = args.host
    _BEARER_TOKEN  = args.token or os.environ.get("STAN_MCP_TOKEN")

    if args.transport == "stdio":
        import sys
        print(f"Stan MCP Server (stdio) — datasets: {_DATASETS_DIR}  results: {_RESULTS_DIR}", file=sys.stderr)
        mcp.run(transport="stdio")
        return

    print(f"Stan MCP Server starting on http://{args.host}:{args.port}/mcp")
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

    mcp.run(transport="streamable-http", host=args.host, port=args.port,
            middleware=token_middleware)


if __name__ == "__main__":
    print("Starting Stan MCP Server (Version 2026/4/26 13:09)...")
    main()
