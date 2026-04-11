#!/usr/bin/env python3
"""Stan MCP Server — structured Bayesian modelling tools for LLM agents.

Serves tools over HTTP (streamable-http transport):

    fit_and_evaluate   Sample + compute NLPD on held-out test responses.
    sample             Sample + return raw posterior draws.
    check_model        Compile-only model check (syntax + log_lik presence).
    get_data_summary   Compact EDA for a named dataset.
    upload_dataset     Push train/test CSV content to the server.
    list_datasets      List available datasets on the server.
    get_run_history    Return the logged run history for a dataset.
    get_capabilities   Describe tools and current server configuration.

Usage
-----
    stan-mcp-server --datasets-dir /path/to/datasets --results-dir /path/to/results
    stan-mcp-server --datasets-dir ./datasets --results-dir ./results --port 8765 --host 0.0.0.0
"""

import argparse
import csv
import hashlib
import json
import re
import socket
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastmcp import FastMCP
from scipy.special import logsumexp

# ── Global path config (set by main() before the server starts) ────────────────
_DATASETS_DIR: Path = Path("datasets")
_RESULTS_DIR:  Path = Path("results")
_MODEL_CACHE:  Path = Path(tempfile.gettempdir()) / "stan_mcp_model_cache"

# ── Default sampling config ────────────────────────────────────────────────────
_DEFAULT_CONFIG = {
    "chains": 4,
    "iter_warmup": 1000,
    "iter_sampling": 1000,
    "seed": 42,
}

mcp = FastMCP("stan")

# ── Helpers ────────────────────────────────────────────────────────────────────

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
        candidates = [p.parent.name for p in _DATASETS_DIR.glob("*/train.csv")]
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
        if csv_col not in test_cols:
            raise ValueError(f"Column '{csv_col}' missing from test.csv")
        dtype = train_vars.get(stan_base, "float")
        if dtype == "int":
            data[f"{stan_base}_train"] = [int(v) for v in train_cols[csv_col]]
            data[f"{stan_base}_test"]  = [int(v) for v in test_cols[csv_col]]
        else:
            data[f"{stan_base}_train"] = train_cols[csv_col].tolist()
            data[f"{stan_base}_test"]  = test_cols[csv_col].tolist()

    if has_J and j_var_bases:
        j_csv_cols = [c for c, b in csv_to_base.items() if b in j_var_bases]
        all_ids: set[int] = set()
        for c in j_csv_cols:
            all_ids.update(int(v) for v in train_cols[c])
            all_ids.update(int(v) for v in test_cols[c])
        data["J"] = len(all_ids)

    y_test_arrays = [test_cols[c].tolist() for c in response_cols if c in test_cols]
    y_test = [v for arr in y_test_arrays for v in arr]  # concatenate all response cols
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

    When `notes`, `rationale`, and `dataset` are all provided the result is
    appended to <results_dir>/<dataset>/log.jsonl.
    """
    if data is None:
        if dataset is None:
            return {"status": "error", "stage": "input", "message": "Either 'data' (with 'y_test') or 'dataset' must be provided."}
        try:
            data, y_test = _load_dataset(dataset)
        except ValueError as exc:
            return {"status": "error", "stage": "data_loading", "message": str(exc)}

    if y_test is None:
        return {"status": "error", "stage": "input", "message": "'y_test' is required when 'data' is passed explicitly."}

    if not re.search(r'\blog_lik\b', stan_code):
        return {"status": "error", "stage": "missing_log_lik", "message": "no 'log_lik' found in stan_code — required for NLPD computation"}

    try:
        model = _get_model(stan_code)
    except Exception as exc:
        return {"status": "error", "stage": "compilation", "message": _extract_compile_error(exc)}

    cfg = _merge_config(config)
    t0 = time.time()
    try:
        fit = model.sample(
            data=data,
            chains=cfg["chains"],
            iter_warmup=cfg["iter_warmup"],
            iter_sampling=cfg["iter_sampling"],
            seed=cfg["seed"],
            show_progress=False,
            show_console=False,
        )
    except Exception as exc:
        return {"status": "error", "stage": "sampling", "message": str(exc)[:500]}
    runtime_sec = round(time.time() - t0, 1)

    all_vars = fit.stan_variables()
    if "log_lik" not in all_vars:
        return {"status": "error", "stage": "missing_log_lik", "message": "'log_lik' not found in generated quantities output"}

    log_lik = np.asarray(all_vars["log_lik"])
    if log_lik.ndim == 1:
        log_lik = log_lik[:, np.newaxis]

    n_test = len(y_test)
    if log_lik.shape[1] != n_test:
        return {"status": "error", "stage": "missing_log_lik", "message": f"log_lik has {log_lik.shape[1]} columns but y_test has {n_test} elements"}

    nlpd = _compute_nlpd(log_lik)

    try:
        diag = _make_diagnostics(fit)
    except Exception:
        diag = {"n_divergences": -1, "r_hat_max": float("nan"), "ess_bulk_min": -1}

    param_summary = _make_param_summary(fit)

    result: dict = {
        "status": "ok",
        "nlpd": round(nlpd, 4),
        "n_divergences": diag["n_divergences"],
        "r_hat_max": diag["r_hat_max"],
        "ess_bulk_min": diag["ess_bulk_min"],
        "runtime_sec": runtime_sec,
        "param_summary": param_summary,
    }

    if dataset is not None and notes is not None:
        existing = _read_log(dataset)
        iter_num = len(existing)
        improved = None if iter_num == 0 else bool(nlpd < min(e["nlpd"] for e in existing if "nlpd" in e))
        _append_log(dataset, {
            "iter": iter_num,
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
    """Sample from a Stan model and return raw posterior draws.

    Draws are flattened across chains.  log_lik is omitted by default;
    pass `return_log_lik: true` inside `config` to include it.
    """
    return_log_lik = bool((config or {}).get("return_log_lik", False))

    try:
        model = _get_model(stan_code)
    except Exception as exc:
        return {"status": "error", "stage": "compilation", "message": _extract_compile_error(exc)}

    cfg = _merge_config(config)
    t0 = time.time()
    try:
        fit = model.sample(
            data=data,
            chains=cfg["chains"],
            iter_warmup=cfg["iter_warmup"],
            iter_sampling=cfg["iter_sampling"],
            seed=cfg["seed"],
            show_progress=False,
            show_console=False,
        )
    except Exception as exc:
        return {"status": "error", "stage": "sampling", "message": str(exc)[:500]}
    runtime_sec = round(time.time() - t0, 1)

    try:
        diag = _make_diagnostics(fit)
    except Exception:
        diag = {"n_divergences": -1, "r_hat_max": float("nan"), "ess_bulk_min": -1}

    draws_out: dict = {}
    for name, draws in fit.stan_variables().items():
        if name == "log_lik" and not return_log_lik:
            continue
        draws = np.asarray(draws)
        if draws.ndim == 1:
            draws_out[name] = draws.tolist()
        elif draws.ndim == 2:
            for i in range(draws.shape[1]):
                draws_out[f"{name}[{i + 1}]"] = draws[:, i].tolist()

    return {
        "status": "ok",
        "n_samples": cfg["chains"] * cfg["iter_sampling"],
        "runtime_sec": runtime_sec,
        "draws": draws_out,
        "diagnostics": diag,
    }


# ── Tool: get_data_summary ─────────────────────────────────────────────────────

@mcp.tool()
def get_data_summary(dataset: str) -> dict:
    """Return a compact EDA summary for a named dataset.

    Reads <datasets_dir>/<dataset>/train.csv and dataset.md.
    The response column of the test set is not exposed (held-out integrity).
    """
    ds_dir     = _DATASETS_DIR / dataset
    train_path = ds_dir / "train.csv"
    md_path    = ds_dir / "dataset.md"
    test_path  = ds_dir / "protected" / "test.csv"

    if not train_path.exists():
        candidates = [p.parent.name for p in _DATASETS_DIR.glob("*/train.csv")]
        return {"status": "error", "message": f"Dataset '{dataset}' not found. Available: {candidates}"}

    train_cols = _load_csv_columns(train_path)
    dataset_md = md_path.read_text() if md_path.exists() else ""
    response_col = _find_response_col(dataset_md, list(train_cols.keys()))

    n_train = len(next(iter(train_cols.values())))
    n_test: Optional[int] = None
    if test_path.exists():
        test_cols = _load_csv_columns(test_path)
        n_test = len(next(iter(test_cols.values())))

    return {
        "dataset": dataset,
        "n_train": n_train,
        "n_test": n_test,
        "columns": {col: _col_stats(arr) for col, arr in train_cols.items()},
        "dataset_md": dataset_md,
    }


# ── Tool: upload_dataset ──────────────────────────────────────────────────────

@mcp.tool()
def upload_dataset(
    name: str,
    train_csv: str,
    test_csv: str,
    dataset_md: Optional[str] = None,
) -> dict:
    """Register a dataset on the server by uploading CSV content as strings.

    Useful when the server is running remotely and files cannot be pre-staged.
    Call this once before calling fit_and_evaluate / get_data_summary with
    the same dataset name.  Overwrites any previously uploaded dataset with
    the same name.

    Parameters
    ----------
    name        : identifier used in subsequent tool calls, e.g. "my_experiment"
    train_csv   : full CSV content (including header row) for training data
    test_csv    : full CSV content (including header row) for held-out test data
    dataset_md  : optional dataset.md content (e.g. with ## Data Interface block
                  for variable-type annotations); pass None to skip
    """
    _UPLOAD_DIR_NAME = "_uploaded"

    # Validate name: alphanumeric, underscores, hyphens only — no path traversal
    if not re.fullmatch(r'[A-Za-z0-9_\-]+', name):
        return {
            "status": "error",
            "message": "Dataset name may only contain letters, digits, underscores, and hyphens.",
        }

    ds_dir = _DATASETS_DIR / _UPLOAD_DIR_NAME / name
    protected_dir = ds_dir / "protected"
    protected_dir.mkdir(parents=True, exist_ok=True)

    (ds_dir / "train.csv").write_text(train_csv)
    (protected_dir / "test.csv").write_text(test_csv)
    if dataset_md is not None:
        (ds_dir / "dataset.md").write_text(dataset_md)

    # Quick sanity check: parse headers and row counts
    try:
        train_cols = _load_csv_columns(ds_dir / "train.csv")
        test_cols  = _load_csv_columns(protected_dir / "test.csv")
    except Exception as exc:
        return {"status": "error", "message": f"CSV parse error: {exc}"}

    return {
        "status": "ok",
        "dataset": f"{_UPLOAD_DIR_NAME}/{name}",
        "n_train": len(next(iter(train_cols.values()))),
        "n_test":  len(next(iter(test_cols.values()))),
        "train_columns": list(train_cols.keys()),
        "test_columns":  list(test_cols.keys()),
    }


# ── Tool: list_datasets ──────────────────────────────────────────────────────

@mcp.tool()
def list_datasets() -> dict:
    """List all available datasets on the server.

    Returns two lists:
      - datasets : top-level pre-staged datasets under --datasets-dir
      - uploaded : datasets pushed via upload_dataset (under _uploaded/)
    """
    top_level = sorted(
        p.parent.name
        for p in _DATASETS_DIR.glob("*/train.csv")
        if p.parent.name != "_uploaded"
    )
    uploaded_dir = _DATASETS_DIR / "_uploaded"
    uploaded = sorted(
        f"_uploaded/{p.parent.name}"
        for p in uploaded_dir.glob("*/train.csv")
    ) if uploaded_dir.exists() else []
    return {"datasets": top_level, "uploaded": uploaded}


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
    return {
        "server": "stan-mcp-server",
        "tools": [
            "get_capabilities",
            "list_datasets",
            "get_data_summary",
            "check_model",
            "fit_and_evaluate",
            "sample",
            "upload_dataset",
            "get_run_history",
        ],
        "default_sampling_config": _DEFAULT_CONFIG,
        "log_lik_contract": (
            "Every model used with fit_and_evaluate must declare "
            "'vector[N_test] log_lik' in generated quantities."
        ),
        "datasets_dir": str(_DATASETS_DIR),
        "results_dir": str(_RESULTS_DIR),
        "model_cache_dir": str(_MODEL_CACHE),
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
    parser.add_argument("--port", default=8765, type=int, help="Bind port (default: 8765)")
    args = parser.parse_args()

    global _DATASETS_DIR, _RESULTS_DIR
    _DATASETS_DIR = args.datasets_dir.resolve()
    _RESULTS_DIR  = args.results_dir.resolve()

    print(f"Stan MCP Server starting on http://{args.host}:{args.port}/mcp")
    print(f"  datasets : {_DATASETS_DIR}")
    print(f"  results  : {_RESULTS_DIR}")
    print(f"  cache    : {_MODEL_CACHE}")

    mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
