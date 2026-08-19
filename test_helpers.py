#!/usr/bin/env python3
"""Fast unit tests for the pure helpers in server.py — no CmdStan required.

These guard the parts of the server that never need a sampler to go wrong:
config clamping, dataset-name validation (path traversal), CSV loading,
the Data Interface parser, the J (group count) logic, and the guarantee
that get_capabilities only advertises tools that are actually registered.

Usage:
    pytest test_helpers.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import stan_mcp_server.server as srv


# ── _merge_config ──────────────────────────────────────────────────────────────

def test_merge_config_defaults():
    assert srv._merge_config(None) == srv._DEFAULT_CONFIG


def test_merge_config_clamps_to_caps():
    cfg = srv._merge_config({"chains": 500, "iter_warmup": 10**9, "iter_sampling": 10**9})
    assert cfg["chains"] == srv._CONFIG_CAPS["chains"]
    assert cfg["iter_warmup"] == srv._CONFIG_CAPS["iter_warmup"]
    assert cfg["iter_sampling"] == srv._CONFIG_CAPS["iter_sampling"]


def test_merge_config_ignores_malformed_and_nonpositive_values():
    cfg = srv._merge_config({"chains": "lots", "iter_sampling": -5, "seed": "7"})
    assert cfg["chains"] == srv._DEFAULT_CONFIG["chains"]
    assert cfg["iter_sampling"] == srv._DEFAULT_CONFIG["iter_sampling"]
    assert cfg["seed"] == 7  # numeric strings are coerced


def test_merge_config_runtime_ceiling_only_lowers():
    assert srv._merge_config({"max_runtime_sec": 60})["max_runtime_sec"] == 60
    asked_high = srv._merge_config({"max_runtime_sec": 10**6})["max_runtime_sec"]
    assert asked_high == srv._DEFAULT_CONFIG["max_runtime_sec"]


# ── _resolve_under (path traversal) ────────────────────────────────────────────

def test_resolve_under_accepts_nested_names(tmp_path):
    (tmp_path / "benchmarks" / "reg1").mkdir(parents=True)
    p = srv._resolve_under(tmp_path, "benchmarks/reg1")
    assert p == (tmp_path / "benchmarks" / "reg1").resolve()


@pytest.mark.parametrize("name", ["../evil", "a/../../evil", "/etc"])
def test_resolve_under_rejects_escapes(tmp_path, name):
    with pytest.raises(ValueError):
        srv._resolve_under(tmp_path, name)


def test_tools_reject_traversal_names(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(srv, "_RESULTS_DIR", tmp_path / "results")
    assert srv.get_data_summary("../secrets")["status"] == "error"
    assert srv.get_run_history("../secrets")["status"] == "error"
    r = srv.fit_and_evaluate(stan_code="// log_lik", dataset="../secrets")
    assert r["status"] == "error" and r["stage"] == "input"


# ── _load_csv_columns / _col_stats ─────────────────────────────────────────────

def test_load_csv_columns_numeric_and_categorical(tmp_path):
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("x,city,y\n1.0,konstanz,2.0\n2.0,zurich,3.5\n3.0,konstanz,\n")
    cols = srv._load_csv_columns(csv_file)
    assert cols["x"].dtype != object
    assert cols["city"].dtype == object          # strings → categorical
    assert cols["y"].dtype == object             # empty cell → not numeric

    stats = srv._col_stats(cols["city"])
    assert stats["type"] == "categorical"
    assert stats["n_levels"] == 2
    assert srv._col_stats(cols["x"])["mean"] == 2.0


def test_get_data_summary_handles_categorical_column(tmp_path, monkeypatch):
    ds = tmp_path / "cat_ds"
    ds.mkdir(parents=True)
    (ds / "train.csv").write_text("x,label,y\n1,a,2.0\n2,b,3.0\n")
    monkeypatch.setattr(srv, "_DATASETS_DIR", tmp_path)
    r = srv.get_data_summary("cat_ds")
    assert r.get("status") != "error"
    assert r["columns"]["label"]["type"] == "categorical"
    assert "mean" in r["columns"]["x"]


# ── Data Interface parsing / response columns ──────────────────────────────────

INTERFACE_MD = """# Demo

## Data Interface

```stan
int<lower=0> N_train;
int<lower=0> N_test;
int<lower=1> J;
vector[N_train] x_train;
vector[N_train] y_train;
array[N_train] int<lower=1, upper=J> g_train;
```
"""


def test_parse_data_interface():
    iface = srv._parse_data_interface(INTERFACE_MD)
    assert iface["train_vars"] == {"x": "float", "y": "float", "g": "int"}
    assert iface["has_J"] is True
    assert iface["j_var_bases"] == ["g"]


def test_find_response_cols():
    assert srv._find_response_cols("response_col: y", ["x", "y"]) == ["y"]
    assert srv._find_response_cols("response_cols: home,away", ["a"]) == ["home", "away"]
    assert srv._find_response_cols("no annotation", ["x", "y"]) == ["y"]  # last column


# ── _load_dataset: J logic and non-numeric guard ───────────────────────────────

def _write_grouped_dataset(root: Path, name: str = "grouped") -> Path:
    ds = root / name
    (ds / "protected").mkdir(parents=True)
    (ds / "dataset.md").write_text(INTERFACE_MD + "\nresponse_col: y\n")
    (ds / "train.csv").write_text("x,y,g\n0.1,1.0,1\n0.2,1.1,5\n0.3,1.2,1\n")
    (ds / "protected" / "test.csv").write_text("x,y,g\n0.4,1.3,9\n")
    return ds


def test_load_dataset_J_is_max_id_not_count(tmp_path, monkeypatch):
    _write_grouped_dataset(tmp_path)
    monkeypatch.setattr(srv, "_DATASETS_DIR", tmp_path)
    data, y_test = srv._load_dataset("grouped")
    # ids are {1, 5, 9}: len would be 3 and crash alpha[9]; must be 9.
    assert data["J"] == 9
    assert data["N_train"] == 3 and data["N_test"] == 1
    assert data["g_train"] == [1, 5, 1] and data["g_test"] == [9]
    assert y_test == [1.3]


def test_load_dataset_rejects_zero_based_group_ids(tmp_path, monkeypatch):
    ds = _write_grouped_dataset(tmp_path)
    (ds / "train.csv").write_text("x,y,g\n0.1,1.0,0\n0.2,1.1,1\n")
    monkeypatch.setattr(srv, "_DATASETS_DIR", tmp_path)
    with pytest.raises(ValueError, match="1-based"):
        srv._load_dataset("grouped")


def test_load_dataset_rejects_non_numeric_column(tmp_path, monkeypatch):
    ds = _write_grouped_dataset(tmp_path)
    (ds / "train.csv").write_text("x,y,g\n0.1,1.0,north\n0.2,1.1,south\n")
    monkeypatch.setattr(srv, "_DATASETS_DIR", tmp_path)
    with pytest.raises(ValueError, match="non-numeric"):
        srv._load_dataset("grouped")


# ── check_model fast path (compile is never reached) ───────────────────────────

def test_check_model_requires_log_lik():
    r = srv.check_model("parameters { real mu; } model { mu ~ normal(0, 1); }")
    assert r["status"] == "error" and r["stage"] == "missing_log_lik"


# ── get_capabilities reflects the live registry ────────────────────────────────

def test_capabilities_tools_come_from_registry():
    names = srv._registered_tool_names()
    caps = srv.get_capabilities()
    assert caps["tools"] == names
    assert "fit_and_evaluate" in names


def test_capabilities_never_advertise_a_withheld_tool():
    """Incident-1 regression: a tool unregistered at startup must not be named
    anywhere in the get_capabilities response."""
    assert "get_run_history" in srv._registered_tool_names()
    try:
        srv.mcp.remove_tool("get_run_history")
        caps = srv.get_capabilities()
        assert "get_run_history" not in caps["tools"]
        assert "get_run_history" not in str(caps)
    finally:
        srv.mcp.tool()(srv.get_run_history)  # re-register for other tests
    assert "get_run_history" in srv._registered_tool_names()


# ── GET /train sidecar endpoint (train data out, protected/ unreachable) ────────

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def train_client(tmp_path, monkeypatch):
    ds = tmp_path / "benchmarks" / "reg1"
    (ds / "protected").mkdir(parents=True)
    (ds / "train.csv").write_text("x,y\n1,2\n")
    (ds / "dataset.md").write_text("# reg1\n")
    (ds / "protected" / "test.csv").write_text("x,y\n9,9\n")
    monkeypatch.setattr(srv, "_DATASETS_DIR", tmp_path)
    return TestClient(srv._upload_app)


def test_get_train_serves_train_csv_and_md(train_client):
    r = train_client.get("/train/benchmarks/reg1")
    assert r.status_code == 200
    assert r.text == "x,y\n1,2\n"
    r = train_client.get("/train/benchmarks/reg1", params={"file": "dataset.md"})
    assert r.status_code == 200
    assert r.text == "# reg1\n"


def test_get_train_refuses_everything_else(train_client):
    # Filename outside the whitelist — the only two servable files are fixed.
    r = train_client.get("/train/benchmarks/reg1", params={"file": "protected/test.csv"})
    assert r.status_code == 400
    r = train_client.get("/train/benchmarks/reg1", params={"file": "../reg1/protected/test.csv"})
    assert r.status_code == 400
    # 'protected' as part of the DATASET name — inside the datasets dir, so the
    # traversal check passes; the belt-and-braces parts guard must refuse it.
    r = train_client.get("/train/benchmarks/reg1/protected")
    assert r.status_code == 403
    # Unknown dataset.
    r = train_client.get("/train/benchmarks/nope")
    assert r.status_code == 404
    # No response body anywhere may carry the held-out values.
    for resp in [
        train_client.get("/train/benchmarks/reg1", params={"file": "protected/test.csv"}),
        train_client.get("/train/benchmarks/reg1/protected"),
    ]:
        assert "9,9" not in resp.text


def test_train_url_advertised(tmp_path, monkeypatch):
    ds = tmp_path / "benchmarks" / "reg1"
    ds.mkdir(parents=True)
    (ds / "train.csv").write_text("x,y\n1,2\n")
    monkeypatch.setattr(srv, "_DATASETS_DIR", tmp_path)
    monkeypatch.setattr(srv, "_UPLOAD_PORT", 8766)
    monkeypatch.setattr(srv, "_UPLOAD_HOST", "127.0.0.1")

    caps = srv.get_capabilities()
    assert caps["train_download_url"] == "http://127.0.0.1:8766/train/{dataset}"

    summary = srv.get_data_summary("benchmarks/reg1")
    assert summary["train_url"] == "http://127.0.0.1:8766/train/benchmarks/reg1"

    # With the sidecar disabled, no URL is advertised.
    monkeypatch.setattr(srv, "_UPLOAD_PORT", 0)
    assert srv.get_capabilities()["train_download_url"] == "disabled"
    assert "train_url" not in srv.get_data_summary("benchmarks/reg1")
