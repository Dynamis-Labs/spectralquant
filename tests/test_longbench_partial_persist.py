"""Tests for the LongBench partial-method persistence + recovery merger.

These tests do NOT spin up the model or HF datasets; they exercise the
filesystem-only contract: ``_write_method_partial_record`` writes a shard
that ``scripts/merge_longbench_partials.py`` can recombine into a single
JSON.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "merge_longbench_partials.py"


def _import_run_longbench():
    sys.path.insert(0, str(REPO_ROOT))
    from experiments import run_longbench  # noqa: WPS433  (test-only import)
    return run_longbench


def test_write_method_partial_record_creates_shard(tmp_path: Path) -> None:
    rl = _import_run_longbench()

    class _StubStatus:
        def __init__(self, sd: Path) -> None:
            self.status_dir = sd
            self.run_id = "rid"
            self.model = "Qwen/Qwen2.5-7B"

    sd = tmp_path / "status"
    (sd / rl.PARTIAL_SUBDIR).mkdir(parents=True, exist_ok=True)
    status = _StubStatus(sd)

    rec = {
        "label": "fp16",
        "avg_bits": 16.0,
        "evidence_ids": ["RUN-LONGBENCH-FP16"],
        "tasks": {"qasper": {"f1": 0.42, "n": 50}},
    }
    rl._write_method_partial_record(status, method="fp16", record=rec)

    shard = sd / rl.PARTIAL_SUBDIR / "method__fp16.json"
    assert shard.exists(), f"shard not written: {shard}"
    blob = json.loads(shard.read_text())
    assert blob["method"] == "fp16"
    assert blob["paper_valid"] is False
    assert blob["partial"] is True
    assert blob["kind"] == "longbench_partial_method_record"
    assert blob["record"]["tasks"]["qasper"]["f1"] == pytest.approx(0.42)


def test_write_method_partial_record_no_status_is_noop(tmp_path: Path) -> None:
    rl = _import_run_longbench()
    # Should not raise.
    rl._write_method_partial_record(None, method="fp16", record={"a": 1})


def _write_shard(d: Path, method: str, rec: dict) -> None:
    payload = {
        "kind": "longbench_partial_method_record",
        "method": method,
        "paper_valid": False,
        "partial": True,
        "record": rec,
    }
    (d / f"method__{method}.json").write_text(json.dumps(payload))


def test_merge_two_methods_marks_partial(tmp_path: Path) -> None:
    pd = tmp_path / "partial"
    pd.mkdir()
    _write_shard(pd, "fp16", {"label": "fp16", "tasks": {"qasper": 0.3}})
    _write_shard(pd, "spectralquant_v2",
                 {"label": "spectralquant_v2", "tasks": {"qasper": 0.25}})
    out = tmp_path / "merged.json"
    rc = subprocess.call([
        sys.executable, str(SCRIPT),
        "--partial-dir", str(pd),
        "--run-id", "rid",
        "--model", "Qwen/Qwen2.5-7B",
        "--avg-bits", "3",
        "--output", str(out),
        # no --paper-valid
    ])
    assert rc == 0
    blob = json.loads(out.read_text())
    assert blob["family"] == "longbench"
    assert set(blob["methods"].keys()) == {"fp16", "spectralquant_v2"}
    assert blob["paper_valid"] is False
    assert blob["partial"] is True
    assert blob["mode"] == "full_partial"


def test_merge_three_methods_with_flag_is_paper_valid(tmp_path: Path) -> None:
    pd = tmp_path / "partial"
    pd.mkdir()
    _write_shard(pd, "fp16", {"label": "fp16", "tasks": {}})
    _write_shard(pd, "spectralquant_v2", {"label": "spectralquant_v2", "tasks": {}})
    _write_shard(pd, "turboquant", {"label": "turboquant", "tasks": {}})
    out = tmp_path / "merged.json"
    rc = subprocess.call([
        sys.executable, str(SCRIPT),
        "--partial-dir", str(pd),
        "--run-id", "rid",
        "--model", "Qwen/Qwen2.5-7B",
        "--avg-bits", "3",
        "--output", str(out),
        "--paper-valid",
    ])
    assert rc == 0
    blob = json.loads(out.read_text())
    assert blob["paper_valid"] is True
    assert blob["partial"] is False
    assert blob["mode"] == "full"


def test_merge_three_methods_without_flag_stays_partial(tmp_path: Path) -> None:
    pd = tmp_path / "partial"
    pd.mkdir()
    _write_shard(pd, "fp16", {"label": "fp16", "tasks": {}})
    _write_shard(pd, "spectralquant_v2", {"label": "spectralquant_v2", "tasks": {}})
    _write_shard(pd, "turboquant", {"label": "turboquant", "tasks": {}})
    out = tmp_path / "merged.json"
    rc = subprocess.call([
        sys.executable, str(SCRIPT),
        "--partial-dir", str(pd),
        "--run-id", "rid",
        "--model", "Qwen/Qwen2.5-7B",
        "--avg-bits", "3",
        "--output", str(out),
    ])
    assert rc == 0
    blob = json.loads(out.read_text())
    assert blob["paper_valid"] is False
