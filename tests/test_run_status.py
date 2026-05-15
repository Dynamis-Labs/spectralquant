"""Tests for ``experiments/run_status.py``.

Covers the heartbeat / progress artifact contract:

* sanitization redacts known token literals and env-var values;
* atomic JSON writes (no partial files at the canonical path);
* StatusWriter writes ``status.json`` and appends ``events.jsonl``;
* emit_failure captures traceback + sanitized stdout/stderr tails;
* configure_persistent_hf_cache sets the expected env vars and creates
  the cache subdirectories on the volume mount;
* derive_run_id matches the result-JSON naming.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_STATUS_PATH = REPO_ROOT / "experiments" / "run_status.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "experiments.run_status", str(RUN_STATUS_PATH)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiments.run_status"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run_status():
    return _load_module()


# --- sanitization --------------------------------------------------------


def test_sanitize_text_redacts_env_secret(run_status, monkeypatch):
    secret = "abcdef0123456789ABCDEF"
    monkeypatch.setenv("HF_TOKEN", secret)
    out = run_status.sanitize_text(f"trace: {secret} end")
    assert secret not in out
    assert "[REDACTED:HF_TOKEN]" in out


def test_sanitize_text_redacts_token_literal(run_status, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    out = run_status.sanitize_text("hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaa here")
    assert "hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in out
    assert "[REDACTED:token-literal]" in out


def test_sanitize_text_skips_short_values(run_status, monkeypatch):
    monkeypatch.setenv("API_KEY", "abc")
    text = "abc abc abc"
    assert run_status.sanitize_text(text) == text


def test_sanitize_text_handles_empty(run_status):
    assert run_status.sanitize_text("") == ""
    assert run_status.sanitize_text(None) is None  # type: ignore[arg-type]


# --- atomic write ---------------------------------------------------------


def test_atomic_write_json_creates_file(run_status, tmp_path):
    target = tmp_path / "nest" / "status.json"
    payload = {"stage": "start", "value": 1}
    run_status.atomic_write_json(target, payload)
    assert target.is_file()
    loaded = json.loads(target.read_text())
    assert loaded == payload
    # No leftover tempfile.
    leftover = [p for p in target.parent.iterdir() if p.name.startswith(".tmp_")]
    assert leftover == []


def test_atomic_write_json_overwrites_existing(run_status, tmp_path):
    target = tmp_path / "status.json"
    run_status.atomic_write_json(target, {"v": 1})
    run_status.atomic_write_json(target, {"v": 2})
    assert json.loads(target.read_text())["v"] == 2


def test_append_jsonl_appends(run_status, tmp_path):
    target = tmp_path / "events.jsonl"
    run_status.append_jsonl(target, {"a": 1})
    run_status.append_jsonl(target, {"a": 2})
    lines = target.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"a": 2}


# --- StatusWriter ---------------------------------------------------------


def test_status_writer_emit_writes_status_json_and_events(
    run_status, tmp_path,
):
    sd = tmp_path / "status" / "run-x"
    w = run_status.StatusWriter(
        status_dir=sd,
        run_id="run-x",
        commit="deadbeef",
        model="m/m",
        avg_bits=3,
        n_calib=4,
        n_eval=2,
        n_layers_sample=2,
    )
    w.emit("start", message="hello")
    assert (sd / "status.json").is_file()
    assert (sd / "events.jsonl").is_file()
    snap = json.loads((sd / "status.json").read_text())
    assert snap["stage"] == "start"
    assert snap["run_id"] == "run-x"
    assert snap["model"] == "m/m"
    assert snap["commit"] == "deadbeef"
    assert "timestamp" in snap
    assert "host" in snap and snap["host"]
    assert "pid" in snap

    w.emit("calibration_start", message="calibrating")
    snap2 = json.loads((sd / "status.json").read_text())
    assert snap2["stage"] == "calibration_start"
    lines = (sd / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["stage"] == "start"
    assert json.loads(lines[1])["stage"] == "calibration_start"


def test_status_writer_redacts_secrets_in_message_and_stderr(
    run_status, tmp_path, monkeypatch,
):
    sentinel = "leakytokenvalue1234567890abcdef"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    sd = tmp_path / "status" / "run-y"
    w = run_status.StatusWriter(
        status_dir=sd, run_id="run-y", model="m/m", avg_bits=3,
    )
    w.emit(
        "subprocess_progress",
        message=f"saw token {sentinel} in trace",
        stdout_tail=f"foo {sentinel} bar",
        stderr_tail=f"err {sentinel}",
    )
    snap = json.loads((sd / "status.json").read_text())
    assert sentinel not in snap.get("message", "")
    assert sentinel not in snap.get("stdout_tail", "")
    assert sentinel not in snap.get("stderr_tail", "")
    assert "[REDACTED:HF_TOKEN]" in snap["message"]


def test_status_writer_emit_failure_captures_traceback(run_status, tmp_path):
    sd = tmp_path / "status" / "run-fail"
    w = run_status.StatusWriter(
        status_dir=sd, run_id="run-fail", model="m/m", avg_bits=3,
    )
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        w.emit_failure(exc, stdout_tail="stdout", stderr_tail="stderr")
    snap = json.loads((sd / "status.json").read_text())
    assert snap["stage"] == "failure"
    assert "RuntimeError: boom" in snap["error"]
    assert "Traceback" in snap["traceback"]
    assert snap["stdout_tail"] == "stdout"
    assert snap["stderr_tail"] == "stderr"


def test_status_writer_truncates_long_tails(run_status, tmp_path):
    sd = tmp_path / "status" / "run-trim"
    w = run_status.StatusWriter(
        status_dir=sd, run_id="run-trim", model="m/m", avg_bits=3,
    )
    big = "x" * 50000
    w.emit("subprocess_progress", stdout_tail=big, stderr_tail=big)
    snap = json.loads((sd / "status.json").read_text())
    # Default truncation cap is 4000 chars.
    assert len(snap["stdout_tail"]) <= 4000
    assert len(snap["stderr_tail"]) <= 4000


# --- derive_run_id / default_status_dir ----------------------------------


def test_derive_run_id_matches_output_naming(run_status):
    rid = run_status.derive_run_id(
        model="mistralai/Mistral-7B-v0.3",
        avg_bits=3,
        seed=42,
        n_calib=32,
        n_eval=8,
    )
    assert rid == "Mistral-7B-v0.3_b3_calib32_eval8_seed42"


def test_derive_run_id_smoke_prefix(run_status):
    rid = run_status.derive_run_id(
        model="Qwen/Qwen2.5-7B",
        avg_bits=3,
        seed=42,
        n_calib=32,
        n_eval=8,
        smoke=True,
    )
    assert rid.startswith("synthetic_smoke__")
    assert rid.endswith("Qwen2.5-7B_b3_calib32_eval8_seed42")


def test_default_status_dir_results_root(run_status):
    sd = run_status.default_status_dir("/results/three_way", "run-z")
    assert str(sd) == "/results/status/run-z"


def test_default_status_dir_local_results(run_status, tmp_path):
    out = tmp_path / "results" / "three_way"
    sd = run_status.default_status_dir(out, "run-z")
    # Falls back to the ``results`` ancestor of the local path.
    assert str(sd).endswith(f"results/status/run-z")


def test_default_status_dir_no_results_ancestor(run_status, tmp_path):
    sd = run_status.default_status_dir(tmp_path, "run-z")
    assert sd == tmp_path / "status" / "run-z"


# --- configure_persistent_hf_cache ---------------------------------------


def test_configure_persistent_hf_cache_sets_env_vars(run_status, tmp_path):
    env: dict = {}
    paths = run_status.configure_persistent_hf_cache(
        env, volume_mount=str(tmp_path), create=True,
    )
    for var in run_status.HF_CACHE_ENV_VARS:
        assert var in env, f"{var} should be in env"
        assert env[var].startswith(str(tmp_path))
    # Caches must be under the volume mount/hf_cache subdir.
    assert env["HUGGINGFACE_HUB_CACHE"] == str(tmp_path / "hf_cache" / "hub")
    assert env["TRANSFORMERS_CACHE"] == str(tmp_path / "hf_cache" / "hub")
    assert env["HF_DATASETS_CACHE"] == str(tmp_path / "hf_cache" / "datasets")
    assert env["HF_HOME"] == str(tmp_path / "hf_cache" / "home")
    # All directories should have been created.
    for p in paths.values():
        assert Path(p).is_dir(), f"{p} should be a directory"


def test_configure_persistent_hf_cache_idempotent(run_status, tmp_path):
    env = {"PYTHONPATH": "/repo/src"}
    run_status.configure_persistent_hf_cache(env, volume_mount=str(tmp_path))
    first = dict(env)
    run_status.configure_persistent_hf_cache(env, volume_mount=str(tmp_path))
    assert env == first


def test_configure_persistent_hf_cache_no_create(run_status, tmp_path):
    env: dict = {}
    paths = run_status.configure_persistent_hf_cache(
        env, volume_mount=str(tmp_path / "missing"), create=False,
    )
    for p in paths.values():
        # Should not have been created.
        assert not Path(p).exists()


# --- known stages contract -----------------------------------------------


def test_known_stages_includes_required_lifecycle(run_status):
    required = (
        "start",
        "import_ok",
        "model_load_start",
        "model_load_end",
        "dataset_load_start",
        "dataset_load_end",
        "calibration_start",
        "calibration_end",
        "eval_start",
        "eval_end",
        "subprocess_start",
        "subprocess_end",
        "success",
        "failure",
    )
    for stage in required:
        assert stage in run_status.KNOWN_STAGES, (
            f"stage {stage!r} missing from KNOWN_STAGES"
        )


def test_known_stages_includes_modal_runner_stages(run_status):
    """The four early stages emitted by ``run_one`` before/around Popen
    must be part of the public KNOWN_STAGES contract so dashboards know
    they exist.
    """
    for stage in (
        "modal_run_one_entered",
        "subprocess_env_configured",
        "subprocess_starting",
        "subprocess_started",
    ):
        assert stage in run_status.KNOWN_STAGES, (
            f"Modal-runner stage {stage!r} missing from KNOWN_STAGES"
        )


# --- inline-corpus-smoke ---------------------------------------------------


def test_known_stages_includes_inline_corpus_stages(run_status):
    """``--inline-corpus-smoke`` adds two new stages around the inline-corpus
    build (replacing dataset_load_*). Both must be public so dashboards know
    they can occur.
    """
    for stage in ("dataset_inline_start", "dataset_inline_end"):
        assert stage in run_status.KNOWN_STAGES, (
            f"Inline-smoke stage {stage!r} missing from KNOWN_STAGES"
        )


def test_derive_run_id_inline_corpus_smoke_prefix(run_status):
    rid = run_status.derive_run_id(
        model="Qwen/Qwen2.5-0.5B",
        avg_bits=3,
        seed=42,
        n_calib=4,
        n_eval=2,
        inline_corpus_smoke=True,
    )
    assert rid.startswith("inline_corpus_smoke__")
    assert rid.endswith("Qwen2.5-0.5B_b3_calib4_eval2_seed42")
    assert "synthetic_smoke" not in rid


def test_derive_run_id_smoke_and_inline_distinct(run_status):
    """Synthetic-smoke and inline-corpus-smoke must produce distinct run_ids
    so harness-validation runs do not collide with synthetic ones on the
    Modal volume."""
    smoke = run_status.derive_run_id(
        model="m/m", avg_bits=3, seed=1, n_calib=4, n_eval=2, smoke=True,
    )
    inline = run_status.derive_run_id(
        model="m/m", avg_bits=3, seed=1, n_calib=4, n_eval=2,
        inline_corpus_smoke=True,
    )
    assert smoke != inline
    assert smoke.startswith("synthetic_smoke__")
    assert inline.startswith("inline_corpus_smoke__")
