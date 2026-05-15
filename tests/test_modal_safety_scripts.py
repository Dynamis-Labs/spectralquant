"""Tests for the lightweight Modal safety scripts.

These tests:
  * import the scripts as modules and exercise their argparse / dispatch logic;
  * never start a model, never call HF, never download anything;
  * never echo any environment variable values.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # required for @dataclass introspection
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preflight():
    return _load_module("preflight_modal", SCRIPTS_DIR / "preflight_modal.py")


@pytest.fixture(scope="module")
def audit():
    return _load_module("audit_results", SCRIPTS_DIR / "audit_results.py")


# --- preflight_modal.py ----------------------------------------------------


def test_preflight_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "preflight_modal.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "preflight" in result.stdout.lower()


def test_preflight_does_not_print_secret_values(monkeypatch):
    sentinel = "DO_NOT_LEAK_THIS_VALUE_12345_abcdefghij"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "preflight_modal.py")],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    combined = result.stdout + result.stderr
    assert sentinel not in combined, "preflight script leaked HF_TOKEN value"


def test_preflight_python_check_passes(preflight):
    report = preflight.Report()
    preflight.check_python_version(report)
    py_check = next(r for r in report.results if r.name == "python_version")
    assert py_check.ok, py_check.message


def test_preflight_env_var_presence_reports_length_only(preflight, monkeypatch):
    sentinel = "abcdef0123456789ZZZZ_supersecret"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    report = preflight.Report()
    preflight.check_env_var_presence(report)
    env_msg = next(r for r in report.results if r.name == "env_HF_TOKEN").message
    assert sentinel not in env_msg
    assert "length=" in env_msg


def test_preflight_env_var_absence_marks_required_as_failed(preflight, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    report = preflight.Report()
    preflight.check_env_var_presence(report)
    hf = next(r for r in report.results if r.name == "env_HF_TOKEN")
    assert not hf.ok
    assert hf.severity == "error"


def test_preflight_secret_path_pattern_matches_dotenv(preflight):
    matched = any(p.search(".env") for p in preflight.SECRET_PATH_PATTERNS)
    assert matched
    matched = any(p.search("modal_token") for p in preflight.SECRET_PATH_PATTERNS)
    assert matched


def test_preflight_disk_space_returns_a_value(preflight, tmp_path):
    report = preflight.Report()
    preflight.check_disk_space(report, str(tmp_path), min_gb=0.0)
    disk = next(r for r in report.results if r.name == "disk_space")
    assert disk.ok


# --- audit_results.py ------------------------------------------------------


def test_audit_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "audit_results.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "audit" in result.stdout.lower()


def test_audit_default_manifest_runs(audit, tmp_path):
    # Use an empty repo root so everything is "missing" (this exercises the
    # presence-check logic without depending on what's checked in).
    a = audit.Audit()
    for group, entries in audit.DEFAULT_MANIFEST.items():
        for entry in entries:
            a.add(group, entry, tmp_path)
    assert len(a.items) > 0
    # All required items should be missing in the empty repo:
    missing = a.missing_required()
    assert len(missing) == sum(
        1 for e in audit.DEFAULT_MANIFEST["three_way_runs"] if e["required"]
    ) + sum(
        1 for e in audit.DEFAULT_MANIFEST["accounting_audit"] if e["required"]
    )


def test_audit_detects_present_file(audit, tmp_path):
    target = tmp_path / "results" / "three_way" / "x.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    a = audit.Audit()
    a.add(
        "three_way_runs",
        {
            "path": "results/three_way/x.json",
            "required": True,
            "label": "test",
        },
        tmp_path,
    )
    assert a.items[0].present
    assert a.items[0].size_bytes == 2
    assert not a.missing_required()


def test_audit_strict_exits_nonzero_when_missing(tmp_path):
    manifest = {
        "g": [
            {"path": "definitely/not/here.json", "required": True, "label": "x"},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "audit_results.py"),
            "--manifest",
            str(manifest_path),
            "--strict",
            "--repo-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1


def test_audit_json_output_mode(tmp_path):
    manifest = {
        "g": [
            {"path": "a.json", "required": False, "label": "a"},
        ],
    }
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "audit_results.py"),
            "--manifest",
            str(manifest_path),
            "--json",
            "--repo-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "items" in payload
    assert payload["summary"]["total"] == 1


def test_audit_invalid_manifest_returns_2(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "audit_results.py"),
            "--manifest",
            str(bad),
            "--repo-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2


# --- gitignore safety ------------------------------------------------------


def test_gitignore_blocks_credential_paths():
    gi = (REPO_ROOT / ".gitignore").read_text()
    for needle in (
        ".env",
        "modal_token",
        "hf_token",
        "secrets/",
        "*.pem",
        "local_results/",
    ):
        assert needle in gi, f".gitignore is missing {needle!r}"


# --- audit_results stale-tmp scan ---------------------------------------


def test_audit_scan_stale_tmp_finds_files(audit, tmp_path):
    """``scan_stale_tmp_files`` must locate ``.tmp_*`` leftovers."""
    target = tmp_path / "results" / "three_way"
    target.mkdir(parents=True)
    (target / ".tmp_abc123").write_text("")  # zero-byte
    (target / ".tmp_xyz").write_bytes(b"partial")
    (target / "real_result.json").write_text("{}")  # not a tempfile
    found = audit.scan_stale_tmp_files(tmp_path)
    paths = [s.path for s in found]
    assert "results/three_way/.tmp_abc123" in paths
    assert "results/three_way/.tmp_xyz" in paths
    assert all("real_result.json" not in p for p in paths)
    sizes = {s.path: s.size_bytes for s in found}
    assert sizes["results/three_way/.tmp_abc123"] == 0
    assert sizes["results/three_way/.tmp_xyz"] == len(b"partial")


def test_audit_scan_stale_tmp_handles_missing_dirs(audit, tmp_path):
    found = audit.scan_stale_tmp_files(tmp_path)
    assert found == []


def test_audit_scan_stale_tmp_recursive(audit, tmp_path):
    """Nested run dirs (e.g. on the Modal volume) should still be scanned."""
    nested = tmp_path / "results" / "three_way" / "run-2026-04-29"
    nested.mkdir(parents=True)
    (nested / ".tmp_nested").write_text("")
    found = audit.scan_stale_tmp_files(tmp_path)
    assert any(s.path.endswith(".tmp_nested") for s in found)


def test_audit_render_includes_stale_section(audit, tmp_path):
    a = audit.Audit()
    a.stale_tmp = [
        audit.StaleTmpFile(
            path="results/three_way/.tmp_zz",
            size_bytes=0,
            mtime=1700000000.0,
        )
    ]
    rendered = a.render_table()
    assert "stale_tmp_files" in rendered
    assert ".tmp_zz" in rendered
    assert "WARNING" in rendered


def test_audit_summary_mentions_stale_count(audit):
    a = audit.Audit()
    a.stale_tmp = [
        audit.StaleTmpFile(path="x/.tmp_a", size_bytes=0, mtime=0.0),
        audit.StaleTmpFile(path="x/.tmp_b", size_bytes=0, mtime=0.0),
    ]
    s = a.summary()
    assert "2 stale" in s


def test_audit_to_dict_includes_stale(audit):
    a = audit.Audit()
    a.stale_tmp = [
        audit.StaleTmpFile(path="x/.tmp_a", size_bytes=5, mtime=42.0),
    ]
    d = a.to_dict()
    assert d["summary"]["stale_tmp_count"] == 1
    assert d["stale_tmp_files"][0]["path"] == "x/.tmp_a"


def test_audit_cli_reports_stale_tmp_in_json(tmp_path):
    """Integration: --json output must include the stale-tmp section."""
    target = tmp_path / "results" / "three_way"
    target.mkdir(parents=True)
    (target / ".tmp_crash").write_text("")
    manifest = {"g": [{"path": "ignored.json", "required": False, "label": "x"}]}
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "audit_results.py"),
            "--manifest", str(manifest_path),
            "--repo-root", str(tmp_path),
            "--json",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"]["stale_tmp_count"] == 1
    paths = [s["path"] for s in payload["stale_tmp_files"]]
    assert any(p.endswith(".tmp_crash") for p in paths)


def test_audit_cli_does_not_delete_by_default(tmp_path):
    target = tmp_path / "results" / "three_way"
    target.mkdir(parents=True)
    stale = target / ".tmp_keep_me"
    stale.write_text("")
    manifest = {"g": [{"path": "ignored.json", "required": False, "label": "x"}]}
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "audit_results.py"),
            "--manifest", str(manifest_path),
            "--repo-root", str(tmp_path),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert stale.exists(), "audit must not delete .tmp_* files by default"


def test_audit_cli_delete_stale_tmp_removes_files(tmp_path):
    target = tmp_path / "results" / "three_way"
    target.mkdir(parents=True)
    stale = target / ".tmp_delete_me"
    stale.write_text("")
    manifest = {"g": [{"path": "ignored.json", "required": False, "label": "x"}]}
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "audit_results.py"),
            "--manifest", str(manifest_path),
            "--repo-root", str(tmp_path),
            "--delete-stale-tmp",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert not stale.exists(), (
        f"--delete-stale-tmp should have removed the file: stderr={result.stderr}"
    )


def test_docs_describe_status_artifact_polling():
    """Both docs must teach operators how to poll the new status JSON."""
    safety = REPO_ROOT / "docs" / "modal_safety_protocol.md"
    runbook = REPO_ROOT / "docs" / "execution_audit_and_modal_runbook.md"
    safety_text = safety.read_text(encoding="utf-8")
    runbook_text = runbook.read_text(encoding="utf-8")
    for path, text in (("modal_safety_protocol.md", safety_text),
                       ("execution_audit_and_modal_runbook.md", runbook_text)):
        assert "status.json" in text, (
            f"docs/{path} must mention the heartbeat status.json artifact"
        )
        assert "/results/status" in text or "results/status" in text, (
            f"docs/{path} must show the canonical status directory path"
        )


def test_modal_safety_protocol_documents_smoke_gate():
    """The protocol must state: no full sweep until HF smoke writes success."""
    safety = (REPO_ROOT / "docs" / "modal_safety_protocol.md").read_text()
    assert "no full sweep" in safety.lower() or "smoke" in safety.lower(), (
        "modal_safety_protocol.md must document the HF-smoke gate before "
        "full sweeps"
    )
    # Be specific: a 'success' status JSON is the gate.
    assert "success" in safety, (
        "modal_safety_protocol.md must reference the 'success' status as "
        "the gate for full sweeps"
    )


def test_modal_safety_protocol_documents_hf_cache():
    """The protocol must explain the persistent HF cache layout."""
    safety = (REPO_ROOT / "docs" / "modal_safety_protocol.md").read_text()
    assert "hf_cache" in safety, (
        "modal_safety_protocol.md must document the /results/hf_cache layout"
    )
    assert "HUGGINGFACE_HUB_CACHE" in safety or "HF_HOME" in safety, (
        "modal_safety_protocol.md must list the HF cache env vars"
    )


def test_env_example_has_no_secret_values():
    example = (REPO_ROOT / ".env.example").read_text()
    for line in example.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "=" in line
        key, _, value = line.partition("=")
        assert value == "" or value.startswith("#"), (
            f".env.example has a non-empty value for {key!r}; "
            "it must contain names only"
        )
