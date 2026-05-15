"""Tests for the four next-stage evaluation harnesses.

Covers:

* schema validity (every new schema is a valid JSON Schema doc)
* synthetic-smoke output of each harness validates against its schema
* dry-run mode writes nothing
* run id / output path determinism
* method-key discipline (unknown labels are rejected)
* token redaction (no real tokens are read; redactor handles known patterns)

These tests intentionally avoid loading any HF model. They run in well
under a second and do not require torch / transformers / datasets.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "schemas"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SRC_DIR = REPO_ROOT / "src"


@pytest.fixture(autouse=True)
def _add_paths(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.syspath_prepend(str(SRC_DIR))


# ---------------------------------------------------------------------------
# Schema validity
# ---------------------------------------------------------------------------

NEW_SCHEMAS = [
    "perplexity.schema.json",
    "longbench.schema.json",
    "generation.schema.json",
    "latency.schema.json",
]


@pytest.mark.parametrize("name", NEW_SCHEMAS)
def test_schema_file_is_valid_json(name: str) -> None:
    path = SCHEMAS_DIR / name
    assert path.is_file()
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj.get("$schema")
    assert obj.get("title")
    assert obj.get("$id")


@pytest.mark.parametrize("name", NEW_SCHEMAS)
def test_schema_is_valid_jsonschema(name: str) -> None:
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator

    path = SCHEMAS_DIR / name
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


# ---------------------------------------------------------------------------
# eval_common helpers
# ---------------------------------------------------------------------------


def test_eval_common_run_id_determinism() -> None:
    from experiments import eval_common

    rid1 = eval_common.derive_run_id(
        "perplexity", "Qwen/Qwen2.5-7B", suffix="x", mode="full",
    )
    rid2 = eval_common.derive_run_id(
        "perplexity", "Qwen/Qwen2.5-7B", suffix="x", mode="full",
    )
    assert rid1 == rid2
    rid3 = eval_common.derive_run_id(
        "perplexity", "Qwen/Qwen2.5-7B", suffix="x",
        mode="synthetic_smoke",
    )
    assert rid3.startswith("synthetic_smoke__")
    assert rid3 != rid1


def test_eval_common_assert_method_keys_accepts_known() -> None:
    from experiments import eval_common
    eval_common.assert_method_keys({"fp16": {}, "spectralquant_v2": {}})


def test_eval_common_assert_method_keys_rejects_unknown() -> None:
    from experiments import eval_common
    with pytest.raises(ValueError, match="Unknown method keys"):
        eval_common.assert_method_keys({"fp16": {}, "made_up": {}})


def test_eval_common_sanitize_text_redacts_token_literals() -> None:
    from experiments import eval_common
    msg = "auth=hf_aaaaaaaaaaaaaaaaaaaaaaaaaa hello"
    redacted = eval_common.sanitize_text(msg)
    assert "hf_aaaaa" not in redacted
    assert "[REDACTED:token-literal]" in redacted


def test_eval_common_inline_corpus_size_and_uniqueness() -> None:
    from experiments import eval_common
    out = eval_common.build_inline_corpus(20)
    assert len(out) == 20
    # Deterministic: every entry has a stable suffix.
    for i, s in enumerate(out):
        assert f"[inline_smoke#{i:03d}]" in s


# ---------------------------------------------------------------------------
# Per-harness synthetic-smoke output validates against its schema.
# ---------------------------------------------------------------------------


def _validate(payload: dict, schema_name: str) -> None:
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator
    schema = json.loads((SCHEMAS_DIR / schema_name).read_text(encoding="utf-8"))
    errs = list(Draft202012Validator(schema).iter_errors(payload))
    assert not errs, "\n".join(
        f"  - {list(e.absolute_path)}: {e.message}" for e in errs
    )


def test_perplexity_synthetic_smoke_payload_validates(tmp_path: Path) -> None:
    from experiments import run_perplexity, eval_common

    args = run_perplexity.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--output-dir", str(tmp_path),
        "--synthetic-smoke",
        "--avg-bits", "3",
        "--methods", "fp16", "spectralquant_v2", "turboquant",
    ])
    methods = run_perplexity.run_synthetic_smoke(args)
    payload = run_perplexity.build_payload(
        args,
        ["--model", "mistralai/Mistral-7B-v0.3"],
        eval_common.MODE_SYNTHETIC_SMOKE,
        methods,
        eval_corpus="synthetic",
        n_eval_sequences=args.n_eval_sequences,
    )
    _validate(payload, "perplexity.schema.json")
    assert payload["family"] == "perplexity"
    assert payload["paper_valid"] is False
    assert payload["mode"] == "synthetic_smoke"
    # Methods present must include all requested.
    assert set(payload["methods"].keys()) == {"fp16", "spectralquant_v2", "turboquant"}


def test_longbench_synthetic_smoke_payload_validates(tmp_path: Path) -> None:
    from experiments import run_longbench, eval_common

    args = run_longbench.parse_args([
        "--model", "Qwen/Qwen2.5-7B",
        "--output-dir", str(tmp_path),
        "--synthetic-smoke",
        "--methods", "fp16", "spectralquant_v2",
        "--subset", "minimal",
        "--n-per-task", "5",
    ])
    methods = run_longbench.run_synthetic_smoke(args)
    payload = run_longbench.build_payload(
        args,
        ["--model", "Qwen/Qwen2.5-7B"],
        eval_common.MODE_SYNTHETIC_SMOKE,
        methods,
    )
    _validate(payload, "longbench.schema.json")
    assert payload["family"] == "longbench"
    assert payload["paper_valid"] is False
    assert payload["subset"]["name"] == "minimal"
    assert payload["tasks"] == run_longbench.SUBSETS["minimal"]


def test_generation_synthetic_smoke_payload_validates(tmp_path: Path) -> None:
    from experiments import run_generation, eval_common

    args = run_generation.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--output-dir", str(tmp_path),
        "--synthetic-smoke",
        "--methods", "fp16", "spectralquant_v2",
    ])
    methods = run_generation.run_synthetic_smoke(args)
    payload = run_generation.build_payload(
        args,
        ["--model", "mistralai/Mistral-7B-v0.3"],
        eval_common.MODE_SYNTHETIC_SMOKE,
        methods,
    )
    _validate(payload, "generation.schema.json")
    assert payload["family"] == "generation"
    assert payload["paper_valid"] is False
    assert len(payload["prompts"]) == len(eval_common.DEFAULT_GENERATION_PROMPTS)


def test_latency_synthetic_smoke_payload_validates(tmp_path: Path) -> None:
    from experiments import run_latency, eval_common

    args = run_latency.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--output-dir", str(tmp_path),
        "--synthetic-smoke",
        "--methods", "fp16", "spectralquant_v2", "turboquant",
        "--batch-sizes", "1,2",
        "--context-lengths", "512,1024",
    ])
    methods = run_latency.run_synthetic_smoke(args)
    payload = run_latency.build_payload(
        args,
        ["--model", "mistralai/Mistral-7B-v0.3"],
        eval_common.MODE_SYNTHETIC_SMOKE,
        methods,
        timer_label="synthetic",
    )
    _validate(payload, "latency.schema.json")
    assert payload["family"] == "latency"
    assert payload["paper_valid"] is False
    # Each method must list 4 operating points (2 batch x 2 ctx).
    for m in ("fp16", "spectralquant_v2", "turboquant"):
        assert len(payload["methods"][m]["operating_points"]) == 4


# ---------------------------------------------------------------------------
# Subprocess invocations of the harnesses (synthetic-smoke writes a JSON).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script,extra_args", [
    ("run_perplexity.py", ["--avg-bits", "3", "--methods", "fp16"]),
    ("run_longbench.py",
     ["--avg-bits", "3", "--methods", "fp16", "--subset", "minimal", "--n-per-task", "5"]),
    ("run_generation.py", ["--avg-bits", "3", "--methods", "fp16"]),
    ("run_latency.py",
     ["--avg-bits", "3", "--methods", "fp16",
      "--batch-sizes", "1", "--context-lengths", "256"]),
])
def test_harness_synthetic_smoke_subprocess_writes_json(
    tmp_path: Path, script: str, extra_args: list
) -> None:
    """Each harness's synthetic-smoke produces exactly one JSON file."""
    cmd = [
        sys.executable, str(EXPERIMENTS_DIR / script),
        "--model", "mistralai/Mistral-7B-v0.3",
        "--output-dir", str(tmp_path),
        "--synthetic-smoke",
        *extra_args,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(SRC_DIR) + os.pathsep + str(REPO_ROOT)
        + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    )
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 1, f"expected 1 output, got {[f.name for f in files]}"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["paper_valid"] is False
    assert payload["mode"] == "synthetic_smoke"


@pytest.mark.parametrize("script,extra_args", [
    ("run_perplexity.py", ["--avg-bits", "3", "--methods", "fp16"]),
    ("run_longbench.py",
     ["--avg-bits", "3", "--methods", "fp16", "--subset", "minimal", "--n-per-task", "5"]),
    ("run_generation.py", ["--avg-bits", "3", "--methods", "fp16"]),
    ("run_latency.py", ["--avg-bits", "3", "--methods", "fp16"]),
])
def test_harness_dry_run_writes_nothing(
    tmp_path: Path, script: str, extra_args: list
) -> None:
    cmd = [
        sys.executable, str(EXPERIMENTS_DIR / script),
        "--model", "mistralai/Mistral-7B-v0.3",
        "--output-dir", str(tmp_path),
        "--dry-run",
        *extra_args,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(SRC_DIR) + os.pathsep + str(REPO_ROOT)
        + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    )
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert list(tmp_path.glob("*.json")) == []


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def test_launcher_build_command_perplexity() -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import launch_modal_eval as L
    finally:
        sys.path.pop(0)
    cfg = L.EvalRunConfig(
        family="perplexity",
        model="mistralai/Mistral-7B-v0.3",
        avg_bits=3,
        methods=("fp16",),
        smoke=True,
    )
    cmd = L.build_command(cfg)
    assert "--synthetic-smoke" in cmd
    assert "--model" in cmd and "mistralai/Mistral-7B-v0.3" in cmd
    assert any(p.endswith("run_perplexity.py") for p in cmd)


def test_launcher_rejects_unknown_family() -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import launch_modal_eval as L
    finally:
        sys.path.pop(0)
    cfg = L.EvalRunConfig(family="bogus", model="x")
    with pytest.raises(ValueError, match="Unknown family"):
        L.build_command(cfg)


def test_launcher_rejects_smoke_plus_dry_run() -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import launch_modal_eval as L
    finally:
        sys.path.pop(0)
    cfg = L.EvalRunConfig(
        family="perplexity", model="x", smoke=True, dry_run=True,
    )
    with pytest.raises(ValueError):
        L.build_command(cfg)


def _import_launcher():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import launch_modal_eval as L  # type: ignore[import-not-found]
        return L
    finally:
        sys.path.pop(0)


def test_family_timeout_table_has_longbench_at_least_4h() -> None:
    """Paper-valid LongBench must not be killed at the old 90-min cap.

    The 2026-04-30 incident (call fc-01KQEQT92W82WXJN3JEEH0Y6RR)
    killed a healthy `spectralquant_v2` LongBench run at the 5400 s
    Modal kill-switch. The fix raises the longbench family default to
    >= 4 h; this test pins that floor so a future refactor cannot
    silently regress it.
    """
    L = _import_launcher()
    assert "longbench" in L.FAMILY_TIMEOUT_SEC
    assert L.FAMILY_TIMEOUT_SEC["longbench"] >= 4 * 60 * 60
    # Default (= max of family caps) must therefore also be >= 4 h.
    assert L.DEFAULT_TIMEOUT_SEC >= 4 * 60 * 60
    # And below the hard ceiling.
    assert L.DEFAULT_TIMEOUT_SEC <= L.MAX_TIMEOUT_SEC


def test_resolve_timeout_explicit_wins() -> None:
    L = _import_launcher()
    out = L.resolve_timeout_sec(
        family="longbench", explicit=900, env={
            "SPECTRALQUANT_MODAL_TIMEOUT_LONGBENCH_SEC": "1234",
            "SPECTRALQUANT_MODAL_TIMEOUT_SEC": "5678",
        },
    )
    assert out == 900


def test_resolve_timeout_family_env_wins_over_global() -> None:
    L = _import_launcher()
    out = L.resolve_timeout_sec(
        family="longbench", explicit=None, env={
            "SPECTRALQUANT_MODAL_TIMEOUT_LONGBENCH_SEC": "12345",
            "SPECTRALQUANT_MODAL_TIMEOUT_SEC": "99999",
        },
    )
    assert out == 12345


def test_resolve_timeout_global_env_wins_over_family_default() -> None:
    L = _import_launcher()
    out = L.resolve_timeout_sec(
        family="longbench", explicit=None, env={
            "SPECTRALQUANT_MODAL_TIMEOUT_SEC": "7777",
        },
    )
    assert out == 7777


def test_resolve_timeout_falls_back_to_family_default() -> None:
    L = _import_launcher()
    out = L.resolve_timeout_sec(
        family="longbench", explicit=None, env={},
    )
    assert out == L.FAMILY_TIMEOUT_SEC["longbench"]


def test_resolve_timeout_unknown_family_falls_back_to_default() -> None:
    L = _import_launcher()
    out = L.resolve_timeout_sec(
        family="bogus", explicit=None, env={},
    )
    assert out == L.DEFAULT_TIMEOUT_SEC


def test_resolve_timeout_rejects_too_small() -> None:
    L = _import_launcher()
    with pytest.raises(ValueError):
        L.resolve_timeout_sec(family="longbench", explicit=10)


def test_resolve_timeout_rejects_too_large() -> None:
    L = _import_launcher()
    with pytest.raises(ValueError):
        L.resolve_timeout_sec(
            family="longbench", explicit=L.MAX_TIMEOUT_SEC + 1,
        )


def test_resolve_timeout_rejects_garbage_env() -> None:
    L = _import_launcher()
    with pytest.raises(ValueError):
        L.resolve_timeout_sec(
            family="longbench", explicit=None,
            env={"SPECTRALQUANT_MODAL_TIMEOUT_LONGBENCH_SEC": "not-a-number"},
        )


def test_resolve_timeout_rejects_negative_env() -> None:
    L = _import_launcher()
    with pytest.raises(ValueError):
        L.resolve_timeout_sec(
            family="longbench", explicit=None,
            env={"SPECTRALQUANT_MODAL_TIMEOUT_SEC": "-1"},
        )


def test_launcher_cli_timeout_default_is_none() -> None:
    """``--timeout-sec`` must default to None so the resolver picks
    the per-family value at build time. A hard-coded default would
    re-introduce the 90-min cap that killed
    ``fc-01KQEQT92W82WXJN3JEEH0Y6RR``.
    """
    L = _import_launcher()
    parser = L.build_parser()
    ns = parser.parse_args([
        "--family", "longbench", "--model", "Qwen/Qwen2.5-7B",
    ])
    assert ns.timeout_sec is None


def test_launcher_status_dir_is_per_family() -> None:
    """The launcher's in-Modal wrapper writes status under
    ``/results/status/<family>/<run_id>/``. The runbook polling
    snippets rely on this exact prefix; a regression that drops the
    family component would break operator polling.
    """
    L = _import_launcher()
    src = (REPO_ROOT / "scripts" / "launch_modal_eval.py").read_text()
    # The default status parent is built from the family.
    assert 'f"/results/status/{cfg.family}"' in src
    # And the printed poll command in main_entry includes the family.
    assert "/results/status/{cfg.family}/" in src


def test_runbook_status_paths_include_family_component() -> None:
    """The polling commands in the runbook must use the
    ``/results/status/longbench/<run_id>/`` form for LongBench, NOT
    the legacy ``/results/status/<run_id>/`` form (which is from the
    three-way launcher and does not apply to launch_modal_eval.py).
    """
    runbook = REPO_ROOT / "docs" / "execution_audit_and_modal_runbook.md"
    text = runbook.read_text()
    # Expect the corrected path to appear in the LongBench polling
    # snippets at least twice (calibration section + generation
    # section).
    assert text.count("/results/status/longbench/<run_id>/") >= 2


def test_launcher_dry_run_subprocess() -> None:
    """The launcher --dry-run must print a command and exit 0 without
    requiring modal or any GPU."""
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "launch_modal_eval.py"),
        "--family", "perplexity",
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--methods", "fp16",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "dry-run" in proc.stdout
    assert "run_perplexity.py" in proc.stdout


# ---------------------------------------------------------------------------
# Paper-valid discipline: smoke modes must NOT produce paper_valid=True.
# ---------------------------------------------------------------------------


def test_no_synthetic_smoke_payload_is_paper_valid(tmp_path: Path) -> None:
    """Regression: every synthetic-smoke output must mark itself as
    not-paper-valid. This is a load-bearing invariant for the evidence
    catalog discipline; a regression here would let smoke output enter
    the catalog as if it were real evidence.
    """
    from experiments import run_perplexity, run_longbench, run_generation, run_latency, eval_common

    cases = [
        (run_perplexity, [
            "--model", "X/Y", "--output-dir", str(tmp_path),
            "--synthetic-smoke", "--avg-bits", "3", "--methods", "fp16",
        ], "perplexity.schema.json", lambda a, m, argv: run_perplexity.build_payload(
            a, argv, eval_common.MODE_SYNTHETIC_SMOKE, m,
            eval_corpus="synthetic",
            n_eval_sequences=a.n_eval_sequences,
        )),
        (run_longbench, [
            "--model", "X/Y", "--output-dir", str(tmp_path),
            "--synthetic-smoke", "--avg-bits", "3", "--methods", "fp16",
            "--subset", "minimal", "--n-per-task", "5",
        ], "longbench.schema.json", lambda a, m, argv: run_longbench.build_payload(
            a, argv, eval_common.MODE_SYNTHETIC_SMOKE, m,
        )),
        (run_generation, [
            "--model", "X/Y", "--output-dir", str(tmp_path),
            "--synthetic-smoke", "--avg-bits", "3", "--methods", "fp16",
        ], "generation.schema.json", lambda a, m, argv: run_generation.build_payload(
            a, argv, eval_common.MODE_SYNTHETIC_SMOKE, m,
        )),
        (run_latency, [
            "--model", "X/Y", "--output-dir", str(tmp_path),
            "--synthetic-smoke", "--avg-bits", "3", "--methods", "fp16",
        ], "latency.schema.json", lambda a, m, argv: run_latency.build_payload(
            a, argv, eval_common.MODE_SYNTHETIC_SMOKE, m,
            timer_label="synthetic",
        )),
    ]
    for module, argv, schema, build in cases:
        args = module.parse_args(argv)
        methods = module.run_synthetic_smoke(args)
        payload = build(args, methods, argv)
        _validate(payload, schema)
        assert payload["paper_valid"] is False, module.__name__
        assert any("not paper-valid" in c.lower() for c in payload["caveats"])
