"""Tests for ``experiments/run_three_way.py`` (spec §13.5).

Scope:

* CLI help exits cleanly.
* ``--dry-run`` validates args, prints plan, writes no JSON.
* ``--synthetic-smoke`` writes a schema-valid JSON deterministically.
* Skip-if-exists is the default; ``--force`` overwrites.
* The result file validates against ``schemas/three_way_result.schema.json``.
* Secrets are never echoed (the script must report presence only).
* The full HF model path raises ``NotImplementedError`` in this slice.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SCHEMAS_DIR = REPO_ROOT / "schemas"
SCRIPT_PATH = EXPERIMENTS_DIR / "run_three_way.py"
THREE_WAY_SCHEMA = SCHEMAS_DIR / "three_way_result.schema.json"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_three_way", str(SCRIPT_PATH)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_three_way"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def harness():
    return _load_module()


# --- CLI help --------------------------------------------------------------


def test_cli_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "three-way" in result.stdout.lower()
    assert "--dry-run" in result.stdout
    assert "--synthetic-smoke" in result.stdout
    assert "--force" in result.stdout


def test_cli_missing_required_args_fails():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--model" in result.stderr or "required" in result.stderr.lower()


# --- argparse object --------------------------------------------------------


def test_parse_args_basic(harness):
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/x",
    ])
    assert args.model == "mistralai/Mistral-7B-v0.3"
    assert args.avg_bits == 3
    assert args.dry_run is False
    assert args.synthetic_smoke is False
    assert args.skip_if_exists is True
    assert args.force is False


def test_parse_args_smoke_alias(harness):
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/x",
        "--smoke",
    ])
    assert args.synthetic_smoke is True


def test_parse_args_force_disables_skip(harness):
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/x",
        "--force",
    ])
    assert args.force is True
    assert args.skip_if_exists is False


def test_parse_args_validates_wf_bounds(harness):
    with pytest.raises(SystemExit):
        harness.parse_args([
            "--model", "x",
            "--avg-bits", "3",
            "--n-calib", "4",
            "--n-eval", "2",
            "--n-layers-sample", "2",
            "--output-dir", "/tmp/x",
            "--wf-min-bits", "5",
            "--wf-max-bits", "2",
        ])


# --- dry-run ----------------------------------------------------------------


def test_dry_run_writes_no_json(tmp_path, harness):
    rc = harness.main([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
        "--dry-run",
    ])
    assert rc == 0
    # No JSON files should have been written under output-dir.
    files = list(tmp_path.rglob("*.json"))
    assert files == [], f"dry-run wrote files: {files}"


def test_dry_run_payload_validates(harness):
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/x",
        "--dry-run",
    ])
    payload = harness.build_payload(args, smoke=None)
    # Validation must succeed (raises on failure).
    pytest.importorskip("jsonschema")
    harness._validate_payload(payload, THREE_WAY_SCHEMA)


# --- synthetic-smoke --------------------------------------------------------


def test_synthetic_smoke_writes_schema_valid_json(tmp_path, harness):
    pytest.importorskip("torch")
    rc = harness.main([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
        "--synthetic-smoke",
    ])
    assert rc == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1, files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["mode"] == "synthetic-smoke"
    assert "v2_allocation_metadata" in payload
    # Re-validate via the helper to lock in schema compatibility.
    pytest.importorskip("jsonschema")
    harness._validate_payload(payload, THREE_WAY_SCHEMA)


# --- resume / force ---------------------------------------------------------


def test_skip_if_exists_default(tmp_path, harness):
    pytest.importorskip("torch")
    args = [
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
        "--synthetic-smoke",
    ]
    assert harness.main(args) == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    mtime_before = files[0].stat().st_mtime_ns

    # Second run should skip and leave the file untouched.
    assert harness.main(args) == 0
    files_after = list(tmp_path.glob("*.json"))
    assert len(files_after) == 1
    assert files_after[0].stat().st_mtime_ns == mtime_before


def test_force_overwrites(tmp_path, harness):
    pytest.importorskip("torch")
    base = [
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
        "--synthetic-smoke",
    ]
    assert harness.main(base) == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    mtime_before = files[0].stat().st_mtime_ns

    # Force a re-write; mtime must advance.
    assert harness.main(base + ["--force"]) == 0
    files_after = list(tmp_path.glob("*.json"))
    assert len(files_after) == 1
    assert files_after[0].stat().st_mtime_ns >= mtime_before


def test_resume_alias_keeps_skip_if_exists_on(harness):
    args = harness.parse_args([
        "--model", "x",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/x",
        "--no-skip-if-exists",
        "--resume",
    ])
    # --resume re-asserts skip even after --no-skip-if-exists.
    assert args.skip_if_exists is True


# --- secret-leak protection -------------------------------------------------


def test_no_secret_value_in_dry_run_output(tmp_path, monkeypatch):
    sentinel = "DO_NOT_LEAK_RUN_THREE_WAY_3a9f8c2e1b"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    monkeypatch.setenv("MODAL_TOKEN_SECRET", sentinel + "_modal")
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--model", "mistralai/Mistral-7B-v0.3",
            "--avg-bits", "3",
            "--n-calib", "4",
            "--n-eval", "2",
            "--n-layers-sample", "2",
            "--output-dir", str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert sentinel not in combined, "harness leaked HF_TOKEN value"
    assert sentinel + "_modal" not in combined, (
        "harness leaked MODAL_TOKEN_SECRET value"
    )
    # But the *names* should be reported, marked as 'set'.
    assert "HF_TOKEN" in combined
    assert "set" in combined


def test_no_secret_value_in_smoke_output(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    sentinel = "ANOTHER_SECRET_VALUE_42abc_runthreeway"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--model", "mistralai/Mistral-7B-v0.3",
            "--avg-bits", "3",
            "--n-calib", "4",
            "--n-eval", "2",
            "--n-layers-sample", "2",
            "--output-dir", str(tmp_path),
            "--synthetic-smoke",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert sentinel not in combined
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    assert sentinel not in written[0].read_text(encoding="utf-8")


# --- full HF model path -----------------------------------------------------


def test_full_path_requires_transformers_when_missing(tmp_path, harness):
    """Without transformers/datasets, the full path must raise a clear
    RuntimeError — not a silent failure or a NotImplementedError."""
    pytest.importorskip("torch")
    try:
        import transformers  # noqa: F401
        pytest.skip("transformers is installed; full path is exercised by Modal tests")
    except ImportError:
        pass
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
    ])
    with pytest.raises(RuntimeError, match="transformers"):
        harness.run(args)


def test_experiments_package_importable_from_outside_repo(tmp_path):
    """Regression: ``run_three_way.py`` does ``from experiments import
    model_adapters`` inside ``run_full_hf``. When the script is loaded via
    ``importlib.util.spec_from_file_location`` (or invoked from a cwd
    outside the repo), the repo root is not automatically on ``sys.path``
    and the import must still succeed because the script itself extends
    ``sys.path``. We simulate that by spawning a fresh interpreter from a
    cwd outside the repo and asserting that loading the module + invoking
    ``run`` either reaches the transformers check (RuntimeError) or fully
    succeeds — never ``ModuleNotFoundError: experiments``.
    """
    pytest.importorskip("torch")
    code = (
        "import sys, importlib.util\n"
        "from pathlib import Path\n"
        f"script = Path({str(SCRIPT_PATH)!r})\n"
        "spec = importlib.util.spec_from_file_location('run_three_way', str(script))\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['run_three_way'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "args = mod.parse_args([\n"
        "    '--model', 'mistralai/Mistral-7B-v0.3',\n"
        "    '--avg-bits', '3', '--n-calib', '4', '--n-eval', '2',\n"
        "    '--n-layers-sample', '2',\n"
        f"    '--output-dir', {str(tmp_path)!r},\n"
        "])\n"
        "try:\n"
        "    mod.run(args)\n"
        "    print('OK')\n"
        "except RuntimeError as exc:\n"
        "    print('RUNTIME:', exc)\n"
        "except ModuleNotFoundError as exc:\n"
        "    print('MNF:', exc)\n"
    )
    # Run from a directory outside the repo so the repo root is NOT on
    # the implicit cwd-derived sys.path.
    outside = tmp_path / "outside"
    outside.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(outside),
        check=False,
    )
    combined = result.stdout + result.stderr
    assert "MNF:" not in combined, (
        f"experiments package not importable from outside the repo:\n{combined}"
    )
    assert "ModuleNotFoundError: No module named 'experiments'" not in combined, combined
    # We expect either a clean transformers-RuntimeError or success.
    assert ("RUNTIME:" in combined and "transformers" in combined) or "OK" in combined, combined


# --- atomic-write helper isolation -----------------------------------------


def test_atomic_write_validates_before_rename(tmp_path, harness):
    """A schema-invalid payload must NOT appear at the canonical path."""
    pytest.importorskip("jsonschema")
    bad_path = tmp_path / "bad.json"
    bad_payload = {"not": "a real run"}
    with pytest.raises(ValueError):
        harness.atomic_write_json(
            bad_path, bad_payload, schema_path=THREE_WAY_SCHEMA,
        )
    assert not bad_path.exists()
    # No leftover tempfiles either.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp_")]
    assert leftovers == []


def test_output_path_deterministic(harness):
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/threeway",
        "--seed", "42",
    ])
    p = harness._output_path(args)
    assert p.name == "Mistral-7B-v0.3_b3_calib4_eval2_seed42.json"


def test_synthetic_smoke_filename_prefix(harness):
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/threeway",
        "--synthetic-smoke",
    ])
    p = harness._output_path(args)
    assert p.name.startswith("synthetic_smoke__")


def test_layer_sample_known_models(harness):
    args = harness.parse_args([
        "--model", "Qwen/Qwen2.5-7B",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "4",
        "--output-dir", "/tmp/x",
    ])
    assert args.layer_sample == [0, 3, 6, 9]


# --- CLI: full-path flags (parse only; no execution) -----------------------


def test_calibration_dir_flags_parse(harness, tmp_path):
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
        "--calibration-dir", str(tmp_path / "calib"),
        "--save-calibration",
    ])
    assert args.calibration_dir == tmp_path / "calib"
    assert args.save_calibration is True
    assert args.load_calibration is False


def test_save_load_calibration_mutually_exclusive(harness, tmp_path):
    with pytest.raises(SystemExit):
        harness.parse_args([
            "--model", "mistralai/Mistral-7B-v0.3",
            "--avg-bits", "3",
            "--n-calib", "4",
            "--n-eval", "2",
            "--n-layers-sample", "2",
            "--output-dir", str(tmp_path),
            "--calibration-dir", str(tmp_path / "calib"),
            "--save-calibration",
            "--load-calibration",
        ])


def test_calibration_save_requires_dir(harness, tmp_path):
    with pytest.raises(SystemExit):
        harness.parse_args([
            "--model", "mistralai/Mistral-7B-v0.3",
            "--avg-bits", "3",
            "--n-calib", "4",
            "--n-eval", "2",
            "--n-layers-sample", "2",
            "--output-dir", str(tmp_path),
            "--save-calibration",
        ])


def test_eval_query_tokens_validation(harness, tmp_path):
    with pytest.raises(SystemExit):
        harness.parse_args([
            "--model", "x",
            "--avg-bits", "3",
            "--n-calib", "4",
            "--n-eval", "2",
            "--n-layers-sample", "2",
            "--output-dir", str(tmp_path),
            "--eval-query-tokens", "0",
        ])


def test_dataset_args_flow_through(harness, tmp_path):
    args = harness.parse_args([
        "--model", "x",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
        "--dataset-name", "wikitext",
        "--dataset-config", "wikitext-103-raw-v1",
        "--dataset-split", "train",
        "--eval-query-tokens", "8",
    ])
    assert args.dataset_name == "wikitext"
    assert args.dataset_config == "wikitext-103-raw-v1"
    assert args.dataset_split == "train"
    assert args.eval_query_tokens == 8


# --- Calibration artifact path -------------------------------------------


def test_calibration_artifact_path_deterministic(harness, tmp_path):
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "32",
        "--n-eval", "8",
        "--n-layers-sample", "8",
        "--output-dir", str(tmp_path),
        "--max-calib-tokens", "384",
        "--seed", "42",
        "--calibration-dir", str(tmp_path / "calib"),
        "--save-calibration",
    ])
    base, meta = harness._calibration_artifact_paths(args)
    assert base.name == "Mistral-7B-v0.3_calib32_tok384_seed42"
    assert meta.name == "Mistral-7B-v0.3_calib32_tok384_seed42_meta.json"


# --- Schema-valid full-path payload via stub ------------------------------


def test_full_path_payload_is_schema_valid(harness, tmp_path):
    """Build a payload from a synthetic 'full' result dict and confirm it
    validates against the three-way schema. This pins the contract that
    run_full_hf must satisfy without actually loading a model."""
    pytest.importorskip("jsonschema")
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
    ])
    layer_indices = args.layer_sample
    head_dim = 128
    cosines = [0.9] * len(layer_indices)
    d_effs = [16] * len(layer_indices)
    methods = {
        name: harness._method_record(
            name, layer_indices, cosines, d_effs,
            head_dim, args.avg_bits,
            label="local" if name == "turboquant" else None,
        )
        for name in ("turboquant", "spectralquant_v1", "spectralquant_v2")
    }
    fake_full = {
        "head_dim": head_dim,
        "layer_indices": layer_indices,
        "d_effs": d_effs,
        "d_eff_stats": {"mean": 16.0, "min": 16.0, "max": 16.0},
        "methods": methods,
        "v2_allocation_metadata": {
            "use_water_fill": True,
            "wf_min_bits": 0,
            "wf_max_bits": None,
            "formula_version": "waterfill-v1",
            "per_head": [],
        },
        "model_block_overrides": {
            "name": "mistralai/Mistral-7B-v0.3",
            "layers": 32, "q_heads": 32, "kv_heads": 8,
            "head_dim": 128, "gqa_ratio": 4,
        },
        "software_overrides": {
            "tokenizer_class": "LlamaTokenizerFast",
            "model_class": "MistralForCausalLM",
            "model_revision": "deadbeef",
            "model_type": "mistral",
        },
        "calibration_artifact": str(tmp_path / "calib_base"),
    }
    payload = harness.build_payload(args, smoke=fake_full)
    # Lock the contract: model + software overrides reach the payload.
    assert payload["model"]["name"] == "mistralai/Mistral-7B-v0.3"
    assert payload["software"]["tokenizer_class"] == "LlamaTokenizerFast"
    assert payload["calibration_artifact"] == str(tmp_path / "calib_base")
    assert payload["mode"] == "full"
    harness._validate_payload(payload, THREE_WAY_SCHEMA)


# --- git commit override (provenance without .git on Modal) ---------------


def test_git_commit_env_override_used_in_payload(harness, monkeypatch):
    """Without ``.git``, the env var ``SPECTRALQUANT_GIT_COMMIT`` must be
    threaded into the result JSON so Modal runs carry real provenance."""
    fake_sha = "deadbeef1234567890abcdef0987654321aabbcc"
    monkeypatch.setenv(harness.GIT_COMMIT_ENV, fake_sha)
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/x",
        "--dry-run",
    ])
    payload = harness.build_payload(args, smoke=None)
    assert payload["commit"] == fake_sha


def test_git_commit_cli_override_beats_env(harness, monkeypatch):
    monkeypatch.setenv(harness.GIT_COMMIT_ENV, "aaaaaaa1111111aaaaaaaaaaaaaaaaaaaaaaaaaa")
    cli_sha = "bbbbbbb2222222bbbbbbbbbbbbbbbbbbbbbbbbbb"
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/x",
        "--dry-run",
        "--git-commit", cli_sha,
    ])
    payload = harness.build_payload(args, smoke=None)
    assert payload["commit"] == cli_sha


def test_git_commit_helper_rejects_garbage_env(harness, monkeypatch, tmp_path):
    """Non-hex / too-short env values must not pollute provenance."""
    monkeypatch.setenv(harness.GIT_COMMIT_ENV, "not-a-sha")
    # Point repo_root at a directory without .git so git rev-parse falls
    # through to the placeholder; we then assert the placeholder is
    # returned rather than the bogus env value.
    out = harness._git_commit(repo_root=tmp_path)
    assert out == "0000000"


def test_git_commit_helper_accepts_valid_env(harness, monkeypatch, tmp_path):
    sha = "0123456789abcdef"
    monkeypatch.setenv(harness.GIT_COMMIT_ENV, sha)
    out = harness._git_commit(repo_root=tmp_path)
    assert out == sha


def test_git_commit_helper_override_arg(harness, monkeypatch, tmp_path):
    monkeypatch.delenv(harness.GIT_COMMIT_ENV, raising=False)
    out = harness._git_commit(repo_root=tmp_path, override="cafebabe1234")
    assert out == "cafebabe1234"


def test_synthetic_smoke_records_env_commit(tmp_path, harness, monkeypatch):
    """The synthetic-smoke output JSON must carry the env-forwarded commit."""
    pytest.importorskip("torch")
    fake_sha = "feedface000111222333444555666777888999aa"
    monkeypatch.setenv(harness.GIT_COMMIT_ENV, fake_sha)
    rc = harness.main([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
        "--synthetic-smoke",
    ])
    assert rc == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["commit"] == fake_sha


# --- inline-corpus-smoke (harness-validation-only) ------------------------
#
# These tests pin the contract: the CLI flag parses, the inline corpus
# loader is deterministic, and the result-payload metadata advertises the
# run as harness-validation rather than paper-valid evidence.


def test_parse_args_inline_corpus_smoke(harness):
    args = harness.parse_args([
        "--model", "Qwen/Qwen2.5-0.5B",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/x",
        "--inline-corpus-smoke",
    ])
    assert args.inline_corpus_smoke is True
    assert args.synthetic_smoke is False


def test_parse_args_smoke_inline_combo_rejected(harness):
    with pytest.raises(SystemExit):
        harness.parse_args([
            "--model", "Qwen/Qwen2.5-0.5B",
            "--avg-bits", "3",
            "--n-calib", "4",
            "--n-eval", "2",
            "--n-layers-sample", "2",
            "--output-dir", "/tmp/x",
            "--synthetic-smoke",
            "--inline-corpus-smoke",
        ])


def test_dry_run_takes_precedence_over_inline_corpus_smoke(harness):
    args = harness.parse_args([
        "--model", "Qwen/Qwen2.5-0.5B",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/x",
        "--inline-corpus-smoke",
        "--dry-run",
    ])
    assert args.dry_run is True
    assert args.inline_corpus_smoke is False


def test_inline_corpus_smoke_filename_prefix(harness):
    args = harness.parse_args([
        "--model", "Qwen/Qwen2.5-0.5B",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", "/tmp/x",
        "--inline-corpus-smoke",
    ])
    p = harness._output_path(args)
    assert p.name.startswith("inline_corpus_smoke__")


def test_inline_corpus_loader_is_deterministic(harness):
    a = harness._build_inline_corpus(7)
    b = harness._build_inline_corpus(7)
    assert a == b
    assert len(a) == 7
    # Each entry must be a non-trivial string and unique across the slice
    # (the trailing index suffix guarantees uniqueness).
    assert len({s for s in a}) == 7
    for s in a:
        assert isinstance(s, str)
        assert len(s) >= 32, s


def test_inline_corpus_loader_cycles_for_large_n(harness):
    base_len = len(harness.INLINE_SMOKE_CORPUS)
    out = harness._build_inline_corpus(base_len * 2 + 3)
    assert len(out) == base_len * 2 + 3


def test_inline_corpus_payload_marks_as_not_paper_valid(harness, tmp_path):
    """The result JSON must clearly advertise that inline-corpus runs are
    not paper-valid evidence."""
    pytest.importorskip("jsonschema")
    args = harness.parse_args([
        "--model", "Qwen/Qwen2.5-0.5B",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
        "--inline-corpus-smoke",
    ])
    # Build a stub "full"-shaped result so we can compose the payload
    # without loading a real model. The mode/data block is what we're
    # locking in here.
    layer_indices = args.layer_sample
    head_dim = 128
    cosines = [0.9] * len(layer_indices)
    d_effs = [16] * len(layer_indices)
    methods = {
        name: harness._method_record(
            name, layer_indices, cosines, d_effs,
            head_dim, args.avg_bits,
            label="local" if name == "turboquant" else None,
        )
        for name in ("turboquant", "spectralquant_v1", "spectralquant_v2")
    }
    fake_full = {
        "head_dim": head_dim,
        "layer_indices": layer_indices,
        "d_effs": d_effs,
        "d_eff_stats": {"mean": 16.0, "min": 16.0, "max": 16.0},
        "methods": methods,
        "v2_allocation_metadata": {
            "use_water_fill": True,
            "wf_min_bits": 0,
            "wf_max_bits": None,
            "formula_version": "waterfill-v1",
            "per_head": [],
        },
        "model_block_overrides": {
            "name": "Qwen/Qwen2.5-0.5B",
            "layers": 24, "q_heads": 14, "kv_heads": 2,
            "head_dim": 64, "gqa_ratio": 7,
        },
        "software_overrides": {
            "tokenizer_class": "Qwen2TokenizerFast",
            "model_class": "Qwen2ForCausalLM",
            "model_revision": "abc123",
            "model_type": "qwen2",
        },
    }
    payload = harness.build_payload(args, smoke=fake_full)
    assert payload["mode"] == "inline-corpus-smoke"
    assert payload["paper_valid"] is False
    assert payload["evidence_ids"] == ["RUN-THREEWAY-INLINESMOKE-001"]
    assert payload["data"]["calibration_corpus"] == "inline_smoke"
    assert payload["data"]["eval_corpus"] == "inline_smoke"
    assert payload["run_id"].startswith("inline_corpus_smoke__")
    harness._validate_payload(payload, THREE_WAY_SCHEMA)


def test_full_path_payload_is_paper_valid(harness, tmp_path):
    """Sanity check: a full-mode payload still asserts paper_valid=True
    even after the new field is added."""
    pytest.importorskip("jsonschema")
    args = harness.parse_args([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
    ])
    layer_indices = args.layer_sample
    head_dim = 128
    cosines = [0.9] * len(layer_indices)
    d_effs = [16] * len(layer_indices)
    methods = {
        name: harness._method_record(
            name, layer_indices, cosines, d_effs,
            head_dim, args.avg_bits,
            label="local" if name == "turboquant" else None,
        )
        for name in ("turboquant", "spectralquant_v1", "spectralquant_v2")
    }
    fake_full = {
        "head_dim": head_dim, "layer_indices": layer_indices,
        "d_effs": d_effs,
        "d_eff_stats": {"mean": 16.0, "min": 16.0, "max": 16.0},
        "methods": methods,
        "v2_allocation_metadata": {
            "use_water_fill": True, "wf_min_bits": 0,
            "wf_max_bits": None, "formula_version": "waterfill-v1",
            "per_head": [],
        },
        "model_block_overrides": {
            "name": "mistralai/Mistral-7B-v0.3",
            "layers": 32, "q_heads": 32, "kv_heads": 8,
            "head_dim": 128, "gqa_ratio": 4,
        },
        "software_overrides": {
            "tokenizer_class": "LlamaTokenizerFast",
            "model_class": "MistralForCausalLM",
            "model_revision": "deadbeef",
            "model_type": "mistral",
        },
    }
    payload = harness.build_payload(args, smoke=fake_full)
    assert payload["mode"] == "full"
    assert payload["paper_valid"] is True
    assert payload["evidence_ids"] == ["RUN-THREEWAY-001"]
    assert payload["data"]["calibration_corpus"] == "WikiText-103"
    harness._validate_payload(payload, THREE_WAY_SCHEMA)


def test_synthetic_smoke_is_not_paper_valid(harness, tmp_path):
    """Synthetic-smoke must also report paper_valid=False so downstream
    tooling can filter out harness-only runs uniformly."""
    pytest.importorskip("torch")
    rc = harness.main([
        "--model", "mistralai/Mistral-7B-v0.3",
        "--avg-bits", "3",
        "--n-calib", "4",
        "--n-eval", "2",
        "--n-layers-sample", "2",
        "--output-dir", str(tmp_path),
        "--synthetic-smoke",
    ])
    assert rc == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["mode"] == "synthetic-smoke"
    assert payload["paper_valid"] is False


def test_inline_corpus_smoke_help_advertised():
    """The CLI help text must mention --inline-corpus-smoke so operators
    discover the harness-validation gate without reading source."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "--inline-corpus-smoke" in result.stdout


# --- Calibration coverage validation --------------------------------------


class _StubCalib:
    """Minimal calibrator stand-in: only implements ``.get(li, hi, ht)``."""

    def __init__(self, populated):
        self._d = set(populated)

    def get(self, li, hi, ht):
        return object() if (li, hi, ht) in self._d else None


def test_missing_calibration_entries_returns_empty_when_full(harness):
    """Full coverage → empty missing list."""
    populated = [
        (li, hi, ht)
        for li in [0, 3, 6]
        for hi in range(4)
        for ht in ("key", "value")
    ]
    calib = _StubCalib(populated)
    out = harness.missing_calibration_entries(calib, [0, 3, 6], 4)
    assert out == []


def test_missing_calibration_entries_lists_holes(harness):
    """Holes → returned in iteration order, no duplicates."""
    # Populate everything except L3 H1 ("value") and L6 H0 ("key").
    populated = [
        (li, hi, ht)
        for li in [0, 3, 6]
        for hi in range(2)
        for ht in ("key", "value")
        if not ((li == 3 and hi == 1 and ht == "value")
                or (li == 6 and hi == 0 and ht == "key"))
    ]
    calib = _StubCalib(populated)
    out = harness.missing_calibration_entries(calib, [0, 3, 6], 2)
    assert out == [(3, 1, "value"), (6, 0, "key")]


def test_missing_calibration_entries_total_when_empty(harness):
    """Empty calibrator → every sampled (layer, head, type) reported."""
    calib = _StubCalib([])
    out = harness.missing_calibration_entries(calib, [0, 3], 2)
    # 2 layers × 2 heads × 2 types = 8 entries.
    assert len(out) == 8
    # First triple is (0, 0, "key"), pinning iteration order.
    assert out[0] == (0, 0, "key")
    assert out[-1] == (3, 1, "value")
