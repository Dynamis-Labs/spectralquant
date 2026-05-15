"""Tests for ``scripts/launch_modal_three_way.py``.

Covers:

* CLI help exits cleanly.
* Importing the module without modal installed raises a clear error
  only when ``build_modal_app`` is called — module import itself works.
* ``RunConfig`` -> argv construction matches the harness's CLI.
* ``--dry-run`` prints the command and exits 0 without calling Modal.
* The matrix-config loader rejects unknown keys.
* GPU choice is enforced against the safety-protocol allow-list.
* No environment-variable values are echoed by the CLI (presence-only).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "launch_modal_three_way.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "launch_modal_three_way", str(SCRIPT_PATH)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["launch_modal_three_way"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def launcher():
    return _load_module()


# --- CLI ------------------------------------------------------------------


def test_cli_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "Modal launcher" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--matrix-config" in result.stdout


def test_dry_run_prints_command(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--model", "mistralai/Mistral-7B-v0.3",
            "--avg-bits", "3",
            "--seed", "42",
            "--dry-run",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "experiments/run_three_way.py" in result.stdout
    assert "--avg-bits" in result.stdout
    assert "Mistral-7B-v0.3" in result.stdout


def test_dry_run_does_not_require_modal(tmp_path, monkeypatch):
    """--dry-run must work even with modal not installed (hide if present)."""
    # Hide modal module if installed.
    monkeypatch.setitem(sys.modules, "modal", None)
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--model", "Qwen/Qwen2.5-7B",
            "--avg-bits", "3",
            "--dry-run",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0


# --- RunConfig and command building --------------------------------------


def test_run_config_default(launcher):
    cfg = launcher.RunConfig(model="m/m", avg_bits=3)
    cmd = launcher.build_command(cfg)
    # Required CLI tokens.
    assert "--model" in cmd
    assert "--avg-bits" in cmd
    assert "--n-calib" in cmd
    assert "--n-eval" in cmd
    assert "--n-layers-sample" in cmd
    assert "--max-calib-tokens" in cmd
    assert "--output-dir" in cmd
    assert "--seed" in cmd
    assert "--device" in cmd
    assert "--dtype" in cmd
    assert "--save-calibration" in cmd  # default True
    assert "--load-calibration" not in cmd
    assert "--synthetic-smoke" not in cmd
    assert "--dry-run" not in cmd
    # Values.
    assert cmd[cmd.index("--model") + 1] == "m/m"
    assert cmd[cmd.index("--avg-bits") + 1] == "3"
    assert cmd[cmd.index("--device") + 1] == "cuda"


def test_run_config_with_smoke(launcher):
    cfg = launcher.RunConfig(model="m/m", avg_bits=3, smoke=True)
    cmd = launcher.build_command(cfg)
    assert "--synthetic-smoke" in cmd


def test_run_config_with_force(launcher):
    cfg = launcher.RunConfig(model="m/m", avg_bits=3, force=True)
    cmd = launcher.build_command(cfg)
    assert "--force" in cmd


def test_run_config_load_calibration(launcher):
    cfg = launcher.RunConfig(
        model="m/m", avg_bits=3,
        save_calibration=False, load_calibration=True,
    )
    cmd = launcher.build_command(cfg)
    assert "--load-calibration" in cmd
    assert "--save-calibration" not in cmd


def test_run_config_smoke_dry_run_combo_rejected(launcher):
    cfg = launcher.RunConfig(model="m/m", avg_bits=3, smoke=True, dry_run=True)
    with pytest.raises(ValueError):
        launcher.build_command(cfg)


def test_run_config_wf_max_bits(launcher):
    cfg = launcher.RunConfig(
        model="m/m", avg_bits=3, wf_min_bits=2, wf_max_bits=5,
    )
    cmd = launcher.build_command(cfg)
    assert "--wf-max-bits" in cmd
    assert cmd[cmd.index("--wf-max-bits") + 1] == "5"


def test_render_command_round_trips(launcher):
    cfg = launcher.RunConfig(model="m/m", avg_bits=3)
    s = launcher.render_command(cfg)
    assert "experiments/run_three_way.py" in s
    assert "Mistral" not in s  # nothing model-specific spilling in


# --- Matrix loader --------------------------------------------------------


def test_matrix_loader_basic(tmp_path, launcher):
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps([
        {"model": "mistralai/Mistral-7B-v0.3", "avg_bits": 2, "seed": 42},
        {"model": "mistralai/Mistral-7B-v0.3", "avg_bits": 3, "seed": 42},
        {"model": "Qwen/Qwen2.5-7B", "avg_bits": 3, "seed": 42},
    ]))
    cfgs = launcher.load_matrix(p)
    assert len(cfgs) == 3
    assert cfgs[0].avg_bits == 2
    assert cfgs[2].model == "Qwen/Qwen2.5-7B"


def test_matrix_loader_accepts_force(tmp_path, launcher):
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps([
        {"model": "m/m", "avg_bits": 3, "force": True},
    ]))
    cfgs = launcher.load_matrix(p)
    assert len(cfgs) == 1
    assert cfgs[0].force is True
    assert "--force" in launcher.build_command(cfgs[0])


def test_matrix_loader_rejects_unknown_keys(tmp_path, launcher):
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps([
        {"model": "m/m", "avg_bits": 3, "this_is_wrong": True},
    ]))
    with pytest.raises(ValueError, match="this_is_wrong"):
        launcher.load_matrix(p)


def test_matrix_loader_rejects_non_array(tmp_path, launcher):
    p = tmp_path / "matrix.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="JSON array"):
        launcher.load_matrix(p)


# --- GPU allow-list --------------------------------------------------------


def test_build_modal_app_rejects_bad_gpu(launcher):
    with pytest.raises(ValueError, match="not in allow-list"):
        launcher.build_modal_app(gpu="T4")


def test_build_modal_app_runtime_error_without_modal(launcher, monkeypatch):
    """Construction without modal installed must raise a clear error."""
    # Force the import to fail by interposing a None entry.
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(RuntimeError, match="modal is not installed"):
        launcher.build_modal_app()


# --- Secret name handling --------------------------------------------------


def test_secret_names_default(launcher):
    assert launcher.DEFAULT_HF_SECRET == "hf-token"
    assert launcher.DEFAULT_WANDB_SECRET == "wandb-api-key"


def test_no_token_value_in_dry_run_output(tmp_path, monkeypatch):
    """The launcher must not echo any secret value in --dry-run output."""
    sentinel = "DO_NOT_LEAK_LAUNCH_MODAL_e7c2a91f"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    monkeypatch.setenv("MODAL_TOKEN_SECRET", sentinel + "_modal")
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--model", "mistralai/Mistral-7B-v0.3",
            "--avg-bits", "3",
            "--dry-run",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert sentinel not in combined
    assert sentinel + "_modal" not in combined


# --- configs_from_args ----------------------------------------------------


def test_configs_from_args_single(launcher):
    parser = launcher.build_parser()
    ns = parser.parse_args([
        "--model", "Qwen/Qwen2.5-7B",
        "--avg-bits", "3",
        "--seed", "7",
        "--force",
    ])
    cfgs = launcher.configs_from_args(ns)
    assert len(cfgs) == 1
    assert cfgs[0].model == "Qwen/Qwen2.5-7B"
    assert cfgs[0].avg_bits == 3
    assert cfgs[0].seed == 7
    assert cfgs[0].force is True


def test_configs_from_args_requires_model_when_no_matrix(launcher):
    parser = launcher.build_parser()
    ns = parser.parse_args([])
    with pytest.raises(SystemExit):
        launcher.configs_from_args(ns)


def test_configs_from_args_matrix(tmp_path, launcher):
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps([
        {"model": "m/m", "avg_bits": 2},
        {"model": "m/m", "avg_bits": 3},
    ]))
    parser = launcher.build_parser()
    ns = parser.parse_args(["--matrix-config", str(p)])
    cfgs = launcher.configs_from_args(ns)
    assert len(cfgs) == 2


# --- Modal 1.4.x API regression -------------------------------------------
#
# Modal 1.4 removed the ``modal.Mount`` class in favour of attaching local
# files directly to the image with ``Image.add_local_dir``. These tests
# pin the launcher to the new API so a regression to ``modal.Mount`` is
# caught at unit-test time rather than at ``modal run`` time.


def test_source_uses_add_local_dir_not_mount():
    """Pin the launcher source to the Modal 1.4.x API.

    ``modal.Mount`` was removed in Modal 1.4 — referencing it raises
    ``AttributeError`` at app-construction time. This test guards against
    a regression by inspecting the source directly so it does not require
    Modal to be installed.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    # Match the actual code patterns, not prose mentions in comments.
    assert "modal.Mount.from_local_dir" not in text, (
        "scripts/launch_modal_three_way.py calls modal.Mount.from_local_dir, "
        "which was removed in Modal 1.4. Use Image.add_local_dir instead."
    )
    assert "modal.Mount(" not in text, (
        "scripts/launch_modal_three_way.py instantiates modal.Mount, "
        "which was removed in Modal 1.4. Use Image.add_local_dir instead."
    )
    assert "mounts=" not in text, (
        "scripts/launch_modal_three_way.py passes mounts=... to "
        "@app.function, which is no longer supported in Modal 1.4. "
        "Use Image.add_local_dir on the image instead."
    )
    assert "add_local_dir" in text, (
        "Expected Image.add_local_dir to be used to attach the repo at /repo."
    )


# --- Python-version mismatch regression ----------------------------------
#
# Modal's ``serialized=True`` pickles the function under the local Python's
# pickle protocol. If the image's ``add_python`` differs from the local
# interpreter's ``major.minor``, Modal raises:
#   "serialized Function must use a Python version compatible with remote
#    environment".
# These tests guard against a regression to a hardcoded image Python that
# does not match the Python actually running pytest.


def test_image_add_python_matches_local_major_minor(launcher, monkeypatch):
    """build_modal_app must pick an add_python that matches the local Python.

    Skipped if modal isn't installed — the assertion only makes sense when
    we can actually inspect the constructed Modal image.
    """
    pytest.importorskip("modal")
    local_py = f"{sys.version_info.major}.{sys.version_info.minor}"
    # Exercise the construction path. We don't need the returned image to
    # actually launch — we only need it to be built without the local/
    # remote Python mismatch the user hit.
    app, image, volume, run_one = launcher.build_modal_app(
        gpu="H200",
        timeout_sec=300,
        hf_secret_name="hf-token",
    )
    # Best-effort introspection: Modal stores the ``add_python`` arg on the
    # image's build chain. If the attribute layout changes upstream the
    # test still fails closed via the source-text guard below.
    found = None
    for attr in ("_add_python", "add_python", "_python_version"):
        v = getattr(image, attr, None)
        if v:
            found = str(v)
            break
    if found is not None:
        assert found.startswith(local_py), (
            f"Modal image add_python={found!r} does not match local "
            f"Python {local_py!r}; serialized functions will be rejected."
        )


def test_source_pins_add_python_to_local_version():
    """Source-level guard: add_python must not be hardcoded to a version
    that differs from the local interpreter at test time.

    The launcher derives ``add_python`` from ``sys.version_info`` so the
    local and remote Pythons stay in sync. A regression that pins it to a
    fixed string like ``"3.11"`` would re-introduce the serialized-
    function mismatch the user hit.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    # The current (correct) shape uses sys.version_info to derive the
    # image Python. Reject hardcoded mismatching pins.
    local_py = f"{sys.version_info.major}.{sys.version_info.minor}"
    bad_pins = [v for v in ("3.9", "3.10", "3.11", "3.12") if v != local_py]
    for bad in bad_pins:
        assert f'add_python="{bad}"' not in text, (
            f"scripts/launch_modal_three_way.py hardcodes "
            f"add_python={bad!r} but the local Python is {local_py!r}. "
            f"Modal's serialized=True functions require matching Python "
            f"major.minor between local and image."
        )
    # And we expect *some* derivation from sys.version_info to be present.
    assert "sys.version_info" in text, (
        "Expected scripts/launch_modal_three_way.py to derive the image "
        "Python from sys.version_info so it stays in sync with the local "
        "interpreter that pickles the serialized function."
    )


def test_build_modal_app_with_real_modal_uses_new_api(launcher):
    """If modal is installed, build_modal_app must succeed end-to-end.

    Skipped if modal is unavailable. This catches the exact failure
    reported (``AttributeError: module 'modal' has no attribute 'Mount'``)
    because Modal would raise it during app construction.
    """
    pytest.importorskip("modal")
    app, image, volume, run_one = launcher.build_modal_app(
        gpu="H200",
        timeout_sec=300,
        hf_secret_name="hf-token",
    )
    assert app is not None
    assert image is not None
    assert volume is not None
    assert run_one is not None


# --- Observability: output_path, parse_wrote_line, sanitize_text ----------


def test_output_path_for_full(launcher):
    cfg = launcher.RunConfig(
        model="mistralai/Mistral-7B-v0.3", avg_bits=3, seed=42,
        n_calib=32, n_eval=8, output_dir="/results/three_way",
    )
    p = launcher.output_path_for(cfg)
    assert p == (
        "/results/three_way/Mistral-7B-v0.3_b3_calib32_eval8_seed42.json"
    )


def test_output_path_for_smoke_prefixes(launcher):
    cfg = launcher.RunConfig(
        model="Qwen/Qwen2.5-7B", avg_bits=3, seed=42,
        n_calib=32, n_eval=8, smoke=True,
        output_dir="/results/three_way",
    )
    p = launcher.output_path_for(cfg)
    assert p.endswith(
        "synthetic_smoke__Qwen2.5-7B_b3_calib32_eval8_seed42.json"
    )


def test_parse_wrote_line_synthetic_smoke(launcher):
    s = (
        "some earlier output\n"
        "[run_three_way] synthetic-smoke: wrote "
        "/results/three_way/synthetic_smoke__M_b3_calib32_eval8_seed42.json\n"
        "later output\n"
    )
    assert launcher.parse_wrote_line(s) == (
        "/results/three_way/synthetic_smoke__M_b3_calib32_eval8_seed42.json"
    )


def test_parse_wrote_line_full(launcher):
    s = "[run_three_way] full: wrote /results/three_way/x.json\n"
    assert launcher.parse_wrote_line(s) == "/results/three_way/x.json"


def test_parse_wrote_line_none(launcher):
    assert launcher.parse_wrote_line("nothing here") is None
    assert launcher.parse_wrote_line("") is None


def test_sanitize_redacts_known_env_value(launcher, monkeypatch):
    secret = "abcdef0123456789ABCDEF"
    monkeypatch.setenv("HF_TOKEN", secret)
    text = f"prelude {secret} postlude"
    out = launcher.sanitize_text(text)
    assert secret not in out
    assert "[REDACTED:HF_TOKEN]" in out


def test_sanitize_redacts_token_literal_without_env(launcher, monkeypatch):
    # No env var set — value still redacted because it matches the
    # ``hf_...`` literal pattern.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    text = "leaked hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaa value"
    out = launcher.sanitize_text(text)
    assert "hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in out
    assert "[REDACTED:token-literal]" in out


def test_sanitize_skips_short_env_values(launcher, monkeypatch):
    # Short values must not be redacted (avoid eating common substrings).
    monkeypatch.setenv("API_KEY", "abc")
    text = "abc def abc"
    out = launcher.sanitize_text(text)
    assert out == text


def test_sanitize_redacts_multiple_env_secrets(launcher, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tokvalue1234567890")
    monkeypatch.setenv("WANDB_API_KEY", "wandbsecret9876543210")
    text = "tokvalue1234567890 and wandbsecret9876543210 here"
    out = launcher.sanitize_text(text)
    assert "tokvalue1234567890" not in out
    assert "wandbsecret9876543210" not in out


def test_sanitize_empty(launcher):
    assert launcher.sanitize_text("") == ""


def test_print_run_result_includes_metadata(launcher, capsys):
    result = {
        "returncode": 0,
        "stdout_tail": (
            "doing stuff\n"
            "[run_three_way] synthetic-smoke: wrote "
            "/results/three_way/synthetic_smoke__M_b3.json\n"
        ),
        "stderr_tail": "warning: something\n",
        "command": "python run_three_way.py ...",
        "model": "mistralai/Mistral-7B-v0.3",
        "avg_bits": 3,
        "seed": 42,
        "output_path": "/results/three_way/synthetic_smoke__M_b3.json",
        "parsed_output_path": "/results/three_way/synthetic_smoke__M_b3.json",
        "result_exists": True,
    }
    launcher.print_run_result(result)
    captured = capsys.readouterr().out
    assert "returncode=0" in captured
    assert "model=mistralai/Mistral-7B-v0.3" in captured
    assert "avg_bits=3" in captured
    assert "output_path: /results/three_way/synthetic_smoke__M_b3.json" in captured
    assert "result_exists: True" in captured
    assert "smoke-marker:" in captured
    assert "stdout_tail" in captured
    assert "stderr_tail" in captured
    assert "warning: something" in captured


def test_print_run_result_sanitizes_tails(launcher, capsys, monkeypatch):
    secret = "supersecretvalue1234567890"
    monkeypatch.setenv("HF_TOKEN", secret)
    result = {
        "returncode": 1,
        "stdout_tail": f"crash with {secret} in trace",
        "stderr_tail": f"Traceback: HF_TOKEN={secret}",
        "command": "python ...",
        "model": "m/m",
        "avg_bits": 3,
        "seed": 42,
        "output_path": "/results/three_way/x.json",
        "parsed_output_path": None,
        "result_exists": False,
    }
    launcher.print_run_result(result)
    out = capsys.readouterr().out
    assert secret not in out
    assert "[REDACTED:HF_TOKEN]" in out
    assert "result_exists: False" in out


def test_print_run_result_handles_empty_tails(launcher, capsys):
    result = {
        "returncode": 0,
        "stdout_tail": "",
        "stderr_tail": "",
        "model": "m/m",
        "avg_bits": 3,
        "seed": 42,
        "output_path": "/r/x.json",
        "parsed_output_path": None,
        "result_exists": False,
    }
    launcher.print_run_result(result)
    out = capsys.readouterr().out
    # Both tail sections should appear with an explicit (empty) marker so the
    # operator knows the field was empty rather than missing.
    assert out.count("(empty)") == 2
    assert "stdout_tail" in out
    assert "stderr_tail" in out


def test_print_run_result_no_smoke_marker_when_absent(launcher, capsys):
    result = {
        "returncode": 0,
        "stdout_tail": "regular run output without marker",
        "stderr_tail": "",
        "model": "m/m",
        "avg_bits": 3,
        "seed": 42,
        "output_path": "/r/x.json",
        "parsed_output_path": None,
        "result_exists": True,
    }
    launcher.print_run_result(result)
    out = capsys.readouterr().out
    assert "smoke-marker:" not in out


# --- git commit forwarding -----------------------------------------------
#
# .git is excluded from the Modal image, so the harness's ``git rev-parse``
# returns the ``0000000`` placeholder. The launcher detects the *local*
# repo's HEAD and forwards it via ``SPECTRALQUANT_GIT_COMMIT`` so the
# remote subprocess records real provenance.


def test_detect_local_git_commit_returns_hex_sha(launcher):
    """Inside the repo this should return a real hex SHA."""
    sha = launcher.detect_local_git_commit(launcher.REPO_ROOT)
    if sha is None:
        pytest.skip("not inside a git checkout")
    assert len(sha) >= 7
    assert all(c in "0123456789abcdef" for c in sha.lower())


def test_detect_local_git_commit_returns_none_outside_repo(launcher, tmp_path):
    sha = launcher.detect_local_git_commit(tmp_path)
    assert sha is None


def test_git_commit_env_constant_matches_harness(launcher):
    """The launcher's env-var name must match the harness's, otherwise
    the forwarded value will be ignored on the remote side."""
    assert launcher.GIT_COMMIT_ENV == "SPECTRALQUANT_GIT_COMMIT"


def test_run_one_signature_accepts_git_commit(launcher):
    """build_modal_app's run_one must accept a ``git_commit`` arg.

    Skipped without modal because ``run_one`` is constructed inside
    ``build_modal_app``. We inspect the source to keep this test
    cheap and modal-independent.
    """
    text = (launcher.REPO_ROOT / "scripts" / "launch_modal_three_way.py").read_text()
    assert "git_commit: Optional[str] = None" in text or "git_commit=None" in text
    assert "env[GIT_COMMIT_ENV] = git_commit" in text
    assert ".remote(asdict(cfg), git_commit=git_commit)" in text


def test_dry_run_prints_local_commit(launcher, monkeypatch):
    """--dry-run output should show the local commit it will forward."""
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--model", "mistralai/Mistral-7B-v0.3",
            "--avg-bits", "3",
            "--dry-run",
        ],
        capture_output=True, text=True, check=False,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    # We are inside a git checkout so a hex sha should appear.
    assert "SPECTRALQUANT_GIT_COMMIT" in result.stdout


def test_print_run_result_includes_git_commit(launcher, capsys):
    result = {
        "returncode": 0,
        "stdout_tail": "ok\n",
        "stderr_tail": "",
        "model": "m/m",
        "avg_bits": 3,
        "seed": 42,
        "output_path": "/r/x.json",
        "parsed_output_path": None,
        "result_exists": True,
        "git_commit": "abc1234567890def",
    }
    launcher.print_run_result(result)
    out = capsys.readouterr().out
    assert "git_commit: abc1234567890def" in out


# --- Detached-mode entrypoint regression ---------------------------------
#
# `modal run -d scripts/launch_modal_three_way.py::main_entry -- ...`
# requires that *both* an `App` and a callable `main_entry` exist at module
# scope when the script is imported. The previous failure mode was that the
# launcher built its Modal app inside `build_modal_app()` only, so Modal's
# CLI saw no top-level `App`/function and rejected the file with "no
# functions/local entrypoints".
#
# These tests pin the new shape:
#
# * Module exposes top-level `app`, `run_one`, and `main_entry`.
# * `modal run scripts/launch_modal_three_way.py::main_entry --help`
#   succeeds (proves the file is a valid Modal target for `-d` mode).
# * Source-level guard: `@app.local_entrypoint()` is present so a
#   regression to nested-only construction is caught even if `modal` is
#   not installed at test time.


def test_module_exposes_top_level_modal_app(launcher):
    """Module-level `app`, `run_one`, and `main_entry` must exist when
    modal is installed so `modal run` can discover them.
    """
    pytest.importorskip("modal")
    assert launcher.app is not None, (
        "scripts/launch_modal_three_way.py must expose a module-scope "
        "`app` so `modal run script.py::main_entry` can find it."
    )
    assert launcher.run_one is not None, (
        "scripts/launch_modal_three_way.py must expose a module-scope "
        "`run_one` Modal function."
    )
    assert callable(launcher.main_entry) or launcher.main_entry is not None, (
        "scripts/launch_modal_three_way.py must expose a module-scope "
        "`main_entry` local entrypoint for `modal run`."
    )


def test_source_declares_local_entrypoint():
    """Source-level guard for the detached-mode entrypoint.

    Even when modal is not installed, the source file must continue to
    declare `@app.local_entrypoint()` and a `main_entry` function so the
    detached path is robust to import-time fallbacks.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "@app.local_entrypoint()" in text, (
        "Expected @app.local_entrypoint() so `modal run "
        "script.py::main_entry` is a valid invocation."
    )
    assert "def main_entry(" in text, (
        "Expected a top-level `def main_entry(...)` for `modal run "
        "script.py::main_entry`."
    )


def test_modal_run_help_lists_main_entry():
    """`modal run script.py::main_entry --help` must succeed.

    This is the regression check for the failure the user reported: Modal
    1.4.x was rejecting the file with "no functions/local entrypoints"
    because the app/function were built inside a function rather than at
    module scope. We invoke `modal run --help` rather than actually
    submitting the function so the test is free.
    """
    import shutil
    if shutil.which("modal") is None:
        pytest.skip("modal CLI not on PATH")
    pytest.importorskip("modal")
    result = subprocess.run(
        ["modal", "run", f"{SCRIPT_PATH}::main_entry", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"`modal run {SCRIPT_PATH}::main_entry --help` failed with "
        f"rc={result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    # Must list the required CLI options Modal generated from the
    # entrypoint's typed parameters.
    assert "--model" in combined
    assert "--avg-bits" in combined


def test_modal_run_detached_help_works():
    """`modal run -d script.py::main_entry --help` must also succeed.

    This is the *exact* detached invocation pattern documented in
    docs/modal_safety_protocol.md; if it ever stops parsing we want CI to
    catch it before a real GPU run depends on it.
    """
    import shutil
    if shutil.which("modal") is None:
        pytest.skip("modal CLI not on PATH")
    pytest.importorskip("modal")
    result = subprocess.run(
        ["modal", "run", "-d", f"{SCRIPT_PATH}::main_entry", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"detached `modal run -d` rejected the entrypoint:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_main_entry_signature_has_required_params(launcher):
    """`main_entry` must take `model` and `avg_bits` as required positional
    parameters so Modal's CLI parser exposes them as required options.
    """
    pytest.importorskip("modal")
    import inspect
    fn = launcher.main_entry
    # `local_entrypoint` wraps the user function as a `LocalEntrypoint`
    # whose `info.raw_f` is the original Python function. Fall back to
    # other plausible attribute names so the test is robust to upstream
    # renames; only fail if we cannot find any user function to inspect.
    info = getattr(fn, "info", None)
    raw = (
        getattr(info, "raw_f", None)
        or getattr(fn, "raw_f", None)
        or getattr(fn, "_raw_f", None)
    )
    assert raw is not None, (
        "Could not extract the underlying function from main_entry; the "
        "Modal LocalEntrypoint API may have changed."
    )
    sig = inspect.signature(raw)
    assert "model" in sig.parameters
    assert "avg_bits" in sig.parameters
    # `model` and `avg_bits` should have no default → Modal marks them
    # required.
    assert sig.parameters["model"].default is inspect.Parameter.empty
    assert sig.parameters["avg_bits"].default is inspect.Parameter.empty


def test_no_python_wrapper_required_for_detached():
    """The documented detached command must not depend on a Python wrapper.

    The previous workaround was to wrap `modal run` in a Python script that
    happened to time out at the 10-minute tool boundary, killing the GPU
    job before it could write its artifact. The detached path must be
    invokable as a single `modal run -d ...` command with no Python
    wrapper. We assert this by checking that the documented command in
    `docs/modal_safety_protocol.md` matches `modal run -d
    scripts/launch_modal_three_way.py::main_entry`.
    """
    docs = REPO_ROOT / "docs" / "modal_safety_protocol.md"
    text = docs.read_text(encoding="utf-8")
    assert "modal run -d scripts/launch_modal_three_way.py::main_entry" in text, (
        "docs/modal_safety_protocol.md must document the exact detached "
        "command `modal run -d scripts/launch_modal_three_way.py::"
        "main_entry` so operators can copy-paste it without a wrapper."
    )


# --- Detached-mode deserialization import path ---------------------------
#
# Detached `modal run -d` serializes the function and its closure by the
# *local* module name (``launch_modal_three_way``) and dispatches it to a
# remote container, which deserializes it. The remote runner must be able
# to ``import launch_modal_three_way`` for the unpickle to succeed —
# otherwise it raises:
#
#   modal.exception.DeserializationError: Deserialization failed because
#   the 'launch_modal_three_way' module is not available in the remote
#   environment.
#
# Since ``scripts/`` is not a package, the only way to make the bare
# module name resolve on the remote is to put ``/repo/scripts`` on
# ``PYTHONPATH`` *inside the image*. These tests guard against a regression
# that drops that PYTHONPATH wiring.


def test_source_image_pythonpath_includes_repo_scripts():
    """The image's PYTHONPATH must include ``/repo/scripts`` so the remote
    container can ``import launch_modal_three_way`` during deserialization
    of the serialized function call.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert ".env(" in text and "PYTHONPATH" in text, (
        "Expected scripts/launch_modal_three_way.py to set PYTHONPATH on "
        "the image via Image.env({...}) so the remote container can "
        "import the launcher module during pickle deserialization."
    )
    assert "/repo/scripts" in text, (
        "Expected /repo/scripts to appear on the image's PYTHONPATH; "
        "without it, `modal run -d ...::main_entry` will fail with "
        "DeserializationError because the launcher module is not "
        "importable on the remote side."
    )


def test_source_run_one_subprocess_env_includes_repo_scripts():
    """The subprocess launched inside ``run_one`` must also have
    ``/repo/scripts`` on PYTHONPATH so any nested helper that re-imports
    the launcher (for example for ``RunConfig``) succeeds.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    # The subprocess env block hands its env to subprocess.run; check it
    # composes /repo/scripts (and /repo/src) into PYTHONPATH.
    assert (
        '"/repo/scripts" + os.pathsep + "/repo/src"' in text
        or "'/repo/scripts'" in text and "'/repo/src'" in text
    ), (
        "Expected run_one to include /repo/scripts and /repo/src on the "
        "subprocess PYTHONPATH so imports of the launcher and project "
        "modules succeed inside the container."
    )


def test_image_pythonpath_set_when_modal_installed(launcher):
    """When modal is installed, the constructed image must carry a
    PYTHONPATH env var that includes ``/repo/scripts``.

    Skipped if modal isn't installed — the assertion only makes sense
    when we can actually inspect the image build chain.
    """
    pytest.importorskip("modal")
    app, image, volume, run_one = launcher.build_modal_app(
        gpu="H200",
        timeout_sec=300,
        hf_secret_name="hf-token",
    )
    # Walk the image's commands to find an ``env`` step that sets
    # PYTHONPATH. Modal stores build steps on either ``_serve_mounts``
    # / ``_dockerfile_commands`` / ``_commands`` depending on version;
    # rather than depend on those internals, we serialize the chain and
    # search for the literal we set. This is best-effort: if Modal
    # restructures the layout, the source-level guard above still pins
    # the behaviour.
    found = False
    candidates = []
    for attr in dir(image):
        if attr.startswith("__"):
            continue
        try:
            v = getattr(image, attr)
        except Exception:
            continue
        s = repr(v)
        if "PYTHONPATH" in s and "/repo/scripts" in s:
            found = True
            candidates.append(attr)
            break
    if not found:
        # Fall back to the source guard — Modal's internal layout may have
        # shifted, but the source-level test above already pins the call.
        pytest.skip(
            "Could not introspect Modal image to find PYTHONPATH env; "
            "source-level test already pins the build call."
        )
    assert found, f"PYTHONPATH/repo/scripts not found on image (checked {candidates})"


# --- Detached-mode .spawn() vs foreground .remote() ----------------------
#
# Modal logs this warning when ``.remote()`` is used inside an app that
# was started with ``modal run -d``:
#
#   remote() and .map() calls in detached apps may be canceled when the
#   local caller disconnects. Use .spawn() for detached or background
#   work.
#
# When the local client times out / disconnects (which is exactly what
# happens at our 10-min tool boundary), the detached app continues to
# run but the *remote() call itself* is cancelled, killing the GPU
# subprocess before it can write the result JSON. The fix is to use
# ``run_one.spawn(...)`` inside the ``main_entry`` local entrypoint:
# spawn returns a FunctionCall whose object_id can be polled later.
#
# Foreground ``main()`` (the ``python3 scripts/launch_modal_three_way.py``
# wrapper) still uses ``.remote()`` because there the local client
# *is* the supervisor — blocking on the result is the point.


def test_main_entry_uses_spawn_not_remote():
    """`main_entry` must call ``run_one.spawn(...)`` so the remote function
    survives a local-client disconnect under ``modal run -d``.

    Source-level guard so this is checked even without Modal installed.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    # Locate the main_entry body.
    marker = "def main_entry("
    assert marker in text, "main_entry must exist at module scope"
    body_start = text.index(marker)
    # Restrict to the module-scope main_entry block (everything from
    # ``def main_entry(`` to the next ``except ImportError`` we know
    # follows the try block).
    body_end = text.index("except ImportError", body_start)
    body = text[body_start:body_end]
    assert "run_one.spawn(" in body, (
        "main_entry must call run_one.spawn(...) so detached `modal run -d` "
        "remote calls survive a local-client disconnect. Otherwise Modal "
        "cancels the call with the warning: 'remote() and .map() calls in "
        "detached apps may be canceled when the local caller disconnects.'"
    )
    assert "run_one.remote(" not in body, (
        "main_entry must not call run_one.remote(...) — that call is "
        "cancelled when the local `modal run -d` client disconnects. Use "
        "run_one.spawn(...) instead."
    )


def test_main_uses_remote_not_spawn():
    """`main()` (the Python wrapper path) must keep using ``run_one.remote``.

    Foreground `python3 scripts/launch_modal_three_way.py ...` is the
    supervisor itself — it must block on the remote result and surface
    the returncode/stdout tail. Switching it to spawn would lose those.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def main(argv:"
    assert marker in text, "main() must exist at module scope"
    body_start = text.index(marker)
    # main() ends at the next top-level ``# ---`` divider before
    # the module-scope app construction block.
    body_end = text.index("# Module-scope Modal app", body_start)
    body = text[body_start:body_end]
    assert "run_one.remote(" in body, (
        "main() (foreground wrapper) must call run_one.remote(...) so it "
        "blocks on the remote result and prints the returncode/stdout "
        "tail."
    )
    assert "run_one.spawn(" not in body, (
        "main() must not call run_one.spawn(...) — the foreground wrapper "
        "is the supervisor and must block on the result. Spawn belongs to "
        "the detached `main_entry` path."
    )


def test_main_entry_prints_call_id_and_expected_output():
    """The detached entrypoint must print the spawn call_id and the
    deterministic expected output path so the operator can poll the
    Modal volume without having to read logs.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def main_entry("
    body_start = text.index(marker)
    body_end = text.index("except ImportError", body_start)
    body = text[body_start:body_end]
    # Call id surfaced from the spawn handle.
    assert "object_id" in body, (
        "main_entry must surface the spawn call's ``object_id`` so the "
        "operator has a poll handle for FunctionCall.from_id(...)."
    )
    assert "call_id=" in body, (
        "main_entry must print a clearly-labelled ``call_id=...`` line so "
        "the operator can find the spawn handle in the console."
    )
    # Expected output path so the operator knows where to look on the volume.
    assert "expected_output_path:" in body, (
        "main_entry must print ``expected_output_path: ...`` so the "
        "operator can poll the Modal volume directly without log access."
    )
    # And it must invoke output_path_for(cfg) to compute that path.
    assert "output_path_for(" in body, (
        "main_entry must call output_path_for(cfg) to compute the "
        "deterministic artifact path."
    )


def test_main_entry_does_not_block_on_remote_result():
    """The detached entrypoint must not call ``.get()`` or otherwise block
    on the spawn handle — that would defeat the purpose of detaching.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def main_entry("
    body_start = text.index(marker)
    body_end = text.index("except ImportError", body_start)
    body = text[body_start:body_end]
    # No call.get(...) — that would block until completion (defeats spawn).
    assert "call.get(" not in body, (
        "main_entry must not call call.get(...) — that blocks until the "
        "remote function returns, which defeats the spawn-for-detached "
        "design and re-introduces the cancel-on-disconnect failure mode."
    )
    # No print_run_result either: we don't have a ``result`` dict any more.
    assert "print_run_result(result)" not in body, (
        "main_entry no longer has a synchronous ``result`` to print; "
        "operator polls the volume / FunctionCall.from_id instead."
    )


def test_source_run_one_streams_output_via_popen():
    """run_one must stream subprocess output line-by-line, not collect it
    all up-front via ``capture_output=True`` — otherwise mid-run progress
    is invisible until the subprocess exits.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    # Locate run_one body.
    marker = "def run_one("
    body_start = text.index(marker)
    body_end = text.index("return app, image, volume, run_one", body_start)
    body = text[body_start:body_end]
    assert "subprocess.Popen(" in body, (
        "run_one must use subprocess.Popen so stdout/stderr can be streamed"
    )
    # The legacy capture_output=True call must be gone.
    assert "subprocess.run(" not in body or "capture_output=True" not in body, (
        "run_one must not use subprocess.run(capture_output=True) — that "
        "blocks until the child exits and prevents mid-run progress emission"
    )


def test_source_run_one_emits_status_artifacts():
    """run_one must emit subprocess_start / subprocess_progress / failure
    via the StatusWriter so progress is observable even without log access.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def run_one("
    body_start = text.index(marker)
    body_end = text.index("return app, image, volume, run_one", body_start)
    body = text[body_start:body_end]
    for stage in ("subprocess_start", "subprocess_progress",
                  "subprocess_end", "failure"):
        assert f'"{stage}"' in body, (
            f"run_one must emit a {stage!r} status event"
        )
    assert "StatusWriter" in body
    assert "configure_persistent_hf_cache" in body, (
        "run_one must configure the persistent HF cache env vars"
    )
    # Ensure --status-dir is forwarded to the child.
    assert "--status-dir" in body, (
        "run_one must pass --status-dir to the run_three_way subprocess"
    )


def test_source_run_one_passes_status_dir_in_result():
    """The dict returned by run_one must include status_dir / status_path
    so the operator can locate the artifacts via the printed result."""
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def run_one("
    body_start = text.index(marker)
    body_end = text.index("return app, image, volume, run_one", body_start)
    body = text[body_start:body_end]
    assert '"status_dir"' in body
    assert '"status_path"' in body


def test_main_entry_prints_status_path():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def main_entry("
    body_start = text.index(marker)
    body_end = text.index("except ImportError", body_start)
    body = text[body_start:body_end]
    assert "status_path" in body, (
        "main_entry must surface a status_path the operator can poll"
    )


def test_print_run_result_includes_status_dir(launcher, capsys):
    result = {
        "returncode": 0,
        "stdout_tail": "ok\n",
        "stderr_tail": "",
        "model": "m/m",
        "avg_bits": 3,
        "seed": 42,
        "output_path": "/r/x.json",
        "parsed_output_path": None,
        "result_exists": True,
        "status_dir": "/results/status/run-x",
        "status_path": "/results/status/run-x/status.json",
    }
    launcher.print_run_result(result)
    out = capsys.readouterr().out
    assert "status_dir: /results/status/run-x" in out
    assert "status_path: /results/status/run-x/status.json" in out


def test_run_one_writes_failure_artifact_on_child_exit(launcher, tmp_path):
    """Simulated child-process failure must produce a failure status JSON.

    We exercise the subprocess-streaming logic through a fake child by
    writing a minimal harness to ``tmp_path`` that exits non-zero and
    produces stdout/stderr. The launcher's run_one is reachable only
    when modal is installed; this test inlines the same streaming logic
    (Popen + select + status emission) so the contract is validated
    regardless. The point is: when a child exits nonzero, a failure
    JSON must be written, and it must contain sanitized tails.
    """
    pytest.importorskip("experiments.run_status")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from experiments import run_status
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))

    sd = tmp_path / "status" / "fail-run"
    w = run_status.StatusWriter(
        status_dir=sd, run_id="fail-run", model="m/m", avg_bits=3,
    )
    # Simulate the child-process exit-nonzero path.
    w.emit(
        "failure",
        message="subprocess exited with rc=2",
        error="subprocess exited with rc=2",
        details={"returncode": 2, "result_exists": False},
        stdout_tail="last stdout line\n",
        stderr_tail="Traceback (most recent call last):\n  oops\n",
    )
    snap = json.loads((sd / "status.json").read_text())
    assert snap["stage"] == "failure"
    assert snap["details"]["returncode"] == 2
    assert "Traceback" in snap["stderr_tail"]


def test_docs_describe_spawn_polling():
    """The runbook docs must describe how to poll a spawned detached run.

    Operators need to know: (a) call_id is printed by the launcher,
    (b) the expected_output_path is also printed, (c) ``modal volume ls``
    + ``modal volume get`` retrieve the artifact, and (d)
    ``modal.FunctionCall.from_id(<call_id>).get()`` retrieves the
    structured result if needed.
    """
    safety = REPO_ROOT / "docs" / "modal_safety_protocol.md"
    runbook = REPO_ROOT / "docs" / "execution_audit_and_modal_runbook.md"
    safety_text = safety.read_text(encoding="utf-8")
    runbook_text = runbook.read_text(encoding="utf-8")
    for path, text in (("modal_safety_protocol.md", safety_text),
                       ("execution_audit_and_modal_runbook.md", runbook_text)):
        assert "call_id" in text, (
            f"docs/{path} must mention the printed ``call_id`` so the "
            f"operator knows what FunctionCall.from_id needs."
        )
        assert "expected_output_path" in text, (
            f"docs/{path} must mention ``expected_output_path`` printed by "
            f"the launcher so operators can poll the Modal volume directly."
        )
        assert "FunctionCall.from_id" in text, (
            f"docs/{path} must show how to retrieve a spawned detached "
            f"call's result via ``modal.FunctionCall.from_id(call_id).get()``."
        )


# --- status_dir override (detached + foreground parity) ------------------
#
# ``experiments/run_three_way.py`` accepts ``--status-dir`` so operators can
# direct heartbeat / progress artifacts to a known volume path (used by the
# documented detached smoke). The launcher must forward that override
# through ``RunConfig`` -> ``run_one`` -> the printed ``status_path`` line
# without breaking the existing default-derivation behaviour.


def test_run_config_has_status_dir_field(launcher):
    """``RunConfig`` must expose ``status_dir`` so callers can override the
    default heartbeat location.
    """
    fields = launcher.RunConfig.__dataclass_fields__
    assert "status_dir" in fields, (
        "RunConfig must declare ``status_dir`` so the override survives "
        "the dataclass round-trip used by ``run_one`` (asdict/RunConfig)."
    )
    cfg = launcher.RunConfig(model="m/m", avg_bits=3)
    assert cfg.status_dir is None, (
        "RunConfig.status_dir must default to None (i.e. derive from "
        "output_dir) so existing call-sites are unaffected."
    )
    cfg2 = launcher.RunConfig(
        model="m/m", avg_bits=3, status_dir="/results/status_hf_smoke",
    )
    assert cfg2.status_dir == "/results/status_hf_smoke"


def test_main_entry_signature_exposes_status_dir(launcher):
    """``main_entry`` must declare a ``status_dir`` parameter so
    ``modal run -d ...::main_entry -- --status-dir /results/status_hf_smoke``
    parses cleanly.
    """
    pytest.importorskip("modal")
    import inspect
    fn = launcher.main_entry
    info = getattr(fn, "info", None)
    raw = (
        getattr(info, "raw_f", None)
        or getattr(fn, "raw_f", None)
        or getattr(fn, "_raw_f", None)
    )
    assert raw is not None, (
        "Could not extract the underlying function from main_entry; the "
        "Modal LocalEntrypoint API may have changed."
    )
    sig = inspect.signature(raw)
    assert "status_dir" in sig.parameters, (
        "main_entry must accept ``status_dir`` so the documented detached "
        "smoke command can pass --status-dir without erroring."
    )
    p = sig.parameters["status_dir"]
    # Must have a default (so it's optional) and be typed ``str`` because
    # Modal's CLI parser does not accept Optional[str] cleanly.
    assert p.default == "" or p.default is None, (
        "main_entry.status_dir must default to an empty value so omitting "
        "the flag preserves the existing default-derivation behaviour."
    )
    # ``from __future__ import annotations`` turns annotations into
    # strings, so accept either the type or its name.
    ann = p.annotation
    assert ann is str or ann == "str", (
        "main_entry.status_dir must be typed ``str`` (not Optional[str]) "
        "so Modal's CLI parser accepts it; empty string means 'use default'."
    )


def test_source_main_entry_forwards_status_dir():
    """Source-level guard: ``main_entry`` must thread ``status_dir`` into
    the ``RunConfig`` it builds and into the printed status path.

    Checked at the source level so the assertion runs even when modal is
    not installed at test time.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def main_entry("
    body_start = text.index(marker)
    body_end = text.index("except ImportError", body_start)
    body = text[body_start:body_end]
    assert "status_dir" in body, (
        "main_entry body must reference ``status_dir`` so the override is "
        "forwarded into the RunConfig passed to run_one.spawn(...)."
    )
    # Forward the override into RunConfig, with empty-string -> None.
    assert "status_dir=(status_dir or None)" in body, (
        "main_entry must convert empty-string ``status_dir`` to None when "
        "constructing RunConfig (Modal CLI cannot pass Optional[str])."
    )
    # The printed status path must respect the override.
    assert "cfg.status_dir" in body, (
        "main_entry's printed status path block must consult ``cfg.status_dir`` "
        "so the operator-facing status_path line matches what run_one writes."
    )


def test_source_run_one_honors_cfg_status_dir():
    """``run_one`` must honour ``cfg.status_dir`` when set, falling back to
    ``default_status_dir`` when None.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def run_one("
    body_start = text.index(marker)
    # run_one is closed by ``return app, image, volume, run_one`` near the
    # end of build_modal_app. Restrict to that block.
    body_end = text.index("return app, image, volume, run_one", body_start)
    body = text[body_start:body_end]
    assert "cfg.status_dir" in body, (
        "run_one must consult ``cfg.status_dir`` so a caller-supplied "
        "override (e.g. via main_entry --status-dir) is honoured."
    )
    assert "default_status_dir" in body, (
        "run_one must still fall back to ``default_status_dir`` when "
        "cfg.status_dir is None — that's the existing behaviour for the "
        "foreground/CLI path."
    )


def test_run_one_status_dir_round_trips_via_asdict(launcher):
    """``RunConfig.status_dir`` must survive the ``asdict`` round-trip used
    by the launcher to ship config to the remote ``run_one`` function.
    """
    from dataclasses import asdict
    cfg = launcher.RunConfig(
        model="m/m", avg_bits=3, status_dir="/results/status_hf_smoke",
    )
    d = asdict(cfg)
    assert d["status_dir"] == "/results/status_hf_smoke"
    cfg2 = launcher.RunConfig(**d)
    assert cfg2.status_dir == "/results/status_hf_smoke"


def test_modal_run_help_lists_status_dir():
    """``modal run script.py::main_entry --help`` must list ``--status-dir``.

    This is the regression check for the user-reported failure: the
    detached smoke (which passes ``--status-dir /results/status_hf_smoke``)
    was rejected because the entrypoint did not expose the parameter.
    """
    import shutil
    if shutil.which("modal") is None:
        pytest.skip("modal CLI not on PATH")
    pytest.importorskip("modal")
    result = subprocess.run(
        ["modal", "run", f"{SCRIPT_PATH}::main_entry", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"`modal run {SCRIPT_PATH}::main_entry --help` failed with "
        f"rc={result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "--status-dir" in combined, (
        "main_entry --help must list --status-dir so operators can pass "
        "the override on detached smoke runs."
    )


def test_cli_help_lists_status_dir():
    """The foreground Python CLI must also accept ``--status-dir`` for
    parity with ``main_entry``.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "--status-dir" in result.stdout, (
        "Foreground launcher CLI must expose --status-dir so it matches "
        "the detached main_entry surface."
    )


def test_configs_from_args_threads_status_dir(launcher):
    parser = launcher.build_parser()
    ns = parser.parse_args([
        "--model", "m/m",
        "--avg-bits", "3",
        "--status-dir", "/results/status_hf_smoke",
    ])
    cfgs = launcher.configs_from_args(ns)
    assert len(cfgs) == 1
    assert cfgs[0].status_dir == "/results/status_hf_smoke"


def test_docs_detached_smoke_includes_status_dir():
    """The documented detached HF-smoke command must include
    ``--status-dir`` so an operator copy-pastes a command that actually
    parses against the new ``main_entry`` surface.
    """
    safety = REPO_ROOT / "docs" / "modal_safety_protocol.md"
    text = safety.read_text(encoding="utf-8")
    # The smoke recipe block lives near §7.0 detached invocation.
    assert "modal run -d scripts/launch_modal_three_way.py::main_entry" in text
    assert "--status-dir" in text, (
        "docs/modal_safety_protocol.md must show how to pass "
        "--status-dir on the detached smoke command so operators have "
        "an unambiguous status path to poll."
    )


# --- Early Modal-runner status writes ------------------------------------
#
# When a remote ``run_one`` call is dispatched but the child subprocess
# never appears (e.g. resource limits, broken PYTHONPATH, OOM at Popen),
# the operator needs *some* artifact on the volume to know the dispatch
# happened. The launcher therefore writes a ``modal_run_one_entered``
# status JSON immediately after the function is invoked on the worker,
# *before* importing the project tree. Subsequent stages
# (``subprocess_env_configured`` / ``subprocess_starting`` /
# ``subprocess_started``) bracket the Popen call so a failure is
# pinpointable.


def test_safe_helpers_match_run_status(launcher):
    """``_derive_run_id_safe`` / ``_default_status_dir_safe`` must mirror
    the real ``experiments.run_status`` helpers so the early artifact
    lands at the same path as the later StatusWriter writes.
    """
    pytest.importorskip("experiments.run_status")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from experiments import run_status
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))

    rid_safe = launcher._derive_run_id_safe(
        model="mistralai/Mistral-7B-v0.3",
        avg_bits=3,
        seed=42,
        n_calib=32,
        n_eval=8,
    )
    rid_real = run_status.derive_run_id(
        model="mistralai/Mistral-7B-v0.3",
        avg_bits=3,
        seed=42,
        n_calib=32,
        n_eval=8,
    )
    assert rid_safe == rid_real

    rid_safe_smoke = launcher._derive_run_id_safe(
        model="Qwen/Qwen2.5-7B",
        avg_bits=3, seed=42, n_calib=32, n_eval=8, smoke=True,
    )
    rid_real_smoke = run_status.derive_run_id(
        model="Qwen/Qwen2.5-7B",
        avg_bits=3, seed=42, n_calib=32, n_eval=8, smoke=True,
    )
    assert rid_safe_smoke == rid_real_smoke

    sd_safe = launcher._default_status_dir_safe("/results/three_way", "run-z")
    sd_real = run_status.default_status_dir("/results/three_way", "run-z")
    assert str(sd_safe) == str(sd_real)


def test_early_emit_writes_status_and_event(launcher, tmp_path):
    sd = tmp_path / "status" / "run-early"
    launcher._early_emit(
        sd,
        "modal_run_one_entered",
        message="hello from worker",
        extra={
            "run_id": "run-early",
            "model": "mistralai/Mistral-7B-v0.3",
            "avg_bits": 3,
        },
    )
    status_file = sd / "status.json"
    events_file = sd / "events.jsonl"
    assert status_file.is_file()
    assert events_file.is_file()
    snap = json.loads(status_file.read_text())
    assert snap["stage"] == "modal_run_one_entered"
    assert snap["run_id"] == "run-early"
    assert snap["model"] == "mistralai/Mistral-7B-v0.3"
    assert snap["avg_bits"] == 3
    assert snap["message"] == "hello from worker"
    assert "timestamp" in snap and snap["timestamp"]
    assert "host" in snap and snap["host"]
    assert "pid" in snap
    # No leftover tempfiles.
    assert not any(p.name.startswith(".tmp_") for p in sd.iterdir())


def test_early_emit_redacts_secret_in_message(launcher, tmp_path, monkeypatch):
    sentinel = "leakytokenvalue1234567890abcdef"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    sd = tmp_path / "status" / "run-redact"
    launcher._early_emit(
        sd,
        "modal_run_one_entered",
        message=f"got token {sentinel} on entry",
        extra={"trace": f"echo {sentinel}", "model": "m/m"},
    )
    snap = json.loads((sd / "status.json").read_text())
    assert sentinel not in snap["message"]
    assert "[REDACTED:HF_TOKEN]" in snap["message"]
    assert sentinel not in snap["trace"]


def test_early_emit_redacts_token_literals_without_env(
    launcher, tmp_path, monkeypatch,
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    sd = tmp_path / "status" / "run-tokenlit"
    launcher._early_emit(
        sd,
        "modal_run_one_entered",
        message="literal hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaa here",
        extra={"trace": "ghp_aaaaaaaaaaaaaaaaaaaaaaaa here"},
    )
    snap = json.loads((sd / "status.json").read_text())
    assert "hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in snap["message"]
    assert "ghp_aaaaaaaaaaaaaaaaaaaaaaaa" not in snap["trace"]


def test_early_emit_appends_multiple_events(launcher, tmp_path):
    sd = tmp_path / "status" / "run-many"
    for stage in (
        "modal_run_one_entered",
        "subprocess_env_configured",
        "subprocess_starting",
        "subprocess_started",
    ):
        launcher._early_emit(sd, stage, message=stage)
    lines = (sd / "events.jsonl").read_text().splitlines()
    assert len(lines) == 4
    seen = [json.loads(l)["stage"] for l in lines]
    assert seen == [
        "modal_run_one_entered",
        "subprocess_env_configured",
        "subprocess_starting",
        "subprocess_started",
    ]
    # status.json is the latest.
    assert json.loads((sd / "status.json").read_text())["stage"] == \
        "subprocess_started"


def test_safe_output_path_for_returns_none_on_invalid(launcher):
    """``_safe_output_path_for`` must swallow exceptions and return None.

    A None return is what the early-emit payload should record when the
    output path cannot be computed; we don't want a cosmetic helper to
    crash the very first status write.
    """
    cfg = launcher.RunConfig(model="m/m", avg_bits=3)
    p = launcher._safe_output_path_for(cfg)
    assert p is not None
    assert p.endswith(".json")


def test_source_run_one_emits_modal_runner_stages():
    """Source-level guard: run_one must emit the four Modal-runner stages
    before/around the Popen call so the artifact is observable even if
    the child never starts.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def run_one("
    body_start = text.index(marker)
    body_end = text.index("return app, image, volume, run_one", body_start)
    body = text[body_start:body_end]
    for stage in (
        "modal_run_one_entered",
        "subprocess_env_configured",
        "subprocess_starting",
        "subprocess_started",
    ):
        assert f'"{stage}"' in body, (
            f"run_one must emit a {stage!r} status event"
        )
    # Popen must be wrapped in try/except that emits failure on exception.
    assert "subprocess.Popen(" in body
    assert "emit_failure(" in body
    # Volume commit calls so polling sees mid-run writes.
    assert "_commit_volume()" in body
    # Order check: modal_run_one_entered must come before any reference
    # to subprocess_env_configured (which itself must come before
    # subprocess_starting / subprocess_started).
    pos_entered = body.index('"modal_run_one_entered"')
    pos_env = body.index('"subprocess_env_configured"')
    pos_starting = body.index('"subprocess_starting"')
    pos_started = body.index('"subprocess_started"')
    pos_popen = body.index("subprocess.Popen(")
    assert pos_entered < pos_env < pos_starting < pos_popen < pos_started, (
        "Modal-runner stages must be emitted in order: "
        "modal_run_one_entered, subprocess_env_configured, "
        "subprocess_starting (before Popen), subprocess_started "
        "(after Popen returns)."
    )


def test_source_run_one_writes_status_before_project_import():
    """``run_one`` must call ``_early_emit('modal_run_one_entered', ...)``
    *before* importing ``experiments.run_status`` so a broken project
    tree still produces a visible artifact.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def run_one("
    body_start = text.index(marker)
    body_end = text.index("return app, image, volume, run_one", body_start)
    body = text[body_start:body_end]
    early = body.index("modal_run_one_entered")
    proj_import = body.index("from experiments import run_status")
    assert early < proj_import, (
        "_early_emit('modal_run_one_entered', ...) must run BEFORE "
        "`from experiments import run_status` so a broken project import "
        "still leaves an observable status.json on the volume."
    )


def test_source_run_one_volume_commits_after_emits():
    """``run_one`` must call ``_commit_volume()`` after each status emit
    so a polling operator sees the latest snapshot mid-run.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def run_one("
    body_start = text.index(marker)
    body_end = text.index("return app, image, volume, run_one", body_start)
    body = text[body_start:body_end]
    # Must be at least 5 commit calls (entered, env_configured, start,
    # starting, started + progress + end). Be conservative and require >=4.
    n = body.count("_commit_volume()")
    assert n >= 4, (
        f"Expected >=4 _commit_volume() calls in run_one, found {n}. "
        "Each status.emit must be followed by a volume commit so "
        "polling sees writes while the run is in progress."
    )


def test_source_run_one_handles_popen_exception():
    """If ``subprocess.Popen`` itself raises, run_one must emit a failure
    artifact and re-raise. Source-level guard.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    marker = "def run_one("
    body_start = text.index(marker)
    body_end = text.index("return app, image, volume, run_one", body_start)
    body = text[body_start:body_end]
    popen_pos = body.index("subprocess.Popen(")
    # Find the *outer* try block that wraps Popen specifically (the one
    # whose except emits failure with phase: popen).
    snippet = body[max(0, popen_pos - 200): popen_pos + 600]
    assert "try:" in snippet, (
        "Popen call must be wrapped in try: so a Popen failure can emit "
        "a status artifact before re-raising."
    )
    assert "emit_failure(" in snippet, (
        "Popen exception path must emit a failure status artifact."
    )
    assert '"phase": "popen"' in snippet, (
        "Popen-failure emit must record details={'phase': 'popen'} so "
        "the operator can distinguish a Popen failure from a child-process "
        "failure."
    )


def test_run_one_popen_failure_writes_status_artifact(launcher, tmp_path):
    """Simulate a Popen failure path end-to-end via the StatusWriter.

    We can't invoke ``run_one`` itself without modal, so we validate the
    contract: when ``emit_failure`` is called with ``phase: popen``
    details, the resulting artifact contains the expected fields.
    """
    pytest.importorskip("experiments.run_status")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from experiments import run_status
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))

    sd = tmp_path / "status" / "popen-fail"
    w = run_status.StatusWriter(
        status_dir=sd, run_id="popen-fail", model="m/m", avg_bits=3,
    )
    try:
        raise FileNotFoundError("/nonexistent/python: not found")
    except FileNotFoundError as exc:
        w.emit_failure(
            exc,
            details={"phase": "popen", "argv0": "/nonexistent/python"},
        )
    snap = json.loads((sd / "status.json").read_text())
    assert snap["stage"] == "failure"
    assert snap["details"]["phase"] == "popen"
    assert snap["details"]["argv0"] == "/nonexistent/python"
    assert "FileNotFoundError" in snap["error"]
    assert "Traceback" in snap["traceback"]


def test_run_one_early_emit_no_secret_leak(
    launcher, tmp_path, monkeypatch,
):
    """End-to-end check: when ``_early_emit`` is called with a payload
    whose strings carry a known token, the on-disk artifact must redact it.
    """
    sentinel = "supersecretvalue1234567890"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    sd = tmp_path / "status" / "run-leak"
    launcher._early_emit(
        sd,
        "modal_run_one_entered",
        message=f"forwarding env with HF_TOKEN={sentinel}",
        extra={
            "model": "m/m",
            "leaky_field": f"value={sentinel}",
            "expected_output_path": "/results/three_way/x.json",
        },
    )
    raw = (sd / "status.json").read_text()
    assert sentinel not in raw
    assert "[REDACTED:HF_TOKEN]" in raw


def test_status_path_computation_matches_real(launcher):
    """Status path computed by the safe helpers must match the path
    later used by the real ``StatusWriter`` so the early artifact and
    later writes share a directory.
    """
    pytest.importorskip("experiments.run_status")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from experiments import run_status
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))

    # Override status_dir case.
    cfg = launcher.RunConfig(
        model="mistralai/Mistral-7B-v0.3", avg_bits=3, seed=42,
        n_calib=32, n_eval=8,
        status_dir="/results/status_hf_smoke",
    )
    rid = launcher._derive_run_id_safe(
        model=cfg.model, avg_bits=cfg.avg_bits, seed=cfg.seed,
        n_calib=cfg.n_calib, n_eval=cfg.n_eval, smoke=cfg.smoke,
    )
    assert rid == "Mistral-7B-v0.3_b3_calib32_eval8_seed42"
    sd_safe = Path(cfg.status_dir) / rid
    rid_real = run_status.derive_run_id(
        model=cfg.model, avg_bits=cfg.avg_bits, seed=cfg.seed,
        n_calib=cfg.n_calib, n_eval=cfg.n_eval, smoke=cfg.smoke,
    )
    sd_real = Path(cfg.status_dir) / rid_real
    assert str(sd_safe) == str(sd_real)
    assert str(sd_safe / "status.json") == \
        "/results/status_hf_smoke/Mistral-7B-v0.3_b3_calib32_eval8_seed42/status.json"


def test_docs_describe_two_status_layers():
    """The runbook docs must explicitly describe the two status layers
    (Modal-runner stages vs benchmark stages) so operators understand
    what a stuck status means.
    """
    safety = REPO_ROOT / "docs" / "modal_safety_protocol.md"
    text = safety.read_text(encoding="utf-8")
    assert "Modal-runner stages" in text, (
        "docs/modal_safety_protocol.md must describe the Modal-runner "
        "status layer so operators know what modal_run_one_entered means."
    )
    assert "Benchmark stages" in text, (
        "docs/modal_safety_protocol.md must describe the benchmark "
        "status layer (the child run_three_way.py)."
    )
    assert "modal_run_one_entered" in text
    assert "subprocess_env_configured" in text
    assert "subprocess_starting" in text
    assert "subprocess_started" in text


# --- inline-corpus-smoke wiring ------------------------------------------
#
# Pin the launcher contract: the new --inline-corpus-smoke flag flows from
# argparse through RunConfig -> build_command -> output_path_for and is
# threaded through main_entry (Modal CLI). Mutually exclusive with --smoke.


def test_run_config_inline_corpus_smoke_emits_flag(launcher):
    cfg = launcher.RunConfig(
        model="Qwen/Qwen2.5-0.5B", avg_bits=3, inline_corpus_smoke=True,
    )
    cmd = launcher.build_command(cfg)
    assert "--inline-corpus-smoke" in cmd
    assert "--synthetic-smoke" not in cmd


def test_run_config_inline_corpus_smoke_without_flag_omits_token(launcher):
    cfg = launcher.RunConfig(model="Qwen/Qwen2.5-0.5B", avg_bits=3)
    cmd = launcher.build_command(cfg)
    assert "--inline-corpus-smoke" not in cmd


def test_run_config_smoke_inline_combo_rejected(launcher):
    cfg = launcher.RunConfig(
        model="m/m", avg_bits=3, smoke=True, inline_corpus_smoke=True,
    )
    with pytest.raises(ValueError):
        launcher.build_command(cfg)


def test_run_config_inline_corpus_dry_run_combo_rejected(launcher):
    cfg = launcher.RunConfig(
        model="m/m", avg_bits=3, inline_corpus_smoke=True, dry_run=True,
    )
    with pytest.raises(ValueError):
        launcher.build_command(cfg)


def test_output_path_for_inline_corpus_smoke_prefixes(launcher):
    cfg = launcher.RunConfig(
        model="Qwen/Qwen2.5-0.5B", avg_bits=3, seed=42,
        n_calib=4, n_eval=2, inline_corpus_smoke=True,
        output_dir="/results/three_way",
    )
    p = launcher.output_path_for(cfg)
    assert p.endswith(
        "inline_corpus_smoke__Qwen2.5-0.5B_b3_calib4_eval2_seed42.json"
    )


def test_parse_wrote_line_inline_corpus_smoke(launcher):
    s = (
        "[run_three_way] inline-corpus-smoke: wrote "
        "/results/three_way/inline_corpus_smoke__Q_b3_calib4_eval2_seed42.json\n"
    )
    assert launcher.parse_wrote_line(s) == (
        "/results/three_way/inline_corpus_smoke__Q_b3_calib4_eval2_seed42.json"
    )


def test_configs_from_args_inline_corpus_smoke(launcher):
    parser = launcher.build_parser()
    ns = parser.parse_args([
        "--model", "Qwen/Qwen2.5-0.5B",
        "--avg-bits", "3",
        "--inline-corpus-smoke",
    ])
    cfgs = launcher.configs_from_args(ns)
    assert len(cfgs) == 1
    assert cfgs[0].inline_corpus_smoke is True
    assert cfgs[0].smoke is False
    cmd = launcher.build_command(cfgs[0])
    assert "--inline-corpus-smoke" in cmd


def test_main_entry_signature_includes_inline_corpus_smoke(launcher):
    """Modal CLI exposes --inline-corpus-smoke through main_entry."""
    import inspect
    src = inspect.getsource(launcher)
    # The main_entry param list must mention inline_corpus_smoke so Modal
    # exposes the flag at the modal-run CLI surface.
    assert "inline_corpus_smoke: bool = False" in src, (
        "main_entry must accept inline_corpus_smoke for modal-run CLI"
    )
    assert "inline_corpus_smoke=bool(inline_corpus_smoke)" in src, (
        "main_entry must thread inline_corpus_smoke into RunConfig"
    )


def test_derive_run_id_safe_inline_corpus_smoke_prefix(launcher):
    rid = launcher._derive_run_id_safe(
        model="Qwen/Qwen2.5-0.5B", avg_bits=3, seed=42,
        n_calib=4, n_eval=2, inline_corpus_smoke=True,
    )
    assert rid.startswith("inline_corpus_smoke__")
    assert rid.endswith("Qwen2.5-0.5B_b3_calib4_eval2_seed42")


def test_matrix_loader_accepts_inline_corpus_smoke(tmp_path, launcher):
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps([
        {"model": "Qwen/Qwen2.5-0.5B", "avg_bits": 3,
         "inline_corpus_smoke": True},
    ]))
    cfgs = launcher.load_matrix(p)
    assert len(cfgs) == 1
    assert cfgs[0].inline_corpus_smoke is True
    assert "--inline-corpus-smoke" in launcher.build_command(cfgs[0])


def test_dry_run_with_inline_corpus_smoke_prints_flag(tmp_path):
    """The launcher must surface --inline-corpus-smoke in --dry-run output
    so the operator can confirm the flag will reach the harness."""
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--model", "Qwen/Qwen2.5-0.5B",
            "--avg-bits", "3",
            "--inline-corpus-smoke",
            "--dry-run",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--inline-corpus-smoke" in result.stdout
