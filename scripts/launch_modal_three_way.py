#!/usr/bin/env python3
"""Modal launcher for ``experiments/run_three_way.py``.

This script is the operator-facing wrapper around the Modal app for the
SpectralQuant v2 three-way benchmark sweep. It:

* Defines the Modal image, GPU, secrets, and volume in code (so the
  whole stack is reviewable in git).
* Constructs the exact ``run_three_way.py`` command line from CLI/config
  inputs.
* Runs **one config at a time** (or a small controlled matrix) and
  honours ``--skip-if-exists`` for resumability.
* Persists results and calibration artifacts to a Modal Volume so a
  crashed sweep can resume without recalibrating.
* Has explicit timeouts per spec §7 of ``docs/modal_safety_protocol.md``.
* Imports cleanly without Modal credentials so unit tests can exercise
  ``build_command`` and the Modal-app spec.

Credential handling:

* Modal Secrets are referenced **by name** only. The names are
  configured through CLI flags (``--hf-secret-name`` etc.) so this file
  embeds no token values.
* The harness inside the Modal function reads ``HF_TOKEN`` from the
  environment but never echoes it. We do not pass tokens via CLI.
* The launcher itself never reads Modal/HF tokens; it just declares
  which named secrets to inject.

Anti-patterns explicitly avoided:

* No ``modal token set --token <value>`` calls.
* No ``os.environ["HF_TOKEN"]`` reads in this file.
* No ``image.run_commands(f"echo {token}")``-style baking.

Local invocation (constructs the command, does not actually launch
without Modal):

    python3 scripts/launch_modal_three_way.py --dry-run \
        --model mistralai/Mistral-7B-v0.3 --avg-bits 3 --seed 42

CLI:

    python3 scripts/launch_modal_three_way.py [--dry-run] \\
        --model MODEL --avg-bits B [--seed SEED] [--n-calib N] \\
        [--n-eval N] [--n-layers-sample N] [--gpu GPU] \\
        [--timeout-sec SEC] [--matrix-config PATH] \\
        [--volume-name NAME] [--app-name NAME] \\
        [--hf-secret-name NAME] [--wandb-secret-name NAME]

The expected end-to-end usage is one of:

    # Foreground (streams logs, dies if local client disconnects):
    python3 scripts/launch_modal_three_way.py \\
        --model mistralai/Mistral-7B-v0.3 --avg-bits 3

    # Detached (Modal keeps running even if the local client times out):
    modal run -d scripts/launch_modal_three_way.py::main_entry -- \\
        --model mistralai/Mistral-7B-v0.3 --avg-bits 3

The detached path requires that ``app`` and a ``main_entry`` local
entrypoint exist at module scope. We build them eagerly when ``modal`` is
importable so ``modal run`` can discover them; tests that monkeypatch
modal away still see ``app is None`` and continue to work via the
in-process ``build_modal_app()`` factory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RUN_THREE_WAY = EXPERIMENTS_DIR / "run_three_way.py"

#: Env var read by ``experiments/run_three_way.py::_git_commit`` so the
#: result JSON records the *local* commit even though the Modal image
#: excludes the ``.git`` directory. Mirrors ``run_three_way.GIT_COMMIT_ENV``.
GIT_COMMIT_ENV = "SPECTRALQUANT_GIT_COMMIT"


def detect_local_git_commit(repo_root: Path = REPO_ROOT) -> Optional[str]:
    """Return the local repo's ``HEAD`` SHA, or None if unavailable.

    Used to forward provenance into the Modal container without mounting
    ``.git``. We accept only hex SHAs of length >= 7 so a stray empty/
    junk value cannot pollute the result JSON.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        sha = out.decode("utf-8").strip()
        if len(sha) >= 7 and re.fullmatch(r"[0-9a-fA-F]+", sha):
            return sha
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Defaults — keep in sync with docs/execution_audit_and_modal_runbook.md §7
# and docs/modal_safety_protocol.md §7.
# ---------------------------------------------------------------------------

DEFAULT_GPU = "H200"
DEFAULT_TIMEOUT_SEC = 90 * 60          # full single-config run, spec §7
SMOKE_TIMEOUT_SEC = 15 * 60            # synthetic-smoke ceiling
SWEEP_TIMEOUT_SEC = 4 * 60 * 60        # full matrix in one function

DEFAULT_VOLUME_NAME = "spectralquant-v2-results"
DEFAULT_APP_NAME = "spectralquant-v2"
DEFAULT_HF_SECRET = "hf-token"
DEFAULT_WANDB_SECRET = "wandb-api-key"

#: Allowed GPU choices, mirroring docs/modal_safety_protocol.md §3.
ALLOWED_GPUS = ("H200", "H100", "B200", "A100-80GB")

#: Three-way run defaults from spec §13.2.
DEFAULT_N_CALIB = 32
DEFAULT_N_EVAL = 8
DEFAULT_N_LAYERS_SAMPLE = 8
DEFAULT_MAX_CALIB_TOKENS = 384


# ---------------------------------------------------------------------------
# Sanitization for stdout/stderr tails
# ---------------------------------------------------------------------------
#
# The remote function captures full subprocess stdout/stderr and we print the
# tail back to the operator console. The harness itself does not echo
# ``HF_TOKEN`` etc., but a noisy traceback or third-party warning could leak
# the value of an environment variable whose name matches a known secret
# pattern. Redact those values defensively before printing.

#: Substrings that, when present in an env-var name (case-insensitive), make
#: the value sensitive. Intentionally broad — false positives are cheap, a
#: leaked token is not.
SECRET_NAME_PATTERNS: Tuple[str, ...] = (
    "TOKEN",
    "SECRET",
    "API_KEY",
    "APIKEY",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
)

#: Token-like literals to redact even when the surrounding env var name is
#: unknown: HF user tokens (``hf_...``), GitHub PATs (``ghp_...`` /
#: ``github_pat_...``), AWS access key ids (``AKIA...``), Modal tokens
#: (``ak-...``, ``as-...``).
_TOKEN_LITERAL_PATTERNS: Tuple["re.Pattern[str]", ...] = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bak-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bas-[A-Za-z0-9_-]{16,}\b"),
)


def _is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(pat in upper for pat in SECRET_NAME_PATTERNS)


def sanitize_text(
    text: str,
    *,
    extra_env: Optional[Iterable[str]] = None,
) -> str:
    """Redact known secret values from a block of text.

    The text is typically a captured stdout/stderr tail from the remote
    subprocess. We redact:

    * The values of environment variables whose names match
      :data:`SECRET_NAME_PATTERNS` (or are in ``extra_env``), as observed in
      the launcher's own ``os.environ``.
    * Token-like literals matching :data:`_TOKEN_LITERAL_PATTERNS` even when
      we never knew the variable name.

    Empty/short values (<8 chars) are skipped to avoid replacing common
    substrings.
    """
    if not text:
        return text
    redacted = text
    seen: set = set()
    env_names: List[str] = []
    for k in os.environ:
        if _is_secret_name(k):
            env_names.append(k)
    if extra_env:
        for k in extra_env:
            if k not in env_names:
                env_names.append(k)
    for name in env_names:
        val = os.environ.get(name, "")
        if not val or len(val) < 8 or val in seen:
            continue
        seen.add(val)
        redacted = redacted.replace(val, f"[REDACTED:{name}]")
    for pat in _TOKEN_LITERAL_PATTERNS:
        redacted = pat.sub("[REDACTED:token-literal]", redacted)
    return redacted


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """One ``(model, avg_bits, seed)`` triplet to run."""
    model: str
    avg_bits: int
    seed: int = 42
    n_calib: int = DEFAULT_N_CALIB
    n_eval: int = DEFAULT_N_EVAL
    n_layers_sample: int = DEFAULT_N_LAYERS_SAMPLE
    max_calib_tokens: int = DEFAULT_MAX_CALIB_TOKENS
    output_dir: str = "/results/three_way"
    calibration_dir: Optional[str] = "/results/calibration"
    save_calibration: bool = True
    load_calibration: bool = False
    smoke: bool = False
    #: Run the *full* HF model path (real model load + adapter discovery +
    #: hooks + calibration + eval) but use a tiny deterministic in-memory
    #: corpus instead of HuggingFace ``datasets.load_dataset``. Harness
    #: validation only — results are *not* paper-valid evidence.
    inline_corpus_smoke: bool = False
    dry_run: bool = False
    force: bool = False
    qjl_projections: int = 64
    wf_min_bits: int = 0
    wf_max_bits: Optional[int] = None
    device: str = "cuda"
    dtype: str = "float16"
    #: Optional override for the *parent* status directory. The per-run
    #: status artifacts live at ``<status_dir>/<run_id>/``. When ``None``
    #: (the default) the parent is derived from ``output_dir`` via
    #: :func:`experiments.run_status.default_status_dir`.
    status_dir: Optional[str] = None
    extra: Tuple[str, ...] = field(default_factory=tuple)


def build_command(cfg: RunConfig) -> List[str]:
    """Build the argv that will run inside the Modal container.

    The result can be fed to ``subprocess.run`` (locally for tests) or
    invoked with ``python3 -m`` inside ``run_three_way.py``.
    """
    if cfg.dry_run and cfg.smoke:
        raise ValueError("Cannot combine smoke=True with dry_run=True")
    if cfg.smoke and cfg.inline_corpus_smoke:
        raise ValueError(
            "Cannot combine smoke=True with inline_corpus_smoke=True"
        )
    if cfg.dry_run and cfg.inline_corpus_smoke:
        raise ValueError(
            "Cannot combine inline_corpus_smoke=True with dry_run=True"
        )

    cmd: List[str] = [
        sys.executable,
        str(RUN_THREE_WAY),
        "--model", cfg.model,
        "--avg-bits", str(cfg.avg_bits),
        "--n-calib", str(cfg.n_calib),
        "--n-eval", str(cfg.n_eval),
        "--n-layers-sample", str(cfg.n_layers_sample),
        "--max-calib-tokens", str(cfg.max_calib_tokens),
        "--output-dir", cfg.output_dir,
        "--seed", str(cfg.seed),
        "--device", cfg.device,
        "--dtype", cfg.dtype,
        "--qjl-projections", str(cfg.qjl_projections),
        "--wf-min-bits", str(cfg.wf_min_bits),
    ]
    if cfg.wf_max_bits is not None:
        cmd += ["--wf-max-bits", str(cfg.wf_max_bits)]
    if cfg.calibration_dir is not None:
        cmd += ["--calibration-dir", cfg.calibration_dir]
        if cfg.save_calibration:
            cmd += ["--save-calibration"]
        if cfg.load_calibration:
            cmd += ["--load-calibration"]
    if cfg.smoke:
        cmd += ["--synthetic-smoke"]
    if cfg.inline_corpus_smoke:
        cmd += ["--inline-corpus-smoke"]
    if cfg.dry_run:
        cmd += ["--dry-run"]
    if cfg.force:
        cmd += ["--force"]
    cmd += list(cfg.extra)
    return cmd


def render_command(cfg: RunConfig) -> str:
    """Return a shell-escaped command string for human review/logging."""
    return " ".join(shlex.quote(p) for p in build_command(cfg))


def _model_short(model_name: str) -> str:
    """File-safe short name from a HF model id (mirror of run_three_way)."""
    base = model_name.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-") or "model"


def output_path_for(cfg: RunConfig) -> str:
    """Compute the deterministic output JSON path used by run_three_way.

    Mirrors ``experiments/run_three_way.py::_output_path`` so the launcher
    can echo the expected artifact location and check existence on the
    Modal volume without re-running the harness.
    """
    short = _model_short(cfg.model)
    fname = (
        f"{short}_b{cfg.avg_bits}_calib{cfg.n_calib}_"
        f"eval{cfg.n_eval}_seed{cfg.seed}.json"
    )
    if cfg.smoke:
        fname = "synthetic_smoke__" + fname
    if cfg.inline_corpus_smoke:
        fname = "inline_corpus_smoke__" + fname
    if cfg.dry_run:
        fname = "dryrun__" + fname
    return str(Path(cfg.output_dir) / fname)


# Pattern emitted by run_three_way.py when a result JSON is written.
_WROTE_LINE_RE = re.compile(
    r"^\[run_three_way\]\s+"
    r"(?:synthetic-smoke|inline-corpus-smoke|full|dry-run):\s+wrote\s+(\S+)",
    re.MULTILINE,
)


def parse_wrote_line(stdout_text: str) -> Optional[str]:
    """Return the path emitted by run_three_way's ``wrote ...`` line, if any."""
    if not stdout_text:
        return None
    m = _WROTE_LINE_RE.search(stdout_text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Import-safe early status helpers
# ---------------------------------------------------------------------------
#
# ``run_one`` writes a *minimal* status artifact immediately after entering
# the remote function so the operator has something to poll even if the
# project's ``experiments`` package fails to import (e.g. PYTHONPATH not
# wired correctly, broken dependency). These helpers duplicate the smallest
# slice of ``experiments/run_status.py`` so they have **no third-party or
# project imports**. Once ``experiments.run_status`` imports cleanly, the
# real ``StatusWriter`` takes over.


def _derive_run_id_safe(
    *,
    model: str,
    avg_bits: int,
    seed: int,
    n_calib: int,
    n_eval: int,
    smoke: bool = False,
    inline_corpus_smoke: bool = False,
) -> str:
    """Mirror of ``run_status.derive_run_id`` with no external imports."""
    short = _model_short(model)
    rid = f"{short}_b{avg_bits}_calib{n_calib}_eval{n_eval}_seed{seed}"
    if smoke:
        rid = "synthetic_smoke__" + rid
    if inline_corpus_smoke:
        rid = "inline_corpus_smoke__" + rid
    return rid


def _default_status_dir_safe(output_dir: str, run_id: str) -> Path:
    """Mirror of ``run_status.default_status_dir`` with no external imports."""
    out = Path(output_dir)
    parts = out.parts
    if len(parts) >= 2 and parts[1] == "results" and parts[0] == "/":
        return Path("/results") / "status" / run_id
    if "results" in parts:
        idx = parts.index("results")
        base = Path(*parts[: idx + 1])
        return base / "status" / run_id
    return out / "status" / run_id


def _safe_output_path_for(cfg: "RunConfig") -> Optional[str]:
    """Best-effort ``output_path_for`` that never raises."""
    try:
        return output_path_for(cfg)
    except Exception:
        return None


def _early_emit(
    status_dir: Path,
    stage: str,
    *,
    message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically write a minimal ``status.json`` + ``events.jsonl`` line.

    Used before ``experiments.run_status`` is imported so the operator
    sees an artifact even if the project tree fails to import. Token
    sanitization is applied via :func:`sanitize_text` on the message and
    on string values inside ``extra`` (one level deep).
    """
    import json as _json
    import socket as _socket
    import tempfile as _tempfile
    from datetime import datetime as _dt, timezone as _tz

    safe_extra: Dict[str, Any] = {}
    if extra:
        for k, v in extra.items():
            if isinstance(v, str):
                safe_extra[k] = sanitize_text(v)
            else:
                safe_extra[k] = v
    payload: Dict[str, Any] = {
        "stage": stage,
        "timestamp": _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "host": _socket.gethostname(),
        "pid": os.getpid(),
    }
    if message is not None:
        payload["message"] = sanitize_text(message)
    payload.update(safe_extra)

    try:
        status_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    status_file = status_dir / "status.json"
    events_file = status_dir / "events.jsonl"
    # Atomic snapshot.
    try:
        fd, tmp_name = _tempfile.mkstemp(
            prefix=".tmp_early_status_", dir=str(status_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                _json.dump(payload, fh, indent=2, sort_keys=True, default=str)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp_name, status_file)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
    except OSError:
        pass
    # Append-only event line.
    try:
        with open(events_file, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(payload, sort_keys=True, default=str) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Modal app construction
# ---------------------------------------------------------------------------
#
# The modal app is built lazily so importing this script in a unit-test
# environment without modal does not blow up. The ``build_modal_app``
# function is the only place that imports modal; tests skip when the
# import fails.


def build_modal_app(
    *,
    app_name: str = DEFAULT_APP_NAME,
    gpu: str = DEFAULT_GPU,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    volume_name: str = DEFAULT_VOLUME_NAME,
    hf_secret_name: str = DEFAULT_HF_SECRET,
    wandb_secret_name: Optional[str] = None,
) -> Any:
    """Construct and return ``(app, image, volume, run_one_function)``.

    Importing modal is **inside** the function; calling this without
    modal installed raises ``RuntimeError`` with an actionable message.

    The returned ``run_one_function`` is decorated as a Modal function and
    accepts a :class:`RunConfig` (or its kwargs) and runs
    ``experiments/run_three_way.py`` with the right command. Results are
    written to the mounted volume at ``cfg.output_dir`` and persisted
    across runs.
    """
    if gpu not in ALLOWED_GPUS:
        raise ValueError(
            f"GPU '{gpu}' not in allow-list {ALLOWED_GPUS}; "
            "see docs/modal_safety_protocol.md §3."
        )
    try:
        import modal  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "modal is not installed locally. Install with `pip install modal`, "
            "then `modal token new` to authenticate. The launcher itself never "
            "reads token values."
        ) from exc

    # Modal 1.4.x removed ``modal.Mount`` in favour of attaching local
    # files directly to the image with ``Image.add_local_dir``. We use
    # ``copy=False`` (the default) so files are added at container start
    # rather than baked into a build layer — keeps the function pinned
    # to the latest local commit without rebuilding the image.
    def _ignore_repo_path(p: "Path") -> bool:
        s = str(p)
        return (
            ".git/" in s
            or s.endswith(".git")
            or "/.git" in s
            or "/results/" in s
            or "/local_results/" in s
            or "__pycache__" in s
        )

    # ``serialized=True`` (set on the @app.function below) pickles this
    # function under the *local* Python's protocol. Modal rejects the call
    # with "serialized Function must use a Python version compatible with
    # remote environment" if the image's Python differs from the local
    # interpreter's ``major.minor``. We therefore pin ``add_python`` to the
    # local Python version, falling back to a known-good default when the
    # local version is outside Modal's supported add_python set.
    _local_py = f"{sys.version_info.major}.{sys.version_info.minor}"
    _MODAL_ADD_PYTHON_SUPPORTED = ("3.10", "3.11", "3.12")
    image_python = (
        _local_py if _local_py in _MODAL_ADD_PYTHON_SUPPORTED else "3.12"
    )
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04",
            add_python=image_python,
        )
        .pip_install(
            "torch>=2.6",
            "transformers>=4.40",
            "datasets>=2.14",
            "numpy",
            "scipy",
            "jsonschema>=4.18",
            "tqdm",
        )
        # Make the launcher module importable on the remote side. With
        # ``serialized=True`` Modal pickles the function (and its closure)
        # by the local module name ``launch_modal_three_way``; in detached
        # mode the remote container deserializes that pickle and must be
        # able to ``import launch_modal_three_way`` to resolve closure
        # references such as ``RunConfig``. ``scripts/`` is not a package,
        # so the only way to make the bare module name resolve on the
        # remote is to put ``/repo/scripts`` on ``PYTHONPATH``. ``/repo/src``
        # is also added so ``experiments`` / project ``src`` imports work.
        .env({"PYTHONPATH": "/repo/scripts:/repo/src:/repo"})
        # Attach the repo at /repo so the function always sees the
        # latest commit without rebuilding the image. Replaces the
        # legacy local-mount API removed in Modal 1.4.
        .add_local_dir(
            str(REPO_ROOT),
            remote_path="/repo",
            ignore=_ignore_repo_path,
        )
    )

    secrets: List[Any] = [modal.Secret.from_name(hf_secret_name)]
    if wandb_secret_name:
        secrets.append(modal.Secret.from_name(wandb_secret_name))

    app = modal.App(app_name)
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)

    # ``serialized=True`` is required because the function is defined
    # inside ``build_modal_app`` rather than at module top-level. Modal
    # 1.4 enforces this for nested-scope functions.
    @app.function(
        image=image,
        gpu=gpu,
        timeout=timeout_sec,
        secrets=secrets,
        volumes={"/results": volume},
        serialized=True,
    )
    def run_one(
        cfg_dict: Dict[str, Any],
        git_commit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Modal-side entrypoint. Runs ``run_three_way.py`` once.

        ``git_commit`` is the local repo's HEAD SHA captured at submit
        time. The Modal image excludes ``.git`` so the harness's
        ``git rev-parse`` would otherwise return the placeholder; we
        forward the local SHA via :data:`GIT_COMMIT_ENV` so the result
        JSON carries real provenance.

        Heartbeat / progress artifacts are written to
        ``/results/status/<run_id>/`` so the operator can poll the
        Modal volume for live progress even when streaming logs are
        unavailable. Child stdout/stderr are *streamed* (line-by-line)
        rather than buffered; the launcher periodically rewrites
        bounded sanitized tails into ``status.json`` so the latest tail
        is observable mid-run.

        Two layers of status are emitted:

        * **Modal-runner stages** (this function): ``modal_run_one_entered``,
          ``subprocess_env_configured``, ``subprocess_starting``,
          ``subprocess_started``, ``subprocess_progress``, ``subprocess_end``,
          plus terminal ``failure`` if Popen itself fails. These are
          observable even when the child process never launches.
        * **Benchmark stages** (the child ``run_three_way.py``): ``start``,
          ``import_ok``, ``model_load_*``, ``calibration_*``, ``eval_*``,
          ``success`` / ``failure``. Both layers write to the same
          ``<status_dir>/<run_id>/`` directory; the child's writes
          overwrite the runner's ``status.json`` as it progresses.
        """
        # Compute run_id and status path *before* importing anything from
        # the project tree so a broken project import still produces an
        # observable artifact. The fallback writer mirrors a minimal slice
        # of ``experiments.run_status.StatusWriter`` and is replaced with
        # the real writer once the import succeeds.
        cfg = RunConfig(**cfg_dict)
        cmd = build_command(cfg)
        # Replace the local repo path with the mounted /repo path.
        for i, tok in enumerate(cmd):
            if tok == str(RUN_THREE_WAY):
                cmd[i] = "/repo/experiments/run_three_way.py"

        run_id = _derive_run_id_safe(
            model=cfg.model,
            avg_bits=cfg.avg_bits,
            seed=cfg.seed,
            n_calib=cfg.n_calib,
            n_eval=cfg.n_eval,
            smoke=cfg.smoke,
            inline_corpus_smoke=cfg.inline_corpus_smoke,
        )
        if cfg.status_dir:
            status_dir = Path(cfg.status_dir) / run_id
        else:
            status_dir = _default_status_dir_safe(cfg.output_dir, run_id)
        status_path = status_dir / "status.json"

        # Try to flush the Modal volume after each emit so a polling
        # operator sees the artifact while the run is still going. We
        # capture the closure ``volume`` from build_modal_app's scope.
        def _commit_volume() -> None:
            try:
                vol = volume  # closure capture
            except NameError:
                return
            commit_fn = getattr(vol, "commit", None)
            if commit_fn is None:
                return
            try:
                commit_fn()
            except Exception:
                pass

        # Stage 1: write a minimal "entered" artifact *immediately* so we
        # have visible output even if subsequent imports fail. Tokens are
        # sanitized; only safe metadata is recorded.
        early_meta = {
            "run_id": run_id,
            "commit": git_commit,
            "model": cfg.model,
            "avg_bits": cfg.avg_bits,
            "seed": cfg.seed,
            "n_calib": cfg.n_calib,
            "n_eval": cfg.n_eval,
            "n_layers_sample": cfg.n_layers_sample,
            "expected_output_path": _safe_output_path_for(cfg),
            "status_path": str(status_path),
            "status_dir": str(status_dir),
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
        }
        _early_emit(
            status_dir,
            "modal_run_one_entered",
            message="run_one entered on Modal worker",
            extra=early_meta,
        )
        _commit_volume()

        # Lazy import: experiments package is on PYTHONPATH inside the
        # image (/repo/src). Importing at module scope would couple the
        # launcher's import-time to the project tree. If this import
        # fails we still want the early artifact above to remain.
        try:
            from experiments import run_status
        except BaseException as exc:
            _early_emit(
                status_dir,
                "failure",
                message=f"failed to import experiments.run_status: "
                        f"{type(exc).__name__}: {exc}",
                extra={**early_meta, "error_type": type(exc).__name__},
            )
            _commit_volume()
            raise

        status = run_status.StatusWriter(
            status_dir=status_dir,
            run_id=run_id,
            commit=git_commit,
            model=cfg.model,
            avg_bits=cfg.avg_bits,
            n_calib=cfg.n_calib,
            n_eval=cfg.n_eval,
            n_layers_sample=cfg.n_layers_sample,
        )
        # Pass --status-dir so the subprocess emits to the same directory
        # (a nested run_id directory under the parent).
        status_parent = str(status_dir.parent)
        cmd = cmd + ["--status-dir", status_parent]

        # Never echo HF_TOKEN — only PWD and CWD.
        env = dict(os.environ)
        # Ensure both /repo/src (for ``experiments`` / project src) and
        # /repo/scripts (for the launcher module itself, see image .env)
        # are on PYTHONPATH for the subprocess as well.
        existing_pp = env.get("PYTHONPATH", "")
        prefix = "/repo/scripts" + os.pathsep + "/repo/src"
        env["PYTHONPATH"] = (
            prefix + (os.pathsep + existing_pp if existing_pp else "")
        )
        if git_commit:
            env[GIT_COMMIT_ENV] = git_commit

        # Persistent HF / datasets cache so retries don't re-download
        # weights or dataset shards. The cache lives on the Modal volume.
        cache_paths = run_status.configure_persistent_hf_cache(
            env, volume_mount="/results"
        )

        # Stage 2: env is fully configured but the child has not yet
        # been spawned. Sanitized snapshot of the cache env for triage.
        status.emit(
            "subprocess_env_configured",
            message="subprocess env configured (PYTHONPATH, HF cache)",
            details={
                "expected_output_path": _safe_output_path_for(cfg),
                "status_path": str(status_path),
                "cache_paths": cache_paths,
                "pythonpath": env["PYTHONPATH"],
                "cwd_for_subprocess": "/repo",
                "python_executable": sys.executable,
                "python_version": (
                    f"{sys.version_info.major}.{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
            },
        )
        _commit_volume()

        status.emit(
            "subprocess_start",
            message="launching run_three_way.py subprocess",
            details={
                "command_argv": [shlex.quote(p) for p in cmd],
                "cache_paths": cache_paths,
            },
        )
        _commit_volume()

        # Stream child output incrementally so the latest tail can be
        # written into ``status.json`` mid-run. We tee stdout and stderr
        # into ring buffers (bounded by character count) so a long run
        # doesn't fill memory with logs.
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        STDOUT_TAIL_CHARS = 8000
        STDERR_TAIL_CHARS = 8000

        def _tail(buf: List[str], cap: int) -> str:
            joined = "".join(buf)
            return joined[-cap:] if len(joined) > cap else joined

        # Stage 3: about to spawn. If Popen raises (e.g. file-not-found,
        # permission, OS-level), the next emit_failure tells the operator
        # the child never started.
        status.emit(
            "subprocess_starting",
            message="about to call subprocess.Popen",
            details={
                "argv0": cmd[0] if cmd else None,
                "argv_len": len(cmd),
            },
        )
        _commit_volume()

        try:
            proc = subprocess.Popen(
                cmd,
                cwd="/repo",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except BaseException as exc:
            status.emit_failure(
                exc,
                details={
                    "phase": "popen",
                    "argv0": cmd[0] if cmd else None,
                    "expected_output_path": _safe_output_path_for(cfg),
                },
            )
            _commit_volume()
            raise

        # Stage 4: Popen succeeded — child has a PID.
        status.emit(
            "subprocess_started",
            message=f"subprocess PID={proc.pid} running",
            details={"pid": proc.pid},
        )
        _commit_volume()
        try:
            import select
            import time as _time

            stdout_fd = proc.stdout.fileno() if proc.stdout else None
            stderr_fd = proc.stderr.fileno() if proc.stderr else None
            fds = [fd for fd in (stdout_fd, stderr_fd) if fd is not None]
            last_status_emit = _time.time()
            STATUS_EMIT_INTERVAL = 30.0  # seconds between progress emits
            line_count = 0
            while fds:
                ready, _, _ = select.select(fds, [], [], 5.0)
                progressed = False
                for fd in ready:
                    if fd == stdout_fd and proc.stdout is not None:
                        line = proc.stdout.readline()
                        if not line:
                            try:
                                fds.remove(fd)
                            except ValueError:
                                pass
                            continue
                        stdout_lines.append(line)
                        if len("".join(stdout_lines)) > 4 * STDOUT_TAIL_CHARS:
                            stdout_lines[:] = [
                                _tail(stdout_lines, STDOUT_TAIL_CHARS)
                            ]
                        # Echo to Modal log stream so logs (when accessible)
                        # also see the child's output in real time.
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        line_count += 1
                        progressed = True
                    elif fd == stderr_fd and proc.stderr is not None:
                        line = proc.stderr.readline()
                        if not line:
                            try:
                                fds.remove(fd)
                            except ValueError:
                                pass
                            continue
                        stderr_lines.append(line)
                        if len("".join(stderr_lines)) > 4 * STDERR_TAIL_CHARS:
                            stderr_lines[:] = [
                                _tail(stderr_lines, STDERR_TAIL_CHARS)
                            ]
                        sys.stderr.write(line)
                        sys.stderr.flush()
                        line_count += 1
                        progressed = True
                # Periodic status snapshot so the operator sees progress
                # even when individual lines aren't dramatic.
                now = _time.time()
                if now - last_status_emit >= STATUS_EMIT_INTERVAL:
                    last_status_emit = now
                    status.emit(
                        "subprocess_progress",
                        message=f"subprocess alive; lines={line_count}",
                        stdout_tail=_tail(stdout_lines, STDOUT_TAIL_CHARS),
                        stderr_tail=_tail(stderr_lines, STDERR_TAIL_CHARS),
                        details={"line_count": line_count},
                    )
                    _commit_volume()
                if not progressed and proc.poll() is not None and not fds:
                    break
                if proc.poll() is not None and not ready:
                    # Drain remaining buffered output then exit.
                    if proc.stdout is not None:
                        rest = proc.stdout.read()
                        if rest:
                            stdout_lines.append(rest)
                            sys.stdout.write(rest)
                    if proc.stderr is not None:
                        rest = proc.stderr.read()
                        if rest:
                            stderr_lines.append(rest)
                            sys.stderr.write(rest)
                    break
            returncode = proc.wait()
        except BaseException as exc:
            # Kill the child if we ever get here so we don't orphan it.
            try:
                proc.kill()
            except Exception:
                pass
            status.emit_failure(
                exc,
                stdout_tail=_tail(stdout_lines, STDOUT_TAIL_CHARS),
                stderr_tail=_tail(stderr_lines, STDERR_TAIL_CHARS),
            )
            _commit_volume()
            raise

        stdout_text = "".join(stdout_lines)
        stderr_text = "".join(stderr_lines)
        # Compute the deterministic output path the harness would have used
        # so the caller can show it even when the subprocess crashed before
        # it could print its own ``wrote ...`` line.
        try:
            expected_output = output_path_for(cfg)
        except Exception:
            expected_output = None
        try:
            result_exists = bool(
                expected_output and Path(expected_output).is_file()
            )
        except OSError:
            result_exists = False
        parsed_output = parse_wrote_line(stdout_text)

        if returncode == 0:
            status.emit(
                "subprocess_end",
                message="subprocess exited cleanly",
                details={
                    "returncode": returncode,
                    "result_exists": result_exists,
                    "output_path": expected_output,
                },
                stdout_tail=stdout_text[-STDOUT_TAIL_CHARS:],
                stderr_tail=stderr_text[-STDERR_TAIL_CHARS:],
            )
            _commit_volume()
        else:
            # Always emit a failure artifact so the operator has something
            # to inspect even when Modal logs are unavailable.
            status.emit(
                "failure",
                message=f"subprocess exited with rc={returncode}",
                error=f"subprocess exited with rc={returncode}",
                details={
                    "returncode": returncode,
                    "result_exists": result_exists,
                    "output_path": expected_output,
                },
                stdout_tail=stdout_text[-STDOUT_TAIL_CHARS:],
                stderr_tail=stderr_text[-STDERR_TAIL_CHARS:],
            )
            _commit_volume()

        return {
            "returncode": returncode,
            "stdout_tail": stdout_text[-4000:],
            "stderr_tail": stderr_text[-4000:],
            "command": " ".join(shlex.quote(p) for p in cmd),
            "model": cfg.model,
            "avg_bits": cfg.avg_bits,
            "seed": cfg.seed,
            "output_path": expected_output,
            "parsed_output_path": parsed_output,
            "result_exists": result_exists,
            "git_commit": git_commit,
            "status_dir": str(status_dir),
            "status_path": str(status.status_path),
        }

    return app, image, volume, run_one


# ---------------------------------------------------------------------------
# Matrix-config support
# ---------------------------------------------------------------------------


def load_matrix(path: Path) -> List[RunConfig]:
    """Load a list of :class:`RunConfig` from a JSON config file.

    The file format is a JSON array of objects whose keys match the
    :class:`RunConfig` fields. Unknown keys raise ``ValueError`` so a typo
    in a config is caught at load time, not silently dropped.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Matrix file {path} must be a JSON array")
    out: List[RunConfig] = []
    valid_keys = {f for f in RunConfig.__dataclass_fields__}
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Matrix entry {i} is not an object")
        unknown = set(entry.keys()) - valid_keys
        if unknown:
            raise ValueError(
                f"Matrix entry {i} has unknown keys {sorted(unknown)}"
            )
        # Tuples are encoded as lists in JSON.
        if "extra" in entry and isinstance(entry["extra"], list):
            entry = {**entry, "extra": tuple(entry["extra"])}
        out.append(RunConfig(**entry))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="launch_modal_three_way.py",
        description=(
            "Modal launcher for the SpectralQuant v2 three-way benchmark. "
            "Constructs the run command for one config (or a matrix file) "
            "and submits it to a Modal function. Use --dry-run to print the "
            "command without launching."
        ),
    )
    p.add_argument("--model", type=str, help="HF model id")
    p.add_argument("--avg-bits", type=int, dest="avg_bits",
                   help="Average bits per element (1..16)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-calib", type=int, default=DEFAULT_N_CALIB,
                   dest="n_calib")
    p.add_argument("--n-eval", type=int, default=DEFAULT_N_EVAL,
                   dest="n_eval")
    p.add_argument("--n-layers-sample", type=int,
                   default=DEFAULT_N_LAYERS_SAMPLE,
                   dest="n_layers_sample")
    p.add_argument("--max-calib-tokens", type=int,
                   default=DEFAULT_MAX_CALIB_TOKENS,
                   dest="max_calib_tokens")
    p.add_argument("--output-dir", type=str,
                   default="/results/three_way",
                   dest="output_dir")
    p.add_argument("--calibration-dir", type=str,
                   default="/results/calibration",
                   dest="calibration_dir")
    p.add_argument("--save-calibration",
                   action="store_true",
                   default=True,
                   dest="save_calibration")
    p.add_argument("--no-save-calibration",
                   action="store_false",
                   dest="save_calibration")
    p.add_argument("--load-calibration", action="store_true",
                   dest="load_calibration",
                   help="Resume from a saved calibration .pt artifact")
    p.add_argument("--smoke", action="store_true",
                   help="Run --synthetic-smoke instead of the full path")
    p.add_argument("--inline-corpus-smoke", action="store_true",
                   dest="inline_corpus_smoke",
                   help=("Run the full HF model path with a deterministic "
                         "inline corpus instead of HF datasets.load_dataset. "
                         "Harness validation only — not paper-valid evidence."))
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print the command that would be launched and exit")
    p.add_argument("--force", action="store_true", dest="force",
                   help=(
                       "Pass --force to run_three_way.py and overwrite an "
                       "existing result JSON. Use only for intentional reruns."
                   ))
    p.add_argument("--gpu", type=str, default=DEFAULT_GPU,
                   choices=ALLOWED_GPUS)
    p.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC,
                   dest="timeout_sec")
    p.add_argument("--matrix-config", type=str, default=None,
                   dest="matrix_config",
                   help="Path to a JSON array of RunConfig dicts to run")
    p.add_argument("--volume-name", type=str, default=DEFAULT_VOLUME_NAME,
                   dest="volume_name")
    p.add_argument("--app-name", type=str, default=DEFAULT_APP_NAME,
                   dest="app_name")
    p.add_argument("--hf-secret-name", type=str, default=DEFAULT_HF_SECRET,
                   dest="hf_secret_name")
    p.add_argument("--wandb-secret-name", type=str, default=None,
                   dest="wandb_secret_name")
    p.add_argument("--qjl-projections", type=int, default=64,
                   dest="qjl_projections")
    p.add_argument("--wf-min-bits", type=int, default=0,
                   dest="wf_min_bits")
    p.add_argument("--wf-max-bits", type=int, default=None,
                   dest="wf_max_bits")
    p.add_argument("--status-dir", type=str, default=None,
                   dest="status_dir",
                   help=(
                       "Optional override for the *parent* directory under "
                       "which heartbeat/progress JSON artifacts are written. "
                       "Per-run artifacts live at <status_dir>/<run_id>/. "
                       "Defaults to a sibling of --output-dir if omitted "
                       "(see experiments/run_status.py::default_status_dir)."
                   ))
    p.add_argument("--print-modal-app", action="store_true",
                   dest="print_modal_app",
                   help="Verify modal-app construction (requires modal "
                        "installed) and print a summary, then exit.")
    return p


def configs_from_args(ns: argparse.Namespace) -> List[RunConfig]:
    """Resolve a list of :class:`RunConfig` from a parsed argparse Namespace."""
    if ns.matrix_config:
        return load_matrix(Path(ns.matrix_config))
    if ns.model is None or ns.avg_bits is None:
        raise SystemExit(
            "When --matrix-config is not given, --model and --avg-bits "
            "are required."
        )
    cfg = RunConfig(
        model=ns.model,
        avg_bits=int(ns.avg_bits),
        seed=int(ns.seed),
        n_calib=int(ns.n_calib),
        n_eval=int(ns.n_eval),
        n_layers_sample=int(ns.n_layers_sample),
        max_calib_tokens=int(ns.max_calib_tokens),
        output_dir=ns.output_dir,
        calibration_dir=ns.calibration_dir,
        save_calibration=bool(ns.save_calibration),
        load_calibration=bool(ns.load_calibration),
        smoke=bool(ns.smoke),
        inline_corpus_smoke=bool(getattr(ns, "inline_corpus_smoke", False)),
        force=bool(ns.force),
        dry_run=False,  # the run itself is full; --dry-run only prints
        qjl_projections=int(ns.qjl_projections),
        wf_min_bits=int(ns.wf_min_bits),
        wf_max_bits=(
            None if ns.wf_max_bits is None else int(ns.wf_max_bits)
        ),
        status_dir=ns.status_dir,
    )
    return [cfg]


def print_run_result(
    result: Dict[str, Any],
    *,
    stream: Any = None,
    tail_chars: int = 2000,
) -> None:
    """Print a sanitized, operator-readable summary of a remote run result.

    Includes returncode, expected/parsed output path, result_exists, and
    the *sanitized* tail of stdout/stderr. Secret values are redacted via
    :func:`sanitize_text` before any output is written.
    """
    out = stream if stream is not None else sys.stdout
    rc = result.get("returncode", "?")
    print(f"[launch_modal_three_way] returncode={rc}", file=out)
    model = result.get("model")
    bits = result.get("avg_bits")
    seed = result.get("seed")
    if model is not None:
        print(
            f"[launch_modal_three_way] config: model={model} "
            f"avg_bits={bits} seed={seed}",
            file=out,
        )
    op = result.get("output_path")
    parsed = result.get("parsed_output_path")
    exists = result.get("result_exists")
    if op:
        print(f"[launch_modal_three_way] output_path: {op}", file=out)
    if parsed and parsed != op:
        print(
            f"[launch_modal_three_way] parsed_output_path: {parsed}",
            file=out,
        )
    if exists is not None:
        print(
            f"[launch_modal_three_way] result_exists: {bool(exists)}",
            file=out,
        )
    git_commit = result.get("git_commit")
    if git_commit:
        print(
            f"[launch_modal_three_way] git_commit: {git_commit}",
            file=out,
        )
    status_dir = result.get("status_dir")
    if status_dir:
        print(
            f"[launch_modal_three_way] status_dir: {status_dir}",
            file=out,
        )
    status_path = result.get("status_path")
    if status_path:
        print(
            f"[launch_modal_three_way] status_path: {status_path}",
            file=out,
        )
    stdout_tail = result.get("stdout_tail") or ""
    stderr_tail = result.get("stderr_tail") or ""
    sanitized_stdout = sanitize_text(stdout_tail)
    sanitized_stderr = sanitize_text(stderr_tail)
    # Highlight the synthetic-smoke / inline-corpus-smoke marker so the
    # operator can confirm the write actually happened.
    for line in sanitized_stdout.splitlines():
        if (
            "[run_three_way] synthetic-smoke: wrote" in line
            or "[run_three_way] inline-corpus-smoke: wrote" in line
        ):
            print(f"[launch_modal_three_way] smoke-marker: {line}", file=out)
            break
    print("[launch_modal_three_way] --- stdout_tail ---", file=out)
    if sanitized_stdout:
        print(sanitized_stdout[-tail_chars:], file=out)
    else:
        print("(empty)", file=out)
    print("[launch_modal_three_way] --- stderr_tail ---", file=out)
    if sanitized_stderr:
        print(sanitized_stderr[-tail_chars:], file=out)
    else:
        print("(empty)", file=out)
    print("[launch_modal_three_way] --- end ---", file=out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    cfgs = configs_from_args(ns)

    if ns.dry_run:
        commit = detect_local_git_commit(REPO_ROOT)
        print(f"[launch_modal_three_way] {len(cfgs)} run(s):")
        if commit:
            print(
                f"  local git commit (forwarded as "
                f"{GIT_COMMIT_ENV}): {commit}"
            )
        else:
            print("  local git commit: <unavailable>")
        for cfg in cfgs:
            print(f"  - model={cfg.model}, bits={cfg.avg_bits}, seed={cfg.seed}")
            print(f"    cmd: {render_command(cfg)}")
        return 0

    if ns.print_modal_app:
        try:
            app, _image, _volume, _run_one = build_modal_app(
                app_name=ns.app_name,
                gpu=ns.gpu,
                timeout_sec=int(ns.timeout_sec),
                volume_name=ns.volume_name,
                hf_secret_name=ns.hf_secret_name,
                wandb_secret_name=ns.wandb_secret_name,
            )
        except RuntimeError as exc:
            print(f"[launch_modal_three_way] {exc}", file=sys.stderr)
            return 2
        print(f"[launch_modal_three_way] app: {ns.app_name}")
        print(f"  gpu: {ns.gpu}")
        print(f"  timeout_sec: {ns.timeout_sec}")
        print(f"  volume: {ns.volume_name}")
        print(f"  hf_secret_name: {ns.hf_secret_name}")
        if ns.wandb_secret_name:
            print(f"  wandb_secret_name: {ns.wandb_secret_name}")
        return 0

    # Default branch: actually launch on Modal. Implemented by deferring
    # to ``modal run`` so the user sees streaming logs and can Ctrl-C
    # cleanly.
    try:
        app, _image, _volume, run_one = build_modal_app(
            app_name=ns.app_name,
            gpu=ns.gpu,
            timeout_sec=int(ns.timeout_sec),
            volume_name=ns.volume_name,
            hf_secret_name=ns.hf_secret_name,
            wandb_secret_name=ns.wandb_secret_name,
        )
    except RuntimeError as exc:
        print(f"[launch_modal_three_way] {exc}", file=sys.stderr)
        print(
            "[launch_modal_three_way] tip: run with --dry-run to see the "
            "command this would launch, or --print-modal-app to verify the "
            "Modal app spec.",
            file=sys.stderr,
        )
        return 2

    git_commit = detect_local_git_commit(REPO_ROOT)
    if git_commit:
        print(
            f"[launch_modal_three_way] forwarding local commit "
            f"{git_commit[:12]} via {GIT_COMMIT_ENV}"
        )
    else:
        print(
            "[launch_modal_three_way] WARNING: could not detect local git "
            "commit; result JSON will record the placeholder '0000000'."
        )

    rc_total = 0
    with app.run():
        for cfg in cfgs:
            print(f"[launch_modal_three_way] launching: {cfg.model} b={cfg.avg_bits} seed={cfg.seed}")
            from dataclasses import asdict
            result = run_one.remote(asdict(cfg), git_commit=git_commit)
            print_run_result(result)
            rc = int(result.get("returncode", -1))
            if rc != 0:
                rc_total = rc
                # Honour the safety protocol: stop the sweep on a permanent
                # error so the operator sees it before more dollars burn.
                print(
                    "[launch_modal_three_way] aborting sweep on non-zero rc",
                    file=sys.stderr,
                )
                break
    return rc_total


# ---------------------------------------------------------------------------
# Module-scope Modal app for ``modal run``
# ---------------------------------------------------------------------------
#
# ``modal run scripts/launch_modal_three_way.py::main_entry -- ...`` requires
# Modal to import this file and find both an ``App`` and a callable named
# ``main_entry`` at module scope. We build them eagerly with default settings
# when ``modal`` is importable so the detached invocation works without a
# Python wrapper:
#
#   modal run -d scripts/launch_modal_three_way.py::main_entry -- \
#       --model mistralai/Mistral-7B-v0.3 --avg-bits 3
#
# The construction is wrapped in a try/except ImportError so unit tests that
# monkeypatch ``modal`` to ``None`` continue to import this module cleanly;
# in that case ``app``, ``run_one``, and ``main_entry`` are all ``None`` and
# ``build_modal_app()`` raises with an actionable message.

app: Any = None
run_one: Any = None
main_entry: Any = None
volume: Any = None

try:
    import modal as _modal_for_module  # noqa: F401  # presence check only
    app, _image, volume, run_one = build_modal_app()

    @app.local_entrypoint()
    def main_entry(
        model: str,
        avg_bits: int,
        seed: int = 42,
        n_calib: int = DEFAULT_N_CALIB,
        n_eval: int = DEFAULT_N_EVAL,
        n_layers_sample: int = DEFAULT_N_LAYERS_SAMPLE,
        max_calib_tokens: int = DEFAULT_MAX_CALIB_TOKENS,
        output_dir: str = "/results/three_way",
        calibration_dir: str = "/results/calibration",
        save_calibration: bool = True,
        load_calibration: bool = False,
        smoke: bool = False,
        inline_corpus_smoke: bool = False,
        force: bool = False,
        qjl_projections: int = 64,
        wf_min_bits: int = 0,
        wf_max_bits: int = -1,
        status_dir: str = "",
    ) -> None:
        """Modal local entrypoint — the detached-friendly invocation path.

        Mirrors the CLI flags of the Python wrapper (``main()``) but is
        invoked through ``modal run [-d] script.py::main_entry``. Modal
        parses the typed parameters into CLI options automatically.

        ``wf_max_bits`` is exposed as ``int`` (default ``-1``) because
        Modal's CLI parser does not accept ``Optional[int]`` cleanly;
        ``-1`` is converted to ``None`` before constructing
        :class:`RunConfig`.

        ``status_dir`` is the *parent* directory for status artifacts;
        the empty string (the default) means "derive from output_dir via
        :func:`experiments.run_status.default_status_dir`". It is exposed
        as ``str`` (not ``Optional[str]``) because Modal's CLI parser
        does not accept ``Optional[str]`` cleanly.
        """
        cfg = RunConfig(
            model=model,
            avg_bits=int(avg_bits),
            seed=int(seed),
            n_calib=int(n_calib),
            n_eval=int(n_eval),
            n_layers_sample=int(n_layers_sample),
            max_calib_tokens=int(max_calib_tokens),
            output_dir=output_dir,
            calibration_dir=calibration_dir,
            save_calibration=bool(save_calibration),
            load_calibration=bool(load_calibration),
            smoke=bool(smoke),
            inline_corpus_smoke=bool(inline_corpus_smoke),
            force=bool(force),
            dry_run=False,
            qjl_projections=int(qjl_projections),
            wf_min_bits=int(wf_min_bits),
            wf_max_bits=None if int(wf_max_bits) < 0 else int(wf_max_bits),
            status_dir=(status_dir or None),
        )
        from dataclasses import asdict
        git_commit = detect_local_git_commit(REPO_ROOT)
        if git_commit:
            print(
                f"[launch_modal_three_way] forwarding local commit "
                f"{git_commit[:12]} via {GIT_COMMIT_ENV}"
            )
        else:
            print(
                "[launch_modal_three_way] WARNING: could not detect local "
                "git commit; result JSON will record placeholder '0000000'."
            )
        print(
            f"[launch_modal_three_way] launching (entrypoint): "
            f"{cfg.model} b={cfg.avg_bits} seed={cfg.seed} "
            f"smoke={cfg.smoke} inline_corpus_smoke={cfg.inline_corpus_smoke}"
        )
        # Detached path: use ``.spawn(...)`` so the remote function keeps
        # running after the local ``modal run -d`` client disconnects.
        # ``.remote()`` is synchronous and Modal cancels the call when the
        # local caller drops the connection (the cause of the
        # "remote() and .map() calls in detached apps may be canceled when
        # the local caller disconnects" warning we observed in stderr).
        # ``spawn`` returns a ``FunctionCall`` whose ``object_id`` is the
        # poll handle the operator uses to fetch the result later.
        call = run_one.spawn(asdict(cfg), git_commit=git_commit)
        expected_output = output_path_for(cfg)
        call_id = getattr(call, "object_id", None) or str(call)
        # Compute the deterministic status dir that ``run_one`` will use.
        # We import lazily so unit tests that monkeypatch the experiments
        # tree still work.
        try:
            from experiments import run_status as _run_status
            run_id = _run_status.derive_run_id(
                model=cfg.model,
                avg_bits=cfg.avg_bits,
                seed=cfg.seed,
                n_calib=cfg.n_calib,
                n_eval=cfg.n_eval,
                smoke=cfg.smoke,
                inline_corpus_smoke=cfg.inline_corpus_smoke,
            )
            if cfg.status_dir:
                status_dir = Path(cfg.status_dir) / run_id
            else:
                status_dir = _run_status.default_status_dir(
                    cfg.output_dir, run_id
                )
            status_path = str(status_dir / "status.json")
        except Exception:
            status_dir = None
            status_path = None
        print(
            f"[launch_modal_three_way] spawned remote call: "
            f"call_id={call_id}"
        )
        print(
            f"[launch_modal_three_way] expected_output_path: "
            f"{expected_output}"
        )
        if status_path:
            print(
                f"[launch_modal_three_way] status_path: {status_path}"
            )
            print(
                f"[launch_modal_three_way] poll status: modal volume get "
                f"{DEFAULT_VOLUME_NAME} {status_path} -"
            )
        print(
            f"[launch_modal_three_way] poll: modal volume ls "
            f"{DEFAULT_VOLUME_NAME} {expected_output}"
        )
        print(
            f"[launch_modal_three_way] result: python3 -c "
            f"\"import modal; "
            f"print(modal.FunctionCall.from_id('{call_id}').get(timeout=0))\""
        )
except ImportError:
    # Modal not installed in the local env; tests and ``--dry-run`` still
    # work because they don't need the module-level app.
    pass
except Exception as _module_app_exc:  # pragma: no cover - defensive
    # If module-level construction fails for any non-ImportError reason
    # (e.g. Modal client can't reach control plane during import), keep
    # ``app``/``run_one``/``main_entry`` as ``None`` so the in-process
    # ``main()`` path can still surface a clear error to the operator.
    print(
        f"[launch_modal_three_way] WARNING: module-level Modal app build "
        f"failed: {type(_module_app_exc).__name__}: {_module_app_exc}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
