"""Heartbeat / progress status artifacts for SpectralQuant v2 runs.

Modal logs are often inaccessible due to control-plane resource limits. To
keep a remote run observable even when the streaming logs are gone, we
write small JSON status files to the durable Modal volume at well-known
paths during each run.

A ``StatusWriter`` is constructed once per run at the top of the harness
(or inside the launcher's remote function) and stages are emitted via
:meth:`StatusWriter.emit`. Each emit writes ``status.json`` (current state)
and appends to ``events.jsonl`` (history). Writes are atomic
(tempfile + ``os.replace``) so a partial write can never appear at the
canonical path even if the process is killed mid-emit.

The artifact path is *deterministic*: same ``output_dir`` + ``run_id``
always picks the same status directory. Operators can poll
``/results/status/<run_id>/status.json`` directly via ``modal volume get``
without needing log access.

Sanitization: every text payload threaded into a status artifact (model
ids, log tails, error messages, traceback) is run through
:func:`sanitize_text` so HF / Modal / W&B / generic token literals are
redacted before being written. The launcher's ``sanitize_text`` covers
the same set; this module duplicates the smaller version so the harness
can run independently without importing the launcher.

This module has **no third-party dependencies** and is safe to import in
any environment, including a clean container with only stdlib + numpy.
"""

from __future__ import annotations

import json
import os
import re
import socket
import tempfile
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Sanitization (duplicate of launcher's, kept lean & dependency-free)
# ---------------------------------------------------------------------------

#: Stages emitted in the canonical order. Documented here so dashboards or
#: tests can compare against the expected progression. Not strictly
#: enforced — emitting an unknown stage is allowed, but new stages should
#: be added here so they are part of the public contract.
KNOWN_STAGES: tuple = (
    # Modal-runner stages (emitted by scripts/launch_modal_three_way.py
    # before/around the child subprocess). These are observable even when
    # the child never starts.
    "modal_run_one_entered",
    "subprocess_env_configured",
    "subprocess_starting",
    "subprocess_started",
    # Benchmark stages (emitted by experiments/run_three_way.py inside
    # the child subprocess).
    "start",
    "import_ok",
    "model_download_start",
    "model_download_end",
    "model_load_start",
    "model_load_end",
    "dataset_load_start",
    "dataset_load_end",
    # Inline-corpus-smoke (harness validation only — bypasses HF datasets).
    "dataset_inline_start",
    "dataset_inline_end",
    "calibration_start",
    "calib_eigh_start",
    "calib_eigh_end",
    "calib_capture_start",
    "calib_capture_end",
    "calib_fit_start",
    "calib_fit_progress",
    "calib_fit_end",
    "calibration_end",
    "eval_start",
    "eval_task_start",
    "eval_progress",
    "eval_task_end",
    "eval_end",
    "subprocess_start",
    "subprocess_progress",
    "subprocess_end",
    "success",
    "failure",
)

SECRET_NAME_PATTERNS: tuple = (
    "TOKEN",
    "SECRET",
    "API_KEY",
    "APIKEY",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
)

_TOKEN_LITERAL_PATTERNS: tuple = (
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

    See ``scripts/launch_modal_three_way.py::sanitize_text`` for the
    matching reference implementation. We duplicate the (small) logic
    here so the harness has zero hard dependency on the launcher module.
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
# Path helpers
# ---------------------------------------------------------------------------

#: Default subdirectory under ``output_dir``'s mount point used to store
#: status artifacts. ``status_dir_for(...)`` builds the per-run path.
DEFAULT_STATUS_SUBDIR = "status"


def _model_short(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-") or "model"


def derive_run_id(
    model: str,
    avg_bits: int,
    seed: int,
    n_calib: int,
    n_eval: int,
    *,
    smoke: bool = False,
    inline_corpus_smoke: bool = False,
) -> str:
    """Build a deterministic ``run_id`` matching the result-JSON naming.

    Same shape as ``experiments/run_three_way.py::_output_path`` so the
    status directory name lines up with the output JSON's filename.

    ``inline_corpus_smoke`` is mutually exclusive with ``smoke`` (the
    harness rejects the combination at parse time); we still apply both
    prefixes here in a stable order so any caller that erroneously sets
    both gets a deterministic, distinct ``run_id``.
    """
    short = _model_short(model)
    rid = f"{short}_b{avg_bits}_calib{n_calib}_eval{n_eval}_seed{seed}"
    if smoke:
        rid = "synthetic_smoke__" + rid
    if inline_corpus_smoke:
        rid = "inline_corpus_smoke__" + rid
    return rid


def default_status_dir(output_dir: Path | str, run_id: str) -> Path:
    """Compute the canonical status directory for a given run.

    Layout::

        <output_dir_mount>/
          status/
            <run_id>/
              status.json
              events.jsonl

    The mount point is taken to be the *parent* of ``output_dir`` when
    ``output_dir`` itself is e.g. ``/results/three_way`` so the status
    tree is a sibling, not a child, of the run output. If we cannot
    identify a clear mount sibling we fall back to a ``status/`` subdir
    of ``output_dir`` itself.
    """
    out = Path(output_dir)
    parts = out.parts
    if len(parts) >= 2 and parts[1] == "results" and parts[0] == "/":
        # Mounted at /results/<something>: place status at /results/status/.
        return Path("/results") / DEFAULT_STATUS_SUBDIR / run_id
    if "results" in parts:
        idx = parts.index("results")
        base = Path(*parts[: idx + 1])
        return base / DEFAULT_STATUS_SUBDIR / run_id
    return out / DEFAULT_STATUS_SUBDIR / run_id


# ---------------------------------------------------------------------------
# Atomic writers
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic JSON write: tempfile in same dir + ``os.replace``.

    Mirrors ``run_three_way.atomic_write_json`` but without schema
    validation — status artifacts are not subject to the result schema.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_status_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append ``payload`` as a single JSON line to ``path``.

    Uses one ``open(..., "a")`` + ``fsync`` per call — adequate for the
    handful of stage events we emit per run. Best-effort: a write that
    fails is logged but never raised, so emitting status never breaks
    the surrounding run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, default=str)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
    except OSError:
        # Do not let a status failure crash the run.
        pass


# ---------------------------------------------------------------------------
# StatusWriter
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _truncate_tail(text: str, *, max_chars: int = 4000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


@dataclass
class StatusWriter:
    """Atomic per-run status emitter.

    See :data:`KNOWN_STAGES` for the canonical stage progression. Each
    :meth:`emit` updates ``status.json`` to the latest snapshot and
    appends a record to ``events.jsonl`` for history.
    """

    status_dir: Path
    run_id: str
    commit: Optional[str] = None
    model: Optional[str] = None
    avg_bits: Optional[int] = None
    n_calib: Optional[int] = None
    n_eval: Optional[int] = None
    n_layers_sample: Optional[int] = None
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    # Internal state — not for direct mutation by callers.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @property
    def status_path(self) -> Path:
        return Path(self.status_dir) / "status.json"

    @property
    def events_path(self) -> Path:
        return Path(self.status_dir) / "events.jsonl"

    def _base_meta(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "run_id": self.run_id,
            "commit": self.commit,
            "model": self.model,
            "avg_bits": self.avg_bits,
            "n_calib": self.n_calib,
            "n_eval": self.n_eval,
            "n_layers_sample": self.n_layers_sample,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        if self.extra_meta:
            meta.update(self.extra_meta)
        return meta

    def emit(
        self,
        stage: str,
        *,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        tb: Optional[str] = None,
        stdout_tail: Optional[str] = None,
        stderr_tail: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write a status snapshot and append an event record.

        Returns the event payload as a dict so the caller can also log
        it to stdout if desired. Sanitization is applied before any data
        leaves this method.
        """
        with self._lock:
            ts = _utc_now_iso()
            sanitized_message = sanitize_text(message) if message else None
            sanitized_error = sanitize_text(error) if error else None
            sanitized_tb = sanitize_text(tb) if tb else None
            sanitized_stdout = (
                sanitize_text(_truncate_tail(stdout_tail)) if stdout_tail else None
            )
            sanitized_stderr = (
                sanitize_text(_truncate_tail(stderr_tail)) if stderr_tail else None
            )
            event: Dict[str, Any] = {
                "stage": str(stage),
                "timestamp": ts,
                **self._base_meta(),
            }
            if sanitized_message is not None:
                event["message"] = sanitized_message
            if details is not None:
                event["details"] = _sanitize_dict(details)
            if sanitized_error is not None:
                event["error"] = sanitized_error
            if sanitized_tb is not None:
                event["traceback"] = sanitized_tb
            if sanitized_stdout is not None:
                event["stdout_tail"] = sanitized_stdout
            if sanitized_stderr is not None:
                event["stderr_tail"] = sanitized_stderr
            # Write status.json (snapshot) and append events.jsonl.
            atomic_write_json(self.status_path, event)
            append_jsonl(self.events_path, event)
            return event

    def emit_failure(
        self,
        exc: BaseException,
        *,
        stdout_tail: Optional[str] = None,
        stderr_tail: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convenience: emit a ``failure`` event derived from an exception."""
        return self.emit(
            "failure",
            message=f"{type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
            tb="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            details=details,
        )


def _sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize string values inside a JSON-able dict (one level deep)."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = sanitize_text(v)
        elif isinstance(v, dict):
            out[k] = _sanitize_dict(v)
        elif isinstance(v, list):
            out[k] = [
                sanitize_text(x) if isinstance(x, str) else x for x in v
            ]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# HF / datasets cache configuration
# ---------------------------------------------------------------------------
#
# To survive retries without re-downloading model weights and dataset
# splits, the launcher pins the HF / datasets cache directories to
# subdirectories of the Modal volume. ``configure_persistent_hf_cache``
# updates ``env`` in place with the relevant variables and creates the
# target directories. It is idempotent and safe to call multiple times.

#: Subpath on the Modal volume where the persistent HF / datasets cache
#: lives. Operators must NOT delete this tree without first auditing for
#: in-flight runs; see ``docs/modal_safety_protocol.md`` §6c.
HF_CACHE_SUBDIR = "hf_cache"

#: Env var names we set/override. Keep in sync with the HuggingFace docs:
#:   * HF_HOME                — root for HF clients (token cache, modules).
#:   * HUGGINGFACE_HUB_CACHE  — model snapshot cache (used by huggingface_hub).
#:   * TRANSFORMERS_CACHE     — legacy transformers cache (still respected).
#:   * HF_DATASETS_CACHE      — datasets library cache.
#:   * XDG_CACHE_HOME         — fallback for libraries that respect XDG.
HF_CACHE_ENV_VARS: tuple = (
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE",
    "XDG_CACHE_HOME",
)


def configure_persistent_hf_cache(
    env: Dict[str, str],
    *,
    volume_mount: str = "/results",
    subdir: str = HF_CACHE_SUBDIR,
    create: bool = True,
) -> Dict[str, str]:
    """Point the HF / datasets caches at a persistent volume directory.

    Mutates ``env`` in place and returns the per-var cache mapping for
    audit. The caller (the launcher's ``run_one`` Modal function) passes
    the mutated env into ``subprocess.run``.

    Layout::

        <volume_mount>/
          hf_cache/
            hub/         # HUGGINGFACE_HUB_CACHE / TRANSFORMERS_CACHE
            datasets/    # HF_DATASETS_CACHE
            xdg/         # XDG_CACHE_HOME
            home/        # HF_HOME (modules, token cache)
    """
    base = Path(volume_mount) / subdir
    paths = {
        "HF_HOME": str(base / "home"),
        "HUGGINGFACE_HUB_CACHE": str(base / "hub"),
        "TRANSFORMERS_CACHE": str(base / "hub"),
        "HF_DATASETS_CACHE": str(base / "datasets"),
        "XDG_CACHE_HOME": str(base / "xdg"),
    }
    if create:
        for p in set(paths.values()):
            try:
                Path(p).mkdir(parents=True, exist_ok=True)
            except OSError:
                # Best-effort: a read-only mount or missing parent will
                # surface as a download error later, with a clearer
                # message than this would produce.
                pass
    for k, v in paths.items():
        env[k] = v
    return paths
