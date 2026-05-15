#!/usr/bin/env python3
"""Modal launcher for the four next-stage evaluation harnesses.

This script is the operator-facing wrapper around a Modal app that runs
one of the v2 evaluation harnesses on a single configuration:

* ``perplexity`` — :mod:`experiments.run_perplexity`
* ``longbench``  — :mod:`experiments.run_longbench`
* ``generation`` — :mod:`experiments.run_generation`
* ``latency``    — :mod:`experiments.run_latency`

The launcher mirrors :mod:`scripts.launch_modal_three_way`'s discipline:

* No tokens are read or echoed; HF / Modal / W&B secrets are referenced
  by name only.
* The harness inside Modal writes results to a persistent volume at a
  deterministic path so retries skip already-completed runs.
* Status artifacts are written under
  ``/results/status/<family>/<run_id>/`` so a polling operator can
  observe progress without log access. (The per-family directory is
  appended by the in-Modal wrapper from the family inferred from the
  config; the launcher's outer ``--status-dir``, when set, overrides
  the parent prefix.)
* Detached mode is supported via ``modal run -d
  scripts/launch_modal_eval.py::main_entry --family ... --model ... ...``.
  Note: a standalone ``--`` separator after the entrypoint name is NOT
  required and previously failed with ``missing --model``; pass the
  flags directly after ``main_entry``.
* ``--dry-run`` prints the command that would be launched and exits.

Per-family Modal timeouts. The longbench harness can take many hours
on a 7B model with ``--subset deterministic``/``full`` because the
SQ-v2 calibration is CPU-bound and the per-task generation loop runs
50 examples per task at 8 192 input tokens / 128 new tokens. To keep
paper-valid evidence reachable without weakening any gate, the
launcher exposes both a CLI flag (``--timeout-sec``) and per-family
defaults below. Operators may also override at module load time via
environment variables — useful for ``modal run -d`` invocations that
bypass the standalone CLI entry point::

    SPECTRALQUANT_MODAL_TIMEOUT_SEC=21600 \
      modal run -d scripts/launch_modal_eval.py::main_entry --family longbench ...

    # Family-specific override takes precedence over the global one.
    SPECTRALQUANT_MODAL_TIMEOUT_LONGBENCH_SEC=21600 \
      modal run -d scripts/launch_modal_eval.py::main_entry --family longbench ...

A larger Modal ``timeout`` raises only the kill-switch; you are billed
for actual GPU minutes. The cost ceiling on a longbench paper-valid
run is therefore set by the harness itself (n_per_task, max_new_tokens,
n_calib, lloyd_max_iter), NOT by this timeout.

Local invocation::

    python3 scripts/launch_modal_eval.py --dry-run \\
        --family perplexity --model mistralai/Mistral-7B-v0.3 \\
        --avg-bits 3 --methods fp16

Detached invocation (real run)::

    modal run -d scripts/launch_modal_eval.py::main_entry \\
        --family perplexity --model mistralai/Mistral-7B-v0.3 \\
        --avg-bits 3 --methods fp16
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

GIT_COMMIT_ENV = "SPECTRALQUANT_GIT_COMMIT"

DEFAULT_GPU = "H200"
# Per-family Modal kill-switch defaults. Modal bills on actual GPU
# minutes, not on the timeout cap, so these are upper bounds rather
# than reservations. The longbench cap is set high enough to cover a
# real paper-valid LongBench run on a 7B model under SQ-v2 + TurboQuant
# replay (CPU-bound calibration + 21 tasks × 50 examples × per-example
# generate); see docs/execution_audit_and_modal_runbook.md §7.7.4a/b
# for the cost/time reasoning.
FAMILY_TIMEOUT_SEC: Dict[str, int] = {
    "perplexity": 90 * 60,        # 1.5 h
    "generation": 90 * 60,        # 1.5 h
    "latency":   60 * 60,         # 1 h
    "longbench": 6 * 60 * 60,     # 6 h — covers paper-valid Qwen2.5-7B
}
# Backwards-compatible default for callers that don't pass a family
# (e.g. unit tests). Always equals the largest per-family cap so it is
# safe regardless of family.
DEFAULT_TIMEOUT_SEC = max(FAMILY_TIMEOUT_SEC.values())
SMOKE_TIMEOUT_SEC = 15 * 60

# Hard ceiling. Modal's per-function timeout is bounded; we refuse to
# request more than 24 h to avoid silently passing through an
# over-large value. Operators that need >24 h must split the run.
MAX_TIMEOUT_SEC = 24 * 60 * 60

# Environment-variable overrides for the module-level app build
# (``modal run -d scripts/launch_modal_eval.py::main_entry ...``). The
# CLI ``--timeout-sec`` flag also accepts these values; the env vars
# exist for the detached path that does not go through the standalone
# CLI.
ENV_TIMEOUT_GLOBAL = "SPECTRALQUANT_MODAL_TIMEOUT_SEC"
ENV_TIMEOUT_FAMILY_PREFIX = "SPECTRALQUANT_MODAL_TIMEOUT_"  # +<FAMILY>_SEC

DEFAULT_VOLUME_NAME = "spectralquant-v2-results"
DEFAULT_APP_NAME = "spectralquant-v2-eval"
DEFAULT_HF_SECRET = "hf-token"

ALLOWED_GPUS = ("H200", "H100", "B200", "A100-80GB")

#: Mapping family -> harness script under ``experiments/``.
FAMILY_SCRIPT: Dict[str, str] = {
    "perplexity": "run_perplexity.py",
    "longbench": "run_longbench.py",
    "generation": "run_generation.py",
    "latency": "run_latency.py",
}

#: Default output sub-directory under ``/results`` for each family.
FAMILY_OUTPUT_SUBDIR: Dict[str, str] = {
    "perplexity": "perplexity",
    "longbench": "longbench",
    "generation": "generation",
    "latency": "latency",
}


def resolve_timeout_sec(
    family: Optional[str] = None,
    explicit: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Resolve the Modal kill-switch timeout for a given family.

    Precedence (highest first):

    1. ``explicit`` — caller-provided value (e.g. ``--timeout-sec``).
    2. ``$SPECTRALQUANT_MODAL_TIMEOUT_<FAMILY>_SEC`` — family-specific env var.
    3. ``$SPECTRALQUANT_MODAL_TIMEOUT_SEC`` — global env-var override.
    4. ``FAMILY_TIMEOUT_SEC[family]`` — per-family default.
    5. ``DEFAULT_TIMEOUT_SEC`` — last-resort fallback (= largest cap).

    Bounds: clamped into ``[60, MAX_TIMEOUT_SEC]``. A non-positive or
    non-integer value raises ``ValueError`` so a typo in the env var
    cannot silently downgrade the run.
    """
    e = env if env is not None else os.environ

    def _coerce(v: Any, source: str) -> int:
        try:
            iv = int(str(v).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"timeout from {source} is not an integer: {v!r}"
            ) from exc
        if iv <= 0:
            raise ValueError(f"timeout from {source} must be positive: {iv}")
        if iv > MAX_TIMEOUT_SEC:
            raise ValueError(
                f"timeout from {source} = {iv}s exceeds MAX_TIMEOUT_SEC "
                f"= {MAX_TIMEOUT_SEC}s; split the run instead"
            )
        if iv < 60:
            raise ValueError(
                f"timeout from {source} = {iv}s is below 60s; refusing"
            )
        return iv

    if explicit is not None:
        return _coerce(explicit, source="explicit argument")
    if family:
        fam_key = f"{ENV_TIMEOUT_FAMILY_PREFIX}{family.upper()}_SEC"
        if fam_key in e:
            return _coerce(e[fam_key], source=fam_key)
    if ENV_TIMEOUT_GLOBAL in e:
        return _coerce(e[ENV_TIMEOUT_GLOBAL], source=ENV_TIMEOUT_GLOBAL)
    if family and family in FAMILY_TIMEOUT_SEC:
        return FAMILY_TIMEOUT_SEC[family]
    return DEFAULT_TIMEOUT_SEC


def detect_local_git_commit(repo_root: Path = REPO_ROOT) -> Optional[str]:
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


# Reuse the launch_modal_three_way sanitization (importing it would be
# circular if that module fails to import modal; we duplicate the small
# function here for the same reason run_status duplicates it).
SECRET_NAME_PATTERNS: Tuple[str, ...] = (
    "TOKEN", "SECRET", "API_KEY", "APIKEY",
    "PASSWORD", "PASSWD", "PRIVATE_KEY",
)
_TOKEN_LITERAL_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bak-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bas-[A-Za-z0-9_-]{16,}\b"),
)


def sanitize_text(text: str) -> str:
    if not text:
        return text
    redacted = text
    seen: set = set()
    for k in os.environ:
        upper = k.upper()
        if any(p in upper for p in SECRET_NAME_PATTERNS):
            val = os.environ.get(k, "")
            if len(val) >= 8 and val not in seen:
                seen.add(val)
                redacted = redacted.replace(val, f"[REDACTED:{k}]")
    for pat in _TOKEN_LITERAL_PATTERNS:
        redacted = pat.sub("[REDACTED:token-literal]", redacted)
    return redacted


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRunConfig:
    family: str
    model: str
    seed: int = 42
    smoke: bool = False
    inline_corpus_smoke: bool = False
    dry_run: bool = False
    force: bool = False
    output_dir: Optional[str] = None  # default per family
    status_dir: Optional[str] = None
    avg_bits: int = 3
    methods: Tuple[str, ...] = ("fp16",)
    device: str = "cuda"
    dtype: str = "float16"

    # Family-specific kwargs are passed through as a list of CLI tokens.
    extra_args: Tuple[str, ...] = field(default_factory=tuple)


def family_output_dir(family: str) -> str:
    return f"/results/{FAMILY_OUTPUT_SUBDIR[family]}"


def build_command(cfg: EvalRunConfig) -> List[str]:
    if cfg.family not in FAMILY_SCRIPT:
        raise ValueError(
            f"Unknown family {cfg.family!r}; permitted: {sorted(FAMILY_SCRIPT)}"
        )
    if cfg.dry_run and cfg.smoke:
        raise ValueError("Cannot combine smoke=True with dry_run=True")
    if cfg.smoke and cfg.inline_corpus_smoke:
        raise ValueError(
            "Cannot combine smoke=True with inline_corpus_smoke=True"
        )

    script = EXPERIMENTS_DIR / FAMILY_SCRIPT[cfg.family]
    out_dir = cfg.output_dir or family_output_dir(cfg.family)
    cmd: List[str] = [
        sys.executable, str(script),
        "--model", cfg.model,
        "--output-dir", out_dir,
        "--seed", str(cfg.seed),
        "--device", cfg.device,
        "--dtype", cfg.dtype,
        "--avg-bits", str(cfg.avg_bits),
        "--methods", *cfg.methods,
    ]
    if cfg.smoke:
        cmd += ["--synthetic-smoke"]
    if cfg.inline_corpus_smoke:
        cmd += ["--inline-corpus-smoke"]
    if cfg.dry_run:
        cmd += ["--dry-run"]
    if cfg.force:
        cmd += ["--force"]
    cmd += list(cfg.extra_args)
    return cmd


def render_command(cfg: EvalRunConfig) -> str:
    return " ".join(shlex.quote(p) for p in build_command(cfg))


def derive_run_id(cfg: EvalRunConfig) -> str:
    """Compute a short label suitable for the status directory name.

    The harnesses each use their own ``derive_run_id``; this is a
    coarse fallback used only by the launcher's pre-spawn early
    artifact emission. Once the subprocess is running the harness
    overwrites the status directory anyway.
    """
    short = re.sub(
        r"[^A-Za-z0-9._-]+", "_", cfg.model.split("/")[-1]
    ).strip("._-") or "model"
    methods_tag = "+".join(sorted(cfg.methods))
    suffix = f"b{cfg.avg_bits}_seed{cfg.seed}_{methods_tag}"
    rid = f"{cfg.family}__{short}__{suffix}"
    if cfg.smoke:
        rid = "synthetic_smoke__" + rid
    elif cfg.inline_corpus_smoke:
        rid = "inline_corpus_smoke__" + rid
    elif cfg.dry_run:
        rid = "dryrun__" + rid
    return rid


# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------


def build_modal_app(
    *,
    app_name: str = DEFAULT_APP_NAME,
    gpu: str = DEFAULT_GPU,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    volume_name: str = DEFAULT_VOLUME_NAME,
    hf_secret_name: str = DEFAULT_HF_SECRET,
) -> Tuple[Any, Any, Any, Any]:
    if gpu not in ALLOWED_GPUS:
        raise ValueError(f"GPU '{gpu}' not in {ALLOWED_GPUS}")
    if not isinstance(timeout_sec, int) or timeout_sec < 60:
        raise ValueError(
            f"timeout_sec must be an int >= 60, got {timeout_sec!r}"
        )
    if timeout_sec > MAX_TIMEOUT_SEC:
        raise ValueError(
            f"timeout_sec={timeout_sec}s exceeds MAX_TIMEOUT_SEC "
            f"= {MAX_TIMEOUT_SEC}s"
        )
    try:
        import modal  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "modal is not installed locally. `pip install modal` then "
            "`modal token new`. Launcher itself never reads token values."
        ) from exc

    def _ignore_repo_path(p: Path) -> bool:
        s = str(p)
        return (
            ".git/" in s or s.endswith(".git") or "/.git" in s
            or "/results/" in s or "/local_results/" in s
            or "__pycache__" in s
        )

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
            # Required by experiments.longbench_dataset's HF Hub fallback
            # path: newer `datasets` rejects script-based datasets, so the
            # LongBench loader downloads `data.zip` from THUDM/LongBench
            # via huggingface_hub and parses the per-task JSONL directly.
            "huggingface_hub>=0.20",
            "numpy",
            "scipy",
            "jsonschema>=4.18",
            "tqdm",
        )
        .env({"PYTHONPATH": "/repo/scripts:/repo/src:/repo"})
        .add_local_dir(
            str(REPO_ROOT),
            remote_path="/repo",
            ignore=_ignore_repo_path,
        )
    )

    secrets = [modal.Secret.from_name(hf_secret_name)]
    app = modal.App(app_name)
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)

    @app.function(
        image=image,
        gpu=gpu,
        timeout=timeout_sec,
        secrets=secrets,
        volumes={"/results": volume},
        serialized=True,
    )
    def run_one(cfg_dict: Dict[str, Any],
                git_commit: Optional[str] = None) -> Dict[str, Any]:
        cfg = EvalRunConfig(**cfg_dict)
        cmd = build_command(cfg)
        # Replace local script path with /repo equivalent.
        for i, tok in enumerate(cmd):
            if tok == str(EXPERIMENTS_DIR / FAMILY_SCRIPT[cfg.family]):
                cmd[i] = f"/repo/experiments/{FAMILY_SCRIPT[cfg.family]}"

        run_id = derive_run_id(cfg)
        status_parent = (
            cfg.status_dir
            or f"/results/status/{cfg.family}"
        )
        cmd = cmd + ["--status-dir", status_parent]

        env = dict(os.environ)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            "/repo/scripts" + os.pathsep + "/repo/src"
            + (os.pathsep + existing_pp if existing_pp else "")
        )
        if git_commit:
            env[GIT_COMMIT_ENV] = git_commit

        # Persistent HF cache.
        try:
            from experiments.run_status import configure_persistent_hf_cache
            configure_persistent_hf_cache(env, volume_mount="/results")
        except Exception:
            pass

        print(f"[launch_modal_eval] launching family={cfg.family} model={cfg.model}")
        try:
            proc = subprocess.run(
                cmd,
                cwd="/repo",
                env=env,
                capture_output=True,
                text=True,
            )
        except BaseException as exc:
            return {
                "returncode": -1,
                "error": f"{type(exc).__name__}: {exc}",
                "command": " ".join(shlex.quote(p) for p in cmd),
                "family": cfg.family,
                "model": cfg.model,
            }

        # Persist the volume so a polling operator sees the artifact.
        try:
            volume.commit()
        except Exception:
            pass

        return {
            "returncode": int(proc.returncode),
            "stdout_tail": sanitize_text(proc.stdout)[-4000:] if proc.stdout else "",
            "stderr_tail": sanitize_text(proc.stderr)[-4000:] if proc.stderr else "",
            "command": " ".join(shlex.quote(p) for p in cmd),
            "family": cfg.family,
            "model": cfg.model,
            "git_commit": git_commit,
            "status_dir": f"{status_parent}/{run_id}",
        }

    return app, image, volume, run_one


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="launch_modal_eval.py",
        description=(
            "Modal launcher for the four next-stage SpectralQuant v2 "
            "evaluation harnesses. Use --dry-run to print the command "
            "that would be launched."
        ),
    )
    p.add_argument("--family", required=True,
                   choices=sorted(FAMILY_SCRIPT.keys()))
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--avg-bits", type=int, default=3, dest="avg_bits")
    p.add_argument("--methods", nargs="+", default=["fp16"])
    p.add_argument("--device", type=str, default="cuda",
                   choices=("cpu", "cuda"))
    p.add_argument("--dtype", type=str, default="float16",
                   choices=("float32", "float16", "bfloat16"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--inline-corpus-smoke", action="store_true",
                   dest="inline_corpus_smoke")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--force", action="store_true")
    p.add_argument("--output-dir", type=str, default=None,
                   dest="output_dir")
    p.add_argument("--status-dir", type=str, default=None,
                   dest="status_dir")
    p.add_argument("--gpu", type=str, default=DEFAULT_GPU,
                   choices=ALLOWED_GPUS)
    # ``None`` here means "resolve from family default + env vars at
    # build_modal_app time". Operators may pass a plain integer (seconds)
    # to pin the kill-switch explicitly.
    p.add_argument("--timeout-sec", type=int, default=None,
                   dest="timeout_sec",
                   help=(
                       "Modal kill-switch timeout in seconds. Default "
                       "is the per-family value in FAMILY_TIMEOUT_SEC "
                       "(longbench=21600s, others=5400s/3600s); env "
                       "vars SPECTRALQUANT_MODAL_TIMEOUT_<FAMILY>_SEC "
                       "and SPECTRALQUANT_MODAL_TIMEOUT_SEC override "
                       "the default but lose to this flag. Capped at "
                       "MAX_TIMEOUT_SEC=86400s."
                   ))
    p.add_argument("--volume-name", type=str, default=DEFAULT_VOLUME_NAME,
                   dest="volume_name")
    p.add_argument("--app-name", type=str, default=DEFAULT_APP_NAME,
                   dest="app_name")
    p.add_argument("--hf-secret-name", type=str, default=DEFAULT_HF_SECRET,
                   dest="hf_secret_name")
    p.add_argument("--print-modal-app", action="store_true",
                   dest="print_modal_app")
    # Anything after ``--`` is passed verbatim to the harness.
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Pass-through args after --extra are forwarded to "
                        "the harness verbatim.")
    return p


def cfg_from_args(ns: argparse.Namespace) -> EvalRunConfig:
    extra = tuple(ns.extra or ())
    if extra and extra[0] == "--":
        extra = extra[1:]
    return EvalRunConfig(
        family=ns.family,
        model=ns.model,
        seed=int(ns.seed),
        smoke=bool(ns.smoke),
        inline_corpus_smoke=bool(ns.inline_corpus_smoke),
        dry_run=False,  # the launch is full; --dry-run is a print-only
        force=bool(ns.force),
        output_dir=ns.output_dir,
        status_dir=ns.status_dir,
        avg_bits=int(ns.avg_bits),
        methods=tuple(ns.methods),
        device=str(ns.device),
        dtype=str(ns.dtype),
        extra_args=extra,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    cfg = cfg_from_args(ns)

    if ns.dry_run:
        commit = detect_local_git_commit(REPO_ROOT)
        cfg = EvalRunConfig(**{**cfg.__dict__, "dry_run": True})
        print(f"[launch_modal_eval] dry-run: family={cfg.family}")
        if commit:
            print(f"  local git commit (forwarded as {GIT_COMMIT_ENV}): {commit}")
        print(f"  cmd: {render_command(cfg)}")
        return 0

    timeout_sec = resolve_timeout_sec(
        family=cfg.family,
        explicit=int(ns.timeout_sec) if ns.timeout_sec is not None else None,
    )

    if ns.print_modal_app:
        try:
            app, _img, _vol, _run_one = build_modal_app(
                app_name=ns.app_name, gpu=ns.gpu,
                timeout_sec=timeout_sec,
                volume_name=ns.volume_name,
                hf_secret_name=ns.hf_secret_name,
            )
        except RuntimeError as exc:
            print(f"[launch_modal_eval] {exc}", file=sys.stderr)
            return 2
        print(f"[launch_modal_eval] app: {ns.app_name}")
        print(f"  family: {cfg.family}")
        print(f"  gpu: {ns.gpu}")
        print(f"  timeout_sec: {timeout_sec}")
        print(f"  volume: {ns.volume_name}")
        return 0

    # Real launch: defer to ``modal run``. Most operators will use
    # ``modal run -d`` against ``main_entry`` instead so the run survives
    # disconnects.
    try:
        app, _img, _vol, run_one = build_modal_app(
            app_name=ns.app_name, gpu=ns.gpu,
            timeout_sec=timeout_sec,
            volume_name=ns.volume_name,
            hf_secret_name=ns.hf_secret_name,
        )
    except RuntimeError as exc:
        print(f"[launch_modal_eval] {exc}", file=sys.stderr)
        return 2

    git_commit = detect_local_git_commit(REPO_ROOT)
    print(f"[launch_modal_eval] launching family={cfg.family} model={cfg.model}")
    from dataclasses import asdict
    with app.run():
        result = run_one.remote(asdict(cfg), git_commit=git_commit)
    rc = int(result.get("returncode", -1))
    print(f"[launch_modal_eval] returncode={rc}")
    print(f"[launch_modal_eval] status_dir: {result.get('status_dir')}")
    if rc != 0:
        print("--- stderr_tail ---")
        print(result.get("stderr_tail", "")[-2000:])
    return rc


# ---------------------------------------------------------------------------
# Module-scope app for ``modal run``
# ---------------------------------------------------------------------------

app: Any = None
run_one: Any = None
main_entry: Any = None
volume: Any = None

try:
    import modal as _modal_for_module  # noqa: F401
    # The module-level Modal app build runs once at import time and
    # backs the ``modal run -d ... main_entry`` detached path. There is
    # no ``--family`` available yet, so we default to the largest
    # per-family cap (longbench) so a longbench detached relaunch is
    # not killed at 90 min. ``$SPECTRALQUANT_MODAL_TIMEOUT_SEC`` and
    # the family-specific env vars still override.
    _module_timeout = resolve_timeout_sec(
        family="longbench",
        explicit=None,
    )
    app, _image, volume, run_one = build_modal_app(timeout_sec=_module_timeout)

    @app.local_entrypoint()
    def main_entry(
        family: str,
        model: str,
        seed: int = 42,
        avg_bits: int = 3,
        methods: str = "fp16",
        device: str = "cuda",
        dtype: str = "float16",
        smoke: bool = False,
        inline_corpus_smoke: bool = False,
        force: bool = False,
        output_dir: str = "",
        status_dir: str = "",
        extra: str = "",
    ) -> None:
        """Modal local entrypoint — detached-friendly invocation.

        ``methods`` is a comma-separated string (Modal's CLI parser does
        not handle list-of-string cleanly). Same for ``extra`` which is
        a single shell-escaped string forwarded verbatim to the harness.
        """
        method_tup = tuple(m.strip() for m in methods.split(",") if m.strip())
        extra_tup = tuple(shlex.split(extra)) if extra else ()
        cfg = EvalRunConfig(
            family=family,
            model=model,
            seed=int(seed),
            avg_bits=int(avg_bits),
            methods=method_tup,
            device=device,
            dtype=dtype,
            smoke=bool(smoke),
            inline_corpus_smoke=bool(inline_corpus_smoke),
            force=bool(force),
            output_dir=(output_dir or None),
            status_dir=(status_dir or None),
            extra_args=extra_tup,
        )
        from dataclasses import asdict
        git_commit = detect_local_git_commit(REPO_ROOT)
        if git_commit:
            print(f"[launch_modal_eval] forwarding commit {git_commit[:12]}")
        print(
            f"[launch_modal_eval] launching family={cfg.family} "
            f"model={cfg.model} smoke={cfg.smoke} "
            f"inline_corpus_smoke={cfg.inline_corpus_smoke}"
        )
        # The Modal kill-switch was set at module import time; surface
        # it (plus the resolved per-family default) so the operator can
        # see whether a longbench launch will survive its expected
        # runtime. This is a log only — main_entry cannot rebuild
        # ``run_one`` per call.
        try:
            family_default = resolve_timeout_sec(family=cfg.family)
        except ValueError:
            family_default = -1
        print(
            f"[launch_modal_eval] modal timeout (kill-switch) = "
            f"{_module_timeout}s; family default for {cfg.family} = "
            f"{family_default}s. Override at module load via "
            f"SPECTRALQUANT_MODAL_TIMEOUT_{cfg.family.upper()}_SEC or "
            f"SPECTRALQUANT_MODAL_TIMEOUT_SEC."
        )
        call = run_one.spawn(asdict(cfg), git_commit=git_commit)
        call_id = getattr(call, "object_id", None) or str(call)
        print(f"[launch_modal_eval] spawned remote call: call_id={call_id}")
        print(
            f"[launch_modal_eval] poll status: modal volume ls "
            f"{DEFAULT_VOLUME_NAME} /results/status/{cfg.family}/"
        )
        print(
            f"[launch_modal_eval] result: python3 -c "
            f"\"import modal; print(modal.FunctionCall.from_id('{call_id}').get(timeout=0))\""
        )
except ImportError:
    pass
except Exception as _exc:  # pragma: no cover - defensive
    print(
        f"[launch_modal_eval] WARNING: module-level Modal app build "
        f"failed: {type(_exc).__name__}: {_exc}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
