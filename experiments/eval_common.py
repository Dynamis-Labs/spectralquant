"""Shared utilities for the v2 evaluation harnesses.

The four next-stage evidence families (perplexity, LongBench, real
generation quality, end-to-end latency) share the same operational
discipline established by ``experiments/run_three_way.py``:

* ``--dry-run`` / ``--synthetic-smoke`` / ``--inline-corpus-smoke``
  modes that need *no* heavy model download.
* Atomic JSON writes with schema validation before the rename.
* Status artifact emission via :class:`experiments.run_status.StatusWriter`.
* Provenance metadata: repo slug, git commit (with ``SPECTRALQUANT_GIT_COMMIT``
  override for Modal containers), software/hardware blocks, sanitized
  command string.
* Token-name redaction in logs and stdout/stderr tails.

This module factors that common surface into one place so the family
harnesses below stay short and testable. None of these helpers import
torch / transformers / datasets; lazy imports stay inside the harness
that needs them.

There are NO secret reads in this module. Token env-var **names** are
referenced for redaction only.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EVAL_COMMON_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_COMMON_DIR.parent
SRC_DIR = REPO_ROOT / "src"
SCHEMAS_DIR = REPO_ROOT / "schemas"

REPO_SLUG = "niashwin/spectralquant-full"
GIT_COMMIT_ENV = "SPECTRALQUANT_GIT_COMMIT"

# Schema version this module emits. Bump when payload shape changes in a
# backward-incompatible way.
SCHEMA_VERSION = "1"

#: Modes a harness can execute in. ``full`` is the only mode that produces
#: paper-valid evidence; everything else is harness-validation only.
MODE_FULL = "full"
MODE_SYNTHETIC_SMOKE = "synthetic_smoke"
MODE_INLINE_CORPUS_SMOKE = "inline_corpus_smoke"
MODE_DRY_RUN = "dry_run"

ALL_MODES = (MODE_FULL, MODE_SYNTHETIC_SMOKE, MODE_INLINE_CORPUS_SMOKE, MODE_DRY_RUN)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_SHA_RE = re.compile(r"^[0-9a-fA-F]+$")


def git_commit(
    repo_root: Path = REPO_ROOT,
    override: Optional[str] = None,
) -> str:
    """Return the git SHA to record in result metadata.

    Resolution order matches ``experiments/run_three_way.py::_git_commit``:
    explicit ``override``, ``SPECTRALQUANT_GIT_COMMIT`` env var, then
    ``git rev-parse HEAD``. Returns ``"0000000"`` if nothing usable is
    available.
    """
    candidates: List[str] = []
    if override:
        candidates.append(override)
    env_val = os.environ.get(GIT_COMMIT_ENV, "")
    if env_val:
        candidates.append(env_val)
    for sha in candidates:
        sha = sha.strip()
        if len(sha) >= 7 and _SHA_RE.fullmatch(sha):
            return sha
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        sha = out.decode("utf-8").strip()
        if len(sha) >= 7 and _SHA_RE.fullmatch(sha):
            return sha
    except Exception:
        pass
    return "0000000"


def safe_command_string(argv: Sequence[str]) -> str:
    """Reconstruct argv into a single-line command string.

    No env values are inlined — only the argv is included so accidental
    leakage of token values cannot happen here. Mirrors
    ``run_three_way._safe_command_string``.
    """
    return " ".join(shlex.quote(a) for a in argv)


def model_short(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-") or "model"


# ---------------------------------------------------------------------------
# Hardware / software metadata
# ---------------------------------------------------------------------------


def build_software_block(extra: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    pyver = ".".join(str(x) for x in sys.version_info[:3])

    def _try_version(mod: str) -> str:
        try:
            return __import__(mod).__version__  # type: ignore[attr-defined]
        except Exception:
            return "unavailable"

    block: Dict[str, Any] = {
        "python": pyver,
        "torch": _try_version("torch"),
        "transformers": _try_version("transformers"),
        "datasets": _try_version("datasets"),
    }
    if extra:
        block.update(extra)
    return block


def build_hardware_block(device: str = "cpu") -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "gpu": "none" if device == "cpu" else "unknown",
        "cuda": "n/a",
        "host": socket.gethostname(),
        "platform": platform.platform(),
    }
    if device == "cuda":
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                block["gpu"] = torch.cuda.get_device_name(0)
                block["cuda"] = (
                    torch.version.cuda or "unknown"  # type: ignore[attr-defined]
                )
                try:
                    props = torch.cuda.get_device_properties(0)
                    block["gpu_total_memory_mb"] = round(
                        props.total_memory / (1024 * 1024), 2
                    )
                except Exception:
                    pass
        except Exception:
            pass
    return block


# ---------------------------------------------------------------------------
# Atomic write + schema validation (shared with run_three_way pattern)
# ---------------------------------------------------------------------------


def _validate_payload(payload: Dict[str, Any], schema_path: Path) -> None:
    try:
        import jsonschema  # type: ignore[import-not-found]
        from jsonschema import Draft202012Validator
    except ImportError:
        return

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload),
                    key=lambda e: list(e.absolute_path))
    if errors:
        msg = "\n".join(
            f"  - at {list(e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        )
        raise ValueError(f"payload does not validate against {schema_path.name}:\n{msg}")


def atomic_write_json(
    path: Path,
    payload: Dict[str, Any],
    schema_path: Optional[Path] = None,
) -> None:
    """Atomically write ``payload`` to ``path``.

    1. Validate against schema (if jsonschema importable).
    2. Write to a tempfile in same dir, fsync, ``os.replace``.

    A half-written or schema-invalid file never appears at the canonical
    path. Mirrors ``experiments/run_three_way.atomic_write_json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if schema_path is not None and schema_path.exists():
        _validate_payload(payload, schema_path)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Output path / run id
# ---------------------------------------------------------------------------


def derive_run_id(
    family: str,
    model: str,
    *,
    suffix: str,
    mode: str,
) -> str:
    """Build a deterministic ``run_id`` for an eval-family run.

    ``suffix`` is a family-specific tail (e.g. ``b3_seed42_wikitext-103``)
    so two configs of the same family on the same model never collide.
    Smoke / dry-run modes get a stable prefix so they never overwrite
    full-mode artifacts.
    """
    short = model_short(model)
    rid = f"{family}__{short}__{suffix}"
    if mode == MODE_SYNTHETIC_SMOKE:
        rid = "synthetic_smoke__" + rid
    elif mode == MODE_INLINE_CORPUS_SMOKE:
        rid = "inline_corpus_smoke__" + rid
    elif mode == MODE_DRY_RUN:
        rid = "dryrun__" + rid
    return rid


def output_path(output_dir: Path, run_id: str) -> Path:
    return Path(output_dir) / f"{run_id}.json"


# ---------------------------------------------------------------------------
# Method label discipline
# ---------------------------------------------------------------------------

#: Method keys used across all four eval families. Methods present in any
#: given run depend on what the harness can produce; we only require >= 1.
KNOWN_METHOD_KEYS: Tuple[str, ...] = (
    "fp16",
    "turboquant",
    "spectralquant_v1",
    "spectralquant_v2",
    "official_turboquant",
)


def assert_method_keys(methods: Dict[str, Any]) -> None:
    """Light sanity check: every method key is recognized.

    Unknown keys are a hint that the harness wrote something downstream
    code won't expect; we surface this as a hard error to keep the
    catalog discipline tight.
    """
    unknown = sorted(set(methods.keys()) - set(KNOWN_METHOD_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown method keys {unknown}; permitted: {KNOWN_METHOD_KEYS}"
        )


# ---------------------------------------------------------------------------
# Inline corpus (matches run_three_way's INLINE_SMOKE_CORPUS spirit)
# ---------------------------------------------------------------------------

INLINE_CORPUS: Tuple[str, ...] = (
    "Quantization research aims to reduce the memory footprint of large "
    "language models without significantly degrading downstream task quality. "
    "Common approaches include weight quantization, key-value cache "
    "compression, and post-training calibration on a small held-out corpus.",
    "Eigenspectral calibration estimates a per-head rotation that aligns the "
    "key-value subspace with the dominant eigendirections of the empirical "
    "covariance matrix observed during a forward pass over short text "
    "samples drawn from the calibration corpus.",
    "Water-filling allocates a non-uniform per-dimension bit budget so that "
    "dimensions carrying more signal receive more bits. This typically "
    "improves attention-output cosine similarity at low average bitrates "
    "compared to uniform semantic quantizers.",
    "The three-way comparison records cosine similarity between FP16 "
    "reference attention weights and the weights produced by TurboQuant, "
    "SpectralQuant v1, and SpectralQuant v2 across a sampled set of layers.",
    "Rotary position embeddings interleave sine and cosine components into "
    "the query and key projections; pre-RoPE quantization keeps the rotation "
    "math identical between the reference path and the quantized path.",
    "Calibration tokens are clipped to a configurable cap so that very long "
    "examples do not dominate the eigenspectrum estimate. Default caps in "
    "the spec sit between 256 and 512 tokens per layer per sample.",
    "Group-query attention factors the key-value heads through a smaller "
    "set of projections shared across query-head groups, which reduces the "
    "memory bandwidth required to serve long context windows in production.",
    "Synthetic smoke runs verify the result-plumbing end to end on small "
    "Q/K/V tensors. Inline-corpus smoke runs additionally exercise the full "
    "HuggingFace model load and hook path without downloading any dataset.",
)


def build_inline_corpus(n_total: int) -> List[str]:
    """Return ``n_total`` deterministic strings; cycles INLINE_CORPUS."""
    if n_total < 1:
        raise ValueError(f"n_total must be >= 1, got {n_total}")
    base = list(INLINE_CORPUS)
    out: List[str] = []
    for i in range(n_total):
        text = base[i % len(base)]
        out.append(f"{text} [inline_smoke#{i:03d}]")
    return out


# ---------------------------------------------------------------------------
# Deterministic prompt set for run_generation.py synthetic / inline modes
# ---------------------------------------------------------------------------

DEFAULT_GENERATION_PROMPTS: Tuple[Dict[str, str], ...] = (
    {
        "id": "summary-001",
        "category": "summarization",
        "prompt": (
            "Summarize the following passage in two sentences. "
            "Passage: SpectralQuant exploits a structural property of the "
            "key-value cache in transformer attention to compress without "
            "destroying downstream attention-output quality."
        ),
    },
    {
        "id": "qa-001",
        "category": "qa",
        "prompt": (
            "Question: What is the main motivation for non-uniform per-dimension "
            "bit allocation in KV cache compression? Answer briefly."
        ),
    },
    {
        "id": "code-001",
        "category": "code",
        "prompt": (
            "Write a short Python function that computes the participation "
            "ratio of a vector of eigenvalues. The function should return "
            "(sum(eig)**2) / sum(eig**2)."
        ),
    },
    {
        "id": "complete-001",
        "category": "completion",
        "prompt": (
            "The participation ratio of a spectrum is defined as"
        ),
    },
    {
        "id": "instruct-001",
        "category": "instruction",
        "prompt": (
            "List three reasons why low-rank approximation typically works "
            "for keys but not for values in transformer attention."
        ),
    },
    {
        "id": "translate-001",
        "category": "translation",
        "prompt": (
            "Translate to French: 'The calibration step takes a few seconds "
            "and is performed once per model.'"
        ),
    },
    {
        "id": "longctx-001",
        "category": "long_context",
        "prompt": (
            "Given the context that water-filling assigns more bits to "
            "dimensions with larger eigenvalues, explain in one paragraph "
            "why this improves attention quality at low average bitrates."
        ),
    },
    {
        "id": "factual-001",
        "category": "factual",
        "prompt": (
            "What does the acronym KV cache stand for in the context of "
            "transformer inference?"
        ),
    },
)


# ---------------------------------------------------------------------------
# Token-name redaction (mirrors launcher / run_status sanitize_text)
# ---------------------------------------------------------------------------

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


def sanitize_text(text: str, *, extra_env: Optional[Iterable[str]] = None) -> str:
    if not text:
        return text
    redacted = text
    seen: set = set()
    env_names: List[str] = list(extra_env or ())
    for k in os.environ:
        upper = k.upper()
        if any(p in upper for p in SECRET_NAME_PATTERNS):
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
# Common payload skeleton
# ---------------------------------------------------------------------------


def base_payload(
    *,
    family: str,
    run_id: str,
    argv: Sequence[str],
    model_name: str,
    mode: str,
    paper_valid: bool,
    device: str = "cpu",
    extra_software: Optional[Dict[str, str]] = None,
    git_commit_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the common top-level fields for an eval-family payload.

    Family-specific fields (``data``, ``decoding``, ``operating_points``,
    ``timing``, ``methods``, ``evidence_ids``, ``caveats``) are added by
    the caller.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "repo": REPO_SLUG,
        "commit": git_commit(override=git_commit_override),
        "command": safe_command_string(argv),
        "mode": mode,
        "paper_valid": bool(paper_valid),
        "model": {"name": model_name},
        "hardware": build_hardware_block(device),
        "software": build_software_block(extra_software),
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def add_common_eval_args(parser: Any) -> None:
    """Add the ``--dry-run / --synthetic-smoke / --inline-corpus-smoke``
    triad and friends to an argparse parser. Mirrors ``run_three_way``.
    """
    parser.add_argument("--model", required=True, type=str,
                        help="HuggingFace model id, e.g. mistralai/Mistral-7B-v0.3")
    parser.add_argument("--output-dir", required=True, type=str,
                        dest="output_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=("cpu", "cuda"))
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--synthetic-smoke", action="store_true",
                        dest="synthetic_smoke")
    parser.add_argument("--inline-corpus-smoke", action="store_true",
                        dest="inline_corpus_smoke")
    parser.add_argument("--force", action="store_true", dest="force")
    parser.add_argument("--skip-if-exists", action="store_true", default=True,
                        dest="skip_if_exists")
    parser.add_argument("--no-skip-if-exists", action="store_false",
                        dest="skip_if_exists")
    parser.add_argument("--git-commit", type=str, default=None,
                        dest="git_commit_override")
    parser.add_argument("--status-dir", type=str, default=None,
                        dest="status_dir")


def resolve_mode(ns: Any) -> str:
    """Translate the parsed namespace's flag triad into a single mode."""
    if getattr(ns, "dry_run", False):
        return MODE_DRY_RUN
    if getattr(ns, "synthetic_smoke", False) and getattr(ns, "inline_corpus_smoke", False):
        raise SystemExit(
            "--synthetic-smoke and --inline-corpus-smoke are mutually exclusive"
        )
    if getattr(ns, "synthetic_smoke", False):
        return MODE_SYNTHETIC_SMOKE
    if getattr(ns, "inline_corpus_smoke", False):
        return MODE_INLINE_CORPUS_SMOKE
    return MODE_FULL


def install_repo_paths_into_sys_path() -> None:
    """Make ``import spectralquant`` and ``from experiments import ...``
    work without ``pip install -e .``. Mirrors run_three_way's bootstrap.
    """
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
