#!/usr/bin/env python3
"""Three-way SpectralQuant v2 benchmark harness (spec §13.5).

This script runs the three-way attention-output cosine comparison between
``TurboQuant`` (local baseline), ``SpectralQuant v1`` (uniform semantic
allocation), and ``SpectralQuant v2`` (water-filled semantic allocation)
for one ``(model, avg_bits, seed)`` triplet.

The full HuggingFace path is *not* exercised by this slice. Loading 7B
weights requires Modal credit and gated-license access, so the real-model
arm raises ``NotImplementedError`` and is gated behind explicit flags.
The harness still:

* validates CLI arguments;
* writes deterministic per-run JSON file paths under ``--output-dir``;
* supports ``--dry-run`` (print plan, no compute, no JSON write);
* supports ``--synthetic-smoke`` (run the full three-way comparison on
  small synthetic Q/K/V tensors so the result-plumbing is exercised end
  to end without downloading any model);
* writes results atomically (tempfile in same dir, validated against
  ``schemas/three_way_result.schema.json`` if jsonschema is available,
  then ``os.replace``);
* skips existing output unless ``--force`` is passed;
* never echoes secret values (HF tokens, Modal tokens) — they are not
  read or printed by this script.

CLI matches ``docs/spectralquant_v2_technical_spec.md`` §13.5 plus the
``docs/modal_safety_protocol.md`` resumability/atomicity contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path setup — make ``import spectralquant`` work when run from a checkout
# without ``pip install -e .``.
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Make ``from experiments import ...`` work regardless of cwd, pytest
# rootdir injection, or whether ``experiments`` has an ``__init__.py``.
# Without this, ``importlib.util.spec_from_file_location`` loaders (used by
# tests) fail at ``from experiments import model_adapters`` because the
# repo root is not on ``sys.path``.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Layer-sample lists from spec §13.4.
LAYER_SAMPLE_LISTS: Dict[str, List[int]] = {
    "mistralai/Mistral-7B-v0.3": [0, 4, 8, 12, 16, 20, 24, 28],
    "Qwen/Qwen2.5-7B": [0, 3, 6, 9, 12, 15, 18, 21],
}

#: Static model-architecture metadata for the spec models. Used to populate
#: the ``model`` block in dry-run / synthetic-smoke without downloading
#: anything; full-model path overwrites with values from
#: ``transformers.AutoConfig``.
MODEL_ARCH_META: Dict[str, Dict[str, int]] = {
    "mistralai/Mistral-7B-v0.3": {
        "layers": 32, "q_heads": 32, "kv_heads": 8,
        "head_dim": 128, "gqa_ratio": 4,
    },
    "Qwen/Qwen2.5-7B": {
        "layers": 28, "q_heads": 28, "kv_heads": 4,
        "head_dim": 128, "gqa_ratio": 7,
    },
}

#: Names of environment variables that may contain credentials. We only
#: read whether they are *set*, never their values.
SECRET_ENV_VARS: Tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "WANDB_API_KEY",
)

REPO_SLUG = "niashwin/spectralquant-full"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: Env var the launcher uses to forward the *local* repo commit into a
#: remote container that does not have ``.git`` mounted (Modal). The
#: harness prefers this over ``git rev-parse`` so result JSONs carry the
#: real provenance even when the remote tree was attached without git.
GIT_COMMIT_ENV = "SPECTRALQUANT_GIT_COMMIT"


def _git_commit(
    repo_root: Path = REPO_ROOT,
    override: Optional[str] = None,
) -> str:
    """Return the git SHA to record in result metadata.

    Resolution order:

    1. Explicit ``override`` argument (CLI ``--git-commit``).
    2. ``SPECTRALQUANT_GIT_COMMIT`` env var (set by the Modal launcher
       so the remote container records the local commit even though
       ``.git`` is excluded from the image).
    3. ``git rev-parse HEAD`` from ``repo_root``.
    4. ``"0000000"`` placeholder.

    Values are accepted only when they look like a hex SHA of length
    >= 7 so a stray empty/garbage value cannot pollute provenance.
    """
    candidates: List[str] = []
    if override:
        candidates.append(override)
    env_val = os.environ.get(GIT_COMMIT_ENV, "")
    if env_val:
        candidates.append(env_val)
    for sha in candidates:
        sha = sha.strip()
        if len(sha) >= 7 and re.fullmatch(r"[0-9a-fA-F]+", sha):
            return sha
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
    return "0000000"


def _safe_command_string(argv: Sequence[str]) -> str:
    """Reconstruct argv into a single-line command string.

    The script never reads secret env values, so this just shell-escapes
    the argv. We deliberately do NOT include any environment variable
    values to avoid accidental leakage (per modal_safety_protocol §1.2).
    """
    safe: List[str] = []
    for arg in argv:
        if re.fullmatch(r"[A-Za-z0-9_./:=@\\-]+", arg or ""):
            safe.append(arg)
        else:
            safe.append("'" + arg.replace("'", "'\\''") + "'")
    return " ".join(safe)


def _model_short(model_name: str) -> str:
    """File-safe short name from a HF model id."""
    base = model_name.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-") or "model"


def _output_path(args: "Args") -> Path:
    """Deterministic per-run JSON path (modal_safety_protocol §4)."""
    short = _model_short(args.model)
    fname = (
        f"{short}_b{args.avg_bits}_calib{args.n_calib}_"
        f"eval{args.n_eval}_seed{args.seed}.json"
    )
    if args.synthetic_smoke and not args.force_real_filename:
        fname = "synthetic_smoke__" + fname
    if args.inline_corpus_smoke and not args.force_real_filename:
        fname = "inline_corpus_smoke__" + fname
    if args.dry_run:
        # Dry-run never writes; surface a deterministic path anyway so
        # logs and tests can compare against it.
        fname = "dryrun__" + fname
    return Path(args.output_dir) / fname


def _validate_payload(
    payload: Dict[str, Any],
    schema_path: Path,
) -> None:
    """Validate payload against the three-way schema, resolving the
    cross-referenced accounting schema from the local schemas/ directory.

    Raises if jsonschema reports any errors. Returns silently otherwise.
    Skipped (no error) if jsonschema is not installed.
    """
    try:
        import jsonschema  # type: ignore[import-not-found]
        from jsonschema import Draft202012Validator
    except ImportError:
        return

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schemas_dir = schema_path.parent
    accounting = json.loads(
        (schemas_dir / "accounting.schema.json").read_text(encoding="utf-8")
    )
    # jsonschema 4.18+ uses Registry; older uses RefResolver. Try both.
    try:
        from referencing import Registry, Resource  # type: ignore[import-not-found]
        from referencing.jsonschema import DRAFT202012  # type: ignore[import-not-found]

        registry = Registry().with_resources([
            (accounting["$id"], Resource(contents=accounting,
                                         specification=DRAFT202012)),
            (schema["$id"], Resource(contents=schema,
                                     specification=DRAFT202012)),
        ])
        validator = Draft202012Validator(schema, registry=registry)
    except ImportError:
        resolver = jsonschema.RefResolver(
            base_uri=schema["$id"],
            referrer=schema,
            store={
                schema["$id"]: schema,
                accounting["$id"]: accounting,
            },
        )
        validator = Draft202012Validator(schema, resolver=resolver)

    errors = sorted(validator.iter_errors(payload),
                    key=lambda e: list(e.absolute_path))
    if errors:
        msg = "\n".join(
            f"  - at {list(e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        )
        raise ValueError(f"payload does not validate:\n{msg}")


def atomic_write_json(
    path: Path,
    payload: Dict[str, Any],
    schema_path: Optional[Path] = None,
) -> None:
    """Atomically write ``payload`` to ``path`` (modal_safety_protocol §5).

    1. Validate ``payload`` against ``schema_path`` (if provided and
       ``jsonschema`` is importable). Validation must succeed before the
       rename; a half-written or schema-invalid file must never appear at
       the canonical path.
    2. Write to a tempfile in the same directory.
    3. fsync and ``os.replace`` for POSIX atomicity.
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
            os.unlink(tmp_name)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


@dataclass
class Args:
    model: str
    avg_bits: int
    n_calib: int
    n_eval: int
    n_layers_sample: int
    output_dir: Path
    seed: int
    device: str
    dtype: str
    max_calib_tokens: int
    dry_run: bool
    synthetic_smoke: bool
    inline_corpus_smoke: bool
    skip_if_exists: bool
    force: bool
    force_real_filename: bool
    qjl_projections: int
    wf_min_bits: int
    wf_max_bits: Optional[int]
    layer_sample: List[int]
    calibration_dir: Optional[Path]
    save_calibration: bool
    load_calibration: bool
    dataset_name: str
    dataset_config: str
    dataset_split: str
    eval_query_tokens: int
    git_commit_override: Optional[str]
    status_dir: Optional[Path]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_three_way.py",
        description=(
            "SpectralQuant v2 three-way benchmark harness "
            "(TurboQuant vs SpectralQuant v1 vs SpectralQuant v2). "
            "Supports --dry-run and --synthetic-smoke for safe local use."
        ),
    )

    p.add_argument("--model", required=True, type=str,
                   help="HuggingFace model id, e.g. mistralai/Mistral-7B-v0.3")
    p.add_argument("--avg-bits", required=True, type=int, dest="avg_bits",
                   choices=range(1, 17), metavar="{1..16}",
                   help="Average bits per element for the K/V quantizer")
    p.add_argument("--n-calib", required=True, type=int, dest="n_calib",
                   help="Number of calibration sequences")
    p.add_argument("--n-eval", required=True, type=int, dest="n_eval",
                   help="Number of evaluation sequences (disjoint from calib)")
    p.add_argument("--n-layers-sample", required=True, type=int,
                   dest="n_layers_sample",
                   help="Number of transformer layers to sample for cosine eval")
    p.add_argument("--output-dir", required=True, type=str, dest="output_dir",
                   help="Directory under which the per-run JSON will be written")

    p.add_argument("--seed", type=int, default=42,
                   help="Master seed for calibration and quantizer init")
    p.add_argument("--device", type=str, default="cpu",
                   choices=("cpu", "cuda"),
                   help="Compute device. CPU is the only safe default for "
                        "synthetic-smoke; the full HF path requires CUDA.")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=("float32", "float16", "bfloat16"),
                   help="Tensor dtype for compute and storage")
    p.add_argument("--max-calib-tokens", type=int, default=384,
                   dest="max_calib_tokens",
                   help="Per-sequence calibration token cap (spec §13.3)")

    # Mode flags.
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Validate args, print plan + output path, write no JSON")
    p.add_argument("--synthetic-smoke", action="store_true",
                   dest="synthetic_smoke",
                   help="Run on small synthetic Q/K/V tensors. No model download.")
    # ``--smoke`` is the v1 alias used by docs/runbook §6.8.
    p.add_argument("--smoke", action="store_true", dest="smoke_alias",
                   help="Alias for --synthetic-smoke (kept for runbook compat)")
    # Inline-corpus harness-validation smoke. Exercises the full HF model
    # path (model load, adapter discovery, hooks, calibration, quantization,
    # eval) but uses a tiny deterministic in-memory corpus instead of
    # HuggingFace ``datasets.load_dataset`` so the run is hermetic and not
    # blocked by dataset downloads. Results from this mode are NOT
    # paper-valid evidence (calibration_corpus="inline_smoke" in the
    # output JSON).
    p.add_argument("--inline-corpus-smoke", action="store_true",
                   dest="inline_corpus_smoke",
                   help=("Use a deterministic inline text corpus instead of "
                         "HF datasets.load_dataset. Full HF model path "
                         "(model load, hooks, calibration, eval) still runs. "
                         "Harness validation only — not paper-valid evidence."))

    # Resumability flags (modal_safety_protocol §4).
    p.add_argument("--skip-if-exists", action="store_true",
                   default=True, dest="skip_if_exists",
                   help="Skip the run if the deterministic output JSON exists")
    p.add_argument("--no-skip-if-exists", action="store_false",
                   dest="skip_if_exists",
                   help="Disable skip-if-exists (useful with --force)")
    p.add_argument("--resume", action="store_true", dest="resume_alias",
                   help="Alias for --skip-if-exists")
    p.add_argument("--force", action="store_true", dest="force",
                   help="Overwrite existing output JSON")
    p.add_argument("--force-real-filename", action="store_true",
                   dest="force_real_filename",
                   help="In --synthetic-smoke, write to the real (non-prefixed) "
                        "filename. Off by default to keep smoke runs out of "
                        "results/three_way/.")

    # v2 engine-config knobs.
    p.add_argument("--qjl-projections", type=int, default=64,
                   dest="qjl_projections",
                   help="QJL projection dimension")
    p.add_argument("--wf-min-bits", type=int, default=0, dest="wf_min_bits",
                   help="Lower bound on per-semantic-dim bits (v2)")
    p.add_argument("--wf-max-bits", type=int, default=None, dest="wf_max_bits",
                   help="Upper bound on per-semantic-dim bits (v2)")

    # Full-HF path knobs (consumed only when not in dry-run / synthetic-smoke).
    p.add_argument("--calibration-dir", type=str, default=None,
                   dest="calibration_dir",
                   help="Directory under which per-(model,seed,n_calib) "
                        "calibration .pt artifacts live. Used by --save-calibration "
                        "and --load-calibration in the full HF path.")
    p.add_argument("--save-calibration", action="store_true",
                   dest="save_calibration",
                   help="In the full HF path, save the calibration artifact "
                        "to --calibration-dir before fitting quantizers.")
    p.add_argument("--load-calibration", action="store_true",
                   dest="load_calibration",
                   help="In the full HF path, load a previously-saved "
                        "calibration artifact from --calibration-dir instead "
                        "of recomputing it.")
    p.add_argument("--dataset-name", type=str, default="wikitext",
                   dest="dataset_name",
                   help="HF datasets name for calibration + eval. "
                        "Default: wikitext.")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1",
                   dest="dataset_config",
                   help="HF datasets config. Default: wikitext-103-raw-v1.")
    p.add_argument("--dataset-split", type=str, default="train",
                   dest="dataset_split",
                   help="HF datasets split for calibration. Default: train. "
                        "Eval uses 'validation' split, kept disjoint from calib.")
    p.add_argument("--eval-query-tokens", type=int, default=4,
                   dest="eval_query_tokens",
                   help="Number of trailing query positions used for the "
                        "attention-cosine score on each eval sequence "
                        "(spec §13.3). Default: 4.")

    p.add_argument("--git-commit", type=str, default=None,
                   dest="git_commit_override",
                   help=(
                       "Override the recorded git commit SHA. Useful on "
                       "Modal where .git is not mounted; the launcher "
                       "forwards the local commit via the "
                       f"{GIT_COMMIT_ENV} env var. Must be hex, len>=7."
                   ))

    p.add_argument("--status-dir", type=str, default=None,
                   dest="status_dir",
                   help=(
                       "Directory under which heartbeat/progress JSON "
                       "artifacts are written (status.json + events.jsonl). "
                       "Defaults to a sibling of --output-dir if omitted "
                       "(see experiments/run_status.py::default_status_dir)."
                   ))

    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> Args:
    parser = build_parser()
    ns = parser.parse_args(argv)

    # --smoke -> synthetic_smoke.
    synthetic_smoke = bool(ns.synthetic_smoke or getattr(ns, "smoke_alias", False))
    skip_if_exists = bool(ns.skip_if_exists or getattr(ns, "resume_alias", False))

    if ns.dry_run and synthetic_smoke:
        # Dry-run wins: print the synthetic-smoke plan but compute nothing.
        synthetic_smoke = False
        print("[run_three_way] note: --dry-run takes precedence over --synthetic-smoke")

    inline_corpus_smoke = bool(getattr(ns, "inline_corpus_smoke", False))
    if synthetic_smoke and inline_corpus_smoke:
        parser.error(
            "--synthetic-smoke and --inline-corpus-smoke are mutually exclusive"
        )
    if ns.dry_run and inline_corpus_smoke:
        inline_corpus_smoke = False
        print(
            "[run_three_way] note: --dry-run takes precedence over "
            "--inline-corpus-smoke"
        )

    if ns.force and skip_if_exists:
        # User-friendly: --force implicitly disables skip.
        skip_if_exists = False

    if ns.n_calib < 1:
        parser.error(f"--n-calib must be >= 1, got {ns.n_calib}")
    if ns.n_eval < 1:
        parser.error(f"--n-eval must be >= 1, got {ns.n_eval}")
    if ns.n_layers_sample < 1:
        parser.error(f"--n-layers-sample must be >= 1, got {ns.n_layers_sample}")

    layer_sample = LAYER_SAMPLE_LISTS.get(
        ns.model,
        list(range(ns.n_layers_sample)),
    )[: ns.n_layers_sample]

    if ns.wf_max_bits is not None and ns.wf_max_bits < ns.wf_min_bits:
        parser.error(
            f"--wf-max-bits ({ns.wf_max_bits}) must be >= --wf-min-bits "
            f"({ns.wf_min_bits})"
        )

    if ns.eval_query_tokens < 1:
        parser.error(
            f"--eval-query-tokens must be >= 1, got {ns.eval_query_tokens}"
        )

    if (ns.save_calibration or ns.load_calibration) and ns.calibration_dir is None:
        parser.error(
            "--save-calibration / --load-calibration require --calibration-dir"
        )

    if ns.save_calibration and ns.load_calibration:
        parser.error(
            "--save-calibration and --load-calibration are mutually exclusive"
        )

    return Args(
        model=ns.model,
        avg_bits=int(ns.avg_bits),
        n_calib=int(ns.n_calib),
        n_eval=int(ns.n_eval),
        n_layers_sample=int(ns.n_layers_sample),
        output_dir=Path(ns.output_dir),
        seed=int(ns.seed),
        device=ns.device,
        dtype=ns.dtype,
        max_calib_tokens=int(ns.max_calib_tokens),
        dry_run=bool(ns.dry_run),
        synthetic_smoke=bool(synthetic_smoke),
        inline_corpus_smoke=bool(inline_corpus_smoke),
        skip_if_exists=bool(skip_if_exists),
        force=bool(ns.force),
        force_real_filename=bool(ns.force_real_filename),
        qjl_projections=int(ns.qjl_projections),
        wf_min_bits=int(ns.wf_min_bits),
        wf_max_bits=None if ns.wf_max_bits is None else int(ns.wf_max_bits),
        layer_sample=list(layer_sample),
        calibration_dir=(Path(ns.calibration_dir)
                          if ns.calibration_dir is not None else None),
        save_calibration=bool(ns.save_calibration),
        load_calibration=bool(ns.load_calibration),
        dataset_name=str(ns.dataset_name),
        dataset_config=str(ns.dataset_config),
        dataset_split=str(ns.dataset_split),
        eval_query_tokens=int(ns.eval_query_tokens),
        git_commit_override=(
            str(ns.git_commit_override)
            if ns.git_commit_override is not None
            else None
        ),
        status_dir=(
            Path(ns.status_dir) if ns.status_dir is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------


def build_software_block(engine_label: str) -> Dict[str, Any]:
    """Software metadata (spec §14)."""
    pyver = ".".join(str(x) for x in sys.version_info[:3])

    def _try_version(mod: str) -> str:
        try:
            return __import__(mod).__version__  # type: ignore[attr-defined]
        except Exception:
            return "unavailable"

    return {
        "python": pyver,
        "torch": _try_version("torch"),
        "transformers": _try_version("transformers"),
        "datasets": _try_version("datasets"),
        "engine": engine_label,
    }


def build_hardware_block(device: str) -> Dict[str, Any]:
    """Best-effort hardware metadata, no GPU access required."""
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
        except Exception:
            pass
    return block


def build_model_block(model_name: str, synthetic: bool) -> Dict[str, Any]:
    """Model architecture metadata.

    For real HF models we'd call ``AutoConfig.from_pretrained``; this slice
    only reads from a static table for the two spec models, plus a generic
    fallback so synthetic-smoke still validates the schema.
    """
    if model_name in MODEL_ARCH_META:
        meta = dict(MODEL_ARCH_META[model_name])
    else:
        meta = {
            "layers": 4, "q_heads": 4, "kv_heads": 2,
            "head_dim": 16, "gqa_ratio": 2,
        }
    return {
        "name": model_name if not synthetic else f"synthetic::{model_name}",
        **meta,
    }


def build_data_block(args: Args) -> Dict[str, Any]:
    """Data block (spec §14).

    ``inline_smoke`` marks a harness-validation run that bypassed the HF
    datasets download. These records are *not* paper-valid evidence — full
    sweeps must use the WikiText / proper eval corpus.
    """
    if args.synthetic_smoke:
        corpus = "synthetic"
    elif args.inline_corpus_smoke:
        corpus = "inline_smoke"
    else:
        corpus = "WikiText-103"
    return {
        "calibration_corpus": corpus,
        "n_calib": args.n_calib,
        "eval_corpus": corpus,
        "n_eval": args.n_eval,
        "max_calib_tokens": args.max_calib_tokens,
        "disjoint_eval": True,
    }


def build_calibration_block(d_eff_stats: Dict[str, float]) -> Dict[str, Any]:
    """Calibration block (spec §14)."""
    return {
        "normalize_keys": True,
        "key_space": "pre_rope",
        "d_eff_method": "participation_ratio",
        "d_eff_rounding": "ceil",
        "d_eff_min": 2,
        "d_eff_max": 126,
        "d_eff_stats": {
            "mean": float(d_eff_stats.get("mean", 0.0)),
            "min": float(d_eff_stats.get("min", 0.0)),
            "max": float(d_eff_stats.get("max", 0.0)),
        },
    }


# ---------------------------------------------------------------------------
# Method-record helpers
# ---------------------------------------------------------------------------


def _per_layer_records(
    layer_indices: Sequence[int],
    cosines: Sequence[float],
    d_effs: Sequence[int],
) -> List[Dict[str, Any]]:
    """Build a per-layer table for one method record."""
    if len(layer_indices) != len(cosines) or len(layer_indices) != len(d_effs):
        raise ValueError("layer_indices, cosines, and d_effs must align")
    out: List[Dict[str, Any]] = []
    for li, c, d in zip(layer_indices, cosines, d_effs):
        c_clamped = max(-1.0, min(1.0, float(c)))
        out.append({
            "layer_index": int(li),
            "d_eff": int(d),
            "attn_cosine_mean": c_clamped,
            "attn_cosine_min": c_clamped,
            "attn_cosine_max": c_clamped,
        })
    return out


def _accounting_record(
    method: str,
    avg_bits: int,
    head_dim: int,
    d_eff: Optional[int],
    waterfill_allocation: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build a CompressionAccounting dict via spectralquant.accounting."""
    from spectralquant.accounting import (
        spectralquant_accounting,
        turboquant_accounting,
    )

    if method == "turboquant":
        return turboquant_accounting(avg_bits=avg_bits, head_dim=head_dim).to_dict()

    # SpectralQuant v1/v2: feed bit components consistent with v1's
    # b_high/b_low layout. For the synthetic-smoke we use a simple
    # account that mirrors v1's regime split; for the real path we'd
    # pull these from the engine's allocation_metadata().
    if d_eff is None:
        d_eff = max(1, head_dim // 4)
    semantic_bits = float(avg_bits * d_eff)
    tail_bits = float(avg_bits * (head_dim - d_eff))
    k_mse_bits = semantic_bits + tail_bits  # full b-bit MSE
    k_qjl_bits = float(d_eff)               # selective QJL on d_eff
    v_mse_bits = float(avg_bits * head_dim)

    if waterfill_allocation is not None and len(waterfill_allocation) == d_eff:
        # Accounting expects exact per-dim sum for the semantic regime.
        wf_sum = sum(int(b) for b in waterfill_allocation)
        # Replace the uniform semantic component with the water-filled sum.
        semantic_bits = float(wf_sum)
        k_mse_bits = semantic_bits + tail_bits

    return spectralquant_accounting(
        avg_bits=avg_bits,
        head_dim=head_dim,
        d_eff=int(d_eff),
        k_mse_bits=k_mse_bits,
        k_qjl_bits=k_qjl_bits,
        v_mse_bits=v_mse_bits,
        method=method,
        waterfill_allocation=waterfill_allocation,
    ).to_dict()


def _method_record(
    method: str,
    layer_indices: Sequence[int],
    cosines: Sequence[float],
    d_effs: Sequence[int],
    head_dim: int,
    avg_bits: int,
    waterfill_allocation: Optional[Sequence[int]] = None,
    label: Optional[str] = None,
    evidence_id: str = "RUN-THREEWAY-SMOKE-001",
) -> Dict[str, Any]:
    cos_arr = [max(-1.0, min(1.0, float(c))) for c in cosines]
    cos_mean = float(sum(cos_arr) / len(cos_arr)) if cos_arr else 0.0
    if len(cos_arr) > 1:
        mean = cos_mean
        var = sum((c - mean) ** 2 for c in cos_arr) / (len(cos_arr) - 1)
        cos_std = float(math.sqrt(max(0.0, var)))
    else:
        cos_std = 0.0
    record: Dict[str, Any] = {
        "attn_cosine_mean": cos_mean,
        "attn_cosine_std": cos_std,
        "compression_accounting": _accounting_record(
            method=method,
            avg_bits=avg_bits,
            head_dim=head_dim,
            d_eff=int(d_effs[0]) if d_effs else None,
            waterfill_allocation=(
                list(waterfill_allocation) if waterfill_allocation else None
            ),
        ),
        "per_layer": _per_layer_records(layer_indices, cos_arr, d_effs),
        "evidence_ids": [evidence_id],
    }
    if label is not None:
        record["label"] = label
    return record


# ---------------------------------------------------------------------------
# Synthetic-smoke pipeline
# ---------------------------------------------------------------------------


def run_synthetic_smoke(args: Args) -> Dict[str, Any]:
    """Drive the three engines on small synthetic Q/K/V tensors.

    The point of this mode is to exercise the *result-plumbing*:
    cosine collection, accounting math, allocation metadata, schema
    validation. The numerical values are not meaningful.
    """
    import numpy as np
    import torch

    from spectralquant import (
        EngineConfig,
        SpectralQuantEngine,
        TurboQuantBaseline,
    )
    from spectralquant.calibration import (
        EigenspectralCalibrator,
        HeadCalibrationData,
    )

    # Tiny architecture so this finishes in well under a second.
    head_dim = 16
    n_kv_heads = 2
    n_q_heads = 4
    n_tokens = 64
    n_eval_tokens = 8

    layer_indices = list(args.layer_sample)
    if not layer_indices:
        layer_indices = [0]

    rng = np.random.default_rng(args.seed)

    # Synthetic eigenspectrum with a sharp gap so v1 and v2 differ.
    big = np.array([100.0, 60.0, 30.0, 12.0], dtype=np.float64)
    tail = np.full(head_dim - big.shape[0], 0.5, dtype=np.float64)
    eigenvalues = np.concatenate([big, tail])

    def _build_calib() -> Tuple[EigenspectralCalibrator, torch.Tensor]:
        calib = EigenspectralCalibrator(max_tokens_per_layer=n_tokens)
        rotated = (
            rng.standard_normal((n_tokens, head_dim)).astype(np.float32)
            * np.sqrt(eigenvalues).astype(np.float32)
        )
        sum_lam = float(eigenvalues.sum())
        sum_sq = float((eigenvalues ** 2).sum())
        d_eff_val = (sum_lam ** 2) / sum_sq if sum_sq > 1e-12 else 1.0
        for layer_idx in layer_indices:
            for head_idx in range(n_kv_heads):
                for head_type in ("key", "value"):
                    A = rng.standard_normal((head_dim, head_dim))
                    Q, _ = np.linalg.qr(A)
                    hcd = HeadCalibrationData(
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        head_type=head_type,
                        eigenvalues=torch.from_numpy(
                            eigenvalues.astype(np.float32)
                        ),
                        eigenvectors=torch.from_numpy(Q.astype(np.float32)),
                        d_eff=float(d_eff_val),
                        spectral_gap=None,
                        var_95=int(min(head_dim, max(1, round(d_eff_val)))),
                        var_99=int(min(head_dim, max(1, round(d_eff_val)))),
                        n_samples=n_tokens,
                        head_dim=head_dim,
                    )
                    calib._calibration_data[
                        (layer_idx, head_idx, head_type)
                    ] = hcd
        calib._is_calibrated = True
        return calib, torch.from_numpy(rotated)

    def _fit_engine(use_water_fill: bool) -> SpectralQuantEngine:
        calib, rotated = _build_calib()
        cfg = EngineConfig(
            avg_bits=float(args.avg_bits),
            qjl_projections=args.qjl_projections,
            use_water_fill=use_water_fill,
            wf_min_bits=args.wf_min_bits,
            wf_max_bits=args.wf_max_bits,
            rotation_seed=args.seed,
            lloyd_seed=args.seed,
        )
        engine = SpectralQuantEngine(calib, cfg)
        rotated_kv = {
            (layer_idx, head_idx, head_type): rotated
            for layer_idx in layer_indices
            for head_idx in range(n_kv_heads)
            for head_type in ("key", "value")
        }
        engine.fit_quantizers(rotated_kv)
        return engine

    engine_v1 = _fit_engine(use_water_fill=False)
    engine_v2 = _fit_engine(use_water_fill=True)

    baseline = TurboQuantBaseline(
        n_layers=max(layer_indices) + 1,
        n_heads=n_kv_heads,
        head_dim=head_dim,
        config=EngineConfig(
            avg_bits=float(args.avg_bits),
            qjl_projections=args.qjl_projections,
            rotation_seed=args.seed,
            lloyd_seed=args.seed,
        ),
    )
    rotated_for_baseline = (
        rng.standard_normal((n_tokens, head_dim)).astype(np.float32)
        * np.sqrt(eigenvalues).astype(np.float32)
    )
    baseline.fit_quantizers({
        (layer_idx, head_idx, head_type):
            torch.from_numpy(rotated_for_baseline)
        for layer_idx in layer_indices
        for head_idx in range(n_kv_heads)
        for head_type in ("key", "value")
    })

    # Build small Q/K/V tensors and collect attention-output cosines.
    g = torch.Generator().manual_seed(args.seed)
    keys = torch.randn(
        1, n_kv_heads, n_eval_tokens, head_dim, generator=g,
    )
    values = torch.randn(
        1, n_kv_heads, n_eval_tokens, head_dim, generator=g,
    )
    queries = torch.randn(
        1, n_q_heads, 4, head_dim, generator=g,
    )

    def _attn_fp16(layer_idx: int) -> torch.Tensor:
        # Reference attention with the same GQA mapping the engine uses.
        ratio = n_q_heads // n_kv_heads
        scores = []
        for h in range(n_kv_heads):
            q_grp = queries[:, h * ratio:(h + 1) * ratio]
            k = keys[:, h]
            scores.append(
                torch.matmul(q_grp, k.transpose(-2, -1))
            )
        s = torch.cat(scores, dim=1) * (head_dim ** -0.5)
        return torch.softmax(s, dim=-1)

    def _cosine(weights_a: torch.Tensor, weights_b: torch.Tensor) -> float:
        a = weights_a.flatten()
        b = weights_b.flatten()
        cs = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0),
        )
        return float(cs.item())

    cos_tq: List[float] = []
    cos_v1: List[float] = []
    cos_v2: List[float] = []
    d_effs: List[int] = []

    for layer_idx in layer_indices:
        ref = _attn_fp16(layer_idx)
        ck_v1 = engine_v1.compress_keys(keys, layer_idx)
        ck_v2 = engine_v2.compress_keys(keys, layer_idx)
        # use_value_rotation default True -> compress values for shape parity
        engine_v1.compress_values(values, layer_idx)
        engine_v2.compress_values(values, layer_idx)
        ck_tq = baseline.compress_keys(keys, layer_idx)

        w_v1 = engine_v1.attention_score(queries, ck_v1, layer_idx)
        w_v2 = engine_v2.attention_score(queries, ck_v2, layer_idx)
        w_tq = baseline.attention_score(queries, ck_tq, layer_idx)

        cos_v1.append(_cosine(w_v1, ref))
        cos_v2.append(_cosine(w_v2, ref))
        cos_tq.append(_cosine(w_tq, ref))
        d_effs.append(int(round(engine_v1.allocation_metadata()
                                ["per_head"][0]["allocation"]["d_eff"])))

    # Allocation metadata comes from v2 engine; pass the first head's
    # bits_per_dim into the accounting record.
    v2_meta = engine_v2.allocation_metadata()
    waterfill_alloc: Optional[List[int]] = None
    if v2_meta["per_head"]:
        first = v2_meta["per_head"][0].get("allocation")
        if first is not None:
            waterfill_alloc = list(first["bits_per_dim"])

    d_eff_mean = float(sum(d_effs) / len(d_effs)) if d_effs else 0.0
    d_eff_min = float(min(d_effs)) if d_effs else 0.0
    d_eff_max = float(max(d_effs)) if d_effs else 0.0

    return {
        "head_dim": head_dim,
        "layer_indices": layer_indices,
        "d_effs": d_effs,
        "d_eff_stats": {
            "mean": d_eff_mean, "min": d_eff_min, "max": d_eff_max,
        },
        "methods": {
            "turboquant": _method_record(
                "turboquant", layer_indices, cos_tq, d_effs,
                head_dim, args.avg_bits, label="local",
            ),
            "spectralquant_v1": _method_record(
                "spectralquant_v1", layer_indices, cos_v1, d_effs,
                head_dim, args.avg_bits,
            ),
            "spectralquant_v2": _method_record(
                "spectralquant_v2", layer_indices, cos_v2, d_effs,
                head_dim, args.avg_bits,
                waterfill_allocation=waterfill_alloc,
            ),
        },
        "v2_allocation_metadata": v2_meta,
    }


# ---------------------------------------------------------------------------
# Full HuggingFace pipeline
# ---------------------------------------------------------------------------
#
# The full path is exercised on Modal where a CUDA GPU and the gated
# Mistral / public Qwen weights are reachable. To keep this module
# importable in a clean checkout, every transformers / datasets call is
# inside the function body and gated by lazy imports — the import-time
# surface is plain Python plus torch.
#
# Pipeline (spec §13.3, §13.4):
#
#   1. Load HuggingFace tokenizer and model in the requested dtype.
#      * The Modal launcher injects HF_TOKEN; we never read it here.
#   2. Pull a calibration corpus (default: WikiText-103 train split) and
#      build a list of text strings of length n_calib + n_eval.
#   3. Either reload a prior calibration .pt + _meta.json (resume) or run
#      EigenspectralCalibrator on the calibration slice, then save the
#      artifact next to the result JSON.
#   4. Build per-(layer,head,type) rotated KV using the calibrator's V^T,
#      and fit the v1 / v2 quantizers via SpectralQuantEngine.fit_quantizers.
#      The TurboQuant baseline gets the un-rotated KV (its random rotation
#      is internal).
#   5. For each sampled layer, register a forward hook to capture the live
#      Q/K/V tensors during a forward pass on each eval sequence, then
#      compute the attention weights produced by:
#        - the FP16 attention (reference)
#        - SpectralQuantEngine v1
#        - SpectralQuantEngine v2
#        - TurboQuantBaseline
#      and accumulate per-layer cosine similarities.
#   6. Aggregate per-method records and emit the schema-valid JSON via
#      atomic_write_json (validates before rename).


def missing_calibration_entries(
    calib: Any,
    layer_sample: Sequence[int],
    n_kv_heads: int,
) -> List[Tuple[int, int, str]]:
    """Return the list of ``(layer_idx, kv_head, head_type)`` triples that the
    eval loop will need but ``calib`` does not have data for.

    ``calib`` is anything with a ``get(layer_idx, head_idx, head_type)``
    method that returns ``None`` when an entry is missing (matching the
    :class:`EigenspectralCalibrator` contract). Returned in the same order
    the eval loop iterates so error messages are stable.

    The helper is exposed at module scope so tests can pin the contract
    without standing up a full HF model.
    """
    missing: List[Tuple[int, int, str]] = []
    for li in layer_sample:
        for hi in range(int(n_kv_heads)):
            for ht in ("key", "value"):
                if calib.get(int(li), int(hi), ht) is None:
                    missing.append((int(li), int(hi), ht))
    return missing


def _calibration_artifact_paths(args: "Args") -> Tuple[Path, Path]:
    """Return (artifact_base, meta_json_path) for the calibration cache.

    File names embed (model_short, n_calib, max_calib_tokens, seed) so a
    different seed or sample size cannot collide silently with another
    artifact. ``EigenspectralCalibrator.save(base)`` writes ``base.pt`` and
    ``base_meta.json``.
    """
    if args.calibration_dir is None:
        raise ValueError("calibration_dir is required for full-path artifacts")
    short = _model_short(args.model)
    base = (
        args.calibration_dir
        / f"{short}_calib{args.n_calib}_tok{args.max_calib_tokens}_seed{args.seed}"
    )
    return base, Path(str(base) + "_meta.json")


#: Deterministic inline corpus used by ``--inline-corpus-smoke``. The strings
#: are long enough to tokenize into >=16 tokens with any common BPE/SP
#: tokenizer so they reach the calibrator's per-layer token cap. The list is
#: cycled to satisfy any ``n_calib + n_eval`` request.
INLINE_SMOKE_CORPUS: Tuple[str, ...] = (
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


def _build_inline_corpus(n_total: int) -> List[str]:
    """Return ``n_total`` deterministic strings for the inline-smoke path.

    Cycles the static :data:`INLINE_SMOKE_CORPUS` list so any reasonable
    ``n_calib + n_eval`` request is satisfied without external I/O.
    Each emitted string is suffixed with its index so the calibration and
    eval slices remain disjoint (mirroring the ``disjoint_eval`` contract
    used by the WikiText path).
    """
    if n_total < 1:
        raise ValueError(f"n_total must be >= 1, got {n_total}")
    base = list(INLINE_SMOKE_CORPUS)
    out: List[str] = []
    for i in range(n_total):
        text = base[i % len(base)]
        out.append(f"{text} [inline_smoke#{i:03d}]")
    return out


def _load_calibration_corpus(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    n_total: int,
) -> List[str]:
    """Load ``n_total`` non-empty text strings from a HF datasets split.

    Lazy-imports ``datasets``. Raises a clear ``RuntimeError`` if datasets
    is not installed (i.e. running outside the Modal image).
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for the full HF path; install it on the "
            f"Modal image (currently missing: {exc!r})."
        ) from exc

    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    out: List[str] = []
    for row in ds:
        text = row.get("text") if isinstance(row, dict) else None
        if not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < 16:
            continue
        out.append(text)
        if len(out) >= n_total:
            break
    if len(out) < n_total:
        raise RuntimeError(
            f"Dataset {dataset_name}/{dataset_config}:{dataset_split} yielded "
            f"only {len(out)} non-empty rows; need {n_total}."
        )
    return out


def _resolve_dtype(name: str) -> Any:
    import torch
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_hf_model_and_tokenizer(
    model_id: str,
    device: str,
    dtype: str,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load tokenizer + model and return (model, tokenizer, software_info).

    The HF_TOKEN env var, if present, is honoured by huggingface_hub
    transparently. We never read or echo it here.
    """
    try:
        import torch  # noqa: F401 — confirm torch is importable
        from transformers import (  # type: ignore[import-not-found]
            AutoTokenizer,
            AutoModelForCausalLM,
        )
        from transformers.utils import logging as hf_logging  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for the full HF path; install it on "
            f"the Modal image (currently missing: {exc!r})."
        ) from exc

    hf_logging.set_verbosity_error()

    torch_dtype = _resolve_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Ensure the Modal function has gpu= set."
            )
        model = model.to("cuda")

    cfg = getattr(model, "config", None)
    info = {
        "tokenizer_class": type(tokenizer).__name__,
        "model_class": type(model).__name__,
        "model_revision": (
            getattr(cfg, "_commit_hash", None) or "unknown"
        ),
        "model_type": getattr(cfg, "model_type", "unknown"),
    }
    return model, tokenizer, info


def _build_calibration(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    args: "Args",
    save_to: Optional[Path],
) -> Any:
    """Run EigenspectralCalibrator on ``texts`` and return the calibrator.

    Side-effect: when ``save_to`` is provided, write ``save_to.pt`` and
    ``save_to_meta.json`` so subsequent runs can resume via
    ``--load-calibration``.
    """
    from spectralquant.calibration import EigenspectralCalibrator

    calib = EigenspectralCalibrator(max_tokens_per_layer=args.max_calib_tokens)
    calib.calibrate(model, tokenizer, list(texts), n_samples=len(texts))
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        calib.save(str(save_to))
    return calib


def _capture_qkv_per_layer(
    model: Any,
    tokenizer: Any,
    text: str,
    layer_indices: Sequence[int],
    max_seq_tokens: int,
) -> Dict[int, Dict[str, Any]]:
    """Run a forward pass and snapshot the live Q/K/V per sampled layer.

    Returns a dict keyed by ``layer_idx`` with sub-dict
    ``{"q", "k", "v"}`` of CPU float tensors of shape
    ``(batch, n_heads_or_kv, seq_len, head_dim)``.

    The hooks call into the layer's ``q_proj``/``k_proj``/``v_proj``
    discovered by :mod:`experiments.model_adapters` so the snapshot is
    pre-RoPE and pre-cache. RoPE is applied later via the calibrator's
    rotated basis (matching spec §13.3 ``key_space=pre_rope``).
    """
    import torch
    from experiments import model_adapters

    layers = model_adapters.find_attention_modules(model)
    by_idx = {layer.layer_idx: layer for layer in layers}

    # Resolve dims from the first sampled layer.
    cfg = getattr(model, "config", None)
    n_q_heads, n_kv_heads, head_dim = model_adapters.get_kv_dims(
        cfg, by_idx[layer_indices[0]].attn_module
    )

    captures: Dict[int, Dict[str, Any]] = {}
    handles: List[Any] = []

    def _make_hook(layer_idx: int, q_proj: Any, k_proj: Any, v_proj: Any):
        # Capture the *outputs* of q_proj/k_proj/v_proj. These are
        # pre-RoPE projections of the hidden state.
        bucket: Dict[str, Any] = {}

        def q_hook(_mod, _inputs, output):
            bucket["q"] = output.detach().to("cpu", dtype=torch.float32)

        def k_hook(_mod, _inputs, output):
            bucket["k"] = output.detach().to("cpu", dtype=torch.float32)

        def v_hook(_mod, _inputs, output):
            bucket["v"] = output.detach().to("cpu", dtype=torch.float32)

        handles.append(q_proj.register_forward_hook(q_hook))
        handles.append(k_proj.register_forward_hook(k_hook))
        handles.append(v_proj.register_forward_hook(v_hook))
        captures[layer_idx] = bucket

    for layer_idx in layer_indices:
        if layer_idx not in by_idx:
            raise RuntimeError(
                f"layer_idx={layer_idx} not found in model "
                f"(model has {len(by_idx)} layers)."
            )
        layer = by_idx[layer_idx]
        _make_hook(layer_idx, layer.q_proj, layer.k_proj, layer.v_proj)

    try:
        device = next(model.parameters()).device
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_tokens,
        ).to(device)
        with torch.no_grad():
            model(**inputs, use_cache=False, output_attentions=False)
    finally:
        for h in handles:
            h.remove()

    # Reshape projections from (batch, seq, n_heads*head_dim) to
    # (batch, n_heads_or_kv, seq, head_dim).
    out: Dict[int, Dict[str, Any]] = {}
    for layer_idx, bucket in captures.items():
        if not all(k in bucket for k in ("q", "k", "v")):
            raise RuntimeError(
                f"Hooks did not fire for layer {layer_idx}; got keys "
                f"{list(bucket.keys())}"
            )
        q = bucket["q"]
        k = bucket["k"]
        v = bucket["v"]
        bsz, seq, _ = q.shape
        q = q.reshape(bsz, seq, n_q_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(bsz, seq, n_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(bsz, seq, n_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        out[layer_idx] = {"q": q, "k": k, "v": v}
    return out


def _reference_attention(
    queries: Any,
    keys: Any,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> Any:
    """Reference attention weights used as the cosine baseline."""
    import torch

    ratio = n_q_heads // n_kv_heads
    scores: List[Any] = []
    for h in range(n_kv_heads):
        q_grp = queries[:, h * ratio:(h + 1) * ratio]
        k = keys[:, h]
        scores.append(torch.matmul(q_grp, k.transpose(-2, -1)))
    s = torch.cat(scores, dim=1) * (head_dim ** -0.5)
    return torch.softmax(s, dim=-1)


def _flatten_cosine(a: Any, b: Any) -> float:
    import torch

    av = a.detach().flatten().to(torch.float32)
    bv = b.detach().flatten().to(torch.float32)
    cs = torch.nn.functional.cosine_similarity(
        av.unsqueeze(0), bv.unsqueeze(0)
    )
    return float(cs.item())


def run_full_hf(args: "Args", status: Any = None) -> Dict[str, Any]:
    """Execute the full HuggingFace three-way benchmark.

    Returns the inner ``smoke``-shaped dict the same way :func:`run_synthetic_smoke`
    does, so :func:`build_payload` can compose the JSON.

    Heavy by design — only call this on Modal or with the right local
    environment. The harness checks ``--dry-run`` / ``--synthetic-smoke``
    before reaching here. ``status`` is an optional
    :class:`run_status.StatusWriter`; when present, key stages emit
    heartbeat artifacts so an operator can poll the Modal volume even
    without log access.
    """
    def _emit(stage: str, **kwargs: Any) -> None:
        if status is not None:
            try:
                status.emit(stage, **kwargs)
            except Exception:
                pass

    import torch

    from spectralquant import (
        EngineConfig,
        SpectralQuantEngine,
        TurboQuantBaseline,
    )
    from spectralquant.calibration import EigenspectralCalibrator
    from experiments import model_adapters

    _emit("import_ok", message="full-path imports ok")

    # 1. Load model.
    _emit(
        "model_load_start",
        message=f"loading {args.model}",
        details={"device": args.device, "dtype": args.dtype},
    )
    model, tokenizer, software_info = _load_hf_model_and_tokenizer(
        args.model, args.device, args.dtype
    )
    _emit("model_load_end", details={"model_class": software_info.get("model_class")})

    cfg = getattr(model, "config", None)
    layers = model_adapters.find_attention_modules(model)
    if not layers:
        raise RuntimeError("No attention layers discovered.")
    n_q_heads, n_kv_heads, head_dim = model_adapters.get_kv_dims(
        cfg, layers[0].attn_module
    )

    # 2. Calibration corpus.
    n_total_texts = args.n_calib + args.n_eval
    if args.inline_corpus_smoke:
        _emit(
            "dataset_inline_start",
            message=(
                "inline-corpus smoke: bypassing HF datasets.load_dataset; "
                "harness validation only — not paper-valid evidence"
            ),
            details={
                "corpus": "inline_smoke",
                "n_total_texts": int(n_total_texts),
                "paper_valid": False,
            },
        )
        texts = _build_inline_corpus(n_total_texts)
        _emit(
            "dataset_inline_end",
            details={
                "corpus": "inline_smoke",
                "n_texts": len(texts),
                "paper_valid": False,
            },
        )
    else:
        _emit(
            "dataset_load_start",
            message=f"loading {args.dataset_name}/{args.dataset_config}",
            details={
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_split": args.dataset_split,
            },
        )
        texts = _load_calibration_corpus(
            args.dataset_name,
            args.dataset_config,
            args.dataset_split,
            n_total=n_total_texts,
        )
        _emit("dataset_load_end", details={"n_texts": len(texts)})
    calib_texts = texts[: args.n_calib]
    eval_texts = texts[args.n_calib : args.n_calib + args.n_eval]

    # 3. Calibration (load or recompute).
    artifact_base: Optional[Path] = None
    if args.calibration_dir is not None:
        artifact_base, _ = _calibration_artifact_paths(args)

    _emit(
        "calibration_start",
        message=("loading calibration artifact" if args.load_calibration
                 else "computing calibration"),
        details={
            "artifact": str(artifact_base) if artifact_base else None,
            "load": bool(args.load_calibration),
            "save": bool(args.save_calibration),
        },
    )
    if args.load_calibration and artifact_base is not None:
        calib = EigenspectralCalibrator(max_tokens_per_layer=args.max_calib_tokens)
        calib.load(str(artifact_base))
    else:
        save_to = artifact_base if args.save_calibration else None
        calib = _build_calibration(
            model, tokenizer, calib_texts, args,
            save_to=save_to,
        )
    _emit("calibration_end")

    # Validate that every (sampled_layer, kv_head, head_type) we will hit
    # during eval has calibration data. Without this, the eval loop would
    # raise a confusing
    #     KeyError: No quantizer for layer=L, head=H, type='key'
    # mid-way through the run, and the operator would have to chase the
    # call stack back to the calibrator.
    missing = missing_calibration_entries(
        calib,
        layer_sample=args.layer_sample,
        n_kv_heads=int(n_kv_heads),
    )
    n_expected = len(args.layer_sample) * int(n_kv_heads) * 2
    if missing:
        sample = ", ".join(
            f"L{li}H{hi}/{ht}" for li, hi, ht in missing[:8]
        )
        msg = (
            f"Calibration missing {len(missing)}/{n_expected} "
            f"required (layer, kv_head, type) entries; first few: {sample}. "
            f"This usually means the K/V hooks did not capture any tokens — "
            f"check the model's k_proj/v_proj layout and that calibration "
            f"texts are non-empty (n_calib={args.n_calib})."
        )
        _emit(
            "calibration_coverage_failed",
            message=msg,
            details={
                "n_expected": n_expected,
                "n_missing": len(missing),
                "missing_sample": [
                    {"layer": li, "head": hi, "type": ht}
                    for li, hi, ht in missing[:16]
                ],
                "sampled_layers": list(args.layer_sample),
                "n_kv_heads": int(n_kv_heads),
            },
        )
        raise RuntimeError(msg)
    _emit(
        "calibration_coverage_ok",
        details={
            "n_expected": n_expected,
            "sampled_layers": list(args.layer_sample),
            "n_kv_heads": int(n_kv_heads),
        },
    )

    # 4. Build rotated KV from calibration data and fit engines.
    rotated_kv: Dict[Tuple[int, int, str], Any] = {}
    for hcd in calib.iter_heads():
        # Synthesise rotated samples by drawing from the calibrated
        # eigenspectrum: x = z * sqrt(λ) where z is unit Gaussian.
        # This matches the v2 spec's "fit on rotated K/V" usage of
        # NonUniformQuantizer when the original K vectors are not
        # cached. It is the same approach as the synthetic-smoke path.
        gen = torch.Generator().manual_seed(
            args.seed + 31 * hcd.layer_idx + hcd.head_idx
        )
        n = max(2 * head_dim, 256)
        z = torch.randn(n, head_dim, generator=gen)
        rotated_kv[(hcd.layer_idx, hcd.head_idx, hcd.head_type)] = (
            z * hcd.eigenvalues.sqrt()
        )

    cfg_v1 = EngineConfig(
        avg_bits=float(args.avg_bits),
        qjl_projections=args.qjl_projections,
        use_water_fill=False,
        rotation_seed=args.seed,
        lloyd_seed=args.seed,
    )
    cfg_v2 = EngineConfig(
        avg_bits=float(args.avg_bits),
        qjl_projections=args.qjl_projections,
        use_water_fill=True,
        wf_min_bits=args.wf_min_bits,
        wf_max_bits=args.wf_max_bits,
        rotation_seed=args.seed,
        lloyd_seed=args.seed,
    )
    engine_v1 = SpectralQuantEngine(calib, cfg_v1)
    engine_v2 = SpectralQuantEngine(calib, cfg_v2)
    engine_v1.fit_quantizers(rotated_kv)
    engine_v2.fit_quantizers(rotated_kv)

    baseline = TurboQuantBaseline(
        n_layers=getattr(cfg, "num_hidden_layers", len(layers)),
        n_heads=n_kv_heads,
        head_dim=head_dim,
        config=cfg_v1,
    )
    baseline.fit_quantizers({
        key: tensor for key, tensor in rotated_kv.items()
    })

    # 5. Per-layer attention cosine on each eval text.
    layer_indices = list(args.layer_sample)
    cos_tq: List[List[float]] = [[] for _ in layer_indices]
    cos_v1: List[List[float]] = [[] for _ in layer_indices]
    cos_v2: List[List[float]] = [[] for _ in layer_indices]
    d_effs: List[int] = []

    for li in layer_indices:
        # Pull d_eff from the calibrator's first head/key entry.
        hcd = calib.get(li, 0, "key")
        d_effs.append(int(round(hcd.d_eff)) if hcd is not None else 0)

    _emit(
        "eval_start",
        message=f"running eval on {len(eval_texts)} texts",
        details={
            "n_eval": len(eval_texts),
            "n_layers": len(layer_indices),
        },
    )
    for ti, text in enumerate(eval_texts):
        if ti > 0 and ti % max(1, len(eval_texts) // 4) == 0:
            _emit(
                "eval_progress",
                details={"completed": ti, "total": len(eval_texts)},
            )
        per_layer_qkv = _capture_qkv_per_layer(
            model, tokenizer, text,
            layer_indices=layer_indices,
            max_seq_tokens=args.max_calib_tokens,
        )
        for slot, li in enumerate(layer_indices):
            qkv = per_layer_qkv[li]
            q = qkv["q"]
            k = qkv["k"]
            # Trim queries to the trailing `eval_query_tokens` positions.
            q = q[:, :, -args.eval_query_tokens:, :]

            ref = _reference_attention(q, k, n_q_heads, n_kv_heads, head_dim)

            ck_v1 = engine_v1.compress_keys(k, li)
            ck_v2 = engine_v2.compress_keys(k, li)
            ck_tq = baseline.compress_keys(k, li)
            w_v1 = engine_v1.attention_score(q, ck_v1, li)
            w_v2 = engine_v2.attention_score(q, ck_v2, li)
            w_tq = baseline.attention_score(q, ck_tq, li)

            cos_v1[slot].append(_flatten_cosine(w_v1, ref))
            cos_v2[slot].append(_flatten_cosine(w_v2, ref))
            cos_tq[slot].append(_flatten_cosine(w_tq, ref))

    _emit("eval_end", details={"n_eval": len(eval_texts)})
    cos_tq_mean = [
        sum(xs) / len(xs) if xs else 0.0 for xs in cos_tq
    ]
    cos_v1_mean = [
        sum(xs) / len(xs) if xs else 0.0 for xs in cos_v1
    ]
    cos_v2_mean = [
        sum(xs) / len(xs) if xs else 0.0 for xs in cos_v2
    ]

    v2_meta = engine_v2.allocation_metadata()
    waterfill_alloc: Optional[List[int]] = None
    if v2_meta["per_head"]:
        first = v2_meta["per_head"][0].get("allocation")
        if first is not None:
            waterfill_alloc = list(first["bits_per_dim"])

    d_eff_mean = float(sum(d_effs) / len(d_effs)) if d_effs else 0.0
    d_eff_min = float(min(d_effs)) if d_effs else 0.0
    d_eff_max = float(max(d_effs)) if d_effs else 0.0

    return {
        "head_dim": head_dim,
        "layer_indices": layer_indices,
        "d_effs": d_effs,
        "d_eff_stats": {
            "mean": d_eff_mean, "min": d_eff_min, "max": d_eff_max,
        },
        "methods": {
            "turboquant": _method_record(
                "turboquant", layer_indices, cos_tq_mean, d_effs,
                head_dim, args.avg_bits, label="local",
                evidence_id="RUN-THREEWAY-001",
            ),
            "spectralquant_v1": _method_record(
                "spectralquant_v1", layer_indices, cos_v1_mean, d_effs,
                head_dim, args.avg_bits,
                evidence_id="RUN-THREEWAY-001",
            ),
            "spectralquant_v2": _method_record(
                "spectralquant_v2", layer_indices, cos_v2_mean, d_effs,
                head_dim, args.avg_bits,
                waterfill_allocation=waterfill_alloc,
                evidence_id="RUN-THREEWAY-001",
            ),
        },
        "v2_allocation_metadata": v2_meta,
        "model_block_overrides": {
            "name": args.model,
            "layers": getattr(cfg, "num_hidden_layers", len(layers)),
            "q_heads": n_q_heads,
            "kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "gqa_ratio": max(1, n_q_heads // n_kv_heads),
        },
        "software_overrides": software_info,
        "calibration_artifact": (
            str(artifact_base) if artifact_base is not None
            and (args.save_calibration or args.load_calibration) else None
        ),
    }


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def build_payload(
    args: Args,
    smoke: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose the full result JSON payload."""
    short = _model_short(args.model)
    run_id = f"{short}_b{args.avg_bits}_seed{args.seed}"
    if args.synthetic_smoke:
        run_id = "synthetic_smoke__" + run_id
    if args.inline_corpus_smoke:
        run_id = "inline_corpus_smoke__" + run_id
    if args.dry_run:
        run_id = "dryrun__" + run_id

    if smoke is None:
        # Plain dry-run / placeholder — no methods executed. We still
        # produce a schema-valid payload so dry-run output can be
        # eyeballed for shape correctness; cosine values are zeroed.
        meta = MODEL_ARCH_META.get(args.model, {
            "layers": 4, "q_heads": 4, "kv_heads": 2,
            "head_dim": 16, "gqa_ratio": 2,
        })
        head_dim = int(meta["head_dim"])
        layer_indices = list(args.layer_sample) or [0]
        cos_zeros = [0.0] * len(layer_indices)
        d_effs = [max(1, head_dim // 4)] * len(layer_indices)
        d_eff_stats = {
            "mean": float(d_effs[0]), "min": float(d_effs[0]), "max": float(d_effs[0]),
        }
        methods = {
            "turboquant": _method_record(
                "turboquant", layer_indices, cos_zeros, d_effs,
                head_dim, args.avg_bits, label="local",
            ),
            "spectralquant_v1": _method_record(
                "spectralquant_v1", layer_indices, cos_zeros, d_effs,
                head_dim, args.avg_bits,
            ),
            "spectralquant_v2": _method_record(
                "spectralquant_v2", layer_indices, cos_zeros, d_effs,
                head_dim, args.avg_bits,
            ),
        }
    else:
        head_dim = int(smoke["head_dim"])
        layer_indices = list(smoke["layer_indices"])
        d_eff_stats = smoke["d_eff_stats"]
        methods = smoke["methods"]

    model_block = build_model_block(args.model, args.synthetic_smoke)
    if smoke is not None and "model_block_overrides" in smoke:
        # Full path: replace the static dims with values pulled from the
        # live HF config so the JSON reflects what was actually loaded.
        model_block = dict(model_block)
        model_block.update(smoke["model_block_overrides"])

    software_block = build_software_block("SpectralQuantEngine")
    if smoke is not None and "software_overrides" in smoke:
        software_block = dict(software_block)
        software_block.update(smoke["software_overrides"])

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": _utc_now_iso(),
        "repo": REPO_SLUG,
        "commit": _git_commit(override=args.git_commit_override),
        "command": _safe_command_string([sys.argv[0], *sys.argv[1:]]),
        "model": model_block,
        "hardware": build_hardware_block(args.device),
        "software": software_block,
        "data": build_data_block(args),
        "calibration": build_calibration_block(d_eff_stats),
        "methods": methods,
        "evidence_ids": [
            "RUN-THREEWAY-SMOKE-001" if args.synthetic_smoke
            else "RUN-THREEWAY-INLINESMOKE-001" if args.inline_corpus_smoke
            else "RUN-THREEWAY-001"
        ],
        "mode": (
            "dry-run" if args.dry_run
            else "synthetic-smoke" if args.synthetic_smoke
            else "inline-corpus-smoke" if args.inline_corpus_smoke
            else "full"
        ),
        "paper_valid": (
            False if args.dry_run or args.synthetic_smoke
            or args.inline_corpus_smoke
            else True
        ),
        "config": {
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "qjl_projections": args.qjl_projections,
            "wf_min_bits": args.wf_min_bits,
            "wf_max_bits": args.wf_max_bits,
            "n_layers_sample": args.n_layers_sample,
            "layer_sample": list(layer_indices),
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "eval_query_tokens": args.eval_query_tokens,
            "calibration_dir": (
                str(args.calibration_dir) if args.calibration_dir is not None else None
            ),
            "save_calibration": args.save_calibration,
            "load_calibration": args.load_calibration,
        },
    }
    if smoke is not None and "v2_allocation_metadata" in smoke:
        payload["v2_allocation_metadata"] = smoke["v2_allocation_metadata"]
    if smoke is not None and smoke.get("calibration_artifact"):
        payload["calibration_artifact"] = smoke["calibration_artifact"]
    return payload


def print_plan(args: Args, output_path: Path) -> None:
    """Human-readable summary of what the run would do. No secrets."""
    secret_presence = {
        name: ("set" if os.environ.get(name) else "unset")
        for name in SECRET_ENV_VARS
    }
    plan = {
        "model": args.model,
        "avg_bits": args.avg_bits,
        "n_calib": args.n_calib,
        "n_eval": args.n_eval,
        "n_layers_sample": args.n_layers_sample,
        "layer_sample": args.layer_sample,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "qjl_projections": args.qjl_projections,
        "wf_min_bits": args.wf_min_bits,
        "wf_max_bits": args.wf_max_bits,
        "output_dir": str(args.output_dir),
        "output_path": str(output_path),
        "mode": (
            "dry-run" if args.dry_run
            else "synthetic-smoke" if args.synthetic_smoke
            else "inline-corpus-smoke" if args.inline_corpus_smoke
            else "full (NotImplemented in this slice)"
        ),
        "skip_if_exists": args.skip_if_exists,
        "force": args.force,
        "secrets_presence": secret_presence,
        "repo": REPO_SLUG,
        "commit": _git_commit(override=args.git_commit_override),
        "commit_source": (
            "cli" if args.git_commit_override
            else ("env" if os.environ.get(GIT_COMMIT_ENV) else "git")
        ),
    }
    print("[run_three_way] plan:")
    for k, v in plan.items():
        print(f"  {k}: {v}")


def _build_status_writer(args: Args) -> Any:
    """Construct a :class:`run_status.StatusWriter` for ``args``.

    The status dir defaults to a sibling of ``args.output_dir`` so the
    launcher and a foreground operator land on the same path without
    extra wiring. Always returns a writer (or a no-op stub if the import
    fails) so callers can emit unconditionally.
    """
    try:
        from experiments import run_status
    except Exception:
        # Fall back to a no-op stub so the harness is robust to missing
        # / partial repo trees on the remote side.
        class _NullWriter:
            status_dir = Path("/dev/null")

            def emit(self, *_a, **_k):  # type: ignore[no-untyped-def]
                return {}

            def emit_failure(self, *_a, **_k):  # type: ignore[no-untyped-def]
                return {}

        return _NullWriter()

    run_id = run_status.derive_run_id(
        model=args.model,
        avg_bits=args.avg_bits,
        seed=args.seed,
        n_calib=args.n_calib,
        n_eval=args.n_eval,
        smoke=args.synthetic_smoke,
        inline_corpus_smoke=args.inline_corpus_smoke,
    )
    if args.status_dir is not None:
        sd = Path(args.status_dir) / run_id
    else:
        sd = run_status.default_status_dir(args.output_dir, run_id)
    return run_status.StatusWriter(
        status_dir=sd,
        run_id=run_id,
        commit=_git_commit(override=args.git_commit_override),
        model=args.model,
        avg_bits=args.avg_bits,
        n_calib=args.n_calib,
        n_eval=args.n_eval,
        n_layers_sample=args.n_layers_sample,
    )


def run(args: Args) -> int:
    output_path = _output_path(args)
    schema_path = REPO_ROOT / "schemas" / "three_way_result.schema.json"

    print_plan(args, output_path)

    if args.dry_run:
        # Dry-run is the cheap validation path: validate plan + payload
        # shape but write *nothing* to disk (the existing contract). We
        # therefore skip status emission entirely so dry-run output dirs
        # remain pristine for downstream tooling that asserts no files
        # are written.
        try:
            payload = build_payload(args, smoke=None)
            _validate_payload(payload, schema_path)
            print("[run_three_way] dry-run: payload is schema-valid")
        except ValueError as exc:
            print(f"[run_three_way] dry-run: schema validation FAILED: {exc}",
                  file=sys.stderr)
            return 1
        return 0

    status = _build_status_writer(args)
    if args.synthetic_smoke:
        run_mode = "synthetic-smoke"
    elif args.inline_corpus_smoke:
        run_mode = "inline-corpus-smoke"
    else:
        run_mode = "full"
    status.emit(
        "start",
        message=f"run starting (mode={run_mode})",
        details={
            "output_path": str(output_path),
            "device": args.device,
            "dtype": args.dtype,
            "mode": run_mode,
            "paper_valid": run_mode == "full",
        },
    )

    if output_path.exists() and args.skip_if_exists and not args.force:
        print(
            f"[run_three_way] skip-if-exists: {output_path} already exists. "
            "Pass --force to overwrite."
        )
        status.emit(
            "success",
            message="skip-if-exists",
            details={"output_path": str(output_path)},
        )
        return 0

    try:
        if args.synthetic_smoke:
            status.emit("calibration_start", message="synthetic-smoke calibration")
            smoke = run_synthetic_smoke(args)
            status.emit(
                "calibration_end",
                details={"layer_indices": smoke.get("layer_indices", [])},
            )
            status.emit("eval_start", message="synthetic-smoke eval")
            payload = build_payload(args, smoke=smoke)
            status.emit("eval_end")
            atomic_write_json(output_path, payload, schema_path=schema_path)
            print(f"[run_three_way] synthetic-smoke: wrote {output_path}")
            status.emit(
                "success",
                message="synthetic-smoke wrote result",
                details={"output_path": str(output_path)},
            )
            return 0

        # Full HuggingFace model path (real HF model load + adapters +
        # calibration + eval). When ``--inline-corpus-smoke`` is set we
        # still go through this path; only the corpus loader is swapped.
        full = run_full_hf(args, status=status)
        payload = build_payload(args, smoke=full)
        atomic_write_json(output_path, payload, schema_path=schema_path)
        marker = "inline-corpus-smoke" if args.inline_corpus_smoke else "full"
        print(f"[run_three_way] {marker}: wrote {output_path}")
        status.emit(
            "success",
            message=f"{marker} path wrote result",
            details={
                "output_path": str(output_path),
                "mode": marker,
                "paper_valid": marker == "full",
            },
        )
        return 0
    except BaseException as exc:
        # Always write a failure artifact so the operator can inspect the
        # error even when Modal logs are unavailable. We re-raise the
        # exception so the run still surfaces a non-zero return code.
        try:
            status.emit_failure(exc)
        except Exception:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
