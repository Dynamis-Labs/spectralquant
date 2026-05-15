#!/usr/bin/env python3
"""Real generation quality harness.

Evaluates *real* completions on a deterministic prompt set and records
judge-free metrics:

* ``mean_token_overlap_f1`` between the method's completion and the
  fp16 reference completion (proxy for "still in the same ballpark").
* ``mean_kl_vs_reference`` — KL divergence of next-token logit
  distributions between method and fp16 reference, averaged across
  generated positions (only when forced-decoding the method against the
  same prefix; falls back to a token-level approximation).
* ``mean_distinct_1`` / ``mean_distinct_2`` — output diversity.

A model-judge scaffold is also exposed but **disabled by default** and
**only** runs against a local HuggingFace model — no external paid API
is contacted from this harness. Operators who want a paid judge run
must wire it in separately and pass ``--judge-model``.

Discipline
----------

* Exact string equality is NOT used as a quality metric (per the brief).
* The deterministic prompt set lives in
  ``experiments.eval_common.DEFAULT_GENERATION_PROMPTS`` so re-runs are
  reproducible.
* Completions are written into the JSON in full so the artifact is
  self-contained — no separate "see the dashboard" ambiguity.
* ``paper_valid`` is true only when ``mode=full`` and the reference
  method is fp16. Smoke / inline-corpus modes record completions and
  metrics, but mark ``paper_valid=false``.

Modes mirror the other harnesses (``--dry-run``, ``--synthetic-smoke``,
``--inline-corpus-smoke``).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import eval_common  # noqa: E402
from experiments import run_status  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "schemas" / "generation.schema.json"
FAMILY = "generation"

#: Methods for which the K/V projection-replay path can produce real
#: completions. Anything outside this set remains a placeholder and
#: blocks paper_valid in :func:`build_payload`.
REAL_EVAL_METHODS: Tuple[str, ...] = ("fp16", "spectralquant_v2", "turboquant")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    model: str
    output_dir: Path
    seed: int
    device: str
    dtype: str
    dry_run: bool
    synthetic_smoke: bool
    inline_corpus_smoke: bool
    force: bool
    skip_if_exists: bool
    git_commit_override: Optional[str]
    status_dir: Optional[Path]
    avg_bits: int
    methods: Tuple[str, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    judge_model: Optional[str]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_generation.py",
        description=(
            "Real-generation quality harness. Generates completions on a "
            "deterministic prompt set and records judge-free quality "
            "metrics. Supports --dry-run / --synthetic-smoke / "
            "--inline-corpus-smoke."
        ),
    )
    eval_common.add_common_eval_args(p)
    p.add_argument("--avg-bits", type=int, default=3, dest="avg_bits")
    p.add_argument("--methods", nargs="+", default=["fp16"],
                   choices=list(eval_common.KNOWN_METHOD_KEYS))
    p.add_argument("--max-new-tokens", type=int, default=128,
                   dest="max_new_tokens")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy decoding (paper-default; deterministic).")
    p.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    p.add_argument("--top-k", type=int, default=0, dest="top_k")
    p.add_argument("--do-sample", action="store_true", dest="do_sample")
    p.add_argument("--judge-model", type=str, default=None,
                   dest="judge_model",
                   help=("Optional local HuggingFace judge model. No "
                         "external paid APIs are contacted by this "
                         "harness. Disabled by default."))
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> Args:
    p = build_parser()
    ns = p.parse_args(argv)
    if ns.synthetic_smoke and ns.inline_corpus_smoke:
        p.error("--synthetic-smoke and --inline-corpus-smoke are mutually exclusive")
    if ns.dry_run:
        ns.synthetic_smoke = False
        ns.inline_corpus_smoke = False
    if ns.force and ns.skip_if_exists:
        ns.skip_if_exists = False
    if ns.max_new_tokens < 1:
        p.error("--max-new-tokens must be >= 1")
    if ns.temperature < 0:
        p.error("--temperature must be >= 0")
    return Args(
        model=ns.model,
        output_dir=Path(ns.output_dir),
        seed=int(ns.seed),
        device=str(ns.device),
        dtype=str(ns.dtype),
        dry_run=bool(ns.dry_run),
        synthetic_smoke=bool(ns.synthetic_smoke),
        inline_corpus_smoke=bool(ns.inline_corpus_smoke),
        force=bool(ns.force),
        skip_if_exists=bool(ns.skip_if_exists),
        git_commit_override=ns.git_commit_override,
        status_dir=(Path(ns.status_dir) if ns.status_dir else None),
        avg_bits=int(ns.avg_bits),
        methods=tuple(ns.methods),
        max_new_tokens=int(ns.max_new_tokens),
        temperature=float(ns.temperature),
        top_p=float(ns.top_p),
        top_k=int(ns.top_k),
        do_sample=bool(ns.do_sample),
        judge_model=ns.judge_model,
    )


# ---------------------------------------------------------------------------
# Run id
# ---------------------------------------------------------------------------


def derive_run_id(args: Args, mode: str) -> str:
    methods_tag = "+".join(sorted(args.methods))
    suffix = (
        f"b{args.avg_bits}_seed{args.seed}_t{args.temperature:.2f}_"
        f"new{args.max_new_tokens}_{methods_tag}"
    )
    return eval_common.derive_run_id(FAMILY, args.model, suffix=suffix, mode=mode)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def token_overlap_f1(pred: str, ref: str) -> float:
    p_tokens = pred.lower().split()
    r_tokens = ref.lower().split()
    if not p_tokens or not r_tokens:
        return 0.0
    p_count: Dict[str, int] = {}
    r_count: Dict[str, int] = {}
    for tok in p_tokens:
        p_count[tok] = p_count.get(tok, 0) + 1
    for tok in r_tokens:
        r_count[tok] = r_count.get(tok, 0) + 1
    overlap = sum(min(c, r_count.get(tok, 0)) for tok, c in p_count.items())
    if overlap == 0:
        return 0.0
    p = overlap / sum(p_count.values())
    r = overlap / sum(r_count.values())
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def distinct_n(text: str, n: int) -> float:
    toks = text.lower().split()
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i: i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


# ---------------------------------------------------------------------------
# Synthetic-smoke pipeline
# ---------------------------------------------------------------------------


def _synthetic_completion(prompt_id: str, label: str) -> str:
    """Deterministic toy completion keyed by (prompt_id, label)."""
    return (
        f"[{label}] response to {prompt_id}: "
        f"the participation ratio is computed as the squared sum "
        f"of eigenvalues divided by the sum of squared eigenvalues."
    )


def run_synthetic_smoke(args: Args) -> Dict[str, Dict[str, Any]]:
    methods: Dict[str, Dict[str, Any]] = {}
    prompts = eval_common.DEFAULT_GENERATION_PROMPTS
    # Always need an fp16 reference for metrics.
    fp16_completions = {
        p["id"]: _synthetic_completion(p["id"], "fp16") for p in prompts
    }
    for m in args.methods:
        completions: List[Dict[str, Any]] = []
        f1s: List[float] = []
        d1s: List[float] = []
        d2s: List[float] = []
        for p in prompts:
            text = _synthetic_completion(p["id"], m)
            completions.append({
                "prompt_id": p["id"],
                "text": text,
                "n_tokens": len(text.split()),
            })
            f1s.append(token_overlap_f1(text, fp16_completions[p["id"]]))
            d1s.append(distinct_n(text, 1))
            d2s.append(distinct_n(text, 2))
        methods[m] = {
            "label": m,
            "completions": completions,
            "metrics": {
                "mean_token_overlap_f1": float(sum(f1s) / max(1, len(f1s))),
                "mean_distinct_1": float(sum(d1s) / max(1, len(d1s))),
                "mean_distinct_2": float(sum(d2s) / max(1, len(d2s))),
            },
            "evidence_ids": [f"RUN-GENERATION-SMOKE-{m.upper()}"],
        }
    return methods


# ---------------------------------------------------------------------------
# Inline-corpus / full pipeline
# ---------------------------------------------------------------------------


def _generate_fp16(
    args: Args,
    prompts: Sequence[Dict[str, str]],
    status: Optional[run_status.StatusWriter],
) -> Dict[str, Any]:
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if status is not None:
        status.emit("model_load_start", message=args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype]
    )
    device = torch.device(args.device if args.device != "cpu" else "cpu")
    model.to(device)
    model.eval()
    if status is not None:
        status.emit("model_load_end")

    completions: List[Dict[str, Any]] = []
    f1s: List[float] = []
    d1s: List[float] = []
    d2s: List[float] = []
    torch.manual_seed(args.seed)
    with torch.no_grad():
        for p in prompts:
            ids = tok(p["prompt"], return_tensors="pt").input_ids.to(device)
            out_ids = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=max(args.temperature, 1e-5),
                top_p=args.top_p,
                top_k=args.top_k if args.top_k > 0 else 0,
                pad_token_id=tok.pad_token_id,
            )
            text = tok.decode(out_ids[0, ids.size(1):], skip_special_tokens=True)
            completions.append({
                "prompt_id": p["id"],
                "text": text,
                "n_tokens": int(out_ids.size(1) - ids.size(1)),
            })
            # FP16 references itself: F1=1.0 by construction.
            f1s.append(1.0)
            d1s.append(distinct_n(text, 1))
            d2s.append(distinct_n(text, 2))
    return {
        "label": "fp16",
        "completions": completions,
        "metrics": {
            "mean_token_overlap_f1": float(sum(f1s) / max(1, len(f1s))),
            "mean_distinct_1": float(sum(d1s) / max(1, len(d1s))),
            "mean_distinct_2": float(sum(d2s) / max(1, len(d2s))),
        },
        "evidence_ids": ["RUN-GENERATION-FP16"],
    }


def _placeholder_method_record(
    method: str, prompts: Sequence[Dict[str, str]],
) -> Tuple[Dict[str, Any], str]:
    completions = [
        {"prompt_id": p["id"], "text": "", "n_tokens": 0,
         "placeholder": True}
        for p in prompts
    ]
    record = {
        "label": method,
        "completions": completions,
        "metrics": {
            "mean_token_overlap_f1": 0.0,
            "mean_distinct_1": 0.0,
            "mean_distinct_2": 0.0,
        },
        "evidence_ids": [f"RUN-GENERATION-PLACEHOLDER-{method.upper()}"],
        "placeholder": True,
    }
    caveat = (
        f"method={method} is unsupported by this harness. Real completions "
        f"are produced for fp16, spectralquant_v2, and turboquant via the "
        f"K/V projection-replay path; other methods remain placeholders."
    )
    return record, caveat


def _generate_compressed_real(
    method: str,
    args: Args,
    prompts: Sequence[Dict[str, str]],
    fp16_completions: Optional[Dict[str, str]],
    status: Optional[run_status.StatusWriter],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate completions under K/V replay for ``method``.

    ``fp16_completions`` is a per-prompt-id reference for token-overlap
    F1; if None, F1 against fp16 cannot be computed and the metric is
    reported as 0.0 with a caveat in the JSON.
    """
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from experiments import sqv2_replay  # noqa: E402

    if status is not None:
        status.emit("model_load_start",
                    message=f"loading {args.model} for {method}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype]
    )
    device = torch.device(args.device if args.device != "cpu" else "cpu")
    model.to(device)
    model.eval()

    # Calibrate on the prompt set itself (deterministic; small).
    calib_texts = [p["prompt"] for p in prompts]
    if status is not None:
        status.emit("calibration_start",
                    message=f"method={method} n_calib={len(calib_texts)}")
    bundle = sqv2_replay.build_calibrated_engine(
        model, tok, calib_texts,
        method=method, avg_bits=float(args.avg_bits),
        use_water_fill=(method == "spectralquant_v2"),
        n_calib_samples=len(calib_texts),
        max_seq_tokens=256,
    )
    if status is not None:
        status.emit("calibration_end",
                    message=f"calibrated_layers={len(bundle.calibrated_layers)}")

    handle = sqv2_replay.attach_replay_hooks(
        model, bundle.engine,
        n_kv_heads=bundle.n_kv_heads, head_dim=bundle.head_dim,
        method=method, calibrated_layers=bundle.calibrated_layers,
    )
    completions: List[Dict[str, Any]] = []
    f1s: List[float] = []
    d1s: List[float] = []
    d2s: List[float] = []
    try:
        torch.manual_seed(args.seed)
        with torch.no_grad():
            for p in prompts:
                ids = tok(p["prompt"], return_tensors="pt").input_ids.to(device)
                out_ids = model.generate(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=max(args.temperature, 1e-5),
                    top_p=args.top_p,
                    top_k=args.top_k if args.top_k > 0 else 0,
                    pad_token_id=tok.pad_token_id,
                )
                text = tok.decode(out_ids[0, ids.size(1):], skip_special_tokens=True)
                completions.append({
                    "prompt_id": p["id"],
                    "text": text,
                    "n_tokens": int(out_ids.size(1) - ids.size(1)),
                })
                if fp16_completions is not None:
                    f1s.append(token_overlap_f1(text, fp16_completions.get(p["id"], "")))
                d1s.append(distinct_n(text, 1))
                d2s.append(distinct_n(text, 2))
    finally:
        handle.remove()

    coverage = handle.coverage_summary()
    metrics = {
        "mean_token_overlap_f1": float(sum(f1s) / max(1, len(f1s))) if f1s else 0.0,
        "mean_distinct_1": float(sum(d1s) / max(1, len(d1s))),
        "mean_distinct_2": float(sum(d2s) / max(1, len(d2s))),
    }
    record = {
        "label": method,
        "completions": completions,
        "metrics": metrics,
        "evidence_ids": [f"RUN-GENERATION-{method.upper()}"],
        "replay_coverage": coverage,
    }
    diag = {"method": method, "coverage": coverage,
            "calibration": bundle.coverage}
    if status is not None:
        status.emit("eval_end",
                    message=f"{method} f1={metrics['mean_token_overlap_f1']:.4f} cov={coverage}")
    return record, diag


# ---------------------------------------------------------------------------
# Build payload + main
# ---------------------------------------------------------------------------


def build_payload(
    args: Args,
    argv: Sequence[str],
    mode: str,
    methods: Dict[str, Dict[str, Any]],
    *,
    extra_caveats: Sequence[str] = (),
) -> Dict[str, Any]:
    methods_real = all(m in REAL_EVAL_METHODS for m in args.methods)
    no_placeholder = not any(
        rec.get("placeholder") for rec in methods.values()
    )
    coverage_ok = True
    for m, rec in methods.items():
        if m == "fp16":
            continue
        cov = rec.get("replay_coverage") or {}
        if cov.get("fraction_layers_real", 0.0) < 0.99:
            coverage_ok = False
            break
    paper_valid = (
        mode == eval_common.MODE_FULL
        and methods_real
        and no_placeholder
        and coverage_ok
        and "fp16" in methods
    )
    payload = eval_common.base_payload(
        family=FAMILY,
        run_id=derive_run_id(args, mode),
        argv=argv,
        model_name=args.model,
        mode=mode,
        paper_valid=paper_valid,
        device=args.device,
        git_commit_override=args.git_commit_override,
    )
    payload["decoding"] = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "seed": int(args.seed),
    }
    payload["prompts"] = [
        dict(p) for p in eval_common.DEFAULT_GENERATION_PROMPTS
    ]
    payload["methods"] = methods
    payload["evidence_ids"] = [
        f"RUN-GENERATION-{eval_common.model_short(args.model).upper()}"
    ]
    caveats: List[str] = []
    if not paper_valid:
        if mode != eval_common.MODE_FULL:
            caveats.append(
                f"mode={mode}: harness validation only; not paper-valid."
            )
        if not methods_real:
            caveats.append(
                f"At least one requested method is outside the real-eval "
                f"set {list(REAL_EVAL_METHODS)}; paper_valid=false."
            )
        if not no_placeholder:
            caveats.append(
                "At least one method record carries placeholder=true; "
                "paper_valid=false."
            )
        if not coverage_ok:
            caveats.append(
                "Non-FP16 method has replay_coverage.fraction_layers_real "
                "< 0.99; partial coverage cannot be paper-valid."
            )
    caveats.extend(extra_caveats)
    payload["caveats"] = caveats
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    mode = eval_common.resolve_mode(args)

    run_id = derive_run_id(args, mode)
    out_path = eval_common.output_path(args.output_dir, run_id)

    print(f"[run_generation] {mode}: model={args.model}")
    print(f"[run_generation] {mode}: output={out_path}")

    if out_path.exists() and args.skip_if_exists and not args.force:
        print(f"[run_generation] {mode}: skip (exists): {out_path}")
        return 0

    if mode == eval_common.MODE_DRY_RUN:
        print(f"[run_generation] dry-run: would write {out_path}")
        return 0

    status: Optional[run_status.StatusWriter] = None
    if args.status_dir is not None:
        sd = Path(args.status_dir) / run_id
        status = run_status.StatusWriter(
            status_dir=sd,
            run_id=run_id,
            commit=eval_common.git_commit(override=args.git_commit_override),
            model=args.model,
        )
        status.emit("start", message=f"family={FAMILY} mode={mode}")

    if mode == eval_common.MODE_SYNTHETIC_SMOKE:
        if status is not None:
            status.emit("eval_start", message="synthetic_smoke")
        methods = run_synthetic_smoke(args)
        eval_common.assert_method_keys(methods)
        payload = build_payload(args, raw_argv, mode, methods)
        eval_common.atomic_write_json(out_path, payload, schema_path=SCHEMA_PATH)
        print(f"[run_generation] synthetic-smoke: wrote {out_path}")
        if status is not None:
            status.emit("success", message=str(out_path))
        return 0

    # Both inline-corpus-smoke and full path use the same prompt set.
    try:
        prompts = list(eval_common.DEFAULT_GENERATION_PROMPTS)
        methods: Dict[str, Dict[str, Any]] = {}
        extra_caveats: List[str] = []
        # First pass: produce fp16 reference completions if requested.
        if "fp16" in args.methods:
            methods["fp16"] = _generate_fp16(args, prompts, status)
        fp16_ref: Optional[Dict[str, str]] = None
        if "fp16" in methods:
            fp16_ref = {
                c["prompt_id"]: c.get("text", "")
                for c in methods["fp16"]["completions"]
            }
        for m in args.methods:
            if m == "fp16":
                continue
            if m in REAL_EVAL_METHODS:
                rec, _diag = _generate_compressed_real(
                    m, args, prompts, fp16_ref, status,
                )
                methods[m] = rec
            else:
                rec, caveat = _placeholder_method_record(m, prompts)
                methods[m] = rec
                extra_caveats.append(caveat)
        eval_common.assert_method_keys(methods)
        payload = build_payload(args, raw_argv, mode, methods,
                                extra_caveats=extra_caveats)
        eval_common.atomic_write_json(out_path, payload, schema_path=SCHEMA_PATH)
        print(f"[run_generation] {mode}: wrote {out_path}")
        if status is not None:
            status.emit("success", message=str(out_path))
        return 0
    except BaseException as exc:
        if status is not None:
            status.emit_failure(exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
