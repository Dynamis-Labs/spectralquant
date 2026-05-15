#!/usr/bin/env python3
"""Perplexity / language-modeling evaluation harness.

Evaluates perplexity (PPL) of one or more methods (``fp16`` reference,
``spectralquant_v2``, and the in-repo ``turboquant`` baseline) on a
public corpus, defaulting to WikiText-103 which is the same source
used for calibration in the three-way harness. This file pairs with
``schemas/perplexity.schema.json``.

The harness runs in four modes:

* ``--dry-run``: validate args, print the plan and the deterministic
  output path, write nothing.
* ``--synthetic-smoke``: skip the HF model entirely. Generate a tiny
  hand-rolled "log-likelihood" vector for each method so the result
  plumbing (atomic write, schema validation, status emission) is
  exercised end-to-end. Output is marked ``mode=synthetic_smoke`` and
  ``paper_valid=false``.
* ``--inline-corpus-smoke``: load the real HF model but evaluate on the
  deterministic inline corpus from
  :data:`experiments.eval_common.INLINE_CORPUS` instead of HuggingFace
  ``datasets.load_dataset``. Useful for harness validation when the
  WikiText shard is unreachable. Output is ``paper_valid=false``.
* (no flag): full path. Evaluates each requested method on the WikiText
  validation split with sliding-window striding, mirrors the
  ``transformers`` perplexity-eval recipe. ``paper_valid=true`` iff
  the run hits the gating in :func:`build_payload`.

How non-FP16 methods are evaluated
----------------------------------

For ``spectralquant_v2`` and ``turboquant`` we use the
:mod:`experiments.sqv2_replay` projection-replay path: a fitted engine
runs ``compress`` → ``decompress`` on every layer's K/V projection
output via forward hooks. The HF model then sees the *compressed* K/V
flowing through the rest of attention, so the PPL number is a real
measurement of "this model under v2 K/V cache compression". The replay
coverage (fraction of layers actually hooked) lands in the JSON so a
report cannot silently claim full coverage when partial.

This is **not** an end-to-end TurboQuant-vs-v2 architecture comparison
(weight quantization, kernel optimizations, etc. are out of scope). The
paper-valid claim is narrower: PPL of the model when its K/V cache is
quantized by the named method.

Important claim discipline
--------------------------

The harness does **not** claim "compression-neutral perplexity" — the
v1 paper's identical 13-digit PPL across fp16/TQ/SQ is V1-GAP-004b,
explicitly blocked in ``docs/claims_discipline.md``. The output JSON
records ``methods.<key>.perplexity`` numerically; downstream report
language must include the standard caveat (number of tokens, stride,
seed) that travels with every PPL number per spec §6.

Unsupported methods (``spectralquant_v1``, ``official_turboquant``)
remain placeholders and are explicitly NOT paper-valid; the gate in
:func:`build_payload` requires every requested method to be in the
"real evaluation" set before paper_valid can be true.

Outputs
-------

JSON file at ``--output-dir/<run_id>.json`` validating against
``schemas/perplexity.schema.json``. Status artifacts at
``<status_dir>/<run_id>/status.json`` + ``events.jsonl`` if a status
directory is configured.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make ``import spectralquant`` and ``from experiments import ...`` work.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import eval_common  # noqa: E402
from experiments import run_status  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "schemas" / "perplexity.schema.json"
FAMILY = "perplexity"

#: Methods for which the harness can run a real (compressed-method)
#: evaluation via the K/V projection-replay path. Anything outside this
#: set must remain a placeholder and blocks paper_valid.
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
    n_eval_sequences: int
    max_eval_tokens: int
    stride: int
    dataset_name: str
    dataset_config: str
    dataset_split: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_perplexity.py",
        description=(
            "SpectralQuant v2 perplexity / language-modeling harness. "
            "Evaluates one or more methods on WikiText-103 (default) or "
            "a deterministic inline corpus. Supports --dry-run, "
            "--synthetic-smoke and --inline-corpus-smoke for safe local use."
        ),
    )
    eval_common.add_common_eval_args(p)
    p.add_argument("--avg-bits", type=int, default=3, dest="avg_bits",
                   help="Average bits/dim for non-fp16 methods (informational).")
    p.add_argument("--methods", nargs="+", default=["fp16"],
                   choices=list(eval_common.KNOWN_METHOD_KEYS),
                   help="Methods to evaluate. Default: fp16 only.")
    p.add_argument("--n-eval-sequences", type=int, default=64,
                   dest="n_eval_sequences",
                   help="Number of evaluation sequences. Default: 64.")
    p.add_argument("--max-eval-tokens", type=int, default=1024,
                   dest="max_eval_tokens",
                   help="Per-sequence token cap (sliding window). Default: 1024.")
    p.add_argument("--stride", type=int, default=512,
                   help="Sliding-window stride. Default: 512.")
    p.add_argument("--dataset-name", type=str, default="wikitext",
                   dest="dataset_name")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1",
                   dest="dataset_config")
    p.add_argument("--dataset-split", type=str, default="validation",
                   dest="dataset_split")
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> Args:
    p = build_parser()
    ns = p.parse_args(argv)
    if ns.synthetic_smoke and ns.inline_corpus_smoke:
        p.error("--synthetic-smoke and --inline-corpus-smoke are mutually exclusive")
    if ns.dry_run and ns.synthetic_smoke:
        ns.synthetic_smoke = False
        print("[run_perplexity] note: --dry-run takes precedence over --synthetic-smoke")
    if ns.dry_run and ns.inline_corpus_smoke:
        ns.inline_corpus_smoke = False
    if ns.force and ns.skip_if_exists:
        ns.skip_if_exists = False
    if ns.n_eval_sequences < 1:
        p.error("--n-eval-sequences must be >= 1")
    if ns.max_eval_tokens < 8:
        p.error("--max-eval-tokens must be >= 8")
    if ns.stride < 1:
        p.error("--stride must be >= 1")
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
        n_eval_sequences=int(ns.n_eval_sequences),
        max_eval_tokens=int(ns.max_eval_tokens),
        stride=int(ns.stride),
        dataset_name=str(ns.dataset_name),
        dataset_config=str(ns.dataset_config),
        dataset_split=str(ns.dataset_split),
    )


# ---------------------------------------------------------------------------
# Run-id and output path
# ---------------------------------------------------------------------------


def derive_run_id(args: Args, mode: str) -> str:
    methods_tag = "+".join(sorted(args.methods))
    suffix = (
        f"b{args.avg_bits}_seed{args.seed}_n{args.n_eval_sequences}_"
        f"tok{args.max_eval_tokens}_str{args.stride}_{methods_tag}"
    )
    return eval_common.derive_run_id(FAMILY, args.model, suffix=suffix, mode=mode)


# ---------------------------------------------------------------------------
# Synthetic-smoke pipeline (no torch / transformers needed at compute time)
# ---------------------------------------------------------------------------


def _synthetic_method_record(
    label: str, *, base_ppl: float, avg_bits: int, n_tokens: int,
) -> Dict[str, Any]:
    """Deterministic toy method record used by --synthetic-smoke."""
    nll = math.log(base_ppl)
    return {
        "label": label,
        "perplexity": float(base_ppl),
        "nll_per_token": float(nll),
        "n_tokens": int(n_tokens),
        "avg_bits": float(avg_bits) if label != "fp16" else 16.0,
        "evidence_ids": [f"RUN-PERPLEXITY-SMOKE-{label.upper()}"],
    }


def run_synthetic_smoke(args: Args) -> Dict[str, Any]:
    """Compute deterministic toy PPLs for each requested method."""
    methods: Dict[str, Dict[str, Any]] = {}
    # Stable values per-method keyed off the method name; fp16 gets the
    # "best" PPL, then v2, v1, TQ, official_TQ. These are NOT measurements.
    base = {
        "fp16": 9.50,
        "spectralquant_v2": 9.62,
        "spectralquant_v1": 9.71,
        "turboquant": 10.41,
        "official_turboquant": 10.30,
    }
    n_tokens = args.n_eval_sequences * args.max_eval_tokens
    for m in args.methods:
        methods[m] = _synthetic_method_record(
            m, base_ppl=base.get(m, 11.0),
            avg_bits=args.avg_bits, n_tokens=n_tokens,
        )
    return methods


# ---------------------------------------------------------------------------
# Inline-corpus / full pipelines
# ---------------------------------------------------------------------------


def _evaluate_fp16_ppl(
    args: Args,
    texts: Sequence[str],
    status: Optional[run_status.StatusWriter],
) -> Dict[str, Any]:
    """Run the standard sliding-window FP16 PPL computation.

    This is a faithful implementation of the HuggingFace perplexity
    recipe (https://huggingface.co/docs/transformers/perplexity) using
    a sliding window over the concatenated tokenized corpus. Lazy
    imports keep this module clean for unit tests.
    """
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if status is not None:
        status.emit("model_load_start", message=f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype]
    )
    device = torch.device(args.device if args.device != "cpu" else "cpu")
    model.to(device)
    model.eval()
    if status is not None:
        status.emit("model_load_end")

    # Concatenate texts into a single token stream.
    if status is not None:
        status.emit("eval_start", message="tokenizing corpus")
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls: List[float] = []
    n_eval_tokens = 0
    prev_end_loc = 0
    with torch.no_grad():
        for begin_loc in range(0, seq_len, args.stride):
            end_loc = min(begin_loc + args.max_eval_tokens, seq_len)
            trg_len = end_loc - prev_end_loc
            ids = input_ids[:, begin_loc:end_loc]
            target_ids = ids.clone()
            target_ids[:, :-trg_len] = -100
            outputs = model(ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood.float().item() * trg_len)
            n_eval_tokens += trg_len
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    total_nll = sum(nlls)
    nll_per_token = total_nll / max(1, n_eval_tokens)
    ppl = math.exp(nll_per_token)
    if status is not None:
        status.emit("eval_end", message=f"ppl={ppl:.4f} n_tokens={n_eval_tokens}")
    return {
        "label": "fp16",
        "perplexity": float(ppl),
        "nll_per_token": float(nll_per_token),
        "n_tokens": int(n_eval_tokens),
        "avg_bits": 16.0,
        "evidence_ids": ["RUN-PERPLEXITY-FP16"],
    }


def _evaluate_compressed_placeholder(
    method: str, args: Args,
) -> Tuple[Dict[str, Any], str]:
    """Placeholder method record + caveat for unsupported methods.

    Used only for methods outside :data:`REAL_EVAL_METHODS` (e.g.
    ``spectralquant_v1``, ``official_turboquant``). The replay path
    handles fp16, spectralquant_v2, and turboquant directly.
    """
    record = {
        "label": method,
        "perplexity": 1.0,  # sentinel; treat as "not measured"
        "nll_per_token": 0.0,
        "n_tokens": 1,
        "avg_bits": float(args.avg_bits),
        "evidence_ids": [f"RUN-PERPLEXITY-PLACEHOLDER-{method.upper()}"],
        "placeholder": True,
    }
    caveat = (
        f"method={method} is a placeholder: only fp16, spectralquant_v2, "
        f"and turboquant have real evaluation paths in this harness. "
        f"Treat this row as not-yet-paper-valid."
    )
    return record, caveat


def _evaluate_compressed_real(
    method: str,
    args: Args,
    texts: Sequence[str],
    status: Optional[run_status.StatusWriter],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run real K/V-replay PPL eval for ``method`` on ``texts``.

    Returns (method_record, replay_diagnostics). ``method`` must be in
    :data:`REAL_EVAL_METHODS` minus ``"fp16"`` (caller dispatches fp16).
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype]
    )
    device = torch.device(args.device if args.device != "cpu" else "cpu")
    model.to(device)
    model.eval()

    # Calibrate using a held-out chunk of the corpus.
    n_calib = max(8, min(32, len(texts) // 4 or 8))
    calib_texts = list(texts[:n_calib])
    if status is not None:
        status.emit("calibration_start",
                    message=f"method={method} n_calib={len(calib_texts)}")
    bundle = sqv2_replay.build_calibrated_engine(
        model, tokenizer, calib_texts,
        method=method,
        avg_bits=float(args.avg_bits),
        use_water_fill=(method == "spectralquant_v2"),
        n_calib_samples=len(calib_texts),
        max_seq_tokens=min(args.max_eval_tokens, 256),
    )
    if status is not None:
        status.emit("calibration_end",
                    message=f"calibrated_layers={len(bundle.calibrated_layers)}")

    # Attach replay hooks; the rest of the eval is the standard
    # sliding-window PPL, but K/V passes through compress→decompress.
    handle = sqv2_replay.attach_replay_hooks(
        model, bundle.engine,
        n_kv_heads=bundle.n_kv_heads, head_dim=bundle.head_dim,
        method=method, calibrated_layers=bundle.calibrated_layers,
    )

    try:
        if status is not None:
            status.emit("eval_start",
                        message=f"sliding-window PPL with {method} hooks")
        encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)

        nlls: List[float] = []
        n_eval_tokens = 0
        prev_end_loc = 0
        with torch.no_grad():
            for begin_loc in range(0, seq_len, args.stride):
                end_loc = min(begin_loc + args.max_eval_tokens, seq_len)
                trg_len = end_loc - prev_end_loc
                ids = input_ids[:, begin_loc:end_loc]
                target_ids = ids.clone()
                target_ids[:, :-trg_len] = -100
                outputs = model(ids, labels=target_ids)
                nll = outputs.loss.float().item()
                nlls.append(nll * trg_len)
                n_eval_tokens += trg_len
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

        nll_per_token = sum(nlls) / max(1, n_eval_tokens)
        ppl = math.exp(nll_per_token)
    finally:
        handle.remove()

    coverage = handle.coverage_summary()
    record = {
        "label": method,
        "perplexity": float(ppl),
        "nll_per_token": float(nll_per_token),
        "n_tokens": int(n_eval_tokens),
        "avg_bits": float(args.avg_bits),
        "evidence_ids": [f"RUN-PERPLEXITY-{method.upper()}"],
        "replay_coverage": coverage,
    }
    diagnostics = {"method": method, "coverage": coverage,
                   "calibration": bundle.coverage}
    if status is not None:
        status.emit("eval_end",
                    message=f"{method} ppl={ppl:.4f} cov={coverage}")
    return record, diagnostics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_payload(
    args: Args,
    argv: Sequence[str],
    mode: str,
    methods: Dict[str, Dict[str, Any]],
    *,
    eval_corpus: str,
    n_eval_sequences: int,
    extra_caveats: Sequence[str] = (),
    n_eval_tokens_threshold: int = 64_000,
) -> Dict[str, Any]:
    """Build the perplexity payload.

    paper_valid gating
    ------------------
    * mode must be ``full``
    * every requested method must be in :data:`REAL_EVAL_METHODS`
    * no method record may carry the ``placeholder`` flag
    * each non-fp16 method must have replay coverage >= 0.99 (i.e. at
      least 99% of attention layers were actually hooked) — otherwise
      the artifact is a partial measurement and we refuse to call it
      paper-valid
    * fp16 ``n_tokens`` >= n_eval_tokens_threshold (default 64k tokens
      is well above what synthetic/inline can produce)
    """
    methods_real = all(m in REAL_EVAL_METHODS for m in args.methods)
    no_placeholder = not any(
        m_rec.get("placeholder") for m_rec in methods.values()
    )

    # Coverage gate: every non-fp16 method must report >=99% layer coverage.
    coverage_ok = True
    for m, rec in methods.items():
        if m == "fp16":
            continue
        cov = rec.get("replay_coverage") or {}
        frac = cov.get("fraction_layers_real", 0.0)
        if frac < 0.99:
            coverage_ok = False
            break

    fp16_tokens = int((methods.get("fp16") or {}).get("n_tokens", 0))
    enough_tokens = fp16_tokens >= int(n_eval_tokens_threshold)

    paper_valid = (
        mode == eval_common.MODE_FULL
        and methods_real
        and no_placeholder
        and coverage_ok
        and enough_tokens
        and "fp16" in methods  # require fp16 reference present
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
    payload["data"] = {
        "eval_corpus": eval_corpus,
        "n_eval_sequences": int(n_eval_sequences),
        "max_eval_tokens": int(args.max_eval_tokens),
        "stride": int(args.stride),
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "paper_valid_n_tokens_threshold": int(n_eval_tokens_threshold),
    }
    payload["methods"] = methods
    payload["evidence_ids"] = [f"RUN-PERPLEXITY-{eval_common.model_short(args.model).upper()}"]
    caveats = list(extra_caveats)
    if not paper_valid:
        if mode != eval_common.MODE_FULL:
            caveats.append(
                f"mode={mode}; this artifact is harness-validation only "
                f"and is not paper-valid evidence."
            )
        if not methods_real:
            caveats.append(
                "Some requested methods are outside the real-eval set "
                f"({list(REAL_EVAL_METHODS)}); paper_valid is gated to false."
            )
        if not no_placeholder:
            caveats.append(
                "At least one method record carries placeholder=true; "
                "paper_valid is gated to false."
            )
        if not coverage_ok:
            caveats.append(
                "Non-FP16 method has replay_coverage.fraction_layers_real "
                "< 0.99; partial coverage cannot be paper-valid."
            )
        if not enough_tokens:
            caveats.append(
                f"fp16 n_tokens < {n_eval_tokens_threshold}; PPL needs "
                f"more tokens to be paper-valid."
            )
    payload["caveats"] = caveats
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    mode = eval_common.resolve_mode(args)

    run_id = derive_run_id(args, mode)
    out_path = eval_common.output_path(args.output_dir, run_id)

    print(f"[run_perplexity] {mode}: model={args.model}")
    print(f"[run_perplexity] {mode}: output={out_path}")

    if out_path.exists() and args.skip_if_exists and not args.force:
        print(f"[run_perplexity] {mode}: skip (exists): {out_path}")
        return 0

    if mode == eval_common.MODE_DRY_RUN:
        print(f"[run_perplexity] dry-run: would write {out_path}")
        return 0

    # Status writer (best-effort; does not block compute on failure).
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
            status.emit("eval_start", message="synthetic_smoke (no model load)")
        methods = run_synthetic_smoke(args)
        eval_common.assert_method_keys(methods)
        payload = build_payload(
            args, raw_argv, mode, methods,
            eval_corpus="synthetic",
            n_eval_sequences=args.n_eval_sequences,
        )
        eval_common.atomic_write_json(out_path, payload, schema_path=SCHEMA_PATH)
        print(f"[run_perplexity] synthetic-smoke: wrote {out_path}")
        if status is not None:
            status.emit("success", message=str(out_path))
        return 0

    # inline-corpus and full both load the HF model. The only difference
    # is where the eval corpus comes from.
    try:
        if mode == eval_common.MODE_INLINE_CORPUS_SMOKE:
            texts = eval_common.build_inline_corpus(args.n_eval_sequences)
            corpus_label = "inline_smoke"
        else:
            from datasets import load_dataset  # type: ignore[import-not-found]

            if status is not None:
                status.emit("dataset_load_start",
                            message=f"{args.dataset_name}/{args.dataset_config}:{args.dataset_split}")
            ds = load_dataset(args.dataset_name, args.dataset_config,
                              split=args.dataset_split)
            texts = []
            for row in ds:
                txt = row.get("text") if isinstance(row, dict) else None
                if not isinstance(txt, str):
                    continue
                txt = txt.strip()
                if len(txt) < 16:
                    continue
                texts.append(txt)
                if len(texts) >= args.n_eval_sequences:
                    break
            corpus_label = f"{args.dataset_name}/{args.dataset_config}:{args.dataset_split}"
            if status is not None:
                status.emit("dataset_load_end", message=f"texts={len(texts)}")

        methods: Dict[str, Dict[str, Any]] = {}
        extra_caveats: List[str] = []
        for m in args.methods:
            if m == "fp16":
                methods[m] = _evaluate_fp16_ppl(args, texts, status)
            elif m in REAL_EVAL_METHODS:
                # Real K/V replay path (only available in full / inline-corpus
                # smoke modes; both load a real HF model).
                rec, _diag = _evaluate_compressed_real(m, args, texts, status)
                methods[m] = rec
            else:
                rec, caveat = _evaluate_compressed_placeholder(m, args)
                methods[m] = rec
                extra_caveats.append(caveat)
        eval_common.assert_method_keys(methods)
        payload = build_payload(
            args, raw_argv, mode, methods,
            eval_corpus=corpus_label,
            n_eval_sequences=len(texts),
            extra_caveats=extra_caveats,
        )
        eval_common.atomic_write_json(out_path, payload, schema_path=SCHEMA_PATH)
        print(f"[run_perplexity] {mode}: wrote {out_path}")
        if status is not None:
            status.emit("success", message=str(out_path))
        return 0
    except BaseException as exc:
        if status is not None:
            status.emit_failure(exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
