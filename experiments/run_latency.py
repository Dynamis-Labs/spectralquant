#!/usr/bin/env python3
"""End-to-end latency harness.

Measures **prefill** and **per-token decode** latency, throughput
(tokens/sec), and peak memory at one or more (batch_size, context_length,
gen_tokens) operating points. Each operating point runs ``warmup_iters``
iterations to warm caches and CUDA kernels, then ``measured_iters``
timed iterations from which we report p50 and p95.

Methodology
-----------

* Timer: ``torch.cuda.Event`` with ``cuda.synchronize()`` when CUDA is
  available; otherwise ``time.perf_counter``. The chosen timer name is
  recorded in the JSON's ``timing.timer`` field.
* Prefill = single forward pass over the full ``context_length`` token
  prefix.
* Per-token decode = wall-clock time of a single autoregressive decode
  step at the *end* of the prefix, averaged across ``gen_tokens`` such
  steps. We measure each step individually and report the median per-
  token latency (not the total ``gen_tokens / t`` ratio, which dilutes
  the slowest step).
* Memory = ``torch.cuda.max_memory_allocated`` between operating points.

Three timing kinds per method
-----------------------------

* **fp16 end-to-end** (``label="fp16"``, ``end_to_end_measured=true``,
  ``production_kernel=true``): the standard HF forward + decode loop.
* **microbenchmark** (``microbenchmark=true``,
  ``microbenchmark_kind="kv_compress_decompress_round_trip"``): times
  the v2 / TurboQuant engine's compress→decompress over a synthetic
  K/V tensor. NOT end-to-end inference latency.
* **hooked replay end-to-end** (``end_to_end_measured=true``,
  ``production_kernel=false``,
  ``measurement_kind="hooked_replay_end_to_end"``): the same HF forward
  + decode loop as fp16, but with K/V projection-replay hooks attached
  so every layer's K/V passes through compress→decompress before the
  rest of attention. This is *real* end-to-end inference timing under
  K/V cache compression — but the per-layer hooks add Python-level
  overhead, so the wall-clock reflects "forward+generate with replay
  hooks", not a production-kernel implementation. Reported separately
  from fp16, never compared head-to-head as a speed claim.

CLI flags
---------

* ``--methods``: which method labels to evaluate.
* ``--include-microbench``: include the K/V compress+decompress
  microbenchmark rows (default: ``True``). Off by default? On by
  default. Distinct from end-to-end rows.
* ``--include-end-to-end-replay``: include hooked replay end-to-end
  rows for non-fp16 methods (default: ``True`` — this was the
  user-requested gap). The microbenchmark and end-to-end-replay rows
  carry distinct labels in the JSON so reports can pick exactly one
  family of numbers.

paper_valid gates
-----------------

* ``mode == full``
* ``device == cuda`` and timer is ``torch.cuda.Event``
* every requested method in ``REAL_EVAL_METHODS``
* no placeholder records
* every non-fp16 method has at least one ``end_to_end_measured=true``
  operating point — purely-microbenchmark rows are valuable harness
  evidence but cannot be the only "latency" signal in a paper-valid
  artifact

Modes mirror the other v2 harnesses (``--dry-run``, ``--synthetic-smoke``,
``--inline-corpus-smoke``).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
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

SCHEMA_PATH = REPO_ROOT / "schemas" / "latency.schema.json"
FAMILY = "latency"

REAL_EVAL_METHODS: Tuple[str, ...] = ("fp16", "spectralquant_v2", "turboquant")
END_TO_END_METHODS: Tuple[str, ...] = ("fp16",)
MICROBENCHMARK_METHODS: Tuple[str, ...] = ("spectralquant_v2", "turboquant")


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
    batch_sizes: Tuple[int, ...]
    context_lengths: Tuple[int, ...]
    gen_tokens: int
    warmup_iters: int
    measured_iters: int
    include_microbench: bool
    include_end_to_end_replay: bool


def _parse_int_list(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = tuple(int(p) for p in parts)
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_latency.py",
        description=(
            "End-to-end latency harness. fp16 is end-to-end; "
            "spectralquant_v2 / turboquant get both a K/V "
            "compress+decompress microbenchmark AND a hooked-replay "
            "end-to-end forward+decode timing."
        ),
    )
    eval_common.add_common_eval_args(p)
    p.add_argument("--avg-bits", type=int, default=3, dest="avg_bits")
    p.add_argument("--methods", nargs="+", default=["fp16"],
                   choices=list(eval_common.KNOWN_METHOD_KEYS))
    p.add_argument("--batch-sizes", type=_parse_int_list, default=(1,),
                   dest="batch_sizes")
    p.add_argument("--context-lengths", type=_parse_int_list,
                   default=(512, 1024, 2048),
                   dest="context_lengths")
    p.add_argument("--gen-tokens", type=int, default=64, dest="gen_tokens")
    p.add_argument("--warmup-iters", type=int, default=2, dest="warmup_iters")
    p.add_argument("--measured-iters", type=int, default=5,
                   dest="measured_iters")
    p.add_argument("--include-microbench", dest="include_microbench",
                   action="store_true", default=True,
                   help="Include K/V compress+decompress microbenchmark rows.")
    p.add_argument("--no-microbench", dest="include_microbench",
                   action="store_false")
    p.add_argument("--include-end-to-end-replay", dest="include_end_to_end_replay",
                   action="store_true", default=True,
                   help=("Include hooked-replay end-to-end forward+decode "
                         "rows for non-fp16 methods. Default: on."))
    p.add_argument("--no-end-to-end-replay", dest="include_end_to_end_replay",
                   action="store_false")
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
    if ns.gen_tokens < 1:
        p.error("--gen-tokens must be >= 1")
    if ns.measured_iters < 1:
        p.error("--measured-iters must be >= 1")
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
        batch_sizes=tuple(ns.batch_sizes),
        context_lengths=tuple(ns.context_lengths),
        gen_tokens=int(ns.gen_tokens),
        warmup_iters=int(ns.warmup_iters),
        measured_iters=int(ns.measured_iters),
        include_microbench=bool(ns.include_microbench),
        include_end_to_end_replay=bool(ns.include_end_to_end_replay),
    )


# ---------------------------------------------------------------------------
# Run id
# ---------------------------------------------------------------------------


def derive_run_id(args: Args, mode: str) -> str:
    methods_tag = "+".join(sorted(args.methods))
    bs_tag = "x".join(str(b) for b in args.batch_sizes)
    ctx_tag = "x".join(str(c) for c in args.context_lengths)
    flag_bits = []
    if not args.include_microbench:
        flag_bits.append("nomicro")
    if not args.include_end_to_end_replay:
        flag_bits.append("noe2erep")
    flag_tag = ("_" + "_".join(flag_bits)) if flag_bits else ""
    suffix = (
        f"b{args.avg_bits}_seed{args.seed}_bs{bs_tag}_ctx{ctx_tag}_"
        f"gen{args.gen_tokens}_wm{args.warmup_iters}_it{args.measured_iters}_"
        f"{methods_tag}{flag_tag}"
    )
    return eval_common.derive_run_id(FAMILY, args.model, suffix=suffix, mode=mode)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    if len(sv) == 1:
        return sv[0]
    idx = (len(sv) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sv[int(idx)]
    frac = idx - lo
    return sv[lo] + (sv[hi] - sv[lo]) * frac


# ---------------------------------------------------------------------------
# Synthetic-smoke pipeline
# ---------------------------------------------------------------------------


def _synthetic_op(
    method: str, bs: int, ctx: int, gen: int, n_iters: int,
) -> Dict[str, Any]:
    base_prefill = 0.05 * ctx + 1.0
    base_decode = 0.5 + 0.001 * ctx
    factor_for = {
        "fp16": 1.0,
        "spectralquant_v2": 0.62,
        "spectralquant_v1": 0.71,
        "turboquant": 0.83,
        "official_turboquant": 0.85,
    }
    f = factor_for.get(method, 0.95)
    p50_pre = base_prefill * f
    p95_pre = p50_pre * 1.10
    p50_dec = base_decode * f
    p95_dec = p50_dec * 1.20
    tps = 1000.0 / max(p50_dec, 1e-3) * bs
    return {
        "batch_size": int(bs),
        "context_length": int(ctx),
        "gen_tokens": int(gen),
        "prefill_ms_p50": float(round(p50_pre, 4)),
        "prefill_ms_p95": float(round(p95_pre, 4)),
        "decode_ms_per_token_p50": float(round(p50_dec, 4)),
        "decode_ms_per_token_p95": float(round(p95_dec, 4)),
        "tokens_per_sec_p50": float(round(tps, 4)),
        "peak_memory_mb": 0.0,
        "n_iters": int(n_iters),
    }


def run_synthetic_smoke(args: Args) -> Dict[str, Dict[str, Any]]:
    methods: Dict[str, Dict[str, Any]] = {}
    for m in args.methods:
        ops = []
        for bs in args.batch_sizes:
            for ctx in args.context_lengths:
                ops.append(_synthetic_op(
                    m, bs, ctx, args.gen_tokens, args.measured_iters,
                ))
        methods[m] = {
            "label": m,
            "operating_points": ops,
            "evidence_ids": [f"RUN-LATENCY-SMOKE-{m.upper()}"],
        }
    return methods


# ---------------------------------------------------------------------------
# Real timing — fp16 end-to-end
# ---------------------------------------------------------------------------


def _measure_end_to_end_op(
    args: Args,
    bs: int,
    ctx: int,
    *,
    model: Any,
    device: Any,
    use_cuda_events: bool,
    label_marker: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure prefill + decode for one operating point on a loaded model.

    ``label_marker`` is merged into the row (e.g. for
    ``hooked_replay_end_to_end`` we set
    ``end_to_end_measured=True, production_kernel=False, measurement_kind=...``).
    """
    import torch  # type: ignore[import-not-found]

    vocab = getattr(model.config, "vocab_size", 32000)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids = torch.randint(
        low=0, high=max(vocab - 1, 1), size=(bs, ctx), generator=g,
    ).to(device)

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prefill_times: List[float] = []
    decode_times: List[float] = []

    with torch.no_grad():
        for _ in range(max(1, args.warmup_iters)):
            out = model(input_ids, use_cache=True)
            past = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            for _step in range(min(2, args.gen_tokens)):
                out = model(next_token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_token = torch.argmax(
                    out.logits[:, -1, :], dim=-1, keepdim=True
                )
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(args.measured_iters):
            if use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = model(input_ids, use_cache=True)
                end.record()
                torch.cuda.synchronize()
                prefill_times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                out = model(input_ids, use_cache=True)
                t1 = time.perf_counter()
                prefill_times.append((t1 - t0) * 1000.0)

            past = out.past_key_values
            next_token = torch.argmax(
                out.logits[:, -1, :], dim=-1, keepdim=True
            )
            step_times: List[float] = []
            for _step in range(args.gen_tokens):
                if use_cuda_events:
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    out = model(next_token, past_key_values=past, use_cache=True)
                    e.record()
                    torch.cuda.synchronize()
                    step_times.append(s.elapsed_time(e))
                else:
                    t0 = time.perf_counter()
                    out = model(next_token, past_key_values=past, use_cache=True)
                    t1 = time.perf_counter()
                    step_times.append((t1 - t0) * 1000.0)
                past = out.past_key_values
                next_token = torch.argmax(
                    out.logits[:, -1, :], dim=-1, keepdim=True
                )
            decode_times.append(statistics.median(step_times))

    peak_mb = 0.0
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    p50_dec = _percentile(decode_times, 0.5)
    row = {
        "batch_size": int(bs),
        "context_length": int(ctx),
        "gen_tokens": int(args.gen_tokens),
        "prefill_ms_p50": float(_percentile(prefill_times, 0.5)),
        "prefill_ms_p95": float(_percentile(prefill_times, 0.95)),
        "decode_ms_per_token_p50": float(p50_dec),
        "decode_ms_per_token_p95": float(_percentile(decode_times, 0.95)),
        "tokens_per_sec_p50": float(1000.0 / max(p50_dec, 1e-6) * bs),
        "peak_memory_mb": float(round(peak_mb, 2)),
        "n_iters": int(args.measured_iters),
    }
    row.update(label_marker)
    return row


def _evaluate_fp16(
    args: Args,
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
    use_cuda_events = bool(
        args.device == "cuda" and torch.cuda.is_available()
    )
    if status is not None:
        status.emit("model_load_end")

    ops: List[Dict[str, Any]] = []
    label_marker = {
        "end_to_end_measured": True,
        "production_kernel": True,
        "measurement_kind": "fp16_end_to_end",
    }
    for bs in args.batch_sizes:
        for ctx in args.context_lengths:
            if status is not None:
                status.emit("eval_progress",
                            message=f"fp16 timing bs={bs} ctx={ctx}")
            ops.append(_measure_end_to_end_op(
                args, bs, ctx,
                model=model, device=device,
                use_cuda_events=use_cuda_events,
                label_marker=label_marker,
            ))
    return {
        "label": "fp16",
        "operating_points": ops,
        "evidence_ids": ["RUN-LATENCY-FP16"],
        "end_to_end_measured": True,
        "production_kernel": True,
    }


def _placeholder_method_record(
    method: str, args: Args, fp16_ops: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    ops = []
    for op in fp16_ops:
        op_copy = dict(op)
        op_copy["placeholder"] = True
        ops.append(op_copy)
    record = {
        "label": method,
        "operating_points": ops,
        "evidence_ids": [f"RUN-LATENCY-PLACEHOLDER-{method.upper()}"],
        "placeholder": True,
    }
    caveat = (
        f"method={method} is unsupported by this harness; only fp16 "
        f"(end-to-end) and spectralquant_v2 / turboquant (microbenchmark "
        f"+ hooked replay end-to-end) have real timing paths."
    )
    return record, caveat


# ---------------------------------------------------------------------------
# Microbenchmark: K/V compress+decompress round trip (synthetic engine)
# ---------------------------------------------------------------------------


def _microbench_compress_decompress(
    method: str,
    args: Args,
    *,
    n_kv_heads: int,
    head_dim: int,
    seq_lengths: Sequence[int],
    use_cuda_events: bool,
    status: Optional[run_status.StatusWriter],
) -> Dict[str, Any]:
    import torch  # type: ignore[import-not-found]
    from spectralquant import (  # noqa: E402
        EngineConfig, SpectralQuantEngine, TurboQuantBaseline,
    )
    from spectralquant.calibration import EigenspectralCalibrator, HeadCalibrationData  # noqa: E402

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    layer_idx = 0
    calib = EigenspectralCalibrator(max_tokens_per_layer=4096)
    for h in range(n_kv_heads):
        for ht in ("key", "value"):
            mat = torch.randn(head_dim, head_dim, generator=g)
            q, _ = torch.linalg.qr(mat)
            eig = torch.linspace(1.0, 0.01, head_dim)
            d_eff = max(2, int(head_dim * 0.4))
            hcd = HeadCalibrationData(
                layer_idx=layer_idx, head_idx=h, head_type=ht,
                eigenvalues=eig, eigenvectors=q,
                d_eff=float(d_eff), spectral_gap=None,
                var_95=int(head_dim * 0.6), var_99=int(head_dim * 0.8),
                n_samples=4096, head_dim=head_dim,
            )
            calib._calibration_data[(layer_idx, h, ht)] = hcd
    calib._is_calibrated = True

    avg_bits = float(args.avg_bits)
    if method == "spectralquant_v2":
        cfg = EngineConfig(avg_bits=avg_bits, use_water_fill=True, wf_min_bits=0)
        engine: Any = SpectralQuantEngine(calib, cfg)
        rotated_kv = {
            (layer_idx, h, ht): torch.randn(256, head_dim, generator=g)
            for h in range(n_kv_heads) for ht in ("key", "value")
        }
        engine.fit_quantizers(rotated_kv)
        round_trip_k = lambda eng, x: _round_trip_keys_v2_local(eng, x, layer_idx)
        round_trip_v = lambda eng, x: _round_trip_values_v2_local(eng, x, layer_idx)
    else:
        cfg = EngineConfig(avg_bits=avg_bits, use_water_fill=False)
        engine = TurboQuantBaseline(
            n_layers=1, n_heads=n_kv_heads, head_dim=head_dim, config=cfg,
        )
        raw_kv = {
            (layer_idx, h, ht): torch.randn(256, head_dim, generator=g)
            for h in range(n_kv_heads) for ht in ("key", "value")
        }
        engine.fit_quantizers(raw_kv)
        round_trip_k = lambda eng, x: _round_trip_keys_baseline_local(eng, x, layer_idx)
        round_trip_v = lambda eng, x: _round_trip_values_baseline_local(eng, x, layer_idx)

    device = torch.device(args.device if args.device != "cpu" else "cpu")
    if hasattr(torch, "cuda") and torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ops: List[Dict[str, Any]] = []
    for bs in args.batch_sizes:
        for ctx in seq_lengths:
            keys = torch.randn(bs, n_kv_heads, ctx, head_dim, generator=g).to(device)
            vals = torch.randn(bs, n_kv_heads, ctx, head_dim, generator=g).to(device)
            for _ in range(max(1, args.warmup_iters)):
                _ = round_trip_k(engine, keys)
                _ = round_trip_v(engine, vals)
            if use_cuda_events:
                torch.cuda.synchronize()
            step_times: List[float] = []
            for _ in range(args.measured_iters):
                if use_cuda_events:
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    _ = round_trip_k(engine, keys)
                    _ = round_trip_v(engine, vals)
                    e.record()
                    torch.cuda.synchronize()
                    step_times.append(s.elapsed_time(e))
                else:
                    t0 = time.perf_counter()
                    _ = round_trip_k(engine, keys)
                    _ = round_trip_v(engine, vals)
                    t1 = time.perf_counter()
                    step_times.append((t1 - t0) * 1000.0)
            p50 = _percentile(step_times, 0.5)
            p95 = _percentile(step_times, 0.95)
            ops.append({
                "batch_size": int(bs),
                "context_length": int(ctx),
                "gen_tokens": int(args.gen_tokens),
                "prefill_ms_p50": float(p50),
                "prefill_ms_p95": float(p95),
                "decode_ms_per_token_p50": float(p50 / max(1, ctx)),
                "decode_ms_per_token_p95": float(p95 / max(1, ctx)),
                "tokens_per_sec_p50": float(1000.0 * ctx / max(p50, 1e-6) * bs),
                "peak_memory_mb": 0.0,
                "n_iters": int(args.measured_iters),
                "microbenchmark": True,
                "microbenchmark_kind": "kv_compress_decompress_round_trip",
                "end_to_end_measured": False,
                "production_kernel": False,
                "measurement_kind": "microbenchmark_kv_round_trip",
            })
            if status is not None:
                status.emit("eval_progress",
                            message=f"micro {method} bs={bs} ctx={ctx} p50={p50:.3f}ms")

    return {
        "label": method,
        "operating_points": ops,
        "evidence_ids": [f"RUN-LATENCY-MICROBENCH-{method.upper()}"],
        "microbenchmark": True,
        "microbenchmark_kind": "kv_compress_decompress_round_trip",
        "end_to_end_measured": False,
        "production_kernel": False,
    }


def _round_trip_keys_v2_local(engine: Any, keys: Any, layer_idx: int) -> Any:
    import torch
    compressed = engine.compress_keys(keys, layer_idx)
    head_hats: List[Any] = []
    for head_idx, cv in sorted(compressed.items()):
        quant = engine._get_quantizer(layer_idx, head_idx, "key")
        k_rot_hat = quant.decompress(cv)
        k_hat = engine._key_rotation.unrotate(k_rot_hat, layer_idx, head_idx)
        head_hats.append(k_hat)
    return torch.stack(head_hats, dim=1)


def _round_trip_values_v2_local(engine: Any, values: Any, layer_idx: int) -> Any:
    cv = engine.compress_values(values, layer_idx)
    return engine.decompress_values(cv, layer_idx)


def _round_trip_keys_baseline_local(engine: Any, keys: Any, layer_idx: int) -> Any:
    ck = engine.compress_keys(keys, layer_idx)
    return engine.decompress_keys(ck, layer_idx)


def _round_trip_values_baseline_local(engine: Any, values: Any, layer_idx: int) -> Any:
    cv = engine.compress_values(values, layer_idx)
    return engine.decompress_values(cv, layer_idx)


# ---------------------------------------------------------------------------
# Hooked-replay end-to-end timing: same forward + decode loop as fp16,
# but with K/V projection-replay hooks attached.
# ---------------------------------------------------------------------------


def _evaluate_hooked_replay_end_to_end(
    method: str,
    args: Args,
    status: Optional[run_status.StatusWriter],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Time the full HF forward+decode loop with replay hooks attached.

    The result is *real end-to-end inference latency* under K/V cache
    compression — but it is **not** a production-kernel measurement,
    because every layer's K/V projection passes through a Python forward
    hook that calls into the engine's compress→decompress on CPU/GPU
    tensors with extra reshapes. We label every operating point with
    ``end_to_end_measured=True`` and ``production_kernel=False`` so a
    downstream report cannot accidentally claim "production speedup".
    """
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from experiments import sqv2_replay  # noqa: E402

    if status is not None:
        status.emit("model_load_start",
                    message=f"hooked-replay end-to-end {method} {args.model}")
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

    # Build a tiny calibration corpus from the inline default. Latency is
    # not sensitive to calibration quality — we only need quantizers
    # fitted, so the engine has codebooks before timing.
    calib_texts: List[str] = list(eval_common.INLINE_CORPUS)[:8]
    if status is not None:
        status.emit("calibration_start",
                    message=f"{method} n_calib={len(calib_texts)}")
    bundle = sqv2_replay.build_calibrated_engine(
        model, tok, calib_texts,
        method=method, avg_bits=float(args.avg_bits),
        use_water_fill=(method == "spectralquant_v2"),
        n_calib_samples=len(calib_texts),
        max_seq_tokens=256,
    )
    handle = sqv2_replay.attach_replay_hooks(
        model, bundle.engine,
        n_kv_heads=bundle.n_kv_heads, head_dim=bundle.head_dim,
        method=method, calibrated_layers=bundle.calibrated_layers,
    )
    if status is not None:
        status.emit("calibration_end",
                    message=f"calibrated_layers={len(bundle.calibrated_layers)}")

    use_cuda_events = bool(
        args.device == "cuda" and torch.cuda.is_available()
    )

    ops: List[Dict[str, Any]] = []
    label_marker = {
        "end_to_end_measured": True,
        "production_kernel": False,
        "measurement_kind": "hooked_replay_end_to_end",
    }
    try:
        for bs in args.batch_sizes:
            for ctx in args.context_lengths:
                if status is not None:
                    status.emit("eval_progress",
                                message=f"hooked-replay {method} bs={bs} ctx={ctx}")
                op = _measure_end_to_end_op(
                    args, bs, ctx,
                    model=model, device=device,
                    use_cuda_events=use_cuda_events,
                    label_marker=label_marker,
                )
                ops.append(op)
    finally:
        handle.remove()

    coverage = handle.coverage_summary()
    rec = {
        "label": method,
        "operating_points": ops,
        "evidence_ids": [f"RUN-LATENCY-E2E-REPLAY-{method.upper()}"],
        "end_to_end_measured": True,
        "production_kernel": False,
        "measurement_kind": "hooked_replay_end_to_end",
        "replay_coverage": coverage,
    }
    diag = {"method": method, "coverage": coverage,
            "calibration": bundle.coverage}
    return rec, diag


# ---------------------------------------------------------------------------
# Per-method merge: if the user asked for both microbench and end-to-end,
# we attach BOTH sets of operating points to one method record (rows are
# distinguishable via measurement_kind).
# ---------------------------------------------------------------------------


def _merge_method_records(
    method: str,
    micro_rec: Optional[Dict[str, Any]],
    e2e_rec: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if micro_rec is None and e2e_rec is None:
        return None
    if micro_rec is None:
        return e2e_rec
    if e2e_rec is None:
        return micro_rec
    # Both present — merge operating points and propagate flags.
    merged_ops = list(micro_rec["operating_points"]) + list(e2e_rec["operating_points"])
    rec: Dict[str, Any] = {
        "label": method,
        "operating_points": merged_ops,
        "evidence_ids": list(set(
            list(micro_rec.get("evidence_ids") or [])
            + list(e2e_rec.get("evidence_ids") or [])
        )),
        # method-level flags reflect at-least-one-of-kind:
        "microbenchmark": True,
        "microbenchmark_kind": "kv_compress_decompress_round_trip",
        "end_to_end_measured": True,
        "production_kernel": False,
        "measurement_kind": "microbenchmark_and_hooked_replay_end_to_end",
    }
    if "replay_coverage" in e2e_rec:
        rec["replay_coverage"] = e2e_rec["replay_coverage"]
    return rec


# ---------------------------------------------------------------------------
# Build payload + main
# ---------------------------------------------------------------------------


def build_payload(
    args: Args,
    argv: Sequence[str],
    mode: str,
    methods: Dict[str, Dict[str, Any]],
    *,
    timer_label: str,
    extra_caveats: Sequence[str] = (),
) -> Dict[str, Any]:
    methods_real = all(m in REAL_EVAL_METHODS for m in args.methods)
    no_placeholder = not any(
        rec.get("placeholder") for rec in methods.values()
    )
    # Each non-fp16 method needs at least one end-to-end op for paper_valid.
    e2e_ok = True
    for m, rec in methods.items():
        if m == "fp16":
            continue
        ops = rec.get("operating_points") or []
        has_e2e = any(op.get("end_to_end_measured") for op in ops)
        if not has_e2e:
            e2e_ok = False
            break
    # Coverage gate for hooked-replay e2e rows.
    coverage_ok = True
    for m, rec in methods.items():
        if m == "fp16":
            continue
        ops = rec.get("operating_points") or []
        has_e2e = any(op.get("end_to_end_measured") for op in ops)
        if not has_e2e:
            continue
        cov = rec.get("replay_coverage") or {}
        if cov.get("fraction_layers_real", 0.0) < 0.99:
            coverage_ok = False
            break

    paper_valid = (
        mode == eval_common.MODE_FULL
        and methods_real
        and no_placeholder
        and args.device == "cuda"
        and timer_label == "torch.cuda.Event"
        and e2e_ok
        and coverage_ok
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
    payload["timing"] = {
        "timer": timer_label,
        "warmup_iters": int(args.warmup_iters),
        "measured_iters": int(args.measured_iters),
        "cuda_synchronize": timer_label == "torch.cuda.Event",
    }
    payload["operating_points"] = [
        {"batch_size": int(b), "context_length": int(c),
         "gen_tokens": int(args.gen_tokens)}
        for b in args.batch_sizes for c in args.context_lengths
    ]
    payload["methods"] = methods
    payload["evidence_ids"] = [
        f"RUN-LATENCY-{eval_common.model_short(args.model).upper()}"
    ]
    caveats: List[str] = []
    if not paper_valid:
        if mode != eval_common.MODE_FULL:
            caveats.append(
                f"mode={mode}: harness validation only; not paper-valid."
            )
        if args.device != "cuda":
            caveats.append(
                f"device={args.device}: latency requires --device cuda."
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
        if not e2e_ok:
            caveats.append(
                "At least one non-FP16 method has no end_to_end_measured "
                "row (microbenchmark-only is not sufficient for paper-valid)."
            )
        if not coverage_ok:
            caveats.append(
                "Hooked-replay end-to-end row has replay_coverage < 0.99; "
                "partial coverage cannot be paper-valid."
            )
    if any(m in MICROBENCHMARK_METHODS for m in args.methods):
        caveats.append(
            "spectralquant_v2 / turboquant rows include MICROBENCHMARK rows "
            "(microbenchmark=true, "
            "microbenchmark_kind=kv_compress_decompress_round_trip) and "
            "HOOKED REPLAY END-TO-END rows (end_to_end_measured=true, "
            "production_kernel=false, "
            "measurement_kind=hooked_replay_end_to_end). The end-to-end "
            "rows include Python-level per-layer hook overhead and are "
            "therefore NOT a production-kernel speed claim. Compare them "
            "to fp16 only with that caveat explicit."
        )
    if timer_label == "time.perf_counter":
        caveats.append(
            "Timer is time.perf_counter (CPU clock); GPU op timings "
            "may include host-side scheduling overhead. Use "
            "torch.cuda.Event for paper-valid GPU latency."
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

    print(f"[run_latency] {mode}: model={args.model}")
    print(f"[run_latency] {mode}: output={out_path}")

    if out_path.exists() and args.skip_if_exists and not args.force:
        print(f"[run_latency] {mode}: skip (exists): {out_path}")
        return 0

    if mode == eval_common.MODE_DRY_RUN:
        print(f"[run_latency] dry-run: would write {out_path}")
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
        payload = build_payload(
            args, raw_argv, mode, methods, timer_label="synthetic",
        )
        eval_common.atomic_write_json(out_path, payload, schema_path=SCHEMA_PATH)
        print(f"[run_latency] synthetic-smoke: wrote {out_path}")
        if status is not None:
            status.emit("success", message=str(out_path))
        return 0

    try:
        import torch  # type: ignore[import-not-found]

        use_cuda_events = (
            args.device == "cuda" and torch.cuda.is_available()
        )
        timer_label = "torch.cuda.Event" if use_cuda_events else "time.perf_counter"

        methods: Dict[str, Dict[str, Any]] = {}
        extra_caveats: List[str] = []
        fp16_record: Optional[Dict[str, Any]] = None
        if "fp16" in args.methods:
            fp16_record = _evaluate_fp16(args, status)
            methods["fp16"] = fp16_record

        # Resolve dims for microbenchmarks.
        if any(m in MICROBENCHMARK_METHODS for m in args.methods):
            if fp16_record is not None:
                from transformers import AutoConfig  # type: ignore[import-not-found]
                cfg = AutoConfig.from_pretrained(args.model)
                n_q = getattr(cfg, "num_attention_heads", 8)
                n_kv = getattr(cfg, "num_key_value_heads", n_q) or n_q
                hidden = getattr(cfg, "hidden_size", n_q * 64)
                head_dim = hidden // n_q
            else:
                n_kv = 8
                head_dim = 64

            for m in args.methods:
                if m not in MICROBENCHMARK_METHODS:
                    continue
                micro_rec: Optional[Dict[str, Any]] = None
                e2e_rec: Optional[Dict[str, Any]] = None
                if args.include_microbench:
                    micro_rec = _microbench_compress_decompress(
                        m, args,
                        n_kv_heads=int(n_kv), head_dim=int(head_dim),
                        seq_lengths=args.context_lengths,
                        use_cuda_events=use_cuda_events,
                        status=status,
                    )
                if args.include_end_to_end_replay:
                    e2e_rec, _diag = _evaluate_hooked_replay_end_to_end(
                        m, args, status,
                    )
                merged = _merge_method_records(m, micro_rec, e2e_rec)
                if merged is not None:
                    methods[m] = merged

        # Truly unsupported methods: emit placeholder.
        for m in args.methods:
            if m in methods:
                continue
            if fp16_record is not None:
                rec, caveat = _placeholder_method_record(
                    m, args, fp16_record["operating_points"],
                )
            else:
                ops = []
                for bs in args.batch_sizes:
                    for ctx in args.context_lengths:
                        ops.append(_synthetic_op(
                            m, bs, ctx, args.gen_tokens, args.measured_iters,
                        ))
                rec = {
                    "label": m,
                    "operating_points": ops,
                    "evidence_ids": [f"RUN-LATENCY-PLACEHOLDER-{m.upper()}"],
                    "placeholder": True,
                }
                caveat = (
                    f"method={m}: unsupported and no fp16 reference; "
                    f"row is fully synthetic."
                )
            methods[m] = rec
            extra_caveats.append(caveat)

        eval_common.assert_method_keys(methods)
        payload = build_payload(
            args, raw_argv, mode, methods,
            timer_label=timer_label, extra_caveats=extra_caveats,
        )
        eval_common.atomic_write_json(out_path, payload, schema_path=SCHEMA_PATH)
        print(f"[run_latency] {mode}: wrote {out_path}")
        if status is not None:
            status.emit("success", message=str(out_path))
        return 0
    except BaseException as exc:
        if status is not None:
            status.emit_failure(exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
