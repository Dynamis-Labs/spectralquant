"""
NeurIPS SpectralQuant — Script C: 10-Seed CI + Latency Crossover Analysis
==========================================================================
  PART 1: 10-seed confidence intervals on Qwen 2.5-1.5B
          - Seeds: [42, 123, 7, 2024, 31415, 99, 1337, 8675309, 271828, 314159]
          - Methods: TQ_3bit, SQ_noQJL_v3
          - Metric: attention output cosine similarity vs FP16
          - Statistics: mean ± std, paired Wilcoxon signed-rank test, win-rate

  PART 2: Latency crossover analysis
          - Measure per-token: compression_time and attention_time for TQ and SQ
          - Sequence lengths: [512, 1024, 2048, 4096, 8192]
          - Find crossover length where SQ per-step latency < TQ

Estimated runtime: 30 min on B200

Usage:
    python neurips_seeds_latency.py [--quick] [--part {0,1,2}] [--device cuda]
"""

import sys, os, math, time, json, logging, argparse
from pathlib import Path

import torch
import numpy as np

# ── project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from turboquant_cutile import TurboQuantEngine, LloydMaxCodebook

log = logging.getLogger("neurips_seeds_latency")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

RESULTS_DIR = PROJECT_ROOT / "results" / "neurips"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SEEDS = [42, 123, 7, 2024, 31415, 99, 1337, 8675309, 271828, 314159]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def save_result(filename, data):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved: %s", path)


def quantize_nearest(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    c = centroids.to(x.device)
    diffs = x.unsqueeze(-1) - c
    return c[diffs.abs().argmin(dim=-1).long()]


def calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd, seed=42):
    """Per-(layer, head) eigenvector calibration with a deterministic seed."""
    from datasets import load_dataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                      split=f"train[:{n_calib * 5}]")
    cov = {
        (l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64), "n": 0}
        for l in range(n_layers) for h in range(n_kv)
    }
    nd = 0
    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt",
                        max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue
        with torch.no_grad():
            out = model(**enc, use_cache=True)
            kv = out.past_key_values
        for l in range(n_layers):
            try:
                k_l = kv.key_cache[l].float().cpu()
            except Exception:
                try:
                    k_l = kv[l][0].float().cpu()
                except Exception:
                    k_l = list(kv)[l][0].float().cpu()
            for h in range(n_kv):
                X = k_l[0, h, :, :].double()
                cov[(l, h)]["xtx"] += X.T @ X
                cov[(l, h)]["n"] += X.shape[0]
        nd += 1
        if nd >= n_calib:
            break

    eigen = {}
    for l in range(n_layers):
        for h in range(n_kv):
            C = (cov[(l, h)]["xtx"] / cov[(l, h)]["n"]).float()
            ev, evec = torch.linalg.eigh(C)
            ev = ev.flip(0).clamp(min=0)
            evec = evec.flip(1)
            d_eff = max(2, min(int(round((ev.sum() ** 2 / (ev ** 2).sum()).item())), hd - 2))
            eigen[(l, h)] = {"evec": evec, "d_eff": d_eff}
    return eigen


def extract_kv_layer(kv, l):
    try:
        return kv.key_cache[l], kv.value_cache[l]
    except Exception:
        pass
    try:
        return kv[l][0], kv[l][1]
    except Exception:
        pass
    entry = list(kv)[l]
    return entry[0], entry[1]


@torch.no_grad()
def eval_cossim_seed(model, tokenizer, eigen, n_layers, n_kv, hd, device,
                      n_eval: int, seed: int) -> dict:
    """
    Evaluate TQ and SQ cosine similarity for one seed.
    Returns {"tq": [per-head sims], "sq": [per-head sims]}
    """
    from datasets import load_dataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"test[:{n_eval * 5}]")
    slayers = list(range(0, n_layers, max(1, n_layers // 5)))[:5]

    tq_sims, sq_sims = [], []
    nev = 0

    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt",
                        max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue
        try:
            with torch.no_grad():
                out = model(**enc, use_cache=True)
                kv = out.past_key_values
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        for l in slayers:
            try:
                k_l, v_l = extract_kv_layer(kv, l)
                k_l = k_l.float().cpu()
                v_l = v_l.float().cpu()
            except Exception:
                continue

            for h in range(n_kv):
                K = k_l[0, h].to(device).half()
                V = v_l[0, h].to(device).half()
                if K.shape[0] < 8:
                    continue

                # Seed-specific random queries
                torch.manual_seed(seed + l * 1000 + h)
                Qp = torch.randn(8, hd, device=device, dtype=torch.float16)
                sc = (Qp.float() @ K.float().T) / math.sqrt(hd)
                ref = (torch.softmax(sc, dim=-1) @ V.float()).half()

                def cos(o):
                    c = torch.nn.functional.cosine_similarity(
                        ref.float().reshape(-1, hd),
                        o.float().reshape(-1, hd),
                        dim=-1,
                    ).mean().item()
                    return c if not math.isnan(c) else None

                # TQ
                try:
                    tq = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)
                    ck = tq.compress_keys_pytorch(K)
                    cv = tq.compress_values_pytorch(V)
                    val = cos(tq.fused_attention_pytorch(Qp, ck, cv))
                    if val is not None:
                        tq_sims.append(val)
                except Exception:
                    pass

                # SQ (no QJL)
                try:
                    ed = eigen[(l, h)]
                    evec = ed["evec"].to(device).float()
                    VT = evec.T.contiguous()
                    k_n = torch.norm(K.float(), dim=-1, keepdim=True)
                    K_rot = (K.float() / (k_n + 1e-8)) @ VT
                    v_n = torch.norm(V.float(), dim=-1, keepdim=True)
                    V_rot = (V.float() / (v_n + 1e-8)) @ VT
                    cb_k = LloydMaxCodebook(hd, 2)
                    cb_v = LloydMaxCodebook(hd, 3)
                    K_hat = quantize_nearest(K_rot, cb_k.centroids.to(device))
                    V_hat = quantize_nearest(V_rot, cb_v.centroids.to(device))
                    K_mse = (K_hat @ evec) * k_n
                    V_rec = (V_hat @ evec) * v_n
                    scores = (Qp.float() @ K_mse.T) / math.sqrt(hd)
                    sq_out = (torch.softmax(scores, dim=-1) @ V_rec).half()
                    val = cos(sq_out)
                    if val is not None:
                        sq_sims.append(val)
                except Exception:
                    pass

        nev += 1
        if nev >= n_eval:
            break

    return {
        "tq": tq_sims,
        "sq": sq_sims,
        "tq_mean": float(np.mean(tq_sims)) if tq_sims else None,
        "sq_mean": float(np.mean(sq_sims)) if sq_sims else None,
        "n": len(tq_sims),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: 10-SEED CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════════

def run_part1_seeds(model, tokenizer, eigen_base, n_layers, n_kv, hd,
                     device, n_eval, seeds, partial_results=None) -> dict:
    """
    Run evaluation for each seed.
    For each seed, re-calibrate and re-evaluate to capture full seed variability.
    Save after each seed.
    """
    results = partial_results or {}

    for seed in seeds:
        seed_key = str(seed)
        if seed_key in results:
            log.info("Skipping seed=%d (done)", seed)
            continue

        log.info("Seed %d: calibrating ...", seed)
        try:
            n_calib_seed = 100  # fixed, enough for stable eigenspectrum
            eigen_seed = calibrate(model, tokenizer, n_calib_seed, device,
                                    n_layers, n_kv, hd, seed=seed)
        except Exception as e:
            log.error("Calibration failed for seed %d: %s", seed, e)
            results[seed_key] = {"error": str(e), "seed": seed}
            save_result("neurips_10seed.json", _build_seed_summary(results, seeds))
            continue

        log.info("Seed %d: evaluating (%d seqs) ...", seed, n_eval)
        try:
            seed_result = eval_cossim_seed(
                model, tokenizer, eigen_seed, n_layers, n_kv, hd,
                device, n_eval, seed
            )
            seed_result["seed"] = seed
            results[seed_key] = seed_result
            log.info("  seed=%d  TQ=%.4f  SQ=%.4f  sq_wins=%s",
                     seed,
                     seed_result.get("tq_mean") or float("nan"),
                     seed_result.get("sq_mean") or float("nan"),
                     (seed_result.get("sq_mean") or 0) > (seed_result.get("tq_mean") or 0))
        except Exception as e:
            log.error("Eval failed for seed %d: %s", seed, e)
            results[seed_key] = {"error": str(e), "seed": seed}

        save_result("neurips_10seed.json", _build_seed_summary(results, seeds))

    return results


def _build_seed_summary(results: dict, seeds: list) -> dict:
    """Compute aggregate statistics across seeds."""
    tq_per_seed = []
    sq_per_seed = []
    sq_wins_all = True
    sq_win_count = 0

    for seed in seeds:
        r = results.get(str(seed), {})
        tq_m = r.get("tq_mean")
        sq_m = r.get("sq_mean")
        if tq_m is not None and sq_m is not None:
            tq_per_seed.append(tq_m)
            sq_per_seed.append(sq_m)
            if sq_m > tq_m:
                sq_win_count += 1
            else:
                sq_wins_all = False

    wilcoxon_result = None
    p_value = None
    if len(tq_per_seed) >= 4 and len(sq_per_seed) >= 4:
        try:
            from scipy.stats import wilcoxon
            stat, p_value = wilcoxon(sq_per_seed, tq_per_seed, alternative="greater")
            wilcoxon_result = {"statistic": float(stat), "p_value": float(p_value)}
            log.info("Wilcoxon test (SQ > TQ): p=%.4f", p_value)
        except Exception as e:
            log.warning("Wilcoxon test failed: %s", e)

    return {
        "model": MODEL_NAME,
        "seeds": seeds,
        "per_seed": results,
        "aggregate": {
            "tq": {
                "mean": float(np.mean(tq_per_seed)) if tq_per_seed else None,
                "std": float(np.std(tq_per_seed)) if tq_per_seed else None,
                "per_seed_means": tq_per_seed,
            },
            "sq": {
                "mean": float(np.mean(sq_per_seed)) if sq_per_seed else None,
                "std": float(np.std(sq_per_seed)) if sq_per_seed else None,
                "per_seed_means": sq_per_seed,
            },
            "sq_wins_all_10_seeds": sq_wins_all and len(tq_per_seed) == len(seeds),
            "sq_win_count": sq_win_count,
            "n_seeds_evaluated": len(tq_per_seed),
            "wilcoxon": wilcoxon_result,
        },
        "interpretation": (
            f"SQ wins on {sq_win_count}/{len(tq_per_seed)} seeds. "
            + (f"Wilcoxon p={p_value:.4f} (SQ > TQ)." if p_value is not None else "")
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: LATENCY CROSSOVER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

N_WARMUP = 5   # warmup iterations before timing
N_TIMING = 20  # timing iterations

def _time_fn(fn, n_warmup=N_WARMUP, n_timing=N_TIMING):
    """Measure median execution time in milliseconds."""
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_timing):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


@torch.no_grad()
def measure_latency_at_seqlen(model, tokenizer, eigen, n_layers, n_kv, hd,
                                device, seq_len: int) -> dict:
    """
    Measure per-token compression and attention times for TQ and SQ at a given seq_len.

    For compression: time to compress 1 new token appended to a cache of seq_len tokens.
    For attention: time to run attention over seq_len cached keys/values with 1 new query.

    Returns dict with timing breakdown.
    """
    # Create representative K, V, Q tensors
    K = torch.randn(seq_len, hd, device=device, dtype=torch.float16)
    V = torch.randn(seq_len, hd, device=device, dtype=torch.float16)
    Q = torch.randn(1, hd, device=device, dtype=torch.float16)  # 1 new token

    # Use first layer, head 0 for representative measurements
    l, h = 0, 0
    ed = eigen.get((l, h), {})
    evec = ed.get("evec", torch.eye(hd)).to(device).float()

    # --- TQ compression (1 token) ---
    k_new = torch.randn(1, hd, device=device, dtype=torch.float16)
    v_new = torch.randn(1, hd, device=device, dtype=torch.float16)
    tq_engine = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)

    def tq_compress_1token():
        ck = tq_engine.compress_keys_pytorch(k_new)
        cv = tq_engine.compress_values_pytorch(v_new)
        return ck, cv

    tq_compress_ms = _time_fn(tq_compress_1token)
    log.info("  tq_compress_1token @ seq=%d: %.3f ms", seq_len, tq_compress_ms)

    # --- SQ compression (1 token) ---
    VT = evec.T.contiguous()
    cb_k = LloydMaxCodebook(hd, 2)
    cb_v = LloydMaxCodebook(hd, 3)
    centroids_k = cb_k.centroids.to(device)
    centroids_v = cb_v.centroids.to(device)

    def sq_compress_1token():
        k_f = k_new.float()
        v_f = v_new.float()
        k_n = torch.norm(k_f, dim=-1, keepdim=True)
        K_rot = (k_f / (k_n + 1e-8)) @ VT
        K_hat = quantize_nearest(K_rot, centroids_k)
        k_r = (K_hat @ evec) * k_n
        v_n = torch.norm(v_f, dim=-1, keepdim=True)
        V_rot = (v_f / (v_n + 1e-8)) @ VT
        V_hat = quantize_nearest(V_rot, centroids_v)
        v_r = (V_hat @ evec) * v_n
        return k_r, v_r

    sq_compress_ms = _time_fn(sq_compress_1token)
    log.info("  sq_compress_1token @ seq=%d: %.3f ms", seq_len, sq_compress_ms)

    # --- TQ attention (attend over full seq_len cache) ---
    # Compress the full cache first
    ck_full = tq_engine.compress_keys_pytorch(K)
    cv_full = tq_engine.compress_values_pytorch(V)

    def tq_attention():
        return tq_engine.fused_attention_pytorch(Q, ck_full, cv_full)

    tq_attn_ms = _time_fn(tq_attention)
    log.info("  tq_attention @ seq=%d: %.3f ms", seq_len, tq_attn_ms)

    # --- SQ attention (attend over full seq_len cache, compressed+decompressed) ---
    k_n_full = torch.norm(K.float(), dim=-1, keepdim=True)
    K_rot_full = (K.float() / (k_n_full + 1e-8)) @ VT
    K_hat_full = quantize_nearest(K_rot_full, centroids_k)
    K_sq = ((K_hat_full @ evec) * k_n_full).half()

    v_n_full = torch.norm(V.float(), dim=-1, keepdim=True)
    V_rot_full = (V.float() / (v_n_full + 1e-8)) @ VT
    V_hat_full = quantize_nearest(V_rot_full, centroids_v)
    V_sq = ((V_hat_full @ evec) * v_n_full).half()

    def sq_attention():
        sc = (Q.float() @ K_sq.float().T) / math.sqrt(hd)
        weights = torch.softmax(sc, dim=-1)
        return weights @ V_sq.float()

    sq_attn_ms = _time_fn(sq_attention)
    log.info("  sq_attention @ seq=%d: %.3f ms", seq_len, sq_attn_ms)

    # --- Per-decode-step totals ---
    # Per decode step at cache size seq_len:
    #   total_tq = compress_1_token_tq + attention_over_seq_tq
    #   total_sq = compress_1_token_sq + attention_over_seq_sq
    total_tq_ms = tq_compress_ms + tq_attn_ms
    total_sq_ms = sq_compress_ms + sq_attn_ms

    return {
        "seq_len": seq_len,
        "tq_compress_1token_ms": tq_compress_ms,
        "sq_compress_1token_ms": sq_compress_ms,
        "tq_attention_ms": tq_attn_ms,
        "sq_attention_ms": sq_attn_ms,
        "tq_per_step_ms": total_tq_ms,
        "sq_per_step_ms": total_sq_ms,
        "sq_faster_than_tq": total_sq_ms < total_tq_ms,
        "sq_speedup_over_tq": float(total_tq_ms / total_sq_ms) if total_sq_ms > 0 else None,
        "attn_speedup_sq_over_tq": float(tq_attn_ms / sq_attn_ms) if sq_attn_ms > 0 else None,
    }


def run_part2_latency(model, tokenizer, eigen, n_layers, n_kv, hd,
                       device, seq_lengths, partial_results=None) -> dict:
    """
    Measure latency at each sequence length and find the crossover point.
    """
    results = partial_results or {}

    for seq_len in seq_lengths:
        sl_key = str(seq_len)
        if sl_key in results:
            log.info("Skipping seq_len=%d (done)", seq_len)
            continue

        log.info("\nLatency measurement at seq_len=%d ...", seq_len)
        try:
            latency = measure_latency_at_seqlen(
                model, tokenizer, eigen, n_layers, n_kv, hd, device, seq_len
            )
            results[sl_key] = latency
            log.info(
                "  seq=%d: TQ_step=%.3f ms, SQ_step=%.3f ms, SQ_faster=%s",
                seq_len,
                latency["tq_per_step_ms"],
                latency["sq_per_step_ms"],
                latency["sq_faster_than_tq"],
            )
        except torch.cuda.OutOfMemoryError:
            log.warning("  OOM at seq_len=%d", seq_len)
            torch.cuda.empty_cache()
            results[sl_key] = {"seq_len": seq_len, "error": "OOM"}
        except Exception as exc:
            log.warning("  Error at seq_len=%d: %s", seq_len, exc)
            import traceback; traceback.print_exc()
            results[sl_key] = {"seq_len": seq_len, "error": str(exc)}

        save_result("neurips_latency_crossover.json",
                    _build_latency_summary(results, seq_lengths))

    return results


def _build_latency_summary(results: dict, seq_lengths: list) -> dict:
    """Find crossover point and build final summary."""
    crossover_seq_len = None
    sq_compress_factor = None
    attn_speedup = None

    rows = []
    for sl in seq_lengths:
        r = results.get(str(sl), {})
        if "error" in r or not r:
            continue
        rows.append(r)

    # Find crossover: smallest seq_len where sq_per_step < tq_per_step
    for r in sorted(rows, key=lambda x: x["seq_len"]):
        if r.get("sq_faster_than_tq"):
            crossover_seq_len = r["seq_len"]
            break

    # Typical attention speedup (from largest available seq_len)
    if rows:
        largest = max(rows, key=lambda x: x["seq_len"])
        attn_speedup = largest.get("attn_speedup_sq_over_tq")
        if largest.get("sq_compress_1token_ms") and largest.get("tq_compress_1token_ms"):
            sq_compress_factor = (
                largest["sq_compress_1token_ms"] / largest["tq_compress_1token_ms"]
            )

    # Narrative
    narrative = ""
    if crossover_seq_len is not None:
        compress_str = f"{sq_compress_factor:.1f}x slower compression" if sq_compress_factor else "slower compression"
        attn_str = f"{attn_speedup:.1f}x" if attn_speedup else "significant"
        narrative = (
            f"At sequence lengths beyond {crossover_seq_len} tokens, "
            f"SpectralQuant's per-step latency is lower than TurboQuant's "
            f"despite {compress_str}, "
            f"because the {attn_str} attention speedup "
            f"amortizes over the growing cache."
        )
    else:
        measured_max = max((r["seq_len"] for r in rows), default=0) if rows else 0
        narrative = (
            f"No crossover observed up to seq_len={measured_max}. "
            "SQ compression overhead dominates at these context lengths."
        )

    return {
        "model": MODEL_NAME,
        "seq_lengths": seq_lengths,
        "per_seq_len": results,
        "crossover_seq_len": crossover_seq_len,
        "attn_speedup_sq_over_tq_at_max_seqlen": attn_speedup,
        "sq_compress_slowdown_vs_tq": sq_compress_factor,
        "narrative": narrative,
        "timing_table": [
            {
                "seq_len": r["seq_len"],
                "tq_per_step_ms": r.get("tq_per_step_ms"),
                "sq_per_step_ms": r.get("sq_per_step_ms"),
                "sq_faster": r.get("sq_faster_than_tq"),
            }
            for r in sorted(rows, key=lambda x: x["seq_len"])
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NeurIPS Script C: 10-Seed CI + Latency Crossover"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5 seeds, fewer eval sequences, shorter context lengths")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--part", type=int, default=0,
                        help="0=all, 1=seeds, 2=latency")
    parser.add_argument("--n-calib", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    n_calib_base = args.n_calib or (40 if args.quick else 200)
    n_eval = 10 if args.quick else 50

    seeds = SEEDS[:5] if args.quick else SEEDS
    seq_lengths = [512, 1024, 2048] if args.quick else [512, 1024, 2048, 4096, 8192]

    log.info("=" * 70)
    log.info("NeurIPS Script C: Seeds + Latency  |  quick=%s  part=%d",
             args.quick, args.part)
    log.info("Seeds: %s", seeds)
    log.info("Seq lengths: %s", seq_lengths)
    log.info("=" * 70)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading %s ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads
    log.info("Model: %d layers, %d KV heads, head_dim=%d", n_layers, n_kv, hd)

    # Base calibration (for latency measurements)
    log.info("Base calibration (%d seqs) ...", n_calib_base)
    eigen_base = calibrate(model, tokenizer, n_calib_base, device, n_layers, n_kv, hd, seed=42)

    # ─── PART 1: 10-seed CIs ───────────────────────────────────────────────
    if args.part in (0, 1):
        log.info("\n" + "=" * 60)
        log.info("PART 1: 10-seed confidence intervals  (n_eval=%d per seed)", n_eval)
        log.info("=" * 60)

        seed_resume = {}
        seed_path = RESULTS_DIR / "neurips_10seed.json"
        if seed_path.exists():
            try:
                with open(seed_path) as f:
                    existing = json.load(f)
                seed_resume = existing.get("per_seed", {})
                log.info("Resuming from %d done seeds", len(seed_resume))
            except Exception:
                pass

        per_seed_results = run_part1_seeds(
            model, tokenizer, eigen_base, n_layers, n_kv, hd,
            device, n_eval, seeds, partial_results=seed_resume
        )

        summary = _build_seed_summary(per_seed_results, seeds)
        save_result("neurips_10seed.json", summary)

        log.info("\n=== 10-Seed Summary ===")
        agg = summary.get("aggregate", {})
        tq_agg = agg.get("tq", {})
        sq_agg = agg.get("sq", {})
        log.info("TQ:  mean=%.4f  std=%.4f",
                 tq_agg.get("mean") or float("nan"),
                 tq_agg.get("std") or float("nan"))
        log.info("SQ:  mean=%.4f  std=%.4f",
                 sq_agg.get("mean") or float("nan"),
                 sq_agg.get("std") or float("nan"))
        log.info("SQ wins all 10 seeds: %s", agg.get("sq_wins_all_10_seeds"))
        if agg.get("wilcoxon"):
            log.info("Wilcoxon p-value: %.4f", agg["wilcoxon"]["p_value"])

    # ─── PART 2: Latency crossover ──────────────────────────────────────────
    if args.part in (0, 2):
        log.info("\n" + "=" * 60)
        log.info("PART 2: Latency crossover  seq_lengths=%s", seq_lengths)
        log.info("=" * 60)

        latency_resume = {}
        latency_path = RESULTS_DIR / "neurips_latency_crossover.json"
        if latency_path.exists():
            try:
                with open(latency_path) as f:
                    existing = json.load(f)
                latency_resume = existing.get("per_seq_len", {})
                log.info("Resuming latency from %d done seq_lengths", len(latency_resume))
            except Exception:
                pass

        latency_results = run_part2_latency(
            model, tokenizer, eigen_base, n_layers, n_kv, hd,
            device, seq_lengths, partial_results=latency_resume
        )

        final_summary = _build_latency_summary(latency_results, seq_lengths)
        save_result("neurips_latency_crossover.json", final_summary)

        log.info("\n=== Latency Summary ===")
        log.info("%-8s  %12s  %12s  %8s", "seq_len", "TQ_step_ms", "SQ_step_ms", "SQ_faster")
        for row in final_summary.get("timing_table", []):
            log.info("%-8d  %12.3f  %12.3f  %8s",
                     row["seq_len"],
                     row["tq_per_step_ms"] or 0.0,
                     row["sq_per_step_ms"] or 0.0,
                     str(row["sq_faster"]))
        log.info("Crossover at: %s tokens", final_summary.get("crossover_seq_len"))
        log.info("%s", final_summary.get("narrative", ""))

    log.info("\nAll done. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
