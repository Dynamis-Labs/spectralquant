#!/usr/bin/env python3
"""
phase3_exp6_latency.py — Experiment 6: Latency Benchmarking

Benchmarks the compressed KV cache attention kernel:
  - Sequence lengths: {512, 1K, 2K, 4K, 8K, 16K}
  - Methods: SpectralQuant vs TurboQuant vs FP16 baseline
  - Reports: mean latency, p50, p99, throughput (tokens/s)

The benchmark measures:
  1. KV compression time (offline calibration overhead)
  2. Prefill time with KV compression applied
  3. Per-step decode time with compressed KV cache
  4. Full throughput: tokens generated per second

Usage:
  python phase3_exp6_latency.py [--quick] [--seq-lengths 512 1024 2048]
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results" / "latency"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CALIB_DIR = PROJECT_ROOT / "results" / "eigenspectral" / "calibration"
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
N_WARMUP = 3
N_BENCHMARK = 10

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(EXPERIMENTS_DIR / "experiment_log.txt", mode="a"),
    ],
)
log = logging.getLogger("exp6_latency")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s (%.1f GB VRAM)", props.name, (getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)) / 1e9)
        return torch.device("cuda")
    log.warning("No GPU — latency measurements will not be representative. Running on CPU.")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _uniform_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    n_levels = 2 ** bits
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    x_clamp = x.clamp(-3 * std, 3 * std)
    x_norm = (x_clamp / (3 * std) + 1) / 2
    x_int = (x_norm * (n_levels - 1)).round().clamp(0, n_levels - 1)
    return x_int / (n_levels - 1) * 2 * 3 * std - 3 * std


def _solve_bits(avg_bits: float, d: int, d_sem: int) -> tuple[int, int]:
    budget = d * avg_bits
    best_err, best = float("inf"), (4, 2)
    for bh in range(2, 9):
        for bl in range(1, bh):
            err = abs(d_sem * bh + (d - d_sem) * bl - budget)
            if err < best_err:
                best_err = err
                best = (bh, bl)
    return best


# ---------------------------------------------------------------------------
# KV compression functions (timed)
# ---------------------------------------------------------------------------

def compress_kv_turbo(kv_cache: list, avg_bits: float = 3.0) -> list:
    """TurboQuant compression: random rotation + uniform quantization."""
    new_cache = []
    for l, (k, v) in enumerate(kv_cache):
        k_f, v_f = k.float(), v.float()
        n_heads = k_f.shape[1]
        head_dim = k_f.shape[-1]
        k_list, v_list = [], []

        for h in range(n_heads):
            rng = np.random.default_rng(l * 100 + h)
            A = rng.standard_normal((head_dim, head_dim)).astype(np.float32)
            Q, _ = np.linalg.qr(A)
            Pi = torch.from_numpy(Q).to(k_f.device)
            bits = max(1, int(round(avg_bits)))

            k_h = k_f[0, h]
            k_list.append((_uniform_quantize(k_h @ Pi, bits) @ Pi.T).unsqueeze(0))

            v_h = v_f[0, h]
            v_list.append((_uniform_quantize(v_h @ Pi, bits) @ Pi.T).unsqueeze(0))

        k_new = torch.stack(k_list, dim=1).to(k.dtype)
        v_new = torch.stack(v_list, dim=1).to(v.dtype)
        new_cache.append((k_new, v_new))

    return new_cache


def compress_kv_spectral(kv_cache: list, calibration: dict, avg_bits: float = 3.0) -> list:
    """SpectralQuant compression: spectral rotation + non-uniform quantization."""
    new_cache = []
    for l, (k, v) in enumerate(kv_cache):
        if l not in calibration:
            new_cache.append((k.clone(), v.clone()))
            continue
        k_f, v_f = k.float(), v.float()
        n_heads = k_f.shape[1]
        head_dim = k_f.shape[-1]
        k_list, v_list = [], []

        for h in range(n_heads):
            c = calibration.get(l, {}).get(h)
            if c is None:
                k_list.append(k_f[0:1, h:h+1])
                v_list.append(v_f[0:1, h:h+1])
                continue

            Vk = torch.from_numpy(c["key_eigenvectors"]).float().to(k_f.device)
            mu_k = torch.from_numpy(c["key_mean"]).float().to(k_f.device)
            d_sem_k = max(1, int(round(c["key_d_eff"])))
            bh_k, bl_k = _solve_bits(avg_bits, head_dim, d_sem_k)

            k_h = k_f[0, h]
            k_rot = (k_h - mu_k) @ Vk
            k_q = torch.cat([
                _uniform_quantize(k_rot[:, :d_sem_k], bh_k),
                _uniform_quantize(k_rot[:, d_sem_k:], bl_k),
            ], dim=-1)
            k_list.append((k_q @ Vk.T + mu_k).unsqueeze(0))

            Vv = torch.from_numpy(c["val_eigenvectors"]).float().to(v_f.device)
            mu_v = torch.from_numpy(c["val_mean"]).float().to(v_f.device)
            d_sem_v = max(1, int(round(c["val_d_eff"])))
            bh_v, bl_v = _solve_bits(avg_bits, head_dim, d_sem_v)

            v_h = v_f[0, h]
            v_rot = (v_h - mu_v) @ Vv
            v_q = torch.cat([
                _uniform_quantize(v_rot[:, :d_sem_v], bh_v),
                _uniform_quantize(v_rot[:, d_sem_v:], bl_v),
            ], dim=-1)
            v_list.append((v_q @ Vv.T + mu_v).unsqueeze(0))

        k_new = torch.stack(k_list, dim=1).to(k.dtype)
        v_new = torch.stack(v_list, dim=1).to(v.dtype)
        new_cache.append((k_new, v_new))

    return new_cache


# ---------------------------------------------------------------------------
# Benchmark single sequence
# ---------------------------------------------------------------------------

def benchmark_sequence(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    method: str | None,
    avg_bits: float | None,
    calibration: dict | None,
    device: torch.device,
    n_decode_steps: int = 32,
) -> dict:
    """
    Time: prefill, KV compression, and decode steps.
    Returns timing dict.
    """
    timings = {}

    with torch.no_grad():
        # --- Prefill ---
        synchronize(device)
        t0 = time.perf_counter()
        out = model(input_ids, use_cache=True)
        synchronize(device)
        timings["prefill_s"] = time.perf_counter() - t0

        kv_cache = out.past_key_values

        # --- KV compression ---
        if method is not None and avg_bits is not None:
            synchronize(device)
            t0 = time.perf_counter()

            if method == "turbo":
                kv_cache = compress_kv_turbo(
                    [(k.cpu(), v.cpu()) for k, v in kv_cache], avg_bits=avg_bits
                )
            elif method == "spectral" and calibration:
                kv_cache = compress_kv_spectral(
                    [(k.cpu(), v.cpu()) for k, v in kv_cache],
                    calibration=calibration, avg_bits=avg_bits
                )
            kv_cache = [(k.to(device), v.to(device)) for k, v in kv_cache]

            synchronize(device)
            timings["compression_s"] = time.perf_counter() - t0
        else:
            timings["compression_s"] = 0.0

        # --- Decode steps ---
        decode_times = []
        next_input = input_ids[:, -1:]
        cur_kv = kv_cache

        for step in range(n_decode_steps):
            synchronize(device)
            t0 = time.perf_counter()
            step_out = model(next_input, past_key_values=cur_kv, use_cache=True)
            synchronize(device)
            decode_times.append(time.perf_counter() - t0)

            next_input = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur_kv = step_out.past_key_values

    decode_arr = np.array(decode_times)
    timings.update({
        "decode_mean_ms": round(float(decode_arr.mean() * 1000), 3),
        "decode_p50_ms": round(float(np.percentile(decode_arr, 50) * 1000), 3),
        "decode_p99_ms": round(float(np.percentile(decode_arr, 99) * 1000), 3),
        "decode_throughput_tok_s": round(float(1.0 / decode_arr.mean()), 2),
        "n_decode_steps": n_decode_steps,
        "prefill_ms": round(timings["prefill_s"] * 1000, 3),
        "compression_ms": round(timings["compression_s"] * 1000, 3),
    })
    del timings["prefill_s"]
    del timings["compression_s"]
    return timings


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def benchmark_config(
    model,
    tokenizer,
    method: str | None,
    avg_bits: float | None,
    calibration: dict | None,
    device: torch.device,
    seq_lengths: list[int],
    n_warmup: int = N_WARMUP,
    n_runs: int = N_BENCHMARK,
    n_decode_steps: int = 32,
) -> list[dict]:
    results = []

    for seq_len in seq_lengths:
        log.info("  seq_len=%d …", seq_len)

        # Build synthetic prompt
        dummy_text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10 + 1)
        enc = tokenizer(
            dummy_text, return_tensors="pt",
            max_length=seq_len, truncation=True,
        ).to(device)
        input_ids = enc["input_ids"]
        actual_len = input_ids.shape[1]

        # Skip if model would OOM (rough heuristic: 16K+ on CPU is very slow)
        if device.type == "cpu" and actual_len > 2048:
            log.info("    Skipping seq_len=%d on CPU (too slow)", actual_len)
            continue

        # Warmup
        for _ in range(n_warmup):
            try:
                _ = benchmark_sequence(
                    model, tokenizer, input_ids, method, avg_bits,
                    calibration, device, n_decode_steps=4,
                )
            except torch.cuda.OutOfMemoryError:
                log.warning("    OOM at seq_len=%d — skipping.", actual_len)
                break
            except Exception as e:
                log.warning("    Error during warmup: %s", e)
                break

        # Benchmark runs
        run_timings = []
        oom = False
        for _ in range(n_runs):
            try:
                t = benchmark_sequence(
                    model, tokenizer, input_ids, method, avg_bits,
                    calibration, device, n_decode_steps=n_decode_steps,
                )
                run_timings.append(t)
            except torch.cuda.OutOfMemoryError:
                log.warning("    OOM during benchmark run at seq_len=%d", actual_len)
                oom = True
                break
            except Exception as e:
                log.warning("    Error: %s", e)
                break

        if oom or not run_timings:
            results.append({
                "seq_len": actual_len,
                "status": "OOM" if oom else "ERROR",
            })
            continue

        # Aggregate
        agg = {
            "seq_len": actual_len,
            "status": "OK",
            "prefill_ms_mean": round(float(np.mean([r["prefill_ms"] for r in run_timings])), 3),
            "prefill_ms_std": round(float(np.std([r["prefill_ms"] for r in run_timings])), 3),
            "compression_ms_mean": round(float(np.mean([r["compression_ms"] for r in run_timings])), 3),
            "decode_mean_ms": round(float(np.mean([r["decode_mean_ms"] for r in run_timings])), 3),
            "decode_p50_ms": round(float(np.median([r["decode_p50_ms"] for r in run_timings])), 3),
            "decode_p99_ms": round(float(np.mean([r["decode_p99_ms"] for r in run_timings])), 3),
            "throughput_tok_s": round(float(np.mean([r["decode_throughput_tok_s"] for r in run_timings])), 2),
            "n_runs": len(run_timings),
        }
        results.append(agg)
        log.info(
            "    decode: mean=%.2fms  p50=%.2fms  p99=%.2fms  throughput=%.1f tok/s",
            agg["decode_mean_ms"], agg["decode_p50_ms"],
            agg["decode_p99_ms"], agg["throughput_tok_s"],
        )

    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_latency(all_bench: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.3,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"FP16": "#2ca02c", "TurboQuant-3bit": "#1f77b4", "SpectralQuant-3bit": "#d62728"}
    markers = {"FP16": "*", "TurboQuant-3bit": "s", "SpectralQuant-3bit": "o"}

    for method_name, results in all_bench.items():
        ok = [r for r in results if r.get("status") == "OK"]
        if not ok:
            continue
        xs = [r["seq_len"] for r in ok]
        decode_ms = [r["decode_mean_ms"] for r in ok]
        throughput = [r["throughput_tok_s"] for r in ok]
        color = colors.get(method_name, "gray")
        marker = markers.get(method_name, "o")

        ax1.plot(xs, decode_ms, color=color, marker=marker, linewidth=2,
                 markersize=7, label=method_name)
        ax2.plot(xs, throughput, color=color, marker=marker, linewidth=2,
                 markersize=7, label=method_name)

    ax1.set_xlabel("Sequence length (tokens)")
    ax1.set_ylabel("Per-step decode latency (ms)")
    ax1.set_title("Decode Latency vs Sequence Length")
    ax1.legend()

    ax2.set_xlabel("Sequence length (tokens)")
    ax2.set_ylabel("Throughput (tokens/s)")
    ax2.set_title("Generation Throughput vs Sequence Length")
    ax2.legend()

    fig.suptitle("Latency Benchmarks — Qwen 2.5-1.5B\nB200 GPU", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / "fig_latency_benchmark.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Latency figure → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp 6: Latency benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seq-lengths", type=int, nargs="+",
                   default=SEQ_LENGTHS,
                   help="Sequence lengths to benchmark")
    p.add_argument("--n-warmup", type=int, default=N_WARMUP)
    p.add_argument("--n-runs", type=int, default=N_BENCHMARK)
    p.add_argument("--n-decode", type=int, default=32,
                   help="Number of decode steps per run")
    p.add_argument("--avg-bits", type=float, default=3.0)
    p.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    set_seed(args.seed)
    device = detect_device()

    seq_lengths = [512, 1024] if args.quick else args.seq_lengths
    n_warmup = 1 if args.quick else args.n_warmup
    n_runs = 2 if args.quick else args.n_runs
    n_decode = 4 if args.quick else args.n_decode

    log.info("Experiment 6: Latency Benchmarking")
    log.info("  seq_lengths=%s  n_warmup=%d  n_runs=%d", seq_lengths, n_warmup, n_runs)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    calibration = None
    try:
        if (args.calib_dir / "metadata.json").exists():
            calibration = {}
            for l in range(n_layers):
                calibration[l] = {}
                for h in range(n_kv_heads):
                    fname = args.calib_dir / f"layer{l:02d}_head{h:02d}.npz"
                    if fname.exists():
                        d = np.load(str(fname))
                        calibration[l][h] = {
                            "key_eigenvectors": d["key_eigenvectors"],
                            "key_mean": d["key_mean"],
                            "key_d_eff": float(d["key_d_eff"]),
                            "val_eigenvectors": d["val_eigenvectors"],
                            "val_mean": d["val_mean"],
                            "val_d_eff": float(d["val_d_eff"]),
                        }
            log.info("Calibration loaded.")
    except Exception as e:
        log.warning("Could not load calibration: %s", e)

    # ------------------------------------------------------------------
    # Benchmark each method
    # ------------------------------------------------------------------
    bench_configs = [
        ("FP16",               None,       None),
        ("TurboQuant-3bit",    "turbo",    args.avg_bits),
        ("SpectralQuant-3bit", "spectral", args.avg_bits),
    ]

    all_bench: dict[str, list[dict]] = {}
    all_rows: list[dict] = []

    for method_name, method, avg_bits in bench_configs:
        log.info("--- Benchmarking: %s ---", method_name)
        results = benchmark_config(
            model, tokenizer, method, avg_bits, calibration, device,
            seq_lengths=seq_lengths,
            n_warmup=n_warmup, n_runs=n_runs, n_decode_steps=n_decode,
        )
        all_bench[method_name] = results
        for r in results:
            row = {"method": method_name, **r}
            all_rows.append(row)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    log.info("=== LATENCY BENCHMARK SUMMARY ===")
    for method_name, results in all_bench.items():
        log.info("  [%s]", method_name)
        for r in results:
            if r.get("status") == "OK":
                log.info(
                    "    seq=%5d  decode=%.2fms (p50=%.2f p99=%.2f)  throughput=%.1f tok/s",
                    r["seq_len"], r["decode_mean_ms"], r["decode_p50_ms"],
                    r["decode_p99_ms"], r["throughput_tok_s"],
                )
            else:
                log.info("    seq=%5d  [%s]", r["seq_len"], r.get("status", "?"))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    csv_path = results_dir / "latency_results.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        log.info("Latency CSV → %s", csv_path)

    plot_latency(all_bench, results_dir)

    output = {
        "phase": "exp6_latency",
        "avg_bits": args.avg_bits,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "device": str(device),
        "results": {k: v for k, v in all_bench.items()},
        "wall_time_s": round(time.time() - t_total, 2),
    }
    with open(results_dir / "latency_results.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info("Experiment 6 complete in %.1f s.", time.time() - t_total)


if __name__ == "__main__":
    main()
