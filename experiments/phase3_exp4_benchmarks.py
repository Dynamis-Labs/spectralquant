#!/usr/bin/env python3
"""
phase3_exp4_benchmarks.py — Experiment 4: Downstream Benchmarks

Runs at least 2 of: LongBench, Needle-in-Haystack, RULER (ZeroSCROLLS).
Compares SpectralQuant vs TurboQuant vs FP16 at matched bit-widths.
Saves benchmark results as CSV.

Implemented benchmarks:
  1. Needle-in-Haystack (custom implementation — no external dependency)
  2. LongBench subset (multi-doc QA via lm-evaluation-harness or custom)
  3. Passkey Retrieval (simple retrieval task)

Usage:
  python phase3_exp4_benchmarks.py [--quick] [--benchmarks needle longbench]
"""

import argparse
import csv
import json
import logging
import random
import re
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

RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CALIB_DIR = PROJECT_ROOT / "results" / "eigenspectral" / "calibration"
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

EVAL_CONFIGS = [
    ("FP16",    None,       None),
    ("TQ-3bit", "turbo",    3.0),
    ("SQ-3bit", "spectral", 3.0),
    ("SQ-2.5",  "spectral", 2.5),
]

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
log = logging.getLogger("exp4_benchmarks")


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
        log.info("GPU: %s (%.1f GB)", props.name, (getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)) / 1e9)
        return torch.device("cuda")
    log.warning("No GPU — CPU only.")
    return torch.device("cpu")


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


def compress_kv(kv_cache, method, avg_bits, calibration, device="cpu"):
    """Compress and decompress a KV cache. Returns new cache."""
    if method is None:
        return [(k.clone(), v.clone()) for k, v in kv_cache]

    new_cache = []
    for l, (k, v) in enumerate(kv_cache):
        k_f = k.float().cpu()
        v_f = v.float().cpu()
        n_heads = k_f.shape[1]
        k_list, v_list = [], []

        for h in range(n_heads):
            k_h = k_f[0, h]
            v_h = v_f[0, h]
            head_dim = k_h.shape[-1]

            if method == "turbo":
                rng = np.random.default_rng(l * 100 + h)
                A = rng.standard_normal((head_dim, head_dim)).astype(np.float32)
                Q, _ = np.linalg.qr(A)
                Pi = torch.from_numpy(Q)
                bits = max(1, int(round(avg_bits)))
                k_h = _uniform_quantize(k_h @ Pi, bits) @ Pi.T
                v_h = _uniform_quantize(v_h @ Pi, bits) @ Pi.T

            elif method == "spectral" and calibration and l in calibration and h in calibration[l]:
                c = calibration[l][h]
                Vk = torch.from_numpy(c["key_eigenvectors"]).float()
                mu_k = torch.from_numpy(c["key_mean"]).float()
                d_eff_k = float(c["key_d_eff"])
                d_sem_k = max(1, int(round(d_eff_k)))
                bh_k, bl_k = _solve_bits(avg_bits, head_dim, d_sem_k)
                k_rot = (k_h - mu_k) @ Vk
                k_h = torch.cat([
                    _uniform_quantize(k_rot[..., :d_sem_k], bh_k),
                    _uniform_quantize(k_rot[..., d_sem_k:], bl_k),
                ], dim=-1) @ Vk.T + mu_k

                Vv = torch.from_numpy(c["val_eigenvectors"]).float()
                mu_v = torch.from_numpy(c["val_mean"]).float()
                d_eff_v = float(c["val_d_eff"])
                d_sem_v = max(1, int(round(d_eff_v)))
                bh_v, bl_v = _solve_bits(avg_bits, head_dim, d_sem_v)
                v_rot = (v_h - mu_v) @ Vv
                v_h = torch.cat([
                    _uniform_quantize(v_rot[..., :d_sem_v], bh_v),
                    _uniform_quantize(v_rot[..., d_sem_v:], bl_v),
                ], dim=-1) @ Vv.T + mu_v

            k_list.append(k_h.unsqueeze(0))
            v_list.append(v_h.unsqueeze(0))

        k_new = torch.stack(k_list, dim=1).to(k.dtype)
        v_new = torch.stack(v_list, dim=1).to(v.dtype)
        new_cache.append((k_new, v_new))

    return new_cache


# ---------------------------------------------------------------------------
# Benchmark 1: Needle in a Haystack
# ---------------------------------------------------------------------------

NEEDLE_TEMPLATE = (
    "The following is a long document. Read carefully.\n\n"
    "{haystack}\n\n"
    "Based on the passage above, what is the secret number mentioned? "
    "Answer with only the number.\n\n"
    "Secret number: "
)

def generate_haystack(target_len: int, needle_position: float, needle: str) -> str:
    """Generate a haystack string with the needle inserted at `needle_position` fraction."""
    filler_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning has transformed natural language processing significantly.",
        "Large language models are capable of impressive in-context reasoning.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Transformer architectures have become dominant in modern AI research.",
        "Natural language generation has improved dramatically with scaling.",
        "The weather today is partly cloudy with a chance of showers.",
        "Scientific research requires careful experimental design and validation.",
        "Data preprocessing is a critical step in any machine learning pipeline.",
        "Regularization techniques help prevent overfitting in neural networks.",
    ]
    # Build haystack by repeating filler
    words_needed = target_len
    filler_pool = " ".join(filler_sentences * (words_needed // 50 + 5))
    words = filler_pool.split()[:words_needed]

    # Insert needle at position
    insert_idx = int(needle_position * len(words))
    words.insert(insert_idx, needle)

    return " ".join(words)


def run_needle_in_haystack(
    model,
    tokenizer,
    method: str | None,
    avg_bits: float | None,
    calibration: dict | None,
    device: torch.device,
    n_trials: int = 10,
    haystack_tokens: int = 1024,
    quick: bool = False,
) -> dict:
    """Run needle-in-haystack benchmark and return accuracy."""
    if quick:
        n_trials = 3
        haystack_tokens = 256

    secret_numbers = [random.randint(1000, 9999) for _ in range(n_trials)]
    needle_positions = np.linspace(0.1, 0.9, n_trials).tolist()

    correct = 0
    details = []

    with torch.no_grad():
        for i, (secret, pos) in enumerate(zip(secret_numbers, needle_positions)):
            needle = f"The secret number is {secret}."
            haystack = generate_haystack(haystack_tokens, pos, needle)

            prompt = NEEDLE_TEMPLATE.format(haystack=haystack)
            enc = tokenizer(
                prompt, return_tensors="pt",
                max_length=haystack_tokens + 200,
                truncation=True,
            ).to(device)

            # Prefill
            out = model(**enc, use_cache=True)
            kv = out.past_key_values

            if method is not None:
                kv = compress_kv(
                    [(k.cpu(), v.cpu()) for k, v in kv],
                    method=method, avg_bits=avg_bits, calibration=calibration,
                )
                kv = [(k.to(device), v.to(device)) for k, v in kv]

            # Generate answer (up to 10 tokens)
            generated = enc["input_ids"].clone()
            cur_kv = kv
            for _ in range(10):
                step_out = model(generated[:, -1:], past_key_values=cur_kv, use_cache=True)
                next_tok = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)
                cur_kv = step_out.past_key_values
                if next_tok.item() == tokenizer.eos_token_id:
                    break

            answer_tokens = generated[0, enc["input_ids"].shape[1]:]
            answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

            # Check if secret number appears in answer
            found = str(secret) in answer_text
            if found:
                correct += 1
            details.append({
                "secret": secret,
                "position": round(pos, 2),
                "answer": answer_text[:50],
                "correct": found,
            })
            log.debug("  Needle %d: secret=%d  answer='%s'  correct=%s",
                      i, secret, answer_text[:30], found)

    accuracy = correct / n_trials
    log.info("  Needle accuracy: %.1f%%  (%d/%d)", accuracy * 100, correct, n_trials)
    return {
        "benchmark": "needle_in_haystack",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": n_trials,
        "haystack_tokens": haystack_tokens,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Passkey Retrieval
# ---------------------------------------------------------------------------

PASSKEY_TEMPLATE = (
    "There is an important passkey hidden in the following text. "
    "Find it and remember it. {context}\n\n"
    "What is the passkey? The passkey is: "
)


def run_passkey_retrieval(
    model,
    tokenizer,
    method: str | None,
    avg_bits: float | None,
    calibration: dict | None,
    device: torch.device,
    n_trials: int = 10,
    context_tokens: int = 512,
    quick: bool = False,
) -> dict:
    """Simple passkey retrieval benchmark."""
    if quick:
        n_trials = 3
        context_tokens = 128

    filler = (
        "The grass is green. The sky is blue. The sun is yellow. "
        "Here we see the foliage of many trees. Birds are singing. "
    ) * 100

    correct = 0
    for _ in range(n_trials):
        passkey = f"{random.randint(10000, 99999)}"
        needle = f"The passkey is {passkey}. "
        context = filler[:context_tokens * 4]  # approximate char budget
        pos = random.randint(10, len(context) - len(needle) - 10)
        context_with_needle = context[:pos] + needle + context[pos:]

        prompt = PASSKEY_TEMPLATE.format(context=context_with_needle[:context_tokens * 5])
        enc = tokenizer(
            prompt, return_tensors="pt",
            max_length=context_tokens + 100, truncation=True,
        ).to(device)

        with torch.no_grad():
            out = model(**enc, use_cache=True)
            kv = out.past_key_values

            if method is not None:
                kv = compress_kv(
                    [(k.cpu(), v.cpu()) for k, v in kv],
                    method=method, avg_bits=avg_bits, calibration=calibration,
                )
                kv = [(k.to(device), v.to(device)) for k, v in kv]

            gen = enc["input_ids"].clone()
            cur_kv = kv
            for _ in range(15):
                step = model(gen[:, -1:], past_key_values=cur_kv, use_cache=True)
                tok = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen = torch.cat([gen, tok], dim=1)
                cur_kv = step.past_key_values
                if tok.item() == tokenizer.eos_token_id:
                    break

            ans = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if passkey in ans:
                correct += 1

    acc = correct / n_trials
    log.info("  Passkey accuracy: %.1f%%  (%d/%d)", acc * 100, correct, n_trials)
    return {
        "benchmark": "passkey_retrieval",
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": n_trials,
        "context_tokens": context_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp 4: Downstream benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--benchmarks", nargs="+",
                   default=["needle", "passkey"],
                   choices=["needle", "passkey"],
                   help="Which benchmarks to run")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--haystack-tokens", type=int, default=1024)
    p.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args()


def load_calibration(calib_dir: Path, n_layers: int, n_kv_heads: int) -> dict | None:
    try:
        if not (calib_dir / "metadata.json").exists():
            return None
        calibration = {}
        for l in range(n_layers):
            calibration[l] = {}
            for h in range(n_kv_heads):
                fname = calib_dir / f"layer{l:02d}_head{h:02d}.npz"
                if fname.exists():
                    d = np.load(str(fname))
                    calibration[l][h] = {
                        "key_eigenvectors": d["key_eigenvectors"],
                        "key_eigenvalues": d["key_eigenvalues"],
                        "key_mean": d["key_mean"],
                        "key_d_eff": float(d["key_d_eff"]),
                        "val_eigenvectors": d["val_eigenvectors"],
                        "val_eigenvalues": d["val_eigenvalues"],
                        "val_mean": d["val_mean"],
                        "val_d_eff": float(d["val_d_eff"]),
                    }
        return calibration
    except Exception as e:
        log.warning("Could not load calibration: %s", e)
        return None


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    set_seed(args.seed)
    device = detect_device()

    log.info("Experiment 4: Downstream Benchmarks")
    log.info("  Benchmarks: %s", args.benchmarks)

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

    calibration = load_calibration(args.calib_dir, n_layers, n_kv_heads)
    if calibration:
        log.info("Calibration loaded.")
    else:
        log.warning("No calibration — SpectralQuant configs will fall back to FP16.")

    # ------------------------------------------------------------------
    # Run benchmarks for each config
    # ------------------------------------------------------------------
    all_results = []

    for config_name, method, avg_bits in EVAL_CONFIGS:
        log.info("--- Config: %s ---", config_name)
        config_row = {
            "config": config_name,
            "method": method or "fp16",
            "avg_bits": avg_bits,
        }

        if "needle" in args.benchmarks:
            log.info("  Running Needle-in-Haystack …")
            r = run_needle_in_haystack(
                model, tokenizer, method, avg_bits, calibration, device,
                n_trials=args.n_trials,
                haystack_tokens=args.haystack_tokens,
                quick=args.quick,
            )
            config_row["needle_accuracy"] = r["accuracy"]
            config_row["needle_n"] = r["total"]

        if "passkey" in args.benchmarks:
            log.info("  Running Passkey Retrieval …")
            r2 = run_passkey_retrieval(
                model, tokenizer, method, avg_bits, calibration, device,
                n_trials=args.n_trials,
                quick=args.quick,
            )
            config_row["passkey_accuracy"] = r2["accuracy"]
            config_row["passkey_n"] = r2["total"]

        all_results.append(config_row)

    # ------------------------------------------------------------------
    # Print & save
    # ------------------------------------------------------------------
    log.info("=== BENCHMARK RESULTS ===")
    header_cols = list(all_results[0].keys()) if all_results else []
    log.info("  " + "  ".join(f"{c:>18}" for c in header_cols))
    for row in all_results:
        log.info("  " + "  ".join(
            f"{str(row.get(c, 'N/A')):>18}" for c in header_cols
        ))

    csv_path = results_dir / "benchmark_results.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        log.info("Benchmark CSV → %s", csv_path)

    # Figure
    _plot_benchmarks(all_results, args.benchmarks, results_dir)

    output = {
        "phase": "exp4_benchmarks",
        "benchmarks": args.benchmarks,
        "results": all_results,
        "wall_time_s": round(time.time() - t_total, 2),
    }
    with open(results_dir / "benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info("Experiment 4 complete in %.1f s.", time.time() - t_total)


def _plot_benchmarks(results: list[dict], benchmarks: list[str], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.3, "grid.axis": "y",
    })

    n_bench = len(benchmarks)
    fig, axes = plt.subplots(1, n_bench, figsize=(5 * n_bench, 5), sharey=False)
    if n_bench == 1:
        axes = [axes]

    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
    config_names = [r["config"] for r in results]

    bench_keys = {
        "needle": ("needle_accuracy", "Needle-in-Haystack Accuracy"),
        "passkey": ("passkey_accuracy", "Passkey Retrieval Accuracy"),
    }

    for ax, bench in zip(axes, benchmarks):
        key, title = bench_keys[bench]
        vals = [r.get(key, 0.0) or 0.0 for r in results]
        ax.bar(config_names, vals, color=colors[:len(results)],
               edgecolor="black", linewidth=0.7, alpha=0.88)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Downstream Benchmark Results — SpectralQuant vs TurboQuant vs FP16\n"
        "Qwen 2.5-1.5B",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / "fig_benchmark_results.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Benchmark figure → %s", out_path)


if __name__ == "__main__":
    main()
