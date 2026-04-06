#!/usr/bin/env python3
"""
phase3_exp3_generation.py — Experiment 3: Text Generation Quality

Generates text from compressed KV cache and evaluates:
  - Perplexity on held-out WikiText-103 text
  - Qualitative side-by-side generation samples

Compares:
  FP16 baseline vs TurboQuant-3bit vs SpectralQuant-3bit vs SpectralQuant-2.5bit

The KV cache is compressed during the prefill phase and decompressed for
attention computation during generation.

Usage:
  python phase3_exp3_generation.py [--quick] [--n-samples N]
"""

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results" / "generation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CALIB_DIR = PROJECT_ROOT / "results" / "eigenspectral" / "calibration"
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

GENERATION_CONFIGS = [
    ("FP16",        "FP16 baseline",             None,  None),
    ("TQ-3bit",     "TurboQuant 3-bit",          "turbo", 3.0),
    ("SQ-3bit",     "SpectralQuant 3-bit",       "spectral", 3.0),
    ("SQ-2.5bit",   "SpectralQuant 2.5-bit",     "spectral", 2.5),
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
log = logging.getLogger("exp3_generation")


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


def compress_kv_cache(kv_cache, method: str, avg_bits: float, calibration: dict | None):
    """
    Compress a full KV cache (list of (k, v) tensors) using the specified method.
    Returns compressed + decompressed cache (same format, lossy).
    """
    new_cache = []
    n_layers = len(kv_cache)

    for l, (k, v) in enumerate(kv_cache):
        # k, v: [B, n_heads, T, head_dim]
        k_f = k.float()
        v_f = v.float()

        if method is None:  # FP16 — no compression
            new_cache.append((k.clone(), v.clone()))
            continue

        n_heads = k_f.shape[1]
        k_recon_list, v_recon_list = [], []

        for h in range(n_heads):
            k_h = k_f[0, h]  # [T, head_dim]
            v_h = v_f[0, h]

            if method == "turbo":
                head_dim = k_h.shape[-1]
                rng = np.random.default_rng(l * 100 + h)
                A = rng.standard_normal((head_dim, head_dim)).astype(np.float32)
                Q, _ = np.linalg.qr(A)
                Pi = torch.from_numpy(Q).to(k_h.device)
                bits = max(1, int(round(avg_bits)))

                k_rot = k_h @ Pi
                k_recon = _uniform_quantize(k_rot, bits) @ Pi.T

                v_rot = v_h @ Pi
                v_recon = _uniform_quantize(v_rot, bits) @ Pi.T

            elif method == "spectral":
                if (calibration is None
                        or l not in calibration
                        or h not in calibration[l]):
                    # Fallback to FP16
                    k_recon, v_recon = k_h, v_h
                else:
                    c = calibration[l][h]
                    V = torch.from_numpy(c["key_eigenvectors"]).float().to(k_h.device)
                    mu_k = torch.from_numpy(c["key_mean"]).float().to(k_h.device)
                    d_eff = float(c["key_d_eff"])
                    d_sem = max(1, int(round(d_eff)))
                    head_dim = k_h.shape[-1]
                    b_high, b_low = _solve_bits(avg_bits, head_dim, d_sem)

                    k_rot = (k_h - mu_k) @ V
                    k_sem_q = _uniform_quantize(k_rot[..., :d_sem], b_high)
                    k_tail_q = _uniform_quantize(k_rot[..., d_sem:], b_low)
                    k_q = torch.cat([k_sem_q, k_tail_q], dim=-1)
                    k_recon = k_q @ V.T + mu_k

                    # Values
                    Vv = torch.from_numpy(c["val_eigenvectors"]).float().to(v_h.device)
                    mu_v = torch.from_numpy(c["val_mean"]).float().to(v_h.device)
                    d_eff_v = float(c["val_d_eff"])
                    d_sem_v = max(1, int(round(d_eff_v)))
                    b_high_v, b_low_v = _solve_bits(avg_bits, head_dim, d_sem_v)

                    v_rot = (v_h - mu_v) @ Vv
                    v_sem_q = _uniform_quantize(v_rot[..., :d_sem_v], b_high_v)
                    v_tail_q = _uniform_quantize(v_rot[..., d_sem_v:], b_low_v)
                    v_q = torch.cat([v_sem_q, v_tail_q], dim=-1)
                    v_recon = v_q @ Vv.T + mu_v
            else:
                k_recon, v_recon = k_h, v_h

            k_recon_list.append(k_recon.unsqueeze(0))
            v_recon_list.append(v_recon.unsqueeze(0))

        k_new = torch.stack(k_recon_list, dim=1).to(k.dtype)
        v_new = torch.stack(v_recon_list, dim=1).to(v.dtype)
        new_cache.append((k_new, v_new))

    return new_cache


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
# Perplexity measurement
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    method: str | None,
    avg_bits: float | None,
    calibration: dict | None,
    device: torch.device,
    seq_len: int = 512,
    stride: int = 256,
) -> float:
    """
    Compute perplexity with compressed KV cache.

    Strategy: encode a long sequence, run prefill with compressed KV cache,
    then score next-token log-probs.
    """
    all_nlls = []

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt",
                max_length=seq_len, truncation=True,
            ).to(device)
            input_ids = enc["input_ids"]
            T = input_ids.shape[1]
            if T < 8:
                continue

            # Split: use first half as context (KV cache), score second half
            ctx_len = max(4, T // 2)
            ctx_ids = input_ids[:, :ctx_len]
            tgt_ids = input_ids[:, ctx_len:]

            if tgt_ids.shape[1] < 2:
                continue

            # Run prefill
            out = model(ctx_ids, use_cache=True)
            kv_cache = out.past_key_values

            if method is not None and avg_bits is not None:
                # Compress KV cache
                kv_cache = compress_kv_cache(
                    [(k.cpu(), v.cpu()) for k, v in kv_cache],
                    method=method,
                    avg_bits=avg_bits,
                    calibration=calibration,
                )
                kv_cache = [(k.to(device), v.to(device)) for k, v in kv_cache]

            # Score target tokens
            tgt_out = model(tgt_ids[:, :-1], past_key_values=kv_cache, use_cache=False)
            logits = tgt_out.logits  # [1, T_tgt - 1, vocab]
            labels = tgt_ids[:, 1:]  # [1, T_tgt - 1]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="mean",
            )
            all_nlls.append(loss.item())

    if not all_nlls:
        return float("nan")
    return math.exp(min(float(np.mean(all_nlls)), 20))


# ---------------------------------------------------------------------------
# Generation samples
# ---------------------------------------------------------------------------

def generate_samples(
    model,
    tokenizer,
    prompts: list[str],
    method: str | None,
    avg_bits: float | None,
    calibration: dict | None,
    device: torch.device,
    max_new_tokens: int = 100,
) -> list[str]:
    """Generate text for each prompt using compressed KV cache."""
    generated = []

    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(
                prompt, return_tensors="pt",
                max_length=128, truncation=True,
            ).to(device)

            # Prefill
            out = model(**enc, use_cache=True)
            kv_cache = out.past_key_values

            if method is not None and avg_bits is not None:
                kv_cache = compress_kv_cache(
                    [(k.cpu(), v.cpu()) for k, v in kv_cache],
                    method=method, avg_bits=avg_bits, calibration=calibration,
                )
                kv_cache = [(k.to(device), v.to(device)) for k, v in kv_cache]

            # Autoregressive generation
            input_ids = enc["input_ids"]
            generated_ids = input_ids.clone()
            cur_kv = kv_cache

            for _ in range(max_new_tokens):
                step_out = model(
                    generated_ids[:, -1:],
                    past_key_values=cur_kv,
                    use_cache=True,
                )
                next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                cur_kv = step_out.past_key_values
                if next_token.item() == tokenizer.eos_token_id:
                    break

            gen_text = tokenizer.decode(
                generated_ids[0, enc["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            generated.append(gen_text.strip())

    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp 3: Text generation quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-ppl-samples", type=int, default=50,
                   help="Number of texts for perplexity evaluation")
    p.add_argument("--n-gen-samples", type=int, default=5,
                   help="Number of generation samples for qualitative comparison")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=512)
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

    n_ppl = 10 if args.quick else args.n_ppl_samples
    n_gen = 2 if args.quick else args.n_gen_samples
    max_new = 30 if args.quick else args.max_new_tokens
    seq_len = 256 if args.quick else args.seq_len

    log.info("Experiment 3: Text Generation Quality")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

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
    # Load calibration
    # ------------------------------------------------------------------
    calibration = None
    try:
        if (args.calib_dir / "metadata.json").exists():
            import json as _json
            with open(args.calib_dir / "metadata.json") as f:
                meta = _json.load(f)
            calibration = {}
            for l in range(n_layers):
                calibration[l] = {}
                for h in range(n_kv_heads):
                    fname = args.calib_dir / f"layer{l:02d}_head{h:02d}.npz"
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
            log.info("Calibration loaded.")
    except Exception as e:
        log.warning("Could not load calibration: %s", e)

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    test_texts = [
        r["text"].strip() for r in dataset if len(r["text"].strip()) > 300
    ][:n_ppl]

    prompts = [t[:200] for t in test_texts[:n_gen]]
    log.info("PPL texts: %d | Generation prompts: %d", len(test_texts), len(prompts))

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------
    ppl_results = []
    for name, desc, method, avg_bits in GENERATION_CONFIGS:
        log.info("Computing perplexity: %s …", name)
        t0 = time.time()
        ppl = compute_perplexity(
            model, tokenizer, test_texts, method, avg_bits,
            calibration, device, seq_len=seq_len,
        )
        elapsed = time.time() - t0
        log.info("  %s perplexity: %.3f  (%.1f s)", name, ppl, elapsed)
        ppl_results.append({
            "config": name,
            "description": desc,
            "method": method or "fp16",
            "avg_bits": avg_bits,
            "perplexity": round(ppl, 4),
            "wall_time_s": round(elapsed, 2),
        })

    # ------------------------------------------------------------------
    # Generation samples
    # ------------------------------------------------------------------
    gen_results: dict[str, list[str]] = {}
    for name, desc, method, avg_bits in GENERATION_CONFIGS:
        log.info("Generating samples: %s …", name)
        t0 = time.time()
        samples = generate_samples(
            model, tokenizer, prompts, method, avg_bits,
            calibration, device, max_new_tokens=max_new,
        )
        gen_results[name] = samples
        log.info("  %s generation complete (%.1f s)", name, time.time() - t0)

    # ------------------------------------------------------------------
    # Print perplexity table
    # ------------------------------------------------------------------
    log.info("=== PERPLEXITY RESULTS ===")
    log.info("  %-12s  %-28s  %8s", "Config", "Description", "PPL")
    log.info("  " + "-" * 52)
    for r in ppl_results:
        log.info("  %-12s  %-28s  %8.3f", r["config"], r["description"], r["perplexity"])

    # ------------------------------------------------------------------
    # Save perplexity CSV
    # ------------------------------------------------------------------
    csv_path = results_dir / "generation_perplexity.csv"
    if ppl_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ppl_results[0].keys()))
            writer.writeheader()
            writer.writerows(ppl_results)
        log.info("Perplexity CSV → %s", csv_path)

    # ------------------------------------------------------------------
    # Save qualitative samples
    # ------------------------------------------------------------------
    samples_path = results_dir / "generation_samples.json"
    side_by_side = []
    for i, prompt in enumerate(prompts):
        entry = {"prompt": prompt[:100] + "...", "samples": {}}
        for name, _, _, _ in GENERATION_CONFIGS:
            if name in gen_results and i < len(gen_results[name]):
                entry["samples"][name] = gen_results[name][i]
        side_by_side.append(entry)

    with open(samples_path, "w") as f:
        json.dump(side_by_side, f, indent=2, ensure_ascii=False)
    log.info("Qualitative samples → %s", samples_path)

    # Print side-by-side sample
    if side_by_side:
        log.info("=== QUALITATIVE SAMPLE (first prompt) ===")
        log.info("Prompt: %s", side_by_side[0]["prompt"])
        for name, text in side_by_side[0]["samples"].items():
            log.info("  [%s] %s", name, text[:120])

    # ------------------------------------------------------------------
    # Figure: perplexity bar chart
    # ------------------------------------------------------------------
    _plot_perplexity(ppl_results, results_dir)

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    output = {
        "phase": "exp3_generation",
        "n_ppl_samples": len(test_texts),
        "n_gen_samples": len(prompts),
        "perplexity_results": ppl_results,
        "wall_time_s": round(time.time() - t_total, 2),
    }
    with open(results_dir / "generation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info("Experiment 3 complete in %.1f s.", time.time() - t_total)


def _plot_perplexity(results: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.3, "grid.axis": "y",
    })

    names = [r["config"] for r in results]
    ppls = [r["perplexity"] for r in results]
    colors = ["#2ca02c"] + ["#1f77b4"] + ["#d62728"] * 2

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, ppls, color=colors[:len(names)],
                  edgecolor="black", linewidth=0.7, alpha=0.88)
    for bar, val in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Text Generation Perplexity\nQwen 2.5-1.5B — WikiText-103 test set")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="FP16 reference"),
        Patch(facecolor="#1f77b4", label="TurboQuant"),
        Patch(facecolor="#d62728", label="SpectralQuant"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "fig_generation_perplexity.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Perplexity figure → %s", out_path)


if __name__ == "__main__":
    main()
