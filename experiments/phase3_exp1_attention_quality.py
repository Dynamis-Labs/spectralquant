#!/usr/bin/env python3
"""
phase3_exp1_attention_quality.py — Experiment 1: Attention Quality (MAIN RESULT)

Runs the same attention quality test as the cuTile repo:
  - 5 sampled layers, all KV heads, 8 query probes per head
  - Tests 6 configurations: T-3.0, T-2.5, T-2.0, S-3.0, S-2.5, S-2.0
  - Metrics: cosine similarity vs FP16, max absolute weight error

REFLECTION GATE 2:
  S ≥ T on at least one comparison → print "GATE 2 PASSED"
  Otherwise: debug instructions and exit

Usage:
  python phase3_exp1_attention_quality.py [--quick] [--layers L L L L L]
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
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results" / "attention_quality"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CALIB_DIR = PROJECT_ROOT / "results" / "eigenspectral" / "calibration"
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
N_PROBE_QUERIES = 8
N_SAMPLED_LAYERS = 5

CONFIGS = [
    # (name, method, avg_bits, rotation, bit_alloc, qjl)
    ("T-3.0", "TurboQuant",    3.0, "random",   "uniform",     "full"),
    ("T-2.5", "TurboQuant",    2.5, "random",   "uniform",     "full"),
    ("T-2.0", "TurboQuant",    2.0, "random",   "uniform",     "full"),
    ("S-3.0", "SpectralQuant", 3.0, "spectral", "non-uniform", "selective"),
    ("S-2.5", "SpectralQuant", 2.5, "spectral", "non-uniform", "selective"),
    ("S-2.0", "SpectralQuant", 2.0, "spectral", "non-uniform", "selective"),
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
log = logging.getLogger("exp1_attention_quality")


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
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s (%.1f GB VRAM)", props.name, (getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)) / 1e9)
    else:
        device = torch.device("cpu")
        log.warning("No GPU detected — running on CPU.")
    return device


def _uniform_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    n_levels = 2 ** bits
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    x_clamp = x.clamp(-3 * std, 3 * std)
    x_norm = (x_clamp / (3 * std) + 1) / 2
    x_int = (x_norm * (n_levels - 1)).round().clamp(0, n_levels - 1)
    return x_int / (n_levels - 1) * 2 * 3 * std - 3 * std


# ---------------------------------------------------------------------------
# Compressors
# ---------------------------------------------------------------------------

class TurboQuantCompressor:
    def __init__(self, head_dim: int, avg_bits: float, seed: int = 0):
        self.head_dim = head_dim
        self.avg_bits = avg_bits
        self.bits = max(1, int(round(avg_bits)))
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((head_dim, head_dim)).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        self.Pi = torch.from_numpy(Q)

    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        Pi = self.Pi.to(x.device)
        x_rot = x @ Pi
        x_q = _uniform_quantize(x_rot, self.bits)
        return x_q @ Pi.T


class SpectralQuantCompressor:
    def __init__(
        self,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        mean: np.ndarray,
        d_eff: float,
        avg_bits: float,
    ):
        self.V = torch.from_numpy(eigenvectors).float()
        self.eigenvalues = eigenvalues
        self.mean = torch.from_numpy(mean).float()
        self.head_dim = eigenvectors.shape[0]
        self.d_sem = max(1, int(round(d_eff)))
        self.avg_bits = avg_bits
        self.b_high, self.b_low = self._solve_bits(avg_bits)

    def _solve_bits(self, avg_bits: float) -> tuple[int, int]:
        d, ds = self.head_dim, self.d_sem
        budget = d * avg_bits
        best_err, best = float("inf"), (4, 2)
        for bh in range(2, 9):
            for bl in range(1, bh):
                err = abs(ds * bh + (d - ds) * bl - budget)
                if err < best_err:
                    best_err = err
                    best = (bh, bl)
        return best

    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        V = self.V.to(x.device)
        mean = self.mean.to(x.device)
        x_rot = (x - mean) @ V  # spectral rotation

        x_sem = _uniform_quantize(x_rot[..., :self.d_sem], self.b_high)
        x_tail = _uniform_quantize(x_rot[..., self.d_sem:], self.b_low)
        x_q = torch.cat([x_sem, x_tail], dim=-1)

        return x_q @ V.T + mean  # unrotate


# ---------------------------------------------------------------------------
# Attention quality evaluation
# ---------------------------------------------------------------------------

def softmax_attention_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    q: [n_q, head_dim]
    k: [T, head_dim]
    Returns: [n_q, T]
    """
    scale = k.shape[-1] ** -0.5
    logits = q @ k.T * scale
    return torch.softmax(logits, dim=-1)


def evaluate_config(
    config_name: str,
    method: str,
    avg_bits: float,
    kv_cache: list,
    query_probes: torch.Tensor,
    sampled_layers: list[int],
    calibration: dict,
    n_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> dict:
    """
    Evaluate one configuration on all sampled layers and heads.

    kv_cache: list of (k, v) tensors per layer, shape [1, n_kv_heads, T, head_dim]
    query_probes: [n_probe, head_dim]
    calibration: {layer: {head: {...}}}
    """
    cos_sims = []
    max_abs_errs = []

    for l in sampled_layers:
        k_fp16 = kv_cache[l][0].squeeze(0).float().to(device)  # [n_kv_heads, T, head_dim]
        v_fp16 = kv_cache[l][1].squeeze(0).float().to(device)

        for h in range(n_kv_heads):
            k_head = k_fp16[h]  # [T, head_dim]

            # Build compressor
            if method == "TurboQuant":
                compressor = TurboQuantCompressor(
                    head_dim=head_dim,
                    avg_bits=avg_bits,
                    seed=l * 100 + h,
                )
            else:  # SpectralQuant
                if calibration is None or l not in calibration or h not in calibration[l]:
                    log.warning("No calibration for layer %d head %d — skipping.", l, h)
                    continue
                c = calibration[l][h]
                compressor = SpectralQuantCompressor(
                    eigenvectors=c["key_eigenvectors"],
                    eigenvalues=c["key_eigenvalues"],
                    mean=c["key_mean"],
                    d_eff=c["key_d_eff"],
                    avg_bits=avg_bits,
                )

            # Compress + decompress keys
            k_recon = compressor.compress_decompress(k_head)

            # Compute attention weights: FP16 reference
            q_probes = query_probes.to(device)  # [n_probe, head_dim]
            w_ref = softmax_attention_weights(q_probes, k_head)      # [n_probe, T]
            w_comp = softmax_attention_weights(q_probes, k_recon)     # [n_probe, T]

            # Cosine similarity between weight vectors
            cos = F.cosine_similarity(w_ref, w_comp, dim=-1).mean().item()
            # Max absolute error
            max_err = (w_ref - w_comp).abs().max().item()

            cos_sims.append(cos)
            max_abs_errs.append(max_err)

    return {
        "config": config_name,
        "method": method,
        "avg_bits": avg_bits,
        "cosine_similarity_mean": round(float(np.mean(cos_sims)), 5) if cos_sims else None,
        "cosine_similarity_std": round(float(np.std(cos_sims)), 5) if cos_sims else None,
        "max_abs_weight_error_mean": round(float(np.mean(max_abs_errs)), 6) if max_abs_errs else None,
        "max_abs_weight_error_std": round(float(np.std(max_abs_errs)), 6) if max_abs_errs else None,
        "n_layer_head_pairs": len(cos_sims),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_comparison_bar(results: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.3, "grid.axis": "y",
    })

    names = [r["config"] for r in results]
    cos_means = [r["cosine_similarity_mean"] or 0.0 for r in results]
    cos_stds = [r["cosine_similarity_std"] or 0.0 for r in results]
    colors = ["#1f77b4"] * 3 + ["#d62728"] * 3  # blue for TurboQuant, red for SpectralQuant

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Cosine similarity
    bars = ax1.bar(names, cos_means, yerr=cos_stds, capsize=4, color=colors,
                   edgecolor="black", linewidth=0.7, alpha=0.85)
    ax1.set_ylim(max(0, min(cos_means) - 0.02), 1.005)
    ax1.set_ylabel("Cosine similarity vs FP16")
    ax1.set_title("Attention Weight Cosine Similarity")
    ax1.axhline(y=0.985, color="gray", linestyle="--", alpha=0.7, label="TurboQuant 3-bit baseline")
    ax1.legend()
    for bar, val in zip(bars, cos_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Max absolute weight error
    max_errs = [r["max_abs_weight_error_mean"] or 0.0 for r in results]
    ax2.bar(names, max_errs, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)
    ax2.set_ylabel("Max absolute attention weight error")
    ax2.set_title("Max Absolute Weight Error vs FP16")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="TurboQuant"),
        Patch(facecolor="#d62728", label="SpectralQuant"),
    ]
    ax2.legend(handles=legend_elements)

    fig.suptitle(
        "Attention Quality: SpectralQuant vs TurboQuant\n"
        f"Qwen 2.5-1.5B — {N_SAMPLED_LAYERS} layers, all KV heads, {N_PROBE_QUERIES} query probes",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / "fig_attention_quality_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved comparison figure → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp 1: Attention quality — SpectralQuant vs TurboQuant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="Reduced evaluation (fewer sequences)")
    p.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-seqs", type=int, default=20,
                   help="Number of test sequences for averaging")
    p.add_argument("--n-probe", type=int, default=N_PROBE_QUERIES,
                   help="Query probes per head")
    p.add_argument("--n-layers", type=int, default=N_SAMPLED_LAYERS,
                   help="Number of layers to sample")
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

    n_seqs = 5 if args.quick else args.n_seqs
    seq_len = 128 if args.quick else args.seq_len
    n_probe = args.n_probe

    log.info("Experiment 1: Attention Quality")
    log.info("  n_seqs=%d  seq_len=%d  n_probe=%d", n_seqs, seq_len, n_probe)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    log.info("Loading model …")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    # Sample layers
    sampled_layers = sorted(
        np.linspace(0, n_layers - 1, args.n_layers, dtype=int).tolist()
    )
    log.info("Sampled layers: %s", sampled_layers)

    # ------------------------------------------------------------------
    # Load calibration
    # ------------------------------------------------------------------
    calibration = None
    if args.calib_dir.exists():
        try:
            from phase2_integration import load_calibration
            calibration, _ = load_calibration(args.calib_dir)
            log.info("Calibration loaded.")
        except Exception as e:
            log.warning("Could not load calibration: %s — SpectralQuant will be skipped.", e)
    else:
        log.warning("Calibration dir not found: %s", args.calib_dir)

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    log.info("Loading test sequences …")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", streaming=False)
    test_texts = [
        r["text"].strip() for r in dataset if len(r["text"].strip()) > 200
    ][:n_seqs]
    log.info("Test texts: %d", len(test_texts))

    # Query probes: random unit vectors
    query_probes = F.normalize(torch.randn(n_probe, head_dim), dim=-1)

    # ------------------------------------------------------------------
    # Accumulate results over test sequences
    # ------------------------------------------------------------------
    all_results_per_config = {name: [] for name, *_ in CONFIGS}

    with torch.no_grad():
        for i, text in enumerate(test_texts):
            log.info("  Sequence %d/%d …", i + 1, len(test_texts))
            enc = tokenizer(
                text, return_tensors="pt", max_length=seq_len,
                truncation=True, padding=False,
            ).to(device)
            if enc["input_ids"].shape[1] < 16:
                continue
            out = model(**enc, use_cache=True, output_hidden_states=False)
            kv_cache = out.past_key_values

            for (name, method, avg_bits, rotation, bit_alloc, qjl) in CONFIGS:
                if method == "SpectralQuant" and calibration is None:
                    continue
                r = evaluate_config(
                    config_name=name,
                    method=method,
                    avg_bits=avg_bits,
                    kv_cache=[(k.cpu(), v.cpu()) for k, v in kv_cache],
                    query_probes=query_probes,
                    sampled_layers=sampled_layers,
                    calibration=calibration,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    device=torch.device("cpu"),
                )
                all_results_per_config[name].append(r)

    # ------------------------------------------------------------------
    # Aggregate across sequences
    # ------------------------------------------------------------------
    final_results = []
    for name, method, avg_bits, rotation, bit_alloc, qjl in CONFIGS:
        entries = all_results_per_config[name]
        if not entries:
            log.warning("No results for config %s", name)
            continue
        cos_means = [e["cosine_similarity_mean"] for e in entries if e["cosine_similarity_mean"] is not None]
        err_means = [e["max_abs_weight_error_mean"] for e in entries if e["max_abs_weight_error_mean"] is not None]
        final_results.append({
            "config": name,
            "method": method,
            "avg_bits": avg_bits,
            "rotation": rotation,
            "bit_alloc": bit_alloc,
            "qjl": qjl,
            "cosine_similarity_mean": round(float(np.mean(cos_means)), 5) if cos_means else None,
            "cosine_similarity_std": round(float(np.std(cos_means)), 5) if cos_means else None,
            "max_abs_weight_error_mean": round(float(np.mean(err_means)), 6) if err_means else None,
            "n_sequences": len(entries),
        })

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    log.info("=== ATTENTION QUALITY RESULTS ===")
    header = f"{'Config':<8}  {'Method':<14}  {'Bits':>5}  {'CosSim':>8}  {'MaxAbsErr':>11}"
    log.info(header)
    log.info("-" * len(header))
    for r in final_results:
        cos = r["cosine_similarity_mean"]
        err = r["max_abs_weight_error_mean"]
        log.info("%-8s  %-14s  %5.1f  %8.5f  %11.6f",
                 r["config"], r["method"], r["avg_bits"],
                 cos if cos else 0.0, err if err else 0.0)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = results_dir / "attention_quality_results.csv"
    if final_results:
        fieldnames = list(final_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_results)
        log.info("Results table → %s", csv_path)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    if final_results:
        plot_comparison_bar(final_results, results_dir)

    # ------------------------------------------------------------------
    # Reflection Gate 2
    # ------------------------------------------------------------------
    log.info("=== REFLECTION GATE 2 ===")
    sq_results = {r["config"]: r for r in final_results if r["method"] == "SpectralQuant"}
    tq_results = {r["config"]: r for r in final_results if r["method"] == "TurboQuant"}

    gate2_comparisons = []
    # Same-bit comparisons: S-3.0 vs T-3.0, S-2.5 vs T-2.5, S-2.0 vs T-2.0
    bit_levels = [3.0, 2.5, 2.0]
    for bits in bit_levels:
        sq_name = f"S-{bits}"
        tq_name = f"T-{bits}"
        if sq_name in sq_results and tq_name in tq_results:
            s_cos = sq_results[sq_name]["cosine_similarity_mean"] or 0.0
            t_cos = tq_results[tq_name]["cosine_similarity_mean"] or 0.0
            gate2_comparisons.append({
                "bits": bits, "spectralquant_cos": s_cos,
                "turboquant_cos": t_cos, "sq_wins": s_cos >= t_cos,
            })
            log.info("  %g-bit: SpectralQuant %.5f vs TurboQuant %.5f — %s",
                     bits, s_cos, t_cos,
                     "SQ WINS" if s_cos >= t_cos else "TQ wins")

    gate2_pass = any(c["sq_wins"] for c in gate2_comparisons)
    if gate2_pass:
        print("\n" + "=" * 60)
        print("GATE 2 PASSED — SpectralQuant ≥ TurboQuant on at least one comparison")
        print("=" * 60 + "\n")
        log.info("GATE 2 PASSED")
    else:
        print("\n" + "=" * 60)
        print("GATE 2 FAILED — SpectralQuant does not beat TurboQuant at any bit level")
        print("\nDebug checklist:")
        print("  1. Were Lloyd-Max codebooks recomputed for non-uniform distributions?")
        print("  2. Try spectral rotation + UNIFORM allocation (Mod 1 only).")
        print("     If that helps → bit allocation implementation bug.")
        print("  3. Try random rotation + non-uniform allocation (Mod 2 only).")
        print("     If that helps → spectral rotation calibration bug.")
        print("  4. Check eigenspectral analysis: is κ > 5 for most heads?")
        print("=" * 60 + "\n")
        log.error("GATE 2 FAILED. See debug checklist above.")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    output = {
        "phase": "exp1_attention_quality",
        "sampled_layers": sampled_layers,
        "n_probe_queries": n_probe,
        "n_sequences": len(test_texts),
        "results": final_results,
        "gate2_comparisons": gate2_comparisons,
        "gate2_pass": gate2_pass,
        "wall_time_s": round(time.time() - t_total, 2),
    }
    with open(results_dir / "attention_quality_results.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info("Experiment 1 complete in %.1f s.", time.time() - t_total)

    if not gate2_pass and not args.quick:
        sys.exit(1)


if __name__ == "__main__":
    main()
