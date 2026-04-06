#!/usr/bin/env python3
"""
phase3_exp7_calibration_cost.py — Experiment 7: Calibration Cost

Measures calibration overhead for SpectralQuant:
  1. How many sequences are needed for stable eigenspectral estimation?
     (Run calibration at 10, 50, 100, 250, 500, 1000 sequences)
  2. Wall-clock time for each calibration set size
  3. Stability: run 3 different random calibration sets at 1000 sequences,
     report d_eff variance across runs

Saves calibration stability results as JSON and CSV.
Generates figures:
  - d_eff convergence vs calibration set size
  - d_eff variance across calibration sets (box plot)

Usage:
  python phase3_exp7_calibration_cost.py [--quick]
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

RESULTS_DIR = PROJECT_ROOT / "results" / "calibration_cost"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
CALIB_SET_SIZES = [10, 50, 100, 250, 500, 1000]
N_STABILITY_RUNS = 3
STABILITY_N_SEQS = 1000  # sequences per stability run

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
log = logging.getLogger("exp7_calibration_cost")


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
    log.warning("No GPU — calibration will be slower.")
    return torch.device("cpu")


def participation_ratio(eigenvalues: np.ndarray) -> float:
    lam = eigenvalues[eigenvalues > 0]
    if len(lam) == 0:
        return 0.0
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def spectral_gap(eigenvalues: np.ndarray, d_eff: float) -> float:
    lam = eigenvalues[eigenvalues > 0]
    k = max(1, min(int(round(d_eff)), len(lam) - 1))
    if lam[k] < 1e-12:
        return float("inf")
    return float(lam[k - 1] / lam[k])


# ---------------------------------------------------------------------------
# Model & data loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(device: torch.device):
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
    return model, tokenizer


def load_all_texts(n_total: int = 2000) -> list[str]:
    from datasets import load_dataset
    log.info("Loading WikiText-103 (up to %d texts) …", n_total)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [
        r["text"].strip() for r in dataset if len(r["text"].strip()) > 200
    ][:n_total]
    log.info("Loaded %d texts", len(texts))
    return texts


# ---------------------------------------------------------------------------
# KV collection & eigenspectral computation
# ---------------------------------------------------------------------------

def collect_and_compute_eigenspectrum(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    seq_len: int = 256,
    layers_to_monitor: list[int] | None = None,
    heads_to_monitor: list[int] | None = None,
) -> tuple[dict, float]:
    """
    Collect KV vectors for the given texts and compute eigenspectra.
    Returns (results dict, wall_clock_seconds).
    Only monitors specified layers and heads for efficiency.
    """
    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads

    if layers_to_monitor is None:
        layers_to_monitor = list(range(n_layers))
    if heads_to_monitor is None:
        heads_to_monitor = list(range(n_kv_heads))

    storage = {
        l: {h: {"keys": [], "values": []} for h in heads_to_monitor}
        for l in layers_to_monitor
    }

    t0 = time.time()

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt",
                max_length=seq_len, truncation=True,
            ).to(device)
            if enc["input_ids"].shape[1] < 8:
                continue

            out = model(**enc, use_cache=True)
            kv = out.past_key_values

            for l in layers_to_monitor:
                k_l = (kv.key_cache[l] if hasattr(kv, "key_cache") else kv[l][0]).float().cpu()
                v_l = (kv.value_cache[l] if hasattr(kv, "key_cache") else kv[l][1]).float().cpu()
                for h in heads_to_monitor:
                    storage[l][h]["keys"].append(k_l[0, h].numpy())
                    storage[l][h]["values"].append(v_l[0, h].numpy())

    # Concatenate and compute eigenspectra
    results = {}
    for l in layers_to_monitor:
        results[l] = {}
        for h in heads_to_monitor:
            K = np.concatenate(storage[l][h]["keys"], axis=0)
            V = np.concatenate(storage[l][h]["values"], axis=0)

            for modality, X in [("keys", K), ("values", V)]:
                mu = X.mean(axis=0)
                Xc = X - mu
                C = (Xc.T @ Xc) / len(X)
                eigvals, _ = np.linalg.eigh(C)
                eigvals = eigvals[::-1]

                d_eff = participation_ratio(eigvals)
                kappa = spectral_gap(eigvals, d_eff)

                if modality not in results[l]:
                    results[l][modality] = {}
                results[l][modality][h] = {
                    "d_eff": d_eff,
                    "kappa": float(kappa) if np.isfinite(kappa) else 999.0,
                }

    elapsed = time.time() - t0
    return results, elapsed


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------

def compute_deff_stats(results: dict, layers: list[int], heads: list[int]) -> dict:
    """Aggregate d_eff across all layers/heads for keys and values."""
    k_deffs, v_deffs, k_kappas, v_kappas = [], [], [], []
    for l in layers:
        for h in heads:
            k_deffs.append(results[l]["keys"][h]["d_eff"])
            v_deffs.append(results[l]["values"][h]["d_eff"])
            k_kappas.append(results[l]["keys"][h]["kappa"])
            v_kappas.append(results[l]["values"][h]["kappa"])

    return {
        "key_d_eff_mean": float(np.mean(k_deffs)),
        "key_d_eff_std": float(np.std(k_deffs)),
        "val_d_eff_mean": float(np.mean(v_deffs)),
        "val_d_eff_std": float(np.std(v_deffs)),
        "key_kappa_mean": float(np.mean([k for k in k_kappas if k < 999])) if any(k < 999 for k in k_kappas) else 0.0,
        "val_kappa_mean": float(np.mean([k for k in v_kappas if k < 999])) if any(k < 999 for k in v_kappas) else 0.0,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_convergence(convergence_data: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.3,
    })

    ns = [r["n_seqs"] for r in convergence_data]
    k_means = [r["key_d_eff_mean"] for r in convergence_data]
    k_stds = [r["key_d_eff_std"] for r in convergence_data]
    v_means = [r["val_d_eff_mean"] for r in convergence_data]
    v_stds = [r["val_d_eff_std"] for r in convergence_data]
    times = [r["wall_time_s"] for r in convergence_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # d_eff convergence
    ax1.errorbar(ns, k_means, yerr=k_stds, color="#1f77b4", marker="o",
                 linewidth=2, capsize=4, label="Key $d_{eff}$")
    ax1.errorbar(ns, v_means, yerr=v_stds, color="#d62728", marker="s",
                 linewidth=2, capsize=4, linestyle="--", label="Value $d_{eff}$")
    ax1.set_xlabel("Number of calibration sequences")
    ax1.set_ylabel("$d_{eff}$ (mean ± std across heads/layers)")
    ax1.set_title("$d_{eff}$ Convergence vs Calibration Set Size")
    ax1.legend()

    # Wall-clock time
    ax2.plot(ns, times, color="#2ca02c", marker="^", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of calibration sequences")
    ax2.set_ylabel("Calibration time (s)")
    ax2.set_title("Calibration Wall-Clock Time")
    for n, t in zip(ns, times):
        ax2.annotate(f"{t:.0f}s", (n, t), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)

    fig.suptitle(
        "Calibration Cost Analysis — Qwen 2.5-1.5B\n"
        "How many sequences are needed for stable eigenspectral estimation?",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / "fig_calibration_convergence.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Convergence figure → %s", out_path)


def plot_stability(stability_data: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.3,
    })

    if not stability_data:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    k_vals = [r["key_d_eff_mean"] for r in stability_data]
    v_vals = [r["val_d_eff_mean"] for r in stability_data]

    ax.bar(
        [0, 1],
        [np.mean(k_vals), np.mean(v_vals)],
        yerr=[np.std(k_vals), np.std(v_vals)],
        color=["#1f77b4", "#d62728"],
        edgecolor="black", linewidth=0.7, capsize=8, alpha=0.88,
        tick_label=["Key $d_{eff}$", "Value $d_{eff}$"],
    )
    for i, vals in enumerate([k_vals, v_vals]):
        for j, v in enumerate(vals):
            ax.scatter([i] * len(vals), vals, color="black", s=40, zorder=5)

    ax.set_ylabel("$d_{eff}$")
    ax.set_title(
        f"Calibration Stability — {len(stability_data)} independent runs\n"
        f"Each run: {STABILITY_N_SEQS} sequences from WikiText-103"
    )

    plt.tight_layout()
    out_path = out_dir / "fig_calibration_stability.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Stability figure → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp 7: Calibration cost analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: fewer calib sizes, fewer stability runs")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--n-total-texts", type=int, default=2000,
                   help="Total texts to preload for calibration experiments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--n-monitor-layers", type=int, default=4,
                   help="Number of layers to monitor (for speed)")
    p.add_argument("--n-monitor-heads", type=int, default=2,
                   help="Number of heads per layer to monitor (for speed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    set_seed(args.seed)
    device = detect_device()

    calib_sizes = [10, 50, 100] if args.quick else CALIB_SET_SIZES
    n_stability_runs = 2 if args.quick else N_STABILITY_RUNS
    stability_n = 100 if args.quick else STABILITY_N_SEQS

    log.info("Experiment 7: Calibration Cost")
    log.info("  calib_sizes=%s  n_stability_runs=%d", calib_sizes, n_stability_runs)

    # ------------------------------------------------------------------
    # Load model and all texts upfront
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(device)
    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads

    # Monitor a subset of layers and heads for efficiency
    monitor_layers = sorted(
        np.linspace(0, n_layers - 1, args.n_monitor_layers, dtype=int).tolist()
    )
    monitor_heads = list(range(min(args.n_monitor_heads, n_kv_heads)))

    n_total = max(max(calib_sizes), stability_n) + 50
    all_texts = load_all_texts(n_total=n_total)
    log.info("Monitoring layers=%s  heads=%s", monitor_layers, monitor_heads)

    # ------------------------------------------------------------------
    # Part 1: Convergence analysis
    # ------------------------------------------------------------------
    log.info("=== Part 1: Convergence Analysis ===")
    convergence_data = []

    for n_seqs in calib_sizes:
        log.info("  Calibrating with %d sequences …", n_seqs)
        texts_subset = all_texts[:n_seqs]
        results, elapsed = collect_and_compute_eigenspectrum(
            model, tokenizer, texts_subset, device,
            seq_len=args.seq_len,
            layers_to_monitor=monitor_layers,
            heads_to_monitor=monitor_heads,
        )
        stats = compute_deff_stats(results, monitor_layers, monitor_heads)
        entry = {
            "n_seqs": n_seqs,
            "wall_time_s": round(elapsed, 2),
            **{k: round(v, 4) for k, v in stats.items()},
        }
        convergence_data.append(entry)
        log.info(
            "    n=%d  time=%.1fs  key_d_eff=%.2f±%.2f  val_d_eff=%.2f±%.2f",
            n_seqs, elapsed,
            stats["key_d_eff_mean"], stats["key_d_eff_std"],
            stats["val_d_eff_mean"], stats["val_d_eff_std"],
        )

    # Save convergence CSV
    csv_path = results_dir / "convergence_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(convergence_data[0].keys()))
        writer.writeheader()
        writer.writerows(convergence_data)
    log.info("Convergence CSV → %s", csv_path)

    # ------------------------------------------------------------------
    # Part 2: Stability analysis (3 random calibration sets)
    # ------------------------------------------------------------------
    log.info("=== Part 2: Stability Analysis (%d runs × %d seqs) ===",
             n_stability_runs, stability_n)
    stability_data = []

    for run_i in range(n_stability_runs):
        run_seed = args.seed + run_i * 100
        set_seed(run_seed)

        # Random subsample of texts
        texts_run = random.sample(all_texts, min(stability_n, len(all_texts)))
        log.info("  Stability run %d/%d (seed=%d, n=%d) …",
                 run_i + 1, n_stability_runs, run_seed, len(texts_run))

        results, elapsed = collect_and_compute_eigenspectrum(
            model, tokenizer, texts_run, device,
            seq_len=args.seq_len,
            layers_to_monitor=monitor_layers,
            heads_to_monitor=monitor_heads,
        )
        stats = compute_deff_stats(results, monitor_layers, monitor_heads)
        entry = {
            "run": run_i,
            "seed": run_seed,
            "n_seqs": len(texts_run),
            "wall_time_s": round(elapsed, 2),
            **{k: round(v, 4) for k, v in stats.items()},
        }
        stability_data.append(entry)
        log.info(
            "    run=%d  time=%.1fs  key_d_eff=%.2f  val_d_eff=%.2f",
            run_i, elapsed, stats["key_d_eff_mean"], stats["val_d_eff_mean"],
        )

    # Compute stability statistics
    k_deffs = [r["key_d_eff_mean"] for r in stability_data]
    v_deffs = [r["val_d_eff_mean"] for r in stability_data]
    stability_summary = {
        "n_runs": n_stability_runs,
        "n_seqs_per_run": stability_n,
        "key_d_eff_mean": round(float(np.mean(k_deffs)), 4),
        "key_d_eff_std": round(float(np.std(k_deffs)), 4),
        "key_d_eff_cv": round(float(np.std(k_deffs) / np.mean(k_deffs)), 4) if np.mean(k_deffs) > 0 else 0.0,
        "val_d_eff_mean": round(float(np.mean(v_deffs)), 4),
        "val_d_eff_std": round(float(np.std(v_deffs)), 4),
        "val_d_eff_cv": round(float(np.std(v_deffs) / np.mean(v_deffs)), 4) if np.mean(v_deffs) > 0 else 0.0,
        "is_stable": float(np.std(k_deffs) / np.mean(k_deffs) < 0.10) if np.mean(k_deffs) > 0 else False,
    }
    log.info("=== STABILITY SUMMARY ===")
    log.info("  Key d_eff: %.3f ± %.3f  (CV=%.1f%%)",
             stability_summary["key_d_eff_mean"],
             stability_summary["key_d_eff_std"],
             stability_summary["key_d_eff_cv"] * 100)
    log.info("  Val d_eff: %.3f ± %.3f  (CV=%.1f%%)",
             stability_summary["val_d_eff_mean"],
             stability_summary["val_d_eff_std"],
             stability_summary["val_d_eff_cv"] * 100)
    if stability_summary["is_stable"]:
        log.info("  d_eff is STABLE (CV < 10%%) — calibration robust to dataset choice.")
    else:
        log.warning(
            "  d_eff UNSTABLE (CV >= 10%%) — consider larger calibration set "
            "or EMA covariance update."
        )

    # Save stability CSV
    csv_path2 = results_dir / "stability_results.csv"
    with open(csv_path2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stability_data[0].keys()))
        writer.writeheader()
        writer.writerows(stability_data)
    log.info("Stability CSV → %s", csv_path2)

    # ------------------------------------------------------------------
    # Print convergence table
    # ------------------------------------------------------------------
    log.info("=== CONVERGENCE RESULTS ===")
    log.info("  %8s  %8s  %12s  %12s", "N_seqs", "Time(s)", "Key_d_eff", "Val_d_eff")
    for r in convergence_data:
        log.info("  %8d  %8.1f  %12.3f  %12.3f",
                 r["n_seqs"], r["wall_time_s"],
                 r["key_d_eff_mean"], r["val_d_eff_mean"])

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_convergence(convergence_data, results_dir)
    plot_stability(stability_data, results_dir)

    # ------------------------------------------------------------------
    # Save full results
    # ------------------------------------------------------------------
    output = {
        "phase": "exp7_calibration_cost",
        "n_monitor_layers": args.n_monitor_layers,
        "n_monitor_heads": args.n_monitor_heads,
        "monitor_layers": monitor_layers,
        "monitor_heads": monitor_heads,
        "convergence_data": convergence_data,
        "stability_data": stability_data,
        "stability_summary": stability_summary,
        "wall_time_s": round(time.time() - t_total, 2),
    }
    with open(results_dir / "calibration_cost_results.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info("Experiment 7 complete in %.1f s.", time.time() - t_total)


if __name__ == "__main__":
    main()
