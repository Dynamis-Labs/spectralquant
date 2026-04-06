#!/usr/bin/env python3
"""
phase1_eigenspectral.py — Eigenspectral Discovery (THE FOUNDATION)

For every layer and attention head of Qwen 2.5-1.5B, collects key/value vectors
over 1000 WikiText-103 sequences and computes the eigenspectral structure:
  - Covariance matrix, eigenvalues, eigenvectors
  - Effective dimensionality d_eff (participation ratio)
  - Spectral gap κ
  - Cumulative variance curves

Produces CSV tables, publication-quality PNG figures, and calibration data
(eigenvectors + eigenvalues) saved for use in Phase 2.

REFLECTION GATE 1:
  κ > 10 (majority of heads) → "GATE 1 PASSED" — strong gap, proceed
  κ 5–10                     → warning, proceed with caution
  κ < 5  (majority of heads) → "GATE 1 FAILED" — instructions printed

Usage:
  python phase1_eigenspectral.py [--quick] [--n-seqs N] [--seq-len L]
"""

import argparse
import json
import logging
import os
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

RESULTS_DIR = PROJECT_ROOT / "results" / "eigenspectral"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

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
log = logging.getLogger("phase1")


# ---------------------------------------------------------------------------
# Helpers
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
        log.warning("No GPU detected — running on CPU. This will be slow.")
    return device


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Effective dimensionality via participation ratio:
      d_eff = (Σ λ_i)² / Σ(λ_i²)

    Only uses positive eigenvalues.
    """
    lam = eigenvalues[eigenvalues > 0]
    if len(lam) == 0:
        return 0.0
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def spectral_gap(eigenvalues: np.ndarray, d_eff: float) -> float:
    """
    Spectral gap κ = λ_{k} / λ_{k+1}  where k = round(d_eff).
    """
    lam = eigenvalues[eigenvalues > 0]
    k = max(1, min(int(round(d_eff)), len(lam) - 1))
    if lam[k] < 1e-12:
        return float("inf")
    return float(lam[k - 1] / lam[k])


def cumulative_variance(eigenvalues: np.ndarray) -> np.ndarray:
    lam = np.abs(eigenvalues)
    total = lam.sum()
    if total == 0:
        return np.zeros_like(lam)
    return np.cumsum(lam) / total


def dims_for_variance(eigenvalues: np.ndarray, thresholds=(0.90, 0.95, 0.99)) -> dict:
    cumvar = cumulative_variance(eigenvalues)
    return {
        f"dims_{int(t*100)}pct": int(np.searchsorted(cumvar, t) + 1)
        for t in thresholds
    }


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    log.info("Loading model …")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    log.info("Model loaded: %d layers, %d KV heads, head_dim=%d",
             model.config.num_hidden_layers,
             model.config.num_key_value_heads,
             model.config.hidden_size // model.config.num_attention_heads)
    return model, tokenizer


def load_calibration_texts(n_seqs: int) -> list[str]:
    from datasets import load_dataset
    log.info("Loading WikiText-103 (%d sequences) …", n_seqs)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=False)
    texts = [
        row["text"].strip()
        for row in dataset
        if len(row["text"].strip()) > 200
    ][:n_seqs]
    log.info("Loaded %d texts", len(texts))
    return texts


def collect_kv_vectors(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    seq_len: int = 256,
    max_seqs: int | None = None,
) -> dict:
    """
    Returns:
      {
        layer_idx: {
          head_idx: {
            'keys':   np.ndarray [n_tokens_total, head_dim]
            'values': np.ndarray [n_tokens_total, head_dim]
          }
        }
      }
    """
    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads
    n_attn_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_attn_heads

    # GQA: each KV head is shared by (n_attn_heads // n_kv_heads) query heads
    kv_per_attn = n_attn_heads // n_kv_heads

    # Accumulate lists; convert to arrays at the end
    storage: dict = {
        l: {h: {"keys": [], "values": []} for h in range(n_kv_heads)}
        for l in range(n_layers)
    }

    seqs = texts[:max_seqs] if max_seqs else texts
    log.info("Collecting KV vectors from %d sequences (seq_len≤%d) …", len(seqs), seq_len)

    t0 = time.time()
    n_processed = 0
    with torch.no_grad():
        for i, text in enumerate(seqs):
            if (i + 1) % 100 == 0:
                log.info("  Processed %d/%d sequences (%.1f s elapsed)",
                         i + 1, len(seqs), time.time() - t0)
            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
                padding=False,
            ).to(device)
            if enc["input_ids"].shape[1] < 8:
                continue

            out = model(**enc, use_cache=True, output_hidden_states=False)
            kv = out.past_key_values  # DynamicCache or list[(k,v)]

            for l in range(n_layers):
                try:
                    k_l = kv.key_cache[l].float().cpu()
                    v_l = kv.value_cache[l].float().cpu()
                except (AttributeError, TypeError):
                    try:
                        k_l = kv[l][0].float().cpu()
                        v_l = kv[l][1].float().cpu()
                    except TypeError:
                        layer_kv = list(kv)[l]
                        k_l = layer_kv[0].float().cpu()
                        v_l = layer_kv[1].float().cpu()
                T = k_l.shape[2]
                for h in range(n_kv_heads):
                    # k_l shape: [1, n_kv_heads, T, head_dim]
                    storage[l][h]["keys"].append(k_l[0, h, :, :].numpy())   # [T, head_dim]
                    storage[l][h]["values"].append(v_l[0, h, :, :].numpy())
            n_processed += 1

    log.info("Collection complete: %d sequences, %.1f s total", n_processed, time.time() - t0)

    # Concatenate
    log.info("Concatenating collected vectors …")
    for l in range(n_layers):
        for h in range(n_kv_heads):
            storage[l][h]["keys"] = np.concatenate(storage[l][h]["keys"], axis=0)
            storage[l][h]["values"] = np.concatenate(storage[l][h]["values"], axis=0)
            n_tok = storage[l][h]["keys"].shape[0]
    log.info("Total tokens per (layer, head): ~%d", n_tok)
    return storage


# ---------------------------------------------------------------------------
# Eigenspectral computation
# ---------------------------------------------------------------------------

def compute_eigenspectrum(X: np.ndarray) -> dict:
    """
    Given X of shape [n, d], compute:
      - covariance C = X^T X / n  (zero-centered)
      - eigenvalues (sorted descending)
      - eigenvectors
      - d_eff, κ, cumulative variance, dims_90/95/99pct
    """
    n, d = X.shape
    # Center
    mu = X.mean(axis=0)
    Xc = X - mu

    # Covariance
    C = (Xc.T @ Xc) / n  # [d, d]

    # Eigen-decomposition (symmetric)
    # Use SVD of Xc for numerical stability when n << d is not the case
    # For d=128 (head_dim), direct eigh is fast and stable
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending order
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]  # columns are eigenvectors, shape [d, d]

    d_eff = participation_ratio(eigvals)
    kappa = spectral_gap(eigvals, d_eff)
    cumvar = cumulative_variance(eigvals)
    dims_info = dims_for_variance(eigvals)

    return {
        "eigenvalues": eigvals,       # [d]
        "eigenvectors": eigvecs,      # [d, d] — columns are eigenvectors
        "mean": mu,                   # [d]
        "d_eff": d_eff,
        "kappa": kappa,
        "cumvar": cumvar,             # [d]
        **dims_info,
    }


def run_eigenspectral_analysis(kv_storage: dict, n_layers: int, n_kv_heads: int) -> dict:
    """
    Compute eigenspectrum for every (layer, head, key/value) triple.

    Returns nested dict: results[layer][head] = {'keys': {...}, 'values': {...}}
    """
    results = {}
    log.info("Computing eigenspectral analysis for %d layers × %d KV heads …",
             n_layers, n_kv_heads)
    t0 = time.time()

    for l in range(n_layers):
        results[l] = {}
        for h in range(n_kv_heads):
            K = kv_storage[l][h]["keys"]    # [n_tokens, head_dim]
            V = kv_storage[l][h]["values"]
            results[l][h] = {
                "keys": compute_eigenspectrum(K),
                "values": compute_eigenspectrum(V),
            }
        if (l + 1) % 4 == 0:
            log.info("  Completed layer %d/%d (%.1f s)", l + 1, n_layers, time.time() - t0)

    log.info("Eigenspectral analysis complete in %.1f s", time.time() - t0)
    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(eigen_results: dict, n_layers: int, n_kv_heads: int) -> dict:
    all_k_deff, all_v_deff, all_k_kappa, all_v_kappa = [], [], [], []

    for l in range(n_layers):
        for h in range(n_kv_heads):
            all_k_deff.append(eigen_results[l][h]["keys"]["d_eff"])
            all_v_deff.append(eigen_results[l][h]["values"]["d_eff"])
            k_kap = eigen_results[l][h]["keys"]["kappa"]
            v_kap = eigen_results[l][h]["values"]["kappa"]
            if np.isfinite(k_kap):
                all_k_kappa.append(k_kap)
            if np.isfinite(v_kap):
                all_v_kappa.append(v_kap)

    return {
        "keys": {
            "mean_d_eff": float(np.mean(all_k_deff)),
            "min_d_eff": float(np.min(all_k_deff)),
            "max_d_eff": float(np.max(all_k_deff)),
            "median_d_eff": float(np.median(all_k_deff)),
            "mean_kappa": float(np.mean(all_k_kappa)) if all_k_kappa else None,
            "median_kappa": float(np.median(all_k_kappa)) if all_k_kappa else None,
            "pct_kappa_gt5": float(np.mean([k > 5 for k in all_k_kappa])) if all_k_kappa else None,
            "pct_kappa_gt10": float(np.mean([k > 10 for k in all_k_kappa])) if all_k_kappa else None,
        },
        "values": {
            "mean_d_eff": float(np.mean(all_v_deff)),
            "min_d_eff": float(np.min(all_v_deff)),
            "max_d_eff": float(np.max(all_v_deff)),
            "median_d_eff": float(np.median(all_v_deff)),
            "mean_kappa": float(np.mean(all_v_kappa)) if all_v_kappa else None,
            "median_kappa": float(np.median(all_v_kappa)) if all_v_kappa else None,
            "pct_kappa_gt5": float(np.mean([k > 5 for k in all_v_kappa])) if all_v_kappa else None,
            "pct_kappa_gt10": float(np.mean([k > 10 for k in all_v_kappa])) if all_v_kappa else None,
        },
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_deff_table(eigen_results: dict, n_layers: int, n_kv_heads: int, out_dir: Path) -> None:
    import csv
    rows = []
    for l in range(n_layers):
        k_deffs = [eigen_results[l][h]["keys"]["d_eff"] for h in range(n_kv_heads)]
        v_deffs = [eigen_results[l][h]["values"]["d_eff"] for h in range(n_kv_heads)]
        k_kappas = [eigen_results[l][h]["keys"]["kappa"] for h in range(n_kv_heads)]
        v_kappas = [eigen_results[l][h]["values"]["kappa"] for h in range(n_kv_heads)]
        rows.append({
            "layer": l,
            "key_d_eff_mean": round(float(np.mean(k_deffs)), 3),
            "key_d_eff_std": round(float(np.std(k_deffs)), 3),
            "val_d_eff_mean": round(float(np.mean(v_deffs)), 3),
            "val_d_eff_std": round(float(np.std(v_deffs)), 3),
            "key_kappa_mean": round(float(np.nanmean([k for k in k_kappas if np.isfinite(k)])), 3)
                              if any(np.isfinite(k) for k in k_kappas) else "inf",
            "val_kappa_mean": round(float(np.nanmean([k for k in v_kappas if np.isfinite(k)])), 3)
                              if any(np.isfinite(k) for k in v_kappas) else "inf",
        })

    csv_path = out_dir / "deff_per_layer.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved d_eff table → %s", csv_path)

    # Print formatted to console
    header = f"{'Layer':>5}  {'Key d_eff':>12}  {'Val d_eff':>12}  {'Key κ':>8}  {'Val κ':>8}"
    log.info("\n%s\n%s", header, "-" * len(header))
    for r in rows:
        log.info("%5d  %12.3f  %12.3f  %8s  %8s",
                 r["layer"], r["key_d_eff_mean"], r["val_d_eff_mean"],
                 r["key_kappa_mean"], r["val_kappa_mean"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_eigenvalue_spectra(
    eigen_results: dict,
    n_layers: int,
    n_kv_heads: int,
    out_dir: Path,
    n_representatives: int = 6,
) -> None:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    # Pick representative (layer, head) pairs: spread across early, middle, late layers
    layer_picks = np.linspace(0, n_layers - 1, n_representatives, dtype=int).tolist()
    head_pick = n_kv_heads // 2

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    axes = axes.flatten()

    for idx, l in enumerate(layer_picks):
        ax = axes[idx]
        k_eigs = eigen_results[l][head_pick]["keys"]["eigenvalues"]
        v_eigs = eigen_results[l][head_pick]["values"]["eigenvalues"]
        dims = np.arange(1, len(k_eigs) + 1)

        ax.semilogy(dims, k_eigs, color="#1f77b4", linewidth=1.5, label="Keys")
        ax.semilogy(dims, np.abs(v_eigs), color="#d62728", linewidth=1.5,
                    linestyle="--", label="Values")

        # Mark d_eff for keys
        d_eff_k = eigen_results[l][head_pick]["keys"]["d_eff"]
        ax.axvline(x=d_eff_k, color="#1f77b4", linestyle=":", alpha=0.7,
                   label=f"$d_{{eff}}^K={d_eff_k:.1f}$")
        d_eff_v = eigen_results[l][head_pick]["values"]["d_eff"]
        ax.axvline(x=d_eff_v, color="#d62728", linestyle=":", alpha=0.7,
                   label=f"$d_{{eff}}^V={d_eff_v:.1f}$")

        ax.set_title(f"Layer {l}, Head {head_pick}")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue (log scale)")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Eigenvalue Spectra — Qwen 2.5-1.5B KV Representations\n"
        f"Representative layers across 0–{n_layers-1}",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / "fig_eigenvalue_spectra.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved eigenvalue spectra figure → %s", out_path)


def plot_deff_heatmap(
    eigen_results: dict,
    n_layers: int,
    n_kv_heads: int,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_matplotlib()

    k_deff = np.zeros((n_layers, n_kv_heads))
    v_deff = np.zeros((n_layers, n_kv_heads))
    for l in range(n_layers):
        for h in range(n_kv_heads):
            k_deff[l, h] = eigen_results[l][h]["keys"]["d_eff"]
            v_deff[l, h] = eigen_results[l][h]["values"]["d_eff"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, n_layers * 0.35)))

    for ax, data, title in [
        (ax1, k_deff, "Key $d_{eff}$"),
        (ax2, v_deff, "Value $d_{eff}$"),
    ]:
        sns.heatmap(
            data,
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=data.max(),
            xticklabels=True if n_kv_heads <= 32 else 4,
            yticklabels=2,
            cbar_kws={"label": "$d_{eff}$"},
            annot=n_layers <= 16 and n_kv_heads <= 8,
            fmt=".1f",
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("KV Head index")
        ax.set_ylabel("Layer index")

    fig.suptitle(
        f"Effective Dimensionality $d_{{eff}}$ across (Layer, KV Head)\n"
        f"Qwen 2.5-1.5B — participation ratio $d_{{eff}} = (\\Sigma\\lambda_i)^2 / \\Sigma\\lambda_i^2$",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / "fig_deff_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved d_eff heatmap → %s", out_path)


def plot_cumulative_variance(
    eigen_results: dict,
    n_layers: int,
    n_kv_heads: int,
    out_dir: Path,
    n_representatives: int = 5,
) -> None:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    layer_picks = np.linspace(0, n_layers - 1, n_representatives, dtype=int).tolist()
    head_pick = n_kv_heads // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_representatives))

    for idx, l in enumerate(layer_picks):
        # Keys
        cumvar_k = eigen_results[l][head_pick]["keys"]["cumvar"]
        dims = np.arange(1, len(cumvar_k) + 1)
        ax1.plot(dims, cumvar_k * 100, color=colors[idx], linewidth=1.8,
                 label=f"Layer {l}")

        # Values
        cumvar_v = eigen_results[l][head_pick]["values"]["cumvar"]
        ax2.plot(dims, cumvar_v * 100, color=colors[idx], linewidth=1.8,
                 label=f"Layer {l}")

    for ax, title in [(ax1, "Keys"), (ax2, "Values")]:
        ax.axhline(y=90, color="gray", linestyle=":", alpha=0.7, label="90%")
        ax.axhline(y=95, color="gray", linestyle="--", alpha=0.7, label="95%")
        ax.axhline(y=99, color="gray", linestyle="-.", alpha=0.7, label="99%")
        ax.set_xlabel("Number of principal components")
        ax.set_ylabel("Cumulative variance explained (%)")
        ax.set_title(f"Cumulative Variance — {title}")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(1, None)
        ax.set_ylim(0, 101)

    fig.suptitle(
        "Cumulative Variance Explained by Principal Components\n"
        f"Qwen 2.5-1.5B, KV Head {head_pick}, representative layers",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / "fig_cumulative_variance.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved cumulative variance figure → %s", out_path)


# ---------------------------------------------------------------------------
# Save calibration data for Phase 2
# ---------------------------------------------------------------------------

def save_calibration_data(eigen_results: dict, n_layers: int, n_kv_heads: int, out_dir: Path) -> None:
    """
    Saves per-(layer, head) eigenvectors and eigenvalues as .npz files.
    Also saves a compact metadata JSON.
    """
    calib_dir = out_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "model": MODEL_NAME,
        "format": "per_layer_head_npz",
    }

    log.info("Saving calibration data (eigenvectors + eigenvalues) …")
    t0 = time.time()
    for l in range(n_layers):
        for h in range(n_kv_heads):
            fname = calib_dir / f"layer{l:02d}_head{h:02d}.npz"
            k_res = eigen_results[l][h]["keys"]
            v_res = eigen_results[l][h]["values"]
            np.savez_compressed(
                fname,
                key_eigenvectors=k_res["eigenvectors"].astype(np.float32),
                key_eigenvalues=k_res["eigenvalues"].astype(np.float32),
                key_mean=k_res["mean"].astype(np.float32),
                key_d_eff=np.float32(k_res["d_eff"]),
                key_kappa=np.float32(k_res["kappa"]) if np.isfinite(k_res["kappa"]) else np.float32(1e6),
                val_eigenvectors=v_res["eigenvectors"].astype(np.float32),
                val_eigenvalues=v_res["eigenvalues"].astype(np.float32),
                val_mean=v_res["mean"].astype(np.float32),
                val_d_eff=np.float32(v_res["d_eff"]),
                val_kappa=np.float32(v_res["kappa"]) if np.isfinite(v_res["kappa"]) else np.float32(1e6),
            )

    with open(calib_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Calibration data saved to %s  (%.1f s)", calib_dir, time.time() - t0)


# ---------------------------------------------------------------------------
# Reflection Gate 1
# ---------------------------------------------------------------------------

def reflection_gate_1(summary: dict) -> str:
    """
    Evaluate spectral gap quality and return gate status string.
    """
    # Combine key + value kappa statistics
    k_pct_gt5 = summary["keys"].get("pct_kappa_gt5") or 0.0
    v_pct_gt5 = summary["values"].get("pct_kappa_gt5") or 0.0
    k_pct_gt10 = summary["keys"].get("pct_kappa_gt10") or 0.0
    v_pct_gt10 = summary["values"].get("pct_kappa_gt10") or 0.0

    mean_k_kappa = summary["keys"].get("mean_kappa") or 0.0
    mean_v_kappa = summary["values"].get("mean_kappa") or 0.0

    majority_gt5 = (k_pct_gt5 + v_pct_gt5) / 2 >= 0.5
    majority_gt10 = (k_pct_gt10 + v_pct_gt10) / 2 >= 0.5

    log.info("=== REFLECTION GATE 1 ===")
    log.info("  Mean key κ: %.2f   (%.0f%% of heads κ>5,  %.0f%% κ>10)",
             mean_k_kappa, k_pct_gt5 * 100, k_pct_gt10 * 100)
    log.info("  Mean val κ: %.2f   (%.0f%% of heads κ>5,  %.0f%% κ>10)",
             mean_v_kappa, v_pct_gt5 * 100, v_pct_gt10 * 100)

    if majority_gt10:
        print("\n" + "=" * 60)
        print("GATE 1 PASSED — Strong spectral gap (κ > 10 for majority of heads)")
        print("SpectralQuant will provide clear improvements over TurboQuant.")
        print("Proceed to Phase 2 with full confidence.")
        print("=" * 60 + "\n")
        log.info("GATE 1 PASSED — strong spectral gap (κ>10 majority)")
        return "PASSED_STRONG"
    elif majority_gt5:
        print("\n" + "=" * 60)
        print("GATE 1 WARNING — Moderate spectral gap (κ 5–10 for majority)")
        print("SpectralQuant improvements may be modest but consistent.")
        print("Proceeding with caution. Calibrate expectations.")
        print("=" * 60 + "\n")
        log.warning("GATE 1 PARTIAL — moderate spectral gap (κ 5–10 majority)")
        return "PASSED_MODERATE"
    else:
        print("\n" + "=" * 60)
        print("GATE 1 FAILED — Weak spectral gap (κ < 5 for majority of heads)")
        print(f"  Mean key κ: {mean_k_kappa:.2f}  |  Mean val κ: {mean_v_kappa:.2f}")
        print()
        print("Instructions:")
        print("  1. PIVOT: Consider emphasizing vector search as primary contribution.")
        print("     Embeddings have well-established sharp spectral gaps.")
        print("  2. INVESTIGATE: Try a larger calibration set (2000+ sequences).")
        print("  3. INVESTIGATE: Try per-token (rather than per-sequence) KV collection.")
        print("  4. If KV gap is weak, re-frame paper as 'KV application with mixed results'")
        print("     and lead with vector search benchmarks.")
        print("=" * 60 + "\n")
        log.error("GATE 1 FAILED — weak spectral gap. See instructions above.")
        return "FAILED"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: Eigenspectral discovery for Qwen 2.5-1.5B KV representations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="Reduced dataset (50 seqs, seq_len=128) for debugging")
    p.add_argument("--n-seqs", type=int, default=1000,
                   help="Number of WikiText-103 sequences")
    p.add_argument("--seq-len", type=int, default=256,
                   help="Maximum token length per sequence")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--skip-collection", action="store_true",
                   help="Skip KV collection (use previously saved data)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    set_seed(args.seed)
    device = detect_device()

    n_seqs = 50 if args.quick else args.n_seqs
    seq_len = 128 if args.quick else args.seq_len

    log.info("Phase 1 — Eigenspectral Discovery")
    log.info("  n_seqs=%d  seq_len=%d  device=%s", n_seqs, seq_len, device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(device)
    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    log.info("Model: %d layers, %d KV heads, head_dim=%d", n_layers, n_kv_heads, head_dim)

    # ------------------------------------------------------------------
    # Calibration data
    # ------------------------------------------------------------------
    texts = load_calibration_texts(n_seqs)

    # ------------------------------------------------------------------
    # KV collection
    # ------------------------------------------------------------------
    if not args.skip_collection:
        t0 = time.time()
        kv_storage = collect_kv_vectors(model, tokenizer, texts, device, seq_len=seq_len)
        log.info("KV collection: %.1f s", time.time() - t0)
        # Free model memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        log.warning("--skip-collection: KV collection step skipped.")
        kv_storage = None  # downstream steps will fail — intended for partial re-runs

    # ------------------------------------------------------------------
    # Eigenspectral analysis
    # ------------------------------------------------------------------
    t0 = time.time()
    eigen_results = run_eigenspectral_analysis(kv_storage, n_layers, n_kv_heads)
    log.info("Eigenspectral computation: %.1f s", time.time() - t0)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    summary = compute_summary_stats(eigen_results, n_layers, n_kv_heads)
    log.info("Summary statistics:")
    for modality in ("keys", "values"):
        s = summary[modality]
        log.info("  %s: d_eff mean=%.2f  min=%.2f  max=%.2f  |  κ mean=%.2f",
                 modality, s["mean_d_eff"], s["min_d_eff"], s["max_d_eff"],
                 s["mean_kappa"] or 0.0)

    summary_path = results_dir / "summary_statistics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved → %s", summary_path)

    # ------------------------------------------------------------------
    # CSV table
    # ------------------------------------------------------------------
    save_deff_table(eigen_results, n_layers, n_kv_heads, results_dir)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    log.info("Generating figures …")
    plot_eigenvalue_spectra(eigen_results, n_layers, n_kv_heads, results_dir)
    plot_deff_heatmap(eigen_results, n_layers, n_kv_heads, results_dir)
    plot_cumulative_variance(eigen_results, n_layers, n_kv_heads, results_dir)

    # ------------------------------------------------------------------
    # Save calibration data for Phase 2
    # ------------------------------------------------------------------
    save_calibration_data(eigen_results, n_layers, n_kv_heads, results_dir)

    # ------------------------------------------------------------------
    # Reflection Gate 1
    # ------------------------------------------------------------------
    gate_status = reflection_gate_1(summary)

    # ------------------------------------------------------------------
    # Final metadata
    # ------------------------------------------------------------------
    meta = {
        "phase": "phase1_eigenspectral",
        "model": MODEL_NAME,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "gate1_status": gate_status,
        "summary": summary,
        "wall_time_s": round(time.time() - t_total, 2),
        "figures": [
            str(results_dir / "fig_eigenvalue_spectra.png"),
            str(results_dir / "fig_deff_heatmap.png"),
            str(results_dir / "fig_cumulative_variance.png"),
        ],
        "calibration_dir": str(results_dir / "calibration"),
    }
    with open(results_dir / "phase1_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if gate_status == "FAILED":
        sys.exit(1)

    log.info("Phase 1 complete in %.1f s. Results in %s", time.time() - t_total, results_dir)


if __name__ == "__main__":
    main()
