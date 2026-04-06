"""
Asymmetric Shaped KV Cache: Exploiting Key/Value Spectral Asymmetry
====================================================================
Keys have d_eff ~ 4, values have d_eff ~ 40-55. Instead of storing all 128
dimensions for both, store only m dimensions for keys and p dimensions for
values after spectral rotation. This is pure truncation in the eigenbasis --
the spectral rotation ensures we keep the highest-variance directions.

Sweep:
  m (key dims):   {2, 4, 8, 16, 32}
  p (value dims): {8, 16, 32, 48, 64, 96, 128}
  quant:          {FP16 (baseline), 4-bit, 3-bit, 2-bit}
  models:         Qwen 2.5-1.5B, Qwen 2.5-7B, Llama 3.1-8B

Metrics at each (m, p, quant) point:
  - Attention output cosine similarity vs FP16 (no compression)
  - Compression ratio (bits per token vs FP16 baseline)
  - Key reconstruction cosine similarity
  - Value reconstruction cosine similarity

Theory: at m=4, p=32, 2-bit quant, we get:
  keys:   4 dims * 2 bits = 8 bits   + 16 bits norm = 24 bits
  values: 32 dims * 3 bits = 96 bits + 16 bits norm = 112 bits
  total = 136 bits/token vs 4096 bits FP16 = 30.1x compression

Usage:
    python shaped_cache_sweep.py [--quick] [--device cuda]

Estimated runtime: ~60 min full, ~10 min quick (on B200)
"""

import sys, os, math, time, json, logging, argparse, gc
from pathlib import Path
from itertools import product

import torch
import torch.nn.functional as F
import numpy as np

# -- project paths --
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from turboquant_cutile import TurboQuantEngine, LloydMaxCodebook

log = logging.getLogger("shaped_cache")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

RESULTS_DIR = PROJECT_ROOT / "results" / "shaped_cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen-1.5B"),
    ("Qwen/Qwen2.5-7B-Instruct",   "Qwen-7B"),
]
if HF_TOKEN:
    MODELS.append(("meta-llama/Llama-3.1-8B-Instruct", "Llama-8B"))

M_VALUES = [2, 4, 8, 16, 32]
P_VALUES = [8, 16, 32, 48, 64, 96, 128]
QUANT_BITS = [0, 4, 3, 2]

N_CALIB = 64
N_EVAL = 32
SEQ_LEN = 512

# Quick overrides
N_CALIB_QUICK = 16
N_EVAL_QUICK = 8
M_VALUES_QUICK = [4, 8, 16]
P_VALUES_QUICK = [16, 32, 64, 128]
QUANT_BITS_QUICK = [0, 3, 2]


# ============================================================================
# UTILITIES
# ============================================================================

def save_result(filename, data):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved: %s", path)


def quantize_nearest(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Scalar nearest-centroid quantization."""
    c = centroids.to(x.device)
    diffs = x.unsqueeze(-1) - c
    return c[diffs.abs().argmin(dim=-1).long()]


def solve_lloyd_max_for_sigma(sigma: float, bits: int, max_iter=200, tol=1e-10):
    """Solve Lloyd-Max for N(0, sigma^2)."""
    from scipy import integrate

    n_levels = 1 << bits
    pdf = lambda x: (
        (1.0 / (math.sqrt(2 * math.pi) * sigma))
        * math.exp(-x * x / (2 * sigma * sigma))
    )
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids
    return torch.tensor(centroids, dtype=torch.float32)


def load_model_tokenizer(model_name: str, device: str):
    """Load a HuggingFace model + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    needs_token = any(x in model_name for x in ["llama", "Llama", "gemma", "Gemma"])
    token = HF_TOKEN if needs_token else None

    log.info("Loading %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        token=token,
    )
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)

    # Detect actual KV head_dim from a forward pass
    try:
        test_ids = tokenizer("test", return_tensors="pt").to(device)
        with torch.no_grad():
            test_out = model(**test_ids, use_cache=True)
        kv = test_out.past_key_values
        try:
            actual_kv_hd = kv.key_cache[0].shape[-1]
        except:
            try:
                actual_kv_hd = kv[0][0].shape[-1]
            except:
                actual_kv_hd = hd
        if actual_kv_hd != hd:
            log.info("KV head_dim=%d differs from Q head_dim=%d, using KV head_dim", actual_kv_hd, hd)
            hd = actual_kv_hd
    except Exception as e:
        log.warning("Could not detect KV head_dim: %s", e)

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info("%s: %d layers, %d kv_heads, head_dim=%d, %.1fB params",
             model_name, n_layers, n_kv, hd, n_params)

    return model, tokenizer, n_layers, n_kv, hd


def extract_kv_layer(kv, l):
    """Extract (key, value) tensors for layer l from past_key_values."""
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


# ============================================================================
# CALIBRATION: per-(layer, head) eigenvectors for KEYS and VALUES separately
# Uses the same method as neurips_models_asymmetry.py (accumulate X^T X)
# ============================================================================

def calibrate_keys_and_values(model, tokenizer, n_calib, device, n_layers, n_kv, hd):
    """
    Calibrate SEPARATELY for keys and values.
    Accumulates X^T X (second moment, not centered covariance) which is the
    correct formulation for the participation ratio in this context.

    Returns:
      eigen_keys: {(l,h): {"evec": Tensor, "ev": Tensor, "d_eff": int}}
      eigen_vals: {(l,h): {"evec": Tensor, "ev": Tensor, "d_eff": int}}
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                      split=f"train[:{n_calib * 5}]")

    cov_keys = {
        (l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64), "n": 0}
        for l in range(n_layers) for h in range(n_kv)
    }
    cov_vals = {
        (l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64), "n": 0}
        for l in range(n_layers) for h in range(n_kv)
    }

    nd = 0
    t0 = time.time()
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
                k_l, v_l = extract_kv_layer(kv, l)
                k_l = k_l.float().cpu()
                v_l = v_l.float().cpu()
            except Exception:
                continue

            for h in range(min(n_kv, k_l.shape[1])):
                X_key = k_l[0, h, :, :].double()  # (seq, hd)
                cov_keys[(l, h)]["xtx"] += X_key.T @ X_key
                cov_keys[(l, h)]["n"] += X_key.shape[0]

                X_val = v_l[0, h, :, :].double()
                cov_vals[(l, h)]["xtx"] += X_val.T @ X_val
                cov_vals[(l, h)]["n"] += X_val.shape[0]

        nd += 1
        if nd >= n_calib:
            break
        if nd % 20 == 0:
            log.info("  Calibration: %d/%d (%.0fs)", nd, n_calib, time.time() - t0)

    def _eigendecompose(cov_dict):
        eigen = {}
        for l in range(n_layers):
            for h in range(n_kv):
                n = cov_dict[(l, h)]["n"]
                if n == 0:
                    eigen[(l, h)] = {
                        "evec": torch.eye(hd),
                        "ev": torch.ones(hd),
                        "d_eff": hd,
                    }
                    continue
                C = (cov_dict[(l, h)]["xtx"] / n).float()
                ev, evec = torch.linalg.eigh(C)
                # Sort descending
                ev = ev.flip(0).clamp(min=0)
                evec = evec.flip(1)
                d_eff = max(2, min(
                    int(round((ev.sum() ** 2 / (ev ** 2).sum()).item())),
                    hd - 2
                ))
                eigen[(l, h)] = {"evec": evec, "ev": ev, "d_eff": d_eff}
        return eigen

    eigen_keys = _eigendecompose(cov_keys)
    eigen_vals = _eigendecompose(cov_vals)

    mean_deff_k = float(np.mean([eigen_keys[(l, h)]["d_eff"]
                                  for l in range(n_layers) for h in range(n_kv)]))
    mean_deff_v = float(np.mean([eigen_vals[(l, h)]["d_eff"]
                                  for l in range(n_layers) for h in range(n_kv)]))
    log.info("Calibration done in %.1fs. mean d_eff: keys=%.1f, values=%.1f",
             time.time() - t0, mean_deff_k, mean_deff_v)

    return eigen_keys, eigen_vals


# ============================================================================
# SHAPED CACHE ENGINE
# ============================================================================

class ShapedCacheEngine:
    """Asymmetric shaped KV cache.

    After spectral rotation (V^T), truncate to keep only the top-m dimensions
    for keys and top-p dimensions for values. Optionally quantize the retained
    dimensions.

    Args:
        key_eigvecs: (head_dim, head_dim) key eigenvectors sorted by eigenvalue desc
        key_eigvals: (head_dim,) key eigenvalues sorted desc
        val_eigvecs: (head_dim, head_dim) value eigenvectors sorted by eigenvalue desc
        val_eigvals: (head_dim,) value eigenvalues sorted desc
        m: number of key dimensions to retain
        p: number of value dimensions to retain
        quant_bits: 0 for FP16 (no quantization), 2/3/4 for scalar quantization
        head_dim: original head dimension
    """

    def __init__(self, key_eigvecs, key_eigvals, val_eigvecs, val_eigvals,
                 m, p, quant_bits=0, head_dim=128):
        self.head_dim = head_dim
        self.m = min(m, head_dim)
        self.p = min(p, head_dim)
        self.quant_bits = quant_bits

        Vk = key_eigvecs.float()
        Vv = val_eigvecs.float()
        assert Vk.shape == (head_dim, head_dim)
        assert Vv.shape == (head_dim, head_dim)

        # Truncated rotation matrices
        self.Vk_trunc = Vk[:, :self.m].contiguous()    # (hd, m)
        self.Vv_trunc = Vv[:, :self.p].contiguous()    # (hd, p)
        self.Vk_trunc_T = self.Vk_trunc.T.contiguous() # (m, hd)
        self.Vv_trunc_T = self.Vv_trunc.T.contiguous() # (p, hd)

        # Build per-regime quantization codebooks
        self.key_centroids = None
        self.val_centroids = None
        if quant_bits > 0:
            ev_k = key_eigvals.float().clamp(min=0)
            sigma_k = float(ev_k[:self.m].mean().clamp(min=1e-8).sqrt().item())
            self.key_centroids = solve_lloyd_max_for_sigma(sigma_k, quant_bits)

            ev_v = val_eigvals.float().clamp(min=0)
            sigma_v = float(ev_v[:self.p].mean().clamp(min=1e-8).sqrt().item())
            self.val_centroids = solve_lloyd_max_for_sigma(sigma_v, quant_bits)

    def compress_keys(self, K):
        """Compress keys: rotate + truncate to m dims + optional quantize."""
        K_f = K.float()
        vec_norms = torch.norm(K_f, dim=-1, keepdim=True)
        K_normed = K_f / (vec_norms + 1e-8)
        rotated = K_normed @ self.Vk_trunc.to(K.device)  # (seq, m)
        if self.quant_bits > 0 and self.key_centroids is not None:
            rotated = quantize_nearest(rotated, self.key_centroids)
        return {"rotated": rotated, "vec_norms": vec_norms.squeeze(-1)}

    def compress_values(self, V):
        """Compress values: rotate + truncate to p dims + optional quantize."""
        V_f = V.float()
        vec_norms = torch.norm(V_f, dim=-1, keepdim=True)
        V_normed = V_f / (vec_norms + 1e-8)
        rotated = V_normed @ self.Vv_trunc.to(V.device)  # (seq, p)
        if self.quant_bits > 0 and self.val_centroids is not None:
            rotated = quantize_nearest(rotated, self.val_centroids)
        return {"rotated": rotated, "vec_norms": vec_norms.squeeze(-1)}

    def decompress_keys(self, compressed_k):
        """Reconstruct keys from compressed representation."""
        rotated = compressed_k["rotated"].float()
        norms = compressed_k["vec_norms"].float()
        device = rotated.device
        return rotated @ self.Vk_trunc_T.to(device) * norms.unsqueeze(-1)

    def decompress_values(self, compressed_v):
        """Reconstruct values from compressed representation."""
        rotated = compressed_v["rotated"].float()
        norms = compressed_v["vec_norms"].float()
        device = rotated.device
        return rotated @ self.Vv_trunc_T.to(device) * norms.unsqueeze(-1)

    def attention_output(self, Q, compressed_k, compressed_v):
        """Compute full attention: softmax(Q @ K_hat^T / sqrt(d)) @ V_hat."""
        K_hat = self.decompress_keys(compressed_k)
        V_hat = self.decompress_values(compressed_v)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = Q.float() @ K_hat.T * scale
        weights = F.softmax(scores, dim=-1)
        return weights @ V_hat

    def bits_per_token(self):
        """Compressed bits per token for this (m, p, quant) config."""
        if self.quant_bits > 0:
            key_bits = self.m * self.quant_bits + 16   # m quantized dims + FP16 norm
            val_bits = self.p * self.quant_bits + 16   # p quantized dims + FP16 norm
        else:
            key_bits = self.m * 16 + 16                # m FP16 dims + FP16 norm
            val_bits = self.p * 16 + 16                # p FP16 dims + FP16 norm
        return key_bits + val_bits

    def compression_ratio(self):
        """Compression ratio vs FP16 KV cache."""
        fp16_bits = 2 * self.head_dim * 16
        return fp16_bits / self.bits_per_token()


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_config(engine, Q_all, K_all, V_all, fp16_output):
    """Evaluate a single (m, p, quant) configuration."""
    compressed_k = engine.compress_keys(K_all)
    compressed_v = engine.compress_values(V_all)

    K_hat = engine.decompress_keys(compressed_k)
    V_hat = engine.decompress_values(compressed_v)

    key_cos = F.cosine_similarity(K_all.float(), K_hat.float(), dim=-1).mean().item()
    val_cos = F.cosine_similarity(V_all.float(), V_hat.float(), dim=-1).mean().item()

    attn_out = engine.attention_output(Q_all, compressed_k, compressed_v)
    attn_cos = F.cosine_similarity(fp16_output.float(), attn_out.float(), dim=-1).mean().item()

    return {
        "key_cos_sim": key_cos,
        "val_cos_sim": val_cos,
        "attn_cos_sim": attn_cos,
        "compression_ratio": engine.compression_ratio(),
        "bits_per_token": engine.bits_per_token(),
        "m": engine.m,
        "p": engine.p,
        "quant_bits": engine.quant_bits,
    }


def run_sweep_for_model(model_name, short_name, m_values, p_values, quant_bits_list,
                         n_calib, n_eval, device):
    """Run the full (m, p, quant) sweep for one model."""
    log.info("=" * 70)
    log.info("MODEL: %s", model_name)
    log.info("=" * 70)

    model, tokenizer, n_layers, n_kv, hd = load_model_tokenizer(model_name, device)

    # Calibrate eigenbasis for keys AND values separately
    log.info("Calibrating eigenbasis from %d sequences...", n_calib)
    eigen_keys, eigen_vals = calibrate_keys_and_values(
        model, tokenizer, n_calib, device, n_layers, n_kv, hd
    )

    # Get d_eff stats
    deff_keys = [eigen_keys[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)]
    deff_vals = [eigen_vals[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)]
    mean_deff_keys = float(np.mean(deff_keys))
    mean_deff_vals = float(np.mean(deff_vals))
    log.info("d_eff: keys=%.2f, values=%.2f, asymmetry=%.1fx",
             mean_deff_keys, mean_deff_vals, mean_deff_vals / max(mean_deff_keys, 0.01))

    # Representative layers + heads for evaluation
    layer_indices = sorted(set([
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
    ]))
    head_indices = list(range(min(n_kv, 4)))
    log.info("Evaluating on layers=%s, heads=%s", layer_indices, head_indices)

    # Collect evaluation KV data (separate forward passes)
    log.info("Collecting evaluation data (%d sequences)...", n_eval)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                      split=f"train[{n_calib * 5}:{(n_calib + n_eval) * 5}]")

    # Store eval KV per (layer, head)
    eval_keys = {(l, h): [] for l in layer_indices for h in head_indices}
    eval_vals = {(l, h): [] for l in layer_indices for h in head_indices}
    nd = 0
    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt",
                        max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 32:
            continue
        with torch.no_grad():
            out = model(**enc, use_cache=True)
            kv = out.past_key_values

        for l in layer_indices:
            try:
                k_l, v_l = extract_kv_layer(kv, l)
                k_l = k_l.float().cpu()
                v_l = v_l.float().cpu()
            except Exception:
                continue
            for h in head_indices:
                if h < k_l.shape[1]:
                    eval_keys[(l, h)].append(k_l[0, h])
                    eval_vals[(l, h)].append(v_l[0, h])

        nd += 1
        if nd >= n_eval:
            break
    log.info("Collected eval data from %d sequences", nd)

    # Run the sweep
    all_results = []
    total_configs = len(m_values) * len(p_values) * len(quant_bits_list)
    config_idx = 0

    for m, p, qb in product(m_values, p_values, quant_bits_list):
        config_idx += 1
        if m > p:
            continue  # keys should use fewer dims than values
        if m > hd or p > hd:
            continue

        if config_idx % 20 == 1:
            log.info("Config %d/%d: m=%d, p=%d, quant=%s",
                     config_idx, total_configs, m, p, f"{qb}-bit" if qb > 0 else "FP16")

        config_metrics = []
        for l in layer_indices:
            for h in head_indices:
                kv_k = eval_keys.get((l, h), [])
                kv_v = eval_vals.get((l, h), [])
                if not kv_k or not kv_v:
                    continue

                K_all = torch.cat(kv_k, dim=0)[:512].to(device).float()
                V_all = torch.cat(kv_v, dim=0)[:512].to(device).float()
                Q_all = K_all.clone()  # Use keys as queries for self-attention test

                if K_all.shape[0] < 32:
                    continue

                # FP16 ground truth
                scale = 1.0 / math.sqrt(hd)
                fp16_scores = Q_all @ K_all.T * scale
                fp16_weights = F.softmax(fp16_scores, dim=-1)
                fp16_output = fp16_weights @ V_all

                engine = ShapedCacheEngine(
                    key_eigvecs=eigen_keys[(l, h)]["evec"].to(device),
                    key_eigvals=eigen_keys[(l, h)]["ev"].to(device),
                    val_eigvecs=eigen_vals[(l, h)]["evec"].to(device),
                    val_eigvals=eigen_vals[(l, h)]["ev"].to(device),
                    m=m, p=p, quant_bits=qb, head_dim=hd,
                )

                metrics = evaluate_config(engine, Q_all, K_all, V_all, fp16_output)
                metrics["layer"] = l
                metrics["head"] = h
                config_metrics.append(metrics)

        if config_metrics:
            avg = {
                "m": m, "p": p,
                "quant_bits": qb,
                "quant_label": f"{qb}-bit" if qb > 0 else "FP16",
                "key_cos_sim": float(np.mean([r["key_cos_sim"] for r in config_metrics])),
                "val_cos_sim": float(np.mean([r["val_cos_sim"] for r in config_metrics])),
                "attn_cos_sim": float(np.mean([r["attn_cos_sim"] for r in config_metrics])),
                "attn_cos_sim_std": float(np.std([r["attn_cos_sim"] for r in config_metrics])),
                "key_cos_sim_std": float(np.std([r["key_cos_sim"] for r in config_metrics])),
                "val_cos_sim_std": float(np.std([r["val_cos_sim"] for r in config_metrics])),
                "compression_ratio": config_metrics[0]["compression_ratio"],
                "bits_per_token": config_metrics[0]["bits_per_token"],
                "n_evals": len(config_metrics),
            }
            all_results.append(avg)

    # Compute baselines: TQ 3-bit, SQ 3-bit (full-dim)
    log.info("Computing baselines...")
    baselines = compute_baselines(
        eigen_keys, eigen_vals, eval_keys, eval_vals,
        layer_indices, head_indices, n_kv, hd, device
    )

    # Per-layer d_eff breakdown
    deff_by_layer_k = {}
    deff_by_layer_v = {}
    for l in range(n_layers):
        dk = [eigen_keys[(l, h)]["d_eff"] for h in range(n_kv)]
        dv = [eigen_vals[(l, h)]["d_eff"] for h in range(n_kv)]
        deff_by_layer_k[l] = float(np.mean(dk))
        deff_by_layer_v[l] = float(np.mean(dv))

    # Eigenvalue spectra for representative layers (for plotting)
    spectra = {}
    for l in layer_indices:
        for h in head_indices[:1]:  # just head 0
            spectra[f"layer_{l}_head_{h}_keys"] = eigen_keys[(l, h)]["ev"][:32].tolist()
            spectra[f"layer_{l}_head_{h}_values"] = eigen_vals[(l, h)]["ev"][:32].tolist()

    result = {
        "model": model_name,
        "short_name": short_name,
        "head_dim": hd,
        "n_layers": n_layers,
        "n_kv_heads": n_kv,
        "mean_d_eff_keys": mean_deff_keys,
        "mean_d_eff_values": mean_deff_vals,
        "asymmetry_ratio": mean_deff_vals / max(mean_deff_keys, 0.01),
        "d_eff_by_layer_keys": deff_by_layer_k,
        "d_eff_by_layer_values": deff_by_layer_v,
        "eigenvalue_spectra": spectra,
        "sweep_results": all_results,
        "baselines": baselines,
        "config": {
            "m_values": m_values,
            "p_values": p_values,
            "quant_bits": quant_bits_list,
            "n_calib": n_calib,
            "n_eval": n_eval,
            "layers_evaluated": layer_indices,
            "heads_evaluated": head_indices,
        },
    }

    save_result(f"shaped_cache_{short_name}.json", result)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def compute_baselines(eigen_keys, eigen_vals, eval_keys, eval_vals,
                      layer_indices, head_indices, n_kv, hd, device):
    """Compute TQ and SQ baselines for comparison."""
    from spectralquant.engine import SpectralQuantEngine

    baselines = {}

    for method_name, use_spectral in [("TQ_3bit", False), ("SQ_noQJL_3bit", True)]:
        cos_sims = []

        for l in layer_indices:
            for h in head_indices:
                kv_k = eval_keys.get((l, h), [])
                kv_v = eval_vals.get((l, h), [])
                if not kv_k or not kv_v:
                    continue

                K_all = torch.cat(kv_k, dim=0)[:512].to(device).float()
                V_all = torch.cat(kv_v, dim=0)[:512].to(device).float()
                Q_all = K_all.clone()

                if K_all.shape[0] < 32:
                    continue

                scale = 1.0 / math.sqrt(hd)
                fp16_scores = Q_all @ K_all.T * scale
                fp16_weights = F.softmax(fp16_scores, dim=-1)
                fp16_output = fp16_weights @ V_all

                try:
                    if use_spectral:
                        evec = eigen_keys[(l, h)]["evec"]
                        ev = eigen_keys[(l, h)]["ev"]
                        d_eff = eigen_keys[(l, h)]["d_eff"]
                        VT = evec.T.contiguous().to(device).float()
                        Vm = evec.to(device).float()

                        k_n = torch.norm(K_all, dim=-1, keepdim=True)
                        K_rot = (K_all / (k_n + 1e-8)) @ VT
                        v_n = torch.norm(V_all, dim=-1, keepdim=True)
                        V_rot = (V_all / (v_n + 1e-8)) @ VT

                        cb_k = LloydMaxCodebook(hd, 2)
                        cb_v = LloydMaxCodebook(hd, 3)
                        K_hat = quantize_nearest(K_rot, cb_k.centroids.to(device))
                        V_hat = quantize_nearest(V_rot, cb_v.centroids.to(device))

                        K_mse = (K_hat @ Vm) * k_n
                        V_rec = (V_hat @ Vm) * v_n
                        scores = Q_all @ K_mse.T * scale
                        output = F.softmax(scores, dim=-1) @ V_rec
                    else:
                        tq = TurboQuantEngine(d=hd, mse_bits=2, seed=42, device=device)
                        ck = tq.compress_keys_pytorch(K_all.half())
                        cv = tq.compress_values_pytorch(V_all.half())
                        scores = tq.attention_scores_pytorch(Q_all.half(), ck)
                        weights = F.softmax(scores.float(), dim=-1)
                        V_hat = tq.decompress_values_pytorch(cv)
                        output = weights @ V_hat.float()

                    cos = F.cosine_similarity(
                        fp16_output.float(), output.float(), dim=-1
                    ).mean().item()
                    cos_sims.append(cos)
                except Exception as e:
                    log.warning("%s baseline failed l=%d h=%d: %s", method_name, l, h, e)

        if cos_sims:
            baselines[method_name] = {
                "attn_cos_sim": float(np.mean(cos_sims)),
                "attn_cos_sim_std": float(np.std(cos_sims)),
                "compression_ratio": 5.95 if use_spectral else 5.02,
                "n_evals": len(cos_sims),
            }
            log.info("  Baseline %s: cos=%.4f +/- %.4f (n=%d)",
                     method_name, baselines[method_name]["attn_cos_sim"],
                     baselines[method_name]["attn_cos_sim_std"],
                     len(cos_sims))

    return baselines


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    quick = args.quick
    device = args.device

    m_vals = M_VALUES_QUICK if quick else M_VALUES
    p_vals = P_VALUES_QUICK if quick else P_VALUES
    qb_vals = QUANT_BITS_QUICK if quick else QUANT_BITS
    n_calib = N_CALIB_QUICK if quick else N_CALIB
    n_eval = N_EVAL_QUICK if quick else N_EVAL

    log.info("=" * 70)
    log.info("SHAPED CACHE SWEEP: %s mode", "QUICK" if quick else "FULL")
    log.info("m_values=%s, p_values=%s, quant=%s", m_vals, p_vals, qb_vals)
    log.info("n_calib=%d, n_eval=%d", n_calib, n_eval)
    log.info("=" * 70)

    all_model_results = {}
    t0 = time.time()

    for model_name, short_name in MODELS:
        try:
            result = run_sweep_for_model(
                model_name, short_name, m_vals, p_vals, qb_vals,
                n_calib, n_eval, device,
            )
            all_model_results[short_name] = result
        except Exception as e:
            log.error("FAILED on %s: %s", model_name, e)
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0

    combined = {
        "experiment": "shaped_cache_sweep",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "mode": "quick" if quick else "full",
        "models": all_model_results,
    }
    save_result("shaped_cache_combined.json", combined)

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("SHAPED CACHE SWEEP COMPLETE (%.1f min)", elapsed / 60)
    log.info("=" * 70)

    for short_name, result in all_model_results.items():
        log.info("\n--- %s (d_eff: keys=%.1f, values=%.1f, asymmetry=%.1fx) ---",
                 short_name, result["mean_d_eff_keys"], result["mean_d_eff_values"],
                 result["asymmetry_ratio"])

        sweep = result["sweep_results"]
        if sweep:
            # Pareto analysis
            high_quality = [r for r in sweep if r["attn_cos_sim"] > 0.95]
            if high_quality:
                best = max(high_quality, key=lambda r: r["compression_ratio"])
                log.info("  Best >0.95 quality: m=%d, p=%d, %s, cos=%.4f, %.1fx",
                         best["m"], best["p"], best["quant_label"],
                         best["attn_cos_sim"], best["compression_ratio"])

            high_compress = [r for r in sweep if r["compression_ratio"] > 10]
            if high_compress:
                best_hc = max(high_compress, key=lambda r: r["attn_cos_sim"])
                log.info("  Best >10x compress: m=%d, p=%d, %s, cos=%.4f, %.1fx",
                         best_hc["m"], best_hc["p"], best_hc["quant_label"],
                         best_hc["attn_cos_sim"], best_hc["compression_ratio"])

            best_overall = max(sweep, key=lambda r: r["attn_cos_sim"])
            log.info("  Best quality:       m=%d, p=%d, %s, cos=%.4f, %.1fx",
                     best_overall["m"], best_overall["p"], best_overall["quant_label"],
                     best_overall["attn_cos_sim"], best_overall["compression_ratio"])

        baselines = result.get("baselines", {})
        for bname, bdata in baselines.items():
            log.info("  Baseline %s: cos=%.4f, %.2fx",
                     bname, bdata["attn_cos_sim"], bdata["compression_ratio"])


if __name__ == "__main__":
    main()
