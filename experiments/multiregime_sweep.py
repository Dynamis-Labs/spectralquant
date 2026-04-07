"""
Multi-Regime Spectral Quantization: Exploiting the Full Eigenvalue Spectrum
============================================================================
Instead of truncating dimensions (which fails) or using 2 regimes (SQ current),
split the eigenbasis into 3+ regimes with different bit widths:

  Regime 1 (top d_eff dims):      High variance -> 4-bit (high precision)
  Regime 2 (medium dims):         Medium variance -> 2-bit
  Regime 3 (tail dims):           Low variance -> 1-bit (sign only)

Key insight: 1-bit sign quantization on the tail preserves the *direction*
of each small component. Since |sigma_i| is tiny for tail dims, the absolute
quantization error is tiny, even though the relative error is ~36%.

Theory predicts:
  Keys:   4*4 + 12*2 + 112*1 + 16(norm) = 168 bits  (12.2x)
  Values: 50*3 + 30*2 + 48*1 + 16(norm) = 274 bits  (7.5x)
  Total:  442 bits per token = 9.3x compression

vs SpectralQuant current: 746 bits = 5.5x compression
That's 1.7x more compression at potentially HIGHER quality (better SQNR).

Usage:
    python multiregime_sweep.py [--quick] [--device cuda]
"""

import sys, os, math, time, json, logging, argparse, gc
from pathlib import Path
from itertools import product

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from turboquant_cutile import TurboQuantEngine, LloydMaxCodebook

log = logging.getLogger("multiregime")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

RESULTS_DIR = PROJECT_ROOT / "results" / "multiregime"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN")


# ============================================================================
# MODELS & CONFIGS
# ============================================================================

MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen-1.5B"),
    ("Qwen/Qwen2.5-7B-Instruct",   "Qwen-7B"),
]
if HF_TOKEN:
    MODELS.append(("meta-llama/Llama-3.1-8B-Instruct", "Llama-8B"))

# Multi-regime configurations to test
# Each config: list of (start_dim_frac, end_dim_frac, bits)
# Fractions are relative to d_eff boundaries computed per (layer, head)
# "d_eff" = boundary 1, "3*d_eff" = boundary 2

# Format: name -> (key_regimes, value_regimes)
# Each regime: (start, end, bits) where start/end are "deff_mult" or absolute
REGIME_CONFIGS = {
    # Current SpectralQuant: 2 regimes
    "SQ_2regime": {
        "key_regimes":  [(0, "1x_deff", 2), ("1x_deff", "all", 3)],
        "val_regimes":  [(0, "1x_deff", 2), ("1x_deff", "all", 3)],
    },
    # 3-regime: high/medium/sign
    "3regime_441": {
        "key_regimes":  [(0, "1x_deff", 4), ("1x_deff", "3x_deff", 2), ("3x_deff", "all", 1)],
        "val_regimes":  [(0, "1x_deff", 4), ("1x_deff", "3x_deff", 2), ("3x_deff", "all", 1)],
    },
    # 3-regime: 4/3/1
    "3regime_431": {
        "key_regimes":  [(0, "1x_deff", 4), ("1x_deff", "3x_deff", 3), ("3x_deff", "all", 1)],
        "val_regimes":  [(0, "1x_deff", 4), ("1x_deff", "3x_deff", 3), ("3x_deff", "all", 1)],
    },
    # 3-regime: aggressive sign tail, keeping medium at 2-bit
    "3regime_421": {
        "key_regimes":  [(0, "1x_deff", 4), ("1x_deff", "4x_deff", 2), ("4x_deff", "all", 1)],
        "val_regimes":  [(0, "1x_deff", 4), ("1x_deff", "4x_deff", 2), ("4x_deff", "all", 1)],
    },
    # 4-regime: 4/3/2/1
    "4regime_4321": {
        "key_regimes":  [(0, "1x_deff", 4), ("1x_deff", "2x_deff", 3), ("2x_deff", "4x_deff", 2), ("4x_deff", "all", 1)],
        "val_regimes":  [(0, "1x_deff", 4), ("1x_deff", "2x_deff", 3), ("2x_deff", "4x_deff", 2), ("4x_deff", "all", 1)],
    },
    # Asymmetric: keys get sign tail sooner (d_eff=4), values keep more bits (d_eff=50)
    "asym_key431_val321": {
        "key_regimes":  [(0, "1x_deff", 4), ("1x_deff", "2x_deff", 3), ("2x_deff", "all", 1)],
        "val_regimes":  [(0, "1x_deff", 3), ("1x_deff", "2x_deff", 2), ("2x_deff", "all", 1)],
    },
    # Ultra-aggressive: 4-bit top + 1-bit everything else
    "2regime_41": {
        "key_regimes":  [(0, "1x_deff", 4), ("1x_deff", "all", 1)],
        "val_regimes":  [(0, "1x_deff", 4), ("1x_deff", "all", 1)],
    },
    # Comparison: uniform 3-bit (TurboQuant-like, but with spectral rotation)
    "uniform_3bit": {
        "key_regimes":  [(0, "all", 3)],
        "val_regimes":  [(0, "all", 3)],
    },
    # Comparison: uniform 2-bit
    "uniform_2bit": {
        "key_regimes":  [(0, "all", 2)],
        "val_regimes":  [(0, "all", 2)],
    },
}

N_CALIB = 64
N_EVAL = 32


# ============================================================================
# UTILITIES (shared with shaped_cache_sweep.py)
# ============================================================================

def save_result(filename, data):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved: %s", path)


def solve_lloyd_max_for_sigma(sigma, bits, max_iter=200, tol=1e-10):
    if bits == 1:
        # Sign quantization: centroids are +/- sigma * sqrt(2/pi)
        c = sigma * math.sqrt(2.0 / math.pi)
        return torch.tensor([-c, c], dtype=torch.float32)

    from scipy import integrate
    n_levels = 1 << bits
    pdf = lambda x: (1.0 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-x*x / (2*sigma*sigma))
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i+1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i+1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids
    return torch.tensor(centroids, dtype=torch.float32)


def quantize_nearest(x, centroids):
    c = centroids.to(x.device)
    diffs = x.unsqueeze(-1) - c
    return c[diffs.abs().argmin(dim=-1).long()]


def load_model_tokenizer(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    needs_token = any(x in model_name for x in ["llama", "Llama", "gemma", "Gemma"])
    token = HF_TOKEN if needs_token else None
    log.info("Loading %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device, token=token)
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    try:
        test_ids = tokenizer("test", return_tensors="pt").to(device)
        with torch.no_grad():
            test_out = model(**test_ids, use_cache=True)
        kv = test_out.past_key_values
        try: actual_kv_hd = kv.key_cache[0].shape[-1]
        except:
            try: actual_kv_hd = kv[0][0].shape[-1]
            except: actual_kv_hd = hd
        if actual_kv_hd != hd: hd = actual_kv_hd
    except: pass
    return model, tokenizer, n_layers, n_kv, hd


def extract_kv_layer(kv, l):
    try: return kv.key_cache[l], kv.value_cache[l]
    except: pass
    try: return kv[l][0], kv[l][1]
    except: pass
    entry = list(kv)[l]
    return entry[0], entry[1]


def calibrate_keys_and_values(model, tokenizer, n_calib, device, n_layers, n_kv, hd):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{n_calib * 5}]")
    cov_keys = {(l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64), "n": 0} for l in range(n_layers) for h in range(n_kv)}
    cov_vals = {(l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64), "n": 0} for l in range(n_layers) for h in range(n_kv)}
    nd = 0
    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100: continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16: continue
        with torch.no_grad():
            out = model(**enc, use_cache=True)
            kv = out.past_key_values
        for l in range(n_layers):
            try: k_l, v_l = extract_kv_layer(kv, l)
            except: continue
            k_l = k_l.float().cpu(); v_l = v_l.float().cpu()
            for h in range(min(n_kv, k_l.shape[1])):
                X_key = k_l[0, h, :, :].double()
                cov_keys[(l, h)]["xtx"] += X_key.T @ X_key
                cov_keys[(l, h)]["n"] += X_key.shape[0]
                X_val = v_l[0, h, :, :].double()
                cov_vals[(l, h)]["xtx"] += X_val.T @ X_val
                cov_vals[(l, h)]["n"] += X_val.shape[0]
        nd += 1
        if nd >= n_calib: break
        if nd % 20 == 0: log.info("  Calibration: %d/%d", nd, n_calib)

    def _eigen(cov_dict):
        eigen = {}
        for l in range(n_layers):
            for h in range(n_kv):
                n = cov_dict[(l, h)]["n"]
                if n == 0:
                    eigen[(l, h)] = {"evec": torch.eye(hd), "ev": torch.ones(hd), "d_eff": hd}
                    continue
                C = (cov_dict[(l, h)]["xtx"] / n).float()
                ev, evec = torch.linalg.eigh(C)
                ev = ev.flip(0).clamp(min=0); evec = evec.flip(1)
                d_eff = max(2, min(int(round((ev.sum()**2 / (ev**2).sum()).item())), hd - 2))
                eigen[(l, h)] = {"evec": evec, "ev": ev, "d_eff": d_eff}
        return eigen
    return _eigen(cov_keys), _eigen(cov_vals)


# ============================================================================
# MULTI-REGIME ENGINE
# ============================================================================

class MultiRegimeEngine:
    """Multi-regime spectral quantization engine.

    Splits the eigenbasis into N regimes, each with its own bit width.
    Regime boundaries are defined relative to d_eff.
    """

    def __init__(self, eigvecs, eigvals, d_eff, regime_spec, head_dim):
        """
        regime_spec: list of (start, end, bits)
          start/end are either int (absolute dim index) or str like "1x_deff", "all"
        """
        self.head_dim = head_dim
        self.V = eigvecs.float()  # (hd, hd)
        self.VT = self.V.T.contiguous()  # (hd, hd)
        self.d_eff = d_eff
        ev = eigvals.float().clamp(min=0)

        # Resolve regime boundaries
        self.regimes = []
        for start_spec, end_spec, bits in regime_spec:
            start = self._resolve_boundary(start_spec, d_eff, head_dim)
            end = self._resolve_boundary(end_spec, d_eff, head_dim)
            if start >= end:
                continue
            # Compute codebook for this regime
            sigma = float(ev[start:end].mean().clamp(min=1e-10).sqrt().item())
            centroids = solve_lloyd_max_for_sigma(sigma, bits)
            self.regimes.append({
                "start": start, "end": end, "bits": bits,
                "centroids": centroids, "n_dims": end - start,
            })

    def _resolve_boundary(self, spec, d_eff, hd):
        if isinstance(spec, int):
            return min(spec, hd)
        if spec == "all":
            return hd
        if spec.endswith("x_deff"):
            mult = float(spec.replace("x_deff", ""))
            return min(int(round(mult * d_eff)), hd)
        return int(spec)

    def compress(self, X):
        """Compress vectors: rotate to eigenbasis, quantize per-regime."""
        X_f = X.float()
        norms = torch.norm(X_f, dim=-1, keepdim=True)
        X_normed = X_f / (norms + 1e-8)
        rotated = X_normed @ self.V  # (seq, hd)

        # Quantize each regime
        quantized = rotated.clone()
        for regime in self.regimes:
            s, e = regime["start"], regime["end"]
            quantized[:, s:e] = quantize_nearest(rotated[:, s:e], regime["centroids"])

        return {"quantized_rotated": quantized, "norms": norms.squeeze(-1)}

    def decompress(self, compressed):
        """Reconstruct: un-rotate and re-scale."""
        qr = compressed["quantized_rotated"].float()
        norms = compressed["norms"].float()
        return (qr @ self.VT.to(qr.device)) * norms.unsqueeze(-1)

    def bits_per_vector(self):
        """Total bits per vector."""
        total = 16  # FP16 norm
        for regime in self.regimes:
            total += regime["n_dims"] * regime["bits"]
        return total

    def compression_ratio(self):
        return (self.head_dim * 16) / self.bits_per_vector()

    def regime_summary(self):
        parts = []
        for r in self.regimes:
            parts.append(f"dims {r['start']+1}-{r['end']}: {r['bits']}-bit ({r['n_dims']} dims)")
        return " | ".join(parts)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_multiregime(key_engine, val_engine, Q, K, V, fp16_output):
    """Evaluate a multi-regime config on one (layer, head)."""
    ck = key_engine.compress(K)
    cv = val_engine.compress(V)
    K_hat = key_engine.decompress(ck)
    V_hat = val_engine.decompress(cv)

    key_cos = F.cosine_similarity(K.float(), K_hat.float(), dim=-1).mean().item()
    val_cos = F.cosine_similarity(V.float(), V_hat.float(), dim=-1).mean().item()

    scale = 1.0 / math.sqrt(key_engine.head_dim)
    scores = Q.float() @ K_hat.T * scale
    weights = F.softmax(scores, dim=-1)
    output = weights @ V_hat

    attn_cos = F.cosine_similarity(fp16_output.float(), output.float(), dim=-1).mean().item()

    return {
        "key_cos_sim": key_cos,
        "val_cos_sim": val_cos,
        "attn_cos_sim": attn_cos,
        "key_bits": key_engine.bits_per_vector(),
        "val_bits": val_engine.bits_per_vector(),
        "total_bits": key_engine.bits_per_vector() + val_engine.bits_per_vector(),
        "key_compress": key_engine.compression_ratio(),
        "val_compress": val_engine.compression_ratio(),
        "total_compress": (2 * key_engine.head_dim * 16) / (key_engine.bits_per_vector() + val_engine.bits_per_vector()),
    }


def run_model(model_name, short_name, n_calib, n_eval, device):
    """Run all multi-regime configs on one model."""
    log.info("=" * 70)
    log.info("MODEL: %s", model_name)
    log.info("=" * 70)

    model, tokenizer, n_layers, n_kv, hd = load_model_tokenizer(model_name, device)
    log.info("Calibrating eigenbasis...")
    eigen_keys, eigen_vals = calibrate_keys_and_values(model, tokenizer, n_calib, device, n_layers, n_kv, hd)

    deff_keys = [eigen_keys[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)]
    deff_vals = [eigen_vals[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)]
    mean_dk = float(np.mean(deff_keys))
    mean_dv = float(np.mean(deff_vals))
    log.info("d_eff: keys=%.1f, values=%.1f", mean_dk, mean_dv)

    # Collect eval data
    layer_indices = sorted(set([n_layers // 4, n_layers // 2, 3 * n_layers // 4]))
    head_indices = list(range(min(n_kv, 4)))

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[{n_calib*5}:{(n_calib+n_eval)*5}]")
    eval_keys = {(l, h): [] for l in layer_indices for h in head_indices}
    eval_vals = {(l, h): [] for l in layer_indices for h in head_indices}
    nd = 0
    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100: continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 32: continue
        with torch.no_grad():
            out = model(**enc, use_cache=True)
            kv = out.past_key_values
        for l in layer_indices:
            try: k_l, v_l = extract_kv_layer(kv, l)
            except: continue
            k_l = k_l.float().cpu(); v_l = v_l.float().cpu()
            for h in head_indices:
                if h < k_l.shape[1]:
                    eval_keys[(l, h)].append(k_l[0, h])
                    eval_vals[(l, h)].append(v_l[0, h])
        nd += 1
        if nd >= n_eval: break
    log.info("Collected eval data from %d sequences", nd)

    # Evaluate each config
    all_results = {}
    for config_name, config in REGIME_CONFIGS.items():
        log.info("  Config: %s", config_name)
        metrics_list = []

        for l in layer_indices:
            for h in head_indices:
                kk = eval_keys.get((l, h), [])
                vv = eval_vals.get((l, h), [])
                if not kk or not vv: continue
                K_all = torch.cat(kk, dim=0)[:512].to(device).float()
                V_all = torch.cat(vv, dim=0)[:512].to(device).float()
                Q_all = K_all.clone()
                if K_all.shape[0] < 32: continue

                scale = 1.0 / math.sqrt(hd)
                fp16_scores = Q_all @ K_all.T * scale
                fp16_output = F.softmax(fp16_scores, dim=-1) @ V_all

                key_engine = MultiRegimeEngine(
                    eigen_keys[(l, h)]["evec"].to(device),
                    eigen_keys[(l, h)]["ev"].to(device),
                    eigen_keys[(l, h)]["d_eff"],
                    config["key_regimes"], hd,
                )
                val_engine = MultiRegimeEngine(
                    eigen_vals[(l, h)]["evec"].to(device),
                    eigen_vals[(l, h)]["ev"].to(device),
                    eigen_vals[(l, h)]["d_eff"],
                    config["val_regimes"], hd,
                )

                m = evaluate_multiregime(key_engine, val_engine, Q_all, K_all, V_all, fp16_output)
                m["layer"] = l; m["head"] = h
                m["key_regime_summary"] = key_engine.regime_summary()
                m["val_regime_summary"] = val_engine.regime_summary()
                metrics_list.append(m)

        if metrics_list:
            avg = {
                "config_name": config_name,
                "attn_cos_sim": float(np.mean([m["attn_cos_sim"] for m in metrics_list])),
                "attn_cos_sim_std": float(np.std([m["attn_cos_sim"] for m in metrics_list])),
                "key_cos_sim": float(np.mean([m["key_cos_sim"] for m in metrics_list])),
                "val_cos_sim": float(np.mean([m["val_cos_sim"] for m in metrics_list])),
                "total_bits": metrics_list[0]["total_bits"],
                "key_bits": metrics_list[0]["key_bits"],
                "val_bits": metrics_list[0]["val_bits"],
                "total_compress": metrics_list[0]["total_compress"],
                "key_compress": metrics_list[0]["key_compress"],
                "val_compress": metrics_list[0]["val_compress"],
                "key_regime_summary": metrics_list[0]["key_regime_summary"],
                "val_regime_summary": metrics_list[0]["val_regime_summary"],
                "n_evals": len(metrics_list),
            }
            all_results[config_name] = avg
            log.info("    attn_cos=%.4f, total_bits=%d, compress=%.1fx",
                     avg["attn_cos_sim"], avg["total_bits"], avg["total_compress"])

    result = {
        "model": model_name, "short_name": short_name,
        "head_dim": hd, "n_layers": n_layers, "n_kv_heads": n_kv,
        "mean_d_eff_keys": mean_dk, "mean_d_eff_values": mean_dv,
        "configs": all_results,
    }
    save_result(f"multiregime_{short_name}.json", result)

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    n_calib = 16 if args.quick else N_CALIB
    n_eval = 8 if args.quick else N_EVAL

    log.info("MULTI-REGIME SWEEP: %s mode", "QUICK" if args.quick else "FULL")

    all_results = {}
    t0 = time.time()
    for model_name, short_name in MODELS:
        try:
            result = run_model(model_name, short_name, n_calib, n_eval, args.device)
            all_results[short_name] = result
        except Exception as e:
            log.error("FAILED on %s: %s", model_name, e)
            import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    save_result("multiregime_combined.json", {
        "experiment": "multiregime_sweep",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "models": all_results,
    })

    log.info("\n" + "=" * 70)
    log.info("MULTI-REGIME SWEEP COMPLETE (%.1f min)", elapsed / 60)
    log.info("=" * 70)

    for short_name, result in all_results.items():
        log.info("\n--- %s (d_eff: K=%.1f, V=%.1f) ---", short_name, result["mean_d_eff_keys"], result["mean_d_eff_values"])
        configs = result["configs"]
        # Sort by compression
        for name, c in sorted(configs.items(), key=lambda x: x[1]["total_compress"], reverse=True):
            log.info("  %20s: cos=%.4f, %d bits, %.1fx | K: %s | V: %s",
                     name, c["attn_cos_sim"], c["total_bits"], c["total_compress"],
                     c["key_regime_summary"], c["val_regime_summary"])


if __name__ == "__main__":
    main()
