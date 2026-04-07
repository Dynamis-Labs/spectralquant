"""
Optimal Per-Dimension Bit Allocation via Reverse Water-Filling
================================================================
The Shannon-optimal way to allocate bits across dimensions with known
variances (eigenvalues) is reverse water-filling:

  b_i = max(b_min, round(0.5 * log2(lambda_i / theta)))

This gives the minimum MSE for a given total bit budget.

We implement this with:
  - b_min = 1 (every dimension gets at least 1 bit = sign)
  - b_max = 6 (diminishing returns beyond 6-bit)
  - Per-dimension Lloyd-Max codebooks calibrated to sigma_i = sqrt(lambda_i)

Sweep:
  - Total bit budgets: 256 to 768 in steps of 64 (per KV pair)
  - Key/value budget split: optimize jointly
  - Models: Qwen 1.5B, Qwen 7B, Llama 8B

This finds the absolute performance ceiling for spectral quantization.

Usage:
    python optimal_allocation_sweep.py [--device cuda]
"""

import sys, os, math, time, json, logging, argparse, gc
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from turboquant_cutile import TurboQuantEngine

log = logging.getLogger("optimal_alloc")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

RESULTS_DIR = PROJECT_ROOT / "results" / "optimal_allocation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN")

MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen-1.5B"),
    ("Qwen/Qwen2.5-7B-Instruct",   "Qwen-7B"),
]
if HF_TOKEN:
    MODELS.append(("meta-llama/Llama-3.1-8B-Instruct", "Llama-8B"))

N_CALIB = 64
N_EVAL = 32

# Bit budgets to sweep (per KV pair, excluding norms)
BIT_BUDGETS = [192, 256, 320, 384, 448, 512, 576, 640, 704, 768]


# ============================================================================
# UTILITIES
# ============================================================================

def save_result(filename, data):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved: %s", path)


def solve_lloyd_max_for_sigma(sigma, bits, max_iter=200, tol=1e-10):
    if bits <= 0:
        return torch.tensor([0.0], dtype=torch.float32)
    if bits == 1:
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


def optimal_bit_allocation(eigenvalues, total_bits, min_bits=1, max_bits=6):
    """Reverse water-filling with minimum bit constraint."""
    d = len(eigenvalues)
    ev = np.array(eigenvalues, dtype=np.float64).clip(min=1e-30)

    # Binary search for theta
    lo, hi = 1e-30, ev.max() * 10
    for _ in range(300):
        theta = (lo + hi) / 2
        bits = np.clip(np.round(0.5 * np.log2(np.maximum(ev / theta, 1e-30))), min_bits, max_bits)
        if bits.sum() > total_bits:
            lo = theta
        else:
            hi = theta

    bits = np.clip(np.round(0.5 * np.log2(np.maximum(ev / theta, 1e-30))), min_bits, max_bits)

    # Fine-tune to hit exact budget
    while bits.sum() > total_bits:
        candidates = np.where(bits > min_bits)[0]
        if len(candidates) == 0: break
        worst = candidates[np.argmin(ev[candidates])]
        bits[worst] -= 1
    while bits.sum() < total_bits:
        candidates = np.where(bits < max_bits)[0]
        if len(candidates) == 0: break
        best = candidates[np.argmax(ev[candidates] / (2.0 ** (2 * bits[candidates])))]
        bits[best] += 1

    return bits.astype(int)


# ============================================================================
# MODEL LOADING & CALIBRATION (reuse from multiregime)
# ============================================================================

def load_model_tokenizer(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    needs_token = any(x in model_name for x in ["llama", "Llama", "gemma", "Gemma"])
    token = HF_TOKEN if needs_token else None
    log.info("Loading %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device, token=token)
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    try:
        test_ids = tokenizer("test", return_tensors="pt").to(device)
        with torch.no_grad(): test_out = model(**test_ids, use_cache=True)
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
    entry = list(kv)[l]; return entry[0], entry[1]


def calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd):
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
                X_key = k_l[0, h].double(); X_val = v_l[0, h].double()
                cov_keys[(l, h)]["xtx"] += X_key.T @ X_key; cov_keys[(l, h)]["n"] += X_key.shape[0]
                cov_vals[(l, h)]["xtx"] += X_val.T @ X_val; cov_vals[(l, h)]["n"] += X_val.shape[0]
        nd += 1
        if nd >= n_calib: break
        if nd % 20 == 0: log.info("  Calibration: %d/%d", nd, n_calib)
    def _eigen(cd):
        eigen = {}
        for l in range(n_layers):
            for h in range(n_kv):
                n = cd[(l, h)]["n"]
                if n == 0:
                    eigen[(l, h)] = {"evec": torch.eye(hd), "ev": torch.ones(hd), "d_eff": hd}; continue
                C = (cd[(l, h)]["xtx"] / n).float()
                ev, evec = torch.linalg.eigh(C)
                ev = ev.flip(0).clamp(min=0); evec = evec.flip(1)
                d_eff = max(2, min(int(round((ev.sum()**2 / (ev**2).sum()).item())), hd - 2))
                eigen[(l, h)] = {"evec": evec, "ev": ev, "d_eff": d_eff}
        return eigen
    return _eigen(cov_keys), _eigen(cov_vals)


# ============================================================================
# OPTIMAL ALLOCATION ENGINE
# ============================================================================

class OptimalAllocEngine:
    """Per-dimension optimal bit allocation with Lloyd-Max codebooks."""

    def __init__(self, eigvecs, eigvals, bit_allocation, head_dim):
        self.head_dim = head_dim
        self.V = eigvecs.float()
        self.VT = self.V.T.contiguous()
        self.bit_allocation = bit_allocation  # (head_dim,) array of ints

        # Build per-dimension codebooks
        ev = eigvals.float().clamp(min=1e-10).cpu().numpy()
        self.codebooks = []
        for i in range(head_dim):
            b = int(bit_allocation[i])
            sigma = float(np.sqrt(ev[i]))
            cb = solve_lloyd_max_for_sigma(max(sigma, 1e-6), max(b, 1))
            self.codebooks.append(cb)

        self.total_bits = int(np.sum(bit_allocation)) + 16  # +16 for norm

    def compress(self, X):
        X_f = X.float()
        norms = torch.norm(X_f, dim=-1, keepdim=True)
        X_normed = X_f / (norms + 1e-8)
        rotated = X_normed @ self.V  # (seq, hd)
        quantized = rotated.clone()
        for i in range(self.head_dim):
            quantized[:, i] = quantize_nearest(rotated[:, i], self.codebooks[i])
        return {"quantized_rotated": quantized, "norms": norms.squeeze(-1)}

    def decompress(self, compressed):
        qr = compressed["quantized_rotated"].float()
        norms = compressed["norms"].float()
        return (qr @ self.VT.to(qr.device)) * norms.unsqueeze(-1)

    def compression_ratio(self):
        return (self.head_dim * 16) / self.total_bits


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_allocation(k_engine, v_engine, Q, K, V, fp16_output):
    ck = k_engine.compress(K); cv = v_engine.compress(V)
    K_hat = k_engine.decompress(ck); V_hat = v_engine.decompress(cv)
    key_cos = F.cosine_similarity(K.float(), K_hat.float(), dim=-1).mean().item()
    val_cos = F.cosine_similarity(V.float(), V_hat.float(), dim=-1).mean().item()
    scale = 1.0 / math.sqrt(k_engine.head_dim)
    scores = Q.float() @ K_hat.T * scale
    output = F.softmax(scores, dim=-1) @ V_hat
    attn_cos = F.cosine_similarity(fp16_output.float(), output.float(), dim=-1).mean().item()
    return {"key_cos_sim": key_cos, "val_cos_sim": val_cos, "attn_cos_sim": attn_cos,
            "total_bits": k_engine.total_bits + v_engine.total_bits,
            "key_bits": k_engine.total_bits, "val_bits": v_engine.total_bits,
            "total_compress": (2 * k_engine.head_dim * 16) / (k_engine.total_bits + v_engine.total_bits)}


def run_model(model_name, short_name, n_calib, n_eval, device):
    log.info("=" * 70)
    log.info("MODEL: %s", model_name)

    model, tokenizer, n_layers, n_kv, hd = load_model_tokenizer(model_name, device)
    eigen_keys, eigen_vals = calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd)

    deff_keys = np.mean([eigen_keys[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)])
    deff_vals = np.mean([eigen_vals[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)])
    log.info("d_eff: keys=%.1f, values=%.1f", deff_keys, deff_vals)

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

    all_results = {}

    for total_budget in BIT_BUDGETS:
        # Try different key/value splits
        best = None
        for key_frac in np.arange(0.2, 0.55, 0.05):
            key_budget = int(total_budget * key_frac)
            val_budget = total_budget - key_budget

            metrics_list = []
            for l in layer_indices:
                for h in head_indices:
                    kk = eval_keys.get((l, h), []); vv = eval_vals.get((l, h), [])
                    if not kk or not vv: continue
                    K_all = torch.cat(kk, dim=0)[:512].to(device).float()
                    V_all = torch.cat(vv, dim=0)[:512].to(device).float()
                    Q_all = K_all.clone()
                    if K_all.shape[0] < 32: continue

                    scale = 1.0 / math.sqrt(hd)
                    fp16_output = F.softmax(Q_all @ K_all.T * scale, dim=-1) @ V_all

                    k_ev = eigen_keys[(l, h)]["ev"].cpu().numpy()
                    v_ev = eigen_vals[(l, h)]["ev"].cpu().numpy()
                    k_bits = optimal_bit_allocation(k_ev, key_budget)
                    v_bits = optimal_bit_allocation(v_ev, val_budget)

                    k_engine = OptimalAllocEngine(
                        eigen_keys[(l, h)]["evec"].to(device),
                        eigen_keys[(l, h)]["ev"].to(device),
                        k_bits, hd)
                    v_engine = OptimalAllocEngine(
                        eigen_vals[(l, h)]["evec"].to(device),
                        eigen_vals[(l, h)]["ev"].to(device),
                        v_bits, hd)

                    m = evaluate_allocation(k_engine, v_engine, Q_all, K_all, V_all, fp16_output)
                    m["key_frac"] = key_frac
                    metrics_list.append(m)

            if metrics_list:
                avg_cos = float(np.mean([m["attn_cos_sim"] for m in metrics_list]))
                if best is None or avg_cos > best["attn_cos_sim"]:
                    # Get representative bit allocation for reporting
                    l0, h0 = layer_indices[1], head_indices[0]
                    k_ev = eigen_keys[(l0, h0)]["ev"].cpu().numpy()
                    v_ev = eigen_vals[(l0, h0)]["ev"].cpu().numpy()
                    kb = int(total_budget * key_frac)
                    vb = total_budget - kb
                    k_bits_rep = optimal_bit_allocation(k_ev, kb)
                    v_bits_rep = optimal_bit_allocation(v_ev, vb)

                    best = {
                        "total_budget": total_budget,
                        "key_budget": kb,
                        "val_budget": vb,
                        "key_frac": float(key_frac),
                        "attn_cos_sim": avg_cos,
                        "attn_cos_sim_std": float(np.std([m["attn_cos_sim"] for m in metrics_list])),
                        "key_cos_sim": float(np.mean([m["key_cos_sim"] for m in metrics_list])),
                        "val_cos_sim": float(np.mean([m["val_cos_sim"] for m in metrics_list])),
                        "total_bits": metrics_list[0]["total_bits"],
                        "total_compress": float(np.mean([m["total_compress"] for m in metrics_list])),
                        "key_bit_allocation": k_bits_rep.tolist(),
                        "val_bit_allocation": v_bits_rep.tolist(),
                        "key_alloc_summary": dict(Counter(k_bits_rep)),
                        "val_alloc_summary": dict(Counter(v_bits_rep)),
                        "n_evals": len(metrics_list),
                    }

        if best:
            all_results[str(total_budget)] = best
            log.info("  Budget=%d: cos=%.4f, compress=%.1fx, split=%.0f%%K/%.0f%%V",
                     total_budget, best["attn_cos_sim"], best["total_compress"],
                     best["key_frac"]*100, (1-best["key_frac"])*100)
            k_counts = Counter(best["key_bit_allocation"])
            v_counts = Counter(best["val_bit_allocation"])
            log.info("    Keys: %s", ", ".join(f"{b}b:{c}" for b, c in sorted(k_counts.items(), reverse=True)))
            log.info("    Vals: %s", ", ".join(f"{b}b:{c}" for b, c in sorted(v_counts.items(), reverse=True)))

    result = {
        "model": model_name, "short_name": short_name,
        "head_dim": hd, "n_layers": n_layers, "n_kv_heads": n_kv,
        "mean_d_eff_keys": float(deff_keys), "mean_d_eff_values": float(deff_vals),
        "budgets": all_results,
    }
    save_result(f"optimal_{short_name}.json", result)

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    all_results = {}
    t0 = time.time()
    for model_name, short_name in MODELS:
        try:
            result = run_model(model_name, short_name, N_CALIB, N_EVAL, args.device)
            all_results[short_name] = result
        except Exception as e:
            log.error("FAILED on %s: %s", model_name, e)
            import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    save_result("optimal_combined.json", {
        "experiment": "optimal_allocation_sweep",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "models": all_results,
    })

    log.info("\n" + "=" * 70)
    log.info("OPTIMAL ALLOCATION SWEEP COMPLETE (%.1f min)", elapsed / 60)
    log.info("=" * 70)

    for sn, result in all_results.items():
        log.info("\n--- %s ---", sn)
        for budget_str, b in sorted(result["budgets"].items(), key=lambda x: int(x[0])):
            log.info("  %s bits: cos=%.4f, %.1fx compress",
                     budget_str, b["attn_cos_sim"], b["total_compress"])


if __name__ == "__main__":
    main()
