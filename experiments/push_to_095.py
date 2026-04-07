"""
Push to 0.95: Asymmetric Bit Allocation (1-bit keys + high-bit values)
======================================================================
Keys need almost no bits (d_eff=4, sign is enough).
Values are the bottleneck. Focus ALL the bit budget on values.

Configs tested:
  1. 1-bit keys + 3-bit values (uniform after rotation)   -> 544 bits = 7.5x
  2. 1-bit keys + 4-bit values (uniform after rotation)   -> 672 bits = 6.1x  
  3. 1-bit keys + 5-bit values (uniform after rotation)   -> 800 bits = 5.1x
  4. 1-bit keys + optimal-alloc values (various budgets)  -> 5-8x
  5. 2-bit keys + 4-bit values (uniform after rotation)   -> 800 bits = 5.1x
  6. 2-bit keys + optimal-alloc values                    -> various
  7. Current SQ 2-regime baseline
  8. Uniform 3-bit baseline

Also: per-dimension key/value cos sim analysis to identify the bottleneck.
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

log = logging.getLogger("push095")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

RESULTS_DIR = PROJECT_ROOT / "results" / "push_095"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN")

MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct",   "Qwen-7B"),
]
if HF_TOKEN:
    MODELS.append(("meta-llama/Llama-3.1-8B-Instruct", "Llama-8B"))

N_CALIB = 64
N_EVAL = 32


def save_result(filename, data):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved: %s", path)


def solve_lloyd_max_for_sigma(sigma, bits, max_iter=200, tol=1e-10):
    if bits <= 0: return torch.tensor([0.0], dtype=torch.float32)
    if bits == 1:
        c = sigma * math.sqrt(2.0 / math.pi)
        return torch.tensor([-c, c], dtype=torch.float32)
    from scipy import integrate
    n_levels = 1 << bits
    pdf = lambda x: (1.0 / (math.sqrt(2*math.pi)*sigma)) * math.exp(-x*x/(2*sigma*sigma))
    lo, hi = -3.5*sigma, 3.5*sigma
    centroids = [lo + (hi-lo)*(i+0.5)/n_levels for i in range(n_levels)]
    for _ in range(max_iter):
        boundaries = [(centroids[i]+centroids[i+1])/2.0 for i in range(n_levels-1)]
        edges = [lo*3] + boundaries + [hi*3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i+1]
            num, _ = integrate.quad(lambda x: x*pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num/den if den>1e-15 else centroids[i])
        if max(abs(new_centroids[i]-centroids[i]) for i in range(n_levels)) < tol: break
        centroids = new_centroids
    return torch.tensor(centroids, dtype=torch.float32)


def quantize_nearest(x, centroids):
    c = centroids.to(x.device)
    diffs = x.unsqueeze(-1) - c
    return c[diffs.abs().argmin(dim=-1).long()]


def optimal_bit_allocation(eigenvalues, total_bits, min_bits=1, max_bits=6):
    d = len(eigenvalues)
    ev = np.array(eigenvalues, dtype=np.float64).clip(min=1e-30)
    lo, hi = 1e-30, ev.max()*10
    for _ in range(300):
        theta = (lo+hi)/2
        bits = np.clip(np.round(0.5*np.log2(np.maximum(ev/theta, 1e-30))), min_bits, max_bits)
        if bits.sum() > total_bits: lo = theta
        else: hi = theta
    bits = np.clip(np.round(0.5*np.log2(np.maximum(ev/theta, 1e-30))), min_bits, max_bits)
    while bits.sum() > total_bits:
        candidates = np.where(bits > min_bits)[0]
        if len(candidates) == 0: break
        worst = candidates[np.argmin(ev[candidates])]; bits[worst] -= 1
    while bits.sum() < total_bits:
        candidates = np.where(bits < max_bits)[0]
        if len(candidates) == 0: break
        best = candidates[np.argmax(ev[candidates]/(2.0**(2*bits[candidates])))]; bits[best] += 1
    return bits.astype(int)


def load_model_tokenizer(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    needs_token = any(x in model_name for x in ["llama", "Llama", "gemma", "Gemma"])
    token = HF_TOKEN if needs_token else None
    log.info("Loading %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device, token=token)
    model.eval()
    cfg = model.config; n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    try:
        test_ids = tokenizer("test", return_tensors="pt").to(device)
        with torch.no_grad(): test_out = model(**test_ids, use_cache=True)
        kv = test_out.past_key_values
        try: actual = kv.key_cache[0].shape[-1]
        except:
            try: actual = kv[0][0].shape[-1]
            except: actual = hd
        if actual != hd: hd = actual
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
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{n_calib*5}]")
    cov_keys = {(l,h): {"xtx": torch.zeros(hd,hd,dtype=torch.float64), "n":0} for l in range(n_layers) for h in range(n_kv)}
    cov_vals = {(l,h): {"xtx": torch.zeros(hd,hd,dtype=torch.float64), "n":0} for l in range(n_layers) for h in range(n_kv)}
    nd = 0
    for item in ds:
        text = item.get("text","")
        if len(text.strip()) < 100: continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16: continue
        with torch.no_grad():
            out = model(**enc, use_cache=True); kv = out.past_key_values
        for l in range(n_layers):
            try: k_l, v_l = extract_kv_layer(kv, l)
            except: continue
            k_l = k_l.float().cpu(); v_l = v_l.float().cpu()
            for h in range(min(n_kv, k_l.shape[1])):
                X_key = k_l[0,h].double(); X_val = v_l[0,h].double()
                cov_keys[(l,h)]["xtx"] += X_key.T @ X_key; cov_keys[(l,h)]["n"] += X_key.shape[0]
                cov_vals[(l,h)]["xtx"] += X_val.T @ X_val; cov_vals[(l,h)]["n"] += X_val.shape[0]
        nd += 1
        if nd >= n_calib: break
    def _eigen(cd):
        eigen = {}
        for l in range(n_layers):
            for h in range(n_kv):
                n = cd[(l,h)]["n"]
                if n == 0: eigen[(l,h)] = {"evec": torch.eye(hd), "ev": torch.ones(hd), "d_eff": hd}; continue
                C = (cd[(l,h)]["xtx"]/n).float()
                ev, evec = torch.linalg.eigh(C)
                ev = ev.flip(0).clamp(min=0); evec = evec.flip(1)
                d_eff = max(2, min(int(round((ev.sum()**2/(ev**2).sum()).item())), hd-2))
                eigen[(l,h)] = {"evec": evec, "ev": ev, "d_eff": d_eff}
        return eigen
    return _eigen(cov_keys), _eigen(cov_vals)


class AsymEngine:
    """Asymmetric bit allocation: fixed key bits + variable value bits."""
    def __init__(self, k_eigvecs, k_eigvals, v_eigvecs, v_eigvals, key_bits_per_dim, val_bits_alloc, hd):
        self.hd = hd
        self.Vk = k_eigvecs.float(); self.VkT = self.Vk.T.contiguous()
        self.Vv = v_eigvecs.float(); self.VvT = self.Vv.T.contiguous()
        
        k_ev = k_eigvals.float().clamp(min=1e-10).cpu().numpy()
        v_ev = v_eigvals.float().clamp(min=1e-10).cpu().numpy()
        
        # Key codebooks (uniform bit width)
        self.key_bits = key_bits_per_dim
        self.k_codebooks = []
        for i in range(hd):
            sigma = float(np.sqrt(k_ev[i]))
            self.k_codebooks.append(solve_lloyd_max_for_sigma(max(sigma,1e-6), key_bits_per_dim))
        
        # Value codebooks (per-dimension allocation)
        self.val_bits_alloc = val_bits_alloc
        self.v_codebooks = []
        for i in range(hd):
            b = int(val_bits_alloc[i])
            sigma = float(np.sqrt(v_ev[i]))
            self.v_codebooks.append(solve_lloyd_max_for_sigma(max(sigma,1e-6), max(b,1)))
        
        self.key_total_bits = hd * key_bits_per_dim + 16
        self.val_total_bits = int(np.sum(val_bits_alloc)) + 16
    
    def compress_keys(self, K):
        K_f = K.float(); norms = torch.norm(K_f, dim=-1, keepdim=True)
        rotated = (K_f / (norms + 1e-8)) @ self.Vk
        quantized = rotated.clone()
        for i in range(self.hd):
            quantized[:,i] = quantize_nearest(rotated[:,i], self.k_codebooks[i])
        return {"qr": quantized, "norms": norms.squeeze(-1)}
    
    def compress_values(self, V):
        V_f = V.float(); norms = torch.norm(V_f, dim=-1, keepdim=True)
        rotated = (V_f / (norms + 1e-8)) @ self.Vv
        quantized = rotated.clone()
        for i in range(self.hd):
            quantized[:,i] = quantize_nearest(rotated[:,i], self.v_codebooks[i])
        return {"qr": quantized, "norms": norms.squeeze(-1)}
    
    def decompress(self, compressed, VT):
        qr = compressed["qr"].float(); norms = compressed["norms"].float()
        return (qr @ VT.to(qr.device)) * norms.unsqueeze(-1)
    
    def evaluate(self, Q, K, V, fp16_output):
        ck = self.compress_keys(K); cv = self.compress_values(V)
        K_hat = self.decompress(ck, self.VkT); V_hat = self.decompress(cv, self.VvT)
        key_cos = F.cosine_similarity(K.float(), K_hat.float(), dim=-1).mean().item()
        val_cos = F.cosine_similarity(V.float(), V_hat.float(), dim=-1).mean().item()
        scale = 1.0 / math.sqrt(self.hd)
        output = F.softmax(Q.float() @ K_hat.T * scale, dim=-1) @ V_hat
        attn_cos = F.cosine_similarity(fp16_output.float(), output.float(), dim=-1).mean().item()
        return {"attn_cos_sim": attn_cos, "key_cos_sim": key_cos, "val_cos_sim": val_cos,
                "key_bits": self.key_total_bits, "val_bits": self.val_total_bits,
                "total_bits": self.key_total_bits + self.val_total_bits,
                "total_compress": (2*self.hd*16) / (self.key_total_bits + self.val_total_bits)}


def run_model(model_name, short_name, device):
    log.info("=" * 70)
    log.info("MODEL: %s", model_name)
    
    model, tokenizer, n_layers, n_kv, hd = load_model_tokenizer(model_name, device)
    eigen_keys, eigen_vals = calibrate(model, tokenizer, N_CALIB, device, n_layers, n_kv, hd)
    
    deff_k = np.mean([eigen_keys[(l,h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)])
    deff_v = np.mean([eigen_vals[(l,h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)])
    log.info("d_eff: keys=%.1f, values=%.1f", deff_k, deff_v)
    
    layer_indices = sorted(set([n_layers//4, n_layers//2, 3*n_layers//4]))
    head_indices = list(range(min(n_kv, 4)))
    
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[{N_CALIB*5}:{(N_CALIB+N_EVAL)*5}]")
    eval_keys = {(l,h): [] for l in layer_indices for h in head_indices}
    eval_vals = {(l,h): [] for l in layer_indices for h in head_indices}
    nd = 0
    for item in ds:
        text = item.get("text","")
        if len(text.strip()) < 100: continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 32: continue
        with torch.no_grad():
            out = model(**enc, use_cache=True); kv = out.past_key_values
        for l in layer_indices:
            try: k_l, v_l = extract_kv_layer(kv, l)
            except: continue
            k_l = k_l.float().cpu(); v_l = v_l.float().cpu()
            for h in head_indices:
                if h < k_l.shape[1]:
                    eval_keys[(l,h)].append(k_l[0,h]); eval_vals[(l,h)].append(v_l[0,h])
        nd += 1
        if nd >= N_EVAL: break
    
    # Define configs: (name, key_bits_per_dim, val_mode, val_param)
    configs = [
        # Fixed key bits + uniform value bits
        ("1bit-K_3bit-V", 1, "uniform", 3),
        ("1bit-K_4bit-V", 1, "uniform", 4),
        ("1bit-K_5bit-V", 1, "uniform", 5),
        ("1bit-K_6bit-V", 1, "uniform", 6),
        ("2bit-K_3bit-V", 2, "uniform", 3),
        ("2bit-K_4bit-V", 2, "uniform", 4),
        ("2bit-K_5bit-V", 2, "uniform", 5),
        # Fixed key bits + optimal value allocation
        ("1bit-K_optV-400", 1, "optimal", 400),
        ("1bit-K_optV-500", 1, "optimal", 500),
        ("1bit-K_optV-600", 1, "optimal", 600),
        ("1bit-K_optV-700", 1, "optimal", 700),
        ("1bit-K_optV-800", 1, "optimal", 800),
        ("2bit-K_optV-500", 2, "optimal", 500),
        ("2bit-K_optV-600", 2, "optimal", 600),
        # Baselines
        ("uniform-3bit", 3, "uniform", 3),
        ("uniform-4bit", 4, "uniform", 4),
    ]
    
    all_results = {}
    for config_name, key_bpd, val_mode, val_param in configs:
        log.info("  Config: %s", config_name)
        metrics_list = []
        
        for l in layer_indices:
            for h in head_indices:
                kk = eval_keys.get((l,h),[]); vv = eval_vals.get((l,h),[])
                if not kk or not vv: continue
                K_all = torch.cat(kk, dim=0)[:512].to(device).float()
                V_all = torch.cat(vv, dim=0)[:512].to(device).float()
                Q_all = K_all.clone()
                if K_all.shape[0] < 32: continue
                
                scale = 1.0 / math.sqrt(hd)
                fp16_output = F.softmax(Q_all @ K_all.T * scale, dim=-1) @ V_all
                
                v_ev = eigen_vals[(l,h)]["ev"].cpu().numpy()
                if val_mode == "uniform":
                    val_alloc = np.full(hd, val_param)
                else:
                    val_alloc = optimal_bit_allocation(v_ev, val_param)
                
                engine = AsymEngine(
                    eigen_keys[(l,h)]["evec"].to(device), eigen_keys[(l,h)]["ev"].to(device),
                    eigen_vals[(l,h)]["evec"].to(device), eigen_vals[(l,h)]["ev"].to(device),
                    key_bpd, val_alloc, hd)
                
                m = engine.evaluate(Q_all, K_all, V_all, fp16_output)
                metrics_list.append(m)
        
        if metrics_list:
            avg = {
                "config": config_name,
                "attn_cos_sim": float(np.mean([m["attn_cos_sim"] for m in metrics_list])),
                "attn_cos_sim_std": float(np.std([m["attn_cos_sim"] for m in metrics_list])),
                "key_cos_sim": float(np.mean([m["key_cos_sim"] for m in metrics_list])),
                "val_cos_sim": float(np.mean([m["val_cos_sim"] for m in metrics_list])),
                "key_bits": metrics_list[0]["key_bits"],
                "val_bits": metrics_list[0]["val_bits"],
                "total_bits": metrics_list[0]["total_bits"],
                "total_compress": metrics_list[0]["total_compress"],
                "n_evals": len(metrics_list),
            }
            all_results[config_name] = avg
            log.info("    attn=%.4f, key=%.4f, val=%.4f | %d bits = %.1fx",
                     avg["attn_cos_sim"], avg["key_cos_sim"], avg["val_cos_sim"],
                     avg["total_bits"], avg["total_compress"])
    
    result = {
        "model": model_name, "short_name": short_name,
        "head_dim": hd, "n_layers": n_layers, "n_kv_heads": n_kv,
        "mean_d_eff_keys": float(deff_k), "mean_d_eff_values": float(deff_v),
        "configs": all_results,
    }
    save_result(f"push095_{short_name}.json", result)
    
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    all_results = {}; t0 = time.time()
    for model_name, short_name in MODELS:
        try:
            result = run_model(model_name, short_name, args.device)
            all_results[short_name] = result
        except Exception as e:
            log.error("FAILED on %s: %s", model_name, e)
            import traceback; traceback.print_exc()
    
    elapsed = time.time() - t0
    save_result("push095_combined.json", {"experiment": "push_to_095", "elapsed_seconds": elapsed, "models": all_results})
    
    log.info("\n" + "=" * 70)
    log.info("PUSH TO 0.95 COMPLETE (%.1f min)", elapsed / 60)
    for sn, result in all_results.items():
        log.info("\n--- %s ---", sn)
        for name, c in sorted(result["configs"].items(), key=lambda x: -x[1]["attn_cos_sim"]):
            log.info("  %25s: attn=%.4f, key=%.4f, val=%.4f | %d bits = %.1fx",
                     name, c["attn_cos_sim"], c["key_cos_sim"], c["val_cos_sim"], c["total_bits"], c["total_compress"])


if __name__ == "__main__":
    main()
