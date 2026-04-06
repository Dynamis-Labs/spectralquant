"""
Memory efficiency maximization: SpectralQuant vs TurboQuant across Qwen model sizes.

Goal: maximize compression ratio while maintaining acceptable quality.
Test aggressive variants: reduce tail bits, skip tail QJL, reduce tail value bits.

Models: Qwen2.5-1.5B, Qwen2.5-7B, Qwen2.5-14B (if VRAM allows)
"""
import sys, os, math, time, json, logging, argparse
from pathlib import Path
import torch
import numpy as np
from scipy import integrate

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from turboquant_cutile import TurboQuantEngine
from turboquant_cutile.codebook import LloydMaxCodebook

log = logging.getLogger("memeff")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")


def quantize_nearest(x, centroids):
    c = centroids.to(x.device)
    diffs = x.unsqueeze(-1) - c
    idx = diffs.abs().argmin(dim=-1)
    return c[idx.long()]


@torch.no_grad()
def sq_flexible(Q, K, V, evec, d_eff, head_dim, key_mse_bits, val_high_bits, val_low_bits, device, use_qjl=True):
    """Flexible SpectralQuant: control key/value bits per regime independently."""
    K_f, V_f, Q_f = K.float(), V.float(), Q.float()
    VT = evec.T.contiguous().to(device).float()
    Vm = evec.to(device).float()

    # TurboQuant-style: normalize then rotate
    k_n = torch.norm(K_f, dim=-1, keepdim=True)
    K_rot = (K_f / (k_n + 1e-8)) @ VT
    v_n = torch.norm(V_f, dim=-1, keepdim=True)
    V_rot = (V_f / (v_n + 1e-8)) @ VT

    # Key quantization: uniform MSE bits across all coords (same codebook)
    cb_key = LloydMaxCodebook(head_dim, key_mse_bits)
    K_hat_rot = quantize_nearest(K_rot, cb_key.centroids)

    # Value quantization: non-uniform — more bits for semantic, fewer for tail
    cb_val_h = LloydMaxCodebook(head_dim, val_high_bits)
    cb_val_l = LloydMaxCodebook(head_dim, val_low_bits)
    V_hat_h = quantize_nearest(V_rot[:, :d_eff], cb_val_h.centroids)
    V_hat_l = quantize_nearest(V_rot[:, d_eff:], cb_val_l.centroids)
    V_hat_rot = torch.cat([V_hat_h, V_hat_l], dim=-1)

    K_mse = (K_hat_rot @ Vm) * k_n
    V_rec = (V_hat_rot @ Vm) * v_n

    # Selective QJL on keys
    residual = K_f - K_mse
    r_norms = torch.norm(residual, dim=-1)
    gen = torch.Generator(device="cpu"); gen.manual_seed(52)
    S = torch.randn(head_dim, head_dim, generator=gen).to(device).float()

    if use_qjl and d_eff >= 2:
        res_sem = (residual @ VT)[:, :d_eff]
        S_sel = S[:d_eff, :d_eff]
        signs = torch.sign(res_sem @ S_sel.T); signs[signs == 0] = 1
        q_sem = (Q_f @ VT)[:, :d_eff]
        qjl_ip = (q_sem @ S_sel.T) @ signs.T
        corr = math.sqrt(math.pi / 2) / d_eff
        scores = (Q_f @ K_mse.T + corr * qjl_ip * r_norms.unsqueeze(0)) / math.sqrt(head_dim)
    else:
        scores = (Q_f @ K_mse.T) / math.sqrt(head_dim)

    return (torch.softmax(scores, dim=-1) @ V_rec).half()


def compute_memory(head_dim, d_eff, key_mse_bits, key_qjl_bits, val_high_bits, val_low_bits, seq_len, n_layers, n_kv_heads):
    """Compute actual memory usage in bytes."""
    # Keys
    key_mse = seq_len * head_dim * key_mse_bits
    key_qjl = seq_len * d_eff * key_qjl_bits  # selective QJL
    key_norms = seq_len * 32  # vec_norm + residual_norm (2 × FP16)

    # Values
    val_sem = seq_len * d_eff * val_high_bits
    val_tail = seq_len * (head_dim - d_eff) * val_low_bits
    val_norms = seq_len * 16  # vec_norm (FP16)

    total_bits = (key_mse + key_qjl + key_norms + val_sem + val_tail + val_norms) * n_layers * n_kv_heads
    total_bytes = total_bits / 8

    fp16_bytes = seq_len * head_dim * 2 * 2 * n_layers * n_kv_heads  # K+V in FP16
    ratio = fp16_bytes / total_bytes if total_bytes > 0 else 0
    avg_bits = total_bits / (seq_len * head_dim * 2 * n_layers * n_kv_heads)

    return {"total_bytes": total_bytes, "fp16_bytes": fp16_bytes,
            "ratio": ratio, "avg_bits": avg_bits, "total_mb": total_bytes/1e6}


def run_for_model(model_name, n_calib, n_eval, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    log.info("Loading %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=device)
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info("  %d layers, %d KV heads, head_dim=%d, %.1fB params", n_layers, n_kv, hd, n_params)

    # Calibrate
    log.info("  Calibrating (%d sequences)...", n_calib)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{n_calib*5}]")
    cov = {(l,h): {"xtx": torch.zeros(hd,hd,dtype=torch.float64), "n":0}
           for l in range(n_layers) for h in range(n_kv)}
    nd = 0
    t0 = time.time()
    for item in ds:
        text = item.get("text","")
        if len(text.strip())<100: continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1]<16: continue
        with torch.no_grad():
            out = model(**enc, use_cache=True); kv = out.past_key_values
        for l in range(n_layers):
            try: k_l = kv.key_cache[l].float().cpu()
            except:
                try: k_l = kv[l][0].float().cpu()
                except: k_l = list(kv)[l][0].float().cpu()
            for h in range(n_kv):
                X = k_l[0,h,:,:].double()
                cov[(l,h)]["xtx"] += X.T@X; cov[(l,h)]["n"] += X.shape[0]
        nd += 1
        if nd >= n_calib: break
        if nd % 50 == 0: log.info("    Cov: %d/%d (%.0fs)", nd, n_calib, time.time()-t0)

    eigen = {}
    for l in range(n_layers):
        for h in range(n_kv):
            C = (cov[(l,h)]["xtx"]/cov[(l,h)]["n"]).float()
            ev, evec = torch.linalg.eigh(C)
            ev = ev.flip(0).clamp(min=0); evec = evec.flip(1)
            d_eff = max(2, min(int(round((ev.sum()**2/(ev**2).sum()).item())), hd-2))
            kappa = min((ev[d_eff-1]/ev[min(d_eff,hd-1)].clamp(min=1e-10)).item(), 1e6)
            eigen[(l,h)] = {"ev":ev, "evec":evec, "d_eff":d_eff, "kappa":kappa}

    mean_deff = np.mean([eigen[(l,h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)])
    mean_kappa = np.mean([eigen[(l,h)]["kappa"] for l in range(n_layers) for h in range(n_kv)])
    log.info("  d_eff=%.1f, κ=%.2f (calibrated in %.1fs)", mean_deff, mean_kappa, time.time()-t0)

    # Define configurations to test
    d_eff_typical = int(round(mean_deff))
    configs = {
        # name: (key_mse_bits, key_has_qjl, val_high_bits, val_low_bits, d_eff)
        "TQ_3bit":          {"key_mse": 2, "qjl": True,  "qjl_full": True,  "vh": 3, "vl": 3, "d": hd},
        "TQ_2bit":          {"key_mse": 1, "qjl": True,  "qjl_full": True,  "vh": 2, "vl": 2, "d": hd},
        "SQ_selQJL":        {"key_mse": 2, "qjl": True,  "qjl_full": False, "vh": 3, "vl": 3, "d": d_eff_typical},
        "SQ_selQJL_v2tail": {"key_mse": 2, "qjl": True,  "qjl_full": False, "vh": 3, "vl": 2, "d": d_eff_typical},
        "SQ_selQJL_v1tail": {"key_mse": 2, "qjl": True,  "qjl_full": False, "vh": 3, "vl": 1, "d": d_eff_typical},
        "SQ_noQJL_v3":      {"key_mse": 2, "qjl": False, "qjl_full": False, "vh": 3, "vl": 3, "d": d_eff_typical},
        "SQ_noQJL_v2tail":  {"key_mse": 2, "qjl": False, "qjl_full": False, "vh": 3, "vl": 2, "d": d_eff_typical},
        "SQ_noQJL_v1tail":  {"key_mse": 2, "qjl": False, "qjl_full": False, "vh": 3, "vl": 1, "d": d_eff_typical},
        "SQ_k1_noQJL_v2t":  {"key_mse": 1, "qjl": False, "qjl_full": False, "vh": 2, "vl": 1, "d": d_eff_typical},
    }

    # Compute memory for each config
    seq_len = 8192
    log.info("\n  MEMORY ANALYSIS (seq_len=%d):", seq_len)
    log.info("  %-25s  %8s  %8s  %8s", "Config", "Ratio", "AvgBits", "MB@8K")

    mem_results = {}
    for name, c in configs.items():
        d = c["d"]
        qjl_bits = 1 if c["qjl"] else 0
        qjl_d = hd if c.get("qjl_full") else d
        mem = compute_memory(hd, qjl_d, c["key_mse"], qjl_bits, c["vh"], c["vl"], seq_len, n_layers, n_kv)
        mem_results[name] = mem
        log.info("  %-25s  %8.2f×  %8.2f  %8.2f", name, mem["ratio"], mem["avg_bits"], mem["total_mb"])

    # Evaluate quality
    log.info("\n  QUALITY EVALUATION (%d sequences)...", n_eval)
    eval_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"test[:{n_eval*5}]")
    slayers = list(range(0, n_layers, max(1, n_layers//5)))[:5]

    quality = {name: [] for name in configs}
    nev = 0
    for item in eval_ds:
        text = item.get("text","")
        if len(text.strip())<100: continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1]<16: continue
        with torch.no_grad():
            out = model(**enc, use_cache=True); kv = out.past_key_values

        for l in slayers:
            try: k_l = kv.key_cache[l].float().cpu(); v_l = kv.value_cache[l].float().cpu()
            except:
                try: k_l = kv[l][0].float().cpu(); v_l = kv[l][1].float().cpu()
                except: lkv=list(kv)[l]; k_l=lkv[0].float().cpu(); v_l=lkv[1].float().cpu()

            for h in range(n_kv):
                K = k_l[0,h].to(device).half(); V = v_l[0,h].to(device).half()
                if K.shape[0]<8: continue
                torch.manual_seed(42+l*1000+h)
                Qp = torch.randn(8, hd, device=device, dtype=torch.float16)
                sc = (Qp.float()@K.float().T)/math.sqrt(hd)
                ref = (torch.softmax(sc,dim=-1)@V.float()).half()
                ed = eigen[(l,h)]

                def cos(out_t):
                    c = torch.nn.functional.cosine_similarity(
                        ref.float().reshape(-1,hd), out_t.float().reshape(-1,hd), dim=-1
                    ).mean().item()
                    return c if not math.isnan(c) else None

                for name, c in configs.items():
                    try:
                        if c.get("qjl_full"):
                            # Standard TurboQuant
                            tq = TurboQuantEngine(head_dim=hd, total_bits=c["key_mse"]+1 if c["qjl"] else c["key_mse"], device=device)
                            ck = tq.compress_keys_pytorch(K); cv = tq.compress_values_pytorch(V)
                            val = cos(tq.fused_attention_pytorch(Qp, ck, cv))
                        else:
                            val = cos(sq_flexible(
                                Qp, K, V, ed["evec"], ed["d_eff"], hd,
                                c["key_mse"], c["vh"], c["vl"], device,
                                use_qjl=c["qjl"]
                            ))
                        if val is not None:
                            quality[name].append(val)
                    except: pass

        nev += 1
        if nev >= n_eval: break
        if nev % 10 == 0: log.info("    Eval: %d/%d", nev, n_eval)

    # Final results
    short = model_name.split("/")[-1]
    log.info("\n" + "=" * 90)
    log.info("RESULTS: %s (d_eff=%d, κ=%.2f, %d layers, %d KV heads)", short, d_eff_typical, mean_kappa, n_layers, n_kv)
    log.info("=" * 90)
    log.info("%-25s  %8s  %8s  %8s  %8s  %6s", "Config", "CosSim", "Ratio", "AvgBits", "MB@8K", "N")

    summary = {"model": short, "n_layers": n_layers, "n_kv_heads": n_kv, "head_dim": hd,
               "d_eff": d_eff_typical, "kappa": float(mean_kappa), "configs": {}}

    for name in configs:
        if quality[name]:
            m = np.mean(quality[name])
            mem = mem_results[name]
            log.info("%-25s  %8.4f  %8.2f×  %8.2f  %8.2f  %6d",
                     name, m, mem["ratio"], mem["avg_bits"], mem["total_mb"], len(quality[name]))
            summary["configs"][name] = {
                "cos_sim": float(m), "ratio": mem["ratio"],
                "avg_bits": mem["avg_bits"], "mb_8k": mem["total_mb"], "n": len(quality[name])
            }

    # Find Pareto-optimal configs
    log.info("\n  PARETO FRONTIER (quality vs compression):")
    items = [(name, summary["configs"][name]) for name in summary["configs"]]
    items.sort(key=lambda x: x[1]["ratio"])
    pareto = []
    best_cos = -1
    for name, data in items:
        if data["cos_sim"] > best_cos:
            pareto.append((name, data))
            best_cos = data["cos_sim"]
    for name, data in reversed(pareto):
        log.info("    %-25s  cos=%.4f  ratio=%.2f×  bits=%.2f", name, data["cos_sim"], data["ratio"], data["avg_bits"])

    out = PROJECT_ROOT / "results" / "memory_efficiency"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"memeff_{short}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("\nSaved to %s", out / f"memeff_{short}.json")

    del model; torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(42); np.random.seed(42)

    n_calib = 100 if args.quick else 500
    n_eval = 30 if args.quick else 200

    models = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    # Try 14B if VRAM allows
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if hasattr(torch.cuda.get_device_properties(0), 'total_memory') else 0
        if vram_gb > 150:
            models.append("Qwen/Qwen2.5-14B-Instruct")
            log.info("VRAM=%.0fGB — will also test 14B", vram_gb)

    all_results = {}
    for m in models:
        log.info("\n" + "="*90)
        log.info("MODEL: %s", m)
        log.info("="*90)
        try:
            all_results[m.split("/")[-1]] = run_for_model(m, n_calib, n_eval, args.device)
        except Exception as e:
            log.error("Failed on %s: %s", m, e)
            import traceback; traceback.print_exc()

    out = PROJECT_ROOT / "results" / "memory_efficiency"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "all_models.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("\nAll results saved to %s", out / "all_models.json")

if __name__ == "__main__":
    main()
