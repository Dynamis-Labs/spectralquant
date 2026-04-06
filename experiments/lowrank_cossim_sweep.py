"""
Phase 1: Low-rank KV cache cosine similarity sweep.
Compress KV vectors to rank r = {2, 4, 8, 16, 32, 64} via eigenvector projection.
No quantization, no codebooks. Just linear projection down and back up.
"""
import sys, os, math, time, json, logging, argparse
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

log = logging.getLogger("lowrank")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

RESULTS_DIR = PROJECT_ROOT / "results" / "lowrank"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd):
    """Collect per-head eigenvectors (same as SpectralQuant calibration)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{n_calib*5}]")
    cov = {(l,h): {"xtx": torch.zeros(hd,hd,dtype=torch.float64), "n":0}
           for l in range(n_layers) for h in range(n_kv)}
    nd = 0
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
    eigen = {}
    for l in range(n_layers):
        for h in range(n_kv):
            C = (cov[(l,h)]["xtx"]/cov[(l,h)]["n"]).float()
            ev, evec = torch.linalg.eigh(C)
            ev = ev.flip(0).clamp(min=0); evec = evec.flip(1)
            d_eff = max(2, min(int(round((ev.sum()**2/(ev**2).sum()).item())), hd-2))
            eigen[(l,h)] = {"evec": evec, "eigenvalues": ev, "d_eff": d_eff}
    return eigen


def lowrank_compress_decompress(k, V_r):
    """Compress key to rank-r and decompress back. k: (seq, d), V_r: (d, r)."""
    norms = torch.norm(k, dim=-1, keepdim=True)  # (seq, 1)
    k_unit = k / (norms + 1e-8)
    proj = k_unit @ V_r        # (seq, r) — project to low-rank space
    k_hat = (proj @ V_r.T) * norms  # (seq, d) — reconstruct
    return k_hat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    log.info("Loading %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=device)
    model.eval()

    cfg = model.config; n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads

    n_calib = 50 if args.quick else 200
    log.info("Calibrating (%d seqs)...", n_calib)
    eigen = calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd)

    # Eval
    log.info("Evaluating cosine similarity at each rank...")
    eval_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"test[:{200 if args.quick else 1000}]")
    slayers = list(range(0, n_layers, max(1, n_layers//5)))[:5]
    n_eval = 20 if args.quick else 100

    ranks = [2, 4, 8, 16, 32, 64]
    results = {r: {"key_cos": [], "val_cos": [], "attn_cos": []} for r in ranks}

    nev = 0
    for item in eval_ds:
        text = item.get("text","")
        if len(text.strip())<100: continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1]<16: continue
        with torch.no_grad():
            out = model(**enc, use_cache=True); kv = out.past_key_values
        for l in slayers:
            try: k_l = kv.key_cache[l].float(); v_l = kv.value_cache[l].float()
            except:
                try: k_l = kv[l][0].float(); v_l = kv[l][1].float()
                except: lkv=list(kv)[l]; k_l=lkv[0].float(); v_l=lkv[1].float()

            for h in range(n_kv):
                K = k_l[0,h]  # (seq, hd)
                V = v_l[0,h]
                if K.shape[0] < 8: continue
                ed = eigen[(l,h)]

                # Random query probes for attention-level comparison
                torch.manual_seed(42+l*1000+h)
                Qp = torch.randn(8, hd, device=K.device, dtype=torch.float32)

                # FP16 reference attention
                scores_ref = (Qp @ K.T) / math.sqrt(hd)
                weights_ref = torch.softmax(scores_ref, dim=-1)
                attn_ref = weights_ref @ V

                for r in ranks:
                    V_r = ed["evec"][:, :r].to(K.device).float()

                    # Compress/decompress keys and values
                    K_hat = lowrank_compress_decompress(K, V_r)
                    V_hat = lowrank_compress_decompress(V, V_r)

                    # Key cosine similarity
                    kcos = torch.nn.functional.cosine_similarity(K, K_hat, dim=-1).mean().item()
                    vcos = torch.nn.functional.cosine_similarity(V, V_hat, dim=-1).mean().item()

                    # Attention output cosine similarity
                    scores_lr = (Qp @ K_hat.T) / math.sqrt(hd)
                    weights_lr = torch.softmax(scores_lr, dim=-1)
                    attn_lr = weights_lr @ V_hat
                    acos = torch.nn.functional.cosine_similarity(
                        attn_ref.reshape(-1, hd), attn_lr.reshape(-1, hd), dim=-1
                    ).mean().item()

                    if not (math.isnan(kcos) or math.isnan(vcos) or math.isnan(acos)):
                        results[r]["key_cos"].append(kcos)
                        results[r]["val_cos"].append(vcos)
                        results[r]["attn_cos"].append(acos)

        nev += 1
        if nev >= n_eval: break
        if nev % 20 == 0: log.info("  Eval: %d/%d", nev, n_eval)

    # Results
    log.info("\n" + "="*75)
    log.info("LOW-RANK KV CACHE: Cosine Similarity Sweep")
    log.info("="*75)
    log.info("%-6s  %10s  %10s  %10s  %10s  %6s", "Rank", "KeyCosSim", "ValCosSim", "AttnCosSim", "Compress", "N")

    summary = {}
    for r in ranks:
        if results[r]["attn_cos"]:
            km = np.mean(results[r]["key_cos"])
            vm = np.mean(results[r]["val_cos"])
            am = np.mean(results[r]["attn_cos"])
            ratio = (16 * hd * 2) / (16 * (r + 1) * 2)  # FP16 both K,V / lowrank both
            log.info("%-6d  %10.4f  %10.4f  %10.4f  %10.1fx  %6d",
                     r, km, vm, am, ratio, len(results[r]["attn_cos"]))
            summary[str(r)] = {
                "key_cos_mean": float(km), "key_cos_std": float(np.std(results[r]["key_cos"])),
                "val_cos_mean": float(vm), "val_cos_std": float(np.std(results[r]["val_cos"])),
                "attn_cos_mean": float(am), "attn_cos_std": float(np.std(results[r]["attn_cos"])),
                "compression_ratio": float(ratio),
                "bits_per_vector": int((r + 1) * 16),
                "n": len(results[r]["attn_cos"]),
            }

    # Reference: SpectralQuant and TurboQuant numbers
    log.info("\nReference:")
    log.info("  TurboQuant 3-bit: attn_cos=0.8443, ratio=5.02x")
    log.info("  SpectralQuant:    attn_cos=0.8615, ratio=5.95x")

    path = RESULTS_DIR / "lowrank_cossim_sweep.json"
    with open(path, "w") as f:
        json.dump({"model": model_name, "head_dim": hd, "results": summary}, f, indent=2)
    log.info("\nSaved: %s", path)

if __name__ == "__main__":
    main()
