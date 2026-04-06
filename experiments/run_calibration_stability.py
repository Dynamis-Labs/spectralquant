"""Calibration stability: run eigendecomposition 3 times on different data splits."""
import sys, os, math, time, json, logging, torch, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))
log = logging.getLogger("stability")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

def calibrate_split(model, tokenizer, split_start, n_samples, device, n_layers, n_kv, hd):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[{split_start}:{split_start+n_samples*5}]")
    cov = {(l,h): {"xtx": torch.zeros(hd,hd,dtype=torch.float64), "n":0} for l in range(n_layers) for h in range(n_kv)}
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
        if nd >= n_samples: break
    d_effs = {}
    for l in range(n_layers):
        for h in range(n_kv):
            C = (cov[(l,h)]["xtx"]/cov[(l,h)]["n"]).float()
            ev, _ = torch.linalg.eigh(C)
            ev = ev.flip(0).clamp(min=0)
            d_eff = max(2, min(int(round((ev.sum()**2/(ev**2).sum()).item())), hd-2))
            d_effs[(l,h)] = d_eff
    return d_effs

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    device = args.device
    n_samples = 50 if args.quick else 200
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("Loading Qwen 2.5-1.5B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16, device_map=device)
    model.eval()
    cfg = model.config
    n_layers, n_kv = cfg.num_hidden_layers, getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads
    splits = [0, 5000, 10000]
    all_d_effs = []
    for i, start in enumerate(splits):
        log.info("Calibration run %d (start=%d, n=%d)...", i+1, start, n_samples)
        d_effs = calibrate_split(model, tokenizer, start, n_samples, device, n_layers, n_kv, hd)
        all_d_effs.append(d_effs)
        log.info("  mean d_eff = %.2f", np.mean(list(d_effs.values())))
    # Compare across runs
    log.info("\n=== STABILITY ANALYSIS ===")
    per_head_cv = []
    for l in range(n_layers):
        for h in range(n_kv):
            vals = [all_d_effs[i][(l,h)] for i in range(3)]
            mean_v = np.mean(vals); std_v = np.std(vals)
            cv = std_v / (mean_v + 1e-10)
            per_head_cv.append(cv)
            if cv > 0.3: log.info("  L%d H%d: d_eff=%s CV=%.2f (UNSTABLE)", l, h, vals, cv)
    log.info("Mean CV across heads: %.4f", np.mean(per_head_cv))
    log.info("Max CV: %.4f", np.max(per_head_cv))
    log.info("Heads with CV > 0.2: %d / %d", sum(1 for c in per_head_cv if c > 0.2), len(per_head_cv))
    stable = np.mean(per_head_cv) < 0.15
    log.info("VERDICT: %s (mean CV=%.4f)", "STABLE" if stable else "UNSTABLE", np.mean(per_head_cv))
    out = PROJECT_ROOT / "results" / "calibration_stability"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"mean_cv": float(np.mean(per_head_cv)), "max_cv": float(np.max(per_head_cv)),
               "stable": stable, "n_samples_per_split": n_samples,
               "per_split_mean_deff": [float(np.mean(list(d.values()))) for d in all_d_effs]}
    with open(out / "stability.json", "w") as f: json.dump(summary, f, indent=2)
    log.info("Saved to %s", out / "stability.json")

if __name__ == "__main__": main()
