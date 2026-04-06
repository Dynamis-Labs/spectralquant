"""
NeurIPS SpectralQuant — Script B: New Model Families + KV Eigenvalue Asymmetry
==============================================================================
  PART 1: Mistral 7B evaluation (TQ vs SQ cosine sim, d_eff, kappa)
  PART 2: Gemma 9B evaluation (head_dim=256 generalization test)
  PART 3: Perplexity on Qwen 2.5-7B (1K–8K)
  PART 4: Key/Value eigenvalue asymmetry across ALL models

Models:
  Qwen 2.5-1.5B-Instruct, 7B-Instruct, 14B-Instruct
  Llama 3.1-8B-Instruct (HF_TOKEN required)
  Mistral 7B-Instruct-v0.3
  Gemma-2-9B-IT (HF_TOKEN required)

Usage:
    python neurips_models_asymmetry.py [--quick] [--part {0,1,2,3,4}] [--device cuda]
"""

import sys, os, math, time, json, logging, argparse, gc
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# ── project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from turboquant_cutile import TurboQuantEngine, LloydMaxCodebook

log = logging.getLogger("neurips_models")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

RESULTS_DIR = PROJECT_ROOT / "results" / "neurips"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def save_result(filename, data):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved: %s", path)


def quantize_nearest(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    c = centroids.to(x.device)
    diffs = x.unsqueeze(-1) - c
    return c[diffs.abs().argmin(dim=-1).long()]


def load_model_tokenizer(model_name: str, device: str):
    """Load a HuggingFace model + tokenizer (with HF_TOKEN for gated models)."""
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
    n_params = sum(p.numel() for p in model.parameters()) / 1e9

    # Detect actual KV head_dim from a forward pass (some models like Gemma 2
    # use different head_dim for Q vs KV)
    try:
        test_ids = tokenizer("test", return_tensors="pt").to(device)
        with torch.no_grad():
            test_out = model(**test_ids, use_cache=True)
        kv = test_out.past_key_values
        try:
            actual_kv_hd = kv.key_cache[0].shape[-1]
        except:
            try: actual_kv_hd = kv[0][0].shape[-1]
            except: actual_kv_hd = hd
        if actual_kv_hd != hd:
            log.info("  KV head_dim=%d differs from config head_dim=%d; using KV dim", actual_kv_hd, hd)
            hd = actual_kv_hd
    except Exception as e:
        log.warning("  Could not detect KV head_dim: %s", e)

    log.info("  %d layers, %d KV heads, head_dim=%d, %.1fB params",
             n_layers, n_kv, hd, n_params)
    return model, tokenizer, n_layers, n_kv, hd


def calibrate_keys_and_values(model, tokenizer, n_calib, device, n_layers, n_kv, hd):
    """
    Calibrate SEPARATELY for keys and values.
    Returns:
      eigen_keys: {(l,h): {"evec", "ev", "d_eff"}}
      eigen_vals: {(l,h): {"evec", "ev", "d_eff"}}
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
                k_l = kv.key_cache[l].float().cpu()
                v_l = kv.value_cache[l].float().cpu()
            except Exception:
                try:
                    k_l = kv[l][0].float().cpu()
                    v_l = kv[l][1].float().cpu()
                except Exception:
                    entry = list(kv)[l]
                    k_l = entry[0].float().cpu()
                    v_l = entry[1].float().cpu()

            for h in range(n_kv):
                X_key = k_l[0, h, :, :].double()
                cov_keys[(l, h)]["xtx"] += X_key.T @ X_key
                cov_keys[(l, h)]["n"] += X_key.shape[0]

                X_val = v_l[0, h, :, :].double()
                cov_vals[(l, h)]["xtx"] += X_val.T @ X_val
                cov_vals[(l, h)]["n"] += X_val.shape[0]

        nd += 1
        if nd >= n_calib:
            break
        if nd % 50 == 0:
            log.info("  Calibration: %d/%d (%.0fs)", nd, n_calib, time.time() - t0)

    def _eigendecompose(cov_dict):
        eigen = {}
        for l in range(n_layers):
            for h in range(n_kv):
                C = (cov_dict[(l, h)]["xtx"] / cov_dict[(l, h)]["n"]).float()
                ev, evec = torch.linalg.eigh(C)
                ev = ev.flip(0).clamp(min=0)
                evec = evec.flip(1)
                d_eff = max(2, min(int(round((ev.sum() ** 2 / (ev ** 2).sum()).item())), hd - 2))
                kappa = float(min((ev[d_eff - 1] / ev[min(d_eff, hd - 1)].clamp(min=1e-10)).item(), 1e6))
                eigen[(l, h)] = {"evec": evec, "ev": ev, "d_eff": d_eff, "kappa": kappa}
        return eigen

    eigen_keys = _eigendecompose(cov_keys)
    eigen_vals = _eigendecompose(cov_vals)

    log.info("Calibration done in %.1fs.", time.time() - t0)
    log.info("  mean d_eff_keys=%.1f, mean d_eff_vals=%.1f",
             float(np.mean([eigen_keys[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)])),
             float(np.mean([eigen_vals[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)])))

    return eigen_keys, eigen_vals


def calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd):
    """Standard key-only calibration (for TQ/SQ eval)."""
    eigen_keys, _ = calibrate_keys_and_values(
        model, tokenizer, n_calib, device, n_layers, n_kv, hd
    )
    return eigen_keys


def extract_kv_layer(kv, l):
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


@torch.no_grad()
def sq_noqjl(Q, K, V, evec, d_eff, hd, device):
    """SpectralQuant: spectral rotation, 2-bit keys, 3-bit values, no QJL."""
    K_f, V_f, Q_f = K.float(), V.float(), Q.float()
    VT = evec.T.contiguous().to(device).float()
    Vm = evec.to(device).float()

    k_n = torch.norm(K_f, dim=-1, keepdim=True)
    K_rot = (K_f / (k_n + 1e-8)) @ VT
    v_n = torch.norm(V_f, dim=-1, keepdim=True)
    V_rot = (V_f / (v_n + 1e-8)) @ VT

    cb_k = LloydMaxCodebook(hd, 2)
    cb_v = LloydMaxCodebook(hd, 3)
    K_hat = quantize_nearest(K_rot, cb_k.centroids)
    V_hat = quantize_nearest(V_rot, cb_v.centroids)

    K_mse = (K_hat @ Vm) * k_n
    V_rec = (V_hat @ Vm) * v_n
    scores = (Q_f @ K_mse.T) / math.sqrt(hd)
    return (torch.softmax(scores, dim=-1) @ V_rec).half()


# ── PPL token-by-token hook infrastructure ────────────────────────────────────

def install_kv_compression_hooks(model, method, eigen, n_layers, n_kv, hd, device):
    """Install compression hooks for token-by-token PPL measurement."""
    handles = []
    if method == "fp16":
        return handles

    for layer_idx in range(n_layers):
        attn = model.model.layers[layer_idx].self_attn

        if method == "tq":
            tq_engine = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)
        else:
            tq_engine = None

        def make_hook(l_idx, engine, meth):
            @torch.no_grad()
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 3:
                    cache = output[2]
                    if cache is None:
                        return output
                    try:
                        k = cache.key_cache[l_idx]
                        v = cache.value_cache[l_idx]
                        k_out = k.clone()
                        v_out = v.clone()
                        for h in range(min(n_kv, k.shape[1])):
                            k_h = k[0, h].half()
                            v_h = v[0, h].half()
                            if meth == "sq":
                                ed = eigen.get((l_idx, h))
                                if ed is None:
                                    continue
                                ev = ed["evec"].to(device).float()
                                VT = ev.T.contiguous()
                                k_n = torch.norm(k_h.float(), dim=-1, keepdim=True)
                                K_rot = (k_h.float() / (k_n + 1e-8)) @ VT
                                cb_k = LloydMaxCodebook(hd, 2)
                                K_hat = quantize_nearest(K_rot, cb_k.centroids.to(device))
                                k_r = ((K_hat @ ev) * k_n).to(k.dtype)
                                v_n = torch.norm(v_h.float(), dim=-1, keepdim=True)
                                V_rot = (v_h.float() / (v_n + 1e-8)) @ VT
                                cb_v = LloydMaxCodebook(hd, 3)
                                V_hat = quantize_nearest(V_rot, cb_v.centroids.to(device))
                                v_r = ((V_hat @ ev) * v_n).to(v.dtype)
                                k_out[0, h] = k_r
                                v_out[0, h] = v_r
                            elif meth == "tq":
                                ck = engine.compress_keys_pytorch(k_h)
                                cv = engine.compress_values_pytorch(v_h)
                                k_out[0, h] = ck["k_mse"].to(k.dtype)
                                v_out[0, h] = engine.decompress_values_pytorch(cv).to(v.dtype)
                        cache.key_cache[l_idx] = k_out
                        cache.value_cache[l_idx] = v_out
                    except (AttributeError, IndexError):
                        pass
                return output
            return hook

        h = attn.register_forward_hook(make_hook(layer_idx, tq_engine, method))
        handles.append(h)

    return handles


def remove_hooks(handles):
    for h in handles:
        if hasattr(h, "remove"):
            h.remove()


# ══════════════════════════════════════════════════════════════════════════════
# TQ vs SQ COSINE SIMILARITY EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_tq_sq_cossim(model, tokenizer, eigen, n_layers, n_kv, hd, device,
                       n_eval: int) -> dict:
    """Evaluate TQ and SQ attention output cosine similarity vs FP16 baseline."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"test[:{n_eval * 5}]")
    slayers = list(range(0, n_layers, max(1, n_layers // 5)))[:5]

    tq_sims, sq_sims = [], []
    nev = 0

    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt",
                        max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue
        try:
            with torch.no_grad():
                out = model(**enc, use_cache=True)
                kv = out.past_key_values
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        for l in slayers:
            try:
                k_l, v_l = extract_kv_layer(kv, l)
                k_l = k_l.float().cpu()
                v_l = v_l.float().cpu()
            except Exception:
                continue

            for h in range(n_kv):
                K = k_l[0, h].to(device).half()
                V = v_l[0, h].to(device).half()
                if K.shape[0] < 8:
                    continue

                torch.manual_seed(42 + l * 1000 + h)
                Qp = torch.randn(8, hd, device=device, dtype=torch.float16)
                sc = (Qp.float() @ K.float().T) / math.sqrt(hd)
                ref = (torch.softmax(sc, dim=-1) @ V.float()).half()

                def cos(o):
                    c = torch.nn.functional.cosine_similarity(
                        ref.float().reshape(-1, hd),
                        o.float().reshape(-1, hd),
                        dim=-1,
                    ).mean().item()
                    return c if not math.isnan(c) else None

                # TQ
                try:
                    tq = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)
                    ck = tq.compress_keys_pytorch(K)
                    cv = tq.compress_values_pytorch(V)
                    val = cos(tq.fused_attention_pytorch(Qp, ck, cv))
                    if val is not None:
                        tq_sims.append(val)
                except Exception:
                    pass

                # SQ
                try:
                    ed = eigen[(l, h)]
                    val = cos(sq_noqjl(Qp, K, V, ed["evec"], ed["d_eff"], hd, device))
                    if val is not None:
                        sq_sims.append(val)
                except Exception:
                    pass

        nev += 1
        if nev >= n_eval:
            break
        if nev % 20 == 0:
            log.info("  Eval: %d/%d", nev, n_eval)

    return {
        "tq_cos_sim_mean": float(np.mean(tq_sims)) if tq_sims else None,
        "tq_cos_sim_std": float(np.std(tq_sims)) if tq_sims else None,
        "sq_cos_sim_mean": float(np.mean(sq_sims)) if sq_sims else None,
        "sq_cos_sim_std": float(np.std(sq_sims)) if sq_sims else None,
        "n": len(tq_sims),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: MISTRAL 7B
# ══════════════════════════════════════════════════════════════════════════════

def run_part1_mistral(device, n_calib, n_eval) -> dict:
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    try:
        model, tokenizer, n_layers, n_kv, hd = load_model_tokenizer(model_name, device)
    except Exception as e:
        log.error("Failed to load Mistral: %s", e)
        return {"error": str(e)}

    log.info("Calibrating Mistral (%d seqs) ...", n_calib)
    eigen_keys, eigen_vals = calibrate_keys_and_values(
        model, tokenizer, n_calib, device, n_layers, n_kv, hd
    )

    mean_deff_k = float(np.mean([eigen_keys[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)]))
    mean_kappa = float(np.mean([eigen_keys[(l, h)]["kappa"] for l in range(n_layers) for h in range(n_kv)]))

    log.info("Evaluating TQ vs SQ cosine similarity (%d seqs) ...", n_eval)
    cossim = eval_tq_sq_cossim(model, tokenizer, eigen_keys, n_layers, n_kv, hd, device, n_eval)

    # Compression ratio
    fp16_bits = hd * 16
    sq_bits = hd * 2 + 16 + hd * 3 + 16  # key 2bit+norm + val 3bit+norm (rough)
    tq_bits = hd * 2 + hd + 16 + hd * 3 + 16  # key 2bit+1bit-QJL+norm + val 3bit+norm
    sq_ratio = (fp16_bits * 2) / (sq_bits)
    tq_ratio = (fp16_bits * 2) / (tq_bits)

    result = {
        "model": model_name,
        "n_layers": n_layers, "n_kv_heads": n_kv, "head_dim": hd,
        "mean_d_eff": mean_deff_k,
        "mean_kappa": mean_kappa,
        "cos_sim": cossim,
        "compression_ratio": {"sq": float(sq_ratio), "tq": float(tq_ratio)},
    }
    log.info("Mistral: d_eff=%.1f κ=%.2f  TQ_sim=%.4f  SQ_sim=%.4f",
             mean_deff_k, mean_kappa,
             cossim.get("tq_cos_sim_mean") or float("nan"),
             cossim.get("sq_cos_sim_mean") or float("nan"))

    del model
    torch.cuda.empty_cache()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: GEMMA 9B  (head_dim=256)
# ══════════════════════════════════════════════════════════════════════════════

def run_part2_gemma(device, n_calib, n_eval) -> dict:
    model_name = "google/gemma-2-9b-it"
    try:
        model, tokenizer, n_layers, n_kv, hd = load_model_tokenizer(model_name, device)
    except Exception as e:
        log.error("Failed to load Gemma: %s", e)
        return {"error": str(e)}

    log.info("Gemma head_dim=%d (expected 256)", hd)
    log.info("Calibrating Gemma (%d seqs) ...", n_calib)
    eigen_keys, eigen_vals = calibrate_keys_and_values(
        model, tokenizer, n_calib, device, n_layers, n_kv, hd
    )

    mean_deff_k = float(np.mean([eigen_keys[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)]))
    mean_deff_v = float(np.mean([eigen_vals[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)]))
    mean_kappa = float(np.mean([eigen_keys[(l, h)]["kappa"] for l in range(n_layers) for h in range(n_kv)]))

    log.info("Gemma d_eff_keys=%.1f d_eff_vals=%.1f (head_dim=%d)", mean_deff_k, mean_deff_v, hd)
    log.info("  d_eff / head_dim = %.3f (is it ~constant across architectures?)",
             mean_deff_k / hd)

    log.info("Evaluating TQ vs SQ (%d seqs) ...", n_eval)
    cossim = eval_tq_sq_cossim(model, tokenizer, eigen_keys, n_layers, n_kv, hd, device, n_eval)

    # Compression ratio
    fp16_bits = hd * 16 * 2  # K + V
    sq_bits_per_tok = hd * 2 + 16 + hd * 3 + 16
    tq_bits_per_tok = hd * 2 + hd + 16 + hd * 3 + 16
    sq_ratio = fp16_bits / sq_bits_per_tok
    tq_ratio = fp16_bits / tq_bits_per_tok

    result = {
        "model": model_name,
        "n_layers": n_layers, "n_kv_heads": n_kv, "head_dim": hd,
        "mean_d_eff_keys": mean_deff_k,
        "mean_d_eff_vals": mean_deff_v,
        "d_eff_over_head_dim": float(mean_deff_k / hd),
        "mean_kappa": mean_kappa,
        "cos_sim": cossim,
        "compression_ratio": {"sq": float(sq_ratio), "tq": float(tq_ratio)},
        "note": "head_dim=256 dimensionality generalization test",
    }
    log.info("Gemma: d_eff=%.1f (%.1f%% of hd=%d)  TQ_sim=%.4f  SQ_sim=%.4f",
             mean_deff_k, 100 * mean_deff_k / hd, hd,
             cossim.get("tq_cos_sim_mean") or float("nan"),
             cossim.get("sq_cos_sim_mean") or float("nan"))

    del model
    torch.cuda.empty_cache()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: PERPLEXITY ON QWEN 2.5-7B
# ══════════════════════════════════════════════════════════════════════════════

def run_ppl_ctx_length(model, tokenizer, method, eigen, n_layers, n_kv, hd,
                        device, text_ids: torch.Tensor, ctx_len: int) -> float:
    seq = text_ids[:, :ctx_len]
    seq_len = seq.shape[1]

    handles = install_kv_compression_hooks(model, method, eigen, n_layers, n_kv, hd, device)
    nlls = []
    past = None

    try:
        for i in range(seq_len - 1):
            token = seq[:, i:i + 1]
            with torch.no_grad():
                out = model(token, past_key_values=past, use_cache=True)
            past = out.past_key_values

            logits = out.logits[:, -1, :]
            target = seq[:, i + 1]
            nll = F.cross_entropy(logits, target, reduction="none")
            nlls.append(nll.item())

            if (i + 1) % 500 == 0:
                log.info("    [%s ctx=%d] %d/%d PPL=%.2f",
                         method, ctx_len, i + 1, seq_len - 1, math.exp(np.mean(nlls)))
    finally:
        remove_hooks(handles)

    return math.exp(float(np.mean(nlls))) if nlls else float("nan")


def run_part3_qwen7b_ppl(device, n_calib, ctx_lengths, partial_results=None) -> dict:
    from datasets import load_dataset

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    try:
        model, tokenizer, n_layers, n_kv, hd = load_model_tokenizer(model_name, device)
    except Exception as e:
        log.error("Failed to load Qwen 7B: %s", e)
        return {"error": str(e)}

    log.info("Calibrating Qwen 7B (%d seqs) ...", n_calib)
    eigen = calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([item["text"] for item in ds if len(item.get("text", "").strip()) > 0])
    max_needed = max(ctx_lengths)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_needed + 1)
    text_ids = enc["input_ids"].to(device)
    log.info("WikiText-2 tokens: %d", text_ids.shape[1])

    results = partial_results or {}
    methods = ["fp16", "tq", "sq"]

    for ctx_len in ctx_lengths:
        ctx_key = str(ctx_len)
        if ctx_key in results:
            log.info("Skipping ctx_len=%d (done)", ctx_len)
            continue
        if text_ids.shape[1] < ctx_len:
            log.warning("Not enough tokens for ctx_len=%d", ctx_len)
            continue

        results[ctx_key] = {}
        for method in methods:
            log.info("PPL [%s] ctx=%d ...", method, ctx_len)
            t0 = time.time()
            try:
                ppl = run_ppl_ctx_length(model, tokenizer, method, eigen,
                                          n_layers, n_kv, hd, device, text_ids, ctx_len)
                elapsed = time.time() - t0
                log.info("  %s ctx=%d: PPL=%.3f (%.1fs)", method, ctx_len, ppl, elapsed)
                results[ctx_key][method] = {"ppl": float(ppl), "time_s": float(elapsed)}
            except torch.cuda.OutOfMemoryError:
                log.warning("  OOM: %s ctx=%d", method, ctx_len)
                torch.cuda.empty_cache()
                results[ctx_key][method] = {"ppl": None, "error": "OOM"}
            except Exception as exc:
                log.warning("  Error %s ctx=%d: %s", method, ctx_len, exc)
                results[ctx_key][method] = {"ppl": None, "error": str(exc)}

            torch.cuda.empty_cache()

        save_result("neurips_qwen7b_ppl.json", {
            "model": model_name, "dataset": "wikitext-2-raw-v1",
            "ctx_lengths": ctx_lengths, "results": results,
        })

    del model
    torch.cuda.empty_cache()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: KEY/VALUE EIGENVALUE ASYMMETRY (all models)
# ══════════════════════════════════════════════════════════════════════════════

ALL_MODELS_FOR_ASYMMETRY = [
    ("Qwen/Qwen2.5-1.5B-Instruct", False),
    ("Qwen/Qwen2.5-7B-Instruct",   False),
    ("Qwen/Qwen2.5-14B-Instruct",  False),
    ("meta-llama/Llama-3.1-8B-Instruct", True),
    ("mistralai/Mistral-7B-Instruct-v0.3", False),
    ("google/gemma-2-9b-it",       True),
]


def run_part4_kv_asymmetry(device, n_calib, partial_results=None) -> dict:
    """
    For each model, calibrate keys and values separately.
    Report d_eff_keys vs d_eff_values, per layer, and save full eigenvalue spectra
    for representative heads.
    """
    results = partial_results or {}

    for model_name, needs_hf_token in ALL_MODELS_FOR_ASYMMETRY:
        short_name = model_name.split("/")[-1]
        if short_name in results:
            log.info("Skipping %s (done)", short_name)
            continue

        log.info("\n" + "=" * 60)
        log.info("KV Asymmetry: %s", model_name)
        log.info("=" * 60)

        try:
            model, tokenizer, n_layers, n_kv, hd = load_model_tokenizer(model_name, device)
        except Exception as e:
            log.error("Failed to load %s: %s", model_name, e)
            results[short_name] = {"error": str(e)}
            save_result("neurips_kv_asymmetry.json", results)
            continue

        try:
            eigen_keys, eigen_vals = calibrate_keys_and_values(
                model, tokenizer, n_calib, device, n_layers, n_kv, hd
            )
        except Exception as e:
            log.error("Calibration failed for %s: %s", model_name, e)
            results[short_name] = {"error": str(e)}
            del model
            torch.cuda.empty_cache()
            save_result("neurips_kv_asymmetry.json", results)
            continue

        # Per-layer mean d_eff
        d_eff_keys_per_layer = [
            float(np.mean([eigen_keys[(l, h)]["d_eff"] for h in range(n_kv)]))
            for l in range(n_layers)
        ]
        d_eff_vals_per_layer = [
            float(np.mean([eigen_vals[(l, h)]["d_eff"] for h in range(n_kv)]))
            for l in range(n_layers)
        ]

        mean_d_eff_keys = float(np.mean([eigen_keys[(l, h)]["d_eff"]
                                          for l in range(n_layers) for h in range(n_kv)]))
        mean_d_eff_vals = float(np.mean([eigen_vals[(l, h)]["d_eff"]
                                          for l in range(n_layers) for h in range(n_kv)]))

        # Save representative eigenvalue spectra:
        # For 3 representative layers (early, middle, late), head 0
        rep_layers = [0, n_layers // 2, n_layers - 1]
        rep_spectra = {}
        for l in rep_layers:
            h = 0  # representative head
            ev_k = eigen_keys[(l, h)]["ev"].tolist()
            ev_v = eigen_vals[(l, h)]["ev"].tolist()
            # Truncate to first 128 values for large head_dims
            rep_spectra[f"layer{l}_head{h}"] = {
                "key_eigenvalues": ev_k[:128],
                "val_eigenvalues": ev_v[:128],
                "d_eff_keys": eigen_keys[(l, h)]["d_eff"],
                "d_eff_vals": eigen_vals[(l, h)]["d_eff"],
            }

        model_result = {
            "model": model_name,
            "n_layers": n_layers,
            "n_kv_heads": n_kv,
            "head_dim": hd,
            "mean_d_eff_keys": mean_d_eff_keys,
            "mean_d_eff_vals": mean_d_eff_vals,
            "d_eff_keys_per_layer": d_eff_keys_per_layer,
            "d_eff_vals_per_layer": d_eff_vals_per_layer,
            "d_eff_ratio_keys_over_vals": float(mean_d_eff_keys / mean_d_eff_vals)
                if mean_d_eff_vals > 0 else None,
            "representative_spectra": rep_spectra,
        }

        log.info("  %s: d_eff_keys=%.1f  d_eff_vals=%.1f  ratio=%.2f",
                 short_name, mean_d_eff_keys, mean_d_eff_vals,
                 mean_d_eff_keys / mean_d_eff_vals if mean_d_eff_vals > 0 else float("nan"))

        results[short_name] = model_result

        del model
        torch.cuda.empty_cache()

        # Save after each model (crash-safe)
        save_result("neurips_kv_asymmetry.json", results)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NeurIPS Script B: New Models + KV Eigenvalue Asymmetry"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1/5 sample sizes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--part", type=int, default=0,
                        help="0=all, 1=mistral, 2=gemma, 3=qwen7b_ppl, 4=kv_asymmetry")
    parser.add_argument("--n-calib", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    n_calib = args.n_calib or (40 if args.quick else 200)
    n_eval = 20 if args.quick else 100

    ppl_ctx_lengths = [1024, 2048] if args.quick else [1024, 2048, 4096, 8192]

    log.info("=" * 70)
    log.info("NeurIPS Script B: Models + KV Asymmetry  |  quick=%s  part=%d",
             args.quick, args.part)
    log.info("=" * 70)

    # ─── PART 1: Mistral 7B ─────────────────────────────────────────────────
    if args.part in (0, 1):
        log.info("\n" + "=" * 60)
        log.info("PART 1: Mistral 7B")
        log.info("=" * 60)
        result = run_part1_mistral(device, n_calib, n_eval)
        save_result("neurips_mistral.json", result)

    # ─── PART 2: Gemma 9B ───────────────────────────────────────────────────
    if args.part in (0, 2):
        log.info("\n" + "=" * 60)
        log.info("PART 2: Gemma 2-9B (head_dim=256 test)")
        log.info("=" * 60)
        result = run_part2_gemma(device, n_calib, n_eval)
        save_result("neurips_gemma.json", result)

    # ─── PART 3: Qwen 7B PPL ────────────────────────────────────────────────
    if args.part in (0, 3):
        log.info("\n" + "=" * 60)
        log.info("PART 3: Qwen 2.5-7B Perplexity  ctx=%s", ppl_ctx_lengths)
        log.info("=" * 60)
        ppl_resume = {}
        ppl_path = RESULTS_DIR / "neurips_qwen7b_ppl.json"
        if ppl_path.exists():
            try:
                with open(ppl_path) as f:
                    existing = json.load(f)
                ppl_resume = existing.get("results", {})
                log.info("Resuming Qwen PPL from %d done ctx_lengths", len(ppl_resume))
            except Exception:
                pass
        run_part3_qwen7b_ppl(device, n_calib, ppl_ctx_lengths, partial_results=ppl_resume)

    # ─── PART 4: KV Asymmetry ───────────────────────────────────────────────
    if args.part in (0, 4):
        log.info("\n" + "=" * 60)
        log.info("PART 4: KV Eigenvalue Asymmetry (all models)")
        log.info("=" * 60)
        asym_resume = {}
        asym_path = RESULTS_DIR / "neurips_kv_asymmetry.json"
        if asym_path.exists():
            try:
                with open(asym_path) as f:
                    asym_resume = json.load(f)
                log.info("Resuming KV asymmetry from %d done models", len(asym_resume))
            except Exception:
                pass
        run_part4_kv_asymmetry(device, n_calib, partial_results=asym_resume)

    log.info("\nAll done. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
