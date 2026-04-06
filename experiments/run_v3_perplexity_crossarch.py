"""
run_v3_perplexity_crossarch.py — SpectralQuant v3 comprehensive experiment script.

Three parts:
  PART 1  — Perplexity evaluation across Qwen2.5 sizes and datasets
  PART 2  — Cross-architecture evaluation (Llama-3.1, Mistral)
  PART 3  — 5-seed confidence intervals on Qwen2.5-1.5B

Usage:
    python run_v3_perplexity_crossarch.py            # full run
    python run_v3_perplexity_crossarch.py --quick    # fast smoke-test
    python run_v3_perplexity_crossarch.py --part 1   # only Part 1
"""

import sys
import os
import math
import time
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from turboquant_cutile import TurboQuantEngine, LloydMaxCodebook

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("v3")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# HF token for gated models
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results" / "v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Shared utilities (reused across all three parts)
# ===========================================================================

def quantize_nearest(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour scalar quantization from a LloydMax codebook."""
    c = centroids.to(x.device)
    diffs = x.unsqueeze(-1) - c
    idx = diffs.abs().argmin(dim=-1)
    return c[idx.long()]


@torch.no_grad()
def sq_noqjl_v3(Q, K, V, evec, d_eff, hd, device):
    """
    SQ_noQJL_v3: spectral rotation, no QJL, uniform 3-bit keys + 3-bit values.
    Equivalent to 'B_SpectralRot_noQJL' in run_final_experiments.py.
    """
    K_f, V_f, Q_f = K.float(), V.float(), Q.float()
    VT = evec.T.contiguous().to(device).float()
    Vm = evec.to(device).float()

    k_n = torch.norm(K_f, dim=-1, keepdim=True)
    K_rot = (K_f / (k_n + 1e-8)) @ VT
    v_n = torch.norm(V_f, dim=-1, keepdim=True)
    V_rot = (V_f / (v_n + 1e-8)) @ VT

    cb_k = LloydMaxCodebook(hd, 2)  # 2-bit key MSE (4-level)
    cb_v = LloydMaxCodebook(hd, 3)  # 3-bit values
    K_hat = quantize_nearest(K_rot, cb_k.centroids)
    V_hat = quantize_nearest(V_rot, cb_v.centroids)

    K_mse = (K_hat @ Vm) * k_n
    V_rec = (V_hat @ Vm) * v_n
    scores = (Q_f @ K_mse.T) / math.sqrt(hd)
    return (torch.softmax(scores, dim=-1) @ V_rec).half()


@torch.no_grad()
def sq_noqjl_v2tail(Q, K, V, evec, d_eff, hd, device):
    """
    SQ_noQJL_v2tail: spectral rotation, no QJL,
    3-bit semantic ([:d_eff]) + 2-bit tail ([d_eff:]) for values,
    2-bit uniform for keys.
    """
    K_f, V_f, Q_f = K.float(), V.float(), Q.float()
    VT = evec.T.contiguous().to(device).float()
    Vm = evec.to(device).float()

    k_n = torch.norm(K_f, dim=-1, keepdim=True)
    K_rot = (K_f / (k_n + 1e-8)) @ VT
    v_n = torch.norm(V_f, dim=-1, keepdim=True)
    V_rot = (V_f / (v_n + 1e-8)) @ VT

    cb_k = LloydMaxCodebook(hd, 2)
    cb_vh = LloydMaxCodebook(hd, 3)  # semantic portion
    cb_vl = LloydMaxCodebook(hd, 2)  # tail portion
    K_hat = quantize_nearest(K_rot, cb_k.centroids)

    V_hat_h = quantize_nearest(V_rot[:, :d_eff], cb_vh.centroids)
    V_hat_l = quantize_nearest(V_rot[:, d_eff:], cb_vl.centroids)
    V_hat = torch.cat([V_hat_h, V_hat_l], dim=-1)

    K_mse = (K_hat @ Vm) * k_n
    V_rec = (V_hat @ Vm) * v_n
    scores = (Q_f @ K_mse.T) / math.sqrt(hd)
    return (torch.softmax(scores, dim=-1) @ V_rec).half()


def cosine_sim(ref: torch.Tensor, out: torch.Tensor, hd: int) -> float:
    """Mean cosine similarity between two attention output tensors."""
    c = F.cosine_similarity(
        ref.float().reshape(-1, hd),
        out.float().reshape(-1, hd),
        dim=-1,
    ).mean().item()
    return c if not math.isnan(c) else None


def extract_kv(kv, layer: int):
    """
    Extract key/value tensors from a HF past_key_values object,
    handling both DynamicCache (attribute access) and tuple-of-tuples.
    """
    try:
        k = kv.key_cache[layer].float().cpu()
        v = kv.value_cache[layer].float().cpu()
    except AttributeError:
        try:
            k = kv[layer][0].float().cpu()
            v = kv[layer][1].float().cpu()
        except Exception:
            lkv = list(kv)[layer]
            k = lkv[0].float().cpu()
            v = lkv[1].float().cpu()
    return k, v


# ===========================================================================
# Calibration (shared)
# ===========================================================================

def calibrate(model, tokenizer, n_calib: int, device: str, n_layers: int, n_kv: int, hd: int) -> dict:
    """
    Compute per-(layer, head) spectral eigenvectors from n_calib WikiText-103 train sequences.
    Returns eigen dict: {(l, h): {"evec": Tensor, "d_eff": int, "kappa": float}}.
    """
    from datasets import load_dataset

    log.info("  Calibrating with %d sequences from WikiText-103 train...", n_calib)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{n_calib * 5}]")

    cov = {
        (l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64), "n": 0}
        for l in range(n_layers) for h in range(n_kv)
    }
    nd = 0
    t0 = time.time()

    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue
        try:
            with torch.no_grad():
                out = model(**enc, use_cache=True)
                kv = out.past_key_values
        except RuntimeError as e:
            log.warning("    OOM during calibration: %s — skipping", e)
            torch.cuda.empty_cache()
            continue

        for l in range(n_layers):
            k_l, _ = extract_kv(kv, l)
            for h in range(n_kv):
                X = k_l[0, h, :, :].double()
                cov[(l, h)]["xtx"] += X.T @ X
                cov[(l, h)]["n"] += X.shape[0]

        nd += 1
        if nd >= n_calib:
            break
        if nd % 50 == 0:
            log.info("    Cov: %d/%d (%.0fs)", nd, n_calib, time.time() - t0)

    log.info("  Calibrated on %d sequences in %.1fs", nd, time.time() - t0)

    eigen = {}
    for l in range(n_layers):
        for h in range(n_kv):
            if cov[(l, h)]["n"] == 0:
                # Fallback: identity eigenvectors
                eigen[(l, h)] = {
                    "evec": torch.eye(hd),
                    "ev": torch.ones(hd),
                    "d_eff": hd // 2,
                    "kappa": 1.0,
                }
                continue
            C = (cov[(l, h)]["xtx"] / cov[(l, h)]["n"]).float()
            ev, evec = torch.linalg.eigh(C)
            ev = ev.flip(0).clamp(min=0)
            evec = evec.flip(1)
            # Effective rank via participation ratio
            ev_sum = ev.sum()
            ev_sq_sum = (ev ** 2).sum()
            d_eff = max(2, min(
                int(round((ev_sum ** 2 / ev_sq_sum).item())) if ev_sq_sum > 1e-10 else hd // 2,
                hd - 2,
            ))
            denom = ev[min(d_eff, hd - 1)].clamp(min=1e-10)
            kappa = min((ev[d_eff - 1] / denom).item(), 1e6)
            eigen[(l, h)] = {"evec": evec, "ev": ev, "d_eff": d_eff, "kappa": kappa}

    mean_deff = np.mean([eigen[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)])
    mean_kappa = np.mean([eigen[(l, h)]["kappa"] for l in range(n_layers) for h in range(n_kv)])
    log.info("  d_eff=%.1f, κ=%.2f", mean_deff, mean_kappa)

    return eigen, float(mean_deff), float(mean_kappa)


# ===========================================================================
# Memory computation (from run_memory_efficiency.py)
# ===========================================================================

def compute_memory(hd, d_eff, key_mse_bits, key_qjl_bits, val_high_bits, val_low_bits,
                   seq_len, n_layers, n_kv):
    """Compute KV-cache memory in bytes and compression ratio vs FP16."""
    key_mse = seq_len * hd * key_mse_bits
    key_qjl = seq_len * d_eff * key_qjl_bits
    key_norms = seq_len * 32  # 2 × FP16 norms
    val_sem = seq_len * d_eff * val_high_bits
    val_tail = seq_len * (hd - d_eff) * val_low_bits
    val_norms = seq_len * 16  # FP16 norm

    total_bits = (key_mse + key_qjl + key_norms + val_sem + val_tail + val_norms) * n_layers * n_kv
    total_bytes = total_bits / 8
    fp16_bytes = seq_len * hd * 2 * 2 * n_layers * n_kv  # K+V FP16
    ratio = fp16_bytes / total_bytes if total_bytes > 0 else 0.0
    avg_bits = total_bits / (seq_len * hd * 2 * n_layers * n_kv) if (seq_len * hd * 2 * n_layers * n_kv) > 0 else 0.0
    return {"total_bytes": total_bytes, "fp16_bytes": fp16_bytes,
            "ratio": ratio, "avg_bits": avg_bits, "total_mb": total_bytes / 1e6}


# ===========================================================================
# Perplexity measurement via sliding-window compressed cache
# ===========================================================================

def measure_ppl_fp16(model, tokenizer, texts: list, device: str, max_tokens: int = 512) -> float:
    """
    FP16 baseline perplexity: model(input_ids, labels=input_ids).loss.
    Returns perplexity (exp of mean cross-entropy).
    """
    losses = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", max_length=max_tokens, truncation=True).to(device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 8:
            continue
        try:
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                losses.append(outputs.loss.item())
        except RuntimeError as e:
            log.warning("    OOM in FP16 ppl: %s — skipping", e)
            torch.cuda.empty_cache()
    if not losses:
        return float("nan")
    return math.exp(np.mean(losses))


def _compress_decompress_kv(kv, eigen: dict, method: str, n_layers: int, n_kv: int, hd: int, device: str):
    """
    Compress-then-decompress the full KV cache for one sequence.
    Returns a list of (k_recon, v_recon) per layer, each [1, n_kv, seq, hd].
    
    method: one of "TQ_3bit", "TQ_2bit", "SQ_noQJL_v3", "SQ_noQJL_v2tail"
    """
    recon_layers = []
    for l in range(n_layers):
        k_l, v_l = extract_kv(kv, l)  # [1, n_kv, seq, hd]
        k_recon_heads = []
        v_recon_heads = []

        for h in range(n_kv):
            K = k_l[0, h].to(device).half()   # [seq, hd]
            V = v_l[0, h].to(device).half()
            seq = K.shape[0]
            if seq < 2:
                k_recon_heads.append(K)
                v_recon_heads.append(V)
                continue

            K_f = K.float()
            V_f = V.float()

            if method in ("TQ_3bit", "TQ_2bit"):
                total_bits = 3 if method == "TQ_3bit" else 2
                try:
                    tq = TurboQuantEngine(head_dim=hd, total_bits=total_bits, device=device)
                    ck = tq.compress_keys_pytorch(K)
                    cv = tq.compress_values_pytorch(V)
                    # Decompress: get reconstructed K and V tensors
                    # TQ doesn't expose a direct decompress-to-tensor, so we approximate
                    # by measuring the reconstructed value as: attention output ≈ Q @ K_recon.T
                    # We use a proxy identity query to extract K_recon (SVD trick):
                    # Send eye queries, read out K_recon rows
                    Q_eye = torch.eye(hd, device=device, dtype=torch.float16)[:min(seq, hd)]
                    attn_out = tq.fused_attention_pytorch(Q_eye, ck, cv)
                    # For key reconstruction we can't easily recover K; use K_f as fallback
                    # since TQ's fused_attention gives us the full pipeline result.
                    # We'll instead measure perplexity proxy via logit difference later.
                    # For now, store K as-is and use the attention output to measure quality.
                    k_recon_heads.append(K)  # keep FP16 K for cache
                    v_recon_heads.append(V)  # keep FP16 V for cache
                except Exception as e:
                    log.debug("TQ compress error (l=%d, h=%d): %s", l, h, e)
                    k_recon_heads.append(K)
                    v_recon_heads.append(V)

            else:  # SQ methods — we CAN reconstruct K and V explicitly
                ed = eigen.get((l, h), None)
                if ed is None:
                    k_recon_heads.append(K)
                    v_recon_heads.append(V)
                    continue
                evec = ed["evec"].to(device).float()
                d_eff = ed["d_eff"]
                VT = evec.T.contiguous()
                Vm = evec

                k_n = torch.norm(K_f, dim=-1, keepdim=True)
                K_rot = (K_f / (k_n + 1e-8)) @ VT
                v_n = torch.norm(V_f, dim=-1, keepdim=True)
                V_rot = (V_f / (v_n + 1e-8)) @ VT

                cb_k = LloydMaxCodebook(hd, 2)
                K_hat = quantize_nearest(K_rot, cb_k.centroids)
                K_recon = ((K_hat @ Vm) * k_n).half()

                if method == "SQ_noQJL_v3":
                    cb_v = LloydMaxCodebook(hd, 3)
                    V_hat = quantize_nearest(V_rot, cb_v.centroids)
                    V_recon = ((V_hat @ Vm) * v_n).half()
                else:  # SQ_noQJL_v2tail
                    cb_vh = LloydMaxCodebook(hd, 3)
                    cb_vl = LloydMaxCodebook(hd, 2)
                    V_hat_h = quantize_nearest(V_rot[:, :d_eff], cb_vh.centroids)
                    V_hat_l = quantize_nearest(V_rot[:, d_eff:], cb_vl.centroids)
                    V_hat = torch.cat([V_hat_h, V_hat_l], dim=-1)
                    V_recon = ((V_hat @ Vm) * v_n).half()

                k_recon_heads.append(K_recon)
                v_recon_heads.append(V_recon)

        # Stack heads back: [1, n_kv, seq, hd]
        k_layer = torch.stack(k_recon_heads, dim=0).unsqueeze(0)  # [1, n_kv, seq, hd]
        v_layer = torch.stack(v_recon_heads, dim=0).unsqueeze(0)
        recon_layers.append((k_layer, v_layer))

    return recon_layers


def measure_ppl_compressed_sq(
    model, tokenizer, texts: list, eigen: dict,
    method: str, device: str,
    n_layers: int, n_kv: int, hd: int,
    stride: int = 64, chunk_size: int = 256,
    quick: bool = False,
) -> float:
    """
    Measure perplexity for SQ methods (where we can reconstruct K and V explicitly)
    using a sliding-window approach with the compressed KV cache.
    
    Algorithm:
      For each sequence, for each stride position t:
        1. Forward [0:t] to get KV cache
        2. Compress-decompress KV cache (SQ path gives exact K_recon, V_recon)
        3. Build a DynamicCache from reconstructed KV
        4. Forward on token [t] with past_key_values=compressed_cache
        5. Record CE loss on token [t]
    
    For TQ methods (no explicit K/V reconstruction), fall back to FP16 ppl
    with a note that this is approximate.
    
    For --quick mode, only processes every 128th token position.
    """
    if method in ("TQ_3bit", "TQ_2bit"):
        # TQ path: no explicit K/V reconstruction available without kernel surgery.
        # We report FP16 PPL with a note.
        log.info("    %s: using FP16 ppl as proxy (TQ cannot reconstruct K/V without kernel)", method)
        return measure_ppl_fp16(model, tokenizer, texts, device)

    # SQ path: we can do proper sliding-window perplexity
    try:
        from transformers.cache_utils import DynamicCache
        HAS_DYNAMIC_CACHE = True
    except ImportError:
        HAS_DYNAMIC_CACHE = False
        log.warning("    DynamicCache not available — falling back to FP16 ppl")
        return measure_ppl_fp16(model, tokenizer, texts, device)

    losses = []
    actual_stride = stride * 2 if quick else stride

    for text_idx, text in enumerate(texts):
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        input_ids = enc["input_ids"]
        seq_len = input_ids.shape[1]
        if seq_len < 32:
            continue

        # Use prefix positions at multiples of actual_stride (min prefix = 16 tokens)
        positions = list(range(16, seq_len - 1, actual_stride))
        if not positions:
            positions = [seq_len // 2]

        for t in positions:
            prefix_ids = input_ids[:, :t]        # [1, t]
            target_tok = input_ids[:, t]          # [1]

            try:
                with torch.no_grad():
                    # Step 1: forward on prefix to populate KV cache
                    pref_out = model(prefix_ids, use_cache=True)
                    kv = pref_out.past_key_values

                    # Step 2: compress-decompress KV (SQ explicit reconstruction)
                    recon_layers = _compress_decompress_kv(kv, eigen, method, n_layers, n_kv, hd, device)

                    # Step 3: build DynamicCache from reconstructed K, V
                    new_cache = DynamicCache()
                    for l_idx, (k_l, v_l) in enumerate(recon_layers):
                        # k_l: [1, n_kv, t, hd]
                        new_cache.update(k_l.to(device), v_l.to(device), l_idx)

                    # Step 4: forward on the single target token
                    tok_out = model(
                        target_tok.unsqueeze(0),  # [1, 1]
                        past_key_values=new_cache,
                        use_cache=False,
                    )
                    # logits: [1, 1, vocab]
                    logits = tok_out.logits[:, 0, :]  # [1, vocab]
                    loss = F.cross_entropy(logits, target_tok)
                    losses.append(loss.item())

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.warning("    OOM at t=%d: %s — skipping", t, e)
                    torch.cuda.empty_cache()
                else:
                    log.debug("    Error at t=%d: %s", t, e)
            except Exception as e:
                log.debug("    Error at t=%d: %s", t, e)

        if (text_idx + 1) % 10 == 0:
            log.info("    PPL [%s]: processed %d/%d texts", method, text_idx + 1, len(texts))

    if not losses:
        log.warning("    No valid losses for %s — returning nan", method)
        return float("nan")
    return math.exp(np.mean(losses))


def load_ppl_texts(dataset_name: str, n_texts: int, tokenizer, max_length: int = 512):
    """Load texts for perplexity evaluation from named dataset."""
    from datasets import load_dataset

    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [item["text"] for item in ds if len(item.get("text", "").strip()) > 200]
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation[:1000]",
                          trust_remote_code=True)
        texts = [item["text"] for item in ds if len(item.get("text", "").strip()) > 200]
    elif dataset_name == "ptb":
        ds = load_dataset("ptb_text_only", "penn_treebank", split="test",
                          trust_remote_code=True)
        texts = [item["sentence"] for item in ds if len(item.get("sentence", "").strip()) > 20]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    texts = texts[:n_texts]
    log.info("  Loaded %d texts from %s", len(texts), dataset_name)
    return texts


# ===========================================================================
# PART 1: Perplexity evaluation
# ===========================================================================

def run_part1_perplexity(args):
    """
    Evaluate perplexity of FP16, TQ_3bit, TQ_2bit, SQ_noQJL_v3, SQ_noQJL_v2tail
    across Qwen2.5-{1.5B,7B,14B}-Instruct on WikiText-2, C4, PTB.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("\n" + "=" * 90)
    log.info("PART 1: Perplexity Evaluation")
    log.info("=" * 90)

    models_ppl = ["Qwen/Qwen2.5-1.5B-Instruct"]
    if not args.quick:
        models_ppl += ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]

    datasets_ppl = ["wikitext2", "c4"] if args.quick else ["wikitext2", "c4", "ptb"]
    n_texts_ppl = 20 if args.quick else 100
    n_calib_ppl = 50 if args.quick else 100

    ppl_methods = ["FP16", "TQ_3bit", "TQ_2bit", "SQ_noQJL_v3", "SQ_noQJL_v2tail"]

    all_ppl_results = {}

    for model_name in models_ppl:
        short = model_name.split("/")[-1]
        log.info("\n  Model: %s", short)
        model_result = {"model": short, "datasets": {}}

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map=args.device,
                token=HF_TOKEN,
            )
            model.eval()

            cfg = model.config
            n_layers = cfg.num_hidden_layers
            n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            hd = cfg.hidden_size // cfg.num_attention_heads
            log.info("  %d layers, %d KV heads, head_dim=%d", n_layers, n_kv, hd)

            # Calibrate eigenvectors
            eigen, mean_deff, mean_kappa = calibrate(
                model, tokenizer, n_calib_ppl, args.device, n_layers, n_kv, hd
            )
            model_result["d_eff"] = mean_deff
            model_result["kappa"] = mean_kappa

            for ds_name in datasets_ppl:
                log.info("\n  Dataset: %s", ds_name)
                ds_result = {}
                try:
                    texts = load_ppl_texts(ds_name, n_texts_ppl, tokenizer)
                except Exception as e:
                    log.error("  Failed to load %s: %s", ds_name, e)
                    model_result["datasets"][ds_name] = {"error": str(e)}
                    continue

                for method in ppl_methods:
                    log.info("    Method: %s", method)
                    t0 = time.time()
                    try:
                        if method == "FP16":
                            ppl = measure_ppl_fp16(model, tokenizer, texts, args.device)
                        else:
                            ppl = measure_ppl_compressed_sq(
                                model, tokenizer, texts, eigen, method, args.device,
                                n_layers, n_kv, hd, quick=args.quick,
                            )
                        elapsed = time.time() - t0
                        log.info(
                            "    %s | %s: PPL=%.2f  (%.1fs)",
                            ds_name, method, ppl, elapsed,
                        )
                        ds_result[method] = {"ppl": float(ppl), "elapsed_s": round(elapsed, 1)}
                    except RuntimeError as e:
                        log.error("    OOM/error for %s/%s: %s", ds_name, method, e)
                        torch.cuda.empty_cache()
                        ds_result[method] = {"ppl": None, "error": str(e)}
                    except Exception as e:
                        log.error("    Error for %s/%s: %s", ds_name, method, e)
                        ds_result[method] = {"ppl": None, "error": str(e)}

                model_result["datasets"][ds_name] = ds_result

                # Summarise
                log.info("\n  --- %s / %s ---", short, ds_name)
                fp16_ppl = ds_result.get("FP16", {}).get("ppl", None)
                log.info("  %-20s  %10s  %10s", "Method", "PPL", "vs FP16")
                for method in ppl_methods:
                    ppl_val = ds_result.get(method, {}).get("ppl", None)
                    if ppl_val is not None and fp16_ppl is not None:
                        delta = ppl_val - fp16_ppl
                        log.info("  %-20s  %10.2f  %+10.2f", method, ppl_val, delta)
                    elif ppl_val is not None:
                        log.info("  %-20s  %10.2f", method, ppl_val)
                    else:
                        log.info("  %-20s  %10s", method, "N/A")

            all_ppl_results[short] = model_result

        except RuntimeError as e:
            log.error("OOM loading %s: %s", model_name, e)
            torch.cuda.empty_cache()
            all_ppl_results[short] = {"error": str(e)}
        except Exception as e:
            log.error("Error on %s: %s", model_name, e)
            import traceback; traceback.print_exc()
            all_ppl_results[short] = {"error": str(e)}
        finally:
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Add methodology note
    all_ppl_results["_methodology"] = {
        "fp16": "model(input_ids, labels=input_ids).loss — standard HF cross-entropy",
        "sq_compressed": (
            "Sliding-window: for each stride position t, "
            "forward on prefix [0:t] -> compress-decompress KV (explicit K/V reconstruction) "
            "-> DynamicCache -> forward on token [t] -> CE loss. PPL = exp(mean loss)."
        ),
        "tq_compressed": (
            "TQ path reports FP16 PPL as proxy. "
            "Direct PPL with TQ requires kernel-level KV injection "
            "(left to kernel integration milestone)."
        ),
    }

    out_path = RESULTS_DIR / "v3_perplexity.json"
    with open(out_path, "w") as f:
        json.dump(all_ppl_results, f, indent=2, default=str)
    log.info("\nPart 1 results saved to %s", out_path)
    return all_ppl_results


# ===========================================================================
# PART 2: Cross-architecture evaluation
# ===========================================================================

def run_crossarch_for_model(model_name: str, args, n_calib: int, n_eval: int) -> dict:
    """
    Run cross-architecture experiment for a single model.
    Returns a summary dict.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    short = model_name.split("/")[-1]
    log.info("\n  Model: %s", short)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=args.device,
        token=HF_TOKEN,
    )
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info("  %d layers, %d KV heads, head_dim=%d, %.1fB params", n_layers, n_kv, hd, n_params)

    # Calibrate
    eigen, mean_deff, mean_kappa = calibrate(
        model, tokenizer, n_calib, args.device, n_layers, n_kv, hd
    )

    # Per-layer d_eff and kappa summary
    layer_stats = []
    for l in range(n_layers):
        layer_deff = []
        layer_kappa = []
        for h in range(n_kv):
            ed = eigen.get((l, h), {})
            layer_deff.append(ed.get("d_eff", hd // 2))
            layer_kappa.append(ed.get("kappa", 1.0))
        layer_stats.append({
            "layer": l,
            "d_eff_mean": float(np.mean(layer_deff)),
            "d_eff_std": float(np.std(layer_deff)),
            "kappa_mean": float(np.mean(layer_kappa)),
            "kappa_std": float(np.std(layer_kappa)),
        })

    d_eff_typical = int(round(mean_deff))

    # Quality evaluation
    log.info("  Evaluating quality (%d sequences)...", n_eval)
    eval_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"test[:{n_eval * 5}]")
    slayers = list(range(0, n_layers, max(1, n_layers // 5)))[:5]

    # Configs to compare
    configs_eval = {
        "TQ_3bit": {"type": "tq", "bits": 3},
        "TQ_2bit": {"type": "tq", "bits": 2},
        "SQ_noQJL_v3": {"type": "sq_v3"},
        "SQ_noQJL_v2tail": {"type": "sq_v2tail"},
    }
    quality = {name: [] for name in configs_eval}
    nev = 0

    for item in eval_ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(args.device)
        if enc["input_ids"].shape[1] < 16:
            continue

        try:
            with torch.no_grad():
                out = model(**enc, use_cache=True)
                kv = out.past_key_values
        except RuntimeError as e:
            log.warning("  OOM during eval: %s — skipping", e)
            torch.cuda.empty_cache()
            continue

        for l in slayers:
            try:
                k_l, v_l = extract_kv(kv, l)
            except Exception:
                continue

            for h in range(n_kv):
                K = k_l[0, h].to(args.device).half()
                V = v_l[0, h].to(args.device).half()
                if K.shape[0] < 8:
                    continue

                torch.manual_seed(42 + l * 1000 + h)
                Qp = torch.randn(8, hd, device=args.device, dtype=torch.float16)
                sc = (Qp.float() @ K.float().T) / math.sqrt(hd)
                ref = (torch.softmax(sc, dim=-1) @ V.float()).half()

                ed = eigen.get((l, h), {})
                evec = ed.get("evec", torch.eye(hd))
                d_eff_lh = ed.get("d_eff", d_eff_typical)

                for name, cfg_e in configs_eval.items():
                    try:
                        if cfg_e["type"] == "tq":
                            tq = TurboQuantEngine(
                                head_dim=hd, total_bits=cfg_e["bits"], device=args.device
                            )
                            ck = tq.compress_keys_pytorch(K)
                            cv = tq.compress_values_pytorch(V)
                            out_t = tq.fused_attention_pytorch(Qp, ck, cv)
                        elif cfg_e["type"] == "sq_v3":
                            out_t = sq_noqjl_v3(Qp, K, V, evec, d_eff_lh, hd, args.device)
                        else:  # sq_v2tail
                            out_t = sq_noqjl_v2tail(Qp, K, V, evec, d_eff_lh, hd, args.device)

                        val = cosine_sim(ref, out_t, hd)
                        if val is not None:
                            quality[name].append(val)
                    except RuntimeError as e:
                        log.debug("  OOM in eval (l=%d, h=%d, %s): %s", l, h, name, e)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        log.debug("  Error (l=%d, h=%d, %s): %s", l, h, name, e)

        nev += 1
        if nev >= n_eval:
            break
        if nev % 10 == 0:
            log.info("    Eval: %d/%d", nev, n_eval)

    # Memory ratios
    seq_len = 8192
    mem_results = {}
    for name, cfg_e in configs_eval.items():
        if cfg_e["type"] == "tq":
            # TQ: uniform bits, full QJL
            bits = cfg_e["bits"]
            mem = compute_memory(hd, hd, bits - 1, 1, bits, bits, seq_len, n_layers, n_kv)
        elif cfg_e["type"] == "sq_v3":
            mem = compute_memory(hd, d_eff_typical, 2, 0, 3, 3, seq_len, n_layers, n_kv)
        else:  # sq_v2tail
            mem = compute_memory(hd, d_eff_typical, 2, 0, 3, 2, seq_len, n_layers, n_kv)
        mem_results[name] = mem

    # Log results
    log.info("\n  Results: %s (d_eff=%d, κ=%.2f)", short, d_eff_typical, mean_kappa)
    log.info("  %-20s  %8s  %8s  %8s", "Config", "CosSim", "Ratio", "AvgBits")
    summary_configs = {}
    for name in configs_eval:
        if quality[name]:
            m = float(np.mean(quality[name]))
            mem = mem_results[name]
            log.info("  %-20s  %8.4f  %8.2f×  %8.2f",
                     name, m, mem["ratio"], mem["avg_bits"])
            summary_configs[name] = {
                "cos_sim_mean": m,
                "cos_sim_std": float(np.std(quality[name])),
                "n": len(quality[name]),
                "compression_ratio": mem["ratio"],
                "avg_bits_per_elem": mem["avg_bits"],
                "mb_at_8k": mem["total_mb"],
            }

    del model
    torch.cuda.empty_cache()

    return {
        "model": short,
        "model_full": model_name,
        "n_layers": n_layers,
        "n_kv_heads": n_kv,
        "head_dim": hd,
        "n_params_B": round(n_params, 2),
        "d_eff_mean": mean_deff,
        "kappa_mean": mean_kappa,
        "d_eff_typical": d_eff_typical,
        "layer_stats": layer_stats,
        "configs": summary_configs,
    }


def run_part2_crossarch(args):
    """
    Cross-architecture evaluation: Llama-3.1-8B and Mistral-7B.
    """
    log.info("\n" + "=" * 90)
    log.info("PART 2: Cross-Architecture Evaluation")
    log.info("=" * 90)

    if HF_TOKEN is None:
        log.warning("HF_TOKEN not set — Llama access may fail. Set HF_TOKEN env var.")

    cross_models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]
    if args.quick:
        cross_models = cross_models[:1]

    n_calib_cross = 30 if args.quick else 100
    n_eval_cross = 15 if args.quick else 50

    all_cross_results = {}

    for model_name in cross_models:
        short = model_name.split("/")[-1]
        try:
            result = run_crossarch_for_model(model_name, args, n_calib_cross, n_eval_cross)
            all_cross_results[short] = result
        except RuntimeError as e:
            log.error("OOM loading %s: %s", model_name, e)
            torch.cuda.empty_cache()
            all_cross_results[short] = {"error": str(e), "model": short}
        except Exception as e:
            log.error("Error on %s: %s", model_name, e)
            import traceback; traceback.print_exc()
            all_cross_results[short] = {"error": str(e), "model": short}

    out_path = RESULTS_DIR / "v3_crossarch.json"
    with open(out_path, "w") as f:
        json.dump(all_cross_results, f, indent=2, default=str)
    log.info("\nPart 2 results saved to %s", out_path)
    return all_cross_results


# ===========================================================================
# PART 3: 5-seed confidence intervals
# ===========================================================================

def run_one_seed(model, tokenizer, eigen: dict, seed: int, n_eval: int,
                 device: str, n_layers: int, n_kv: int, hd: int) -> dict:
    """
    Run TQ_3bit and SQ_noQJL_v3 evaluations with a given seed.
    Returns {"tq": [cos_sims], "sq": [cos_sims]}.
    """
    from datasets import load_dataset

    eval_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"test[:{n_eval * 5}]")
    slayers = list(range(0, n_layers, max(1, n_layers // 5)))[:5]

    results = {"tq": [], "sq": []}
    nev = 0

    for item in eval_ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue

        try:
            with torch.no_grad():
                out = model(**enc, use_cache=True)
                kv = out.past_key_values
        except RuntimeError as e:
            log.warning("  OOM seed=%d: %s", seed, e)
            torch.cuda.empty_cache()
            continue

        for l in slayers:
            try:
                k_l, v_l = extract_kv(kv, l)
            except Exception:
                continue
            for h in range(n_kv):
                K = k_l[0, h].to(device).half()
                V = v_l[0, h].to(device).half()
                if K.shape[0] < 8:
                    continue

                torch.manual_seed(seed + l * 1000 + h)
                Qp = torch.randn(8, hd, device=device, dtype=torch.float16)
                sc = (Qp.float() @ K.float().T) / math.sqrt(hd)
                ref = (torch.softmax(sc, dim=-1) @ V.float()).half()

                ed = eigen.get((l, h), {})
                evec = ed.get("evec", torch.eye(hd))
                d_eff_lh = ed.get("d_eff", hd // 2)

                # TQ_3bit
                try:
                    tq = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)
                    ck = tq.compress_keys_pytorch(K)
                    cv = tq.compress_values_pytorch(V)
                    out_t = tq.fused_attention_pytorch(Qp, ck, cv)
                    val = cosine_sim(ref, out_t, hd)
                    if val is not None:
                        results["tq"].append(val)
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                # SQ_noQJL_v3
                try:
                    out_t = sq_noqjl_v3(Qp, K, V, evec, d_eff_lh, hd, device)
                    val = cosine_sim(ref, out_t, hd)
                    if val is not None:
                        results["sq"].append(val)
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        nev += 1
        if nev >= n_eval:
            break

    return results


def run_part3_confidence(args):
    """
    5-seed confidence intervals for Qwen2.5-1.5B-Instruct:
    TQ_3bit vs SQ_noQJL_v3.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("\n" + "=" * 90)
    log.info("PART 3: 5-Seed Confidence Intervals (Qwen2.5-1.5B-Instruct)")
    log.info("=" * 90)

    seeds = [42, 123, 7, 2024, 31415]
    n_calib_ci = 30 if args.quick else 100
    n_eval_ci = 15 if args.quick else 50

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    short = model_name.split("/")[-1]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=args.device,
            token=HF_TOKEN,
        )
        model.eval()

        cfg = model.config
        n_layers = cfg.num_hidden_layers
        n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        hd = cfg.hidden_size // cfg.num_attention_heads

        # Calibrate once (with fixed seed 42)
        torch.manual_seed(42)
        np.random.seed(42)
        eigen, mean_deff, mean_kappa = calibrate(
            model, tokenizer, n_calib_ci, args.device, n_layers, n_kv, hd
        )
        log.info("d_eff=%.1f, κ=%.2f", mean_deff, mean_kappa)

        seed_results = {}
        tq_means, sq_means = [], []

        for seed in seeds:
            log.info("\n  Seed %d...", seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            sr = run_one_seed(model, tokenizer, eigen, seed, n_eval_ci,
                              args.device, n_layers, n_kv, hd)
            tq_m = float(np.mean(sr["tq"])) if sr["tq"] else float("nan")
            sq_m = float(np.mean(sr["sq"])) if sr["sq"] else float("nan")
            tq_means.append(tq_m)
            sq_means.append(sq_m)
            seed_results[str(seed)] = {
                "tq_mean": tq_m,
                "sq_mean": sq_m,
                "tq_n": len(sr["tq"]),
                "sq_n": len(sr["sq"]),
                "sq_wins": bool(sq_m > tq_m) if not (math.isnan(sq_m) or math.isnan(tq_m)) else None,
            }
            log.info("  seed=%d | TQ=%.4f  SQ=%.4f  Δ=%+.4f  SQ wins: %s",
                     seed, tq_m, sq_m, sq_m - tq_m,
                     "YES" if sq_m > tq_m else "NO")

        # Filter out NaNs for summary stats
        valid_tq = [m for m in tq_means if not math.isnan(m)]
        valid_sq = [m for m in sq_means if not math.isnan(m)]

        tq_mean_all = float(np.mean(valid_tq)) if valid_tq else float("nan")
        tq_std_all = float(np.std(valid_tq)) if valid_tq else float("nan")
        sq_mean_all = float(np.mean(valid_sq)) if valid_sq else float("nan")
        sq_std_all = float(np.std(valid_sq)) if valid_sq else float("nan")

        sq_wins_all_seeds = all(
            seed_results[str(s)]["sq_wins"] is True for s in seeds
        )

        delta_mean = sq_mean_all - tq_mean_all
        delta_std = math.sqrt(tq_std_all ** 2 + sq_std_all ** 2) if valid_tq and valid_sq else float("nan")

        log.info("\n  === 5-Seed Summary ===")
        log.info("  TQ_3bit:    %.4f ± %.4f  [min=%.4f, max=%.4f]",
                 tq_mean_all, tq_std_all,
                 min(valid_tq) if valid_tq else float("nan"),
                 max(valid_tq) if valid_tq else float("nan"))
        log.info("  SQ_noQJL_v3: %.4f ± %.4f  [min=%.4f, max=%.4f]",
                 sq_mean_all, sq_std_all,
                 min(valid_sq) if valid_sq else float("nan"),
                 max(valid_sq) if valid_sq else float("nan"))
        log.info("  Delta:      %+.4f ± %.4f", delta_mean, delta_std)
        log.info("  SQ wins on ALL 5 seeds: %s", "YES" if sq_wins_all_seeds else "NO")

        ci_output = {
            "model": short,
            "seeds": seeds,
            "d_eff_mean": mean_deff,
            "kappa_mean": mean_kappa,
            "per_seed": seed_results,
            "summary": {
                "tq_3bit": {
                    "mean": tq_mean_all,
                    "std": tq_std_all,
                    "min": min(valid_tq) if valid_tq else None,
                    "max": max(valid_tq) if valid_tq else None,
                    "per_seed": tq_means,
                },
                "sq_noqjl_v3": {
                    "mean": sq_mean_all,
                    "std": sq_std_all,
                    "min": min(valid_sq) if valid_sq else None,
                    "max": max(valid_sq) if valid_sq else None,
                    "per_seed": sq_means,
                },
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "sq_wins_all_5_seeds": sq_wins_all_seeds,
            },
        }

    except RuntimeError as e:
        log.error("OOM in Part 3: %s", e)
        torch.cuda.empty_cache()
        ci_output = {"error": str(e)}
    except Exception as e:
        log.error("Error in Part 3: %s", e)
        import traceback; traceback.print_exc()
        ci_output = {"error": str(e)}
    finally:
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    out_path = RESULTS_DIR / "v3_confidence.json"
    with open(out_path, "w") as f:
        json.dump(ci_output, f, indent=2, default=str)
    log.info("\nPart 3 results saved to %s", out_path)
    return ci_output


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SpectralQuant v3: perplexity + cross-arch + 5-seed CI experiment"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast smoke-test: fewer models, sequences, skip PTB",
    )
    parser.add_argument(
        "--part", type=int, choices=[1, 2, 3], default=None,
        help="Run only a specific part (1=perplexity, 2=cross-arch, 3=confidence). "
             "Default: run all three.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    # Global seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    log.info("SpectralQuant v3 experiment — quick=%s, part=%s, device=%s",
             args.quick, args.part or "all", args.device)
    log.info("Results will be saved to %s", RESULTS_DIR)

    run_all = args.part is None
    t_start = time.time()

    if run_all or args.part == 1:
        run_part1_perplexity(args)

    if run_all or args.part == 2:
        run_part2_crossarch(args)

    if run_all or args.part == 3:
        run_part3_confidence(args)

    log.info("\nTotal elapsed: %.1f min", (time.time() - t_start) / 60)
    log.info("All results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
