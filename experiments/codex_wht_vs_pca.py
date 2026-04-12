"""
Experiment 4: PCA vs WHT basis in the favorable 1-bit-key proxy setting.

Primary setup:
- Q = K
- non-causal self-attention proxy

Goal:
- isolate whether the strong 1-bit-key result depends on a learned per-head
  PCA basis or survives under a TurboQuant-style WHT rotation.
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from benchmark_utils import (  # noqa: E402
    causal_attention_output,
    collect_eval_qkv,
    noncausal_attention_output,
    select_head_indices,
    select_layer_indices,
)

log = logging.getLogger("codex_wht_vs_pca")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

RESULTS_DIR = PROJECT_ROOT / "results" / "codex_wht_vs_pca"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN")

MAIN_GRID = [
    ("pca", "1bit-K_6bit-V", 1, 6),
    ("wht_only", "1bit-K_6bit-V", 1, 6),
    ("wht_polarquant", "1bit-K_6bit-V", 1, 6),
    ("pca", "1bit-K_4bit-V", 1, 4),
    ("wht_only", "1bit-K_4bit-V", 1, 4),
    ("wht_polarquant", "1bit-K_4bit-V", 1, 4),
    ("pca", "2bit-K_6bit-V", 2, 6),
    ("wht_only", "2bit-K_6bit-V", 2, 6),
    ("wht_polarquant", "2bit-K_6bit-V", 2, 6),
]


def save_result(filename: str, data: dict) -> Path:
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved: %s", path)
    return path


def solve_lloyd_max_for_sigma(sigma, bits, max_iter=200, tol=1e-10):
    if bits <= 0:
        return torch.tensor([0.0], dtype=torch.float32)
    if bits == 1:
        c = sigma * math.sqrt(2.0 / math.pi)
        return torch.tensor([-c, c], dtype=torch.float32)
    from scipy import integrate

    n_levels = 1 << bits
    pdf = lambda x: (1.0 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-x * x / (2 * sigma * sigma))
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


def solve_polarquant_centroids(bits: int, d: int) -> torch.Tensor:
    if bits <= 0:
        return torch.tensor([0.0], dtype=torch.float32)
    if bits == 1:
        c = math.sqrt(2.0 / (math.pi * d))
        return torch.tensor([-c, c], dtype=torch.float32)
    if bits == 2:
        return torch.tensor([-1.51, -0.453, 0.453, 1.51], dtype=torch.float32) / math.sqrt(d)
    return solve_lloyd_max_for_sigma(1.0 / math.sqrt(d), bits)


def quantize_nearest(x, centroids):
    c = centroids.to(x.device)
    diffs = x.unsqueeze(-1) - c
    return c[diffs.abs().argmin(dim=-1).long()]


def load_model_tokenizer(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    needs_token = any(x in model_name for x in ["llama", "Llama", "gemma", "Gemma"])
    token = HF_TOKEN if needs_token else None
    log.info("Loading %s on %s ...", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"token": token}
    if device == "mps":
        model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    elif device == "cpu":
        model_kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    else:
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = device
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    try:
        test_ids = tokenizer("test", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**test_ids, use_cache=True)
        kv = out.past_key_values
        try:
            actual_hd = kv.key_cache[0].shape[-1]
        except Exception:
            actual_hd = kv[0][0].shape[-1]
        if actual_hd != hd:
            hd = actual_hd
    except Exception:
        pass
    return model, tokenizer, n_layers, n_kv, hd


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


def calibrate_pca(model, tokenizer, n_calib, device, n_layers, n_kv, hd, seq_len=512):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{n_calib * 5}]")
    cov_keys = {(l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64), "n": 0} for l in range(n_layers) for h in range(n_kv)}
    cov_vals = {(l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64), "n": 0} for l in range(n_layers) for h in range(n_kv)}
    nd = 0
    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue
        with torch.no_grad():
            out = model(**enc, use_cache=True)
        kv = out.past_key_values
        for l in range(n_layers):
            try:
                k_l, v_l = extract_kv_layer(kv, l)
            except Exception:
                continue
            k_l = k_l.float().cpu()
            v_l = v_l.float().cpu()
            for h in range(min(n_kv, k_l.shape[1])):
                xk = k_l[0, h].double()
                xv = v_l[0, h].double()
                cov_keys[(l, h)]["xtx"] += xk.T @ xk
                cov_keys[(l, h)]["n"] += xk.shape[0]
                cov_vals[(l, h)]["xtx"] += xv.T @ xv
                cov_vals[(l, h)]["n"] += xv.shape[0]
        nd += 1
        if nd >= n_calib:
            break

    def _eigen(cov_dict):
        eigen = {}
        for l in range(n_layers):
            for h in range(n_kv):
                n = cov_dict[(l, h)]["n"]
                if n == 0:
                    eigen[(l, h)] = {"evec": torch.eye(hd), "ev": torch.ones(hd)}
                    continue
                c = (cov_dict[(l, h)]["xtx"] / n).float()
                ev, evec = torch.linalg.eigh(c)
                eigen[(l, h)] = {"evec": evec.flip(1), "ev": ev.flip(0).clamp(min=0)}
        return eigen

    return _eigen(cov_keys), _eigen(cov_vals)


def hadamard_matrix(n: int) -> torch.Tensor:
    if n & (n - 1):
        raise ValueError(f"Hadamard transform requires power-of-two dimension, got {n}")
    h = torch.tensor([[1.0]], dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    return h / math.sqrt(n)


def build_wht_basis(hd: int, *, seed: int) -> torch.Tensor:
    base = hadamard_matrix(hd)
    g = torch.Generator()
    g.manual_seed(seed)
    signs1 = torch.where(torch.rand(hd, generator=g) > 0.5, 1.0, -1.0).float()
    signs2 = torch.where(torch.rand(hd, generator=g) > 0.5, 1.0, -1.0).float()
    d1 = torch.diag(signs1)
    d2 = torch.diag(signs2)
    return d1 @ base @ d2


def build_wht_bank(n_layers: int, n_kv: int, hd: int):
    keys = {}
    vals = {}
    for l in range(n_layers):
        for h in range(n_kv):
            keys[(l, h)] = {"evec": build_wht_basis(hd, seed=17_000 + l * 257 + h), "ev": torch.ones(hd)}
            vals[(l, h)] = {"evec": build_wht_basis(hd, seed=29_000 + l * 257 + h), "ev": torch.ones(hd)}
    return keys, vals


def calibrate_wht_variances(model, tokenizer, n_calib, device, n_layers, n_kv, hd, wht_keys, wht_vals, seq_len=512):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{n_calib * 5}]")
    key_stats = {(l, h): {"sumsq": torch.zeros(hd, dtype=torch.float64), "n": 0} for l in range(n_layers) for h in range(n_kv)}
    val_stats = {(l, h): {"sumsq": torch.zeros(hd, dtype=torch.float64), "n": 0} for l in range(n_layers) for h in range(n_kv)}
    nd = 0
    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue
        with torch.no_grad():
            out = model(**enc, use_cache=True)
        kv = out.past_key_values
        for l in range(n_layers):
            try:
                k_l, v_l = extract_kv_layer(kv, l)
            except Exception:
                continue
            k_l = k_l.float().cpu()
            v_l = v_l.float().cpu()
            for h in range(min(n_kv, k_l.shape[1])):
                k_basis = wht_keys[(l, h)]["evec"]
                v_basis = wht_vals[(l, h)]["evec"]
                xk = k_l[0, h]
                xv = v_l[0, h]
                xk = xk / (torch.norm(xk, dim=-1, keepdim=True) + 1e-8)
                xv = xv / (torch.norm(xv, dim=-1, keepdim=True) + 1e-8)
                rot_k = xk @ k_basis
                rot_v = xv @ v_basis
                key_stats[(l, h)]["sumsq"] += rot_k.double().pow(2).sum(dim=0)
                key_stats[(l, h)]["n"] += rot_k.shape[0]
                val_stats[(l, h)]["sumsq"] += rot_v.double().pow(2).sum(dim=0)
                val_stats[(l, h)]["n"] += rot_v.shape[0]
        nd += 1
        if nd >= n_calib:
            break

    for l in range(n_layers):
        for h in range(n_kv):
            kn = key_stats[(l, h)]["n"]
            vn = val_stats[(l, h)]["n"]
            if kn > 0:
                wht_keys[(l, h)]["ev"] = (key_stats[(l, h)]["sumsq"] / kn).float().clamp(min=1e-6)
            if vn > 0:
                wht_vals[(l, h)]["ev"] = (val_stats[(l, h)]["sumsq"] / vn).float().clamp(min=1e-6)
    return wht_keys, wht_vals


class BasisEngine:
    def __init__(self, k_basis, k_eigs, v_basis, v_eigs, key_bits_per_dim, value_bits_per_dim, hd):
        self.hd = hd
        self.vk = k_basis.float()
        self.vk_t = self.vk.T.contiguous()
        self.vv = v_basis.float()
        self.vv_t = self.vv.T.contiguous()

        k_ev = k_eigs.float().clamp(min=1e-6).cpu().numpy()
        v_ev = v_eigs.float().clamp(min=1e-6).cpu().numpy()
        self.k_codebooks = [solve_lloyd_max_for_sigma(float(np.sqrt(k_ev[i])), key_bits_per_dim) for i in range(hd)]
        self.v_codebooks = [solve_lloyd_max_for_sigma(float(np.sqrt(v_ev[i])), value_bits_per_dim) for i in range(hd)]
        self.key_total_bits = hd * key_bits_per_dim + 16
        self.val_total_bits = hd * value_bits_per_dim + 16

    def compress_keys(self, k):
        norms = torch.norm(k.float(), dim=-1, keepdim=True)
        rotated = (k.float() / (norms + 1e-8)) @ self.vk
        quant = rotated.clone()
        for i in range(self.hd):
            quant[:, i] = quantize_nearest(rotated[:, i], self.k_codebooks[i])
        return {"qr": quant, "norms": norms.squeeze(-1)}

    def compress_values(self, v):
        norms = torch.norm(v.float(), dim=-1, keepdim=True)
        rotated = (v.float() / (norms + 1e-8)) @ self.vv
        quant = rotated.clone()
        for i in range(self.hd):
            quant[:, i] = quantize_nearest(rotated[:, i], self.v_codebooks[i])
        return {"qr": quant, "norms": norms.squeeze(-1)}

    def decompress(self, compressed, vt):
        qr = compressed["qr"].float()
        norms = compressed["norms"].float()
        return (qr @ vt.to(qr.device)) * norms.unsqueeze(-1)

    def evaluate(self, q, k, v, *, attention_mode: str):
        ck = self.compress_keys(k)
        cv = self.compress_values(v)
        k_hat = self.decompress(ck, self.vk_t)
        v_hat = self.decompress(cv, self.vv_t)
        attention_fn = noncausal_attention_output if attention_mode == "noncausal" else causal_attention_output
        out_ref = attention_fn(q, k, v, self.hd)
        out_hat = attention_fn(q, k_hat, v_hat, self.hd)
        scores_ref = q.float() @ k.float().T / math.sqrt(self.hd)
        scores_hat = q.float() @ k_hat.float().T / math.sqrt(self.hd)
        if attention_mode == "causal" and scores_ref.shape[0] == scores_ref.shape[1]:
            mask = torch.triu(
                torch.full((scores_ref.shape[0], scores_ref.shape[1]), float("-inf"), device=scores_ref.device, dtype=scores_ref.dtype),
                diagonal=1,
            )
            scores_ref = scores_ref + mask
            scores_hat = scores_hat + mask
        probs_ref = F.softmax(scores_ref, dim=-1)
        probs_hat = F.softmax(scores_hat, dim=-1)
        eps = 1e-8
        kl = torch.sum(probs_ref.clamp_min(eps) * (torch.log(probs_ref.clamp_min(eps)) - torch.log(probs_hat.clamp_min(eps))), dim=-1).mean().item()
        return {
            "attn_cos_sim": F.cosine_similarity(out_ref.float(), out_hat.float(), dim=-1).mean().item(),
            "attn_kl_fp16_to_compressed": kl,
            "key_cos_sim": F.cosine_similarity(k.float(), k_hat.float(), dim=-1).mean().item(),
            "val_cos_sim": F.cosine_similarity(v.float(), v_hat.float(), dim=-1).mean().item(),
            "key_bits": self.key_total_bits,
            "val_bits": self.val_total_bits,
            "total_bits": self.key_total_bits + self.val_total_bits,
            "total_compress": (2 * self.hd * 16) / (self.key_total_bits + self.val_total_bits),
        }


class WHTPolarQuantEngine:
    def __init__(self, k_basis, v_basis, key_bits_per_dim, value_bits_per_dim, hd):
        self.hd = hd
        self.vk = k_basis.float()
        self.vk_t = self.vk.T.contiguous()
        self.vv = v_basis.float()
        self.vv_t = self.vv.T.contiguous()
        self.k_codebook = solve_polarquant_centroids(key_bits_per_dim, hd)
        self.v_codebook = solve_polarquant_centroids(value_bits_per_dim, hd)
        self.key_total_bits = hd * key_bits_per_dim + 16
        self.val_total_bits = hd * value_bits_per_dim + 16

    def _compress(self, x, basis, codebook):
        norms = torch.norm(x.float(), dim=-1, keepdim=True)
        rotated = (x.float() / (norms + 1e-8)) @ basis
        quant = quantize_nearest(rotated, codebook)
        return {"qr": quant, "norms": norms.squeeze(-1)}

    def compress_keys(self, k):
        return self._compress(k, self.vk, self.k_codebook)

    def compress_values(self, v):
        return self._compress(v, self.vv, self.v_codebook)

    def decompress(self, compressed, vt):
        qr = compressed["qr"].float()
        norms = compressed["norms"].float()
        qr_norms = torch.norm(qr, dim=-1, keepdim=True)
        qr = qr / torch.where(qr_norms > 1e-10, qr_norms, torch.ones_like(qr_norms))
        return (qr @ vt.to(qr.device)) * norms.unsqueeze(-1)

    def evaluate(self, q, k, v, *, attention_mode: str):
        ck = self.compress_keys(k)
        cv = self.compress_values(v)
        k_hat = self.decompress(ck, self.vk_t)
        v_hat = self.decompress(cv, self.vv_t)
        attention_fn = noncausal_attention_output if attention_mode == "noncausal" else causal_attention_output
        out_ref = attention_fn(q, k, v, self.hd)
        out_hat = attention_fn(q, k_hat, v_hat, self.hd)
        scores_ref = q.float() @ k.float().T / math.sqrt(self.hd)
        scores_hat = q.float() @ k_hat.float().T / math.sqrt(self.hd)
        if attention_mode == "causal" and scores_ref.shape[0] == scores_ref.shape[1]:
            mask = torch.triu(
                torch.full((scores_ref.shape[0], scores_ref.shape[1]), float("-inf"), device=scores_ref.device, dtype=scores_ref.dtype),
                diagonal=1,
            )
            scores_ref = scores_ref + mask
            scores_hat = scores_hat + mask
        probs_ref = F.softmax(scores_ref, dim=-1)
        probs_hat = F.softmax(scores_hat, dim=-1)
        eps = 1e-8
        kl = torch.sum(probs_ref.clamp_min(eps) * (torch.log(probs_ref.clamp_min(eps)) - torch.log(probs_hat.clamp_min(eps))), dim=-1).mean().item()
        return {
            "attn_cos_sim": F.cosine_similarity(out_ref.float(), out_hat.float(), dim=-1).mean().item(),
            "attn_kl_fp16_to_compressed": kl,
            "key_cos_sim": F.cosine_similarity(k.float(), k_hat.float(), dim=-1).mean().item(),
            "val_cos_sim": F.cosine_similarity(v.float(), v_hat.float(), dim=-1).mean().item(),
            "key_bits": self.key_total_bits,
            "val_bits": self.val_total_bits,
            "total_bits": self.key_total_bits + self.val_total_bits,
            "total_compress": (2 * self.hd * 16) / (self.key_total_bits + self.val_total_bits),
        }


def run_model(
    model_name,
    short_name,
    device,
    *,
    n_calib,
    n_eval,
    seq_len,
    layer_mode,
    head_mode,
    grid,
    query_mode: str,
    attention_mode: str,
):
    model, tokenizer, n_layers, n_kv, hd = load_model_tokenizer(model_name, device)
    layer_indices = select_layer_indices(n_layers, layer_mode)
    head_indices = select_head_indices(n_kv, head_mode)

    pca_keys, pca_vals = calibrate_pca(model, tokenizer, n_calib, device, n_layers, n_kv, hd, seq_len=512)
    wht_keys, wht_vals = build_wht_bank(n_layers, n_kv, hd)
    wht_keys, wht_vals = calibrate_wht_variances(
        model,
        tokenizer,
        n_calib,
        device,
        n_layers,
        n_kv,
        hd,
        wht_keys,
        wht_vals,
        seq_len=512,
    )

    eval_bundle = collect_eval_qkv(
        model,
        tokenizer,
        device,
        n_eval=n_eval,
        seq_len=seq_len,
        head_dim=hd,
        layer_indices=layer_indices,
        head_indices=head_indices,
        extract_kv_layer=extract_kv_layer,
        split_start=n_calib * 5,
    )

    results = {
        "model": model_name,
        "short_name": short_name,
        "device": device,
        "seq_len": seq_len,
        "n_calib": n_calib,
        "n_eval": n_eval,
        "layer_indices": layer_indices,
        "head_indices": head_indices,
        "primary_setup": {"query_mode": query_mode, "attention_mode": attention_mode},
        "grid": {},
    }

    for basis_type, config_name, key_bits, value_bits in grid:
        metrics_list = []
        for l in layer_indices:
            for h in head_indices:
                kk = eval_bundle["keys"].get((l, h), [])
                vv = eval_bundle["values"].get((l, h), [])
                qq = eval_bundle["queries"].get((l, h), [])
                if not kk or not vv:
                    continue
                k_all = torch.cat(kk, dim=0).to(device).float()
                v_all = torch.cat(vv, dim=0).to(device).float()
                if query_mode == "proxy_qeqk":
                    q_eval = k_all
                else:
                    if not qq:
                        continue
                    q_eval = torch.cat(qq, dim=0).to(device).float()
                if basis_type == "pca":
                    engine = BasisEngine(
                        pca_keys[(l, h)]["evec"].to(device),
                        pca_keys[(l, h)]["ev"].to(device),
                        pca_vals[(l, h)]["evec"].to(device),
                        pca_vals[(l, h)]["ev"].to(device),
                        key_bits,
                        value_bits,
                        hd,
                    )
                elif basis_type == "wht_only":
                    engine = BasisEngine(
                        wht_keys[(l, h)]["evec"].to(device),
                        wht_keys[(l, h)]["ev"].to(device),
                        wht_vals[(l, h)]["evec"].to(device),
                        wht_vals[(l, h)]["ev"].to(device),
                        key_bits,
                        value_bits,
                        hd,
                    )
                else:
                    engine = WHTPolarQuantEngine(
                        wht_keys[(l, h)]["evec"].to(device),
                        wht_vals[(l, h)]["evec"].to(device),
                        key_bits,
                        value_bits,
                        hd,
                    )
                metrics_list.append(engine.evaluate(q_eval, k_all, v_all, attention_mode=attention_mode))

        avg = {key: float(np.mean([m[key] for m in metrics_list])) for key in metrics_list[0]}
        avg["n_evals"] = len(metrics_list)
        results["grid"][f"{basis_type}:{config_name}"] = {"primary": avg}

    del model
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--mode", choices=["main", "confirm"], default="main")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--short-name", default=None)
    parser.add_argument("--n-calib", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument(
        "--grid-mode",
        choices=["threeway", "legacy"],
        default="threeway",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["proxy_qeqk_noncausal", "real_q_causal"],
        default="proxy_qeqk_noncausal",
    )
    args = parser.parse_args()

    t0 = time.time()
    if args.evaluation_mode == "proxy_qeqk_noncausal":
        query_mode = "proxy_qeqk"
        attention_mode = "noncausal"
        suffix = "proxy"
    else:
        query_mode = "real_q"
        attention_mode = "causal"
        suffix = "realq_causal"

    if args.grid_mode == "threeway":
        suffix = f"{suffix}_threeway"
        main_grid = MAIN_GRID
        confirm_grid = [
            ("pca", "1bit-K_6bit-V", 1, 6),
            ("wht_only", "1bit-K_6bit-V", 1, 6),
            ("wht_polarquant", "1bit-K_6bit-V", 1, 6),
        ]
    else:
        main_grid = [
            ("pca", "1bit-K_6bit-V", 1, 6),
            ("wht_polarquant", "1bit-K_6bit-V", 1, 6),
            ("pca", "1bit-K_4bit-V", 1, 4),
            ("wht_polarquant", "1bit-K_4bit-V", 1, 4),
            ("pca", "2bit-K_6bit-V", 2, 6),
            ("wht_polarquant", "2bit-K_6bit-V", 2, 6),
        ]
        confirm_grid = [("pca", "1bit-K_6bit-V", 1, 6), ("wht_polarquant", "1bit-K_6bit-V", 1, 6)]

    if args.mode == "main":
        model_name = args.model_name or "Qwen/Qwen2.5-0.5B-Instruct"
        short_name = args.short_name or "Qwen-0.5B"
        n_calib = args.n_calib if args.n_calib is not None else 4
        n_eval = args.n_eval if args.n_eval is not None else 1
        grid = main_grid
    else:
        model_name = args.model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        short_name = args.short_name or "TinyLlama-1.1B"
        n_calib = args.n_calib if args.n_calib is not None else 2
        n_eval = args.n_eval if args.n_eval is not None else 1
        grid = confirm_grid

    if args.mode == "main":
        result = run_model(
            model_name,
            short_name,
            args.device,
            n_calib=n_calib,
            n_eval=n_eval,
            seq_len=512,
            layer_mode="sampled",
            head_mode="sampled",
            grid=grid,
            query_mode=query_mode,
            attention_mode=attention_mode,
        )
        result["elapsed_seconds"] = time.time() - t0
        save_result(f"codex_wht_vs_pca_{suffix}_{short_name}.json", result)
    else:
        result = run_model(
            model_name,
            short_name,
            args.device,
            n_calib=n_calib,
            n_eval=n_eval,
            seq_len=512,
            layer_mode="sampled",
            head_mode="sampled",
            grid=grid,
            query_mode=query_mode,
            attention_mode=attention_mode,
        )
        result["elapsed_seconds"] = time.time() - t0
        save_result(f"codex_wht_vs_pca_{suffix}_{short_name}.json", result)


if __name__ == "__main__":
    main()
