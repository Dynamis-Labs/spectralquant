"""
Script C — v3 experiment suite:
  PART 1: d_eff sweep + Pareto frontier (quality vs compression)
  PART 2: Distribution shift robustness (wiki/code/math/multilingual)
  PART 3: Latency benchmarks (PyTorch vs PyTorch, fair comparison)
  PART 4: Attention pattern visualization

Usage:
  python run_v3_deff_distshift_latency.py              # full run
  python run_v3_deff_distshift_latency.py --quick      # fast smoke-test

Outputs (all under results/v3/):
  v3_deff_sweep.json
  v3_distribution_shift.json
  v3_latency.json
  fig_deff_pareto.png
  v3_attention_patterns.png
"""
import sys, os, math, time, json, logging, argparse, warnings
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from turboquant_cutile import TurboQuantEngine, LloydMaxCodebook

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("v3")


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers (adapted from run_final_experiments.py)
# ──────────────────────────────────────────────────────────────────────────────

def quantize_nearest(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid quantization (no gradient)."""
    c = centroids.to(x.device)
    diffs = x.unsqueeze(-1) - c
    return c[diffs.abs().argmin(dim=-1).long()]


@torch.no_grad()
def sq_noqjl(Q, K, V, evec, d_eff_override, hd, device):
    """SpectralQuant: spectral rotation, no QJL, uniform 3-bit values.

    d_eff_override: if not None, forces a specific d_eff instead of the
    per-head calibrated value.  Used for the sweep.
    """
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


@torch.no_grad()
def sq_noqjl_deff(Q, K, V, evec, d_eff, hd, device):
    """SQ with an explicit d_eff for the sweep.

    d_eff controls the non-uniform allocation:
      - Keys: 2-bit MSE quantization on all coordinates (uniform, no QJL)
      - Values: 3-bit MSE on top d_eff coords, 2-bit on remaining (d - d_eff)
    This way d_eff directly affects quality and compression.
    """
    K_f, V_f, Q_f = K.float(), V.float(), Q.float()
    VT = evec.T.contiguous().to(device).float()
    Vm = evec.to(device).float()

    k_n = torch.norm(K_f, dim=-1, keepdim=True)
    K_rot = (K_f / (k_n + 1e-8)) @ VT
    v_n = torch.norm(V_f, dim=-1, keepdim=True)
    V_rot = (V_f / (v_n + 1e-8)) @ VT

    # Keys: uniform 2-bit MSE (same as SQ_noQJL)
    cb_k = LloydMaxCodebook(hd, 2)
    K_hat = quantize_nearest(K_rot, cb_k.centroids)

    # Values: NON-UNIFORM — 3-bit semantic (top d_eff), 2-bit tail (rest)
    cb_v_high = LloydMaxCodebook(hd, 3)
    cb_v_low = LloydMaxCodebook(hd, 2)
    V_hat_high = quantize_nearest(V_rot[:, :d_eff], cb_v_high.centroids)
    V_hat_low = quantize_nearest(V_rot[:, d_eff:], cb_v_low.centroids)
    V_hat = torch.cat([V_hat_high, V_hat_low], dim=-1)

    K_mse = (K_hat @ Vm) * k_n
    V_rec = (V_hat @ Vm) * v_n

    scores = (Q_f @ K_mse.T) / math.sqrt(hd)
    return (torch.softmax(scores, dim=-1) @ V_rec).half()


def compression_ratio_analytical(hd: int, d_eff: int = 4, k_bits: int = 2,
                                  v_high_bits: int = 3, v_low_bits: int = 2) -> float:
    """Bits per token: FP16 vs compressed with non-uniform value allocation.

    FP16  : (16 + 16) * hd  per token (K + V)
    SQ    : k_bits*hd + d_eff*v_high + (hd-d_eff)*v_low + 2*16 + 16
    ratio = fp16_bits / compressed_bits
    """
    fp16_bits = 32.0 * hd  # K + V both fp16
    k_total = k_bits * hd
    v_total = v_high_bits * d_eff + v_low_bits * (hd - d_eff)
    comp_bits = k_total + v_total + 48.0  # + 3 fp16 norms (k_norm, k_res_norm, v_norm)
    return fp16_bits / comp_bits


def calibrate_from_data(model, tokenizer, texts, device, n_layers, n_kv, hd,
                        n_calib=50, max_length=512):
    """Compute per-(layer,head) eigenvectors from a list of raw text strings."""
    cov = {
        (l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64), "n": 0}
        for l in range(n_layers) for h in range(n_kv)
    }
    nd = 0
    for text in texts:
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt",
                        max_length=max_length, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue
        with torch.no_grad():
            out = model(**enc, use_cache=True)
            kv = out.past_key_values
        for l in range(n_layers):
            try:
                k_l = kv.key_cache[l].float().cpu()
            except Exception:
                try:
                    k_l = kv[l][0].float().cpu()
                except Exception:
                    k_l = list(kv)[l][0].float().cpu()
            for h in range(n_kv):
                X = k_l[0, h, :, :].double()
                cov[(l, h)]["xtx"] += X.T @ X
                cov[(l, h)]["n"] += X.shape[0]
        nd += 1
        if nd >= n_calib:
            break

    eigen = {}
    for l in range(n_layers):
        for h in range(n_kv):
            n = cov[(l, h)]["n"]
            if n == 0:
                # Fallback: identity
                evec = torch.eye(hd)
                d_eff_auto = hd // 4
            else:
                C = (cov[(l, h)]["xtx"] / n).float()
                ev, evec = torch.linalg.eigh(C)
                ev = ev.flip(0).clamp(min=0)
                evec = evec.flip(1)
                d_eff_auto = max(
                    2, min(int(round((ev.sum() ** 2 / (ev ** 2).sum()).item())), hd - 2)
                )
            eigen[(l, h)] = {"evec": evec, "d_eff": d_eff_auto}
    return eigen


def extract_kv(kv, l):
    """Robustly extract K, V tensors for layer l from various HF cache formats."""
    try:
        return kv.key_cache[l].float(), kv.value_cache[l].float()
    except Exception:
        try:
            return kv[l][0].float(), kv[l][1].float()
        except Exception:
            lkv = list(kv)[l]
            return lkv[0].float(), lkv[1].float()


def cosine_sim(ref: torch.Tensor, out: torch.Tensor, hd: int) -> float:
    c = torch.nn.functional.cosine_similarity(
        ref.float().reshape(-1, hd),
        out.float().reshape(-1, hd),
        dim=-1,
    ).mean().item()
    return c if not math.isnan(c) else None


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: d_eff sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_deff_sweep(model, tokenizer, eigen, eval_texts, hd, n_kv, device,
                   deff_values, slayers, n_eval, n_q=8):
    """Sweep d_eff and measure cosine similarity of attention output."""
    results = {d: [] for d in deff_values}
    nev = 0
    for text in eval_texts:
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt",
                        max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue
        with torch.no_grad():
            out_m = model(**enc, use_cache=True)
            kv = out_m.past_key_values

        for l in slayers:
            k_l, v_l = extract_kv(kv, l)
            for h in range(n_kv):
                K = k_l[0, h].to(device).half()
                V = v_l[0, h].to(device).half()
                if K.shape[0] < 8:
                    continue
                torch.manual_seed(42 + l * 1000 + h)
                Qp = torch.randn(n_q, hd, device=device, dtype=torch.float16)
                # FP16 reference
                sc_ref = (Qp.float() @ K.float().T) / math.sqrt(hd)
                ref_out = (torch.softmax(sc_ref, dim=-1) @ V.float()).half()
                ed = eigen[(l, h)]
                evec = ed["evec"].to(device)
                for d_eff in deff_values:
                    # clamp to valid range
                    d_eff_c = max(2, min(d_eff, hd - 2))
                    try:
                        sq_out = sq_noqjl_deff(Qp, K, V, evec, d_eff_c, hd, device)
                        c = cosine_sim(ref_out, sq_out, hd)
                        if c is not None:
                            results[d_eff].append(c)
                    except Exception as e:
                        log.debug("d_eff=%d l=%d h=%d error: %s", d_eff, l, h, e)
        nev += 1
        if nev >= n_eval:
            break

    summary = {}
    for d_eff in deff_values:
        vals = results[d_eff]
        cr = compression_ratio_analytical(hd, d_eff=d_eff)
        summary[d_eff] = {
            "cos_sim_mean": float(np.mean(vals)) if vals else None,
            "cos_sim_std": float(np.std(vals)) if vals else None,
            "compression_ratio": float(cr),
            "n": len(vals),
        }
    return summary


def plot_deff_pareto(sweep_data: dict, out_path: Path):
    """Plot Pareto frontier: cosine similarity vs compression ratio for each model."""
    fig, axes = plt.subplots(1, len(sweep_data), figsize=(7 * len(sweep_data), 5),
                             constrained_layout=True)
    if len(sweep_data) == 1:
        axes = [axes]

    cmap = plt.cm.viridis

    for ax, (model_label, model_sweep) in zip(axes, sweep_data.items()):
        deff_vals = sorted(model_sweep.keys(), key=int)
        cos_sims = []
        comp_ratios = []
        valid_deffs = []
        for d in deff_vals:
            entry = model_sweep[d]
            if entry["cos_sim_mean"] is not None:
                cos_sims.append(entry["cos_sim_mean"])
                comp_ratios.append(entry["compression_ratio"])
                valid_deffs.append(int(d))

        if not cos_sims:
            ax.set_title(f"{model_label}\n(no data)")
            continue

        colors = cmap(np.linspace(0.15, 0.85, len(valid_deffs)))
        sc = ax.scatter(comp_ratios, cos_sims, c=np.linspace(0.15, 0.85, len(valid_deffs)),
                        cmap=cmap, s=90, zorder=3, edgecolors="k", linewidths=0.5)
        ax.plot(comp_ratios, cos_sims, "--", color="gray", alpha=0.5, lw=1.2, zorder=2)
        for x, y, d in zip(comp_ratios, cos_sims, valid_deffs):
            ax.annotate(f"d={d}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)

        ax.set_xlabel("Compression Ratio (FP16 bits / compressed bits)", fontsize=10)
        ax.set_ylabel("Cosine Similarity (attention output vs FP16)", fontsize=10)
        ax.set_title(f"Pareto Frontier — {model_label}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax, label="d_eff (low→high)")
        cbar.set_ticks([])

    fig.suptitle("d_eff Sweep: Quality vs Compression (SpectralQuant)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved Pareto figure → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: Distribution shift
# ──────────────────────────────────────────────────────────────────────────────

def load_domain_texts(domain: str, n: int = 200) -> list[str]:
    """Load text samples for a given domain. Returns list of raw strings."""
    from datasets import load_dataset

    if domain == "wiki":
        try:
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train",
                              trust_remote_code=True)
            texts = [item["text"] for item in ds
                     if len(item.get("text", "").strip()) > 100][:n]
            log.info("Loaded %d wiki texts", len(texts))
            return texts
        except Exception as e:
            log.warning("WikiText load failed: %s", e)
            return []

    elif domain == "wiki_eval":
        try:
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test",
                              trust_remote_code=True)
            texts = [item["text"] for item in ds
                     if len(item.get("text", "").strip()) > 100][:n]
            log.info("Loaded %d wiki eval texts", len(texts))
            return texts
        except Exception as e:
            log.warning("WikiText eval load failed: %s", e)
            return []

    elif domain == "code":
        try:
            ds = load_dataset("openai_humaneval", split="test",
                              trust_remote_code=True)
            texts = [item["prompt"] for item in ds
                     if len(item.get("prompt", "").strip()) > 30][:n]
            log.info("Loaded %d HumanEval code prompts", len(texts))
            return texts
        except Exception as e:
            log.warning("HumanEval load failed (%s), trying fallback", e)
            # Fallback: synthetic code-like strings
            try:
                ds = load_dataset("codeparrot/github-code", split="train",
                                  streaming=True, trust_remote_code=True)
                texts = []
                for item in ds:
                    t = item.get("code", "") or item.get("content", "")
                    if len(t.strip()) > 80:
                        texts.append(t[:1024])
                    if len(texts) >= n:
                        break
                log.info("Loaded %d code texts from codeparrot fallback", len(texts))
                return texts
            except Exception as e2:
                log.warning("Code fallback failed: %s", e2)
                return []

    elif domain == "math":
        try:
            ds = load_dataset("gsm8k", "main", split="train",
                              trust_remote_code=True)
            texts = [item["question"] for item in ds
                     if len(item.get("question", "").strip()) > 20][:n]
            log.info("Loaded %d GSM8K math questions", len(texts))
            return texts
        except Exception as e:
            log.warning("GSM8K load failed: %s", e)
            return []

    elif domain == "multilingual":
        for lang_cfg in ["20231101.zh", "20231101.de", "20231101.fr"]:
            try:
                ds = load_dataset("wikimedia/wikipedia", lang_cfg, split="train",
                                  streaming=True, trust_remote_code=True)
                texts = []
                for item in ds:
                    t = item.get("text", "")
                    if len(t.strip()) > 100:
                        texts.append(t[:512])
                    if len(texts) >= n:
                        break
                if texts:
                    log.info("Loaded %d multilingual texts (%s)", len(texts), lang_cfg)
                    return texts
            except Exception as e:
                log.warning("Multilingual %s failed: %s", lang_cfg, e)
        return []

    else:
        log.warning("Unknown domain: %s", domain)
        return []


def evaluate_cos_sim(model, tokenizer, eval_texts, eigen, hd, n_kv, device,
                     slayers, n_eval, n_q=8):
    """Measure mean cosine similarity between SQ and FP16 attention outputs.
    Also returns TQ cosine similarity for delta computation.
    """
    sq_scores = []
    tq_scores = []
    nev = 0
    for text in eval_texts:
        if len(text.strip()) < 30:
            continue
        enc = tokenizer(text, return_tensors="pt",
                        max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 8:
            continue
        with torch.no_grad():
            out_m = model(**enc, use_cache=True)
            kv = out_m.past_key_values

        for l in slayers:
            k_l, v_l = extract_kv(kv, l)
            for h in range(n_kv):
                K = k_l[0, h].to(device).half()
                V = v_l[0, h].to(device).half()
                if K.shape[0] < 4:
                    continue
                torch.manual_seed(42 + l * 1000 + h)
                Qp = torch.randn(n_q, hd, device=device, dtype=torch.float16)
                sc_ref = (Qp.float() @ K.float().T) / math.sqrt(hd)
                ref_out = (torch.softmax(sc_ref, dim=-1) @ V.float()).half()
                ed = eigen[(l, h)]
                # SQ
                try:
                    sq_out = sq_noqjl(Qp, K, V, ed["evec"], ed["d_eff"], hd, device)
                    c = cosine_sim(ref_out, sq_out, hd)
                    if c is not None:
                        sq_scores.append(c)
                except Exception:
                    pass
                # TQ
                try:
                    tq = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)
                    ck = tq.compress_keys_pytorch(K)
                    cv = tq.compress_values_pytorch(V)
                    tq_out = tq.fused_attention_pytorch(Qp, ck, cv)
                    c = cosine_sim(ref_out, tq_out, hd)
                    if c is not None:
                        tq_scores.append(c)
                except Exception:
                    pass
        nev += 1
        if nev >= n_eval:
            break

    sq_mean = float(np.mean(sq_scores)) if sq_scores else None
    tq_mean = float(np.mean(tq_scores)) if tq_scores else None
    return sq_mean, tq_mean, len(sq_scores)


def run_distribution_shift(model, tokenizer, hd, n_kv, device, slayers,
                            n_calib, n_eval, quick):
    """Calibrate on two domains, evaluate on four. Return delta table."""
    # Load all domain texts upfront
    n_text = 100 if quick else 300
    log.info("Loading domain texts...")

    wiki_train = load_domain_texts("wiki", n=n_text)
    wiki_eval_texts = load_domain_texts("wiki_eval", n=n_text)
    code_texts = load_domain_texts("code", n=n_text)
    math_texts = load_domain_texts("math", n=n_text)
    multi_texts = load_domain_texts("multilingual", n=min(50, n_text))

    calib_domains = [
        ("wiki", wiki_train),
        ("code", code_texts),
    ]

    eval_domains_base = [
        ("wiki", wiki_eval_texts),
        ("code", code_texts),
    ]
    if not quick:
        eval_domains_base += [
            ("math", math_texts),
            ("multilingual", multi_texts),
        ]
    # Filter out empty domains
    eval_domains = [(k, v) for k, v in eval_domains_base if v]

    n_layers = model.config.num_hidden_layers
    results = {}

    for calib_label, calib_texts in calib_domains:
        if not calib_texts:
            log.warning("No calibration texts for domain: %s, skipping", calib_label)
            continue
        log.info("Calibrating on domain: %s (%d texts)", calib_label, len(calib_texts))
        eigen = calibrate_from_data(
            model, tokenizer, calib_texts, device, n_layers, n_kv, hd,
            n_calib=n_calib, max_length=512
        )
        for eval_label, eval_texts in eval_domains:
            if not eval_texts:
                continue
            log.info("  Evaluating on domain: %s", eval_label)
            sq_mean, tq_mean, n_samples = evaluate_cos_sim(
                model, tokenizer, eval_texts, eigen, hd, n_kv, device,
                slayers, n_eval=n_eval
            )
            key = f"{calib_label}→{eval_label}"
            results[key] = {
                "calib_domain": calib_label,
                "eval_domain": eval_label,
                "sq_cos_sim": sq_mean,
                "tq_cos_sim": tq_mean,
                "delta_vs_tq": (float(sq_mean - tq_mean)
                                if sq_mean is not None and tq_mean is not None
                                else None),
                "n_samples": n_samples,
            }
            log.info(
                "    %s: SQ=%.4f  TQ=%.4f  Δ=%+.4f  (n=%d)",
                key,
                sq_mean if sq_mean else float("nan"),
                tq_mean if tq_mean else float("nan"),
                results[key]["delta_vs_tq"] if results[key]["delta_vs_tq"] else float("nan"),
                n_samples,
            )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: Latency benchmarks (PyTorch vs PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_op(fn, n_warmup=3, n_runs=10):
    """Time a zero-argument callable fn. Returns (mean_ms, std_ms)."""
    for _ in range(n_warmup):
        fn()
        _sync()
    times = []
    for _ in range(n_runs):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)  # ms
    return float(np.mean(times)), float(np.std(times))


def run_latency_benchmarks(hd, device, seq_lengths, n_warmup=3, n_runs=10):
    """Pure PyTorch latency for TQ and SQ at various sequence lengths."""
    results = {}
    for seq_len in seq_lengths:
        log.info("  seq_len=%d", seq_len)
        K = torch.randn(seq_len, hd, device=device, dtype=torch.float16)
        V = torch.randn(seq_len, hd, device=device, dtype=torch.float16)
        Q = torch.randn(8, hd, device=device, dtype=torch.float16)

        # Dummy eigen for SQ (random orthogonal)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(7)
        G = torch.randn(hd, hd, generator=gen)
        evec, _ = torch.linalg.qr(G)
        evec = evec.to(device)

        tq = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)

        # ── TQ compression
        def tq_compress():
            return tq.compress_keys_pytorch(K), tq.compress_values_pytorch(V)

        tq_comp_mean, tq_comp_std = time_op(tq_compress, n_warmup, n_runs)

        # ── TQ attention (using pre-compressed)
        ck_ref, cv_ref = tq.compress_keys_pytorch(K), tq.compress_values_pytorch(V)

        def tq_attn():
            return tq.fused_attention_pytorch(Q, ck_ref, cv_ref)

        tq_attn_mean, tq_attn_std = time_op(tq_attn, n_warmup, n_runs)

        # ── SQ compression
        def sq_compress():
            K_f = K.float()
            k_n = torch.norm(K_f, dim=-1, keepdim=True)
            K_rot = (K_f / (k_n + 1e-8)) @ evec.T.float()
            cb_k = LloydMaxCodebook(hd, 2)
            K_hat = quantize_nearest(K_rot, cb_k.centroids)
            K_mse = (K_hat @ evec.float()) * k_n

            V_f = V.float()
            v_n = torch.norm(V_f, dim=-1, keepdim=True)
            V_rot = (V_f / (v_n + 1e-8)) @ evec.T.float()
            cb_v = LloydMaxCodebook(hd, 3)
            V_hat = quantize_nearest(V_rot, cb_v.centroids)
            V_rec = (V_hat @ evec.float()) * v_n
            return K_mse, V_rec

        sq_comp_mean, sq_comp_std = time_op(sq_compress, n_warmup, n_runs)

        # ── SQ attention
        K_mse_ref, V_rec_ref = sq_compress()

        def sq_attn():
            sc = (Q.float() @ K_mse_ref.T) / math.sqrt(hd)
            return (torch.softmax(sc, dim=-1) @ V_rec_ref).half()

        sq_attn_mean, sq_attn_std = time_op(sq_attn, n_warmup, n_runs)

        # ── Calibration time (eigendecomposition of hd×hd covariance)
        cov_mat = (K.float().T @ K.float()) / seq_len

        def calib_op():
            ev, evec_c = torch.linalg.eigh(cov_mat)
            return evec_c

        calib_mean, calib_std = time_op(calib_op, n_warmup, n_runs)

        results[seq_len] = {
            "tq": {
                "compression_ms_mean": tq_comp_mean,
                "compression_ms_std": tq_comp_std,
                "attention_ms_mean": tq_attn_mean,
                "attention_ms_std": tq_attn_std,
                "calibration_ms_mean": None,  # TQ has no calibration
                "calibration_ms_std": None,
            },
            "sq": {
                "compression_ms_mean": sq_comp_mean,
                "compression_ms_std": sq_comp_std,
                "attention_ms_mean": sq_attn_mean,
                "attention_ms_std": sq_attn_std,
                "calibration_ms_mean": calib_mean,
                "calibration_ms_std": calib_std,
            },
        }
        log.info(
            "    TQ  compress=%.2f±%.2f ms  attn=%.2f±%.2f ms",
            tq_comp_mean, tq_comp_std, tq_attn_mean, tq_attn_std,
        )
        log.info(
            "    SQ  compress=%.2f±%.2f ms  attn=%.2f±%.2f ms  calib=%.2f±%.2f ms",
            sq_comp_mean, sq_comp_std, sq_attn_mean, sq_attn_std,
            calib_mean, calib_std,
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: Attention pattern visualization
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_attention_weights_fp16(Q, K, hd):
    sc = (Q.float() @ K.float().T) / math.sqrt(hd)
    return torch.softmax(sc, dim=-1).cpu().numpy()


@torch.no_grad()
def get_attention_weights_tq(Q, K, V, hd, device):
    tq = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)
    ck = tq.compress_keys_pytorch(K)
    sc = tq.attention_scores_pytorch(Q, ck)
    return torch.softmax(sc, dim=-1).cpu().numpy()


@torch.no_grad()
def get_attention_weights_sq(Q, K, V, evec, hd, device):
    K_f = K.float()
    k_n = torch.norm(K_f, dim=-1, keepdim=True)
    VT = evec.T.contiguous().to(device).float()
    Vm = evec.to(device).float()
    K_rot = (K_f / (k_n + 1e-8)) @ VT
    cb_k = LloydMaxCodebook(hd, 2)
    K_hat = quantize_nearest(K_rot, cb_k.centroids)
    K_mse = (K_hat @ Vm) * k_n
    sc = (Q.float() @ K_mse.T) / math.sqrt(hd)
    return torch.softmax(sc, dim=-1).cpu().numpy()


def run_attention_viz(model, tokenizer, wiki_texts, eigen, hd, n_kv, device,
                      target_layer, out_path):
    """Plot side-by-side FP16 / TQ / SQ attention heatmaps."""
    sample_texts = [t for t in wiki_texts if len(t.strip()) > 150][:2]
    if not sample_texts:
        log.warning("No suitable texts for attention visualization")
        return

    fig = plt.figure(figsize=(15, 5 * len(sample_texts)), constrained_layout=True)
    outer = gridspec.GridSpec(len(sample_texts), 1, figure=fig, hspace=0.35)

    for si, text in enumerate(sample_texts):
        enc = tokenizer(text, return_tensors="pt",
                        max_length=64, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 8:
            continue
        with torch.no_grad():
            out_m = model(**enc, use_cache=True)
            kv = out_m.past_key_values

        k_l, v_l = extract_kv(kv, target_layer)
        seq_len = k_l.shape[2]

        # Pick head 0
        h = 0
        K = k_l[0, h].to(device).half()
        V = v_l[0, h].to(device).half()
        if K.shape[0] < 4:
            continue
        # Use actual token query vectors from the model for realism
        # (simulate by using random Qs seeded deterministically)
        torch.manual_seed(99)
        n_q = min(seq_len, 16)
        Qp = torch.randn(n_q, hd, device=device, dtype=torch.float16)

        w_fp16 = get_attention_weights_fp16(Qp, K, hd)         # (n_q, seq_len)
        w_tq   = get_attention_weights_tq(Qp, K, V, hd, device)
        ed = eigen[(target_layer, h)]
        w_sq   = get_attention_weights_sq(Qp, K, V, ed["evec"], hd, device)

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[si], wspace=0.08
        )
        titles = ["FP16 (reference)", "TurboQuant (3-bit)", "SpectralQuant (3-bit)"]
        weights_list = [w_fp16, w_tq, w_sq]
        vmin = min(w.min() for w in weights_list)
        vmax = max(w.max() for w in weights_list)

        for ci, (title, w) in enumerate(zip(titles, weights_list)):
            ax = fig.add_subplot(inner[ci])
            im = ax.imshow(w, aspect="auto", cmap="viridis",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Key position", fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"Sample {si+1} (layer {target_layer}, head {h})\nQuery position",
                               fontsize=8)
            else:
                ax.set_ylabel("")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="Attention weight" if ci == 2 else "")

    fig.suptitle("Attention Weight Patterns: FP16 vs TQ vs SQ",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved attention pattern figure → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpectralQuant v3: d_eff sweep, dist shift, latency, attn viz"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test mode: smaller d_eff grid, fewer samples, seq_len=512 only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+",
                        default=["Qwen/Qwen2.5-1.5B-Instruct"],
                        help="HuggingFace model IDs to sweep (Part 1 multi-model)")
    args = parser.parse_args()
    device = args.device

    # ── Experiment parameters ─────────────────────────────────────────────────
    if args.quick:
        deff_values  = [2, 4, 8, 16, 32]
        n_calib      = 30
        n_eval       = 15
        seq_lengths  = [512]
        n_warmup     = 2
        n_runs_lat   = 5
    else:
        deff_values  = list(range(2, 34, 2))        # 2,4,...,32
        n_calib      = 100
        n_eval       = 50
        seq_lengths  = [512, 2048, 8192]
        n_warmup     = 3
        n_runs_lat   = 10

    out_dir = PROJECT_ROOT / "results" / "v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # ── Primary model (1.5B) ─────────────────────────────────────────────────
    primary_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    log.info("Loading primary model: %s", primary_model_name)
    tokenizer = AutoTokenizer.from_pretrained(primary_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        primary_model_name, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads

    # Sample layers evenly for evaluation speed
    slayers = list(range(0, n_layers, max(1, n_layers // 5)))[:5]
    log.info("Model config: n_layers=%d  n_kv=%d  hd=%d  slayers=%s",
             n_layers, n_kv, hd, slayers)

    # ── Calibration (wiki, used as baseline) ─────────────────────────────────
    log.info("Calibrating on WikiText (baseline)...")
    try:
        wiki_ds_train = load_dataset("wikitext", "wikitext-103-raw-v1",
                                     split="train", trust_remote_code=True)
        wiki_calib_texts = [item["text"] for item in wiki_ds_train
                            if len(item.get("text", "").strip()) > 100][:n_calib * 5]
    except Exception as e:
        log.error("Failed to load WikiText: %s", e)
        wiki_calib_texts = []

    eigen_wiki = calibrate_from_data(
        model, tokenizer, wiki_calib_texts, device, n_layers, n_kv, hd,
        n_calib=n_calib
    )
    mean_deff = int(round(np.mean(
        [eigen_wiki[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)]
    )))
    log.info("Mean calibrated d_eff = %d", mean_deff)

    # ── WikiText eval texts ───────────────────────────────────────────────────
    try:
        wiki_ds_test = load_dataset("wikitext", "wikitext-103-raw-v1",
                                    split="test", trust_remote_code=True)
        wiki_eval_texts = [item["text"] for item in wiki_ds_test
                           if len(item.get("text", "").strip()) > 100][:n_eval * 5]
    except Exception as e:
        log.warning("WikiText test load failed: %s; using train tail", e)
        wiki_eval_texts = wiki_calib_texts[n_calib:]

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: d_eff sweep (primary model)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n=== PART 1: d_eff Sweep ===")
    all_sweep_data = {}

    # Primary model sweep
    log.info("Sweeping d_eff for %s ...", primary_model_name)
    sweep_primary = run_deff_sweep(
        model, tokenizer, eigen_wiki, wiki_eval_texts,
        hd, n_kv, device, deff_values, slayers, n_eval
    )
    all_sweep_data[primary_model_name.split("/")[-1]] = {
        str(k): v for k, v in sweep_primary.items()
    }
    log.info("d_eff sweep (primary): %s",
             {d: f"{v['cos_sim_mean']:.4f}" for d, v in sweep_primary.items()
              if v["cos_sim_mean"] is not None})

    # Optional 7B model sweep (only if explicitly requested)
    secondary_name = "Qwen/Qwen2.5-7B-Instruct"
    if secondary_name in args.models:
        log.info("Loading secondary model: %s", secondary_name)
        try:
            model7 = AutoModelForCausalLM.from_pretrained(
                secondary_name, torch_dtype=torch.float16, device_map=device
            )
            model7.eval()
            cfg7 = model7.config
            n_layers7 = cfg7.num_hidden_layers
            n_kv7 = getattr(cfg7, "num_key_value_heads", cfg7.num_attention_heads)
            hd7 = cfg7.hidden_size // cfg7.num_attention_heads
            slayers7 = list(range(0, n_layers7, max(1, n_layers7 // 5)))[:5]
            eigen7 = calibrate_from_data(
                model7, tokenizer, wiki_calib_texts, device,
                n_layers7, n_kv7, hd7, n_calib=n_calib
            )
            sweep7 = run_deff_sweep(
                model7, tokenizer, eigen7, wiki_eval_texts,
                hd7, n_kv7, device, deff_values, slayers7, n_eval
            )
            all_sweep_data[secondary_name.split("/")[-1]] = {
                str(k): v for k, v in sweep7.items()
            }
            del model7
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            log.warning("7B model sweep failed: %s", e)

    # Save d_eff sweep results
    sweep_out = out_dir / "v3_deff_sweep.json"
    with open(sweep_out, "w") as f:
        json.dump(all_sweep_data, f, indent=2)
    log.info("Saved d_eff sweep → %s", sweep_out)

    # Plot Pareto frontier
    pareto_path = out_dir / "fig_deff_pareto.png"
    plot_deff_pareto(all_sweep_data, pareto_path)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: Distribution shift robustness
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n=== PART 2: Distribution Shift Robustness ===")
    dist_results = run_distribution_shift(
        model, tokenizer, hd, n_kv, device, slayers,
        n_calib=n_calib, n_eval=n_eval, quick=args.quick
    )

    dist_out = out_dir / "v3_distribution_shift.json"
    with open(dist_out, "w") as f:
        json.dump(dist_results, f, indent=2)
    log.info("Saved distribution shift results → %s", dist_out)

    # Print summary table
    log.info("\nDistribution Shift Summary:")
    log.info("%-25s  %8s  %8s  %+8s  %6s",
             "calib→eval", "SQ", "TQ", "Δ(SQ-TQ)", "n")
    for key, v in dist_results.items():
        sq = v["sq_cos_sim"]
        tq = v["tq_cos_sim"]
        delta = v["delta_vs_tq"]
        log.info("%-25s  %8s  %8s  %+8s  %6d",
                 key,
                 f"{sq:.4f}" if sq is not None else "N/A",
                 f"{tq:.4f}" if tq is not None else "N/A",
                 f"{delta:.4f}" if delta is not None else "N/A",
                 v["n_samples"])

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: Latency benchmarks (PyTorch vs PyTorch)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n=== PART 3: Latency Benchmarks ===")
    log.info("Running on device=%s, seq_lengths=%s", device, seq_lengths)
    lat_results = run_latency_benchmarks(
        hd, device, seq_lengths, n_warmup=n_warmup, n_runs=n_runs_lat
    )

    lat_out = out_dir / "v3_latency.json"
    with open(lat_out, "w") as f:
        json.dump(lat_results, f, indent=2)
    log.info("Saved latency results → %s", lat_out)

    # Print latency table
    log.info("\nLatency Summary (mean ± std, ms):")
    log.info("%-10s  %-6s  %16s  %16s  %16s",
             "seq_len", "method", "compression", "attention", "calibration")
    for sl in seq_lengths:
        if sl not in lat_results:
            continue
        for method in ["tq", "sq"]:
            r = lat_results[sl][method]
            calib_str = (
                f"{r['calibration_ms_mean']:.2f}±{r['calibration_ms_std']:.2f}"
                if r["calibration_ms_mean"] is not None else "—"
            )
            log.info("%-10d  %-6s  %16s  %16s  %16s",
                     sl, method.upper(),
                     f"{r['compression_ms_mean']:.2f}±{r['compression_ms_std']:.2f}",
                     f"{r['attention_ms_mean']:.2f}±{r['attention_ms_std']:.2f}",
                     calib_str)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 4: Attention pattern visualization
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n=== PART 4: Attention Pattern Visualization ===")
    target_layer = slayers[len(slayers) // 2]  # middle sampled layer
    attn_out = out_dir / "v3_attention_patterns.png"
    run_attention_viz(
        model, tokenizer, wiki_eval_texts, eigen_wiki, hd, n_kv, device,
        target_layer=target_layer, out_path=attn_out
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("v3 experiment suite complete.")
    log.info("Output directory: %s", out_dir)
    log.info("  v3_deff_sweep.json        → d_eff sweep results")
    log.info("  v3_distribution_shift.json→ distribution shift table")
    log.info("  v3_latency.json           → latency benchmarks")
    log.info("  fig_deff_pareto.png       → Pareto frontier plot")
    log.info("  v3_attention_patterns.png → attention heatmaps")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
