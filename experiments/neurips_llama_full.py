"""
NeurIPS SpectralQuant — Script A: Llama 3.1-8B Full Evaluation
==============================================================
All Llama 3.1-8B experiments:
  PART 1: LongBench (narrativeqa, qasper, hotpotqa, gov_report, trec, triviaqa)
  PART 2: NIAH at extended contexts (4K–32K)
  PART 3: Perplexity at extended contexts (1K–8K)

Methods: FP16, TQ_3bit, SQ_noQJL_v3
Estimated runtime: 6–8 hours on B200

Usage:
    python neurips_llama_full.py [--quick] [--part {0,1,2,3}] [--device cuda]
    --quick : 1/5 sample sizes, shorter context lengths
    --part  : 0=all, 1=longbench, 2=niah, 3=ppl
"""

import sys, os, math, time, json, logging, argparse, gc, re, string
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np

# ── project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from turboquant_cutile import TurboQuantEngine, LloydMaxCodebook

log = logging.getLogger("neurips_llama")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

RESULTS_DIR = PROJECT_ROOT / "results" / "neurips"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def save_result(filename, data):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved: %s", path)


def quantize_nearest(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid quantisation (no gradient)."""
    c = centroids.to(x.device)
    diffs = x.unsqueeze(-1) - c
    return c[diffs.abs().argmin(dim=-1).long()]


def calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd):
    """Compute per-(layer, head) eigenvectors from key covariance on WikiText-103."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                      split=f"train[:{n_calib * 5}]")
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
        if nd % 50 == 0:
            log.info("  Calibration: %d/%d (%.0fs)", nd, n_calib, time.time() - t0)

    eigen = {}
    for l in range(n_layers):
        for h in range(n_kv):
            C = (cov[(l, h)]["xtx"] / cov[(l, h)]["n"]).float()
            ev, evec = torch.linalg.eigh(C)
            ev = ev.flip(0).clamp(min=0)
            evec = evec.flip(1)
            d_eff = max(2, min(int(round((ev.sum() ** 2 / (ev ** 2).sum()).item())), hd - 2))
            eigen[(l, h)] = {"evec": evec, "d_eff": d_eff, "ev": ev}
    log.info("Calibration done in %.1fs. mean d_eff=%.1f",
             time.time() - t0,
             float(np.mean([eigen[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv)])))
    return eigen


def extract_kv_layer(kv, l):
    """Extract (k, v) tensors for layer l from various cache formats."""
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
def compress_decompress_kv_sq(k_h, v_h, evec, hd, device):
    """SpectralQuant: spectral rotation, 2-bit keys, 3-bit values, no QJL."""
    K_f = k_h.float()
    V_f = v_h.float()
    ev = evec.to(device).float()
    VT = ev.T.contiguous()

    k_n = torch.norm(K_f, dim=-1, keepdim=True)
    K_rot = (K_f / (k_n + 1e-8)) @ VT
    cb_k = LloydMaxCodebook(hd, 2)
    K_hat = quantize_nearest(K_rot, cb_k.centroids.to(device))
    k_recon = ((K_hat @ ev) * k_n).half()

    v_n = torch.norm(V_f, dim=-1, keepdim=True)
    V_rot = (V_f / (v_n + 1e-8)) @ VT
    cb_v = LloydMaxCodebook(hd, 3)
    V_hat = quantize_nearest(V_rot, cb_v.centroids.to(device))
    v_recon = ((V_hat @ ev) * v_n).half()

    return k_recon, v_recon


@torch.no_grad()
def compress_decompress_kv_tq(k_h, v_h, engine):
    """TurboQuant 3-bit compress/decompress."""
    ck = engine.compress_keys_pytorch(k_h)
    k_recon = ck["k_mse"].half()
    cv = engine.compress_values_pytorch(v_h)
    v_recon = engine.decompress_values_pytorch(cv).half()
    return k_recon, v_recon


@torch.no_grad()
def rebuild_cache(model_kv, method, eigen, n_layers, n_kv, hd, device, tq_engine=None):
    """Rebuild a DynamicCache by compressing + decompressing each layer/head."""
    from transformers import DynamicCache

    new_cache = DynamicCache()
    for l in range(n_layers):
        k_full, v_full = extract_kv_layer(model_kv, l)
        k_full = k_full.to(device)
        v_full = v_full.to(device)
        k_recon = torch.zeros_like(k_full)
        v_recon = torch.zeros_like(v_full)

        for h in range(n_kv):
            k_h = k_full[0, h]  # [seq, head_dim]
            v_h = v_full[0, h]
            if method == "fp16":
                k_recon[0, h] = k_h
                v_recon[0, h] = v_h
            elif method == "tq":
                kr, vr = compress_decompress_kv_tq(k_h, v_h, tq_engine)
                k_recon[0, h] = kr
                v_recon[0, h] = vr
            elif method == "sq":
                ed = eigen[(l, h)]
                kr, vr = compress_decompress_kv_sq(k_h, v_h, ed["evec"], hd, device)
                k_recon[0, h] = kr
                v_recon[0, h] = vr

        new_cache.update(k_recon, v_recon, l)
    return new_cache


# ── KV compression hooks for token-by-token PPL ───────────────────────────────

class KVCompressor:
    """Compresses K,V in-place inside the attention forward hook."""

    def __init__(self, method, eigen, layer_idx, n_kv, hd, device):
        self.method = method
        self.eigen = eigen
        self.layer_idx = layer_idx
        self.n_kv = n_kv
        self.hd = hd
        self.device = device
        if method == "tq":
            self.tq_engine = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)
        else:
            self.tq_engine = None

    @torch.no_grad()
    def compress_cache_layer(self, cache, l_idx):
        """Compress key_cache[l_idx] and value_cache[l_idx] in place."""
        try:
            k = cache.key_cache[l_idx]
            v = cache.value_cache[l_idx]
        except (AttributeError, IndexError):
            return

        k_out = k.clone()
        v_out = v.clone()

        for h in range(min(self.n_kv, k.shape[1])):
            k_h = k[0, h].half()
            v_h = v[0, h].half()
            if self.method == "sq":
                ed = self.eigen.get((l_idx, h))
                if ed is None:
                    continue
                kr, vr = compress_decompress_kv_sq(k_h, v_h, ed["evec"], self.hd, self.device)
                k_out[0, h] = kr.to(k.dtype)
                v_out[0, h] = vr.to(v.dtype)
            elif self.method == "tq":
                kr, vr = compress_decompress_kv_tq(k_h, v_h, self.tq_engine)
                k_out[0, h] = kr.to(k.dtype)
                v_out[0, h] = vr.to(v.dtype)

        cache.key_cache[l_idx] = k_out
        cache.value_cache[l_idx] = v_out


def install_kv_compression(model, method, eigen, n_layers, n_kv, hd, device):
    """Register forward hooks on each attention layer to compress K,V after each token."""
    handles = []
    if method == "fp16":
        return handles

    for layer_idx in range(n_layers):
        attn = model.model.layers[layer_idx].self_attn
        compressor = KVCompressor(method, eigen, layer_idx, n_kv, hd, device)

        def make_hook(comp, l_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 3:
                    cache = output[2]
                    if cache is not None:
                        try:
                            comp.compress_cache_layer(cache, l_idx)
                        except Exception:
                            pass
                return output
            return hook

        h = attn.register_forward_hook(make_hook(compressor, layer_idx))
        handles.append(h)

    return handles


def remove_kv_compression(handles):
    for h in handles:
        if hasattr(h, "remove"):
            h.remove()


# ══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    m, n = len(gt_tokens), len(pred_tokens)
    m_cap, n_cap = min(m, 1000), min(n, 1000)
    gt_cap = gt_tokens[:m_cap]
    pred_cap = pred_tokens[:n_cap]
    dp = [[0] * (n_cap + 1) for _ in range(m_cap + 1)]
    for i in range(1, m_cap + 1):
        for j in range(1, n_cap + 1):
            if gt_cap[i - 1] == pred_cap[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m_cap][n_cap]
    if lcs == 0:
        return 0.0
    prec = lcs / n_cap
    rec = lcs / m_cap
    return 2 * prec * rec / (prec + rec)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def score_answer(prediction: str, references, metric: str) -> float:
    if isinstance(references, str):
        refs = [references]
    elif isinstance(references, list):
        flat = []
        for r in references:
            if isinstance(r, str):
                flat.append(r)
            elif isinstance(r, dict):
                flat.extend(v for v in r.values() if isinstance(v, str))
        refs = flat if flat else [str(references)]
    else:
        refs = [str(references)]

    if metric == "f1":
        return max(token_f1(prediction, r) for r in refs)
    elif metric == "rouge_l":
        return max(rouge_l(prediction, r) for r in refs)
    elif metric == "exact":
        return max(exact_match(prediction, r) for r in refs)
    return 0.0


# Subtask metadata
SUBTASK_META = {
    "narrativeqa": {"metric": "f1",     "ref_key": "answers"},
    "qasper":      {"metric": "f1",     "ref_key": "answers"},
    "hotpotqa":    {"metric": "f1",     "ref_key": "answers"},
    "gov_report":  {"metric": "rouge_l","ref_key": "answers"},
    "trec":        {"metric": "exact",  "ref_key": "answers"},
    "triviaqa":    {"metric": "f1",     "ref_key": "answers"},
}
CORE_SUBTASKS = ["narrativeqa", "qasper", "hotpotqa", "gov_report", "trec", "triviaqa"]


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: LONGBENCH
# ══════════════════════════════════════════════════════════════════════════════

def build_longbench_prompt(example: dict) -> tuple:
    context = example.get("context", "")
    question = example.get("input", "")
    ref = example.get("answers", "")
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    return prompt, ref


@torch.no_grad()
def run_longbench_example(
    model, tokenizer, prompt: str, method: str,
    eigen, n_layers, n_kv, hd, device,
    max_ctx: int = 3900, max_new_tokens: int = 100,
) -> str:
    enc = tokenizer(
        prompt, return_tensors="pt",
        max_length=max_ctx, truncation=True
    ).to(device)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 4:
        return ""

    # Prefill full context
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        kv = out.past_key_values

    if method == "fp16":
        gen_ids = model.generate(
            input_ids,
            past_key_values=kv,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        tq_engine = None
        if method == "tq":
            tq_engine = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)
        new_cache = rebuild_cache(kv, method, eigen, n_layers, n_kv, hd, device, tq_engine=tq_engine)
        gen_ids = model.generate(
            input_ids,
            past_key_values=new_cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip()


def run_part1_longbench(model, tokenizer, eigen, n_layers, n_kv, hd, device,
                        n_examples: int, partial_results: dict = None) -> dict:
    """Run LongBench for all subtasks and methods. Saves after each subtask."""
    from datasets import load_dataset

    methods = ["fp16", "tq", "sq"]
    results = partial_results or {}

    for subtask in CORE_SUBTASKS:
        if subtask in results:
            log.info("Skipping already-done subtask: %s", subtask)
            continue

        log.info("LongBench subtask: %s  (n=%d)", subtask, n_examples)
        meta = SUBTASK_META.get(subtask, {"metric": "f1", "ref_key": "answers"})
        metric = meta["metric"]

        try:
            ds = load_dataset("THUDM/LongBench", subtask, split="test", trust_remote_code=True)
            log.info("  Loaded %d examples for %s", len(ds), subtask)
        except Exception as e:
            log.warning("  Could not load %s: %s", subtask, e)
            results[subtask] = {"error": str(e)}
            save_result("neurips_llama_longbench.json", {"model": MODEL_NAME, "results": results})
            continue

        subtask_scores = {m: [] for m in methods}
        count = 0

        for example in ds:
            if count >= n_examples:
                break
            prompt, ref = build_longbench_prompt(example)

            for method in methods:
                try:
                    answer = run_longbench_example(
                        model, tokenizer, prompt, method,
                        eigen, n_layers, n_kv, hd, device,
                    )
                    score = score_answer(answer, ref, metric)
                    subtask_scores[method].append(score)
                    log.debug("  [%s/%s] ex=%d score=%.3f ans=%s", subtask, method, count, score, answer[:60])
                except torch.cuda.OutOfMemoryError:
                    log.warning("  OOM: %s/%s example %d — skipping", subtask, method, count)
                    torch.cuda.empty_cache()
                except Exception as exc:
                    log.warning("  Error %s/%s example %d: %s", subtask, method, count, exc)

            count += 1
            if count % 10 == 0:
                log.info("  [%s] %d/%d done", subtask, count, n_examples)

        results[subtask] = {"metric": metric}
        for method in methods:
            scores = subtask_scores[method]
            n = len(scores)
            mean = float(np.mean(scores)) if scores else None
            se = float(np.std(scores) / math.sqrt(n)) if n > 1 else None
            results[subtask][method] = {
                "mean": mean,
                "se": se,
                "n": n,
                "scores": [float(s) for s in scores],
            }
        log.info(
            "  [%s] FP16=%.3f  TQ=%.3f  SQ=%.3f",
            subtask,
            results[subtask]["fp16"]["mean"] or 0.0,
            results[subtask]["tq"]["mean"] or 0.0,
            results[subtask]["sq"]["mean"] or 0.0,
        )

        # Save after each subtask (survive crashes)
        save_result("neurips_llama_longbench.json", {"model": MODEL_NAME, "results": results})

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: NIAH (extended contexts 4K–32K)
# ══════════════════════════════════════════════════════════════════════════════

NEEDLE = "The special magic number is 7392."
QUESTION_NIAH = "What is the special magic number mentioned in the text? Answer with only the number."

FILLERS = [
    "The history of artificial intelligence spans decades of research and innovation across multiple disciplines. Scientists and engineers have worked tirelessly to develop systems capable of reasoning, learning, and solving complex problems that were once thought to be exclusively within the domain of human cognition. The field has seen remarkable transformations from early symbolic approaches to modern deep learning paradigms that power today's most capable AI systems.",
    "Modern agriculture relies on a complex network of supply chains, from seed production to distribution of harvested crops. Climate patterns, soil conditions, and water availability all play crucial roles in determining crop yields across different geographic regions around the world. Sustainable farming practices are becoming increasingly important as global food demand continues to grow with expanding populations.",
    "The development of quantum computing represents one of the most significant technological challenges of our era. Researchers are working to overcome decoherence, error correction, and scalability issues that currently limit the practical application of quantum processors in solving real-world computational problems that classical computers cannot handle efficiently.",
    "Urban planning in the twenty-first century must balance economic development with environmental sustainability. Cities around the globe are experimenting with green infrastructure, mixed-use zoning, and public transportation investments to create more livable and resilient urban environments for their growing populations and diverse communities.",
    "The study of marine biology has revealed extraordinary biodiversity in ocean ecosystems that continues to surprise researchers. From the sunlit surface waters to the crushing depths of hydrothermal vents, life has adapted to fill virtually every available ecological niche in the world's vast and largely unexplored oceans.",
    "Advances in materials science have led to the development of metamaterials with properties not found anywhere in nature. These carefully engineered structures can manipulate electromagnetic waves in unprecedented ways, enabling potential applications ranging from perfect lenses to novel antenna designs and even theoretical invisibility cloaking devices.",
    "The global financial system operates through an intricate web of institutions, regulations, and market mechanisms that span national borders. Central banks, commercial banks, and investment firms each play distinct yet interconnected roles in maintaining economic stability and facilitating the efficient flow of capital throughout the world economy.",
    "Archaeological discoveries continue to reshape our understanding of ancient civilizations and human history. Recent excavations across multiple continents have revealed sophisticated urban planning, advanced metallurgical techniques, and surprisingly complex social hierarchies in cultures far older than previously believed by mainstream historians and scholars.",
]


def build_niah_prompt(tokenizer, ctx_len: int, depth: float) -> tuple:
    """Build a budget-managed NIAH prompt using chat template.
    Returns (input_ids tensor, actual_len)."""
    question_part = "\n\n" + QUESTION_NIAH

    # Measure overhead from chat template
    test_messages = [{"role": "user", "content": "X"}]
    test_prompt = tokenizer.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True
    )
    overhead_tokens = len(tokenizer.encode(test_prompt, add_special_tokens=False))
    needle_tokens = len(tokenizer.encode(NEEDLE, add_special_tokens=False))
    question_tokens = len(tokenizer.encode(question_part, add_special_tokens=False))

    # Filler budget
    budget = ctx_len - overhead_tokens - needle_tokens - question_tokens - 10

    # Build diverse filler text exactly to budget
    filler_text = ""
    para_idx = 0
    while len(tokenizer.encode(filler_text, add_special_tokens=False)) < budget:
        filler_text += FILLERS[para_idx % len(FILLERS)] + "\n\n"
        para_idx += 1
    filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)
    if len(filler_tokens) > budget:
        filler_text = tokenizer.decode(filler_tokens[:budget], skip_special_tokens=True)

    # Insert needle at depth
    insert_pos = int(depth * len(filler_text))
    context = filler_text[:insert_pos] + "\n" + NEEDLE + "\n" + filler_text[insert_pos:]

    messages = [{"role": "user", "content": context + question_part}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    return input_ids, input_ids.shape[1]


def run_part2_niah(model, tokenizer, eigen, n_layers, n_kv, hd, device,
                   ctx_lengths, depths, partial_records=None) -> list:
    """Run NIAH for all (ctx_len, depth, method) combinations. Saves after each (ctx_len, depth)."""
    methods = ["fp16", "tq", "sq"]
    records = partial_records or []

    # Build a set of already-done keys
    done_keys = {
        (r["ctx_len"], r["depth"], r["method"])
        for r in records
    }

    for ctx_len in ctx_lengths:
        for depth in depths:
            # Check if all methods done for this (ctx_len, depth)
            pending_methods = [m for m in methods if (ctx_len, depth, m) not in done_keys]
            if not pending_methods:
                log.info("Skipping NIAH ctx=%d depth=%.2f (all done)", ctx_len, depth)
                continue

            try:
                input_ids, actual_len = build_niah_prompt(tokenizer, ctx_len, depth)
            except Exception as e:
                log.error("Failed to build NIAH prompt ctx=%d depth=%.2f: %s", ctx_len, depth, e)
                continue

            input_ids = input_ids.to(device)
            log.info("NIAH ctx=%d depth=%.0f%% actual_tokens=%d", ctx_len, depth * 100, actual_len)

            # Prefill once, reuse for all methods
            try:
                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                    kv_fp16 = out.past_key_values
            except torch.cuda.OutOfMemoryError:
                log.warning("  OOM on prefill ctx=%d — skipping all methods", ctx_len)
                torch.cuda.empty_cache()
                for m in pending_methods:
                    records.append({
                        "ctx_len": ctx_len, "depth": depth, "method": m,
                        "correct": None, "response": "OOM_prefill",
                    })
                save_result("neurips_llama_niah.json", _build_niah_output(records))
                continue

            for method in pending_methods:
                try:
                    tq_engine = None
                    if method == "tq":
                        tq_engine = TurboQuantEngine(head_dim=hd, total_bits=3, device=device)

                    if method == "fp16":
                        gen_ids = model.generate(
                            input_ids,
                            past_key_values=kv_fp16,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    else:
                        new_cache = rebuild_cache(
                            kv_fp16, method, eigen, n_layers, n_kv, hd, device, tq_engine=tq_engine
                        )
                        gen_ids = model.generate(
                            input_ids,
                            past_key_values=new_cache,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    response = tokenizer.decode(gen_ids[0, actual_len:], skip_special_tokens=True).strip()
                    correct = "7392" in response
                    log.info("  [%s] %s — \"%s\"", method, "PASS" if correct else "FAIL", response[:80])
                    records.append({
                        "ctx_len": ctx_len, "depth": depth, "method": method,
                        "correct": correct, "response": response[:200],
                        "actual_tokens": actual_len,
                    })
                except torch.cuda.OutOfMemoryError:
                    log.warning("  OOM: ctx=%d method=%s", ctx_len, method)
                    torch.cuda.empty_cache()
                    records.append({
                        "ctx_len": ctx_len, "depth": depth, "method": method,
                        "correct": None, "response": "OOM",
                    })
                except Exception as exc:
                    log.warning("  Error ctx=%d method=%s: %s", ctx_len, method, exc)
                    records.append({
                        "ctx_len": ctx_len, "depth": depth, "method": method,
                        "correct": None, "response": str(exc)[:100],
                    })

            # Save heatmap after each (ctx_len, depth)
            save_result("neurips_llama_niah.json", _build_niah_output(records))

            del kv_fp16
            torch.cuda.empty_cache()

    # Summary
    log.info("\n=== NIAH Summary ===")
    for method in ["fp16", "tq", "sq"]:
        mr = [r for r in records if r["method"] == method and r["correct"] is not None]
        correct = sum(1 for r in mr if r.get("correct", False))
        log.info("  %s: %d/%d correct (%.1f%%)", method, correct, len(mr),
                 100.0 * correct / len(mr) if mr else 0.0)

    return records


def _build_niah_output(records):
    hm = {}
    for r in records:
        m, ct, dp = r["method"], str(r["ctx_len"]), str(r["depth"])
        hm.setdefault(m, {}).setdefault(ct, {})[dp] = r.get("correct")
    return {"model": MODEL_NAME, "records": records, "heatmap": hm}


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: PERPLEXITY at extended contexts (1K–8K)
# ══════════════════════════════════════════════════════════════════════════════

def run_ppl_at_context_length(model, tokenizer, method, eigen,
                               n_layers, n_kv, hd, device,
                               text_ids: torch.Tensor, ctx_len: int) -> float:
    """Compute PPL on text_ids[:ctx_len] using token-by-token generation with compression hooks."""
    seq = text_ids[:, :ctx_len]
    seq_len = seq.shape[1]

    handles = install_kv_compression(model, method, eigen, n_layers, n_kv, hd, device)
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
                log.info("    [%s ctx=%d] token %d/%d, running PPL=%.2f",
                         method, ctx_len, i + 1, seq_len - 1, math.exp(np.mean(nlls)))
    finally:
        remove_kv_compression(handles)

    if not nlls:
        return float("nan")
    return math.exp(float(np.mean(nlls)))


def run_part3_perplexity(model, tokenizer, eigen, n_layers, n_kv, hd, device,
                          ctx_lengths, partial_results: dict = None) -> dict:
    """Run PPL at each context length for all methods. Saves after each ctx_len."""
    from datasets import load_dataset

    methods = ["fp16", "tq", "sq"]
    results = partial_results or {}

    # Load WikiText-2 test set
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([item["text"] for item in ds if len(item.get("text", "").strip()) > 0])

    max_needed = max(ctx_lengths)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_needed + 1)
    text_ids = enc["input_ids"].to(device)
    log.info("WikiText-2 total tokens: %d (need up to %d)", text_ids.shape[1], max_needed)

    for ctx_len in ctx_lengths:
        ctx_key = str(ctx_len)
        if ctx_key in results:
            log.info("Skipping already-done ctx_len=%d", ctx_len)
            continue

        if text_ids.shape[1] < ctx_len:
            log.warning("Not enough tokens for ctx_len=%d, have %d", ctx_len, text_ids.shape[1])
            continue

        results[ctx_key] = {}
        for method in methods:
            log.info("PPL [%s] ctx_len=%d ...", method, ctx_len)
            t0 = time.time()
            try:
                ppl = run_ppl_at_context_length(
                    model, tokenizer, method, eigen,
                    n_layers, n_kv, hd, device, text_ids, ctx_len
                )
                elapsed = time.time() - t0
                log.info("  %s ctx=%d: PPL=%.3f (%.1fs)", method, ctx_len, ppl, elapsed)
                results[ctx_key][method] = {"ppl": float(ppl), "time_s": float(elapsed)}
            except torch.cuda.OutOfMemoryError:
                log.warning("  OOM: method=%s ctx=%d", method, ctx_len)
                torch.cuda.empty_cache()
                results[ctx_key][method] = {"ppl": None, "error": "OOM"}
            except Exception as exc:
                log.warning("  Error method=%s ctx=%d: %s", method, ctx_len, exc)
                results[ctx_key][method] = {"ppl": None, "error": str(exc)}

            torch.cuda.empty_cache()

        # Save after each ctx_len
        save_result("neurips_llama_ppl.json", {
            "model": MODEL_NAME, "dataset": "wikitext-2-raw-v1",
            "ctx_lengths": ctx_lengths, "results": results,
        })

    # Summary table
    log.info("\n%-8s  %10s  %10s  %10s", "ctx_len", "FP16_PPL", "TQ_PPL", "SQ_PPL")
    for ctx_key in sorted(results, key=lambda x: int(x)):
        r = results[ctx_key]
        fp_ppl = r.get("fp16", {}).get("ppl")
        tq_ppl = r.get("tq", {}).get("ppl")
        sq_ppl = r.get("sq", {}).get("ppl")
        log.info("%-8s  %10s  %10s  %10s",
                 ctx_key,
                 f"{fp_ppl:.3f}" if fp_ppl else "N/A",
                 f"{tq_ppl:.3f}" if tq_ppl else "N/A",
                 f"{sq_ppl:.3f}" if sq_ppl else "N/A")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NeurIPS Script A: Llama 3.1-8B Full Evaluation"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1/5 sample sizes, fewer context lengths")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--part", type=int, default=0,
                        help="0=all, 1=longbench, 2=niah, 3=ppl")
    parser.add_argument("--n-calib", type=int, default=None)
    args = parser.parse_args()

    device = args.device

    # Settings based on --quick
    n_calib = args.n_calib or (40 if args.quick else 200)
    n_lb_examples = 20 if args.quick else 100

    niah_ctx_lengths = [4096, 8192] if args.quick else [4096, 8192, 16384, 32768]
    niah_depths = [0.1, 0.25, 0.5, 0.75, 0.9]

    ppl_ctx_lengths = [1024, 2048] if args.quick else [1024, 2048, 4096, 8192]

    log.info("=" * 70)
    log.info("NeurIPS Script A: Llama 3.1-8B  |  quick=%s  |  part=%d", args.quick, args.part)
    log.info("=" * 70)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading %s ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
        token=HF_TOKEN,
    )
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads
    log.info("Model: %d layers, %d KV heads, head_dim=%d", n_layers, n_kv, hd)

    # Calibrate once, shared across parts
    log.info("Calibrating (%d seqs) ...", n_calib)
    eigen = calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd)

    # ─── PART 1: LongBench ──────────────────────────────────────────────────
    if args.part in (0, 1):
        log.info("\n" + "=" * 60)
        log.info("PART 1: LongBench  (n=%d per subtask)", n_lb_examples)
        log.info("=" * 60)

        # Resume from existing results if available
        lb_resume = {}
        lb_path = RESULTS_DIR / "neurips_llama_longbench.json"
        if lb_path.exists():
            try:
                with open(lb_path) as f:
                    existing = json.load(f)
                lb_resume = existing.get("results", {})
                log.info("Resuming LongBench from %d done subtasks", len(lb_resume))
            except Exception:
                pass

        lb_results = run_part1_longbench(
            model, tokenizer, eigen, n_layers, n_kv, hd, device,
            n_examples=n_lb_examples,
            partial_results=lb_resume,
        )
        log.info("LongBench complete.")

    # ─── PART 2: NIAH ───────────────────────────────────────────────────────
    if args.part in (0, 2):
        log.info("\n" + "=" * 60)
        log.info("PART 2: NIAH  ctx_lengths=%s", niah_ctx_lengths)
        log.info("=" * 60)

        niah_resume = []
        niah_path = RESULTS_DIR / "neurips_llama_niah.json"
        if niah_path.exists():
            try:
                with open(niah_path) as f:
                    existing = json.load(f)
                niah_resume = existing.get("records", [])
                log.info("Resuming NIAH from %d done records", len(niah_resume))
            except Exception:
                pass

        niah_records = run_part2_niah(
            model, tokenizer, eigen, n_layers, n_kv, hd, device,
            ctx_lengths=niah_ctx_lengths,
            depths=niah_depths,
            partial_records=niah_resume,
        )
        log.info("NIAH complete.")

    # ─── PART 3: Perplexity ─────────────────────────────────────────────────
    if args.part in (0, 3):
        log.info("\n" + "=" * 60)
        log.info("PART 3: Perplexity  ctx_lengths=%s", ppl_ctx_lengths)
        log.info("=" * 60)

        ppl_resume = {}
        ppl_path = RESULTS_DIR / "neurips_llama_ppl.json"
        if ppl_path.exists():
            try:
                with open(ppl_path) as f:
                    existing = json.load(f)
                ppl_resume = existing.get("results", {})
                log.info("Resuming PPL from %d done ctx_lengths", len(ppl_resume))
            except Exception:
                pass

        ppl_results = run_part3_perplexity(
            model, tokenizer, eigen, n_layers, n_kv, hd, device,
            ctx_lengths=ppl_ctx_lengths,
            partial_results=ppl_resume,
        )
        log.info("Perplexity complete.")

    log.info("\nAll done. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
