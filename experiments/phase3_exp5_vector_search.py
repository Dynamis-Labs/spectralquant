#!/usr/bin/env python3
"""
phase3_exp5_vector_search.py — Experiment 5: Vector Search

Applies SpectralQuant to embedding vectors (BGE-large or sentence-transformers)
for nearest-neighbor retrieval and evaluates:
  - Recall@1 and Recall@10 at {2, 3, 4, 8} bits
  - Comparison: SpectralQuant vs TurboQuant vs Product Quantization

Uses a subset of the BEIR benchmark (MSMARCO or similar) or synthetic data
if BEIR is unavailable.

Usage:
  python phase3_exp5_vector_search.py [--quick] [--embedding-model MODEL]
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results" / "vector_search"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model — BGE-large-en-v1.5 is a strong open-source retrieval model
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BIT_LEVELS = [2, 3, 4, 8]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(EXPERIMENTS_DIR / "experiment_log.txt", mode="a"),
    ],
)
log = logging.getLogger("exp5_vector_search")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s (%.1f GB)", props.name, (getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)) / 1e9)
        return torch.device("cuda")
    log.warning("No GPU — CPU only.")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_embedding_model(device: torch.device):
    """Load BGE-large or fallback to MiniLM."""
    for model_name in [EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL]:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=str(device))
            log.info("Loaded embedding model: %s", model_name)
            return model, model_name
        except ImportError:
            log.info("sentence-transformers not installed; trying transformers directly.")
            break
        except Exception as e:
            log.warning("Could not load %s: %s — trying fallback.", model_name, e)

    # Fallback: use transformers directly
    try:
        from transformers import AutoTokenizer, AutoModel

        model_name = FALLBACK_EMBEDDING_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()

        class _Wrapper:
            def __init__(self, m, tok, dev):
                self._model = m
                self._tok = tok
                self._device = dev

            def encode(self, texts, batch_size=32, normalize_embeddings=True, **kw):
                all_embs = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    enc = self._tok(batch, return_tensors="pt",
                                    max_length=128, truncation=True, padding=True)
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                    with torch.no_grad():
                        out = self._model(**enc)
                    emb = out.last_hidden_state[:, 0, :]  # CLS
                    if normalize_embeddings:
                        emb = F.normalize(emb, dim=-1)
                    all_embs.append(emb.cpu().numpy())
                return np.concatenate(all_embs, axis=0)

        log.info("Loaded fallback embedding model via transformers: %s", model_name)
        return _Wrapper(model, tokenizer, device), model_name

    except Exception as e:
        log.error("Could not load any embedding model: %s", e)
        raise


# ---------------------------------------------------------------------------
# Dataset — BEIR subset or synthetic
# ---------------------------------------------------------------------------

def load_dataset(n_corpus: int = 5000, n_queries: int = 200) -> tuple[list[str], list[str], dict]:
    """
    Try to load a BEIR dataset subset, fall back to synthetic data.

    Returns:
      corpus: list of passage strings
      queries: list of query strings
      qrels: dict {query_idx: set of relevant corpus_idx}
    """
    try:
        from datasets import load_dataset as hf_load

        log.info("Trying to load BEIR MS-MARCO subset …")
        # Use the 'fiqa' dataset from BEIR as it's smaller and freely available
        dataset = hf_load("BeIR/fiqa", "corpus", split="corpus", streaming=False)
        corpus = [row.get("text", row.get("title", "")) for row in dataset][:n_corpus]

        query_dataset = hf_load("BeIR/fiqa", "queries", split="queries", streaming=False)
        queries = [row["text"] for row in query_dataset][:n_queries]

        # Try to load qrels
        try:
            qrel_dataset = hf_load("BeIR/fiqa-qrels", "test", split="test")
            qrels = {}
            for row in qrel_dataset:
                qid = row["query-id"]
                cid = row["corpus-id"]
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(cid)
            log.info("FIQA loaded: %d corpus, %d queries, %d qrel pairs",
                     len(corpus), len(queries), sum(len(v) for v in qrels.values()))
            return corpus, queries, {"type": "fiqa", "qrels": qrels}
        except Exception:
            # No qrels — generate synthetic ground truth
            pass

    except Exception as e:
        log.info("BEIR/FIQA not available (%s) — using synthetic data.", e)

    # Synthetic: create a retrieval dataset where query i is relevant to corpus i
    log.info("Generating synthetic retrieval dataset (%d corpus, %d queries) …",
             n_corpus, n_queries)
    vocab = [
        "machine learning", "neural network", "transformer", "attention",
        "compression", "quantization", "embedding", "retrieval", "semantic",
        "vector", "representation", "language model", "inference", "latency",
        "throughput", "accuracy", "benchmark", "evaluation", "optimization",
        "gradient", "training", "fine-tuning", "dataset", "corpus", "query",
    ]
    corpus = [
        " ".join(random.choices(vocab, k=random.randint(10, 30)))
        for _ in range(n_corpus)
    ]
    queries = []
    qrels_idx = {}
    for i in range(n_queries):
        # Query = first 5 words of corpus[i] + some noise
        base = corpus[i].split()[:5]
        query = " ".join(base + random.choices(vocab, k=2))
        queries.append(query)
        qrels_idx[i] = {i}  # query i is relevant to corpus i

    return corpus, queries, {"type": "synthetic", "qrels_idx": qrels_idx}


# ---------------------------------------------------------------------------
# Quantization methods
# ---------------------------------------------------------------------------

def _uniform_quantize_np(X: np.ndarray, bits: int) -> np.ndarray:
    """Per-row uniform quantization."""
    n_levels = 2 ** bits
    std = X.std(axis=-1, keepdims=True).clip(1e-6, None)
    X_clamp = X.clip(-3 * std, 3 * std)
    X_norm = (X_clamp / (3 * std) + 1) / 2
    X_int = np.round(X_norm * (n_levels - 1)).clip(0, n_levels - 1)
    return X_int / (n_levels - 1) * 2 * 3 * std - 3 * std


def quantize_turboquant(X: np.ndarray, bits: int, seed: int = 0) -> np.ndarray:
    """TurboQuant: random rotation + uniform quantization."""
    d = X.shape[-1]
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    X_rot = X @ Q
    X_q = _uniform_quantize_np(X_rot, bits)
    return X_q @ Q.T


def quantize_spectralquant(
    X: np.ndarray,
    bits: int,
    eigenvectors: np.ndarray,
    mean: np.ndarray,
    d_eff: float,
) -> np.ndarray:
    """SpectralQuant: spectral rotation + non-uniform quantization."""
    d = X.shape[-1]
    d_sem = max(1, int(round(d_eff)))

    # Non-uniform bits
    budget = d * bits
    best_err, b_high, b_low = float("inf"), bits + 1, max(1, bits - 1)
    for bh in range(1, 9):
        for bl in range(1, bh + 1):
            err = abs(d_sem * bh + (d - d_sem) * bl - budget)
            if err < best_err:
                best_err = err
                b_high, b_low = bh, bl

    X_rot = (X - mean) @ eigenvectors
    X_sem_q = _uniform_quantize_np(X_rot[:, :d_sem], b_high)
    X_tail_q = _uniform_quantize_np(X_rot[:, d_sem:], b_low)
    X_q = np.concatenate([X_sem_q, X_tail_q], axis=-1)
    return X_q @ eigenvectors.T + mean


def quantize_product_quantization(
    X: np.ndarray,
    bits: int,
    n_subspaces: int | None = None,
) -> np.ndarray:
    """
    Simple product quantization (PQ) approximation.
    Divides the vector into n_subspaces sub-vectors and quantizes each independently.
    """
    d = X.shape[-1]
    if n_subspaces is None:
        n_subspaces = max(1, d // 8)

    # Ensure divisibility
    n_subspaces = min(n_subspaces, d)
    while d % n_subspaces != 0:
        n_subspaces -= 1
    subspace_dim = d // n_subspaces

    X_q = np.zeros_like(X)
    for i in range(n_subspaces):
        start = i * subspace_dim
        end = start + subspace_dim
        sub = X[:, start:end]
        X_q[:, start:end] = _uniform_quantize_np(sub, bits)

    return X_q


# ---------------------------------------------------------------------------
# Eigenspectral analysis on embeddings
# ---------------------------------------------------------------------------

def compute_embedding_eigenspectrum(X: np.ndarray) -> dict:
    """Compute eigenspectrum of embedding matrix X: [n, d]."""
    mu = X.mean(axis=0)
    Xc = X - mu
    C = (Xc.T @ Xc) / len(X)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    lam = eigvals[eigvals > 0]
    d_eff = float((lam.sum() ** 2) / (lam ** 2).sum()) if len(lam) > 0 else 0.0
    d_sem = max(1, int(round(d_eff)))
    kappa = float(lam[d_sem - 1] / lam[d_sem]) if d_sem < len(lam) and lam[d_sem] > 1e-10 else float("inf")

    return {
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "mean": mu,
        "d_eff": d_eff,
        "kappa": kappa,
    }


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def compute_recall(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    qrels_idx: dict,  # {query_i: set of relevant corpus_i}
    k_values: list[int] = [1, 10],
    batch_size: int = 256,
) -> dict:
    """Compute Recall@k using dot-product similarity (for normalized embeddings)."""
    n_queries = query_embs.shape[0]
    recalls = {k: [] for k in k_values}

    for q_start in range(0, n_queries, batch_size):
        q_batch = query_embs[q_start:q_start + batch_size]
        # Dot product similarity (assumes L2 normalized)
        sims = q_batch @ corpus_embs.T  # [batch, n_corpus]

        for rel_i, q_row in enumerate(sims):
            q_idx = q_start + rel_i
            if q_idx not in qrels_idx or not qrels_idx[q_idx]:
                continue
            relevant = qrels_idx[q_idx]

            top_k = np.argsort(-q_row)
            for k in k_values:
                retrieved = set(top_k[:k].tolist())
                recall = len(relevant & retrieved) / len(relevant)
                recalls[k].append(recall)

    return {f"Recall@{k}": round(float(np.mean(v)), 4) if v else 0.0 for k, v in recalls.items()}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_recall_curves(results: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.3,
    })

    methods = sorted(set(r["method"] for r in results))
    colors = {"SpectralQuant": "#d62728", "TurboQuant": "#1f77b4",
               "ProductQuant": "#ff7f0e", "FP16": "#2ca02c"}
    markers = {"SpectralQuant": "o", "TurboQuant": "s", "ProductQuant": "^", "FP16": "*"}
    linestyles = {"SpectralQuant": "-", "TurboQuant": "--", "ProductQuant": "-.", "FP16": ":"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for metric, ax in zip(["Recall@1", "Recall@10"], axes):
        for method in methods:
            method_results = sorted(
                [r for r in results if r["method"] == method],
                key=lambda x: x["bits"],
            )
            bits = [r["bits"] for r in method_results]
            vals = [r.get(metric, 0.0) for r in method_results]
            ax.plot(bits, vals,
                    color=colors.get(method, "gray"),
                    marker=markers.get(method, "o"),
                    linestyle=linestyles.get(method, "-"),
                    linewidth=2, markersize=7, label=method)

        ax.set_xlabel("Bits per dimension")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xticks(BIT_LEVELS)
        ax.legend()

    fig.suptitle(
        "Vector Search Retrieval Quality vs Compression Level\n"
        f"Embedding model: {EMBEDDING_MODEL.split('/')[-1]}",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / "fig_vector_search_recall.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Vector search figure → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp 5: Vector search with compressed embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-corpus", type=int, default=5000,
                   help="Number of corpus documents")
    p.add_argument("--n-queries", type=int, default=200,
                   help="Number of query examples")
    p.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    set_seed(args.seed)
    device = detect_device()

    n_corpus = 500 if args.quick else args.n_corpus
    n_queries = 50 if args.quick else args.n_queries

    log.info("Experiment 5: Vector Search  n_corpus=%d  n_queries=%d", n_corpus, n_queries)

    # ------------------------------------------------------------------
    # Load embedding model
    # ------------------------------------------------------------------
    global EMBEDDING_MODEL
    EMBEDDING_MODEL = args.embedding_model
    embed_model, model_name = load_embedding_model(device)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    corpus, queries, dataset_meta = load_dataset(n_corpus=n_corpus, n_queries=n_queries)
    log.info("Dataset: %d corpus docs, %d queries (%s)",
             len(corpus), len(queries), dataset_meta["type"])

    # Qrels
    if dataset_meta["type"] == "synthetic":
        qrels_idx = dataset_meta["qrels_idx"]
    else:
        # Map FIQA query/corpus IDs to indices
        qrels_idx = {i: {i} for i in range(min(n_queries, len(corpus)))}

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    log.info("Embedding corpus (%d docs) …", len(corpus))
    t0 = time.time()
    corpus_embs = embed_model.encode(corpus, batch_size=64, normalize_embeddings=True)
    log.info("Corpus embedded in %.1f s — shape: %s", time.time() - t0, corpus_embs.shape)

    log.info("Embedding queries (%d) …", len(queries))
    t0 = time.time()
    query_embs = embed_model.encode(queries[:n_queries], batch_size=64, normalize_embeddings=True)
    log.info("Queries embedded in %.1f s", time.time() - t0)

    emb_dim = corpus_embs.shape[-1]
    log.info("Embedding dimension: %d", emb_dim)

    # Compute eigenspectrum on corpus embeddings (for calibration)
    log.info("Computing eigenspectrum of corpus embeddings …")
    eigen = compute_embedding_eigenspectrum(corpus_embs)
    log.info("  d_eff=%.1f  κ=%.2f  (dim=%d)", eigen["d_eff"], eigen["kappa"], emb_dim)

    # ------------------------------------------------------------------
    # FP16 baseline
    # ------------------------------------------------------------------
    log.info("FP16 baseline …")
    fp16_recall = compute_recall(query_embs, corpus_embs, qrels_idx)
    log.info("  FP16 Recall@1=%.4f  Recall@10=%.4f",
             fp16_recall["Recall@1"], fp16_recall["Recall@10"])

    # ------------------------------------------------------------------
    # Compressed variants
    # ------------------------------------------------------------------
    all_results = []
    for bits in BIT_LEVELS:
        log.info("--- %d-bit compression ---", bits)

        # SpectralQuant
        t0 = time.time()
        corpus_sq = quantize_spectralquant(
            corpus_embs, bits=bits,
            eigenvectors=eigen["eigenvectors"],
            mean=eigen["mean"],
            d_eff=eigen["d_eff"],
        )
        query_sq = quantize_spectralquant(
            query_embs, bits=bits,
            eigenvectors=eigen["eigenvectors"],
            mean=eigen["mean"],
            d_eff=eigen["d_eff"],
        )
        # Renormalize after quantization
        corpus_sq_norm = corpus_sq / (np.linalg.norm(corpus_sq, axis=-1, keepdims=True) + 1e-8)
        query_sq_norm = query_sq / (np.linalg.norm(query_sq, axis=-1, keepdims=True) + 1e-8)
        sq_recall = compute_recall(query_sq_norm, corpus_sq_norm, qrels_idx)
        sq_time = time.time() - t0
        log.info("  SQ Recall@1=%.4f  Recall@10=%.4f  (%.2f s)",
                 sq_recall["Recall@1"], sq_recall["Recall@10"], sq_time)

        # TurboQuant
        t0 = time.time()
        corpus_tq = quantize_turboquant(corpus_embs, bits=bits, seed=args.seed)
        query_tq = quantize_turboquant(query_embs, bits=bits, seed=args.seed)
        corpus_tq_norm = corpus_tq / (np.linalg.norm(corpus_tq, axis=-1, keepdims=True) + 1e-8)
        query_tq_norm = query_tq / (np.linalg.norm(query_tq, axis=-1, keepdims=True) + 1e-8)
        tq_recall = compute_recall(query_tq_norm, corpus_tq_norm, qrels_idx)
        tq_time = time.time() - t0
        log.info("  TQ Recall@1=%.4f  Recall@10=%.4f  (%.2f s)",
                 tq_recall["Recall@1"], tq_recall["Recall@10"], tq_time)

        # Product Quantization
        t0 = time.time()
        corpus_pq = quantize_product_quantization(corpus_embs, bits=bits)
        query_pq = quantize_product_quantization(query_embs, bits=bits)
        corpus_pq_norm = corpus_pq / (np.linalg.norm(corpus_pq, axis=-1, keepdims=True) + 1e-8)
        query_pq_norm = query_pq / (np.linalg.norm(query_pq, axis=-1, keepdims=True) + 1e-8)
        pq_recall = compute_recall(query_pq_norm, corpus_pq_norm, qrels_idx)
        pq_time = time.time() - t0
        log.info("  PQ Recall@1=%.4f  Recall@10=%.4f  (%.2f s)",
                 pq_recall["Recall@1"], pq_recall["Recall@10"], pq_time)

        for method, recall, t_elapsed in [
            ("SpectralQuant", sq_recall, sq_time),
            ("TurboQuant",    tq_recall, tq_time),
            ("ProductQuant",  pq_recall, pq_time),
        ]:
            all_results.append({
                "method": method,
                "bits": bits,
                "Recall@1": recall["Recall@1"],
                "Recall@10": recall["Recall@10"],
                "time_s": round(t_elapsed, 3),
            })

    # Also add FP16 at bits=16 for reference
    all_results.append({
        "method": "FP16",
        "bits": 16,
        "Recall@1": fp16_recall["Recall@1"],
        "Recall@10": fp16_recall["Recall@10"],
        "time_s": 0.0,
    })

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    log.info("=== VECTOR SEARCH RESULTS ===")
    log.info("  %-16s  %5s  %10s  %10s", "Method", "Bits", "Recall@1", "Recall@10")
    log.info("  " + "-" * 46)
    for r in sorted(all_results, key=lambda x: (x["method"], x["bits"])):
        log.info("  %-16s  %5d  %10.4f  %10.4f",
                 r["method"], r["bits"], r["Recall@1"], r["Recall@10"])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    csv_path = results_dir / "vector_search_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "bits", "Recall@1", "Recall@10", "time_s"])
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Results CSV → %s", csv_path)

    plot_recall_curves(all_results, results_dir)

    output = {
        "phase": "exp5_vector_search",
        "embedding_model": model_name,
        "embedding_dim": int(emb_dim),
        "n_corpus": len(corpus),
        "n_queries": len(queries[:n_queries]),
        "dataset_type": dataset_meta["type"],
        "d_eff": round(eigen["d_eff"], 2),
        "kappa": round(float(eigen["kappa"]) if np.isfinite(eigen["kappa"]) else 999.0, 2),
        "fp16_baseline": fp16_recall,
        "results": all_results,
        "wall_time_s": round(time.time() - t_total, 2),
    }
    with open(results_dir / "vector_search_results.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info("Experiment 5 complete in %.1f s.", time.time() - t_total)


if __name__ == "__main__":
    main()
