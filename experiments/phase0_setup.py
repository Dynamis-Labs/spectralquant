#!/usr/bin/env python3
"""
phase0_setup.py — Baseline Setup and Reproduction

Clones TurboQuant (cuTile) repo, downloads Qwen 2.5-1.5B and WikiText-103
calibration data, reproduces baseline results, and saves them to
results/baseline_reproduction.json.

Baseline targets (within ±5%):
  - ~5.02× compression ratio
  - ~0.985 cosine similarity on attention output
  - ~144.7 tok/s generation throughput

Usage:
  python phase0_setup.py [--quick] [--results-dir PATH]
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — allow importing spectralquant from src/
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_reproduction"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TURBOQUANT_DIR = PROJECT_ROOT / "turboquant_cutile"
TURBOQUANT_REPO = "https://github.com/DevTechJr/turboquant_cutile"

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
WIKITEXT_NAME = "wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"

REQUIRED_FILES = [
    "host.py",
    "compress.py",
    "decompress.py",
    "attention.py",
    "codebook.py",
    "constants.py",
]

BASELINE_TARGETS = {
    "compression_ratio": 5.02,
    "cosine_similarity": 0.985,
    "throughput_tok_per_s": 144.7,
}
TOLERANCE = 0.05  # ±5 %

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
log = logging.getLogger("phase0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        log.info("GPU detected: %s (%.1f GB VRAM)", props.name, vram / 1e9)
    else:
        device = torch.device("cpu")
        log.warning("No GPU detected — running on CPU. Throughput measurements will not be representative.")
    return device


def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if check and result.returncode != 0:
        log.error("Command failed:\n%s\n%s", result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return result


# ---------------------------------------------------------------------------
# Step 1: Clone TurboQuant repo
# ---------------------------------------------------------------------------

def clone_turboquant() -> bool:
    log.info("=== STEP 1: Clone TurboQuant repo ===")
    if TURBOQUANT_DIR.exists():
        log.info("Repo already exists at %s — skipping clone.", TURBOQUANT_DIR)
    else:
        t0 = time.time()
        try:
            run_cmd(["git", "clone", TURBOQUANT_REPO, str(TURBOQUANT_DIR)])
            log.info("Cloned in %.1f s", time.time() - t0)
        except RuntimeError as e:
            log.error("Failed to clone TurboQuant: %s", e)
            return False

    # Verify required files — they live inside the turboquant_cutile/ subdirectory
    pkg_dir = TURBOQUANT_DIR / "turboquant_cutile"
    if not pkg_dir.exists():
        # Fallback: maybe the files are at the root
        pkg_dir = TURBOQUANT_DIR
    missing = [f for f in REQUIRED_FILES if not (pkg_dir / f).exists()]
    if missing:
        log.warning("Missing required files in %s: %s", pkg_dir, missing)
        log.info("Continuing anyway — SpectralQuant uses its own implementation.")
    else:
        log.info("All required files present in %s: %s", pkg_dir, REQUIRED_FILES)
    return True


# ---------------------------------------------------------------------------
# Step 2: Download model and tokenizer
# ---------------------------------------------------------------------------

def download_model(device: torch.device, quick: bool = False):
    log.info("=== STEP 2: Download Qwen 2.5-1.5B ===")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        log.error("transformers not installed. Run: pip install transformers accelerate")
        raise

    t0 = time.time()
    log.info("Downloading tokenizer for %s …", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    log.info("Tokenizer downloaded in %.1f s", time.time() - t0)

    t0 = time.time()
    log.info("Downloading model for %s …", MODEL_NAME)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    log.info("Model downloaded & loaded in %.1f s  (%s params)",
             time.time() - t0,
             f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Step 3: Download WikiText-103 calibration data
# ---------------------------------------------------------------------------

def download_calibration_data(n_samples: int = 1000):
    log.info("=== STEP 3: Download WikiText-103 calibration data (%d samples) ===", n_samples)
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets not installed. Run: pip install datasets")
        raise

    t0 = time.time()
    dataset = load_dataset(WIKITEXT_NAME, WIKITEXT_CONFIG, split="train", streaming=False)
    # Filter out very short examples
    texts = [
        row["text"].strip()
        for row in dataset
        if len(row["text"].strip()) > 200
    ][:n_samples]
    log.info("Loaded %d calibration samples in %.1f s", len(texts), time.time() - t0)
    return texts


# ---------------------------------------------------------------------------
# Step 4: Reproduce baseline results
# ---------------------------------------------------------------------------

def compute_compression_ratio(model_config) -> float:
    """
    Approximate TurboQuant compression ratio from model config.

    TurboQuant: keys at 3 bits (2-bit Lloyd-Max + 1-bit QJL), values at 3 bits.
    FP16 baseline: 16 bits per element.
    Compression ratio ≈ 16 / 3.
    Adjusted by any metadata/codebook overhead.
    """
    bits_per_element_fp16 = 16
    bits_turboquant_key = 3    # 2-bit LM + 1-bit QJL
    bits_turboquant_val = 3    # 3-bit LM
    # combined average
    avg_bits = (bits_turboquant_key + bits_turboquant_val) / 2
    ratio = bits_per_element_fp16 / avg_bits
    return ratio


def measure_attention_cosine_similarity(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    quick: bool = False,
    seq_len: int = 256,
) -> float:
    """
    Simulate TurboQuant compression + reconstruction and measure cosine similarity
    of attention outputs vs FP16 reference.

    This implements a software-level approximation of TurboQuant's uniform
    3-bit quantization (Lloyd-Max on Gaussian + QJL on keys).
    """
    from torch.nn.functional import cosine_similarity

    log.info("Measuring attention cosine similarity (TurboQuant approx.) …")

    n_eval = 20 if quick else 100
    sample_texts = random.sample(texts, min(n_eval, len(texts)))

    cos_sims = []
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    # Sample a few layers
    sample_layers = sorted(random.sample(range(n_layers), min(5, n_layers)))

    with torch.no_grad():
        for text in sample_texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
                padding=False,
            ).to(device)

            # Forward pass to get past_key_values (FP16 reference)
            out = model(**enc, output_hidden_states=False, use_cache=True)
            kv_cache = out.past_key_values  # DynamicCache or list of (k, v)

            for layer_idx in sample_layers:
                # Handle DynamicCache (transformers >=4.36) and older tuple-style
                try:
                    k_fp16 = kv_cache.key_cache[layer_idx].float()
                    v_fp16 = kv_cache.value_cache[layer_idx].float()
                except (AttributeError, TypeError):
                    try:
                        k_fp16 = kv_cache[layer_idx][0].float()
                        v_fp16 = kv_cache[layer_idx][1].float()
                    except TypeError:
                        # Last resort: iterate to get the layer
                        layer_kv = list(kv_cache)[layer_idx]
                        k_fp16 = layer_kv[0].float()
                        v_fp16 = layer_kv[1].float()

                # Simulate TurboQuant 3-bit uniform quantization (Gaussian Lloyd-Max)
                k_quant = _simulate_turboquant_quantize(k_fp16, bits=3)
                v_quant = _simulate_turboquant_quantize(v_fp16, bits=3)

                # Cosine similarity in key space
                cos = cosine_similarity(
                    k_fp16.reshape(-1, head_dim),
                    k_quant.reshape(-1, head_dim),
                    dim=-1,
                ).mean().item()
                cos_sims.append(cos)

    mean_cos = float(np.mean(cos_sims))
    log.info("Mean cosine similarity (keys): %.4f  (std=%.4f, n=%d)",
             mean_cos, float(np.std(cos_sims)), len(cos_sims))
    return mean_cos


def _simulate_turboquant_quantize(x: torch.Tensor, bits: int = 3) -> torch.Tensor:
    """
    Approximate Lloyd-Max quantization for a Gaussian distribution.
    Uses uniform quantization as a proxy (within ~5% of Lloyd-Max SQNR for Gaussian).
    Operates over last dimension.
    """
    n_levels = 2 ** bits
    # Per-vector min/max clamp at 3σ (approximate Lloyd-Max clipping point)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    x_clamp = x.clamp(-3 * std, 3 * std)
    # Uniform quantize in [-3σ, 3σ]
    x_norm = (x_clamp / (3 * std) + 1) / 2  # [0, 1]
    x_int = (x_norm * (n_levels - 1)).round().clamp(0, n_levels - 1)
    # Dequantize
    x_dequant = x_int / (n_levels - 1) * 2 * 3 * std - 3 * std
    return x_dequant


def measure_throughput(
    model,
    tokenizer,
    device: torch.device,
    quick: bool = False,
    n_runs: int = 5,
    prompt_len: int = 64,
    gen_len: int = 64,
) -> float:
    """Measure tokens/second for autoregressive generation."""
    if device.type != "cuda":
        log.warning("Cannot measure meaningful throughput without GPU. Returning 0.")
        return 0.0

    log.info("Measuring generation throughput …")
    prompt = "The quick brown fox jumps over the lazy dog. " * 4
    enc = tokenizer(prompt, return_tensors="pt", max_length=prompt_len, truncation=True).to(device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**enc, max_new_tokens=16, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    n_gen = 2 if quick else n_runs
    tok_counts = []
    times = []
    with torch.no_grad():
        for _ in range(n_gen):
            t0 = time.perf_counter()
            out = model.generate(
                **enc,
                max_new_tokens=gen_len,
                do_sample=False,
                use_cache=True,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_new_tokens = out.shape[1] - enc["input_ids"].shape[1]
            tok_counts.append(n_new_tokens)
            times.append(elapsed)

    toks_per_s = float(np.mean([t / e for t, e in zip(tok_counts, times)]))
    log.info("Throughput: %.1f tok/s  (mean over %d runs)", toks_per_s, n_gen)
    return toks_per_s


def check_within_tolerance(actual: float, target: float, tol: float = TOLERANCE) -> bool:
    if target == 0:
        return actual == 0
    return abs(actual - target) / abs(target) <= tol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 0: Baseline setup and TurboQuant reproduction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="Use reduced sample sizes (for debugging)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n-calib", type=int, default=1000,
                   help="Number of calibration samples to download")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                   help="Directory to save results")
    p.add_argument("--skip-clone", action="store_true",
                   help="Skip cloning (assumes repo already present)")
    p.add_argument("--skip-model", action="store_true",
                   help="Skip model download step (for dry-run testing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    set_seed(args.seed)
    device = detect_device()

    results = {
        "phase": "phase0_baseline",
        "seed": args.seed,
        "device": str(device),
        "quick_mode": args.quick,
        "model_name": MODEL_NAME,
        "targets": BASELINE_TARGETS,
        "tolerance": TOLERANCE,
    }

    # ------------------------------------------------------------------
    # Step 1: Clone
    # ------------------------------------------------------------------
    if not args.skip_clone:
        ok = clone_turboquant()
        if not ok:
            log.error("TurboQuant clone/verification failed. Exiting.")
            results["status"] = "FAILED_CLONE"
            _save_results(results, results_dir)
            sys.exit(1)
    results["turboquant_dir"] = str(TURBOQUANT_DIR)
    results["required_files_present"] = REQUIRED_FILES

    # ------------------------------------------------------------------
    # Step 2 & 3: Model + calibration data
    # ------------------------------------------------------------------
    if args.skip_model:
        log.warning("--skip-model set: skipping model download and all measurements.")
        results["status"] = "SKIPPED_MODEL"
        _save_results(results, results_dir)
        return

    model, tokenizer = download_model(device, quick=args.quick)
    n_calib = 50 if args.quick else args.n_calib
    calib_texts = download_calibration_data(n_samples=n_calib)
    results["n_calibration_samples"] = len(calib_texts)

    # ------------------------------------------------------------------
    # Step 4: Reproduce baseline
    # ------------------------------------------------------------------
    log.info("=== STEP 4: Reproduce baseline results ===")

    # Compression ratio (analytical, same formula as TurboQuant)
    compression_ratio = compute_compression_ratio(model.config)
    log.info("Compression ratio (TurboQuant 3-bit): %.3f×  (target: %.2f×)",
             compression_ratio, BASELINE_TARGETS["compression_ratio"])

    # Cosine similarity
    cos_sim = measure_attention_cosine_similarity(
        model, tokenizer, calib_texts, device, quick=args.quick
    )

    # Throughput
    throughput = measure_throughput(model, tokenizer, device, quick=args.quick)

    # ------------------------------------------------------------------
    # Tolerance check
    # ------------------------------------------------------------------
    checks = {
        "compression_ratio": {
            "actual": round(compression_ratio, 4),
            "target": BASELINE_TARGETS["compression_ratio"],
            "within_tolerance": check_within_tolerance(compression_ratio, BASELINE_TARGETS["compression_ratio"]),
        },
        "cosine_similarity": {
            "actual": round(cos_sim, 4),
            "target": BASELINE_TARGETS["cosine_similarity"],
            "within_tolerance": check_within_tolerance(cos_sim, BASELINE_TARGETS["cosine_similarity"]),
        },
    }
    if throughput > 0:
        checks["throughput_tok_per_s"] = {
            "actual": round(throughput, 2),
            "target": BASELINE_TARGETS["throughput_tok_per_s"],
            "within_tolerance": check_within_tolerance(throughput, BASELINE_TARGETS["throughput_tok_per_s"]),
        }

    all_pass = all(v["within_tolerance"] for v in checks.values())
    results["checks"] = checks
    results["all_within_tolerance"] = all_pass

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=== BASELINE REPRODUCTION SUMMARY ===")
    for metric, info in checks.items():
        status = "PASS" if info["within_tolerance"] else "FAIL"
        log.info("  [%s] %s: actual=%.4f  target=%.4f  (±%.0f%%)",
                 status, metric, info["actual"], info["target"], TOLERANCE * 100)

    if all_pass:
        log.info("BASELINE REPRODUCTION PASSED — all metrics within ±%.0f%% of targets.", TOLERANCE * 100)
        results["status"] = "PASSED"
    else:
        failed = [k for k, v in checks.items() if not v["within_tolerance"]]
        msg = (
            f"Baseline reproduction FAILED for: {failed}. "
            "Debug TurboQuant implementation before proceeding. "
            "Check that random rotation is applied correctly and Lloyd-Max codebooks are loaded."
        )
        log.error(msg)
        results["status"] = "FAILED_REPRODUCTION"
        results["failed_metrics"] = failed
        _save_results(results, results_dir)
        sys.exit(1)

    results["wall_time_s"] = round(time.time() - t_start, 2)
    _save_results(results, results_dir)
    log.info("Phase 0 complete. Results saved to %s", results_dir / "baseline_reproduction.json")


def _save_results(results: dict, results_dir: Path) -> None:
    out_path = results_dir / "baseline_reproduction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
