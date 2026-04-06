#!/usr/bin/env python3
"""
phase2_integration.py — SpectralQuant Implementation Integration

Loads Phase 1 calibration data (eigenvectors, eigenvalues), constructs
SpectralQuantEngine instances for each head, and verifies:
  1. Spectral rotation preserves dot products (V^T is orthogonal)
  2. Non-uniform bit allocation respects the total bit budget
  3. Selective QJL uses fewer bits than full QJL

All integration test results are saved to results/integration/.

Usage:
  python phase2_integration.py [--quick] [--calib-dir PATH]
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results" / "integration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CALIB_DIR = PROJECT_ROOT / "results" / "eigenspectral" / "calibration"

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

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
log = logging.getLogger("phase2")


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
        log.info("GPU: %s (%.1f GB VRAM)", props.name, (getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)) / 1e9)
    else:
        device = torch.device("cpu")
        log.warning("No GPU — running on CPU.")
    return device


# ---------------------------------------------------------------------------
# Load Phase 1 calibration data
# ---------------------------------------------------------------------------

def load_calibration(calib_dir: Path) -> dict:
    """
    Returns dict: calibration[layer][head] = {
      'key_eigenvectors': np.ndarray [head_dim, head_dim],
      'key_eigenvalues':  np.ndarray [head_dim],
      'key_mean':         np.ndarray [head_dim],
      'key_d_eff':        float,
      'key_kappa':        float,
      'val_*':            same structure,
    }
    """
    meta_path = calib_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Calibration metadata not found at {meta_path}. "
            "Run phase1_eigenspectral.py first."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    n_layers = meta["n_layers"]
    n_kv_heads = meta["n_kv_heads"]
    log.info("Loading calibration: %d layers × %d KV heads from %s",
             n_layers, n_kv_heads, calib_dir)

    calibration = {}
    for l in range(n_layers):
        calibration[l] = {}
        for h in range(n_kv_heads):
            fname = calib_dir / f"layer{l:02d}_head{h:02d}.npz"
            data = np.load(str(fname))
            calibration[l][h] = {
                "key_eigenvectors": data["key_eigenvectors"],
                "key_eigenvalues":  data["key_eigenvalues"],
                "key_mean":         data["key_mean"],
                "key_d_eff":        float(data["key_d_eff"]),
                "key_kappa":        float(data["key_kappa"]),
                "val_eigenvectors": data["val_eigenvectors"],
                "val_eigenvalues":  data["val_eigenvalues"],
                "val_mean":         data["val_mean"],
                "val_d_eff":        float(data["val_d_eff"]),
                "val_kappa":        float(data["val_kappa"]),
            }

    log.info("Calibration loaded (%d layer-head pairs)", n_layers * n_kv_heads)
    return calibration, meta


# ---------------------------------------------------------------------------
# SpectralQuant Engine (pure Python / PyTorch)
# ---------------------------------------------------------------------------

class SpectralQuantEngine:
    """
    Per-head SpectralQuant compressor/decompressor.

    Implements:
      - Spectral rotation:  x_rot = V^T @ (x - mean)
      - Non-uniform bit allocation:  b_high for top d_eff dims, b_low for tail
      - Selective QJL:  QJL sign correction applied to top d_eff dims only

    Parameters
    ----------
    eigenvectors : np.ndarray, shape [head_dim, head_dim]
        Columns are eigenvectors sorted by decreasing eigenvalue.
    eigenvalues : np.ndarray, shape [head_dim]
        Sorted descending.
    mean : np.ndarray, shape [head_dim]
    d_eff : float
        Effective dimensionality (participation ratio).
    avg_bits : float
        Target average bits per coordinate (e.g., 3.0).
    b_high : int
        Bits for semantic (top d_eff) coordinates. If None, computed automatically.
    b_low : int
        Bits for tail coordinates. If None, computed automatically.
    """

    def __init__(
        self,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        mean: np.ndarray,
        d_eff: float,
        avg_bits: float = 3.0,
        b_high: int | None = None,
        b_low: int | None = None,
    ):
        self.V = torch.from_numpy(eigenvectors).float()  # [head_dim, head_dim]
        self.eigenvalues = torch.from_numpy(eigenvalues).float()
        self.mean = torch.from_numpy(mean).float()
        self.head_dim = eigenvectors.shape[0]
        self.d_eff = d_eff
        self.d_sem = max(1, int(round(d_eff)))  # semantic regime cutoff
        self.avg_bits = avg_bits

        # Determine bit allocation
        if b_high is not None and b_low is not None:
            self.b_high = b_high
            self.b_low = b_low
        else:
            self.b_high, self.b_low = self._compute_bit_allocation(avg_bits)

        # Verify budget
        total = self.d_sem * self.b_high + (self.head_dim - self.d_sem) * self.b_low
        budget = self.head_dim * avg_bits
        self._budget_error = abs(total - budget) / budget

        log.debug(
            "SpectralQuantEngine: head_dim=%d d_eff=%.1f d_sem=%d "
            "avg_bits=%.1f b_high=%d b_low=%d  budget_err=%.2f%%",
            self.head_dim, d_eff, self.d_sem, avg_bits,
            self.b_high, self.b_low, self._budget_error * 100,
        )

    def _compute_bit_allocation(self, avg_bits: float) -> tuple[int, int]:
        """
        Solve for (b_high, b_low) integer pair that minimises budget error
        with b_high > b_low >= 1.

        Strategy:
          b_high = ceil(avg_bits + 1)
          solve for b_low from budget constraint (continuous), then round.
        """
        d, d_sem = self.head_dim, self.d_sem
        budget = d * avg_bits

        best_err = float("inf")
        best_bh, best_bl = 4, 2

        for bh in range(2, 9):
            # Continuous b_low that satisfies budget
            bl_cont = (budget - d_sem * bh) / (d - d_sem) if d > d_sem else bh
            for bl in [max(1, int(bl_cont)), max(1, int(bl_cont) + 1)]:
                if bl >= bh:
                    continue
                err = abs(d_sem * bh + (d - d_sem) * bl - budget)
                if err < best_err:
                    best_err = err
                    best_bh, best_bl = bh, bl

        return best_bh, best_bl

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spectral rotation: x_rot = V^T @ (x - mean)
        x: [..., head_dim]
        Returns: [..., head_dim]
        """
        V = self.V.to(x.device)
        mean = self.mean.to(x.device)
        return (x - mean) @ V  # [..., head_dim]  (equivalent to V^T @ (x - mean) row-wise)

    def unrotate(self, x_rot: torch.Tensor) -> torch.Tensor:
        """
        Inverse rotation: x = V @ x_rot + mean
        """
        V = self.V.to(x_rot.device)
        mean = self.mean.to(x_rot.device)
        return x_rot @ V.T + mean

    def quantize(self, x_rot: torch.Tensor, modality: str = "key") -> torch.Tensor:
        """
        Non-uniform quantization:
          - Top d_sem dimensions: b_high bits
          - Remaining dimensions: b_low bits
        """
        x_sem = x_rot[..., :self.d_sem]
        x_tail = x_rot[..., self.d_sem:]

        x_sem_q = _uniform_quantize(x_sem, self.b_high)
        x_tail_q = _uniform_quantize(x_tail, self.b_low)

        return torch.cat([x_sem_q, x_tail_q], dim=-1)

    def dequantize(self, x_q: torch.Tensor) -> torch.Tensor:
        """Dequantize (no-op in current approximation since quantize/dequantize is paired)."""
        return x_q

    def compress(self, x: torch.Tensor, modality: str = "key") -> dict:
        """Full compression pipeline: rotate → quantize → (selective QJL metadata)."""
        x_rot = self.rotate(x)
        x_q = self.quantize(x_rot, modality=modality)
        # Selective QJL: sign bits for top d_sem coordinates
        # (in full implementation this is a 1-bit correction per semantic dim)
        qjl_signs_sem = torch.sign(x_rot[..., :self.d_sem])
        return {
            "x_quantized": x_q,          # full rotated-quantized vector
            "qjl_signs":   qjl_signs_sem, # only d_sem sign bits (selective)
            "d_sem":       self.d_sem,
        }

    def decompress(self, compressed: dict) -> torch.Tensor:
        """Reconstruct vector from compressed representation."""
        x_rot_recon = compressed["x_quantized"]
        return self.unrotate(x_rot_recon)

    def bits_per_vector(self) -> float:
        """Actual bits per head_dim element (including selective QJL)."""
        # Semantic: b_high + 1 bit QJL
        # Tail: b_low (no QJL)
        sem_bits = self.d_sem * (self.b_high + 1)
        tail_bits = (self.head_dim - self.d_sem) * self.b_low
        return (sem_bits + tail_bits) / self.head_dim

    def budget_error_pct(self) -> float:
        return self._budget_error * 100


class TurboQuantBaseline:
    """
    TurboQuant baseline: random rotation + uniform b-bit quantization + full QJL.
    """

    def __init__(self, head_dim: int, avg_bits: float = 3.0, seed: int = 42):
        self.head_dim = head_dim
        self.avg_bits = avg_bits
        self.bits = int(round(avg_bits))

        # Generate random orthogonal matrix (random rotation)
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((head_dim, head_dim)).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        self.Pi = torch.from_numpy(Q)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        Pi = self.Pi.to(x.device)
        return x @ Pi  # random rotation

    def unrotate(self, x_rot: torch.Tensor) -> torch.Tensor:
        Pi = self.Pi.to(x_rot.device)
        return x_rot @ Pi.T

    def compress(self, x: torch.Tensor) -> dict:
        x_rot = self.rotate(x)
        x_q = _uniform_quantize(x_rot, self.bits)
        qjl_signs = torch.sign(x_rot)  # full d-dim QJL
        return {"x_quantized": x_q, "qjl_signs": qjl_signs}

    def decompress(self, compressed: dict) -> torch.Tensor:
        return self.unrotate(compressed["x_quantized"])

    def bits_per_vector(self) -> float:
        return self.avg_bits + 1.0  # uniform b bits + 1 bit QJL per coordinate


def _uniform_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-vector range uniform quantization (proxy for Lloyd-Max on Gaussian)."""
    n_levels = 2 ** bits
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    x_clamp = x.clamp(-3 * std, 3 * std)
    x_norm = (x_clamp / (3 * std) + 1) / 2
    x_int = (x_norm * (n_levels - 1)).round().clamp(0, n_levels - 1)
    x_dequant = x_int / (n_levels - 1) * 2 * 3 * std - 3 * std
    return x_dequant


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_orthogonality(engine: SpectralQuantEngine, n_pairs: int = 100) -> dict:
    """
    Verify that spectral rotation V^T is orthogonal:
      For random vectors u, v: <V^T u, V^T v> ≈ <u, v>
    """
    d = engine.head_dim
    results = []
    for _ in range(n_pairs):
        u = torch.randn(d)
        v = torch.randn(d)
        dot_orig = float((u * v).sum())
        u_rot = engine.rotate(u.unsqueeze(0)).squeeze(0)
        v_rot = engine.rotate(v.unsqueeze(0)).squeeze(0)
        dot_rot = float((u_rot * v_rot).sum())
        results.append(abs(dot_rot - dot_orig) / (abs(dot_orig) + 1e-8))

    return {
        "test": "orthogonality",
        "mean_relative_error": float(np.mean(results)),
        "max_relative_error": float(np.max(results)),
        "passed": float(np.max(results)) < 1e-4,
    }


def test_budget_constraint(engine: SpectralQuantEngine) -> dict:
    """
    Verify that bit allocation respects budget:
      d_sem * b_high + (d - d_sem) * b_low ≈ d * avg_bits
    """
    d = engine.head_dim
    d_sem = engine.d_sem
    allocated = d_sem * engine.b_high + (d - d_sem) * engine.b_low
    budget = d * engine.avg_bits
    error_pct = abs(allocated - budget) / budget * 100

    return {
        "test": "budget_constraint",
        "d_sem": d_sem,
        "b_high": engine.b_high,
        "b_low": engine.b_low,
        "avg_bits": engine.avg_bits,
        "allocated_bits": float(allocated),
        "budget_bits": float(budget),
        "error_pct": round(float(error_pct), 3),
        "passed": error_pct < 25,  # within 25% — integer rounding causes some slack
    }


def test_selective_qjl(
    engine: SpectralQuantEngine,
    turbo: TurboQuantBaseline,
) -> dict:
    """
    Verify that selective QJL (applied to d_sem dims) uses fewer bits than full QJL
    (applied to all d dims).
    """
    d = engine.head_dim
    d_sem = engine.d_sem

    # Full QJL: d bits
    # Selective QJL: d_sem bits
    full_qjl_bits_per_vec = float(d)
    selective_qjl_bits_per_vec = float(d_sem)
    saving = full_qjl_bits_per_vec - selective_qjl_bits_per_vec
    saving_pct = saving / full_qjl_bits_per_vec * 100

    x = torch.randn(100, d)
    compressed_sq = engine.compress(x)
    compressed_tq = turbo.compress(x)

    sq_qjl_bits = compressed_sq["qjl_signs"].shape[-1]
    tq_qjl_bits = compressed_tq["qjl_signs"].shape[-1]

    return {
        "test": "selective_qjl",
        "head_dim": d,
        "d_sem": d_sem,
        "spectralquant_qjl_bits_per_token": sq_qjl_bits,
        "turboquant_qjl_bits_per_token": tq_qjl_bits,
        "qjl_bit_saving": tq_qjl_bits - sq_qjl_bits,
        "qjl_bit_saving_pct": round(saving_pct, 2),
        "passed": sq_qjl_bits < tq_qjl_bits,
    }


def test_reconstruction_quality(
    engine: SpectralQuantEngine,
    turbo: TurboQuantBaseline,
    n_vectors: int = 1000,
) -> dict:
    """
    Compare reconstruction quality (cosine similarity) of SpectralQuant vs TurboQuant
    on random vectors drawn from approximately the true distribution.
    """
    from torch.nn.functional import cosine_similarity

    d = engine.head_dim
    # Simulate distribution: anisotropic Gaussian aligned with eigenvectors
    # (higher variance on top dimensions)
    std_vec = (engine.eigenvalues.float().clamp(min=1e-6) ** 0.5).to("cpu")
    x_rot_true = torch.randn(n_vectors, d) * std_vec.unsqueeze(0)
    # Rotate back to original space
    x_orig = engine.unrotate(x_rot_true)

    # SpectralQuant
    comp_sq = engine.compress(x_orig)
    x_recon_sq = engine.decompress(comp_sq)
    cos_sq = cosine_similarity(x_orig, x_recon_sq, dim=-1).mean().item()

    # TurboQuant
    comp_tq = turbo.compress(x_orig)
    x_recon_tq = turbo.decompress(comp_tq)
    cos_tq = cosine_similarity(x_orig, x_recon_tq, dim=-1).mean().item()

    return {
        "test": "reconstruction_quality",
        "n_vectors": n_vectors,
        "spectralquant_cosine_sim": round(float(cos_sq), 5),
        "turboquant_cosine_sim": round(float(cos_tq), 5),
        "delta_cosine_sim": round(float(cos_sq - cos_tq), 5),
        "spectralquant_wins": cos_sq >= cos_tq,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2: SpectralQuant integration tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quick", action="store_true", help="Test fewer layers/heads")
    p.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR,
                   help="Phase 1 calibration directory")
    p.add_argument("--avg-bits", type=float, default=3.0,
                   help="Average bits for quantization")
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

    log.info("Phase 2 — Integration Tests")
    log.info("  calib_dir=%s  avg_bits=%.1f", args.calib_dir, args.avg_bits)

    # ------------------------------------------------------------------
    # Load calibration
    # ------------------------------------------------------------------
    calibration, meta = load_calibration(args.calib_dir)
    n_layers = meta["n_layers"]
    n_kv_heads = meta["n_kv_heads"]

    # Quick mode: test first layer only
    layers_to_test = [0] if args.quick else list(range(n_layers))
    heads_to_test = [0] if args.quick else list(range(n_kv_heads))

    all_test_results = []
    aggregate = {
        "orthogonality_pass_rate": [],
        "budget_pass_rate": [],
        "selective_qjl_pass_rate": [],
        "sq_cosine_sims": [],
        "tq_cosine_sims": [],
        "qjl_savings_pct": [],
    }

    log.info("Running integration tests on %d layers × %d heads …",
             len(layers_to_test), len(heads_to_test))

    for l in layers_to_test:
        for h in heads_to_test:
            c = calibration[l][h]
            head_dim = c["key_eigenvectors"].shape[0]

            engine = SpectralQuantEngine(
                eigenvectors=c["key_eigenvectors"],
                eigenvalues=c["key_eigenvalues"],
                mean=c["key_mean"],
                d_eff=c["key_d_eff"],
                avg_bits=args.avg_bits,
            )
            turbo = TurboQuantBaseline(
                head_dim=head_dim,
                avg_bits=args.avg_bits,
                seed=args.seed,
            )

            t1 = test_orthogonality(engine, n_pairs=50 if args.quick else 200)
            t2 = test_budget_constraint(engine)
            t3 = test_selective_qjl(engine, turbo)
            t4 = test_reconstruction_quality(
                engine, turbo, n_vectors=100 if args.quick else 500
            )

            entry = {
                "layer": l,
                "head": h,
                "d_eff": c["key_d_eff"],
                "b_high": engine.b_high,
                "b_low": engine.b_low,
                "bits_per_vector_sq": engine.bits_per_vector(),
                "bits_per_vector_tq": turbo.bits_per_vector(),
                "tests": {
                    "orthogonality": t1,
                    "budget_constraint": t2,
                    "selective_qjl": t3,
                    "reconstruction_quality": t4,
                },
            }
            all_test_results.append(entry)

            aggregate["orthogonality_pass_rate"].append(t1["passed"])
            aggregate["budget_pass_rate"].append(t2["passed"])
            aggregate["selective_qjl_pass_rate"].append(t3["passed"])
            aggregate["sq_cosine_sims"].append(t4["spectralquant_cosine_sim"])
            aggregate["tq_cosine_sims"].append(t4["turboquant_cosine_sim"])
            aggregate["qjl_savings_pct"].append(t3["qjl_bit_saving_pct"])

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    log.info("=== INTEGRATION TEST SUMMARY ===")
    log.info("  Orthogonality:    %.0f%% PASS (max relative error)",
             100 * np.mean(aggregate["orthogonality_pass_rate"]))
    log.info("  Budget constraint: %.0f%% PASS",
             100 * np.mean(aggregate["budget_pass_rate"]))
    log.info("  Selective QJL:    %.0f%% PASS (fewer bits than full QJL)",
             100 * np.mean(aggregate["selective_qjl_pass_rate"]))
    log.info("  Reconstruction cosine sim:")
    log.info("    SpectralQuant: %.4f ± %.4f",
             np.mean(aggregate["sq_cosine_sims"]),
             np.std(aggregate["sq_cosine_sims"]))
    log.info("    TurboQuant:    %.4f ± %.4f",
             np.mean(aggregate["tq_cosine_sims"]),
             np.std(aggregate["tq_cosine_sims"]))
    log.info("  QJL bit savings:  %.1f%% on average",
             np.mean(aggregate["qjl_savings_pct"]))

    all_pass = (
        np.mean(aggregate["orthogonality_pass_rate"]) > 0.95
        and np.mean(aggregate["budget_pass_rate"]) > 0.90
        and np.mean(aggregate["selective_qjl_pass_rate"]) > 0.95
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    summary = {
        "phase": "phase2_integration",
        "avg_bits": args.avg_bits,
        "layers_tested": len(layers_to_test),
        "heads_tested": len(heads_to_test),
        "all_tests_pass": all_pass,
        "aggregate": {
            k: float(np.mean(v)) if v else None
            for k, v in aggregate.items()
        },
        "wall_time_s": round(time.time() - t_total, 2),
    }

    out_path = results_dir / "integration_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_head": all_test_results[:50]}, f, indent=2)
    log.info("Results saved → %s", out_path)

    if all_pass:
        log.info("INTEGRATION TESTS PASSED — SpectralQuantEngine ready for Phase 3.")
    else:
        log.warning("Some integration tests failed. Check results for details before Phase 3.")

    log.info("Phase 2 complete in %.1f s.", time.time() - t_total)


if __name__ == "__main__":
    main()
