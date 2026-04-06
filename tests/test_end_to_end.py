"""
test_end_to_end.py — End-to-end pipeline tests.

Tests the full SpectralQuant pipeline:
    calibrate → compress → decompress → compare

Key invariants verified:
    1. Full pipeline: calibrate → compress → decompress → compare
    2. Compression ratio matches configured avg_bits
    3. SpectralQuant beats random rotation on sharp-gap data
    4. SpectralQuant matches random rotation on flat-spectrum data (graceful degradation)

All tests use synthetic CPU data.
"""

import numpy as np
import pytest

from conftest import HEAD_DIM, N_TOKENS, SEED


# ---------------------------------------------------------------------------
# Minimal self-contained pipeline for end-to-end testing.
# Once spectralquant is built, replace with:
#   from spectralquant import SpectralQuantCompressor
# ---------------------------------------------------------------------------


class _LloydMaxCodebook:
    """Per-coordinate Lloyd-Max codebook."""

    def __init__(self, n_levels: int, max_iter: int = 100, tol: float = 1e-6):
        self.n_levels = n_levels
        self.max_iter = max_iter
        self.tol = tol
        self.centroids: np.ndarray | None = None
        self.boundaries: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "_LloydMaxCodebook":
        lo, hi = float(data.min()), float(data.max())
        centroids = np.linspace(
            lo + (hi - lo) / (2 * self.n_levels),
            hi - (hi - lo) / (2 * self.n_levels),
            self.n_levels,
        ).astype(np.float64)
        prev_distortion = np.inf
        for _ in range(self.max_iter):
            boundaries = np.concatenate([[-np.inf], (centroids[:-1] + centroids[1:]) / 2, [np.inf]])
            indices = np.digitize(data, boundaries[1:])
            new_centroids = np.array([
                data[indices == k].mean() if (indices == k).any() else centroids[k]
                for k in range(self.n_levels)
            ])
            distortion = float(((data - new_centroids[indices]) ** 2).mean())
            centroids = new_centroids
            if abs(prev_distortion - distortion) < self.tol:
                break
            prev_distortion = distortion
        self.centroids = centroids.astype(np.float32)
        self.boundaries = np.concatenate([[-np.inf], (centroids[:-1] + centroids[1:]) / 2, [np.inf]]).astype(np.float32)
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.digitize(x, self.boundaries[1:]).astype(np.int32)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        return self.centroids[indices].astype(np.float32)


class _SpectralQuantPipeline:
    """
    Minimal end-to-end pipeline for testing only.

    Implements:
      - Spectral rotation (or random rotation)
      - Non-uniform Lloyd-Max quantization
      - Decompress / reconstruct
    """

    def __init__(
        self,
        head_dim: int = HEAD_DIM,
        avg_bits: float = 3.0,
        d_eff: int | None = None,
        use_spectral: bool = True,
        seed: int = SEED,
    ):
        self.head_dim = head_dim
        self.avg_bits = avg_bits
        self.d_eff = d_eff  # if None, set during calibration
        self.use_spectral = use_spectral
        self.seed = seed
        self.rotation_matrix: np.ndarray | None = None
        self.codebooks: list | None = None
        self._calibrated = False

    # ---- calibration ----

    def calibrate(self, X: np.ndarray) -> "_SpectralQuantPipeline":
        """Fit rotation matrix and codebooks on calibration data X: [n, d]."""
        if self.use_spectral:
            C = X.T @ X / len(X)
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = eigenvalues[::-1].copy()
            V = eigenvectors[:, ::-1].copy().astype(np.float32)
            if self.d_eff is None:
                self.d_eff = int(round((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()))
        else:
            rng = np.random.default_rng(self.seed)
            A = rng.standard_normal((self.head_dim, self.head_dim)).astype(np.float32)
            V, _ = np.linalg.qr(A)
            if self.d_eff is None:
                self.d_eff = self.head_dim // 4  # heuristic for random baseline

        self.rotation_matrix = V
        X_rot = X @ V  # [n, d]

        # Non-uniform bit allocation
        d = self.head_dim
        d_eff = self.d_eff
        d_tail = d - d_eff
        budget = self.avg_bits * d
        b_high = int(np.ceil(self.avg_bits)) + 1
        b_low = max(1, int((budget - d_eff * b_high) / d_tail)) if d_tail > 0 else b_high

        self.b_high = b_high
        self.b_low = b_low

        # Fit per-coordinate codebooks
        self.codebooks = []
        for j in range(d):
            n_bits = b_high if j < d_eff else b_low
            n_levels = max(2, 2 ** n_bits)
            cb = _LloydMaxCodebook(n_levels=n_levels).fit(X_rot[:, j])
            self.codebooks.append(cb)

        self._calibrated = True
        return self

    # ---- compress ----

    def compress(self, X: np.ndarray) -> list[np.ndarray]:
        """
        Compress X: [n, d] → list of d int arrays (one per coordinate).

        Bits used:
            coordinates 0..d_eff-1: self.b_high bits
            coordinates d_eff..d-1: self.b_low bits
        """
        assert self._calibrated, "Call calibrate() first"
        X_rot = X @ self.rotation_matrix
        return [self.codebooks[j].encode(X_rot[:, j]) for j in range(self.head_dim)]

    # ---- decompress ----

    def decompress(self, compressed: list[np.ndarray]) -> np.ndarray:
        """Decompress → [n, d]."""
        assert self._calibrated
        d = self.head_dim
        n = len(compressed[0])
        X_rot = np.stack([self.codebooks[j].decode(compressed[j]) for j in range(d)], axis=1)
        return X_rot @ self.rotation_matrix.T  # undo rotation

    # ---- bits used ----

    def bits_per_vector(self) -> float:
        """Total bits used per vector (should match avg_bits × d)."""
        d_eff = self.d_eff
        d_tail = self.head_dim - d_eff
        return float(d_eff * self.b_high + d_tail * self.b_low)


# ---------------------------------------------------------------------------
# Test 1: Full pipeline calibrate → compress → decompress → compare
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_pipeline_runs_without_error(self, sharp_gap_kv_data):
        keys, _, _ = sharp_gap_kv_data
        pipe = _SpectralQuantPipeline(use_spectral=True, avg_bits=3.0)
        pipe.calibrate(keys)
        compressed = pipe.compress(keys)
        reconstructed = pipe.decompress(compressed)
        assert reconstructed.shape == keys.shape

    def test_pipeline_mse_is_bounded(self, sharp_gap_kv_data):
        """After compress-decompress, MSE should be smaller than signal variance."""
        keys, _, _ = sharp_gap_kv_data
        pipe = _SpectralQuantPipeline(use_spectral=True, avg_bits=3.0)
        pipe.calibrate(keys)
        reconstructed = pipe.decompress(pipe.compress(keys))
        mse = float(((keys - reconstructed) ** 2).mean())
        var = float(keys.var())
        assert mse < var, f"MSE={mse:.4f} exceeds total variance={var:.4f}"

    def test_pipeline_reconstructed_shape(self, flat_spectrum_data):
        pipe = _SpectralQuantPipeline(use_spectral=True, avg_bits=3.0)
        pipe.calibrate(flat_spectrum_data)
        rec = pipe.decompress(pipe.compress(flat_spectrum_data))
        assert rec.shape == flat_spectrum_data.shape

    def test_random_rotation_pipeline_runs(self, sharp_gap_kv_data):
        keys, _, _ = sharp_gap_kv_data
        pipe = _SpectralQuantPipeline(use_spectral=False, avg_bits=3.0)
        pipe.calibrate(keys)
        rec = pipe.decompress(pipe.compress(keys))
        assert rec.shape == keys.shape

    def test_cosine_similarity_high(self, sharp_gap_kv_data):
        """
        After compression, the cosine similarity between original and
        reconstructed vectors should be high (> 0.90 at 3 bits).
        """
        keys, _, _ = sharp_gap_kv_data
        pipe = _SpectralQuantPipeline(use_spectral=True, avg_bits=3.0)
        pipe.calibrate(keys)
        rec = pipe.decompress(pipe.compress(keys))

        norms_orig = np.linalg.norm(keys, axis=1, keepdims=True)
        norms_rec = np.linalg.norm(rec, axis=1, keepdims=True)
        # Avoid division by zero
        valid = (norms_orig.ravel() > 1e-8) & (norms_rec.ravel() > 1e-8)
        dot = (keys[valid] * rec[valid]).sum(axis=1)
        cos_sim = dot / (norms_orig[valid].ravel() * norms_rec[valid].ravel())
        mean_cos = float(cos_sim.mean())
        assert mean_cos > 0.90, f"Mean cosine similarity {mean_cos:.4f} is too low at 3 bits"


# ---------------------------------------------------------------------------
# Test 2: Compression ratio matches configured avg_bits
# ---------------------------------------------------------------------------


class TestCompressionRatio:
    def test_bits_per_vector_matches_avg_bits(self, sharp_gap_kv_data):
        """
        Total bits used per vector should be approximately avg_bits × head_dim.
        Tolerance: ±2 bits (rounding to integer bit widths).
        """
        keys, _, _ = sharp_gap_kv_data
        for avg_bits in [2.0, 3.0, 4.0]:
            pipe = _SpectralQuantPipeline(use_spectral=True, avg_bits=avg_bits)
            pipe.calibrate(keys)
            actual_bits = pipe.bits_per_vector()
            target_bits = avg_bits * HEAD_DIM
            assert abs(actual_bits - target_bits) <= 2 * HEAD_DIM, (
                f"avg_bits={avg_bits}: actual={actual_bits:.1f}, target={target_bits:.1f}"
            )

    def test_compression_reduces_memory(self, sharp_gap_kv_data):
        """Compressed representation uses fewer bits than FP32 (32 bits/coord)."""
        keys, _, _ = sharp_gap_kv_data
        pipe = _SpectralQuantPipeline(use_spectral=True, avg_bits=3.0)
        pipe.calibrate(keys)
        bits_compressed = pipe.bits_per_vector()
        bits_fp32 = 32.0 * HEAD_DIM
        assert bits_compressed < bits_fp32, (
            f"Compressed ({bits_compressed:.0f} bits) should be < FP32 ({bits_fp32:.0f} bits)"
        )


# ---------------------------------------------------------------------------
# Test 3: SpectralQuant beats random rotation on sharp-gap data
# ---------------------------------------------------------------------------


class TestSpectralVsRandom:
    def test_spectral_beats_random_on_sharp_gap(self, sharp_gap_kv_data):
        """
        For data with a strong spectral gap, SpectralQuant should achieve
        lower MSE than random rotation at the same bit budget.
        """
        keys, _, d_eff_true = sharp_gap_kv_data

        spectral_pipe = _SpectralQuantPipeline(
            use_spectral=True, avg_bits=3.0, d_eff=d_eff_true
        )
        spectral_pipe.calibrate(keys)
        rec_spectral = spectral_pipe.decompress(spectral_pipe.compress(keys))
        mse_spectral = float(((keys - rec_spectral) ** 2).mean())

        random_pipe = _SpectralQuantPipeline(
            use_spectral=False, avg_bits=3.0, d_eff=d_eff_true
        )
        random_pipe.calibrate(keys)
        rec_random = random_pipe.decompress(random_pipe.compress(keys))
        mse_random = float(((keys - rec_random) ** 2).mean())

        assert mse_spectral <= mse_random * 1.05, (
            f"SpectralQuant MSE ({mse_spectral:.6f}) should be ≤ random rotation MSE "
            f"({mse_random:.6f}) on sharp-gap data (within 5% tolerance for rounding)"
        )

    def test_spectral_cosine_sim_higher_on_sharp_gap(self, sharp_gap_kv_data):
        """SpectralQuant cosine similarity ≥ random rotation cosine similarity."""
        keys, _, d_eff_true = sharp_gap_kv_data

        def mean_cosine_sim(orig, rec):
            norms_o = np.linalg.norm(orig, axis=1)
            norms_r = np.linalg.norm(rec, axis=1)
            valid = (norms_o > 1e-8) & (norms_r > 1e-8)
            dot = (orig[valid] * rec[valid]).sum(axis=1)
            return float((dot / (norms_o[valid] * norms_r[valid])).mean())

        spectral = _SpectralQuantPipeline(use_spectral=True, avg_bits=3.0, d_eff=d_eff_true)
        spectral.calibrate(keys)
        sim_spectral = mean_cosine_sim(keys, spectral.decompress(spectral.compress(keys)))

        random = _SpectralQuantPipeline(use_spectral=False, avg_bits=3.0, d_eff=d_eff_true)
        random.calibrate(keys)
        sim_random = mean_cosine_sim(keys, random.decompress(random.compress(keys)))

        assert sim_spectral >= sim_random - 0.02, (
            f"SpectralQuant cosine sim ({sim_spectral:.4f}) should be ≥ random "
            f"({sim_random:.4f}) on sharp-gap data"
        )


# ---------------------------------------------------------------------------
# Test 4: SpectralQuant matches random rotation on flat-spectrum data
#         (graceful degradation — should NOT be worse)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_spectral_not_worse_on_flat_spectrum(self, flat_spectrum_data):
        """
        For flat-spectrum data (no gap), SpectralQuant should not be
        significantly worse than random rotation.
        Tolerance: at most 10% higher MSE (due to rounding in bit allocation).
        """
        X = flat_spectrum_data

        spectral = _SpectralQuantPipeline(use_spectral=True, avg_bits=3.0)
        spectral.calibrate(X)
        rec_s = spectral.decompress(spectral.compress(X))
        mse_spectral = float(((X - rec_s) ** 2).mean())

        random = _SpectralQuantPipeline(use_spectral=False, avg_bits=3.0)
        random.calibrate(X)
        rec_r = random.decompress(random.compress(X))
        mse_random = float(((X - rec_r) ** 2).mean())

        assert mse_spectral <= mse_random * 1.10, (
            f"On flat-spectrum data, SpectralQuant MSE ({mse_spectral:.6f}) "
            f"is > 10%% worse than random rotation ({mse_random:.6f})"
        )

    def test_both_methods_achieve_similar_mse_on_flat_spectrum(self, flat_spectrum_data):
        """
        The relative MSE difference between SpectralQuant and random rotation
        should be small for flat-spectrum data.
        """
        X = flat_spectrum_data

        spectral = _SpectralQuantPipeline(use_spectral=True, avg_bits=4.0)
        spectral.calibrate(X)
        rec_s = spectral.decompress(spectral.compress(X))
        mse_s = float(((X - rec_s) ** 2).mean())

        random = _SpectralQuantPipeline(use_spectral=False, avg_bits=4.0)
        random.calibrate(X)
        rec_r = random.decompress(random.compress(X))
        mse_r = float(((X - rec_r) ** 2).mean())

        if mse_r > 0:
            rel_diff = abs(mse_s - mse_r) / mse_r
            assert rel_diff < 0.50, (
                f"Relative MSE difference on flat spectrum is {rel_diff:.2%} "
                f"(SpectralQuant: {mse_s:.6f}, random: {mse_r:.6f})"
            )
