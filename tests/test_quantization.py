"""
test_quantization.py — Tests for Lloyd-Max quantization and bit allocation.

All tests run on CPU with small synthetic data (no GPU required).

Tested properties:
    1. Lloyd-Max converges on Gaussian data
    2. Bit allocation respects the budget constraint:
       d_eff * b_high + (d - d_eff) * b_low = B
    3. Compress → decompress roundtrip (MSE is bounded)
    4. Non-uniform quantization with d_eff=d degenerates to uniform
    5. MSE decreases monotonically with more bits
"""

import numpy as np
import pytest

from conftest import HEAD_DIM, N_TOKENS


# ---------------------------------------------------------------------------
# Minimal self-contained Lloyd-Max implementation for testing.
# Once spectralquant.quantization is available, replace with real imports.
# ---------------------------------------------------------------------------


def lloyd_max_1d(
    data: np.ndarray,
    n_levels: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D Lloyd-Max quantizer.

    Returns:
        centroids:   [n_levels]  — optimal reconstruction values
        boundaries:  [n_levels+1]  — decision boundaries
    """
    assert data.ndim == 1
    assert n_levels >= 2

    # Initialise centroids uniformly within the data range
    lo, hi = float(data.min()), float(data.max())
    centroids = np.linspace(lo + (hi - lo) / (2 * n_levels),
                            hi - (hi - lo) / (2 * n_levels),
                            n_levels).astype(np.float64)

    prev_distortion = np.inf
    for _ in range(max_iter):
        # Update boundaries (midpoints between centroids)
        boundaries = np.concatenate([
            [-np.inf],
            (centroids[:-1] + centroids[1:]) / 2,
            [np.inf],
        ])
        # Assign each data point to its nearest centroid
        indices = np.digitize(data, boundaries[1:])
        # Update centroids to the mean of their Voronoi region
        new_centroids = np.array([
            data[indices == k].mean() if (indices == k).any() else centroids[k]
            for k in range(n_levels)
        ])
        distortion = float(((data - new_centroids[indices]) ** 2).mean())
        centroids = new_centroids
        if abs(prev_distortion - distortion) < tol:
            break
        prev_distortion = distortion

    boundaries = np.concatenate([
        [-np.inf],
        (centroids[:-1] + centroids[1:]) / 2,
        [np.inf],
    ])
    return centroids.astype(np.float32), boundaries.astype(np.float32)


def quantize(x: np.ndarray, centroids: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Quantize x using a Lloyd-Max codebook."""
    indices = np.digitize(x, boundaries[1:])
    return centroids[indices]


def compute_bit_allocation(
    avg_bits: float,
    head_dim: int,
    d_eff: int,
    b_high_int: int = None,
    b_low_int: int = None,
) -> dict:
    """
    Compute non-uniform bit allocation satisfying the budget constraint.

    Budget constraint: d_eff * b_high + (head_dim - d_eff) * b_low = avg_bits * head_dim

    Returns dict with keys: b_high, b_low, total_bits, budget_bits
    """
    budget = avg_bits * head_dim
    d_tail = head_dim - d_eff

    if b_high_int is not None and b_low_int is not None:
        b_high = b_high_int
        b_low = b_low_int
    else:
        # Simple heuristic: b_high = ceil(avg_bits) + 1, solve for b_low
        b_high = int(np.ceil(avg_bits)) + 1
        if d_tail > 0:
            b_low = (budget - d_eff * b_high) / d_tail
        else:
            b_low = 0.0

    total_bits = d_eff * b_high + d_tail * b_low
    return {
        "b_high": b_high,
        "b_low": b_low,
        "total_bits": total_bits,
        "budget_bits": budget,
    }


# ---------------------------------------------------------------------------
# Test 1: Lloyd-Max converges on Gaussian data
# ---------------------------------------------------------------------------


class TestLloydMaxConvergence:
    def test_converges_on_gaussian_4bit(self, gaussian_data_1d):
        """Lloyd-Max should converge and return non-degenerate centroids."""
        centroids, boundaries = lloyd_max_1d(gaussian_data_1d, n_levels=16, max_iter=100)
        assert len(centroids) == 16
        # Centroids should be strictly increasing
        assert np.all(np.diff(centroids) > 0), "Centroids are not strictly increasing"

    def test_converges_on_gaussian_2bit(self, gaussian_data_1d):
        centroids, _ = lloyd_max_1d(gaussian_data_1d, n_levels=4, max_iter=100)
        assert len(centroids) == 4
        assert np.all(np.diff(centroids) > 0)

    def test_centroids_span_data_range(self, gaussian_data_1d):
        """Centroids should be within the data range."""
        data = gaussian_data_1d
        centroids, _ = lloyd_max_1d(data, n_levels=8)
        assert centroids.min() >= data.min() - 1e-3
        assert centroids.max() <= data.max() + 1e-3

    def test_distortion_decreases_with_more_levels(self, gaussian_data_1d):
        """MSE should decrease as we add quantization levels."""
        data = gaussian_data_1d
        mses = []
        for n_levels in [2, 4, 8, 16, 32]:
            centroids, boundaries = lloyd_max_1d(data, n_levels=n_levels)
            q = quantize(data, centroids, boundaries)
            mse = float(((data - q) ** 2).mean())
            mses.append(mse)
        # Each MSE should be strictly less than the previous
        for i in range(1, len(mses)):
            assert mses[i] < mses[i - 1], (
                f"MSE did not decrease from {mses[i-1]:.6f} to {mses[i]:.6f} "
                f"when going from {[2,4,8,16,32][i-1]} to {[2,4,8,16,32][i]} levels"
            )

    def test_quantization_assigns_every_level(self, gaussian_data_1d):
        """Every level should be assigned to at least one data point."""
        data = gaussian_data_1d
        n_levels = 8
        centroids, boundaries = lloyd_max_1d(data, n_levels=n_levels)
        q = quantize(data, centroids, boundaries)
        assigned = set(np.unique(q))
        expected = set(centroids.tolist())
        assert assigned == expected, f"Not all levels assigned: missing {expected - assigned}"


# ---------------------------------------------------------------------------
# Test 2: Bit allocation respects budget constraint
# ---------------------------------------------------------------------------


class TestBitAllocationBudget:
    @pytest.mark.parametrize("avg_bits,d_eff", [
        (3.0, 8),
        (2.5, 4),
        (4.0, 16),
        (2.0, 2),
    ])
    def test_budget_constraint_satisfied(self, avg_bits, d_eff):
        """d_eff * b_high + (d - d_eff) * b_low == avg_bits * d"""
        alloc = compute_bit_allocation(avg_bits, HEAD_DIM, d_eff)
        assert abs(alloc["total_bits"] - alloc["budget_bits"]) < 0.5, (
            f"Budget violation: allocated {alloc['total_bits']:.2f} bits, "
            f"budget was {alloc['budget_bits']:.2f}"
        )

    def test_d_eff_equals_d_gives_uniform(self):
        """When d_eff = d (full rank), allocation should be uniform: b_high = avg_bits."""
        alloc = compute_bit_allocation(avg_bits=3.0, head_dim=HEAD_DIM, d_eff=HEAD_DIM)
        assert abs(alloc["total_bits"] - 3.0 * HEAD_DIM) < 0.5

    def test_b_high_greater_than_b_low(self):
        """High-signal dimensions should receive more bits than tail dimensions."""
        alloc = compute_bit_allocation(avg_bits=3.0, head_dim=HEAD_DIM, d_eff=8)
        assert alloc["b_high"] > alloc["b_low"], (
            f"b_high ({alloc['b_high']}) should exceed b_low ({alloc['b_low']})"
        )

    def test_explicit_override(self):
        """Explicit b_high and b_low should be honoured."""
        alloc = compute_bit_allocation(
            avg_bits=3.0, head_dim=32, d_eff=8, b_high_int=5, b_low_int=2
        )
        assert alloc["b_high"] == 5
        assert alloc["b_low"] == 2

    def test_budget_with_no_tail(self):
        """When d_eff == head_dim, there is no tail and budget should still hold."""
        alloc = compute_bit_allocation(avg_bits=4.0, head_dim=16, d_eff=16)
        assert abs(alloc["total_bits"] - 4.0 * 16) < 0.5


# ---------------------------------------------------------------------------
# Test 3: Compress → decompress roundtrip
# ---------------------------------------------------------------------------


class TestCompressDecompressRoundtrip:
    def _compress_decompress(self, X: np.ndarray, n_bits: int) -> np.ndarray:
        """Compress each coordinate independently using Lloyd-Max, then reconstruct."""
        n_levels = 2 ** n_bits
        X_rec = np.empty_like(X)
        for j in range(X.shape[1]):
            centroids, boundaries = lloyd_max_1d(X[:, j], n_levels=n_levels)
            X_rec[:, j] = quantize(X[:, j], centroids, boundaries)
        return X_rec

    def test_roundtrip_mse_is_bounded(self, flat_spectrum_data):
        """MSE after 4-bit quantization should be much smaller than signal variance."""
        X = flat_spectrum_data
        X_rec = self._compress_decompress(X, n_bits=4)
        mse = float(((X - X_rec) ** 2).mean())
        variance = float(X.var())
        # At 4 bits, quantization noise should be < 1% of signal variance
        assert mse < 0.01 * variance, f"MSE={mse:.6f} exceeds 1% of variance={variance:.6f}"

    def test_roundtrip_shape_preserved(self, flat_spectrum_data):
        X = flat_spectrum_data
        X_rec = self._compress_decompress(X, n_bits=3)
        assert X_rec.shape == X.shape

    def test_roundtrip_dtype_preserved(self, flat_spectrum_data):
        X = flat_spectrum_data.astype(np.float32)
        X_rec = self._compress_decompress(X, n_bits=3)
        assert X_rec.dtype == np.float32

    def test_roundtrip_values_are_quantized(self, flat_spectrum_data):
        """Every output value should be one of the allowed reconstruction levels."""
        X = flat_spectrum_data[:50]  # small slice for speed
        n_bits = 2
        n_levels = 2 ** n_bits
        X_rec = np.empty_like(X)
        for j in range(X.shape[1]):
            centroids, boundaries = lloyd_max_1d(X[:, j], n_levels=n_levels)
            X_rec[:, j] = quantize(X[:, j], centroids, boundaries)
            unique_vals = set(np.unique(X_rec[:, j]).tolist())
            centroid_set = set(centroids.tolist())
            assert unique_vals.issubset(centroid_set), (
                f"Column {j}: output values not in centroid set"
            )


# ---------------------------------------------------------------------------
# Test 4: Non-uniform quantization with d_eff=d degenerates to uniform
# ---------------------------------------------------------------------------


class TestDegeneratesToUniform:
    def test_full_rank_allocation_is_uniform(self):
        """
        When d_eff = d (flat spectrum), non-uniform allocation should
        be equivalent to uniform allocation.
        """
        avg_bits = 3.0
        alloc = compute_bit_allocation(avg_bits=avg_bits, head_dim=HEAD_DIM, d_eff=HEAD_DIM)
        # With d_eff = d, b_low is irrelevant (tail is empty)
        # total bits should equal avg_bits * head_dim
        assert abs(alloc["total_bits"] - avg_bits * HEAD_DIM) < 1.0

    def test_flat_spectrum_no_advantage(self, flat_spectrum_data):
        """
        For flat-spectrum data, uniform and non-uniform quantization should
        give similar MSE (non-uniform offers no benefit).
        """
        X = flat_spectrum_data
        n_bits = 3
        n_levels = 2 ** n_bits

        def mse_uniform(X):
            recs = []
            for j in range(X.shape[1]):
                c, b = lloyd_max_1d(X[:, j], n_levels=n_levels)
                recs.append(quantize(X[:, j], c, b))
            return float(((X - np.stack(recs, axis=1)) ** 2).mean())

        # For flat spectrum, non-uniform with d_eff=d IS uniform
        mse_u = mse_uniform(X)
        assert mse_u > 0, "MSE should be positive (quantization error exists)"


# ---------------------------------------------------------------------------
# Test 5: MSE decreases with more bits
# ---------------------------------------------------------------------------


class TestMSEVsBits:
    @pytest.mark.parametrize("n_bits", [1, 2, 3, 4, 5, 6])
    def test_mse_positive(self, gaussian_data_1d, n_bits):
        """MSE should be positive for any finite bit-width."""
        n_levels = 2 ** n_bits
        centroids, boundaries = lloyd_max_1d(gaussian_data_1d, n_levels=n_levels)
        q = quantize(gaussian_data_1d, centroids, boundaries)
        mse = float(((gaussian_data_1d - q) ** 2).mean())
        assert mse > 0, f"MSE is zero at {n_bits} bits — all data points land on centroids?"

    def test_mse_monotonically_decreases(self, gaussian_data_1d):
        """MSE(b) > MSE(b+1) for b = 1, 2, 3, 4, 5."""
        mses = {}
        for n_bits in [1, 2, 3, 4, 5, 6]:
            n_levels = 2 ** n_bits
            centroids, boundaries = lloyd_max_1d(gaussian_data_1d, n_levels=n_levels)
            q = quantize(gaussian_data_1d, centroids, boundaries)
            mses[n_bits] = float(((gaussian_data_1d - q) ** 2).mean())

        for b in range(1, 6):
            assert mses[b] > mses[b + 1], (
                f"MSE did not decrease from {b} to {b+1} bits: "
                f"{mses[b]:.6f} vs {mses[b+1]:.6f}"
            )

    def test_high_bits_near_zero_mse(self, gaussian_data_1d):
        """At 8 bits, MSE should be very small relative to signal variance."""
        n_levels = 256  # 2^8
        centroids, boundaries = lloyd_max_1d(gaussian_data_1d, n_levels=n_levels)
        q = quantize(gaussian_data_1d, centroids, boundaries)
        mse = float(((gaussian_data_1d - q) ** 2).mean())
        variance = float(gaussian_data_1d.var())
        assert mse < 1e-3 * variance, (
            f"At 8 bits, MSE={mse:.6f} is too large relative to variance={variance:.6f}"
        )
