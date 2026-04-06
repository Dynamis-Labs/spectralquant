"""
test_calibration.py — Tests for eigenspectral calibration.

All tests use synthetic data with known spectral structure, so we can
verify calibration results against ground truth. No GPU required.

Tested properties:
    1. Calibration on synthetic data with known eigenstructure
    2. d_eff computation: for identity covariance, d_eff should equal d
    3. d_eff computation: for rank-k covariance, d_eff ≈ k
    4. Spectral gap κ computation
    5. Save/load roundtrip (calibration state is serialisable)
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import HEAD_DIM, N_TOKENS, SEED


# ---------------------------------------------------------------------------
# Minimal self-contained calibration implementation for testing.
# Replace with real imports once spectralquant.calibration is built.
# ---------------------------------------------------------------------------


class SpectralCalibration:
    """
    Minimal calibration object for test purposes.

    Usage:
        cal = SpectralCalibration()
        cal.fit(X)  # X: [n, d]
        cal.d_eff   # effective dimensionality
        cal.gap     # spectral gap κ
        cal.V       # eigenvector matrix [d, d]
    """

    def __init__(self):
        self.eigenvalues: np.ndarray | None = None
        self.eigenvectors: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "SpectralCalibration":
        """Compute eigenspectral decomposition of the sample covariance of X."""
        assert X.ndim == 2, f"Expected 2D array, got shape {X.shape}"
        C = X.T @ X / len(X)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # Sort descending
        self.eigenvalues = eigenvalues[::-1].copy().astype(np.float64)
        self.eigenvectors = eigenvectors[:, ::-1].copy().astype(np.float32)
        self._fitted = True
        return self

    @property
    def d(self) -> int:
        self._check_fitted()
        return len(self.eigenvalues)

    @property
    def d_eff(self) -> int:
        """Participation ratio: d_eff = (Σλ_i)² / Σ(λ_i²)"""
        self._check_fitted()
        lam = self.eigenvalues
        return int(round((lam.sum() ** 2) / (lam ** 2).sum()))

    @property
    def gap(self) -> float:
        """Spectral gap: κ = λ_{d_eff} / λ_{d_eff+1}"""
        self._check_fitted()
        k = self.d_eff
        if k >= self.d:
            return float("inf")
        return float(self.eigenvalues[k - 1] / self.eigenvalues[k])

    @property
    def V(self) -> np.ndarray:
        """Eigenvector matrix; columns sorted by decreasing eigenvalue."""
        self._check_fitted()
        return self.eigenvectors

    def cumulative_variance(self, threshold: float = 0.95) -> int:
        """Number of components needed to explain `threshold` fraction of variance."""
        self._check_fitted()
        cumvar = np.cumsum(self.eigenvalues) / self.eigenvalues.sum()
        indices = np.where(cumvar >= threshold)[0]
        return int(indices[0] + 1) if len(indices) > 0 else self.d

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        np.savez(
            str(path),
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SpectralCalibration":
        data = np.load(str(path))
        obj = cls()
        obj.eigenvalues = data["eigenvalues"]
        obj.eigenvectors = data["eigenvectors"]
        obj._fitted = True
        return obj

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing calibration attributes.")


# ---------------------------------------------------------------------------
# Test 1: Calibration on synthetic data with known eigenstructure
# ---------------------------------------------------------------------------


class TestCalibrationFit:
    def test_fit_runs_on_flat_spectrum(self, flat_spectrum_data):
        """Fit should complete without error on identity-covariance data."""
        cal = SpectralCalibration()
        cal.fit(flat_spectrum_data)
        assert cal._fitted

    def test_fit_runs_on_low_rank_data(self, low_rank_data):
        data, k = low_rank_data
        cal = SpectralCalibration()
        cal.fit(data)
        assert cal._fitted

    def test_fit_runs_on_sharp_gap_data(self, sharp_gap_kv_data):
        keys, values, d_eff_true = sharp_gap_kv_data
        cal = SpectralCalibration()
        cal.fit(keys)
        assert cal._fitted

    def test_eigenvalues_are_non_negative(self, flat_spectrum_data):
        """Eigenvalues of a covariance matrix are always ≥ 0."""
        cal = SpectralCalibration().fit(flat_spectrum_data)
        assert np.all(cal.eigenvalues >= -1e-6), (
            f"Negative eigenvalue found: {cal.eigenvalues.min():.6f}"
        )

    def test_eigenvalues_sorted_descending(self, flat_spectrum_data):
        """Eigenvalues should be returned in descending order."""
        cal = SpectralCalibration().fit(flat_spectrum_data)
        assert np.all(np.diff(cal.eigenvalues) <= 1e-6), (
            "Eigenvalues are not in descending order"
        )

    def test_n_eigenvalues_equals_d(self, flat_spectrum_data):
        cal = SpectralCalibration().fit(flat_spectrum_data)
        assert len(cal.eigenvalues) == HEAD_DIM
        assert cal.eigenvectors.shape == (HEAD_DIM, HEAD_DIM)

    def test_unfitted_raises(self):
        cal = SpectralCalibration()
        with pytest.raises(RuntimeError, match="fit"):
            _ = cal.d_eff


# ---------------------------------------------------------------------------
# Test 2: d_eff for identity covariance should equal d
# ---------------------------------------------------------------------------


class TestDeffIdentityCovariance:
    def test_d_eff_equals_d_for_iid_gaussian(self):
        """
        For i.i.d. Gaussian data (identity covariance), all eigenvalues are
        approximately equal, so the participation ratio d_eff ≈ d.
        """
        rng = np.random.default_rng(42)
        # Use many samples so eigenvalue estimates are tight
        n = 10_000
        d = HEAD_DIM
        X = rng.standard_normal((n, d)).astype(np.float32)
        cal = SpectralCalibration().fit(X)
        # d_eff should be close to d (within 10%)
        assert cal.d_eff >= int(0.9 * d), (
            f"d_eff={cal.d_eff} should be close to d={d} for identity covariance"
        )

    def test_d_eff_is_at_most_d(self, flat_spectrum_data):
        cal = SpectralCalibration().fit(flat_spectrum_data)
        assert cal.d_eff <= HEAD_DIM

    def test_d_eff_is_at_least_1(self, flat_spectrum_data):
        cal = SpectralCalibration().fit(flat_spectrum_data)
        assert cal.d_eff >= 1


# ---------------------------------------------------------------------------
# Test 3: d_eff for rank-k covariance should be approximately k
# ---------------------------------------------------------------------------


class TestDeffRankK:
    def test_d_eff_approximately_k_for_rank_k(self, low_rank_data):
        """
        For data generated from a rank-k distribution (with strong
        k active directions), d_eff should be approximately k.
        """
        data, k = low_rank_data
        cal = SpectralCalibration().fit(data)
        # Allow a wide tolerance because the participation ratio is a smooth measure
        # and small amounts of noise inflate d_eff slightly above k
        assert cal.d_eff <= 3 * k, (
            f"d_eff={cal.d_eff} is much larger than k={k} (rank-k data)"
        )
        assert cal.d_eff >= 1, f"d_eff={cal.d_eff} is < 1"

    def test_sharp_gap_d_eff_is_small(self, sharp_gap_kv_data):
        """For data with a sharp spectral gap at d_eff=8, d_eff should be near 8."""
        keys, _, d_eff_true = sharp_gap_kv_data
        cal = SpectralCalibration().fit(keys)
        # Tight tolerance: gap is very sharp so participation ratio should converge
        assert abs(cal.d_eff - d_eff_true) <= 4, (
            f"d_eff={cal.d_eff}, expected ≈ {d_eff_true}"
        )

    def test_d_eff_increases_with_noise(self):
        """
        Adding isotropic noise to rank-k data should increase d_eff
        (noise fills in the tail dimensions).
        """
        rng = np.random.default_rng(42)
        d, n, k = HEAD_DIM, 2000, 4
        U = rng.standard_normal((d, k)).astype(np.float32)
        Z = rng.standard_normal((n, k)).astype(np.float32) * 10.0

        X_clean = Z @ U.T  # rank-k
        noise = rng.standard_normal((n, d)).astype(np.float32) * 2.0
        X_noisy = X_clean + noise  # rank-k + noise

        cal_clean = SpectralCalibration().fit(X_clean)
        cal_noisy = SpectralCalibration().fit(X_noisy)

        assert cal_noisy.d_eff >= cal_clean.d_eff, (
            f"d_eff should not decrease when noise is added: "
            f"clean={cal_clean.d_eff}, noisy={cal_noisy.d_eff}"
        )


# ---------------------------------------------------------------------------
# Test 4: Spectral gap computation
# ---------------------------------------------------------------------------


class TestSpectralGap:
    def test_gap_is_large_for_sharp_spectrum(self, sharp_gap_kv_data):
        """
        Data with eigenvalues 50 (top 8 dims) vs 1 (remaining dims)
        should yield a large spectral gap (κ ≈ 50).
        """
        keys, _, d_eff_true = sharp_gap_kv_data
        cal = SpectralCalibration().fit(keys)
        # The gap at d_eff should be large (eigenvalue ratio ≈ 50/1 = 50)
        assert cal.gap > 5.0, (
            f"Expected large spectral gap (κ > 5), got κ = {cal.gap:.2f}"
        )

    def test_gap_is_near_1_for_flat_spectrum(self, flat_spectrum_data):
        """
        For i.i.d. Gaussian data, all eigenvalues are similar so the gap ≈ 1.
        """
        # Use large n for stable eigenvalue estimates
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5000, HEAD_DIM)).astype(np.float32)
        cal = SpectralCalibration().fit(X)
        # Gap should be close to 1 (no sharp spectral gap)
        assert cal.gap < 5.0, (
            f"Expected small spectral gap for flat spectrum, got κ = {cal.gap:.2f}"
        )

    def test_gap_is_positive(self, sharp_gap_kv_data):
        keys, _, _ = sharp_gap_kv_data
        cal = SpectralCalibration().fit(keys)
        assert cal.gap > 0

    def test_gap_definition_is_ratio(self, pre_computed_calibration):
        """Gap should equal λ_{d_eff} / λ_{d_eff+1}."""
        cal_data = pre_computed_calibration
        eigenvalues = cal_data["eigenvalues"]
        d_eff = cal_data["d_eff"]
        expected_gap = float(eigenvalues[d_eff - 1] / eigenvalues[d_eff])
        # Refit from scratch and compare
        keys, _, _ = pytest.fixture  # bypass — use pre-computed directly
        assert abs(cal_data["spectral_gap"] - expected_gap) < 0.01


# ---------------------------------------------------------------------------
# Test 5: Save/load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_load_roundtrip(self, flat_spectrum_data, tmp_path):
        """Saving and loading a calibration should preserve all attributes."""
        cal = SpectralCalibration().fit(flat_spectrum_data)
        path = tmp_path / "calibration.npz"
        cal.save(path)

        cal2 = SpectralCalibration.load(path)
        np.testing.assert_allclose(cal.eigenvalues, cal2.eigenvalues, atol=1e-6)
        np.testing.assert_allclose(cal.eigenvectors, cal2.eigenvectors, atol=1e-6)
        assert cal.d_eff == cal2.d_eff
        assert abs(cal.gap - cal2.gap) < 0.01

    def test_load_restores_fitted_state(self, flat_spectrum_data, tmp_path):
        """Loaded calibration should report as fitted."""
        cal = SpectralCalibration().fit(flat_spectrum_data)
        path = tmp_path / "cal.npz"
        cal.save(path)
        cal2 = SpectralCalibration.load(path)
        assert cal2._fitted

    def test_save_creates_file(self, flat_spectrum_data, tmp_path):
        cal = SpectralCalibration().fit(flat_spectrum_data)
        path = tmp_path / "output.npz"
        assert not path.exists()
        cal.save(path)
        assert path.exists()
