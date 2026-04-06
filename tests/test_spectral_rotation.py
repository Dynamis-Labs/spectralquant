"""
test_spectral_rotation.py — Tests for spectral and random rotation matrices.

All tests run on CPU with small synthetic matrices (no GPU required).

Key properties being tested:
    1. Dot-product preservation (isometry)
    2. Orthogonality: V^T @ V = I
    3. Rotation + un-rotation = identity
    4. Batch processing consistency
    5. Random rotation baseline has the same structural properties
"""

import numpy as np
import pytest
import torch

from conftest import HEAD_DIM, N_TOKENS


# ---------------------------------------------------------------------------
# Helpers: thin wrappers that mimic what spectralquant.rotation will expose.
# These are defined here so tests can run even before the full library is built.
# Once spectralquant.rotation is available, replace these with real imports.
# ---------------------------------------------------------------------------


def compute_spectral_rotation(X: np.ndarray) -> np.ndarray:
    """
    Compute the spectral rotation matrix V from data matrix X.

    V has shape [d, d]; columns are eigenvectors of the sample covariance,
    sorted by decreasing eigenvalue.  Rotation is V^T @ x.
    """
    C = X.T @ X / len(X)
    _, V = np.linalg.eigh(C)
    # eigh returns ascending; reverse to descending
    V = V[:, ::-1].copy()
    return V.astype(np.float32)


def compute_random_rotation(d: int, seed: int = 0) -> np.ndarray:
    """Random orthogonal matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q


def rotate(V: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply rotation: Y = X @ V  (equivalent to (V^T @ x^T)^T for row vectors)."""
    return X @ V


def unrotate(V: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Undo rotation: X = Y @ V^T."""
    return Y @ V.T


# ---------------------------------------------------------------------------
# Test 1: Spectral rotation preserves dot products (isometry)
# ---------------------------------------------------------------------------


class TestDotProductPreservation:
    def test_spectral_rotation_preserves_dot_product(self, flat_spectrum_data):
        """V^T is orthogonal, so (V^T a) · (V^T b) = a · b."""
        X = flat_spectrum_data
        V = compute_spectral_rotation(X)

        a = X[0]
        b = X[1]
        dot_original = float(np.dot(a, b))

        a_rot = rotate(V, a)
        b_rot = rotate(V, b)
        dot_rotated = float(np.dot(a_rot, b_rot))

        assert abs(dot_original - dot_rotated) < 1e-4, (
            f"Dot product not preserved: {dot_original:.6f} vs {dot_rotated:.6f}"
        )

    def test_random_rotation_preserves_dot_product(self, flat_spectrum_data):
        """Random rotation also preserves dot products."""
        X = flat_spectrum_data
        V = compute_random_rotation(HEAD_DIM, seed=123)

        a = X[0]
        b = X[1]
        dot_original = float(np.dot(a, b))
        a_rot = rotate(V, a)
        b_rot = rotate(V, b)
        dot_rotated = float(np.dot(a_rot, b_rot))

        assert abs(dot_original - dot_rotated) < 1e-4

    def test_norm_preserved_under_rotation(self, flat_spectrum_data):
        """||V^T x||_2 = ||x||_2."""
        X = flat_spectrum_data
        V = compute_spectral_rotation(X)
        norms_orig = np.linalg.norm(X, axis=1)
        norms_rot = np.linalg.norm(rotate(V, X), axis=1)
        np.testing.assert_allclose(norms_orig, norms_rot, rtol=1e-4, atol=1e-4)

    def test_inner_product_matrix_preserved(self, flat_spectrum_data):
        """
        For a batch of vectors, the Gram matrix K = X @ X^T should equal
        the Gram matrix of the rotated vectors.
        """
        X = flat_spectrum_data[:20]  # use 20 vectors
        V = compute_spectral_rotation(flat_spectrum_data)
        X_rot = rotate(V, X)

        K_orig = X @ X.T
        K_rot = X_rot @ X_rot.T

        np.testing.assert_allclose(K_orig, K_rot, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 2: Orthogonality V^T @ V = I
# ---------------------------------------------------------------------------


class TestOrthogonality:
    def test_spectral_rotation_is_orthogonal(self, flat_spectrum_data):
        """V^T @ V should be identity."""
        V = compute_spectral_rotation(flat_spectrum_data)
        product = V.T @ V
        np.testing.assert_allclose(
            product, np.eye(HEAD_DIM, dtype=np.float32), atol=1e-5,
            err_msg="Spectral rotation V^T @ V is not identity."
        )

    def test_spectral_rotation_columns_are_unit_vectors(self, flat_spectrum_data):
        """Each column of V should have unit norm."""
        V = compute_spectral_rotation(flat_spectrum_data)
        col_norms = np.linalg.norm(V, axis=0)
        np.testing.assert_allclose(
            col_norms, np.ones(HEAD_DIM, dtype=np.float32), atol=1e-5
        )

    def test_random_rotation_is_orthogonal(self):
        """Random rotation Q from QR decomposition satisfies Q^T @ Q = I."""
        Q = compute_random_rotation(HEAD_DIM)
        product = Q.T @ Q
        np.testing.assert_allclose(
            product, np.eye(HEAD_DIM, dtype=np.float32), atol=1e-5
        )

    def test_spectral_rotation_determinant_is_pm1(self, flat_spectrum_data):
        """det(V) = ±1 for a proper orthogonal matrix."""
        V = compute_spectral_rotation(flat_spectrum_data)
        det = float(np.linalg.det(V))
        assert abs(abs(det) - 1.0) < 1e-4, f"det(V) = {det}, expected ±1"


# ---------------------------------------------------------------------------
# Test 3: Rotation + unrotation = identity
# ---------------------------------------------------------------------------


class TestRotationRoundTrip:
    def test_rotate_unrotate_is_identity(self, flat_spectrum_data):
        """unrotate(V, rotate(V, X)) ≈ X."""
        X = flat_spectrum_data
        V = compute_spectral_rotation(X)
        X_reconstructed = unrotate(V, rotate(V, X))
        np.testing.assert_allclose(X, X_reconstructed, atol=1e-5, rtol=1e-5)

    def test_rotate_unrotate_single_vector(self, flat_spectrum_data):
        """Test on a single vector (1D input)."""
        V = compute_spectral_rotation(flat_spectrum_data)
        x = flat_spectrum_data[42]
        x_reconstructed = unrotate(V, rotate(V, x))
        np.testing.assert_allclose(x, x_reconstructed, atol=1e-5)

    def test_random_rotation_round_trip(self, flat_spectrum_data):
        """Random rotation + unrotation = identity."""
        X = flat_spectrum_data
        Q = compute_random_rotation(HEAD_DIM)
        X_reconstructed = unrotate(Q, rotate(Q, X))
        np.testing.assert_allclose(X, X_reconstructed, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: Batch processing
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    def test_batch_equals_individual(self, flat_spectrum_data):
        """
        Rotating a batch of vectors at once should give the same result as
        rotating each vector individually.
        """
        X = flat_spectrum_data[:10]
        V = compute_spectral_rotation(flat_spectrum_data)

        # Batch rotation
        X_rot_batch = rotate(V, X)

        # Individual rotation
        X_rot_individual = np.stack([rotate(V, X[i]) for i in range(len(X))])

        np.testing.assert_allclose(X_rot_batch, X_rot_individual, atol=1e-6)

    def test_rotation_with_torch_tensors(self, flat_spectrum_data):
        """Rotation should work equivalently as a torch matmul."""
        X = flat_spectrum_data[:10]
        V = compute_spectral_rotation(flat_spectrum_data)

        X_np = rotate(V, X)

        X_t = torch.from_numpy(X)
        V_t = torch.from_numpy(V)
        X_torch = (X_t @ V_t).numpy()

        np.testing.assert_allclose(X_np, X_torch, atol=1e-5)

    def test_rotation_dtype_consistency(self, flat_spectrum_data):
        """Output dtype should match input dtype."""
        X = flat_spectrum_data.astype(np.float32)
        V = compute_spectral_rotation(X)
        X_rot = rotate(V, X)
        assert X_rot.dtype == np.float32, f"Expected float32, got {X_rot.dtype}"


# ---------------------------------------------------------------------------
# Test 5: Random rotation has same structural properties
# ---------------------------------------------------------------------------


class TestRandomRotationBaseline:
    def test_random_rotation_orthogonal(self):
        Q = compute_random_rotation(HEAD_DIM)
        np.testing.assert_allclose(Q.T @ Q, np.eye(HEAD_DIM, dtype=np.float32), atol=1e-5)

    def test_random_rotation_preserves_norms(self, flat_spectrum_data):
        X = flat_spectrum_data
        Q = compute_random_rotation(HEAD_DIM)
        np.testing.assert_allclose(
            np.linalg.norm(X, axis=1),
            np.linalg.norm(rotate(Q, X), axis=1),
            atol=1e-4,
        )

    def test_spectral_rotation_sorts_variance(self, sharp_gap_kv_data):
        """
        After spectral rotation, coordinate variances should be in descending order.
        After random rotation, this property should NOT hold (coords are mixed).
        """
        keys, _, _ = sharp_gap_kv_data
        V_spectral = compute_spectral_rotation(keys)
        Q_random = compute_random_rotation(HEAD_DIM, seed=7)

        rotated_spectral = rotate(V_spectral, keys)
        rotated_random = rotate(Q_random, keys)

        var_spectral = rotated_spectral.var(axis=0)
        var_random = rotated_random.var(axis=0)

        # Spectral: variance should be (approximately) monotonically decreasing
        assert np.all(np.diff(var_spectral) <= 0.5), (
            "After spectral rotation, coordinate variances are not sorted descending"
        )
        # Random: variance should NOT be sorted (with high probability for sharp-gap data)
        is_sorted_random = np.all(np.diff(var_random) <= 0)
        assert not is_sorted_random, (
            "After random rotation, variances happen to be sorted (unexpected)"
        )
