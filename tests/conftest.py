"""
conftest.py — Shared pytest fixtures for SpectralQuant tests.

All fixtures use small synthetic data and run entirely on CPU,
so no GPU is required to execute the test suite.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

HEAD_DIM = 32       # small head_dim for fast tests (real models use 64–128)
N_TOKENS = 256      # number of tokens to simulate
N_HEADS = 4         # number of attention heads
N_LAYERS = 3        # number of transformer layers
SEED = 42


# ---------------------------------------------------------------------------
# Synthetic model / data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng():
    """Seeded NumPy RNG shared across the session."""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def torch_rng():
    """Seeded Torch generator."""
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return gen


@pytest.fixture(scope="session")
def flat_spectrum_data(rng):
    """
    Sample data with a FLAT eigenspectrum (identity covariance).
    d_eff should equal HEAD_DIM.
    Shape: [N_TOKENS, HEAD_DIM]
    """
    X = rng.standard_normal((N_TOKENS, HEAD_DIM)).astype(np.float32)
    return X


@pytest.fixture(scope="session")
def low_rank_data(rng):
    """
    Sample data from a rank-4 distribution (only 4 active eigendirections).
    d_eff should be approximately 4.
    Shape: [N_TOKENS, HEAD_DIM]
    """
    k = 4
    U = rng.standard_normal((HEAD_DIM, k)).astype(np.float32)
    # Strong eigenvalues for top-k directions
    eigenvalues = np.array([100.0, 50.0, 20.0, 10.0], dtype=np.float32)
    Z = rng.standard_normal((N_TOKENS, k)).astype(np.float32)
    X = Z * np.sqrt(eigenvalues)[None, :] @ U.T
    # Add tiny isotropic noise
    X += rng.standard_normal((N_TOKENS, HEAD_DIM)).astype(np.float32) * 0.01
    return X, k


@pytest.fixture(scope="session")
def sharp_gap_kv_data(rng):
    """
    Synthetic KV cache data with a strong spectral gap (κ >> 1).

    Returns:
        keys:   [N_TOKENS, HEAD_DIM]  — key vectors with sharp spectral structure
        values: [N_TOKENS, HEAD_DIM]  — value vectors with sharp spectral structure
        d_eff_true: int               — ground-truth effective dimension
    """
    d_eff = 8
    # Build covariance: top d_eff dims have eigenvalue 50, rest have eigenvalue 1
    eigenvalues = np.concatenate([
        np.full(d_eff, 50.0),
        np.full(HEAD_DIM - d_eff, 1.0),
    ]).astype(np.float32)

    keys = (rng.standard_normal((N_TOKENS, HEAD_DIM)) * np.sqrt(eigenvalues)[None, :]).astype(np.float32)
    values = (rng.standard_normal((N_TOKENS, HEAD_DIM)) * np.sqrt(eigenvalues)[None, :]).astype(np.float32)
    return keys, values, d_eff


@pytest.fixture(scope="session")
def gaussian_data_1d(rng):
    """
    1D Gaussian data for Lloyd-Max quantizer tests.
    Shape: [5000]  (large enough for convergence tests)
    """
    return rng.standard_normal(5000).astype(np.float32)


@pytest.fixture(scope="session")
def pre_computed_calibration(sharp_gap_kv_data, rng):
    """
    A pre-computed calibration result for re-use across tests.
    Avoids repeated PCA computation.

    Returns a dict with:
        eigenvectors: [HEAD_DIM, HEAD_DIM]
        eigenvalues:  [HEAD_DIM]
        d_eff:        int
        spectral_gap: float
    """
    keys, values, d_eff_true = sharp_gap_kv_data
    X = keys  # calibrate on keys
    C = X.T @ X / len(X)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # eigh returns ascending order; reverse to descending
    eigenvalues = eigenvalues[::-1].copy().astype(np.float32)
    eigenvectors = eigenvectors[:, ::-1].copy().astype(np.float32)
    d_eff = int(round((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()))
    gap = float(eigenvalues[d_eff - 1] / eigenvalues[d_eff]) if d_eff < HEAD_DIM else float("inf")
    return {
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "d_eff": d_eff,
        "spectral_gap": gap,
        "d_eff_true": d_eff_true,
    }


@pytest.fixture(scope="session")
def orthogonal_matrix(rng):
    """
    A random orthogonal matrix of shape [HEAD_DIM, HEAD_DIM].
    Useful for rotation tests.
    """
    A = rng.standard_normal((HEAD_DIM, HEAD_DIM)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q
