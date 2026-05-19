import math
import sys
from pathlib import Path

import pytest
import torch
from scipy.linalg import hadamard as scipy_hadamard

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from experiments.codex_wht_vs_pca import build_wht_bank, build_wht_basis, hadamard_matrix


def test_hadamard_matrix_matches_scipy_reference():
    n = 8
    expected = torch.tensor(scipy_hadamard(n), dtype=torch.float32) / math.sqrt(n)
    actual = hadamard_matrix(n)
    assert torch.allclose(actual, expected)


def test_hadamard_matrix_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power-of-two"):
        hadamard_matrix(12)


def test_wht_basis_is_deterministic_for_same_seed():
    basis_a = build_wht_basis(64, seed=1234)
    basis_b = build_wht_basis(64, seed=1234)
    basis_c = build_wht_basis(64, seed=1235)

    assert torch.allclose(basis_a, basis_b)
    assert not torch.allclose(basis_a, basis_c)


def test_wht_basis_is_orthogonal_and_norm_preserving():
    basis = build_wht_basis(64, seed=7)
    identity = torch.eye(64, dtype=torch.float32)

    assert torch.allclose(basis.T @ basis, identity, atol=1e-5, rtol=1e-5)
    assert torch.allclose(basis @ basis.T, identity, atol=1e-5, rtol=1e-5)

    x = torch.randn(64, dtype=torch.float32)
    y = x @ basis

    assert torch.allclose(torch.linalg.vector_norm(x), torch.linalg.vector_norm(y), atol=1e-5, rtol=1e-5)
    assert torch.allclose(y @ basis.T, x, atol=1e-5, rtol=1e-5)


def test_wht_bank_returns_orthogonal_bases_for_keys_and_values():
    keys, values = build_wht_bank(n_layers=2, n_kv=2, hd=64)
    identity = torch.eye(64, dtype=torch.float32)

    assert set(keys.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert set(values.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    for bank in (keys, values):
        for item in bank.values():
            basis = item["evec"]
            assert basis.shape == (64, 64)
            assert torch.allclose(basis.T @ basis, identity, atol=1e-5, rtol=1e-5)
            assert torch.all(item["ev"] == 1.0)
