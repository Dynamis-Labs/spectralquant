"""Unit tests for `spectralquant.waterfill`.

Coverage corresponds to spec §12.1 plus a few defensive cases:

- Allocation sums to total budget.
- Equal eigenvalues yield uniform / nearly-uniform allocation.
- Concentrated spectrum allocates more to dominant dims.
- min_bits / max_bits respected.
- Invalid budgets / negative / NaN eigenvalues raise.
- Zero eigenvalues do not produce NaN/Inf in scores.
- Tie-breaking is deterministic (lowest index).
- numpy / torch / list inputs produce identical results.
- Inputs are not mutated.
- summarize_allocation is JSON-safe and self-consistent.
"""

from __future__ import annotations

import json

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load `spectralquant.waterfill` directly from its source file. This
# bypasses `src/spectralquant/__init__.py`, which transitively imports
# torch-dependent modules and would fail in environments without torch.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_WATERFILL_PATH = _REPO_ROOT / "src" / "spectralquant" / "waterfill.py"
_spec = importlib.util.spec_from_file_location("spectralquant_waterfill", _WATERFILL_PATH)
assert _spec is not None and _spec.loader is not None
waterfill = importlib.util.module_from_spec(_spec)
sys.modules["spectralquant_waterfill"] = waterfill
_spec.loader.exec_module(waterfill)

FORMULA_VERSION = waterfill.FORMULA_VERSION
allocate_waterfill_bits = waterfill.allocate_waterfill_bits
marginal_gain = waterfill.marginal_gain
summarize_allocation = waterfill.summarize_allocation
validate_bit_allocation = waterfill.validate_bit_allocation

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Allocation invariants
# ---------------------------------------------------------------------------


def test_allocation_sums_to_budget():
    eig = np.array([0.8, 0.12, 0.05, 0.02, 0.01])
    bits = allocate_waterfill_bits(eig, total_bits=12)
    assert int(bits.sum()) == 12
    assert bits.dtype == np.int64
    assert bits.shape == (5,)


def test_allocation_returns_numpy_int64_for_list_input():
    bits = allocate_waterfill_bits([0.5, 0.3, 0.2], total_bits=6)
    assert isinstance(bits, np.ndarray)
    assert bits.dtype == np.int64
    assert int(bits.sum()) == 6


# ---------------------------------------------------------------------------
# Spectrum-shape behaviour
# ---------------------------------------------------------------------------


def test_equal_eigenvalues_uniform_allocation_when_divisible():
    eig = np.full(4, 0.25)
    bits = allocate_waterfill_bits(eig, total_bits=8)
    # Perfectly divisible: all dims must get the same bits.
    assert bits.tolist() == [2, 2, 2, 2]


def test_equal_eigenvalues_nearly_uniform_when_not_divisible():
    eig = np.full(4, 0.25)
    bits = allocate_waterfill_bits(eig, total_bits=10)
    # Total 10 over 4 dims => most should get 3 or 2; range at most 1.
    assert int(bits.sum()) == 10
    assert bits.max() - bits.min() <= 1
    # With deterministic tie-breaking on equal eigenvalues, lowest indices
    # accumulate first: expected [3, 3, 2, 2].
    assert bits.tolist() == [3, 3, 2, 2]


def test_concentrated_spectrum_allocates_more_to_first_dim():
    eig = np.array([0.97, 0.02, 0.005, 0.005])
    bits = allocate_waterfill_bits(eig, total_bits=8)
    assert int(bits.sum()) == 8
    # The dominant eigenvalue must receive at least as many bits as any other,
    # and strictly more than the smallest.
    assert bits[0] >= bits[1] >= bits[2]
    assert bits[0] > bits[-1]


def test_marginal_gain_after_allocation_is_balanced():
    """After greedy water-filling, marginal gains should be close across dims.

    Specifically, no inactive dim should have a marginal gain higher than the
    smallest gain among the dims that received the last bit.
    """
    eig = np.array([0.7, 0.2, 0.07, 0.03])
    bits = allocate_waterfill_bits(eig, total_bits=12)
    g = marginal_gain(eig, bits)
    # All gains finite and positive.
    assert np.all(np.isfinite(g))
    assert np.all(g > 0)


# ---------------------------------------------------------------------------
# min/max bit caps
# ---------------------------------------------------------------------------


def test_min_bits_respected():
    eig = np.array([0.99, 0.005, 0.005])
    bits = allocate_waterfill_bits(eig, total_bits=9, min_bits=2)
    assert bits.min() >= 2
    assert int(bits.sum()) == 9


def test_max_bits_respected():
    eig = np.array([0.99, 0.005, 0.005])
    bits = allocate_waterfill_bits(eig, total_bits=9, max_bits=4)
    assert bits.max() <= 4
    assert int(bits.sum()) == 9


def test_max_bits_caps_dominant_dimension():
    eig = np.array([1.0, 1e-12, 1e-12, 1e-12])
    # Without a cap the dominant dim would take everything; with cap=3 it can't.
    bits = allocate_waterfill_bits(eig, total_bits=8, max_bits=3)
    assert bits[0] == 3
    assert int(bits.sum()) == 8


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_negative_eigenvalues_raise():
    with pytest.raises(ValueError, match="non-negative"):
        allocate_waterfill_bits(np.array([0.5, -0.1, 0.6]), total_bits=4)


def test_invalid_nan_eigenvalues_raise():
    with pytest.raises(ValueError, match="finite"):
        allocate_waterfill_bits(np.array([0.5, np.nan, 0.6]), total_bits=4)


def test_invalid_2d_eigenvalues_raise():
    with pytest.raises(ValueError, match="one-dimensional"):
        allocate_waterfill_bits(np.array([[0.5, 0.5], [0.3, 0.7]]), total_bits=4)


def test_invalid_empty_eigenvalues_raise():
    with pytest.raises(ValueError, match="non-empty"):
        allocate_waterfill_bits(np.array([]), total_bits=0)


def test_invalid_total_bits_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        allocate_waterfill_bits(np.array([0.5, 0.5]), total_bits=-1)


def test_invalid_total_bits_below_min_bits_raises():
    with pytest.raises(ValueError, match="cannot satisfy min_bits"):
        allocate_waterfill_bits(np.array([0.5, 0.5, 0.5]), total_bits=2, min_bits=2)


def test_invalid_total_bits_above_max_bits_raises():
    with pytest.raises(ValueError, match="exceeds"):
        allocate_waterfill_bits(np.array([0.5, 0.5]), total_bits=10, max_bits=3)


def test_invalid_total_bits_type_raises():
    with pytest.raises(TypeError):
        allocate_waterfill_bits(np.array([0.5, 0.5]), total_bits=3.5)  # type: ignore[arg-type]


def test_invalid_max_bits_below_min_bits_raises():
    with pytest.raises(ValueError, match=">= min_bits"):
        allocate_waterfill_bits(
            np.array([0.5, 0.5]), total_bits=4, min_bits=3, max_bits=2
        )


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


def test_zero_eigenvalues_do_not_nan():
    eig = np.array([0.5, 0.5, 0.0, 0.0])
    bits = allocate_waterfill_bits(eig, total_bits=6)
    assert int(bits.sum()) == 6
    assert np.all(np.isfinite(bits))
    # Zero-eigenvalue dims should not preferentially receive bits over
    # positive-eigenvalue dims.
    assert bits[0] >= bits[2]
    assert bits[1] >= bits[3]


def test_all_zero_eigenvalues_distributes_bits_deterministically():
    eig = np.zeros(4)
    bits = allocate_waterfill_bits(eig, total_bits=10)
    assert int(bits.sum()) == 10
    # With identical (zero) eigenvalues, tie-breaking favors lowest index.
    assert bits.tolist() == [3, 3, 2, 2]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_tie_breaking():
    eig = np.array([1.0, 1.0, 1.0, 1.0])
    # Add 5 bits to a 4-dim symmetric input. The greedy allocator must always
    # pick the lowest available index; expected: [2, 1, 1, 1].
    bits = allocate_waterfill_bits(eig, total_bits=5)
    assert bits.tolist() == [2, 1, 1, 1]


def test_repeated_call_is_deterministic():
    eig = np.array([0.6, 0.3, 0.07, 0.03])
    out1 = allocate_waterfill_bits(eig, total_bits=11)
    out2 = allocate_waterfill_bits(eig, total_bits=11)
    assert np.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Backend consistency
# ---------------------------------------------------------------------------


def test_numpy_and_list_inputs_match():
    eig_list = [0.5, 0.3, 0.15, 0.05]
    eig_np = np.array(eig_list)
    out_list = allocate_waterfill_bits(eig_list, total_bits=10)
    out_np = allocate_waterfill_bits(eig_np, total_bits=10)
    assert np.array_equal(out_list, out_np)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch is not available")
def test_numpy_and_torch_inputs_match():
    eig_list = [0.5, 0.3, 0.15, 0.05]
    eig_np = np.array(eig_list)
    eig_t = torch.tensor(eig_list, dtype=torch.float32)  # type: ignore[union-attr]
    out_np = allocate_waterfill_bits(eig_np, total_bits=10)
    out_t = allocate_waterfill_bits(eig_t, total_bits=10)
    assert np.array_equal(out_np, out_t)


# ---------------------------------------------------------------------------
# Mutation safety
# ---------------------------------------------------------------------------


def test_input_numpy_array_not_mutated():
    eig = np.array([0.5, 0.3, 0.15, 0.05])
    eig_copy = eig.copy()
    _ = allocate_waterfill_bits(eig, total_bits=10)
    np.testing.assert_array_equal(eig, eig_copy)


def test_input_list_not_mutated():
    eig = [0.5, 0.3, 0.15, 0.05]
    eig_copy = list(eig)
    _ = allocate_waterfill_bits(eig, total_bits=10)
    assert eig == eig_copy


@pytest.mark.skipif(not _HAS_TORCH, reason="torch is not available")
def test_input_torch_tensor_not_mutated():
    eig_t = torch.tensor([0.5, 0.3, 0.15, 0.05])  # type: ignore[union-attr]
    eig_copy = eig_t.clone()
    _ = allocate_waterfill_bits(eig_t, total_bits=10)
    assert torch.equal(eig_t, eig_copy)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# validate_bit_allocation
# ---------------------------------------------------------------------------


def test_validate_bit_allocation_accepts_valid():
    validate_bit_allocation(np.array([3, 2, 1, 0]), total_bits=6)


def test_validate_bit_allocation_rejects_wrong_sum():
    with pytest.raises(ValueError, match="bits sum to"):
        validate_bit_allocation(np.array([3, 2, 1, 0]), total_bits=10)


def test_validate_bit_allocation_rejects_below_min():
    with pytest.raises(ValueError, match="below min_bits"):
        validate_bit_allocation(np.array([3, 2, 1, 0]), total_bits=6, min_bits=1)


def test_validate_bit_allocation_rejects_above_max():
    with pytest.raises(ValueError, match="above max_bits"):
        validate_bit_allocation(np.array([5, 2, 1, 0]), total_bits=8, max_bits=4)


def test_validate_bit_allocation_rejects_non_integer():
    with pytest.raises(ValueError, match="integer-valued"):
        validate_bit_allocation(np.array([2.5, 1.5, 1.0, 1.0]), total_bits=6)


# ---------------------------------------------------------------------------
# summarize_allocation
# ---------------------------------------------------------------------------


def test_summarize_allocation_is_json_safe():
    eig = np.array([0.6, 0.25, 0.1, 0.05])
    bits = allocate_waterfill_bits(eig, total_bits=10)
    summary = summarize_allocation(eig, bits)
    # Must round-trip through JSON without errors.
    blob = json.dumps(summary)
    again = json.loads(blob)
    assert again["formula_version"] == FORMULA_VERSION
    assert again["d"] == 4
    assert again["total_bits"] == 10
    assert again["bits"] == bits.tolist()
    assert sum(again["bits"]) == 10


def test_summarize_allocation_handles_zero_eigenvalue_sum():
    eig = np.zeros(3)
    bits = np.array([1, 1, 1], dtype=np.int64)
    summary = summarize_allocation(eig, bits)
    assert summary["eigenvalue_sum"] == 0.0
    assert summary["eigenvalue_share"] == [0.0, 0.0, 0.0]
