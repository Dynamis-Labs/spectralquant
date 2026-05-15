"""Pure water-filling utilities for SpectralQuant v2.

This module is a self-contained implementation of the greedy bit allocation
described in `docs/spectralquant_v2_technical_spec.md` §9.3-9.4. The functions
here have no dependency on the rest of the SpectralQuant engine: they accept
numpy arrays, Python lists, or torch tensors (when torch is available) and
always return numpy arrays or JSON-safe Python primitives.

Allocation rule (greedy, per-bit):

    i* = argmax_i  lambda_i / 4 ** b_i

with deterministic tie-breaking by lowest index.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

try:  # optional torch support
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch absent in this env
    torch = None  # type: ignore
    _HAS_TORCH = False


ArrayLike = Union[np.ndarray, Sequence[float], "torch.Tensor"]  # noqa: F821

FORMULA_VERSION = "waterfill-v1"


def _to_float64_array(eigenvalues: ArrayLike) -> np.ndarray:
    """Coerce a numpy array, torch tensor, or python sequence to float64 numpy.

    The original input is never mutated. A torch tensor on any device is
    detached and copied to CPU before conversion.
    """
    if _HAS_TORCH and isinstance(eigenvalues, torch.Tensor):  # type: ignore[arg-type]
        arr = eigenvalues.detach().cpu().numpy()
    elif isinstance(eigenvalues, np.ndarray):
        arr = eigenvalues
    elif isinstance(eigenvalues, (list, tuple)):
        arr = np.asarray(eigenvalues)
    else:
        # Last-resort: try numpy's coercion. Will raise for unsupported types.
        arr = np.asarray(eigenvalues)
    # Always make a fresh float64 copy so callers' inputs are untouched.
    return np.array(arr, dtype=np.float64, copy=True)


def allocate_waterfill_bits(
    eigenvalues: ArrayLike,
    total_bits: int,
    min_bits: int = 0,
    max_bits: Optional[int] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Allocate integer bits across dimensions by greedy water-filling.

    Parameters
    ----------
    eigenvalues : 1-D non-negative array (numpy / torch / list).
    total_bits  : total integer bit budget to allocate, sum_i b_i = total_bits.
    min_bits    : per-dimension lower bound (default 0).
    max_bits    : per-dimension upper bound, or None for no cap.
    eps         : floor to avoid division blow-up when an eigenvalue is 0.

    Returns
    -------
    np.ndarray of int64 with shape (d,), summing to total_bits.

    Tie-breaking is deterministic: the lowest index wins when scores tie.
    """
    eig = _to_float64_array(eigenvalues)

    if eig.ndim != 1:
        raise ValueError(
            f"eigenvalues must be one-dimensional, got shape {eig.shape}"
        )
    if eig.size == 0:
        raise ValueError("eigenvalues must be non-empty")
    if not np.all(np.isfinite(eig)):
        raise ValueError("eigenvalues must be finite (no NaN/inf)")
    if np.any(eig < 0):
        raise ValueError("eigenvalues must be non-negative")

    if not isinstance(total_bits, (int, np.integer)) or isinstance(total_bits, bool):
        raise TypeError(f"total_bits must be a non-negative integer, got {type(total_bits).__name__}")
    if total_bits < 0:
        raise ValueError(f"total_bits must be non-negative, got {total_bits}")
    if not isinstance(min_bits, (int, np.integer)) or isinstance(min_bits, bool):
        raise TypeError("min_bits must be an integer")
    if min_bits < 0:
        raise ValueError(f"min_bits must be non-negative, got {min_bits}")
    if max_bits is not None:
        if not isinstance(max_bits, (int, np.integer)) or isinstance(max_bits, bool):
            raise TypeError("max_bits must be an integer or None")
        if max_bits < min_bits:
            raise ValueError(
                f"max_bits ({max_bits}) must be >= min_bits ({min_bits})"
            )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    d = int(eig.size)
    if total_bits < d * min_bits:
        raise ValueError(
            f"total_bits={total_bits} cannot satisfy min_bits={min_bits} over d={d} dims"
        )
    if max_bits is not None and total_bits > d * max_bits:
        raise ValueError(
            f"total_bits={total_bits} exceeds d*max_bits={d * max_bits}"
        )

    eig_safe = np.maximum(eig, eps)
    bits = np.full(d, int(min_bits), dtype=np.int64)
    remaining = int(total_bits) - d * int(min_bits)

    for _ in range(remaining):
        # Marginal gain per dimension under current allocation.
        scores = eig_safe / np.power(4.0, bits.astype(np.float64))
        if max_bits is not None:
            scores = np.where(bits >= max_bits, -np.inf, scores)
        # np.argmax already returns the lowest index on ties (deterministic).
        i = int(np.argmax(scores))
        bits[i] += 1

    return bits


def marginal_gain(
    eigenvalues: ArrayLike,
    bits: ArrayLike,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-dimension marginal score lambda_i / 4 ** b_i.

    This is the score the greedy allocator maximizes. It is exposed for
    diagnostics and for tests that want to confirm allocation properties
    without re-implementing the rule.
    """
    eig = _to_float64_array(eigenvalues)
    b = _to_float64_array(bits)
    if eig.shape != b.shape:
        raise ValueError(
            f"eigenvalues shape {eig.shape} does not match bits shape {b.shape}"
        )
    if eig.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    if np.any(eig < 0):
        raise ValueError("eigenvalues must be non-negative")
    if np.any(b < 0):
        raise ValueError("bits must be non-negative")
    eig_safe = np.maximum(eig, eps)
    return eig_safe / np.power(4.0, b)


def validate_bit_allocation(
    bits: ArrayLike,
    total_bits: int,
    min_bits: int = 0,
    max_bits: Optional[int] = None,
) -> None:
    """Raise ValueError if ``bits`` is not a valid water-fill allocation.

    Checks: integer-valued, 1-D, non-negative, in [min_bits, max_bits], sums
    to ``total_bits``.
    """
    b = _to_float64_array(bits)
    if b.ndim != 1:
        raise ValueError(f"bits must be one-dimensional, got shape {b.shape}")
    if b.size == 0:
        raise ValueError("bits must be non-empty")
    if not np.all(np.isfinite(b)):
        raise ValueError("bits must be finite")
    if not np.all(b == np.round(b)):
        raise ValueError("bits must all be integer-valued")
    b_int = b.astype(np.int64)
    if np.any(b_int < min_bits):
        raise ValueError(f"bits below min_bits={min_bits}: {b_int.tolist()}")
    if max_bits is not None and np.any(b_int > max_bits):
        raise ValueError(f"bits above max_bits={max_bits}: {b_int.tolist()}")
    s = int(b_int.sum())
    if s != int(total_bits):
        raise ValueError(
            f"bits sum to {s}, expected total_bits={total_bits}"
        )


def summarize_allocation(
    eigenvalues: ArrayLike,
    bits: ArrayLike,
) -> Dict[str, Any]:
    """Return a JSON-safe summary of an allocation for logging/serialization."""
    eig = _to_float64_array(eigenvalues)
    b = _to_float64_array(bits)
    if eig.shape != b.shape:
        raise ValueError(
            f"eigenvalues shape {eig.shape} does not match bits shape {b.shape}"
        )
    if eig.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    b_int = b.astype(np.int64)
    eig_sum = float(eig.sum())
    eig_share = (eig / eig_sum).tolist() if eig_sum > 0 else [0.0] * eig.size
    return {
        "formula_version": FORMULA_VERSION,
        "d": int(eig.size),
        "total_bits": int(b_int.sum()),
        "bits": [int(x) for x in b_int.tolist()],
        "min_bits": int(b_int.min()),
        "max_bits": int(b_int.max()),
        "mean_bits": float(b_int.mean()),
        "eigenvalues": [float(x) for x in eig.tolist()],
        "eigenvalue_sum": eig_sum,
        "eigenvalue_share": [float(x) for x in eig_share],
    }


__all__ = [
    "FORMULA_VERSION",
    "allocate_waterfill_bits",
    "marginal_gain",
    "validate_bit_allocation",
    "summarize_allocation",
]
