"""Compression-accounting utilities for SpectralQuant v2.

This module computes per-method compression ratios from explicit bit-layout
components rather than from headline targets. It is deliberately decoupled
from the rest of the engine so it can be exercised by unit tests and by a
future ``run_compression_accounting_audit.py`` script without loading any
heavy dependencies.

The exposed dataclasses match the schema in
``schemas/accounting.schema.json`` and the API in
``docs/spectralquant_v2_technical_spec.md`` §10.

Two important rules from the spec are enforced here:

1. Every reported compression ratio is *derived* from stored bit components.
   Hard-coded ratios are forbidden (see G3 in ``docs/claims_discipline.md``).
2. The simple appendix formula for SpectralQuant given in spec §10 does not
   yield the report's headline 5.95x for ``b=3, d=3``. ``check_headline_ratio``
   surfaces that discrepancy explicitly rather than papering over it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Formula identifiers
# ---------------------------------------------------------------------------

# These strings are written into ``CompressionAccounting.formula_version`` so
# that any saved artifact can be unambiguously matched to the formula that
# produced it.
TURBOQUANT_FORMULA_VERSION = "turboquant-spec-v1"
SPECTRALQUANT_SPEC_FORMULA_VERSION = "spectralquant-spec-v1"
SPECTRALQUANT_FLEXIBLE_FORMULA_VERSION = "spectralquant-flex-v1"

FP16_BITS_PER_SCALAR = 16


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class CompressionAccounting:
    """Bit-level accounting for one (method, operating point).

    All ``*_bits`` quantities are *per KV slot* (i.e. per (token, KV head))
    in units of bits. ``head_dim`` is the per-head feature dimension.

    The compression ratio is computed against an FP16 reference of
    ``2 * head_dim * 16`` bits per (K, V) pair averaged to per-slot
    ``head_dim * 16``.
    """

    method: str
    avg_bits_arg: int
    head_dim: int
    d_eff: Optional[int]
    k_mse_bits: float
    k_qjl_bits: float
    k_norm_bits: float
    v_mse_bits: float
    v_norm_bits: float
    total_k_bits: float
    total_v_bits: float
    average_slot_bits: float
    fp16_slot_bits: float
    compression_ratio: float
    formula_version: str
    waterfill_allocation: Optional[List[int]] = None
    notes: Optional[str] = None

    # ---- helpers -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict conforming to ``schemas/accounting.schema.json``."""
        d = asdict(self)
        # Drop optional fields when not used so the dict stays clean.
        if d["waterfill_allocation"] is None:
            d.pop("waterfill_allocation")
        if d.get("notes") is None:
            d.pop("notes", None)
        return d

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, blob: Dict[str, Any]) -> "CompressionAccounting":
        # Permissive: ignore unknown keys so saved artifacts that gain extra
        # fields in the future still load.
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in blob.items() if k in known})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(avg_bits: int, head_dim: int) -> None:
    if not isinstance(avg_bits, int) or isinstance(avg_bits, bool):
        raise TypeError(f"avg_bits must be int, got {type(avg_bits).__name__}")
    if avg_bits < 1:
        raise ValueError(f"avg_bits must be >= 1, got {avg_bits}")
    if not isinstance(head_dim, int) or isinstance(head_dim, bool):
        raise TypeError("head_dim must be int")
    if head_dim < 1:
        raise ValueError(f"head_dim must be >= 1, got {head_dim}")


def _ratio_from_components(
    head_dim: int,
    total_k_bits: float,
    total_v_bits: float,
) -> tuple[float, float, float]:
    """Return (avg_slot_bits, fp16_slot_bits, ratio)."""
    avg_slot = (total_k_bits + total_v_bits) / 2.0
    fp16_slot = float(head_dim * FP16_BITS_PER_SCALAR)
    if avg_slot <= 0:
        raise ValueError("average_slot_bits must be positive")
    return avg_slot, fp16_slot, fp16_slot / avg_slot


# ---------------------------------------------------------------------------
# TurboQuant accounting
# ---------------------------------------------------------------------------


def turboquant_accounting(
    avg_bits: int,
    head_dim: int = 128,
    *,
    k_norm_bits: float = 32.0,
    v_norm_bits: float = 16.0,
) -> CompressionAccounting:
    """TurboQuant per-slot bit accounting per spec §10.

    Layout (per spec):

      K = (b - 1) * D + D + norm_K = b * D + norm_K
        - (b - 1) * D  : MSE quant residual after one QJL bit per dim
        - + D          : full-dimensional 1-bit QJL signs
        - + norm_K     : 32-bit K norm
      V = b * D + norm_V
        - b * D        : MSE quant
        - + norm_V     : 16-bit V norm

    With ``head_dim = 128`` and the default norms this reproduces the
    headline ratios:

      b=3 -> 2048 / 408 ≈ 5.02x
      b=5 -> 2048 / 664 ≈ 3.08x
    """
    _validate_inputs(avg_bits, head_dim)

    # K side: (b-1) MSE + 1 QJL per dim collapses to b * D + norm.
    k_mse_bits = float((avg_bits - 1) * head_dim)
    k_qjl_bits = float(head_dim)  # 1 bit per dim, full dimensional
    v_mse_bits = float(avg_bits * head_dim)

    total_k = k_mse_bits + k_qjl_bits + k_norm_bits
    total_v = v_mse_bits + v_norm_bits
    avg_slot, fp16_slot, ratio = _ratio_from_components(head_dim, total_k, total_v)

    return CompressionAccounting(
        method="turboquant",
        avg_bits_arg=avg_bits,
        head_dim=head_dim,
        d_eff=None,
        k_mse_bits=k_mse_bits,
        k_qjl_bits=k_qjl_bits,
        k_norm_bits=float(k_norm_bits),
        v_mse_bits=v_mse_bits,
        v_norm_bits=float(v_norm_bits),
        total_k_bits=total_k,
        total_v_bits=total_v,
        average_slot_bits=avg_slot,
        fp16_slot_bits=fp16_slot,
        compression_ratio=ratio,
        formula_version=TURBOQUANT_FORMULA_VERSION,
    )


# ---------------------------------------------------------------------------
# SpectralQuant accounting
# ---------------------------------------------------------------------------


def spectralquant_accounting(
    avg_bits: int,
    head_dim: int,
    d_eff: int,
    *,
    k_mse_bits: float,
    k_qjl_bits: float,
    v_mse_bits: float,
    method: str = "spectralquant_v2",
    k_norm_bits: float = 32.0,
    v_norm_bits: float = 16.0,
    waterfill_allocation: Optional[Sequence[int]] = None,
    formula_version: str = SPECTRALQUANT_FLEXIBLE_FORMULA_VERSION,
    notes: Optional[str] = None,
) -> CompressionAccounting:
    """Flexible SpectralQuant accounting from explicit K/V bit components.

    Unlike :func:`turboquant_accounting`, this does not assume a single
    formula. Callers must pass the actual per-slot K MSE bits, K QJL bits,
    and V MSE bits the engine will write. This is the path the v2 engine
    will use to log accounting alongside each run, so that the reported
    ratio matches the bits actually stored.

    Use :func:`spectralquant_spec_accounting` for the simple spec §10
    formula (which is known not to reproduce the 5.95x headline).
    """
    if method not in ("spectralquant_v1", "spectralquant_v2"):
        raise ValueError(
            f"method must be spectralquant_v1 or spectralquant_v2, got {method!r}"
        )
    _validate_inputs(avg_bits, head_dim)

    if not (0 <= d_eff <= head_dim):
        raise ValueError(f"d_eff must be in [0, {head_dim}], got {d_eff}")
    for label, val in (
        ("k_mse_bits", k_mse_bits),
        ("k_qjl_bits", k_qjl_bits),
        ("k_norm_bits", k_norm_bits),
        ("v_mse_bits", v_mse_bits),
        ("v_norm_bits", v_norm_bits),
    ):
        if val < 0 or not math.isfinite(val):
            raise ValueError(f"{label} must be a finite non-negative number, got {val}")

    if waterfill_allocation is not None:
        alloc = [int(b) for b in waterfill_allocation]
        if any(b < 0 for b in alloc):
            raise ValueError("waterfill_allocation must be non-negative")
        if len(alloc) != d_eff:
            raise ValueError(
                f"waterfill_allocation length {len(alloc)} does not match d_eff={d_eff}"
            )
    else:
        alloc = None

    total_k = float(k_mse_bits) + float(k_qjl_bits) + float(k_norm_bits)
    total_v = float(v_mse_bits) + float(v_norm_bits)
    avg_slot, fp16_slot, ratio = _ratio_from_components(head_dim, total_k, total_v)

    return CompressionAccounting(
        method=method,
        avg_bits_arg=avg_bits,
        head_dim=head_dim,
        d_eff=int(d_eff),
        k_mse_bits=float(k_mse_bits),
        k_qjl_bits=float(k_qjl_bits),
        k_norm_bits=float(k_norm_bits),
        v_mse_bits=float(v_mse_bits),
        v_norm_bits=float(v_norm_bits),
        total_k_bits=total_k,
        total_v_bits=total_v,
        average_slot_bits=avg_slot,
        fp16_slot_bits=fp16_slot,
        compression_ratio=ratio,
        formula_version=formula_version,
        waterfill_allocation=alloc,
        notes=notes,
    )


def spectralquant_spec_accounting(
    avg_bits: int,
    d_eff: int,
    head_dim: int = 128,
    *,
    method: str = "spectralquant_v2",
    k_norm_bits: float = 32.0,
    v_norm_bits: float = 16.0,
    waterfill_allocation: Optional[Sequence[int]] = None,
) -> CompressionAccounting:
    """Implements the simple spec §10 SpectralQuant formula:

        K = b * D + d_eff + norm_K   (full b-bit MSE + selective QJL on d_eff)
        V = b * D + norm_V

    This is *deliberately* not what the engine should report — the spec notes
    that this formula does not yield 5.95x at (b=3, d_eff=3). It exists as a
    reference for the discrepancy check in :func:`check_headline_ratio` and
    for documentation, not as the canonical engine output.
    """
    _validate_inputs(avg_bits, head_dim)
    if not (0 <= d_eff <= head_dim):
        raise ValueError(f"d_eff must be in [0, {head_dim}], got {d_eff}")

    k_mse_bits = float(avg_bits * head_dim)
    k_qjl_bits = float(d_eff)
    v_mse_bits = float(avg_bits * head_dim)

    return spectralquant_accounting(
        avg_bits=avg_bits,
        head_dim=head_dim,
        d_eff=d_eff,
        k_mse_bits=k_mse_bits,
        k_qjl_bits=k_qjl_bits,
        v_mse_bits=v_mse_bits,
        method=method,
        k_norm_bits=k_norm_bits,
        v_norm_bits=v_norm_bits,
        waterfill_allocation=waterfill_allocation,
        formula_version=SPECTRALQUANT_SPEC_FORMULA_VERSION,
        notes=(
            "Simple spec §10 formula. Known not to reproduce the 5.95x headline "
            "at (avg_bits=3, d_eff=3, head_dim=128); see check_headline_ratio."
        ),
    )


# ---------------------------------------------------------------------------
# Discrepancy check
# ---------------------------------------------------------------------------


@dataclass
class HeadlineRatioCheck:
    """Result of comparing a computed ratio against a headline target."""

    label: str
    target_ratio: float
    computed_ratio: float
    tolerance: float
    matches: bool
    formula_version: str
    diagnostic: str


def check_headline_ratio(
    accounting: CompressionAccounting,
    target_ratio: float,
    *,
    tolerance: float = 0.05,
    label: Optional[str] = None,
) -> HeadlineRatioCheck:
    """Compare a computed compression ratio against a stated headline.

    The check fails *softly* — it returns a ``HeadlineRatioCheck`` with
    ``matches=False`` and a human-readable diagnostic, instead of pretending
    the computed value matches. Tests assert on ``matches``.
    """
    if not math.isfinite(target_ratio) or target_ratio <= 0:
        raise ValueError(f"target_ratio must be positive, got {target_ratio}")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    delta = accounting.compression_ratio - target_ratio
    matches = abs(delta) <= tolerance
    diag = (
        f"{accounting.method}@b={accounting.avg_bits_arg}"
        f" d_eff={accounting.d_eff} D={accounting.head_dim}: "
        f"computed={accounting.compression_ratio:.4f}, "
        f"target={target_ratio:.4f}, delta={delta:+.4f}, "
        f"tol={tolerance:.4f} -> {'OK' if matches else 'MISMATCH'}"
    )
    return HeadlineRatioCheck(
        label=label or f"{accounting.method}_b{accounting.avg_bits_arg}",
        target_ratio=float(target_ratio),
        computed_ratio=float(accounting.compression_ratio),
        tolerance=float(tolerance),
        matches=matches,
        formula_version=accounting.formula_version,
        diagnostic=diag,
    )


__all__ = [
    "CompressionAccounting",
    "HeadlineRatioCheck",
    "TURBOQUANT_FORMULA_VERSION",
    "SPECTRALQUANT_SPEC_FORMULA_VERSION",
    "SPECTRALQUANT_FLEXIBLE_FORMULA_VERSION",
    "FP16_BITS_PER_SCALAR",
    "turboquant_accounting",
    "spectralquant_accounting",
    "spectralquant_spec_accounting",
    "check_headline_ratio",
]
