"""Unit tests for `spectralquant.accounting`.

Coverage corresponds to spec §10 plus the explicit-discrepancy requirement:

- TurboQuant 3-bit and 5-bit ratios match documented values (~5.02x, ~3.08x).
- SpectralQuant flexible formula round-trips through K/V components.
- v1 and v2 (flexible-form) yield identical compression when only the
  semantic bit allocation reshuffles inside a fixed budget.
- ``CompressionAccounting`` round-trips through JSON.
- ``check_headline_ratio`` fires when the simple spec formula is asked to
  reproduce 5.95x at (b=3, d_eff=3, head_dim=128) — the known discrepancy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load `spectralquant.accounting` directly to avoid importing
# `src/spectralquant/__init__.py`, which transitively requires torch.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ACCOUNTING_PATH = _REPO_ROOT / "src" / "spectralquant" / "accounting.py"
_spec = importlib.util.spec_from_file_location(
    "spectralquant_accounting", _ACCOUNTING_PATH
)
assert _spec is not None and _spec.loader is not None
accounting = importlib.util.module_from_spec(_spec)
sys.modules["spectralquant_accounting"] = accounting
_spec.loader.exec_module(accounting)

CompressionAccounting = accounting.CompressionAccounting
turboquant_accounting = accounting.turboquant_accounting
spectralquant_accounting = accounting.spectralquant_accounting
spectralquant_spec_accounting = accounting.spectralquant_spec_accounting
check_headline_ratio = accounting.check_headline_ratio
TURBOQUANT_FORMULA_VERSION = accounting.TURBOQUANT_FORMULA_VERSION
SPECTRALQUANT_SPEC_FORMULA_VERSION = accounting.SPECTRALQUANT_SPEC_FORMULA_VERSION
SPECTRALQUANT_FLEXIBLE_FORMULA_VERSION = accounting.SPECTRALQUANT_FLEXIBLE_FORMULA_VERSION


# ---------------------------------------------------------------------------
# TurboQuant
# ---------------------------------------------------------------------------


def test_turboquant_3bit_ratio_is_5_02():
    a = turboquant_accounting(avg_bits=3, head_dim=128)
    # Documented spec §10 ratio: 2048 / 408 = 5.0196...
    assert a.compression_ratio == pytest.approx(2048 / 408, rel=1e-9)
    assert a.compression_ratio == pytest.approx(5.02, abs=0.01)
    assert a.formula_version == TURBOQUANT_FORMULA_VERSION
    assert a.method == "turboquant"
    assert a.d_eff is None


def test_turboquant_5bit_ratio_is_3_08():
    a = turboquant_accounting(avg_bits=5, head_dim=128)
    assert a.compression_ratio == pytest.approx(2048 / 664, rel=1e-9)
    assert a.compression_ratio == pytest.approx(3.08, abs=0.01)


def test_turboquant_components_sum_correctly():
    a = turboquant_accounting(avg_bits=3, head_dim=128)
    assert a.total_k_bits == a.k_mse_bits + a.k_qjl_bits + a.k_norm_bits
    assert a.total_v_bits == a.v_mse_bits + a.v_norm_bits
    assert a.average_slot_bits == pytest.approx(
        (a.total_k_bits + a.total_v_bits) / 2.0
    )
    assert a.fp16_slot_bits == 128 * 16


def test_turboquant_qjl_is_full_dim():
    a = turboquant_accounting(avg_bits=4, head_dim=64)
    # QJL is one bit per dim.
    assert a.k_qjl_bits == 64


def test_turboquant_invalid_inputs_raise():
    with pytest.raises(ValueError):
        turboquant_accounting(avg_bits=0)
    with pytest.raises(ValueError):
        turboquant_accounting(avg_bits=3, head_dim=0)
    with pytest.raises(TypeError):
        turboquant_accounting(avg_bits="3")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SpectralQuant — flexible
# ---------------------------------------------------------------------------


def test_spectralquant_flexible_round_trip_components():
    """The flexible builder writes back the exact components it was given."""
    a = spectralquant_accounting(
        avg_bits=3,
        head_dim=128,
        d_eff=4,
        k_mse_bits=380.0,
        k_qjl_bits=4.0,
        v_mse_bits=384.0,
    )
    assert a.method == "spectralquant_v2"
    assert a.k_mse_bits == 380.0
    assert a.k_qjl_bits == 4.0
    assert a.v_mse_bits == 384.0
    assert a.total_k_bits == 380.0 + 4.0 + 32.0
    assert a.total_v_bits == 384.0 + 16.0
    assert a.formula_version == SPECTRALQUANT_FLEXIBLE_FORMULA_VERSION


def test_spectralquant_flexible_rejects_bad_method():
    with pytest.raises(ValueError):
        spectralquant_accounting(
            avg_bits=3,
            head_dim=128,
            d_eff=3,
            k_mse_bits=384.0,
            k_qjl_bits=3.0,
            v_mse_bits=384.0,
            method="turboquant",
        )


def test_spectralquant_flexible_rejects_negative_components():
    with pytest.raises(ValueError):
        spectralquant_accounting(
            avg_bits=3,
            head_dim=128,
            d_eff=3,
            k_mse_bits=-1.0,
            k_qjl_bits=3.0,
            v_mse_bits=384.0,
        )


def test_spectralquant_flexible_rejects_d_eff_out_of_range():
    with pytest.raises(ValueError):
        spectralquant_accounting(
            avg_bits=3,
            head_dim=128,
            d_eff=200,
            k_mse_bits=384.0,
            k_qjl_bits=3.0,
            v_mse_bits=384.0,
        )


def test_spectralquant_flexible_waterfill_length_must_match_d_eff():
    with pytest.raises(ValueError, match="waterfill_allocation length"):
        spectralquant_accounting(
            avg_bits=3,
            head_dim=128,
            d_eff=3,
            k_mse_bits=384.0,
            k_qjl_bits=3.0,
            v_mse_bits=384.0,
            waterfill_allocation=[3, 3],
        )


def test_spectralquant_flexible_accepts_matching_waterfill():
    a = spectralquant_accounting(
        avg_bits=3,
        head_dim=128,
        d_eff=3,
        k_mse_bits=384.0,
        k_qjl_bits=3.0,
        v_mse_bits=384.0,
        waterfill_allocation=[5, 3, 1],
    )
    assert a.waterfill_allocation == [5, 3, 1]


# ---------------------------------------------------------------------------
# SpectralQuant — simple spec formula
# ---------------------------------------------------------------------------


def test_spectralquant_spec_formula_b3_d3():
    """Simple spec §10 formula at (b=3, d_eff=3, D=128).

    Spec yields:
        K = 384 + 3 + 32 = 419
        V = 384 + 16     = 400
        avg = 409.5
        ratio = 2048 / 409.5 ≈ 5.001
    Critically, this is NOT 5.95x.
    """
    a = spectralquant_spec_accounting(avg_bits=3, d_eff=3, head_dim=128)
    assert a.total_k_bits == 419.0
    assert a.total_v_bits == 400.0
    assert a.average_slot_bits == pytest.approx(409.5, rel=1e-9)
    assert a.compression_ratio == pytest.approx(2048 / 409.5, rel=1e-9)
    assert a.compression_ratio < 5.1  # well below 5.95
    assert a.formula_version == SPECTRALQUANT_SPEC_FORMULA_VERSION
    assert a.notes is not None
    assert "5.95" in a.notes  # discrepancy is documented inline


def test_spectralquant_spec_formula_method_default_is_v2():
    a = spectralquant_spec_accounting(avg_bits=3, d_eff=3, head_dim=128)
    assert a.method == "spectralquant_v2"


def test_spectralquant_spec_formula_method_v1_allowed():
    a = spectralquant_spec_accounting(
        avg_bits=3, d_eff=3, head_dim=128, method="spectralquant_v1"
    )
    assert a.method == "spectralquant_v1"


# ---------------------------------------------------------------------------
# v1 vs v2 with only allocation changing
# ---------------------------------------------------------------------------


def test_v1_v2_equal_compression_when_only_allocation_changes():
    """When the engine only reshuffles bits *within* the semantic budget,
    total stored bits — and therefore compression — must be identical."""
    head_dim = 128
    d_eff = 4
    avg_bits = 3
    semantic_budget = avg_bits * d_eff  # 12 bits

    # v1: uniform allocation [3, 3, 3, 3]
    v1 = spectralquant_accounting(
        avg_bits=avg_bits,
        head_dim=head_dim,
        d_eff=d_eff,
        k_mse_bits=float(avg_bits * head_dim),  # uniform b across all dims
        k_qjl_bits=float(d_eff),
        v_mse_bits=float(avg_bits * head_dim),
        method="spectralquant_v1",
        waterfill_allocation=[3, 3, 3, 3],
    )
    # v2: water-filled allocation [6, 3, 2, 1] — same sum.
    v2 = spectralquant_accounting(
        avg_bits=avg_bits,
        head_dim=head_dim,
        d_eff=d_eff,
        k_mse_bits=float(avg_bits * head_dim),
        k_qjl_bits=float(d_eff),
        v_mse_bits=float(avg_bits * head_dim),
        method="spectralquant_v2",
        waterfill_allocation=[6, 3, 2, 1],
    )
    assert sum(v1.waterfill_allocation or []) == semantic_budget
    assert sum(v2.waterfill_allocation or []) == semantic_budget
    assert v1.compression_ratio == v2.compression_ratio
    assert v1.average_slot_bits == v2.average_slot_bits


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def test_json_round_trip_preserves_values():
    a = turboquant_accounting(avg_bits=3, head_dim=128)
    blob = a.to_json()
    parsed = json.loads(blob)
    assert parsed["method"] == "turboquant"
    assert parsed["compression_ratio"] == pytest.approx(a.compression_ratio)
    assert parsed["formula_version"] == TURBOQUANT_FORMULA_VERSION
    # waterfill_allocation should not appear when None.
    assert "waterfill_allocation" not in parsed


def test_json_round_trip_with_waterfill():
    a = spectralquant_accounting(
        avg_bits=3,
        head_dim=128,
        d_eff=3,
        k_mse_bits=384.0,
        k_qjl_bits=3.0,
        v_mse_bits=384.0,
        waterfill_allocation=[5, 3, 1],
    )
    parsed = json.loads(a.to_json())
    assert parsed["waterfill_allocation"] == [5, 3, 1]


def test_dataclass_from_dict_drops_unknown_fields():
    blob = turboquant_accounting(avg_bits=3, head_dim=128).to_dict()
    blob["future_unknown_field"] = 123
    restored = CompressionAccounting.from_dict(blob)
    assert restored.method == "turboquant"
    assert restored.compression_ratio == pytest.approx(2048 / 408)


def test_accounting_validates_against_schema():
    """Each accounting object must validate against `accounting.schema.json`."""
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = _REPO_ROOT / "schemas" / "accounting.schema.json"
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    for obj in (
        turboquant_accounting(avg_bits=3, head_dim=128),
        turboquant_accounting(avg_bits=5, head_dim=128),
        spectralquant_spec_accounting(avg_bits=3, d_eff=3, head_dim=128),
        spectralquant_accounting(
            avg_bits=3,
            head_dim=128,
            d_eff=4,
            k_mse_bits=384.0,
            k_qjl_bits=4.0,
            v_mse_bits=384.0,
            waterfill_allocation=[5, 4, 2, 1],
        ),
    ):
        errors = list(validator.iter_errors(obj.to_dict()))
        assert not errors, f"schema errors: {errors}"


# ---------------------------------------------------------------------------
# Headline-ratio discrepancy detection
# ---------------------------------------------------------------------------


def test_check_headline_ratio_passes_for_turboquant():
    a = turboquant_accounting(avg_bits=3, head_dim=128)
    chk = check_headline_ratio(a, target_ratio=5.02, tolerance=0.01)
    assert chk.matches, chk.diagnostic
    assert chk.formula_version == TURBOQUANT_FORMULA_VERSION
    assert "OK" in chk.diagnostic


def test_check_headline_ratio_detects_5_95_discrepancy():
    """The simple SpectralQuant spec formula must NOT match 5.95x at (b=3, d=3)."""
    a = spectralquant_spec_accounting(avg_bits=3, d_eff=3, head_dim=128)
    chk = check_headline_ratio(a, target_ratio=5.95, tolerance=0.05)
    assert not chk.matches, (
        "Spec §10 SpectralQuant formula unexpectedly matches the 5.95x headline."
        f" Computed ratio: {a.compression_ratio:.4f}."
    )
    assert "MISMATCH" in chk.diagnostic
    # The computed ratio is around 5.00x.
    assert chk.computed_ratio == pytest.approx(2048 / 409.5, rel=1e-9)


def test_check_headline_ratio_invalid_target_raises():
    a = turboquant_accounting(avg_bits=3, head_dim=128)
    with pytest.raises(ValueError):
        check_headline_ratio(a, target_ratio=-1.0)
    with pytest.raises(ValueError):
        check_headline_ratio(a, target_ratio=5.0, tolerance=-0.1)


def test_check_headline_ratio_diagnostic_includes_method_and_bits():
    a = turboquant_accounting(avg_bits=5, head_dim=128)
    chk = check_headline_ratio(a, target_ratio=3.08, tolerance=0.01)
    assert "turboquant" in chk.diagnostic
    assert "b=5" in chk.diagnostic
