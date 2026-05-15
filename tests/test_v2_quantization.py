"""Synthetic tests for the v2 water-filling integration in NonUniformQuantizer.

These tests exercise the per-semantic-dimension water-fill path added in
``src/spectralquant/nonuniform_quantization.py``. They are deliberately
small and deterministic — no model loading, no GPU, no calibration runs —
and they focus on the contract of the integration:

1. ``use_water_fill=False`` is the default and reproduces the v1 single-
   semantic-codebook path exactly (allocation metadata still populated).
2. ``use_water_fill=True`` produces a non-uniform allocation when
   eigenvalues are non-uniform.
3. The total semantic bit budget is preserved between v1 and v2 for the
   same (d_eff, b_high).
4. Allocation metadata fields have the right shapes and types.
5. Invalid inputs fail clearly.
6. compress / decompress preserves shapes on the v2 path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectralquant.nonuniform_quantization import (
    CompressedVector,
    NonUniformQuantizer,
    WaterfillAllocation,
)
from spectralquant.waterfill import FORMULA_VERSION as WATERFILL_FORMULA_VERSION


HEAD_DIM = 16
N_TOKENS = 512
SEED = 13


def _sharp_spectrum(d_eff: int = 4, head_dim: int = HEAD_DIM) -> np.ndarray:
    """Return a strictly descending eigenspectrum with a sharp gap at d_eff."""
    big = np.array([100.0, 60.0, 30.0, 12.0])[:d_eff].astype(np.float64)
    if d_eff < 4:
        big = big[:d_eff]
    if d_eff > 4:
        # Pad with descending values if the caller asked for more semantic dims.
        extra = np.linspace(8.0, 5.0, d_eff - 4, dtype=np.float64)
        big = np.concatenate([big, extra])
    tail = np.full(head_dim - d_eff, 0.5, dtype=np.float64)
    return np.concatenate([big, tail])


def _flat_spectrum(head_dim: int = HEAD_DIM) -> np.ndarray:
    return np.full(head_dim, 1.0, dtype=np.float64)


def _make_rotated_data(eigenvalues: np.ndarray, n_tokens: int = N_TOKENS) -> torch.Tensor:
    """Synthesize 'rotated' data with given per-coord variances."""
    rng = np.random.default_rng(SEED)
    head_dim = eigenvalues.shape[0]
    x = rng.standard_normal((n_tokens, head_dim)).astype(np.float32)
    x *= np.sqrt(eigenvalues).astype(np.float32)[None, :]
    return torch.from_numpy(x)


# ---------------------------------------------------------------------------
# Defaults / v1 backward compatibility
# ---------------------------------------------------------------------------


class TestV1BackwardCompatibility:
    def test_default_flag_is_false(self):
        eig = torch.from_numpy(_sharp_spectrum())
        q = NonUniformQuantizer(eigenvalues=eig, avg_bits=4.0, seed=SEED)
        assert q.use_water_fill is False
        # Metadata is None until fit.
        assert q.waterfill_allocation is None
        assert q.semantic_bits_per_dim is None

    def test_v1_uses_single_semantic_codebook(self):
        eig_np = _sharp_spectrum()
        data = _make_rotated_data(eig_np)
        q = NonUniformQuantizer(
            eigenvalues=torch.from_numpy(eig_np), avg_bits=4.0, seed=SEED
        )
        q.fit(data, d_eff=4)

        # v1: one shared semantic codebook, no per-dim list.
        assert q._semantic_quantizer is not None
        assert q._per_dim_semantic_quantizers is None

        meta = q.waterfill_allocation
        assert isinstance(meta, WaterfillAllocation)
        assert meta.use_water_fill is False
        assert meta.formula_version == "uniform-v1"
        # Uniform v1 allocation is just [b_high] * d_eff.
        assert meta.bits_per_dim == [q._b_high] * q._d_eff_int
        assert meta.total_semantic_bits == q._b_high * q._d_eff_int

    def test_v1_compress_decompress_unchanged_with_flag_off(self):
        # Re-fitting twice with flag off must be deterministic and produce
        # identical outputs (no hidden water-fill state leaks in).
        eig_np = _sharp_spectrum()
        data = _make_rotated_data(eig_np)
        eig = torch.from_numpy(eig_np)

        q1 = NonUniformQuantizer(eigenvalues=eig, avg_bits=4.0, seed=SEED)
        q1.fit(data, d_eff=4)
        c1 = q1.compress(data)

        q2 = NonUniformQuantizer(eigenvalues=eig, avg_bits=4.0, seed=SEED)
        q2.fit(data, d_eff=4)
        c2 = q2.compress(data)

        assert c1.semantic_bits_per_dim is None
        assert c2.semantic_bits_per_dim is None
        assert torch.equal(c1.semantic_indices, c2.semantic_indices)
        assert torch.equal(c1.tail_indices, c2.tail_indices)
        assert c1.actual_bits_used == c2.actual_bits_used
        assert c1.b_high == c2.b_high
        assert c1.b_low == c2.b_low


# ---------------------------------------------------------------------------
# v2 water-fill path
# ---------------------------------------------------------------------------


class TestV2Allocation:
    def test_non_uniform_eigenvalues_change_allocation(self):
        eig_np = _sharp_spectrum()
        data = _make_rotated_data(eig_np)
        eig = torch.from_numpy(eig_np)

        q_v1 = NonUniformQuantizer(eigenvalues=eig, avg_bits=4.0, seed=SEED)
        q_v1.fit(data, d_eff=4)
        v1_alloc = q_v1.semantic_bits_per_dim
        assert v1_alloc is not None

        q_v2 = NonUniformQuantizer(
            eigenvalues=eig, avg_bits=4.0, seed=SEED, use_water_fill=True
        )
        q_v2.fit(data, d_eff=4)
        v2_alloc = q_v2.semantic_bits_per_dim
        assert v2_alloc is not None

        # Non-uniform spectrum -> v2 allocation must differ from the uniform v1.
        assert v2_alloc != v1_alloc
        # Larger eigenvalues should not receive fewer bits than smaller ones.
        # i.e. allocation is non-increasing along a strictly descending spectrum.
        for i in range(len(v2_alloc) - 1):
            assert v2_alloc[i] >= v2_alloc[i + 1]

    def test_total_semantic_bit_budget_is_preserved(self):
        eig_np = _sharp_spectrum()
        data = _make_rotated_data(eig_np)
        eig = torch.from_numpy(eig_np)

        q_v1 = NonUniformQuantizer(eigenvalues=eig, avg_bits=4.0, seed=SEED)
        q_v1.fit(data, d_eff=4)

        q_v2 = NonUniformQuantizer(
            eigenvalues=eig, avg_bits=4.0, seed=SEED, use_water_fill=True
        )
        q_v2.fit(data, d_eff=4)

        v1_total = q_v1.waterfill_allocation.total_semantic_bits
        v2_total = q_v2.waterfill_allocation.total_semantic_bits
        assert v1_total == v2_total
        assert sum(q_v2.semantic_bits_per_dim) == v2_total
        # And v2's total equals b_high * d_eff exactly.
        assert v2_total == q_v2._b_high * q_v2._d_eff_int

    def test_flat_spectrum_collapses_to_uniform(self):
        # With a flat spectrum, water-filling should fill round-robin and
        # produce a uniform allocation when the budget is divisible by d_eff.
        eig_np = _flat_spectrum()
        data = _make_rotated_data(eig_np)
        eig = torch.from_numpy(eig_np)

        q = NonUniformQuantizer(
            eigenvalues=eig, avg_bits=4.0, seed=SEED, use_water_fill=True
        )
        q.fit(data, d_eff=4)
        assert q.semantic_bits_per_dim is not None
        # Round-robin from index 0 with a divisible budget -> uniform.
        assert len(set(q.semantic_bits_per_dim)) == 1

    def test_allocation_metadata_shape_and_types(self):
        eig_np = _sharp_spectrum()
        data = _make_rotated_data(eig_np)
        eig = torch.from_numpy(eig_np)

        q = NonUniformQuantizer(
            eigenvalues=eig,
            avg_bits=4.0,
            seed=SEED,
            use_water_fill=True,
            wf_min_bits=0,
            wf_max_bits=8,
        )
        q.fit(data, d_eff=4)

        meta = q.waterfill_allocation
        assert isinstance(meta, WaterfillAllocation)
        assert meta.use_water_fill is True
        assert meta.d_eff == 4
        assert len(meta.bits_per_dim) == 4
        assert all(isinstance(b, int) for b in meta.bits_per_dim)
        assert len(meta.eigenvalues) == 4
        assert meta.formula_version == WATERFILL_FORMULA_VERSION
        assert meta.min_bits == 0
        assert meta.max_bits == 8
        assert meta.actual_min_bits == min(meta.bits_per_dim)
        assert meta.actual_max_bits == max(meta.bits_per_dim)

        d = meta.to_dict()
        # JSON-safe round-trip: every value is a primitive / list of primitives.
        for key in (
            "use_water_fill",
            "eigenvalues",
            "bits_per_dim",
            "d_eff",
            "total_semantic_bits",
            "min_bits",
            "max_bits",
            "actual_min_bits",
            "actual_max_bits",
            "formula_version",
        ):
            assert key in d

    def test_per_dim_codebook_count_matches_d_eff(self):
        eig_np = _sharp_spectrum()
        data = _make_rotated_data(eig_np)
        eig = torch.from_numpy(eig_np)
        q = NonUniformQuantizer(
            eigenvalues=eig, avg_bits=4.0, seed=SEED, use_water_fill=True
        )
        q.fit(data, d_eff=4)
        assert q._per_dim_semantic_quantizers is not None
        assert len(q._per_dim_semantic_quantizers) == 4

        for cb, b_i in zip(
            q._per_dim_semantic_quantizers, q.semantic_bits_per_dim or []
        ):
            # When the allocator gives the dim 0 bits we deliberately keep a
            # 1-bit Lloyd-Max instance with collapsed centroids; otherwise
            # n_bits matches the allocation exactly.
            if b_i == 0:
                assert cb.n_bits == 1
            else:
                assert cb.n_bits == b_i


# ---------------------------------------------------------------------------
# Compress / decompress shape and roundtrip on the v2 path
# ---------------------------------------------------------------------------


class TestV2CompressDecompress:
    def test_v2_compress_decompress_shapes(self):
        eig_np = _sharp_spectrum()
        data = _make_rotated_data(eig_np)
        eig = torch.from_numpy(eig_np)
        q = NonUniformQuantizer(
            eigenvalues=eig, avg_bits=4.0, seed=SEED, use_water_fill=True
        )
        q.fit(data, d_eff=4)

        comp = q.compress(data)
        assert isinstance(comp, CompressedVector)
        assert comp.semantic_indices.shape == (N_TOKENS, 4)
        assert comp.tail_indices.shape == (N_TOKENS, HEAD_DIM - 4)
        assert comp.semantic_bits_per_dim == q.semantic_bits_per_dim

        recon = q.decompress(comp)
        assert recon.shape == data.shape
        assert recon.dtype == torch.float32

    def test_v2_actual_bits_match_per_dim_allocation(self):
        eig_np = _sharp_spectrum()
        data = _make_rotated_data(eig_np)
        eig = torch.from_numpy(eig_np)
        q = NonUniformQuantizer(
            eigenvalues=eig, avg_bits=4.0, seed=SEED, use_water_fill=True
        )
        q.fit(data, d_eff=4)

        comp = q.compress(data)
        d_eff = comp.d_eff
        d_tail = HEAD_DIM - d_eff
        per_vec = sum(q.semantic_bits_per_dim) + d_tail * q._b_low
        expected = N_TOKENS * per_vec
        assert comp.actual_bits_used == float(expected)

    def test_v2_batched_input_shapes(self):
        # Ensure the per-dim path works with leading batch dims, not just
        # (n_tokens, head_dim).
        eig_np = _sharp_spectrum()
        eig = torch.from_numpy(eig_np)
        q = NonUniformQuantizer(
            eigenvalues=eig, avg_bits=4.0, seed=SEED, use_water_fill=True
        )
        train = _make_rotated_data(eig_np)
        q.fit(train, d_eff=4)

        rng = np.random.default_rng(SEED + 1)
        batched = rng.standard_normal((3, 7, HEAD_DIM)).astype(np.float32)
        batched *= np.sqrt(eig_np).astype(np.float32)[None, None, :]
        x = torch.from_numpy(batched)
        comp = q.compress(x)
        assert comp.semantic_indices.shape == (3, 7, 4)
        assert comp.tail_indices.shape == (3, 7, HEAD_DIM - 4)
        recon = q.decompress(comp)
        assert recon.shape == x.shape

    def test_compress_override_rejected_for_water_fill(self):
        eig_np = _sharp_spectrum()
        data = _make_rotated_data(eig_np)
        eig = torch.from_numpy(eig_np)
        q = NonUniformQuantizer(
            eigenvalues=eig, avg_bits=4.0, seed=SEED, use_water_fill=True
        )
        q.fit(data, d_eff=4)
        # Changing avg_bits would change b_high and invalidate the per-dim
        # codebooks that were fit for the original budget.
        with pytest.raises(ValueError, match="water-fill"):
            q.compress(data, avg_bits=2.0)


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


class TestV2InvalidInputs:
    def test_eigenvalues_must_be_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            NonUniformQuantizer(
                eigenvalues=torch.zeros(4, 4), avg_bits=4.0, use_water_fill=True
            )

    def test_use_water_fill_must_be_bool(self):
        eig = torch.from_numpy(_sharp_spectrum())
        with pytest.raises(TypeError, match="use_water_fill"):
            NonUniformQuantizer(eigenvalues=eig, use_water_fill=1)  # type: ignore[arg-type]

    def test_min_bits_must_be_non_negative(self):
        eig = torch.from_numpy(_sharp_spectrum())
        with pytest.raises(ValueError, match="wf_min_bits"):
            NonUniformQuantizer(
                eigenvalues=eig, use_water_fill=True, wf_min_bits=-1
            )

    def test_max_bits_must_be_ge_min_bits(self):
        eig = torch.from_numpy(_sharp_spectrum())
        with pytest.raises(ValueError, match="wf_max_bits"):
            NonUniformQuantizer(
                eigenvalues=eig,
                use_water_fill=True,
                wf_min_bits=2,
                wf_max_bits=1,
            )

    def test_negative_eigenvalues_rejected_by_allocator(self):
        eig_np = _sharp_spectrum().copy()
        eig_np[2] = -1.0  # poison one eigenvalue
        eig = torch.from_numpy(eig_np)
        data = _make_rotated_data(np.abs(eig_np))
        q = NonUniformQuantizer(
            eigenvalues=eig, avg_bits=4.0, seed=SEED, use_water_fill=True
        )
        with pytest.raises(ValueError, match="non-negative"):
            q.fit(data, d_eff=4)
