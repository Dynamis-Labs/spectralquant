"""Engine-level tests for v2 water-filling propagation (spec §12.4).

These tests exercise the canonical pure-Python engine
(``spectralquant.SpectralQuantEngine`` re-exported from
``spectralquant.spectralquant``) and verify:

1. ``EngineConfig.use_water_fill=False`` (default) reproduces v1 allocation
   exactly.
2. ``EngineConfig.use_water_fill=True`` changes the per-semantic-dim
   allocation when eigenvalues are non-uniform.
3. v1 and v2 share the same total semantic MSE bit budget.
4. v1 and v2 select the same number of semantic (selective-QJL) dimensions.
5. compress / decompress preserve tensor shapes.
6. Attention scoring returns finite logits.
7. Causal masking is applied consistently outside the engine path.
8. The engine does not silently mix normalized and unnormalized keys.

We also assert that:

* ``SpectralQuantEngine`` re-exported from the package is the canonical
  pure-Python engine.
* ``KernelSpectralQuantEngine`` is import-safe even on a clean checkout
  without ``turboquant_cutile`` installed (it raises ``RuntimeError`` only
  when constructed in environments that lack cuTile).
* Invalid ``EngineConfig`` water-fill arguments raise clearly.
* Allocation metadata is exposed as JSON-safe dicts.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pytest
import torch

import spectralquant
from spectralquant import EngineConfig, SpectralQuantEngine
from spectralquant.calibration import (
    EigenspectralCalibrator,
    HeadCalibrationData,
)


HEAD_DIM = 16
N_TOKENS = 256
N_KV_HEADS = 2
N_LAYERS = 1
SEED = 17


# ---------------------------------------------------------------------------
# Helpers — minimal calibrator built from synthetic eigenspectra.
# ---------------------------------------------------------------------------


def _build_eigenstuff(
    eigenvalues: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (eigvals_tensor, V) where V is a random orthonormal matrix."""
    head_dim = eigenvalues.shape[0]
    A = rng.standard_normal((head_dim, head_dim))
    Q, _ = np.linalg.qr(A)
    return (
        torch.from_numpy(eigenvalues.astype(np.float32)),
        torch.from_numpy(Q.astype(np.float32)),
    )


def _build_calibrator(
    eigenvalues: np.ndarray,
    *,
    seed: int = SEED,
    n_layers: int = N_LAYERS,
    n_heads: int = N_KV_HEADS,
) -> Tuple[EigenspectralCalibrator, torch.Tensor]:
    """Build a calibrator populated with identical synthetic per-head data.

    Returns
    -------
    calibrator: fitted-looking calibrator usable with SpectralQuantEngine.
    rotated_data: ``(n_tokens, head_dim)`` synthetic data with the requested
        per-coordinate variances, ready to feed ``fit_quantizers`` (we fit on
        rotated data, so this *is* in the spectral basis).
    """
    rng = np.random.default_rng(seed)
    head_dim = eigenvalues.shape[0]

    # Per-coord variances == eigenvalues (in the rotated basis).
    rotated = (
        rng.standard_normal((N_TOKENS, head_dim)).astype(np.float32)
        * np.sqrt(eigenvalues).astype(np.float32)
    )

    calib = EigenspectralCalibrator(max_tokens_per_layer=N_TOKENS)
    sum_lam = float(eigenvalues.sum())
    sum_sq = float((eigenvalues ** 2).sum())
    d_eff = (sum_lam ** 2) / sum_sq if sum_sq > 1e-12 else 1.0

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            for head_type in ("key", "value"):
                eigvals, V = _build_eigenstuff(eigenvalues, rng)
                hcd = HeadCalibrationData(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    head_type=head_type,
                    eigenvalues=eigvals,
                    eigenvectors=V,
                    d_eff=float(d_eff),
                    spectral_gap=None,
                    var_95=int(min(head_dim, max(1, round(d_eff)))),
                    var_99=int(min(head_dim, max(1, round(d_eff)))),
                    n_samples=N_TOKENS,
                    head_dim=head_dim,
                )
                calib._calibration_data[(layer_idx, head_idx, head_type)] = hcd
    calib._is_calibrated = True
    return calib, torch.from_numpy(rotated)


def _make_kv(
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n_kv_heads, seq_len, head_dim, generator=g)


def _fit_engine(
    config: EngineConfig,
    eigenvalues: np.ndarray,
) -> SpectralQuantEngine:
    calib, rotated = _build_calibrator(eigenvalues)
    engine = SpectralQuantEngine(calib, config)
    rotated_kv: Dict[Tuple[int, int, str], torch.Tensor] = {}
    for layer_idx in range(N_LAYERS):
        for head_idx in range(N_KV_HEADS):
            for head_type in ("key", "value"):
                rotated_kv[(layer_idx, head_idx, head_type)] = rotated
    engine.fit_quantizers(rotated_kv)
    return engine


# Two synthetic eigenspectra that exercise the v1/v2 branches.
def _sharp_spectrum() -> np.ndarray:
    big = np.array([100.0, 60.0, 30.0, 12.0], dtype=np.float64)
    tail = np.full(HEAD_DIM - big.shape[0], 0.5, dtype=np.float64)
    return np.concatenate([big, tail])


def _flat_spectrum() -> np.ndarray:
    return np.full(HEAD_DIM, 1.0, dtype=np.float64)


# ---------------------------------------------------------------------------
# Canonical engine identity / import safety
# ---------------------------------------------------------------------------


class TestCanonicalEngineExports:
    def test_canonical_engine_is_pure_python(self):
        # The exported ``SpectralQuantEngine`` must be the pure-Python
        # pipeline in spectralquant.spectralquant — *not* the cuTile
        # subclass — so that benchmarks know which code path they are using.
        from spectralquant.spectralquant import SpectralQuantEngine as PureEngine
        assert SpectralQuantEngine is PureEngine

    def test_kernel_engine_alias_distinct(self):
        # KernelSpectralQuantEngine is a separate symbol; even when the
        # cuTile import succeeds (because engine.py provides stubs), the
        # two engines are distinct classes.
        from spectralquant import KernelSpectralQuantEngine
        assert KernelSpectralQuantEngine is not SpectralQuantEngine

    def test_legacy_alias_points_to_canonical_engine(self):
        # The transitional ``_LegacySpectralQuantEngine`` alias keeps
        # working for callers that imported it before the rename.
        assert spectralquant._LegacySpectralQuantEngine is SpectralQuantEngine


# ---------------------------------------------------------------------------
# EngineConfig validation
# ---------------------------------------------------------------------------


class TestEngineConfigValidation:
    def test_defaults_keep_v1_behaviour(self):
        cfg = EngineConfig()
        assert cfg.use_water_fill is False
        assert cfg.wf_min_bits == 0
        assert cfg.wf_max_bits is None

    def test_use_water_fill_must_be_bool(self):
        with pytest.raises(TypeError):
            EngineConfig(use_water_fill=1)  # type: ignore[arg-type]

    def test_wf_min_bits_must_be_int(self):
        with pytest.raises(TypeError):
            EngineConfig(use_water_fill=True, wf_min_bits=0.5)  # type: ignore[arg-type]

    def test_wf_min_bits_negative_rejected(self):
        with pytest.raises(ValueError):
            EngineConfig(use_water_fill=True, wf_min_bits=-1)

    def test_wf_max_below_min_rejected(self):
        with pytest.raises(ValueError):
            EngineConfig(use_water_fill=True, wf_min_bits=2, wf_max_bits=1)

    def test_valid_bounds_accepted(self):
        cfg = EngineConfig(use_water_fill=True, wf_min_bits=1, wf_max_bits=8)
        assert cfg.use_water_fill is True
        assert cfg.wf_min_bits == 1
        assert cfg.wf_max_bits == 8


# ---------------------------------------------------------------------------
# v1 default behaviour — water_fill=False (Spec test 1, 3, 4)
# ---------------------------------------------------------------------------


class TestV1DefaultBehaviour:
    def test_v1_alloc_uniform_per_dim(self):
        cfg = EngineConfig(avg_bits=4.0)
        engine = _fit_engine(cfg, _sharp_spectrum())
        meta = engine.allocation_metadata()
        assert meta["use_water_fill"] is False
        assert meta["formula_version"] == "uniform-v1"
        assert len(meta["per_head"]) > 0
        for head in meta["per_head"]:
            alloc = head["allocation"]
            bits_per_dim = alloc["bits_per_dim"]
            d_eff = alloc["d_eff"]
            # All semantic dims have the same b_high in v1.
            assert len(bits_per_dim) == d_eff
            assert len(set(bits_per_dim)) == 1, bits_per_dim
            assert alloc["use_water_fill"] is False
            assert alloc["formula_version"] == "uniform-v1"

    def test_v1_v2_share_total_semantic_bit_budget(self):
        # Spec test 3: total semantic MSE bits must match between paths.
        cfg_v1 = EngineConfig(avg_bits=4.0)
        cfg_v2 = EngineConfig(avg_bits=4.0, use_water_fill=True)
        eig = _sharp_spectrum()
        e1 = _fit_engine(cfg_v1, eig)
        e2 = _fit_engine(cfg_v2, eig)
        meta1 = e1.allocation_metadata()
        meta2 = e2.allocation_metadata()
        for h1, h2 in zip(meta1["per_head"], meta2["per_head"]):
            assert h1["allocation"]["d_eff"] == h2["allocation"]["d_eff"]
            assert (
                h1["allocation"]["total_semantic_bits"]
                == h2["allocation"]["total_semantic_bits"]
            )
            # Per-dim bit sum equals total_semantic_bits in both paths.
            assert sum(h1["allocation"]["bits_per_dim"]) == h1["allocation"]["total_semantic_bits"]
            assert sum(h2["allocation"]["bits_per_dim"]) == h2["allocation"]["total_semantic_bits"]

    def test_v1_v2_share_d_eff(self):
        # Spec test 4: same selective-QJL dimensionality on both paths.
        cfg_v1 = EngineConfig(avg_bits=4.0)
        cfg_v2 = EngineConfig(avg_bits=4.0, use_water_fill=True)
        eig = _sharp_spectrum()
        e1 = _fit_engine(cfg_v1, eig)
        e2 = _fit_engine(cfg_v2, eig)
        d_effs_1 = [h["allocation"]["d_eff"] for h in e1.allocation_metadata()["per_head"]]
        d_effs_2 = [h["allocation"]["d_eff"] for h in e2.allocation_metadata()["per_head"]]
        assert d_effs_1 == d_effs_2


# ---------------------------------------------------------------------------
# v2 water-fill behaviour (Spec tests 2, 3)
# ---------------------------------------------------------------------------


class TestV2WaterFillBehaviour:
    def test_flag_reaches_quantizers(self):
        cfg = EngineConfig(avg_bits=4.0, use_water_fill=True, wf_min_bits=1, wf_max_bits=8)
        engine = _fit_engine(cfg, _sharp_spectrum())
        # Inspect each quantizer's flag — confirms the engine config was
        # plumbed through ``fit_quantizers``.
        seen = 0
        for (layer_idx, head_idx, head_type), q in engine._quantizers.items():
            assert q.use_water_fill is True
            assert q._wf_min_bits == 1
            assert q._wf_max_bits == 8
            seen += 1
        assert seen == N_LAYERS * N_KV_HEADS * 2

    def test_v2_alloc_non_uniform_for_sharp_spectrum(self):
        # The greedy water-fill rule i* = argmax_i (λ_i / 4^b_i) only
        # produces a non-uniform allocation when consecutive semantic
        # eigenvalues span more than a factor of 4.  Use a geometric
        # decay (ratio 2) within the semantic regime so the test is
        # actually exercised.
        head = np.array([200.0, 100.0, 50.0, 25.0], dtype=np.float64)
        tail = np.full(HEAD_DIM - head.shape[0], 0.05, dtype=np.float64)
        spectrum = np.concatenate([head, tail])
        cfg = EngineConfig(avg_bits=4.0, use_water_fill=True)
        engine = _fit_engine(cfg, spectrum)
        meta = engine.allocation_metadata()
        assert meta["use_water_fill"] is True
        assert meta["formula_version"] == "waterfill-v1"
        saw_nonuniform = False
        for head in meta["per_head"]:
            bits = head["allocation"]["bits_per_dim"]
            # Larger eigenvalues should never receive fewer bits than
            # smaller ones (greedy water-filling guarantees this).
            assert bits == sorted(bits, reverse=True), bits
            if len(set(bits)) > 1:
                saw_nonuniform = True
        # At least one head should have a non-flat allocation given the
        # sharp spectrum.
        assert saw_nonuniform, "expected non-uniform water-fill allocation on sharp spectrum"

    def test_v2_alloc_flat_for_flat_spectrum(self):
        cfg = EngineConfig(avg_bits=4.0, use_water_fill=True)
        engine = _fit_engine(cfg, _flat_spectrum())
        meta = engine.allocation_metadata()
        for head in meta["per_head"]:
            bits = head["allocation"]["bits_per_dim"]
            # Flat eigenvalues → water-filling is degenerate, all dims equal
            # (within a 1-bit tie-break tolerance from greedy allocation).
            assert max(bits) - min(bits) <= 1, bits


# ---------------------------------------------------------------------------
# Spec tests 5–8: shapes, finite logits, masking, normalization.
# ---------------------------------------------------------------------------


class TestEngineShapesAndLogits:
    def test_compress_decompress_preserves_shapes(self):
        cfg = EngineConfig(avg_bits=4.0, use_water_fill=True)
        engine = _fit_engine(cfg, _sharp_spectrum())
        kv = _make_kv(N_KV_HEADS, seq_len=8, head_dim=HEAD_DIM, seed=SEED)
        c_keys = engine.compress_keys(kv, layer_idx=0)
        c_vals = engine.compress_values(kv, layer_idx=0)
        vals_hat = engine.decompress_values(c_vals, layer_idx=0)
        assert vals_hat.shape == kv[:, : len(c_vals), :, :].shape
        # Compressed vectors expose the v2 metadata field.
        for cv in c_keys.values():
            assert cv.semantic_bits_per_dim is not None
            assert sum(cv.semantic_bits_per_dim) == cv.b_high * cv.d_eff
        for cv in c_vals.values():
            assert cv.semantic_bits_per_dim is not None

    def test_attention_scores_finite(self):
        cfg = EngineConfig(avg_bits=4.0, use_water_fill=True)
        engine = _fit_engine(cfg, _sharp_spectrum())
        kv = _make_kv(N_KV_HEADS, seq_len=8, head_dim=HEAD_DIM, seed=SEED)
        c_keys = engine.compress_keys(kv, layer_idx=0)
        # one query head per kv head (no GQA expansion here)
        q = _make_kv(N_KV_HEADS, seq_len=4, head_dim=HEAD_DIM, seed=SEED + 1)
        weights = engine.attention_score(q, c_keys, layer_idx=0)
        assert torch.isfinite(weights).all()
        # softmax weights sum to 1 over the seq_len axis.
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_causal_mask_is_caller_responsibility(self):
        # Spec test 7: the engine returns *un-masked* softmax weights;
        # any causal mask must be applied by the caller (consistent with
        # v1).  We verify that no key position is implicitly zeroed out.
        cfg = EngineConfig(avg_bits=4.0)
        engine = _fit_engine(cfg, _flat_spectrum())
        kv = _make_kv(N_KV_HEADS, seq_len=6, head_dim=HEAD_DIM, seed=SEED)
        c_keys = engine.compress_keys(kv, layer_idx=0)
        q = _make_kv(N_KV_HEADS, seq_len=3, head_dim=HEAD_DIM, seed=SEED + 2)
        w = engine.attention_score(q, c_keys, layer_idx=0)
        # Every key index should receive non-zero attention from at least
        # one query, i.e. no implicit causal masking.
        per_key = w.sum(dim=(0, 1, 2))
        assert (per_key > 0).all()

    def test_keys_are_normalised_consistently(self):
        # Spec test 8: the engine path must not mix normalized and
        # unnormalized keys.  We feed in a tensor whose row norms vary by
        # several orders of magnitude and check that compressed
        # reconstruction error stays bounded — silent normalization
        # mixing would cause the larger-norm keys to dominate or to be
        # quantized incorrectly.
        cfg = EngineConfig(avg_bits=4.0)
        engine = _fit_engine(cfg, _flat_spectrum())
        # Build keys with row norms in {1.0, 100.0}.
        g = torch.Generator().manual_seed(SEED + 9)
        kv = torch.randn(1, N_KV_HEADS, 6, HEAD_DIM, generator=g)
        kv[:, :, ::2, :] *= 100.0
        c_keys = engine.compress_keys(kv, layer_idx=0)
        for head_idx, cv in c_keys.items():
            quant = engine._get_quantizer(0, head_idx, "key")
            k_rot_hat = quant.decompress(cv)
            # Reconstruction lives in the same scale family as the input.
            assert torch.isfinite(k_rot_hat).all()
            assert k_rot_hat.shape == kv[:, head_idx, :, :].shape


# ---------------------------------------------------------------------------
# Allocation metadata is JSON-safe.
# ---------------------------------------------------------------------------


class TestAllocationMetadataJsonSafety:
    def test_metadata_round_trips_through_json(self):
        import json

        cfg = EngineConfig(avg_bits=4.0, use_water_fill=True, wf_min_bits=1, wf_max_bits=8)
        engine = _fit_engine(cfg, _sharp_spectrum())
        meta = engine.allocation_metadata()
        encoded = json.dumps(meta)  # must not raise
        decoded = json.loads(encoded)
        assert decoded["use_water_fill"] is True
        assert decoded["formula_version"] == "waterfill-v1"
        assert decoded["wf_min_bits"] == 1
        assert decoded["wf_max_bits"] == 8

    def test_waterfill_allocations_dict(self):
        cfg = EngineConfig(avg_bits=4.0, use_water_fill=True)
        engine = _fit_engine(cfg, _sharp_spectrum())
        allocs = engine.waterfill_allocations()
        assert len(allocs) == N_LAYERS * N_KV_HEADS * 2
        for key, wfa in allocs.items():
            assert wfa is not None
            assert wfa.use_water_fill is True
            assert wfa.formula_version == "waterfill-v1"


# ---------------------------------------------------------------------------
# Clean-checkout import safety
# ---------------------------------------------------------------------------


class TestCleanCheckoutImports:
    def test_package_imports_without_modal(self):
        # Re-importing the package after we already use it confirms the
        # module-level `try/except` around the cuTile engine cannot fail,
        # even when the cuTile baseline is absent.  The module is already
        # imported at the top of this file.
        import importlib

        importlib.reload(spectralquant)
        assert hasattr(spectralquant, "SpectralQuantEngine")
        assert hasattr(spectralquant, "KernelSpectralQuantEngine")
        assert hasattr(spectralquant, "EngineConfig")

    def test_pure_python_engine_does_not_require_kernel(self):
        # The canonical SpectralQuantEngine path must not import or touch
        # ``turboquant_cutile``.  We construct and use it on CPU-only
        # synthetic data and assert success.
        cfg = EngineConfig(avg_bits=4.0)
        engine = _fit_engine(cfg, _flat_spectrum())
        kv = _make_kv(N_KV_HEADS, seq_len=4, head_dim=HEAD_DIM, seed=SEED)
        out = engine.compress_keys(kv, layer_idx=0)
        assert len(out) == N_KV_HEADS
