import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from experiments.codex_wht_vs_pca import WHTPolarQuantEngine, build_wht_basis, solve_polarquant_centroids


def _reference_polarquant_roundtrip(x: torch.Tensor, basis: torch.Tensor, bits: int) -> torch.Tensor:
    d = x.shape[-1]
    centroids = solve_polarquant_centroids(bits, d).to(x.device)
    norms = torch.norm(x.float(), dim=-1, keepdim=True)
    rotated = (x.float() / (norms + 1e-8)) @ basis
    quant = centroids[(rotated.unsqueeze(-1) - centroids).abs().argmin(dim=-1)]
    quant_norms = torch.norm(quant, dim=-1, keepdim=True)
    quant = quant / torch.where(quant_norms > 1e-10, quant_norms, torch.ones_like(quant_norms))
    return (quant @ basis.T) * norms


def test_polarquant_centroids_match_tom_reference_formulas():
    one_bit = solve_polarquant_centroids(1, 128)
    expected_one_bit = torch.tensor(
        [-math.sqrt(2.0 / (math.pi * 128)), math.sqrt(2.0 / (math.pi * 128))],
        dtype=torch.float32,
    )
    assert torch.allclose(one_bit, expected_one_bit)

    two_bit = solve_polarquant_centroids(2, 128)
    expected_two_bit = torch.tensor([-1.51, -0.453, 0.453, 1.51], dtype=torch.float32) / math.sqrt(128)
    assert torch.allclose(two_bit, expected_two_bit)


def test_wht_engine_matches_tom_style_reference_roundtrip_for_keys_and_values():
    hd = 128
    k_basis = build_wht_basis(hd, seed=17)
    v_basis = build_wht_basis(hd, seed=29)
    engine = WHTPolarQuantEngine(k_basis, v_basis, key_bits_per_dim=1, value_bits_per_dim=4, hd=hd)

    g = torch.Generator().manual_seed(123)
    keys = torch.randn(8, hd, generator=g, dtype=torch.float32) * 7.5
    values = torch.randn(8, hd, generator=g, dtype=torch.float32) * 3.0

    key_hat = engine.decompress(engine.compress_keys(keys), engine.vk_t)
    val_hat = engine.decompress(engine.compress_values(values), engine.vv_t)

    ref_key_hat = _reference_polarquant_roundtrip(keys, k_basis, bits=1)
    ref_val_hat = _reference_polarquant_roundtrip(values, v_basis, bits=4)

    assert torch.allclose(key_hat, ref_key_hat, atol=1e-6, rtol=1e-6)
    assert torch.allclose(val_hat, ref_val_hat, atol=1e-6, rtol=1e-6)


def test_wht_engine_norm_correction_preserves_zero_vectors():
    hd = 64
    basis = build_wht_basis(hd, seed=5)
    engine = WHTPolarQuantEngine(basis, basis, key_bits_per_dim=3, value_bits_per_dim=3, hd=hd)
    zeros = torch.zeros(4, hd, dtype=torch.float32)

    key_hat = engine.decompress(engine.compress_keys(zeros), engine.vk_t)
    val_hat = engine.decompress(engine.compress_values(zeros), engine.vv_t)

    assert torch.allclose(key_hat, zeros, atol=1e-8, rtol=1e-8)
    assert torch.allclose(val_hat, zeros, atol=1e-8, rtol=1e-8)
