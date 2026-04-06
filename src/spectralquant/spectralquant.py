"""
Main pipeline class for SpectralQuant KV cache compression.

Provides:
- ``EngineConfig``: Dataclass controlling compression hyperparameters.
- ``SpectralQuantEngine``: Full pipeline — spectral rotation → non-uniform
  quantization → selective QJL.
- ``TurboQuantBaseline``: Baseline matching TurboQuant (random rotation +
  uniform quantization + full QJL).

Both classes share an identical interface so they can be swapped in any
evaluation script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectralquant.calibration import EigenspectralCalibrator, HeadCalibrationData
from spectralquant.spectral_rotation import SpectralRotation, RandomRotation
from spectralquant.nonuniform_quantization import (
    BitAllocator,
    LloydMaxQuantizer,
    NonUniformQuantizer,
    CompressedVector,
)
from spectralquant.selective_qjl import SelectiveQJL, FullQJL
from spectralquant.metrics import (
    cosine_similarity,
    weighted_mse,
    inner_product_error,
    compression_ratio,
    max_absolute_weight_error,
)
from spectralquant.utils import get_model_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Hyperparameters for :class:`SpectralQuantEngine`.

    Attributes
    ----------
    avg_bits:
        Target average bits per key/value element (default 4.0).
    qjl_projections:
        Number of QJL sketch dimensions (default 64).
    lloyd_max_iter:
        Maximum iterations for Lloyd-Max codebook fitting (default 200).
    lloyd_seed:
        Random seed for Lloyd-Max initialisation.
    rotation_seed:
        Random seed used by :class:`~spectralquant.spectral_rotation.RandomRotation`
        (only relevant for the baseline).
    n_calibration_tokens:
        Max tokens per layer to collect during calibration-time quantizer
        fitting.  Used when calling :meth:`SpectralQuantEngine.fit_quantizers`.
    use_value_rotation:
        If ``True``, rotate value vectors as well as keys.
    """

    avg_bits: float = 4.0
    qjl_projections: int = 64
    lloyd_max_iter: int = 200
    lloyd_seed: int = 0
    rotation_seed: int = 42
    n_calibration_tokens: int = 10_000
    use_value_rotation: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _softmax_attn_weights(
    scores: torch.Tensor, scale: Optional[float] = None
) -> torch.Tensor:
    """Apply scaling and softmax to raw attention scores.

    Parameters
    ----------
    scores:
        Raw dot-product scores.  Shape ``(..., n_queries, seq_len)``.
    scale:
        Optional pre-scale (default: 1/sqrt(head_dim) inferred if possible).

    Returns
    -------
    torch.Tensor
        Softmax attention weights, same shape as ``scores``.
    """
    if scale is not None:
        scores = scores * scale
    return torch.softmax(scores, dim=-1)


# ---------------------------------------------------------------------------
# SpectralQuantEngine
# ---------------------------------------------------------------------------

class SpectralQuantEngine:
    """Full SpectralQuant compression pipeline.

    Pipeline per head:

    1. **Spectral rotation**: Rotate KV vectors into the eigenbasis using
       calibrated V^T (energy concentrated in first d_eff coords).
    2. **Non-uniform quantization**: Allocate b_high bits to the semantic
       regime and b_low bits to the tail regime via Lloyd-Max codebooks.
    3. **Selective QJL**: Estimate attention scores using only the top d_eff
       rotated-key coordinates.

    Decompression pipeline:

    1. Dequantize the two regimes using stored codebooks.
    2. Inverse-rotate (multiply by V) to return to the original basis.

    Parameters
    ----------
    calibrator:
        A fitted :class:`~spectralquant.calibration.EigenspectralCalibrator`.
    config:
        Engine hyperparameters.  Defaults to :class:`EngineConfig`.

    Examples
    --------
    >>> engine = SpectralQuantEngine(calibrator, EngineConfig(avg_bits=4.0))
    >>> engine.fit_quantizers(rotated_keys_per_layer)
    >>> c_keys = engine.compress_keys(keys, layer_idx=0)
    >>> c_vals = engine.compress_values(values, layer_idx=0)
    >>> attn = engine.attention_score(queries, c_keys, layer_idx=0)
    >>> vals_hat = engine.decompress_values(c_vals, layer_idx=0)
    """

    def __init__(
        self,
        calibrator: EigenspectralCalibrator,
        config: Optional[EngineConfig] = None,
    ) -> None:
        self._calibrator = calibrator
        self._config = config or EngineConfig()
        self._key_rotation = SpectralRotation(calibrator, head_type="key")
        self._val_rotation = SpectralRotation(calibrator, head_type="value")
        self._qjl = SelectiveQJL(
            n_projections=self._config.qjl_projections,
            seed=self._config.rotation_seed,
        )
        # Quantizers fitted per (layer_idx, head_idx, head_type)
        self._quantizers: Dict[Tuple[int, int, str], NonUniformQuantizer] = {}
        self._is_fitted: bool = False

    def fit_quantizers(
        self,
        rotated_kv_data: Dict[Tuple[int, int, str], torch.Tensor],
    ) -> None:
        """Fit per-head Lloyd-Max quantizers from pre-rotated KV data.

        Parameters
        ----------
        rotated_kv_data:
            Dictionary keyed by ``(layer_idx, head_idx, head_type)`` mapping
            to rotated data tensors of shape ``(n_tokens, head_dim)``.
        """
        for (layer_idx, head_idx, head_type), data in rotated_kv_data.items():
            hcd = self._calibrator.get(layer_idx, head_idx, head_type)
            if hcd is None:
                logger.warning(
                    "No calibration data for L%d H%d %s; skipping quantizer fitting.",
                    layer_idx, head_idx, head_type,
                )
                continue
            quant = NonUniformQuantizer(
                eigenvalues=hcd.eigenvalues,
                avg_bits=self._config.avg_bits,
                max_lloyd_iter=self._config.lloyd_max_iter,
                seed=self._config.lloyd_seed,
            ).fit(data, d_eff=hcd.d_eff)
            self._quantizers[(layer_idx, head_idx, head_type)] = quant
        self._is_fitted = True
        logger.info("Fitted %d per-head quantizers.", len(self._quantizers))

    def _get_quantizer(
        self, layer_idx: int, head_idx: int, head_type: str
    ) -> NonUniformQuantizer:
        key = (layer_idx, head_idx, head_type)
        if key not in self._quantizers:
            raise KeyError(
                f"No quantizer for layer={layer_idx}, head={head_idx}, "
                f"type='{head_type}'.  Call fit_quantizers() first."
            )
        return self._quantizers[key]

    # ------------------------------------------------------------------
    # Public compression API
    # ------------------------------------------------------------------

    def compress_keys(
        self,
        keys: torch.Tensor,
        layer_idx: int,
    ) -> Dict[int, CompressedVector]:
        """Compress all key heads for a given layer.

        Parameters
        ----------
        keys:
            Key tensor of shape ``(batch, n_kv_heads, seq_len, head_dim)``.
        layer_idx:
            Transformer layer index.

        Returns
        -------
        Dict[int, CompressedVector]
            Per-head compressed representations keyed by ``head_idx``.
        """
        n_kv_heads = keys.shape[1]
        result: Dict[int, CompressedVector] = {}
        for head_idx in range(n_kv_heads):
            # keys[:, head_idx, :, :] -> (batch, seq_len, head_dim)
            k_h = keys[:, head_idx, :, :]
            k_rot = self._key_rotation.rotate(k_h, layer_idx, head_idx)
            quant = self._get_quantizer(layer_idx, head_idx, "key")
            compressed = quant.compress(k_rot)
            result[head_idx] = compressed
        return result

    def compress_values(
        self,
        values: torch.Tensor,
        layer_idx: int,
    ) -> Dict[int, CompressedVector]:
        """Compress all value heads for a given layer.

        Parameters
        ----------
        values:
            Value tensor of shape ``(batch, n_kv_heads, seq_len, head_dim)``.
        layer_idx:
            Transformer layer index.

        Returns
        -------
        Dict[int, CompressedVector]
            Per-head compressed representations keyed by ``head_idx``.
        """
        n_kv_heads = values.shape[1]
        result: Dict[int, CompressedVector] = {}
        for head_idx in range(n_kv_heads):
            v_h = values[:, head_idx, :, :]
            if self._config.use_value_rotation:
                v_rot = self._val_rotation.rotate(v_h, layer_idx, head_idx)
            else:
                v_rot = v_h
            quant = self._get_quantizer(layer_idx, head_idx, "value")
            compressed = quant.compress(v_rot)
            result[head_idx] = compressed
        return result

    def decompress_values(
        self,
        compressed: Dict[int, CompressedVector],
        layer_idx: int,
    ) -> torch.Tensor:
        """Reconstruct value tensor from compressed representations.

        Parameters
        ----------
        compressed:
            Output of :meth:`compress_values`, keyed by ``head_idx``.
        layer_idx:
            Transformer layer index.

        Returns
        -------
        torch.Tensor
            Reconstructed values of shape
            ``(batch, n_kv_heads, seq_len, head_dim)``.
        """
        head_indices = sorted(compressed.keys())
        heads: List[torch.Tensor] = []
        for head_idx in head_indices:
            cv = compressed[head_idx]
            quant = self._get_quantizer(layer_idx, head_idx, "value")
            v_rot_hat = quant.decompress(cv)  # (..., head_dim)
            if self._config.use_value_rotation:
                v_hat = self._val_rotation.unrotate(v_rot_hat, layer_idx, head_idx)
            else:
                v_hat = v_rot_hat
            heads.append(v_hat)

        # Stack along head dimension: list of (batch, seq_len, head_dim)
        # -> (batch, n_kv_heads, seq_len, head_dim)
        return torch.stack(heads, dim=1)

    def attention_score(
        self,
        queries: torch.Tensor,
        compressed_keys: Dict[int, CompressedVector],
        layer_idx: int,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute attention weights using QJL-based score estimation.

        Combines:
        1. Selective QJL inner-product estimate over the semantic regime.
        2. Exact inner product over the (dequantized) tail regime.

        Parameters
        ----------
        queries:
            Query tensor of shape ``(batch, n_heads, n_queries, head_dim)``.
        compressed_keys:
            Output of :meth:`compress_keys`, keyed by ``head_idx``.
        layer_idx:
            Transformer layer index.
        scale:
            Attention scale factor (default: ``1 / sqrt(head_dim)``).

        Returns
        -------
        torch.Tensor
            Softmax attention weights of shape
            ``(batch, n_heads, n_queries, seq_len)``.
        """
        batch, n_heads, n_queries, head_dim = queries.shape
        if scale is None:
            scale = head_dim ** -0.5

        all_weights: List[torch.Tensor] = []

        # Assume n_heads >= n_kv_heads (GQA: map query heads to kv heads)
        n_kv_heads = len(compressed_keys)
        head_ratio = n_heads // n_kv_heads

        for head_idx, cv in sorted(compressed_keys.items()):
            # Query heads that attend to this kv head
            q_start = head_idx * head_ratio
            q_end = q_start + head_ratio
            q_group = queries[:, q_start:q_end, :, :]  # (batch, head_ratio, n_q, d)

            # Rotate queries into spectral basis
            # Process each query head in the group
            q_rot_list: List[torch.Tensor] = []
            for qi in range(head_ratio):
                q_h = q_group[:, qi, :, :]  # (batch, n_q, d)
                q_rot_list.append(
                    self._key_rotation.rotate(q_h, layer_idx, head_idx)
                )
            q_rot = torch.stack(q_rot_list, dim=1)  # (batch, head_ratio, n_q, d)

            # Reconstruct rotated keys (semantic + tail separately)
            quant = self._get_quantizer(layer_idx, head_idx, "key")
            k_rot_hat = quant.decompress(cv)  # (batch, seq_len, d) or (..., d)

            d_eff = cv.d_eff

            # Selective QJL score for semantic regime
            # Process each query head
            scores_list: List[torch.Tensor] = []
            for qi in range(head_ratio):
                q_h_rot = q_rot[:, qi, :, :]  # (batch, n_q, d)
                # k_rot_hat: (batch, seq_len, d) or (seq_len, d) etc.
                k_h_rot = k_rot_hat  # same for all query heads in group

                qjl_score = self._qjl.compute_correction(
                    keys=k_h_rot,  # (batch, seq_len, d)
                    queries=q_h_rot,  # (batch, n_q, d)
                    d_eff=d_eff,
                )  # (batch, n_q, seq_len)

                # Tail exact contribution: q_tail · k_tail
                q_tail = q_h_rot[..., d_eff:]  # (batch, n_q, d-d_eff)
                k_tail = k_h_rot[..., d_eff:]  # (batch, seq_len, d-d_eff)
                tail_score = torch.bmm(q_tail, k_tail.transpose(-2, -1))  # (batch, n_q, seq)

                total_score = qjl_score + tail_score  # (batch, n_q, seq)
                scores_list.append(total_score)

            # (batch, head_ratio, n_q, seq_len)
            group_scores = torch.stack(scores_list, dim=1)
            weights = _softmax_attn_weights(group_scores * scale)
            all_weights.append(weights)

        # Concatenate all head groups: (batch, n_heads, n_q, seq)
        return torch.cat(all_weights, dim=1)

    def get_compression_ratio(
        self,
        keys: Optional[torch.Tensor] = None,
    ) -> float:
        """Return the average bits-per-element achieved by the engine.

        If ``keys`` is provided, the ratio is computed for that specific
        tensor.  Otherwise, the theoretical ratio based on the config is
        returned.

        Parameters
        ----------
        keys:
            Optional key tensor for an empirical estimate.

        Returns
        -------
        float
            Compression ratio (original_bits / compressed_bits).
        """
        original_bpe = 16  # fp16 baseline
        compressed_bpe = self._config.avg_bits
        return original_bpe / compressed_bpe

    def compare_with_baseline(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        baseline: "TurboQuantBaseline",
        layer_idx: int = 0,
    ) -> Dict[str, Any]:
        """Compare SpectralQuant vs TurboQuant baseline on a batch of KV.

        Parameters
        ----------
        keys:
            Key tensor of shape ``(batch, n_kv_heads, seq_len, head_dim)``.
        values:
            Value tensor, same shape as ``keys``.
        queries:
            Query tensor of shape ``(batch, n_heads, n_queries, head_dim)``.
        baseline:
            A :class:`TurboQuantBaseline` instance to compare against.
        layer_idx:
            Layer index (for rotation lookup).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing, for each method (``"spectralquant"`` and
            ``"turboquant"``), the keys:

            * ``"key_cosine_sim"``: Mean cosine similarity of compressed keys.
            * ``"value_cosine_sim"``: Mean cosine similarity of reconstructed values.
            * ``"key_mse"``: MSE of key reconstruction.
            * ``"value_mse"``: MSE of value reconstruction.
            * ``"compression_ratio"``: Bits-per-element compression ratio.
        """
        results: Dict[str, Any] = {}

        # --- SpectralQuant ---
        try:
            ck = self.compress_keys(keys, layer_idx)
            cv = self.compress_values(values, layer_idx)

            # Reconstruct keys from compressed
            key_hats: List[torch.Tensor] = []
            for head_idx, cvk in sorted(ck.items()):
                quant_k = self._get_quantizer(layer_idx, head_idx, "key")
                k_rot_hat = quant_k.decompress(cvk)
                k_hat = self._key_rotation.unrotate(k_rot_hat, layer_idx, head_idx)
                key_hats.append(k_hat)
            keys_hat = torch.stack(key_hats, dim=1)

            vals_hat = self.decompress_values(cv, layer_idx)

            results["spectralquant"] = {
                "key_cosine_sim": float(
                    cosine_similarity(
                        keys_hat.flatten(0, 2),
                        keys[:, : keys_hat.shape[1], :, :].flatten(0, 2),
                    ).mean().item()
                ),
                "value_cosine_sim": float(
                    cosine_similarity(
                        vals_hat.flatten(0, 2),
                        values[:, : vals_hat.shape[1], :, :].flatten(0, 2),
                    ).mean().item()
                ),
                "key_mse": float(
                    (keys_hat - keys[:, : keys_hat.shape[1], :, :]).pow(2).mean().item()
                ),
                "value_mse": float(
                    (vals_hat - values[:, : vals_hat.shape[1], :, :]).pow(2).mean().item()
                ),
                "compression_ratio": self.get_compression_ratio(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("SpectralQuant comparison failed: %s", exc)
            results["spectralquant"] = {"error": str(exc)}

        # --- Baseline ---
        try:
            results["turboquant"] = baseline.compare(keys, values, queries, layer_idx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TurboQuant baseline comparison failed: %s", exc)
            results["turboquant"] = {"error": str(exc)}

        return results


# ---------------------------------------------------------------------------
# TurboQuant baseline
# ---------------------------------------------------------------------------

class TurboQuantBaseline:
    """TurboQuant-style baseline: random rotation + uniform quantization + full QJL.

    Designed to be a fair comparison point for :class:`SpectralQuantEngine`:
    * Uses Haar-distributed random orthogonal rotation (no calibration needed).
    * Applies uniform (Lloyd-Max) quantization with the same avg_bits but
      without the two-regime split.
    * Uses full QJL on all d dimensions.

    Parameters
    ----------
    n_layers:
        Number of transformer layers.
    n_heads:
        Number of attention heads per layer.
    head_dim:
        Dimension of each attention head.
    config:
        Engine configuration (avg_bits, qjl_projections, etc.).

    Examples
    --------
    >>> baseline = TurboQuantBaseline(n_layers=32, n_heads=32, head_dim=128)
    >>> baseline.fit_quantizers(keys_per_layer)
    >>> result = baseline.compare(keys, values, queries, layer_idx=0)
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        config: Optional[EngineConfig] = None,
    ) -> None:
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._config = config or EngineConfig()

        self._rotation = RandomRotation(
            head_dim=head_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            global_seed=self._config.rotation_seed,
        )
        self._qjl = FullQJL(
            n_projections=self._config.qjl_projections,
            seed=self._config.rotation_seed,
        )
        # Uniform quantizers per (layer_idx, head_idx, head_type)
        self._quantizers: Dict[Tuple[int, int, str], LloydMaxQuantizer] = {}
        self._is_fitted: bool = False

    def fit_quantizers(
        self,
        kv_data: Dict[Tuple[int, int, str], torch.Tensor],
    ) -> None:
        """Fit per-head uniform Lloyd-Max quantizers.

        Parameters
        ----------
        kv_data:
            Dictionary keyed by ``(layer_idx, head_idx, head_type)`` mapping
            to key or value tensors of shape ``(n_tokens, head_dim)``.
        """
        n_bits = max(1, round(self._config.avg_bits))
        for (layer_idx, head_idx, head_type), data in kv_data.items():
            # Rotate with random matrix, then fit uniform quantizer
            data_rot = self._rotation.rotate(data.float(), layer_idx, head_idx)
            lm = LloydMaxQuantizer(
                n_bits=n_bits,
                max_iter=self._config.lloyd_max_iter,
                seed=self._config.lloyd_seed,
            ).fit(data_rot.flatten())
            self._quantizers[(layer_idx, head_idx, head_type)] = lm
        self._is_fitted = True
        logger.info("TurboQuantBaseline: fitted %d quantizers.", len(self._quantizers))

    def compress_keys(
        self,
        keys: torch.Tensor,
        layer_idx: int,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Size]]:
        """Compress keys with uniform quantization.

        Parameters
        ----------
        keys:
            Shape ``(batch, n_kv_heads, seq_len, head_dim)``.
        layer_idx:
            Layer index.

        Returns
        -------
        Dict[int, Tuple[torch.Tensor, torch.Size]]
            Per-head ``(quantized_indices, original_shape)``.
        """
        n_kv_heads = keys.shape[1]
        result: Dict[int, Tuple[torch.Tensor, torch.Size]] = {}
        for head_idx in range(n_kv_heads):
            k_h = keys[:, head_idx, :, :]
            k_rot = self._rotation.rotate(k_h.float(), layer_idx, head_idx)
            lm = self._quantizers.get((layer_idx, head_idx, "key"))
            if lm is None:
                logger.warning("No quantizer for L%d H%d key; using identity.", layer_idx, head_idx)
                result[head_idx] = (k_rot, k_rot.shape)
            else:
                idx = lm.quantize(k_rot)
                result[head_idx] = (idx, k_rot.shape)
        return result

    def decompress_keys(
        self,
        compressed: Dict[int, Tuple[torch.Tensor, torch.Size]],
        layer_idx: int,
    ) -> torch.Tensor:
        """Reconstruct keys from uniform quantization.

        Parameters
        ----------
        compressed:
            Output of :meth:`compress_keys`.
        layer_idx:
            Layer index.

        Returns
        -------
        torch.Tensor
            Reconstructed keys of shape ``(batch, n_kv_heads, seq_len, head_dim)``.
        """
        head_list: List[torch.Tensor] = []
        for head_idx, (idx, orig_shape) in sorted(compressed.items()):
            lm = self._quantizers.get((layer_idx, head_idx, "key"))
            if lm is None:
                k_rot_hat = idx.float()
            else:
                k_rot_hat = lm.dequantize(idx).reshape(orig_shape)
            k_hat = self._rotation.unrotate(k_rot_hat, layer_idx, head_idx)
            head_list.append(k_hat)
        return torch.stack(head_list, dim=1)

    def decompress_values(
        self,
        compressed: Dict[int, Tuple[torch.Tensor, torch.Size]],
        layer_idx: int,
    ) -> torch.Tensor:
        """Reconstruct values from uniform quantization.

        Parameters
        ----------
        compressed:
            Compressed value dict from :meth:`compress_values`.
        layer_idx:
            Layer index.

        Returns
        -------
        torch.Tensor
            Reconstructed values of shape ``(batch, n_kv_heads, seq_len, head_dim)``.
        """
        head_list: List[torch.Tensor] = []
        for head_idx, (idx, orig_shape) in sorted(compressed.items()):
            lm = self._quantizers.get((layer_idx, head_idx, "value"))
            if lm is None:
                v_rot_hat = idx.float()
            else:
                v_rot_hat = lm.dequantize(idx).reshape(orig_shape)
            v_hat = self._rotation.unrotate(v_rot_hat, layer_idx, head_idx)
            head_list.append(v_hat)
        return torch.stack(head_list, dim=1)

    def compress_values(
        self,
        values: torch.Tensor,
        layer_idx: int,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Size]]:
        """Compress values with uniform quantization.

        Parameters
        ----------
        values:
            Shape ``(batch, n_kv_heads, seq_len, head_dim)``.
        layer_idx:
            Layer index.

        Returns
        -------
        Dict[int, Tuple[torch.Tensor, torch.Size]]
            Per-head ``(quantized_indices, original_shape)``.
        """
        n_kv_heads = values.shape[1]
        result: Dict[int, Tuple[torch.Tensor, torch.Size]] = {}
        for head_idx in range(n_kv_heads):
            v_h = values[:, head_idx, :, :]
            v_rot = self._rotation.rotate(v_h.float(), layer_idx, head_idx)
            lm = self._quantizers.get((layer_idx, head_idx, "value"))
            if lm is None:
                result[head_idx] = (v_rot, v_rot.shape)
            else:
                idx = lm.quantize(v_rot)
                result[head_idx] = (idx, v_rot.shape)
        return result

    def attention_score(
        self,
        queries: torch.Tensor,
        compressed_keys: Dict[int, Tuple[torch.Tensor, torch.Size]],
        layer_idx: int,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute attention weights using full QJL.

        Parameters
        ----------
        queries:
            Query tensor of shape ``(batch, n_heads, n_queries, head_dim)``.
        compressed_keys:
            Output of :meth:`compress_keys`.
        layer_idx:
            Layer index.
        scale:
            Attention scale factor.

        Returns
        -------
        torch.Tensor
            Softmax attention weights of shape
            ``(batch, n_heads, n_queries, seq_len)``.
        """
        batch, n_heads, n_queries, head_dim = queries.shape
        if scale is None:
            scale = head_dim ** -0.5

        n_kv_heads = len(compressed_keys)
        head_ratio = n_heads // n_kv_heads

        all_weights: List[torch.Tensor] = []

        for head_idx, (idx, orig_shape) in sorted(compressed_keys.items()):
            lm = self._quantizers.get((layer_idx, head_idx, "key"))
            if lm is None:
                k_rot_hat = idx.float()
            else:
                k_rot_hat = lm.dequantize(idx).reshape(orig_shape)

            q_start = head_idx * head_ratio
            q_end = q_start + head_ratio
            q_group = queries[:, q_start:q_end, :, :]

            scores_list: List[torch.Tensor] = []
            for qi in range(head_ratio):
                q_h = q_group[:, qi, :, :]  # (batch, n_q, d)
                q_rot = self._rotation.rotate(q_h.float(), layer_idx, head_idx)
                score = self._qjl.compute_correction(
                    keys=k_rot_hat,
                    queries=q_rot,
                    d_eff=head_dim,  # FullQJL ignores d_eff but we pass for interface parity
                )
                scores_list.append(score)

            group_scores = torch.stack(scores_list, dim=1)  # (batch, hr, n_q, seq)
            weights = _softmax_attn_weights(group_scores * scale)
            all_weights.append(weights)

        return torch.cat(all_weights, dim=1)

    def get_compression_ratio(self) -> float:
        """Return theoretical compression ratio (fp16 → avg_bits).

        Returns
        -------
        float
            Original bits (16) divided by avg_bits.
        """
        return 16.0 / self._config.avg_bits

    def compare(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        layer_idx: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate baseline reconstruction quality on a batch.

        Parameters
        ----------
        keys:
            Shape ``(batch, n_kv_heads, seq_len, head_dim)``.
        values:
            Same shape as ``keys``.
        queries:
            Shape ``(batch, n_heads, n_queries, head_dim)``.
        layer_idx:
            Layer index.

        Returns
        -------
        Dict[str, Any]
            Dictionary with ``"key_cosine_sim"``, ``"value_cosine_sim"``,
            ``"key_mse"``, ``"value_mse"``, ``"compression_ratio"``.
        """
        ck = self.compress_keys(keys, layer_idx)
        cv = self.compress_values(values, layer_idx)

        keys_hat = self.decompress_keys(ck, layer_idx)
        vals_hat = self.decompress_values(cv, layer_idx)

        nk = keys_hat.shape[1]
        nv = vals_hat.shape[1]

        return {
            "key_cosine_sim": float(
                cosine_similarity(
                    keys_hat.flatten(0, 2), keys[:, :nk, :, :].flatten(0, 2)
                ).mean().item()
            ),
            "value_cosine_sim": float(
                cosine_similarity(
                    vals_hat.flatten(0, 2), values[:, :nv, :, :].flatten(0, 2)
                ).mean().item()
            ),
            "key_mse": float((keys_hat - keys[:, :nk, :, :]).pow(2).mean().item()),
            "value_mse": float((vals_hat - values[:, :nv, :, :]).pow(2).mean().item()),
            "compression_ratio": self.get_compression_ratio(),
        }
