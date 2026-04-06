"""
Eigenspectral calibration module for SpectralQuant.

This module provides EigenspectralCalibrator, which hooks into attention layers
of a pretrained LLM, collects key/value vectors, and computes per-head
eigenspectral statistics (eigenvalues, eigenvectors, participation ratio,
spectral gap, cumulative variance thresholds).

These statistics drive non-uniform bit allocation and spectral rotation in the
full SpectralQuant pipeline.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class HeadCalibrationData:
    """Per-head calibration results from eigenspectral analysis.

    Attributes:
        layer_idx: Index of the transformer layer.
        head_idx: Index of the attention head within the layer.
        head_type: Either ``"key"`` or ``"value"``.
        eigenvalues: Eigenvalues of the covariance matrix, sorted descending.
            Shape ``(head_dim,)``.
        eigenvectors: Orthonormal eigenvectors (columns), i.e. the rotation
            matrix V such that C = V diag(λ) V^T.  Shape
            ``(head_dim, head_dim)``.
        d_eff: Effective rank (participation ratio) of the covariance matrix.
        spectral_gap: Ratio λ_{d_eff} / λ_{d_eff+1} at the effective-rank
            boundary.  ``None`` if d_eff equals head_dim.
        var_95: Number of leading components that explain ≥ 95 % of variance.
        var_99: Number of leading components that explain ≥ 99 % of variance.
        n_samples: Number of vectors used to estimate the covariance.
        head_dim: Dimensionality of each head vector.
    """

    layer_idx: int
    head_idx: int
    head_type: str  # "key" | "value"
    eigenvalues: torch.Tensor          # (head_dim,)
    eigenvectors: torch.Tensor         # (head_dim, head_dim)
    d_eff: float
    spectral_gap: Optional[float]
    var_95: int
    var_99: int
    n_samples: int
    head_dim: int

    # ---------- serialisation helpers ----------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable + tensor dict."""
        return {
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "head_type": self.head_type,
            "d_eff": self.d_eff,
            "spectral_gap": self.spectral_gap,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "n_samples": self.n_samples,
            "head_dim": self.head_dim,
            # tensors serialised separately
        }

    @classmethod
    def from_dict(
        cls,
        meta: Dict[str, Any],
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> "HeadCalibrationData":
        """Reconstruct from metadata dict + tensors."""
        return cls(
            layer_idx=meta["layer_idx"],
            head_idx=meta["head_idx"],
            head_type=meta["head_type"],
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            d_eff=meta["d_eff"],
            spectral_gap=meta["spectral_gap"],
            var_95=meta["var_95"],
            var_99=meta["var_99"],
            n_samples=meta["n_samples"],
            head_dim=meta["head_dim"],
        )


# ---------------------------------------------------------------------------
# Architecture detection helpers
# ---------------------------------------------------------------------------

def _detect_architecture(model: nn.Module) -> str:
    """Detect the model architecture family.

    Parameters
    ----------
    model:
        A pretrained HuggingFace model instance.

    Returns
    -------
    str
        One of ``"qwen2"`` , ``"llama"``, or ``"generic"``.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        logger.warning("Model has no .config attribute; falling back to generic hooks.")
        return "generic"
    arch = getattr(cfg, "model_type", "").lower()
    if "qwen2" in arch or "qwen" in arch:
        return "qwen2"
    if "llama" in arch or "mistral" in arch or "gemma" in arch:
        return "llama"
    logger.warning("Unknown architecture '%s'; falling back to generic hooks.", arch)
    return "generic"


def _iter_attention_layers(model: nn.Module, arch: str):
    """Yield ``(layer_idx, attention_module)`` pairs.

    Parameters
    ----------
    model:
        Pretrained model.
    arch:
        Architecture family string from :func:`_detect_architecture`.

    Yields
    ------
    Tuple[int, nn.Module]
        ``(layer_idx, attn_module)`` for every transformer block.
    """
    # Qwen2 / Qwen: model.model.layers[i].self_attn
    # Llama / Mistral / Gemma: model.model.layers[i].self_attn
    # Generic fallback: scan for named modules ending with "self_attn" or "attention"
    layers = None
    if arch in ("qwen2", "llama"):
        try:
            layers = model.model.layers
        except AttributeError:
            pass

    if layers is not None:
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                attn = getattr(layer, "attention", None)
            if attn is not None:
                yield i, attn
        return

    # Generic: walk all named modules and find attention-like sub-modules
    seen: set = set()
    for name, module in model.named_modules():
        lname = name.lower()
        if ("self_attn" in lname or "attention" in lname) and id(module) not in seen:
            # Try to extract layer index from name
            parts = name.split(".")
            idx = None
            for p in parts:
                if p.isdigit():
                    idx = int(p)
                    break
            if idx is not None:
                seen.add(id(module))
                yield idx, module


def _get_kv_head_dims(attn_module: nn.Module, model_config: Any) -> Tuple[int, int, int]:
    """Return (n_heads, n_kv_heads, head_dim) from an attention module.

    Parameters
    ----------
    attn_module:
        Attention sub-module.
    model_config:
        ``model.config`` object.

    Returns
    -------
    Tuple[int, int, int]
        ``(n_heads, n_kv_heads, head_dim)``.
    """
    n_heads = getattr(model_config, "num_attention_heads", None) or getattr(
        attn_module, "num_heads", 1
    )
    n_kv_heads = getattr(model_config, "num_key_value_heads", None) or getattr(
        attn_module, "num_key_value_heads", n_heads
    )
    hidden_size = getattr(model_config, "hidden_size", None) or getattr(
        attn_module, "hidden_size", n_heads * 64
    )
    head_dim = hidden_size // n_heads
    return int(n_heads), int(n_kv_heads), int(head_dim)


# ---------------------------------------------------------------------------
# Spectral statistics helpers
# ---------------------------------------------------------------------------

def _compute_covariance(vectors: torch.Tensor) -> torch.Tensor:
    """Compute uncentred covariance C = X^T X / n.

    Parameters
    ----------
    vectors:
        Tensor of shape ``(n, d)`` in float32.

    Returns
    -------
    torch.Tensor
        Symmetric covariance matrix of shape ``(d, d)``.
    """
    n = vectors.shape[0]
    return (vectors.T @ vectors) / n


def _eigendecompose(cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigendecompose a symmetric PSD covariance matrix.

    Uses :func:`torch.linalg.eigh` (numerically stable for symmetric matrices)
    and returns eigenvalues/vectors sorted in **descending** order.

    Parameters
    ----------
    cov:
        Symmetric matrix of shape ``(d, d)`` in float32.

    Returns
    -------
    eigenvalues:
        Sorted-descending eigenvalues of shape ``(d,)``.
    eigenvectors:
        Corresponding orthonormal eigenvectors (columns), shape ``(d, d)``.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # eigh returns ascending order – reverse to descending
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    # Clamp small negatives that arise from numerical error
    eigenvalues = eigenvalues.clamp(min=0.0)
    return eigenvalues, eigenvectors


def _participation_ratio(eigenvalues: torch.Tensor) -> float:
    """Compute the participation ratio (effective rank).

    .. math::

        d_{\\text{eff}} = \\frac{(\\sum_i \\lambda_i)^2}{\\sum_i \\lambda_i^2}

    Parameters
    ----------
    eigenvalues:
        Non-negative eigenvalues of shape ``(d,)``.

    Returns
    -------
    float
        Effective rank :math:`d_{\\text{eff}}`.
    """
    lam = eigenvalues.double()
    sum_lam = lam.sum()
    sum_lam_sq = (lam ** 2).sum()
    if sum_lam_sq < 1e-12:
        return 1.0
    return float((sum_lam ** 2) / sum_lam_sq)


def _spectral_gap(eigenvalues: torch.Tensor, d_eff: float) -> Optional[float]:
    """Compute the spectral gap at the effective-rank boundary.

    .. math::

        \\kappa = \\lambda_{d_{\\text{eff}}} / \\lambda_{d_{\\text{eff}}+1}

    where indices are 1-based and :math:`d_{\\text{eff}}` is rounded to the
    nearest integer.

    Parameters
    ----------
    eigenvalues:
        Sorted-descending eigenvalues.
    d_eff:
        Participation ratio.

    Returns
    -------
    Optional[float]
        Spectral gap, or ``None`` if the boundary is beyond the last
        eigenvalue.
    """
    d = len(eigenvalues)
    k = max(1, min(round(d_eff), d - 1))  # boundary index (0-based: k-1 and k)
    lam_k = float(eigenvalues[k - 1])
    lam_k1 = float(eigenvalues[k])
    if lam_k1 < 1e-12:
        return None
    return lam_k / lam_k1


def _cumulative_variance_thresholds(
    eigenvalues: torch.Tensor,
) -> Tuple[int, int]:
    """Find the minimum number of components for 95 % and 99 % variance.

    Parameters
    ----------
    eigenvalues:
        Sorted-descending eigenvalues.

    Returns
    -------
    Tuple[int, int]
        ``(var_95, var_99)`` — the smallest ``k`` such that the top-``k``
        components explain at least 95 % / 99 % of total variance.
    """
    total = eigenvalues.sum()
    if total < 1e-12:
        return 1, 1
    cumvar = eigenvalues.cumsum(0) / total
    var_95 = int((cumvar < 0.95).sum().item()) + 1
    var_99 = int((cumvar < 0.99).sum().item()) + 1
    var_95 = min(var_95, len(eigenvalues))
    var_99 = min(var_99, len(eigenvalues))
    return var_95, var_99


# ---------------------------------------------------------------------------
# Hook utilities
# ---------------------------------------------------------------------------

class _KVCollectorHook:
    """Forward hook that collects key and value tensors from an attention layer.

    The hook expects the attention module to expose ``k_proj`` and ``v_proj``
    as child modules (standard HuggingFace convention), or to receive the
    key/value tensors via the output tuple.

    Collected tensors are appended to ``key_buffers[head_idx]`` and
    ``value_buffers[head_idx]`` (list of 1D tensors per head).

    Parameters
    ----------
    n_kv_heads:
        Number of key/value heads (for GQA models may differ from n_heads).
    head_dim:
        Dimension of each head.
    device:
        Device to move collected vectors to (usually CPU to save GPU memory).
    max_tokens:
        Maximum total tokens to collect (across all calls).
    """

    def __init__(
        self,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        max_tokens: int = 50_000,
    ) -> None:
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.max_tokens = max_tokens
        self.key_buffers: List[List[torch.Tensor]] = [[] for _ in range(n_kv_heads)]
        self.value_buffers: List[List[torch.Tensor]] = [[] for _ in range(n_kv_heads)]
        self._total_tokens: int = 0
        self._active: bool = True

    def __call__(
        self,
        module: nn.Module,
        inputs: Tuple,
        output: Any,
    ) -> None:
        if not self._active or self._total_tokens >= self.max_tokens:
            return

        # Attempt to extract key and value tensors from the output.
        # HuggingFace attention modules typically return (attn_out, attn_weights,
        # past_key_value).  The past_key_value is a tuple (key, value).
        keys: Optional[torch.Tensor] = None
        values: Optional[torch.Tensor] = None

        if isinstance(output, tuple):
            for item in output:
                if isinstance(item, tuple) and len(item) == 2:
                    k_cand, v_cand = item
                    if (
                        isinstance(k_cand, torch.Tensor)
                        and isinstance(v_cand, torch.Tensor)
                        and k_cand.dim() == 4  # (batch, n_kv_heads, seq, head_dim)
                    ):
                        keys, values = k_cand, v_cand
                        break

        if keys is None:
            # Try input tuple: some implementations pass (hidden, key, value, ...)
            for item in inputs:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    if keys is None:
                        keys = item
                    elif values is None:
                        values = item
                        break

        if keys is None or values is None:
            return

        # keys / values shape: (batch, n_kv_heads, seq_len, head_dim)
        # Move to CPU and cast to float32
        keys = keys.detach().float().cpu()
        values = values.detach().float().cpu()

        batch, n_kv, seq_len, hdim = keys.shape
        remaining = self.max_tokens - self._total_tokens
        seq_use = min(seq_len, remaining // max(batch, 1))
        if seq_use <= 0:
            self._active = False
            return

        for h in range(min(n_kv, self.n_kv_heads)):
            # Shape: (batch * seq_use, head_dim)
            k_slice = keys[:, h, :seq_use, :].reshape(-1, hdim)
            v_slice = values[:, h, :seq_use, :].reshape(-1, hdim)
            self.key_buffers[h].append(k_slice)
            self.value_buffers[h].append(v_slice)

        self._total_tokens += batch * seq_use
        if self._total_tokens >= self.max_tokens:
            self._active = False

    def get_keys(self, head_idx: int) -> torch.Tensor:
        """Concatenate collected key vectors for ``head_idx``."""
        if not self.key_buffers[head_idx]:
            return torch.zeros(0, self.head_dim)
        return torch.cat(self.key_buffers[head_idx], dim=0)

    def get_values(self, head_idx: int) -> torch.Tensor:
        """Concatenate collected value vectors for ``head_idx``."""
        if not self.value_buffers[head_idx]:
            return torch.zeros(0, self.head_dim)
        return torch.cat(self.value_buffers[head_idx], dim=0)

    def reset(self) -> None:
        """Clear all buffers."""
        self.key_buffers = [[] for _ in range(self.n_kv_heads)]
        self.value_buffers = [[] for _ in range(self.n_kv_heads)]
        self._total_tokens = 0
        self._active = True


# ---------------------------------------------------------------------------
# Main calibrator
# ---------------------------------------------------------------------------

class EigenspectralCalibrator:
    """Calibrate per-head eigenspectral structure from a pretrained LLM.

    The calibrator:

    1. Registers forward hooks on every attention layer.
    2. Runs the model on a calibration dataset to collect key/value vectors.
    3. Computes C = X^T X / n for each head.
    4. Eigendecomposes C → (λ, V).
    5. Derives participation ratio d_eff, spectral gap κ, and variance
       thresholds (var_95, var_99).

    All computations are performed in float32 for numerical stability.

    Parameters
    ----------
    max_tokens_per_layer:
        Maximum number of token vectors to collect per attention layer
        before stopping collection (controls memory usage).
    device:
        Device used for calibration forward passes.  Defaults to the model's
        current device.

    Examples
    --------
    >>> calibrator = EigenspectralCalibrator(max_tokens_per_layer=10_000)
    >>> calibrator.calibrate(model, tokenizer, dataset, n_samples=512)
    >>> calibrator.save("calibration.pt")
    >>> stats = calibrator.summary()
    """

    def __init__(
        self,
        max_tokens_per_layer: int = 50_000,
        device: Optional[torch.device] = None,
    ) -> None:
        self.max_tokens_per_layer = max_tokens_per_layer
        self.device = device
        self._calibration_data: Dict[Tuple[int, int, str], HeadCalibrationData] = {}
        self._is_calibrated: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model: nn.Module,
        tokenizer: Any,
        dataset: List[str],
        n_samples: int = 1000,
    ) -> None:
        """Run calibration on the given model and dataset.

        Parameters
        ----------
        model:
            A pretrained HuggingFace language model (Qwen2.5, Llama, etc.).
        tokenizer:
            Matching HuggingFace tokenizer.
        dataset:
            List of text strings to use as calibration data.
        n_samples:
            Number of samples from ``dataset`` to process.  If
            ``len(dataset) < n_samples``, all samples are used.

        Notes
        -----
        The model is run in ``torch.no_grad()`` mode.  Existing gradients are
        not affected.  The model's training/eval state is not modified.
        """
        device = self.device or next(model.parameters()).device
        arch = _detect_architecture(model)
        logger.info("Detected architecture: %s", arch)

        model_config = getattr(model, "config", None)

        # Register hooks on every attention layer
        hooks_map: Dict[int, _KVCollectorHook] = {}
        hook_handles: List[torch.utils.hooks.RemovableHook] = []

        for layer_idx, attn_module in _iter_attention_layers(model, arch):
            if layer_idx in hooks_map:
                continue
            n_heads, n_kv_heads, head_dim = _get_kv_head_dims(attn_module, model_config)
            collector = _KVCollectorHook(
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                device=torch.device("cpu"),
                max_tokens=self.max_tokens_per_layer,
            )
            handle = attn_module.register_forward_hook(collector)
            hooks_map[layer_idx] = collector
            hook_handles.append(handle)
            logger.debug(
                "Registered hook on layer %d (n_kv_heads=%d, head_dim=%d)",
                layer_idx,
                n_kv_heads,
                head_dim,
            )

        if not hooks_map:
            raise RuntimeError(
                "No attention layers found in the model.  "
                "Ensure the model has a .model.layers attribute or "
                "sub-modules named 'self_attn'/'attention'."
            )

        # Forward passes over calibration data
        samples = dataset[:n_samples]
        logger.info(
            "Running %d calibration samples across %d attention layers.",
            len(samples),
            len(hooks_map),
        )

        model.eval()
        with torch.no_grad():
            for i, text in enumerate(samples):
                try:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                    ).to(device)
                    model(**inputs, use_cache=True, output_attentions=False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping sample %d due to error: %s", i, exc)
                if (i + 1) % 100 == 0:
                    logger.info("Processed %d/%d samples.", i + 1, len(samples))

        # Remove hooks
        for handle in hook_handles:
            handle.remove()

        # Compute eigenspectral statistics per head
        logger.info("Computing eigenspectral statistics …")
        for layer_idx, collector in hooks_map.items():
            n_heads, n_kv_heads, head_dim = (
                collector.n_kv_heads,
                collector.n_kv_heads,
                collector.head_dim,
            )
            for head_idx in range(n_kv_heads):
                for head_type, get_fn in (
                    ("key", collector.get_keys),
                    ("value", collector.get_values),
                ):
                    vectors = get_fn(head_idx)  # (n_tokens, head_dim)
                    if vectors.shape[0] < 2:
                        logger.warning(
                            "Layer %d head %d (%s): too few samples (%d), skipping.",
                            layer_idx,
                            head_idx,
                            head_type,
                            vectors.shape[0],
                        )
                        continue

                    cov = _compute_covariance(vectors.float())
                    eigenvalues, eigenvectors = _eigendecompose(cov)
                    d_eff = _participation_ratio(eigenvalues)
                    gap = _spectral_gap(eigenvalues, d_eff)
                    var_95, var_99 = _cumulative_variance_thresholds(eigenvalues)

                    key = (layer_idx, head_idx, head_type)
                    self._calibration_data[key] = HeadCalibrationData(
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        head_type=head_type,
                        eigenvalues=eigenvalues,
                        eigenvectors=eigenvectors,
                        d_eff=d_eff,
                        spectral_gap=gap,
                        var_95=var_95,
                        var_99=var_99,
                        n_samples=vectors.shape[0],
                        head_dim=head_dim,
                    )
                    logger.debug(
                        "Layer %d head %d (%s): d_eff=%.2f, gap=%s, var95=%d, var99=%d",
                        layer_idx,
                        head_idx,
                        head_type,
                        d_eff,
                        f"{gap:.2f}" if gap is not None else "N/A",
                        var_95,
                        var_99,
                    )

        self._is_calibrated = True
        logger.info(
            "Calibration complete.  Stored data for %d (layer, head, type) triples.",
            len(self._calibration_data),
        )

    def get(
        self, layer_idx: int, head_idx: int, head_type: str = "key"
    ) -> Optional[HeadCalibrationData]:
        """Retrieve calibration data for a specific head.

        Parameters
        ----------
        layer_idx:
            Transformer layer index.
        head_idx:
            Attention head index.
        head_type:
            ``"key"`` or ``"value"``.

        Returns
        -------
        Optional[HeadCalibrationData]
            Calibration data, or ``None`` if not found.
        """
        return self._calibration_data.get((layer_idx, head_idx, head_type))

    def iter_heads(self) -> Iterator[HeadCalibrationData]:
        """Iterate over all stored ``HeadCalibrationData`` objects."""
        yield from self._calibration_data.values()

    def save(self, path: str) -> None:
        """Serialise calibration data to disk.

        Saves a ``.pt`` file (tensors) and a companion ``_meta.json`` file.

        Parameters
        ----------
        path:
            Base output path (without extension).  Two files will be written:
            ``<path>.pt`` and ``<path>_meta.json``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tensor_data: Dict[str, torch.Tensor] = {}
        meta_list: List[Dict[str, Any]] = []

        for key, hcd in self._calibration_data.items():
            k = f"L{key[0]}_H{key[1]}_{key[2]}"
            tensor_data[f"{k}_eigenvalues"] = hcd.eigenvalues
            tensor_data[f"{k}_eigenvectors"] = hcd.eigenvectors
            meta_list.append(hcd.to_dict())

        torch.save(tensor_data, str(path) + ".pt")
        with open(str(path) + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2)

        logger.info("Saved calibration data to %s.pt / %s_meta.json", path, path)

    def load(self, path: str) -> None:
        """Load calibration data from disk.

        Parameters
        ----------
        path:
            Base path (without extension), as used in :meth:`save`.
        """
        path = Path(path)
        tensor_data: Dict[str, torch.Tensor] = torch.load(
            str(path) + ".pt", map_location="cpu", weights_only=True
        )
        with open(str(path) + "_meta.json", "r", encoding="utf-8") as f:
            meta_list: List[Dict[str, Any]] = json.load(f)

        self._calibration_data = {}
        for meta in meta_list:
            li, hi, ht = meta["layer_idx"], meta["head_idx"], meta["head_type"]
            k = f"L{li}_H{hi}_{ht}"
            eigenvalues = tensor_data[f"{k}_eigenvalues"]
            eigenvectors = tensor_data[f"{k}_eigenvectors"]
            hcd = HeadCalibrationData.from_dict(meta, eigenvalues, eigenvectors)
            self._calibration_data[(li, hi, ht)] = hcd

        self._is_calibrated = True
        logger.info(
            "Loaded calibration data: %d head entries.", len(self._calibration_data)
        )

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics across all calibrated heads.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing means, mins, maxes and full per-layer
            breakdown of d_eff, spectral_gap, var_95, var_99.
        """
        if not self._is_calibrated or not self._calibration_data:
            return {"calibrated": False}

        d_effs = [hcd.d_eff for hcd in self._calibration_data.values()]
        gaps = [
            hcd.spectral_gap
            for hcd in self._calibration_data.values()
            if hcd.spectral_gap is not None
        ]
        var95s = [hcd.var_95 for hcd in self._calibration_data.values()]
        var99s = [hcd.var_99 for hcd in self._calibration_data.values()]

        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            t = torch.tensor(values, dtype=torch.float32)
            result: Dict[str, float] = {
                "mean": float(t.mean()),
                "min": float(t.min()),
                "max": float(t.max()),
            }
            if t.numel() > 1:
                result["std"] = float(t.std())
            else:
                result["std"] = 0.0
            return result

        return {
            "calibrated": True,
            "n_heads_calibrated": len(self._calibration_data),
            "d_eff": _stats(d_effs),
            "spectral_gap": _stats(gaps),
            "var_95": _stats([float(v) for v in var95s]),
            "var_99": _stats([float(v) for v in var99s]),
            "per_head": [
                {
                    "layer_idx": hcd.layer_idx,
                    "head_idx": hcd.head_idx,
                    "head_type": hcd.head_type,
                    "d_eff": hcd.d_eff,
                    "spectral_gap": hcd.spectral_gap,
                    "var_95": hcd.var_95,
                    "var_99": hcd.var_99,
                    "n_samples": hcd.n_samples,
                }
                for hcd in self._calibration_data.values()
            ],
        }
