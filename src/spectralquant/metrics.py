"""
Evaluation metrics for SpectralQuant.

All metrics are batched-friendly torch functions.  Each accepts and returns
tensors unless otherwise noted.

Metrics:
- ``cosine_similarity``: Batched cosine similarity.
- ``attention_output_cosine_sim``: Cosine similarity between attention outputs.
- ``max_absolute_weight_error``: Max absolute error between attention weight matrices.
- ``weighted_mse``: MSE weighted by eigenvalue importance.
- ``compression_ratio``: Ratio of original to compressed bit count.
- ``inner_product_error``: Error between exact and approximate query-key inner products.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute batched cosine similarity between tensors ``x`` and ``y``.

    The cosine similarity is computed along the **last** dimension.

    Parameters
    ----------
    x:
        First tensor.  Shape ``(..., d)``.
    y:
        Second tensor.  Same shape as ``x``.
    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Cosine similarity values.  Shape ``(...)`` (last dimension reduced).

    Examples
    --------
    >>> sim = cosine_similarity(k_approx, k_exact)  # (batch, seq_len)
    >>> print(sim.mean())  # mean over all tokens
    """
    x_norm = torch.nn.functional.normalize(x.float(), dim=-1, eps=eps)
    y_norm = torch.nn.functional.normalize(y.float(), dim=-1, eps=eps)
    return (x_norm * y_norm).sum(dim=-1)


def attention_output_cosine_sim(
    attn_out_approx: torch.Tensor,
    attn_out_fp16: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine similarity between approximate and reference attention outputs.

    Parameters
    ----------
    attn_out_approx:
        Approximate attention output from the compressed pipeline.
        Shape ``(batch, seq_len, hidden_dim)`` or ``(seq_len, hidden_dim)``.
    attn_out_fp16:
        Reference attention output (full-precision baseline).
        Same shape as ``attn_out_approx``.
    eps:
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Per-token cosine similarities.  Shape ``(batch, seq_len)`` or
        ``(seq_len,)``.

    Examples
    --------
    >>> sim = attention_output_cosine_sim(approx_out, fp16_out)
    >>> print(f"Mean cosine sim: {sim.mean():.4f}")
    """
    return cosine_similarity(attn_out_approx, attn_out_fp16, eps=eps)


def max_absolute_weight_error(
    weights_approx: torch.Tensor,
    weights_fp16: torch.Tensor,
) -> torch.Tensor:
    """Maximum absolute error between approximate and reference attention weights.

    Parameters
    ----------
    weights_approx:
        Approximate softmax attention weights.  Shape
        ``(batch, n_heads, n_queries, seq_len)`` or any broadcastable shape.
    weights_fp16:
        Reference attention weights.  Same shape as ``weights_approx``.

    Returns
    -------
    torch.Tensor
        Max absolute error per query position.  Shape
        ``(batch, n_heads, n_queries)`` or reduced accordingly.

    Examples
    --------
    >>> err = max_absolute_weight_error(approx_weights, ref_weights)
    >>> print(f"Max error: {err.max():.6f}")
    """
    return (weights_approx.float() - weights_fp16.float()).abs().max(dim=-1).values


def weighted_mse(
    x: torch.Tensor,
    y: torch.Tensor,
    eigenvalues: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """MSE weighted by eigenvalue importance.

    Errors in high-eigenvalue (high-energy) dimensions are penalised more
    heavily, reflecting the fact that these dimensions contribute more to the
    attention score.

    .. math::

        \\text{wMSE} = \\frac{\\sum_i \\lambda_i (x_i - y_i)^2}{\\sum_i \\lambda_i}

    Parameters
    ----------
    x:
        Approximate tensor.  Shape ``(..., head_dim)``.
    y:
        Reference tensor.  Same shape as ``x``.
    eigenvalues:
        Non-negative eigenvalues of shape ``(head_dim,)``, used as weights.
    eps:
        Small constant to avoid division by zero in normalisation.

    Returns
    -------
    torch.Tensor
        Scalar weighted MSE.

    Examples
    --------
    >>> wmse = weighted_mse(k_approx, k_exact, eigenvalues)
    """
    weights = eigenvalues.float().to(x.device)
    weights = weights / (weights.sum() + eps)
    sq_err = (x.float() - y.float()).pow(2)  # (..., head_dim)
    return (sq_err * weights).sum(dim=-1).mean()


def compression_ratio(
    original_bits: float,
    compressed_bits: float,
) -> float:
    """Compression ratio: original storage / compressed storage.

    Parameters
    ----------
    original_bits:
        Total bits for the uncompressed representation (e.g. n_elements * 16
        for fp16).
    compressed_bits:
        Total bits for the compressed representation.

    Returns
    -------
    float
        Compression ratio ≥ 1.0 if compression is effective.  Returns
        ``0.0`` if ``compressed_bits`` is zero.

    Examples
    --------
    >>> ratio = compression_ratio(original_bits=32*1024, compressed_bits=4*1024)
    >>> print(f"Compression ratio: {ratio:.1f}x")
    """
    if compressed_bits <= 0:
        logger.warning("compressed_bits is zero; returning 0.0.")
        return 0.0
    return original_bits / compressed_bits


def inner_product_error(
    q: torch.Tensor,
    k_approx: torch.Tensor,
    k_exact: torch.Tensor,
    normalise: bool = True,
) -> torch.Tensor:
    """Error between exact and approximate query-key inner products.

    Measures how accurately the compressed pipeline preserves the attention
    scores (pre-softmax logits).

    .. math::

        \\text{IPE} = \\langle q, k_{\\text{approx}} \\rangle
                    - \\langle q, k_{\\text{exact}} \\rangle

    Parameters
    ----------
    q:
        Query tensor.  Shape ``(..., head_dim)``.
    k_approx:
        Approximate (reconstructed) key tensor.  Same leading shape as ``q``.
    k_exact:
        Reference key tensor.  Same shape as ``k_approx``.
    normalise:
        If ``True``, normalise the error by ``‖q‖ ‖k_exact‖`` to make it
        dimensionless (relative error).

    Returns
    -------
    torch.Tensor
        Scalar (mean) inner-product error.

    Examples
    --------
    >>> err = inner_product_error(queries, k_hat, keys)
    >>> print(f"Mean IP error: {err:.6f}")
    """
    q_f = q.float()
    k_ap = k_approx.float()
    k_ex = k_exact.float()

    ip_approx = (q_f * k_ap).sum(dim=-1)   # (...,)
    ip_exact = (q_f * k_ex).sum(dim=-1)    # (...,)
    err = ip_approx - ip_exact              # (...,)

    if normalise:
        denom = (q_f.norm(dim=-1) * k_ex.norm(dim=-1)).clamp(min=1e-8)
        err = err / denom

    return err.abs().mean()
