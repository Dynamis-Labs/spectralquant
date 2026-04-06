"""
Non-uniform bit allocation and Lloyd-Max quantization for SpectralQuant.

After spectral rotation, key/value vectors decompose into two regimes:
* **Semantic regime** (first d_eff coordinates): high variance, deserves more
  bits (b_high).
* **Tail regime** (remaining d - d_eff coordinates): low variance, quantized
  with fewer bits (b_low).

This module provides:
- ``BitAllocator``: Solves the bit-budget constraint for the two-regime split.
- ``LloydMaxQuantizer``: Classic iterative optimal scalar quantizer.
- ``NonUniformQuantizer``: End-to-end compress/decompress using the above.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CompressedVector:
    """Container for a non-uniformly compressed vector.

    Attributes
    ----------
    semantic_indices:
        Quantized indices for the semantic regime (top d_eff coordinates).
        Shape ``(batch, seq_len, d_eff)`` or ``(..., d_eff)``.
    tail_indices:
        Quantized indices for the tail regime (remaining d - d_eff coords).
        Shape ``(batch, seq_len, d - d_eff)`` or ``(..., d - d_eff)``.
    d_eff:
        Number of semantic-regime coordinates.
    head_dim:
        Total head dimension.
    b_high:
        Bits per semantic coordinate.
    b_low:
        Bits per tail coordinate.
    original_shape:
        Shape of the original (uncompressed) tensor.
    actual_bits_used:
        Total bits used for this compressed vector.
    mse:
        Quantization MSE (set after compression if computable).
    """

    semantic_indices: torch.Tensor
    tail_indices: torch.Tensor
    d_eff: int
    head_dim: int
    b_high: int
    b_low: int
    original_shape: Tuple[int, ...]
    actual_bits_used: float = 0.0
    mse: Optional[float] = None


# ---------------------------------------------------------------------------
# Bit allocator
# ---------------------------------------------------------------------------

class BitAllocator:
    """Solve the two-regime bit allocation problem.

    Given a total bit budget ``B`` per vector and a head dimension ``d``,
    splits into:

    .. math::

        d_{\\text{eff}} \\cdot b_{\\text{high}} + (d - d_{\\text{eff}}) \\cdot b_{\\text{low}} = B

    subject to :math:`b_{\\text{high}} \\geq b_{\\text{low}} \\geq 1` and
    both being integers.

    The default strategy allocates the extra bits to the semantic regime:
    ``b_high = B_per_d + extra_for_semantic``.

    Parameters
    ----------
    min_bits:
        Minimum bits for the tail regime (default 1).
    max_bits:
        Maximum bits for either regime (default 8).

    Examples
    --------
    >>> allocator = BitAllocator()
    >>> b_high, b_low = allocator.allocate(d_eff=32, avg_bits=4.0, head_dim=128)
    >>> # semantic gets b_high bits, tail gets b_low bits
    """

    def __init__(self, min_bits: int = 1, max_bits: int = 8) -> None:
        self.min_bits = min_bits
        self.max_bits = max_bits

    def allocate(
        self,
        d_eff: float,
        avg_bits: float,
        head_dim: int,
    ) -> Tuple[int, int]:
        """Compute per-regime bit widths.

        Parameters
        ----------
        d_eff:
            Effective rank (participation ratio), may be fractional.  Rounded
            to the nearest integer for allocation.
        avg_bits:
            Target average bits per dimension across the full vector.
        head_dim:
            Total head dimension ``d``.

        Returns
        -------
        Tuple[int, int]
            ``(b_high, b_low)`` — bits for semantic and tail regimes.

        Raises
        ------
        ValueError
            If no valid integer allocation satisfying the constraint exists.
        """
        d_eff_int = max(1, min(round(d_eff), head_dim - 1))
        d_tail = head_dim - d_eff_int
        total_budget = avg_bits * head_dim  # total bits across all dimensions

        # Try all valid (b_high, b_low) pairs in descending quality order
        best: Optional[Tuple[int, int]] = None
        best_slack = math.inf

        for b_high in range(self.max_bits, self.min_bits - 1, -1):
            for b_low in range(b_high, self.min_bits - 1, -1):
                used = d_eff_int * b_high + d_tail * b_low
                slack = abs(used - total_budget)
                if slack < best_slack:
                    best_slack = slack
                    best = (b_high, b_low)

        if best is None:
            raise ValueError(
                f"Cannot find valid bit allocation for d_eff={d_eff_int}, "
                f"avg_bits={avg_bits}, head_dim={head_dim}."
            )

        b_high, b_low = best
        actual_avg = (d_eff_int * b_high + d_tail * b_low) / head_dim
        logger.debug(
            "BitAllocator: d_eff=%d, d_tail=%d, b_high=%d, b_low=%d, "
            "target_avg=%.2f, actual_avg=%.2f",
            d_eff_int,
            d_tail,
            b_high,
            b_low,
            avg_bits,
            actual_avg,
        )
        return b_high, b_low


# ---------------------------------------------------------------------------
# Lloyd-Max quantizer
# ---------------------------------------------------------------------------

class LloydMaxQuantizer:
    """Optimal scalar quantizer via the Lloyd-Max iterative algorithm.

    Minimises the mean-squared quantization error for a given scalar
    distribution by iteratively updating:

    * **Decision boundaries**: midpoints between adjacent centroids.
    * **Centroids**: conditional means within each Voronoi region.

    Parameters
    ----------
    n_bits:
        Bit width (number of quantization levels = 2^n_bits).
    max_iter:
        Maximum number of Lloyd-Max iterations.
    tol:
        Convergence tolerance on centroid movement.
    seed:
        Random seed for initialisation.

    Examples
    --------
    >>> lm = LloydMaxQuantizer(n_bits=4)
    >>> lm.fit(data_samples)        # data_samples: 1D float tensor
    >>> indices = lm.quantize(x)   # integer indices
    >>> x_hat = lm.dequantize(indices)
    """

    def __init__(
        self,
        n_bits: int = 4,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if n_bits < 1 or n_bits > 16:
            raise ValueError(f"n_bits must be in [1, 16], got {n_bits}.")
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self._centroids: Optional[torch.Tensor] = None  # (n_levels,)
        self._is_fitted: bool = False

    @property
    def centroids(self) -> torch.Tensor:
        """Learned centroids of shape ``(n_levels,)``."""
        if not self._is_fitted:
            raise RuntimeError("LloydMaxQuantizer has not been fitted.  Call fit() first.")
        return self._centroids  # type: ignore[return-value]

    def fit(self, data: torch.Tensor) -> "LloydMaxQuantizer":
        """Fit the codebook from scalar data samples.

        Parameters
        ----------
        data:
            1D tensor of scalar samples (any dtype, cast to float32).

        Returns
        -------
        LloydMaxQuantizer
            ``self`` (for chaining).
        """
        data = data.float().flatten()
        if data.numel() < self.n_levels:
            logger.warning(
                "LloydMax: fewer data samples (%d) than quantization levels (%d); "
                "using uniform initialisation.",
                data.numel(),
                self.n_levels,
            )

        # Initialise centroids uniformly between [min, max]
        d_min, d_max = float(data.min()), float(data.max())
        if d_min == d_max:
            # Degenerate case: all values identical
            self._centroids = torch.full((self.n_levels,), d_min)
            self._is_fitted = True
            return self

        centroids = torch.linspace(d_min, d_max, self.n_levels, dtype=torch.float32)

        for iteration in range(self.max_iter):
            # Step 1: assign each sample to the nearest centroid
            dists = (data.unsqueeze(1) - centroids.unsqueeze(0)).abs()  # (n, n_levels)
            assignments = dists.argmin(dim=1)  # (n,)

            # Step 2: update centroids as the mean of assigned samples
            new_centroids = centroids.clone()
            for k in range(self.n_levels):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = data[mask].mean()

            # Check convergence
            shift = (new_centroids - centroids).abs().max().item()
            centroids = new_centroids
            if shift < self.tol:
                logger.debug("LloydMax converged at iteration %d (shift=%.2e).", iteration, shift)
                break
        else:
            logger.debug("LloydMax reached max_iter=%d without convergence.", self.max_iter)

        self._centroids = centroids.sort().values  # keep sorted for binary search
        self._is_fitted = True
        return self

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor ``x`` to integer indices.

        Parameters
        ----------
        x:
            Input tensor of any shape (scalar elements).

        Returns
        -------
        torch.Tensor
            Integer tensor of indices (dtype ``torch.int32``), same shape
            as ``x``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before quantize().")
        centroids = self._centroids.to(x.device)  # type: ignore[union-attr]
        orig_shape = x.shape
        flat = x.float().flatten()

        # Assign each element to nearest centroid
        dists = (flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()  # (n, n_levels)
        indices = dists.argmin(dim=1).to(torch.int32)
        return indices.reshape(orig_shape)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct values from integer indices.

        Parameters
        ----------
        indices:
            Integer tensor of indices (as returned by :meth:`quantize`).

        Returns
        -------
        torch.Tensor
            Reconstructed float32 tensor, same shape as ``indices``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before dequantize().")
        centroids = self._centroids.to(indices.device)  # type: ignore[union-attr]
        idx_clamped = indices.long().clamp(0, self.n_levels - 1)
        return centroids[idx_clamped]

    def mse(self, x: torch.Tensor) -> float:
        """Compute quantization MSE on ``x``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        float
            Mean squared error after quantize → dequantize round-trip.
        """
        x_hat = self.dequantize(self.quantize(x))
        return float((x.float() - x_hat).pow(2).mean().item())


# ---------------------------------------------------------------------------
# Non-uniform quantizer
# ---------------------------------------------------------------------------

class NonUniformQuantizer:
    """End-to-end non-uniform quantizer combining bit allocation and Lloyd-Max.

    Operates on pre-rotated vectors where the first d_eff coordinates are in
    the semantic (high-energy) regime and the rest are in the tail regime.

    * Fits **separate** Lloyd-Max codebooks for semantic and tail coordinates,
      because their distributions differ substantially after spectral rotation.
    * Allocates bits via :class:`BitAllocator`.

    Parameters
    ----------
    eigenvalues:
        Per-head eigenvalues (descending) used to determine d_eff and to
        weight the codebook fitting.  Shape ``(head_dim,)``.
    avg_bits:
        Target average bits per dimension.
    max_lloyd_iter:
        Max iterations for Lloyd-Max fitting.
    seed:
        Random seed for Lloyd-Max initialisation.

    Examples
    --------
    >>> quantizer = NonUniformQuantizer(eigenvalues, avg_bits=4.0)
    >>> quantizer.fit(rotated_keys)  # shape: (n_tokens, head_dim)
    >>> compressed = quantizer.compress(x, d_eff=32, avg_bits=4.0)
    >>> x_hat = quantizer.decompress(compressed)
    """

    def __init__(
        self,
        eigenvalues: torch.Tensor,
        avg_bits: float = 4.0,
        max_lloyd_iter: int = 200,
        seed: int = 0,
    ) -> None:
        self._eigenvalues = eigenvalues.float()
        self._avg_bits = avg_bits
        self._head_dim = eigenvalues.shape[0]
        self._max_lloyd_iter = max_lloyd_iter
        self._seed = seed

        self._allocator = BitAllocator()
        self._semantic_quantizer: Optional[LloydMaxQuantizer] = None
        self._tail_quantizer: Optional[LloydMaxQuantizer] = None
        self._d_eff_int: int = 0
        self._b_high: int = 0
        self._b_low: int = 0
        self._is_fitted: bool = False

    def fit(
        self,
        rotated_data: torch.Tensor,
        d_eff: Optional[float] = None,
    ) -> "NonUniformQuantizer":
        """Fit Lloyd-Max codebooks from rotated data samples.

        Parameters
        ----------
        rotated_data:
            Pre-rotated tensor of shape ``(n_tokens, head_dim)``.
        d_eff:
            Effective rank.  If ``None``, computed from ``eigenvalues`` via
            the participation ratio formula.

        Returns
        -------
        NonUniformQuantizer
            ``self`` (for chaining).
        """
        if d_eff is None:
            lam = self._eigenvalues.double()
            sum_lam = lam.sum()
            sum_sq = (lam ** 2).sum()
            d_eff = float((sum_lam ** 2) / sum_sq) if sum_sq > 1e-12 else 1.0

        self._d_eff_int = max(1, min(round(d_eff), self._head_dim - 1))
        self._b_high, self._b_low = self._allocator.allocate(
            d_eff=d_eff, avg_bits=self._avg_bits, head_dim=self._head_dim
        )

        # Fit semantic codebook on the high-energy coordinates
        semantic_data = rotated_data[:, : self._d_eff_int].float().flatten()
        self._semantic_quantizer = LloydMaxQuantizer(
            n_bits=self._b_high, max_iter=self._max_lloyd_iter, seed=self._seed
        ).fit(semantic_data)

        # Fit tail codebook on the low-energy coordinates
        tail_data = rotated_data[:, self._d_eff_int :].float().flatten()
        self._tail_quantizer = LloydMaxQuantizer(
            n_bits=self._b_low, max_iter=self._max_lloyd_iter, seed=self._seed + 1
        ).fit(tail_data)

        self._is_fitted = True
        logger.info(
            "NonUniformQuantizer fitted: d_eff=%d, b_high=%d, b_low=%d, head_dim=%d",
            self._d_eff_int,
            self._b_high,
            self._b_low,
            self._head_dim,
        )
        return self

    def compress(
        self,
        x: torch.Tensor,
        d_eff: Optional[float] = None,
        avg_bits: Optional[float] = None,
    ) -> CompressedVector:
        """Compress a (batch of) rotated vector(s).

        Parameters
        ----------
        x:
            Pre-rotated tensor of shape ``(..., head_dim)``.
        d_eff:
            Override d_eff (uses fitted value if ``None``).
        avg_bits:
            Override avg_bits (uses fitted value if ``None``).

        Returns
        -------
        CompressedVector
            Compressed representation.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before compress().")

        # Optionally re-fit with different d_eff / avg_bits
        if d_eff is not None or avg_bits is not None:
            _d = d_eff if d_eff is not None else float(self._d_eff_int)
            _b = avg_bits if avg_bits is not None else self._avg_bits
            d_eff_int = max(1, min(round(_d), self._head_dim - 1))
            b_high, b_low = self._allocator.allocate(_d, _b, self._head_dim)
        else:
            d_eff_int = self._d_eff_int
            b_high, b_low = self._b_high, self._b_low

        orig_shape = x.shape
        x_f = x.float()

        semantic_part = x_f[..., :d_eff_int]
        tail_part = x_f[..., d_eff_int:]

        sem_idx = self._semantic_quantizer.quantize(semantic_part)  # type: ignore
        tail_idx = self._tail_quantizer.quantize(tail_part)  # type: ignore

        # Compute actual bits used
        n_elements = x_f[..., 0].numel()  # number of vectors
        actual_bits = n_elements * (d_eff_int * b_high + (self._head_dim - d_eff_int) * b_low)

        # Compute MSE
        sem_hat = self._semantic_quantizer.dequantize(sem_idx)  # type: ignore
        tail_hat = self._tail_quantizer.dequantize(tail_idx)  # type: ignore
        x_hat = torch.cat([sem_hat, tail_hat], dim=-1)
        mse_val = float((x_f - x_hat).pow(2).mean().item())

        return CompressedVector(
            semantic_indices=sem_idx,
            tail_indices=tail_idx,
            d_eff=d_eff_int,
            head_dim=self._head_dim,
            b_high=b_high,
            b_low=b_low,
            original_shape=orig_shape,
            actual_bits_used=float(actual_bits),
            mse=mse_val,
        )

    def decompress(self, compressed: CompressedVector) -> torch.Tensor:
        """Reconstruct the original (rotated) vector from a CompressedVector.

        Parameters
        ----------
        compressed:
            Output of :meth:`compress`.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape ``compressed.original_shape``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before decompress().")

        sem_hat = self._semantic_quantizer.dequantize(compressed.semantic_indices)  # type: ignore
        tail_hat = self._tail_quantizer.dequantize(compressed.tail_indices)  # type: ignore
        return torch.cat([sem_hat, tail_hat], dim=-1)

    def compression_ratio(self, x: torch.Tensor) -> float:
        """Compute the compression ratio relative to float32 storage.

        Parameters
        ----------
        x:
            Original tensor (used to determine size).

        Returns
        -------
        float
            Ratio of original bits (float32) to compressed bits.
        """
        original_bits = x.numel() * 32  # float32
        d_tail = self._head_dim - self._d_eff_int
        compressed_bits = (
            (x[..., 0].numel()) * (self._d_eff_int * self._b_high + d_tail * self._b_low)
        )
        if compressed_bits == 0:
            return 0.0
        return original_bits / compressed_bits
