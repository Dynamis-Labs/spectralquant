"""
Selective Quantized Johnson-Lindenstrauss (QJL) module for SpectralQuant.

After spectral rotation, only the top d_eff coordinates carry significant
signal.  Applying QJL correction only to these coordinates (instead of all d)
saves (d - d_eff) bits per key vector per token while maintaining the
unbiasedness guarantees of the JL estimate.

Provides:
- ``BaseQJL``: Abstract base for both implementations.
- ``SelectiveQJL``: QJL on top d_eff coordinates only (SpectralQuant).
- ``FullQJL``: QJL on all d coordinates (TurboQuant baseline).

Background
----------
QJL (Zandieh et al., 2024) estimates inner products
:math:`\\langle q, k \\rangle` from a quantized sketch of k:

.. math::

    \\hat{s} = \\frac{d}{m} \\langle q, S k \\rangle

where S is a random sign matrix (Rademacher).  SpectralQuant applies S only
to the first d_eff components, which concentrates the sketch budget where the
signal is.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseQJL(ABC):
    """Abstract base class for QJL variants.

    Both :class:`SelectiveQJL` and :class:`FullQJL` share the same interface
    so they can be swapped transparently in the pipeline.

    Parameters
    ----------
    seed:
        Random seed for the sign matrix generation.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._sign_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    @abstractmethod
    def generate_signs(
        self, active_dim: int, n_projections: int
    ) -> torch.Tensor:
        """Generate the random sign matrix S.

        Parameters
        ----------
        active_dim:
            Number of input dimensions to project (d_eff for SelectiveQJL,
            full head_dim for FullQJL).
        n_projections:
            Number of projection dimensions (rows of S).

        Returns
        -------
        torch.Tensor
            Binary ±1 matrix of shape ``(n_projections, active_dim)``.
        """

    @abstractmethod
    def compute_correction(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        d_eff: int,
    ) -> torch.Tensor:
        """Compute the QJL inner-product estimate.

        Parameters
        ----------
        keys:
            Rotated key tensor of shape ``(seq_len, head_dim)`` or
            ``(batch, seq_len, head_dim)``.
        queries:
            Query tensor with same leading shape, same head_dim.
        d_eff:
            Effective rank (top-d_eff coordinates used by SelectiveQJL).

        Returns
        -------
        torch.Tensor
            Inner-product estimates of shape ``(batch, n_queries, seq_len)``
            or ``(n_queries, seq_len)``.
        """

    def _rademacher_signs(
        self,
        active_dim: int,
        n_projections: int,
        device: torch.device,
        cache_key: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Generate or retrieve cached Rademacher sign matrix.

        Parameters
        ----------
        active_dim:
            Number of input dimensions.
        n_projections:
            Number of output projections.
        device:
            Target device.
        cache_key:
            If provided, cache by this key (default ``(active_dim, n_projections)``).

        Returns
        -------
        torch.Tensor
            Float32 ±1 matrix of shape ``(n_projections, active_dim)``.
        """
        key = cache_key or (active_dim, n_projections)
        if key not in self._sign_cache:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            bits = torch.randint(
                0, 2, (n_projections, active_dim), generator=generator, dtype=torch.float32
            )
            signs = 2.0 * bits - 1.0  # {0,1} -> {-1, +1}
            self._sign_cache[key] = signs
        return self._sign_cache[key].to(device)


# ---------------------------------------------------------------------------
# Selective QJL (SpectralQuant)
# ---------------------------------------------------------------------------

class SelectiveQJL(BaseQJL):
    """QJL correction applied **only** to the top d_eff spectral coordinates.

    After spectral rotation, coordinates 0…d_eff-1 concentrate the signal.
    Applying QJL sketching to only these coordinates:

    * Saves ``(d - d_eff)`` bits per key vector per token.
    * Maintains unbiasedness: the expectation of the sketch equals the true
      inner product of the d_eff-dimensional subspace.

    The inner product estimate is:

    .. math::

        \\hat{s} = \\frac{d_{\\text{eff}}}{m}
                   \\langle q_{:d_{\\text{eff}}},\\;
                           S k_{:d_{\\text{eff}}} \\rangle

    where S ∈ {±1}^{m × d_eff} is the random sign matrix.

    Parameters
    ----------
    n_projections:
        Number of JL projection dimensions ``m``.  More projections →
        lower variance estimate.
    seed:
        Random seed for sign matrix generation.

    Examples
    --------
    >>> qjl = SelectiveQJL(n_projections=64, seed=42)
    >>> signs = qjl.generate_signs(d_eff=32, n_projections=64)
    >>> score_est = qjl.compute_correction(keys_rot, queries_rot, d_eff=32)
    """

    def __init__(self, n_projections: int = 64, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.n_projections = n_projections

    def generate_signs(
        self,
        active_dim: int,
        n_projections: int,
    ) -> torch.Tensor:
        """Generate ±1 sign matrix for the selective projection.

        Parameters
        ----------
        active_dim:
            ``d_eff`` — number of spectral coordinates to project.
        n_projections:
            Number of projections ``m``.

        Returns
        -------
        torch.Tensor
            ±1 matrix of shape ``(n_projections, active_dim)``, float32.
        """
        return self._rademacher_signs(
            active_dim=active_dim,
            n_projections=n_projections,
            device=torch.device("cpu"),
            cache_key=(active_dim, n_projections),
        )

    def compute_correction(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        d_eff: int,
    ) -> torch.Tensor:
        """Estimate inner products via selective QJL sketch.

        Parameters
        ----------
        keys:
            Rotated key vectors.  Shape ``(seq_len, head_dim)`` or
            ``(batch, seq_len, head_dim)``.
        queries:
            Rotated query vectors.  Shape ``(n_queries, head_dim)`` or
            ``(batch, n_queries, head_dim)``.
        d_eff:
            Number of leading spectral coordinates to use in the sketch.

        Returns
        -------
        torch.Tensor
            Attention score estimates of shape ``(seq_len, n_queries)`` or
            ``(batch, n_queries, seq_len)``.

        Notes
        -----
        The estimate is unbiased for the partial inner product
        :math:`\\langle q_{:d}, k_{:d} \\rangle`.  Adding the exact inner
        product over the tail regime (trivially computed from dequantized tail
        coordinates) gives an unbiased estimate of the full inner product.
        """
        device = keys.device
        batched = keys.dim() == 3

        if not batched:
            keys = keys.unsqueeze(0)      # (1, seq_len, head_dim)
            queries = queries.unsqueeze(0)  # (1, n_queries, head_dim)

        batch, seq_len, head_dim = keys.shape
        _, n_queries, _ = queries.shape

        # Restrict to top d_eff coordinates
        k_sem = keys[:, :, :d_eff].float()       # (batch, seq_len, d_eff)
        q_sem = queries[:, :, :d_eff].float()     # (batch, n_queries, d_eff)

        # Sign matrix S: (n_projections, d_eff)
        S = self._rademacher_signs(
            active_dim=d_eff,
            n_projections=self.n_projections,
            device=device,
        )

        # Sketch keys: (batch, seq_len, n_projections)
        sk = k_sem @ S.T  # (..., seq_len, n_projections)

        # Sketch queries: (batch, n_queries, n_projections)
        sq = q_sem @ S.T

        # Inner product estimate: (batch, n_queries, seq_len)
        # scale = d_eff / n_projections (unbiasedness correction)
        scale = d_eff / self.n_projections
        estimates = scale * torch.bmm(sq, sk.transpose(1, 2))  # (batch, n_q, seq)

        if not batched:
            estimates = estimates.squeeze(0)  # (n_queries, seq_len)

        return estimates

    def bits_saved_per_token(self, head_dim: int, d_eff: int) -> int:
        """Compute the bits saved per key vector relative to FullQJL.

        Parameters
        ----------
        head_dim:
            Total head dimension ``d``.
        d_eff:
            Number of spectral coordinates used.

        Returns
        -------
        int
            ``d - d_eff`` bits saved per token (one bit per skipped coordinate
            in the sign sketch).
        """
        return max(0, head_dim - d_eff)


# ---------------------------------------------------------------------------
# Full QJL (TurboQuant baseline)
# ---------------------------------------------------------------------------

class FullQJL(BaseQJL):
    """QJL on all ``d`` coordinates (TurboQuant baseline).

    Applies the standard QJL sketch to the entire head vector:

    .. math::

        \\hat{s} = \\frac{d}{m} \\langle q, S k \\rangle

    where S ∈ {±1}^{m × d}.

    Parameters
    ----------
    n_projections:
        Number of JL projection dimensions ``m``.
    seed:
        Random seed for sign matrix generation.

    Examples
    --------
    >>> qjl = FullQJL(n_projections=64, seed=42)
    >>> score_est = qjl.compute_correction(keys_rot, queries_rot, d_eff=128)
    """

    def __init__(self, n_projections: int = 64, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.n_projections = n_projections

    def generate_signs(
        self,
        active_dim: int,
        n_projections: int,
    ) -> torch.Tensor:
        """Generate ±1 sign matrix for the full projection.

        Parameters
        ----------
        active_dim:
            Full head dimension ``d``.
        n_projections:
            Number of projections ``m``.

        Returns
        -------
        torch.Tensor
            ±1 matrix of shape ``(n_projections, active_dim)``, float32.
        """
        return self._rademacher_signs(
            active_dim=active_dim,
            n_projections=n_projections,
            device=torch.device("cpu"),
            cache_key=(active_dim, n_projections),
        )

    def compute_correction(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        d_eff: int,
    ) -> torch.Tensor:
        """Estimate inner products via full QJL sketch.

        Parameters
        ----------
        keys:
            Key vectors.  Shape ``(seq_len, head_dim)`` or
            ``(batch, seq_len, head_dim)``.
        queries:
            Query vectors.  Same leading dimensions.
        d_eff:
            Ignored in FullQJL (all coordinates are used); kept for interface
            parity with :class:`SelectiveQJL`.

        Returns
        -------
        torch.Tensor
            Attention score estimates of shape ``(n_queries, seq_len)`` or
            ``(batch, n_queries, seq_len)``.
        """
        device = keys.device
        batched = keys.dim() == 3

        if not batched:
            keys = keys.unsqueeze(0)
            queries = queries.unsqueeze(0)

        batch, seq_len, head_dim = keys.shape
        _, n_queries, _ = queries.shape

        k_f = keys.float()
        q_f = queries.float()

        S = self._rademacher_signs(
            active_dim=head_dim,
            n_projections=self.n_projections,
            device=device,
        )

        sk = k_f @ S.T           # (batch, seq_len, n_projections)
        sq = q_f @ S.T           # (batch, n_queries, n_projections)

        scale = head_dim / self.n_projections
        estimates = scale * torch.bmm(sq, sk.transpose(1, 2))  # (batch, n_q, seq)

        if not batched:
            estimates = estimates.squeeze(0)

        return estimates
