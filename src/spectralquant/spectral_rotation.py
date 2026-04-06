"""
Spectral rotation module for SpectralQuant.

Provides:
- ``BaseRotation``: Abstract base class defining the rotation interface.
- ``SpectralRotation``: Data-driven orthogonal rotation using calibrated
  eigenvectors (V^T projects vectors into the spectral domain).
- ``RandomRotation``: Haar-distributed random orthogonal rotation
  (TurboQuant baseline).

All rotation classes operate on batched tensors of shape
``(batch, seq_len, head_dim)`` or ``(seq_len, head_dim)`` and preserve
inner products (orthogonality).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from spectralquant.calibration import EigenspectralCalibrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRotation(ABC):
    """Abstract base class for rotation transforms applied to KV head vectors.

    Subclasses must implement :meth:`rotate` and :meth:`unrotate`.  Both
    methods must be inverses of each other: ``unrotate(rotate(x)) ≡ x``.

    The transform is parameterised by ``(layer_idx, head_idx)`` so that
    different heads can use different rotation matrices.
    """

    @abstractmethod
    def rotate(
        self,
        x: torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> torch.Tensor:
        """Apply the forward rotation R to ``x``.

        .. math::

            \\hat{x} = R^\\top x \\quad \\text{(i.e. } \\hat{x}_i = v_i^\\top x \\text{)}

        Parameters
        ----------
        x:
            Input tensor.  Shape ``(..., head_dim)``.
        layer_idx:
            Layer index (selects the per-layer rotation matrix).
        head_idx:
            Head index (selects the per-head rotation matrix).

        Returns
        -------
        torch.Tensor
            Rotated tensor, same shape as ``x``.
        """

    @abstractmethod
    def unrotate(
        self,
        x: torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> torch.Tensor:
        """Apply the inverse rotation R to ``x``.

        .. math::

            x = R \\hat{x}

        Parameters
        ----------
        x:
            Rotated tensor.  Shape ``(..., head_dim)``.
        layer_idx:
            Layer index.
        head_idx:
            Head index.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor in the original basis.
        """

    def rotate_batch(
        self,
        x: torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> torch.Tensor:
        """Convenience wrapper: rotate ``(batch, seq_len, head_dim)`` tensors.

        Delegates to :meth:`rotate`.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, seq_len, head_dim)``.
        layer_idx:
            Layer index.
        head_idx:
            Head index.

        Returns
        -------
        torch.Tensor
            Rotated tensor, same shape as ``x``.
        """
        return self.rotate(x, layer_idx, head_idx)

    def unrotate_batch(
        self,
        x: torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> torch.Tensor:
        """Convenience wrapper: unrotate ``(batch, seq_len, head_dim)`` tensors.

        Delegates to :meth:`unrotate`.

        Parameters
        ----------
        x:
            Rotated tensor of shape ``(batch, seq_len, head_dim)``.
        layer_idx:
            Layer index.
        head_idx:
            Head index.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor, same shape as ``x``.
        """
        return self.unrotate(x, layer_idx, head_idx)


# ---------------------------------------------------------------------------
# Spectral rotation (data-driven)
# ---------------------------------------------------------------------------

class SpectralRotation(BaseRotation):
    """Data-driven orthogonal rotation using per-head calibrated eigenvectors.

    Given the eigenvector matrix V (columns = eigenvectors, descending λ order)
    computed by :class:`~spectralquant.calibration.EigenspectralCalibrator`:

    * **Forward rotation** (project into spectral basis):
      :math:`\\hat{x} = V^\\top x`
    * **Inverse rotation** (reconstruct original basis):
      :math:`x = V \\hat{x}`

    Because V is orthogonal, :math:`V V^\\top = V^\\top V = I`, so dot products
    are exactly preserved and the transform is lossless.

    After rotation, the first :math:`d_{\\text{eff}}` coordinates concentrate
    most of the signal energy (semantic regime) and the remaining coordinates
    carry little energy (tail regime), enabling non-uniform quantization.

    Parameters
    ----------
    calibrator:
        Fitted :class:`~spectralquant.calibration.EigenspectralCalibrator`.
    head_type:
        Whether to use key (``"key"``) or value (``"value"``) eigenvectors.
        Defaults to ``"key"``; set to ``"value"`` when rotating value tensors.

    Examples
    --------
    >>> rot = SpectralRotation(calibrator, head_type="key")
    >>> k_rot = rot.rotate(keys, layer_idx=0, head_idx=3)  # V^T @ keys
    >>> k_rec = rot.unrotate(k_rot, layer_idx=0, head_idx=3)  # V @ k_rot
    """

    def __init__(
        self,
        calibrator: EigenspectralCalibrator,
        head_type: str = "key",
    ) -> None:
        if head_type not in ("key", "value"):
            raise ValueError(f"head_type must be 'key' or 'value', got '{head_type}'.")
        self._calibrator = calibrator
        self._head_type = head_type
        # Cache: (layer_idx, head_idx) -> (V, V^T)
        self._cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_matrices(
        self, layer_idx: int, head_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(V, V^T)`` for the requested head, with caching."""
        key = (layer_idx, head_idx)
        if key not in self._cache:
            hcd = self._calibrator.get(layer_idx, head_idx, self._head_type)
            if hcd is None:
                raise KeyError(
                    f"No calibration data for layer={layer_idx}, head={head_idx}, "
                    f"type='{self._head_type}'.  Run calibrator.calibrate() first."
                )
            V = hcd.eigenvectors.float()   # (head_dim, head_dim)
            Vt = V.T.contiguous()          # (head_dim, head_dim)
            self._cache[key] = (V, Vt)
        return self._cache[key]

    def rotate(
        self,
        x: torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> torch.Tensor:
        """Project ``x`` into the spectral basis: :math:`\\hat{x} = V^\\top x`.

        Parameters
        ----------
        x:
            Tensor of shape ``(..., head_dim)``.
        layer_idx:
            Transformer layer index.
        head_idx:
            Attention head index.

        Returns
        -------
        torch.Tensor
            Spectrally rotated tensor, same shape as ``x``.
        """
        _, Vt = self._get_matrices(layer_idx, head_idx)
        Vt = Vt.to(x.device)
        # x: (..., head_dim), Vt: (head_dim, head_dim)
        # Result: (..., head_dim) = x @ V (column-wise: each row of x multiplied by V)
        # Equivalently: (V^T x^T)^T = x @ V
        return x @ Vt.T  # x @ V == (V^T x^T)^T

    def unrotate(
        self,
        x: torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> torch.Tensor:
        """Reconstruct original basis: :math:`x = V \\hat{x}`.

        Parameters
        ----------
        x:
            Rotated tensor of shape ``(..., head_dim)``.
        layer_idx:
            Transformer layer index.
        head_idx:
            Attention head index.

        Returns
        -------
        torch.Tensor
            Tensor in the original basis, same shape as ``x``.
        """
        V, _ = self._get_matrices(layer_idx, head_idx)
        V = V.to(x.device)
        return x @ V.T  # x @ V^T == (V x^T)^T; note V^T = (V^T)^T ... unrotate via V

    def get_eigenvectors(
        self, layer_idx: int, head_idx: int
    ) -> torch.Tensor:
        """Return the eigenvector matrix V for a specific head.

        Parameters
        ----------
        layer_idx:
            Transformer layer index.
        head_idx:
            Attention head index.

        Returns
        -------
        torch.Tensor
            Orthonormal eigenvector matrix of shape ``(head_dim, head_dim)``.
        """
        V, _ = self._get_matrices(layer_idx, head_idx)
        return V


# ---------------------------------------------------------------------------
# Random rotation (TurboQuant baseline)
# ---------------------------------------------------------------------------

def _haar_random_orthogonal(d: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate a Haar-distributed random orthogonal matrix of size ``(d, d)``.

    Uses the QR decomposition of a standard Gaussian matrix, which yields
    a matrix uniformly distributed on the orthogonal group O(d).

    Parameters
    ----------
    d:
        Matrix dimension.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Orthogonal matrix of shape ``(d, d)``, float32.
    """
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None

    Z = torch.randn(d, d, generator=generator, dtype=torch.float32)
    Q, R = torch.linalg.qr(Z)
    # Ensure uniform distribution by correcting sign of columns
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs.unsqueeze(0)
    return Q


class RandomRotation(BaseRotation):
    """Haar-distributed random orthogonal rotation (TurboQuant baseline).

    Generates a fixed random orthogonal matrix Π for each ``(layer, head)``
    pair (seeded deterministically from layer/head indices for reproducibility).
    All heads share the same dimensional rotation if a global seed is provided.

    Parameters
    ----------
    head_dim:
        Dimension of each attention head vector.
    n_layers:
        Number of transformer layers.
    n_heads:
        Number of attention heads per layer.
    global_seed:
        Base random seed.  The per-head seed is derived as
        ``global_seed + layer_idx * n_heads + head_idx``.

    Examples
    --------
    >>> rot = RandomRotation(head_dim=128, n_layers=32, n_heads=32, global_seed=42)
    >>> k_rot = rot.rotate(keys, layer_idx=0, head_idx=0)
    >>> k_rec = rot.unrotate(k_rot, layer_idx=0, head_idx=0)
    """

    def __init__(
        self,
        head_dim: int,
        n_layers: int,
        n_heads: int,
        global_seed: int = 42,
    ) -> None:
        self._head_dim = head_dim
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._global_seed = global_seed
        # Lazy cache: (layer_idx, head_idx) -> (Pi, Pi^T)
        self._cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_matrices(
        self, layer_idx: int, head_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Π, Π^T) for the given head, with lazy generation and caching."""
        key = (layer_idx, head_idx)
        if key not in self._cache:
            seed = self._global_seed + layer_idx * self._n_heads + head_idx
            Pi = _haar_random_orthogonal(self._head_dim, seed=seed)
            self._cache[key] = (Pi, Pi.T.contiguous())
        return self._cache[key]

    def rotate(
        self,
        x: torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> torch.Tensor:
        """Apply the random orthogonal rotation: :math:`\\hat{x} = \\Pi^\\top x`.

        Parameters
        ----------
        x:
            Tensor of shape ``(..., head_dim)``.
        layer_idx:
            Layer index.
        head_idx:
            Head index.

        Returns
        -------
        torch.Tensor
            Rotated tensor, same shape as ``x``.
        """
        Pi, _ = self._get_matrices(layer_idx, head_idx)
        Pi = Pi.to(x.device)
        return x @ Pi

    def unrotate(
        self,
        x: torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> torch.Tensor:
        """Inverse rotation: :math:`x = \\Pi \\hat{x}`.

        Parameters
        ----------
        x:
            Rotated tensor of shape ``(..., head_dim)``.
        layer_idx:
            Layer index.
        head_idx:
            Head index.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor in the original basis.
        """
        _, PiT = self._get_matrices(layer_idx, head_idx)
        PiT = PiT.to(x.device)
        return x @ PiT

    def get_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """Return the random orthogonal matrix Π for a specific head.

        Parameters
        ----------
        layer_idx:
            Layer index.
        head_idx:
            Head index.

        Returns
        -------
        torch.Tensor
            Orthogonal matrix of shape ``(head_dim, head_dim)``.
        """
        Pi, _ = self._get_matrices(layer_idx, head_idx)
        return Pi
