"""SpectralQuantEngine — drop-in replacement for TurboQuantEngine.

Replaces random rotation (Pi) with calibrated spectral rotation (V^T),
uniform bit allocation with non-uniform allocation, and full QJL with
selective QJL on the semantic (high-eigenvalue) regime only.

All cuTile CUDA kernels are reused transparently because we subclass
TurboQuantEngine and only replace the rotation matrices, codebooks,
and the PyTorch fallback paths.  When ``cuda.tile`` is unavailable (e.g.
during local development) the pure-PyTorch paths are used exclusively.

Key changes vs TurboQuantEngine:
    1. ``Pi``  ← V   (eigenvector matrix, columns sorted by eigenvalue desc)
    2. ``PiT`` ← V^T (spectral rotation — projects into eigenbasis)
    3. ``key_codebook``   ← unified fallback + per-regime codebooks
    4. ``val_codebook``   ← unified fallback + per-regime codebooks
    5. ``compress_keys_pytorch``   — non-uniform bits + selective QJL
    6. ``compress_values_pytorch`` — non-uniform bits (two-regime)
    7. ``decompress_values_pytorch`` — two-regime dequantization
    8. ``attention_scores_pytorch`` — selective QJL correction (÷ d_eff, not ÷ d)

The ``launch_*`` methods inherited from TurboQuantEngine will automatically
use the spectral Pi/PiT/codebooks when CUDA kernels are available,
giving the spectral-rotation benefit at kernel speed.  The per-regime
logic (non-uniform bit allocation) is only exercised on the PyTorch path;
on the CUDA path a single codebook tuned to ``d_eff`` is used.
"""

from __future__ import annotations

import math
import sys
import logging
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Bootstrap: add TurboQuant to sys.path so we can import it.
# On Modal the repo is at /root/spectralquant/baseline/turboquant_cutile/,
# locally it may not exist yet (tests mock it).
# ---------------------------------------------------------------------------
_BASELINE_DIR = Path(__file__).parent.parent.parent / "baseline" / "turboquant_cutile"
if _BASELINE_DIR.exists():
    sys.path.insert(0, str(_BASELINE_DIR))

try:
    from turboquant_cutile.host import TurboQuantEngine, _generate_qjl_matrix
    from turboquant_cutile.codebook import LloydMaxCodebook, solve_lloyd_max
    from turboquant_cutile.constants import HEAD_DIM, DEFAULT_SEED
    _TURBOQUANT_AVAILABLE = True
except ImportError:  # pragma: no cover — not installed locally
    _TURBOQUANT_AVAILABLE = False
    # Provide stubs so the module is importable for linting / offline testing.
    HEAD_DIM = 128
    DEFAULT_SEED = 42

    class _StubCodebook:
        """Minimal stub for offline use."""
        def __init__(self, d: int, bits: int):
            self.d = d
            self.bits = bits
            self.n_levels = 1 << bits
            sigma = 1.0 / math.sqrt(d)
            # Uniform centroids as placeholder
            self.centroids = torch.linspace(-3.5 * sigma, 3.5 * sigma, self.n_levels)
            self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2.0

    class _StubEngine:
        """Minimal stub so SpectralQuantEngine can be defined offline."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "turboquant_cutile is not installed.  "
                "Install it from /baseline/turboquant_cutile or run on Modal."
            )

    TurboQuantEngine = _StubEngine  # type: ignore[assignment,misc]
    LloydMaxCodebook = _StubCodebook  # type: ignore[assignment,misc]

    def _generate_qjl_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:  # type: ignore[misc]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 10000)
        return torch.randn(d, d, generator=gen).to(device)

    def solve_lloyd_max(d: int, bits: int, **kwargs):  # type: ignore[misc]
        raise RuntimeError("turboquant_cutile not available")


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solve_lloyd_max_for_sigma(
    sigma: float,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> torch.Tensor:
    """Solve Lloyd-Max for N(0, sigma^2) distribution.

    The standard ``solve_lloyd_max(d, bits)`` assumes sigma = 1/sqrt(d).
    This helper lets us specify any sigma, which is needed for per-regime
    codebooks where the variance is proportional to the eigenvalues.

    Args:
        sigma: Standard deviation of the target Gaussian.
        bits: Number of quantization bits (2^bits levels).
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance.

    Returns:
        centroids: (2^bits,) float32 tensor of optimal centroids.
    """
    from scipy import integrate

    n_levels = 1 << bits
    pdf = lambda x: (
        (1.0 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-x * x / (2 * sigma * sigma))
    )

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    return torch.tensor(centroids, dtype=torch.float32)


# ---------------------------------------------------------------------------
# SpectralQuantEngine
# ---------------------------------------------------------------------------

class SpectralQuantEngine(TurboQuantEngine):
    """SpectralQuant: data-aware spectral rotation + non-uniform bit allocation.

    Drop-in replacement for ``TurboQuantEngine``.  Same external interface
    (``compress_keys_pytorch``, ``compress_values_pytorch``,
    ``decompress_values_pytorch``, ``attention_scores_pytorch``,
    ``fused_attention_pytorch``), same CUDA kernels — but the rotation
    matrix and codebooks are replaced to exploit per-head spectral structure.

    How it works
    ------------
    After applying V^T (the eigenvector rotation), coordinate ``i`` has
    variance proportional to eigenvalue ``λ_i``.  The first ``d_eff``
    coordinates (the *semantic regime*) carry most of the energy and
    deserve more bits; the remaining ``d - d_eff`` (*tail regime*) have
    low variance and can be coarsely quantized.

    Specifically:

    * **Keys** — semantic coords get ``b_high`` bits (including 1 bit
      reserved for selective QJL); tail coords get ``b_low`` bits with
      no QJL.
    * **Values** — same split (``b_high`` / ``b_low``) but no QJL at all.
    * **QJL** — only the residual projected onto the semantic subspace
      is signed; the correction term divides by ``d_eff`` instead of
      ``d`` (since only ``d_eff`` projections are used).

    CUDA kernel compatibility
    -------------------------
    The ``launch_*`` methods inherited from TurboQuantEngine use ``self.Pi``
    and ``self.val_codebook.centroids`` directly.  Because we replace
    ``self.Pi``/``self.PiT`` with V/V^T and set ``self.val_codebook`` to a
    codebook calibrated to the average post-rotation variance, the CUDA
    path already benefits from the spectral rotation.  The per-regime
    non-uniform quantization is only applied on the pure-PyTorch path.

    Args:
        eigenvectors: [head_dim, head_dim] — columns are eigenvectors
            sorted by eigenvalue *descending* (column 0 = largest λ).
        eigenvalues:  [head_dim] — eigenvalues sorted descending.
        d_eff: Effective dimensionality (participation ratio).  The first
            ``d_eff`` rotated coordinates form the semantic regime.
        head_dim: Dimension per head (default 128).
        total_bits: Target average bits per coordinate (default 3).
        b_high: Bits for semantic regime.  If ``None``, solved automatically.
        b_low:  Bits for tail regime.    If ``None``, solved automatically.
        seed: RNG seed for the QJL projection matrix.
        device: ``'cuda'`` or ``'cpu'``.
    """

    def __init__(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        d_eff: int,
        head_dim: int = HEAD_DIM,
        total_bits: int = 3,
        b_high: Optional[int] = None,
        b_low: Optional[int] = None,
        seed: int = DEFAULT_SEED,
        device: str = "cpu",
    ):
        # Skip super().__init__() — we build all state ourselves so that
        # we don't waste time generating a random Pi we'll immediately discard.
        self.head_dim = head_dim
        self.total_bits = total_bits
        self.d_eff = d_eff
        self.eigenvalues = eigenvalues.to(device)
        self.device = device

        # ------------------------------------------------------------------
        # SPECTRAL ROTATION: V^T replaces the Haar-random Pi^T.
        #
        # V's columns are eigenvectors sorted by eigenvalue (descending).
        # Rotating by V^T maps each key/value into the eigenbasis so that:
        #   coord 0 has the highest variance ~ λ_0
        #   coord d-1 has the lowest variance  ~ λ_{d-1}
        # This is exactly what TurboQuant assumes its random rotation
        # achieves on average — but spectral rotation achieves it exactly
        # for this specific head's distribution.
        # ------------------------------------------------------------------
        V = eigenvectors.to(device).float()  # [head_dim, head_dim]
        assert V.shape == (head_dim, head_dim), (
            f"Expected eigenvectors of shape ({head_dim}, {head_dim}), got {V.shape}"
        )
        # Pi is the "un-rotation" matrix (used to go back to original basis).
        # PiT is the "rotation" matrix (maps vectors into eigenbasis).
        self.Pi = V                       # [head_dim, head_dim]
        self.PiT = V.T.contiguous()       # [head_dim, head_dim]

        # ------------------------------------------------------------------
        # QJL MATRIX: still random (only rotation changes in SpectralQuant).
        # ------------------------------------------------------------------
        self.S = _generate_qjl_matrix(head_dim, seed, device)
        self.ST = self.S.T.contiguous()

        # ------------------------------------------------------------------
        # NON-UNIFORM BIT ALLOCATION
        #
        # TurboQuant uses uniform bits (mse_bits) for every coordinate.
        # We split coordinates into two regimes:
        #   semantic (0 : d_eff)   — high eigenvalue, high variance → more bits
        #   tail     (d_eff : d)   — low  eigenvalue, low  variance → fewer bits
        #
        # For keys: semantic regime gets (mse_bits_high + 1 QJL sign) bits
        #           tail regime gets mse_bits_low bits, no QJL.
        # For values: same split but no QJL in either regime.
        # ------------------------------------------------------------------
        if b_high is None or b_low is None:
            b_high, b_low = self._solve_bit_allocation(total_bits, d_eff, head_dim)
        self.b_high = b_high
        self.b_low = b_low
        # Reserve 1 bit for QJL sign in the semantic regime of keys.
        self.mse_bits_high = max(b_high - 1, 1)
        self.mse_bits_low = b_low              # tail: no QJL
        self.mse_bits = max(total_bits - 1, 1) # for compatibility with base class

        # ------------------------------------------------------------------
        # CODEBOOKS
        #
        # Uniform fallback (single codebook, same interface as TurboQuant).
        # These are used by the inherited CUDA launch_* methods when
        # cuda.tile is available.
        #
        # The average post-rotation variance is:
        #   sigma^2 = (1/d) * sum(λ_i) = trace(C) / d
        # so we calibrate the unified codebook to N(0, sigma^2).
        # Because LloydMaxCodebook uses sigma = 1/sqrt(d_eff_param) internally
        # (sigma = 1/sqrt(d)), we pass d_eff_param = d * total_var (scaling trick).
        # ------------------------------------------------------------------
        total_var = float(eigenvalues.sum().item())  # trace(C)
        mean_var = total_var / head_dim              # average per-coordinate variance
        # d_eff_param such that 1/sqrt(d_eff_param) = sqrt(mean_var)
        # => d_eff_param = 1 / mean_var
        if mean_var > 0:
            d_uniform = max(1, round(1.0 / mean_var))
        else:
            d_uniform = head_dim
        self.key_codebook = LloydMaxCodebook(d_uniform, self.mse_bits)
        self.val_codebook = LloydMaxCodebook(d_uniform, total_bits)

        # ------------------------------------------------------------------
        # PER-REGIME CODEBOOKS (used on the PyTorch path only)
        #
        # After spectral rotation, semantic coords ~ N(0, σ_high²) and
        # tail coords ~ N(0, σ_low²), where:
        #   σ_high² = mean(λ_0 : λ_{d_eff})    (average semantic eigenvalue)
        #   σ_low²  = mean(λ_{d_eff} : λ_d)    (average tail eigenvalue)
        #
        # We scale the codebook boundaries accordingly instead of relying
        # on the N(0, 1/d) assumption that TurboQuant makes.
        # ------------------------------------------------------------------
        ev = eigenvalues.cpu()

        sigma_high = float(ev[:d_eff].mean().clamp(min=1e-8).sqrt().item())
        sigma_low  = float(ev[d_eff:].mean().clamp(min=1e-8).sqrt().item()) if d_eff < head_dim else sigma_high
        # Guard: if sigma_low is extremely small relative to sigma_high,
        # clamp it to avoid degenerate codebooks that produce NaN
        if sigma_low < sigma_high * 1e-4:
            sigma_low = sigma_high * 1e-4
            log.debug("Clamped sigma_low to %.6f (was near-zero)", sigma_low)

        # Build per-regime centroids (as tensors; we embed them in thin
        # wrapper objects to preserve the .centroids attribute interface).
        self._centroids_key_high = _solve_lloyd_max_for_sigma(sigma_high, self.mse_bits_high)
        self._centroids_key_low  = _solve_lloyd_max_for_sigma(sigma_low,  self.mse_bits_low)
        self._centroids_val_high = _solve_lloyd_max_for_sigma(sigma_high, b_high)
        self._centroids_val_low  = _solve_lloyd_max_for_sigma(sigma_low,  b_low)

        # ------------------------------------------------------------------
        # Scalar constants
        # ------------------------------------------------------------------
        self.scale = 1.0 / math.sqrt(head_dim)
        # Selective QJL uses only d_eff projections → correction scales by
        # sqrt(π/2) / d_eff (vs. sqrt(π/2) / d in full QJL).
        self.correction_scale = math.sqrt(math.pi / 2) / d_eff

        log.debug(
            "SpectralQuantEngine: head_dim=%d, d_eff=%d, b_high=%d, b_low=%d, "
            "σ_high=%.4f, σ_low=%.4f",
            head_dim, d_eff, b_high, b_low, sigma_high, sigma_low,
        )

    # ------------------------------------------------------------------
    # Bit allocation solver
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_bit_allocation(avg_bits: int, d_eff: int, d: int) -> tuple[int, int]:
        """Solve integer bit widths for semantic and tail regimes.

        Constraint: ``d_eff * b_high + (d - d_eff) * b_low ≈ d * avg_bits``.

        For keys, ``b_high`` includes one bit reserved for the QJL sign,
        so the MSE quantizer in the semantic regime gets ``b_high - 1`` bits.
        This naturally gives ``b_high = b_low + 1``, and the budget equation
        becomes:

            d_eff * (b + 1) + (d - d_eff) * b = d * avg_bits
            d * b + d_eff = d * avg_bits
            b = avg_bits - d_eff / d

        Args:
            avg_bits: Target average bits per coordinate.
            d_eff: Number of semantic (high-eigenvalue) coordinates.
            d: Total head dimension.

        Returns:
            (b_high, b_low): Integer bit widths for semantic and tail regimes.
        """
        b_low = max(1, round(avg_bits - d_eff / d))
        b_high = b_low + 1

        actual_avg = (d_eff * b_high + (d - d_eff) * b_low) / d
        log.debug(
            "_solve_bit_allocation: avg_bits=%d, d_eff=%d, d=%d → "
            "b_high=%d, b_low=%d, actual_avg=%.3f",
            avg_bits, d_eff, d, b_high, b_low, actual_avg,
        )
        return b_high, b_low

    # ------------------------------------------------------------------
    # Internal quantize helpers (per-regime)
    # ------------------------------------------------------------------

    def _quantize_regime(
        self, x: torch.Tensor, centroids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest-centroid scalar quantization.

        Args:
            x: (..., dim) float32 input.
            centroids: (n_levels,) float32 codebook centroids.

        Returns:
            indices: (..., dim) uint8 quantized indices.
            y_hat:   (..., dim) float32 reconstructed values.
        """
        c = centroids.to(x.device)
        diffs = x.unsqueeze(-1) - c                     # (..., dim, n_levels)
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)
        y_hat = c[indices.long()]                       # (..., dim)
        return indices, y_hat

    # ------------------------------------------------------------------
    # Key compression (override)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compress_keys_pytorch(self, K: torch.Tensor) -> dict:
        """Compress keys with spectral rotation and non-uniform bit allocation.

        **Differences from TurboQuantEngine.compress_keys_pytorch:**

        1. *Rotation*: uses V^T (spectral) instead of random Pi^T.
        2. *Semantic regime* (coords ``0 : d_eff``): quantized with
           ``mse_bits_high`` bits.
        3. *Tail regime* (coords ``d_eff : head_dim``): quantized with
           ``mse_bits_low`` bits.
        4. *Selective QJL*: only the residual projected onto the
           ``d_eff``-dimensional semantic subspace is signed; tail
           dimensions receive zero signs in the output dict.

        The returned dict is a strict superset of TurboQuantEngine's output,
        so any downstream code that only reads ``k_mse``, ``qjl_signs``,
        ``vec_norms``, and ``residual_norms`` will work unchanged.

        Args:
            K: (seq_k, head_dim) key vectors (FP16 or FP32).

        Returns:
            Compressed dict with keys:
                ``indices``       — (seq_k, head_dim) uint8 quantized indices
                                    (semantic then tail, concatenated).
                ``k_mse``         — (seq_k, head_dim) FP16 MSE reconstruction.
                ``qjl_signs``     — (seq_k, head_dim) int8 QJL signs
                                    (non-zero only for first ``d_eff`` dims).
                ``vec_norms``     — (seq_k,) FP16 original L2 norms.
                ``residual_norms``— (seq_k,) FP16 norms of QJL residuals.
                ``d_eff``         — int, number of semantic coordinates.
                ``b_high``        — int, bits used in semantic regime.
                ``b_low``         — int, bits used in tail regime.
        """
        K_f = K.float()
        vec_norms = torch.norm(K_f, dim=-1, keepdim=True)           # (seq_k, 1)
        K_normed = K_f / (vec_norms + 1e-8)

        # --- Spectral rotation ---
        rotated = K_normed @ self.PiT.float()                        # (seq_k, head_dim)
        rotated_high = rotated[:, :self.d_eff]                       # semantic regime
        rotated_low  = rotated[:, self.d_eff:]                       # tail regime

        # --- Quantize semantic regime (mse_bits_high bits) ---
        idx_high, y_hat_high = self._quantize_regime(
            rotated_high, self._centroids_key_high
        )

        # --- Quantize tail regime (mse_bits_low bits) ---
        idx_low, y_hat_low = self._quantize_regime(
            rotated_low, self._centroids_key_low
        )

        # --- Reconstruct in rotated basis, then un-rotate → original basis ---
        y_hat = torch.cat([y_hat_high, y_hat_low], dim=-1)           # (seq_k, head_dim)
        indices = torch.cat([idx_high, idx_low], dim=-1)             # (seq_k, head_dim) uint8

        # k_mse: un-rotate then rescale by original norm
        k_mse = (y_hat @ self.Pi.float()) * vec_norms                # (seq_k, head_dim)

        # --- Selective QJL: only on the semantic subspace ---
        # The QJL residual correction estimates the inner product of the
        # residual with the query.  We only use d_eff projections (the tail
        # has very low energy and would only add noise).
        residual = K_f - k_mse
        residual_norms = torch.norm(residual, dim=-1)                # (seq_k,)

        # Project residual into semantic subspace via the spectral basis.
        residual_rotated  = residual @ self.PiT.float()              # (seq_k, head_dim)
        residual_semantic = residual_rotated[:, :self.d_eff]         # (seq_k, d_eff)

        # Use the top-left d_eff × d_eff block of the QJL matrix S.
        # This is sufficient because both query and residual are projected
        # to d_eff dims before the sign sketch.
        S_sel = self.S[:self.d_eff, :self.d_eff].float()             # (d_eff, d_eff)
        projected = residual_semantic @ S_sel.T                      # (seq_k, d_eff)
        signs_semantic = torch.sign(projected).to(torch.int8)
        signs_semantic[signs_semantic == 0] = 1

        # Pad signs to full head_dim so downstream code can handle the
        # dict uniformly.  Tail slots are zero — no correction applied.
        full_signs = torch.zeros(
            K.shape[0], self.head_dim, dtype=torch.int8, device=K.device
        )
        full_signs[:, :self.d_eff] = signs_semantic

        return {
            "indices":        indices,
            "k_mse":          k_mse.half(),
            "qjl_signs":      full_signs,
            "vec_norms":      vec_norms.squeeze(-1).half(),
            "residual_norms": residual_norms.half(),
            # Spectral metadata (not in TurboQuantEngine output — ignored by
            # downstream code that doesn't know about regimes).
            "d_eff": self.d_eff,
            "b_high": self.b_high,
            "b_low":  self.b_low,
        }

    # ------------------------------------------------------------------
    # Value compression (override)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compress_values_pytorch(self, V: torch.Tensor) -> dict:
        """Compress values with spectral rotation and non-uniform bit allocation.

        **Differences from TurboQuantEngine.compress_values_pytorch:**

        1. *Rotation*: uses V^T (spectral) instead of random Pi^T.
        2. *Non-uniform bits*: semantic regime uses ``b_high`` bits,
           tail regime uses ``b_low`` bits.
        3. The returned dict stores concatenated indices just like the
           key dict, but there is no QJL (values are only decompressed,
           never used in inner-product estimation).

        The returned dict is format-compatible with
        ``decompress_values_pytorch`` (overridden below) and also with
        the parent class's CUDA ``launch_*`` methods (which will see the
        unified codebook via ``self.val_codebook``).

        Args:
            V: (seq_v, head_dim) value vectors (FP16 or FP32).

        Returns:
            Compressed dict with keys:
                ``indices``   — (seq_v, head_dim) uint8 concatenated indices.
                ``vec_norms`` — (seq_v,) FP16 original L2 norms.
                ``d_eff``     — int, number of semantic coordinates.
                ``b_high``    — int, bits used in semantic regime.
                ``b_low``     — int, bits used in tail regime.
        """
        V_f = V.float()
        vec_norms = torch.norm(V_f, dim=-1, keepdim=True)            # (seq_v, 1)
        V_normed = V_f / (vec_norms + 1e-8)

        # --- Spectral rotation ---
        rotated = V_normed @ self.PiT.float()                         # (seq_v, head_dim)

        # --- Non-uniform quantization ---
        idx_high, _ = self._quantize_regime(
            rotated[:, :self.d_eff], self._centroids_val_high
        )
        idx_low, _ = self._quantize_regime(
            rotated[:, self.d_eff:], self._centroids_val_low
        )

        indices = torch.cat([idx_high, idx_low], dim=-1)              # (seq_v, head_dim) uint8

        return {
            "indices":   indices,
            "vec_norms": vec_norms.squeeze(-1).half(),
            "d_eff":     self.d_eff,
            "b_high":    self.b_high,
            "b_low":     self.b_low,
        }

    # ------------------------------------------------------------------
    # Value decompression (override)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decompress_values_pytorch(self, compressed_v: dict) -> torch.Tensor:
        """Reconstruct values from the two-regime compressed representation.

        **Differences from TurboQuantEngine.decompress_values_pytorch:**

        Uses separate codebooks for semantic and tail regimes, then
        un-rotates by V (= ``self.Pi``) to return to the original basis.

        If the compressed dict does not contain ``d_eff`` (e.g. it was
        produced by the parent class), falls back gracefully to the
        parent's single-codebook reconstruction.

        Args:
            compressed_v: Dict as returned by ``compress_values_pytorch``.

        Returns:
            (seq_v, head_dim) FP16 reconstructed value vectors.
        """
        # Graceful fallback: if this dict came from TurboQuantEngine
        # (no 'd_eff' key), delegate to parent implementation.
        if "d_eff" not in compressed_v:
            return super().decompress_values_pytorch(compressed_v)

        indices   = compressed_v["indices"]                           # (seq_v, head_dim) uint8
        vec_norms = compressed_v["vec_norms"].float()                 # (seq_v,)
        d_eff     = compressed_v["d_eff"]

        # Split indices back into regimes.
        idx_high = indices[:, :d_eff].long()
        idx_low  = indices[:, d_eff:].long()

        c_high = self._centroids_val_high.to(indices.device)
        c_low  = self._centroids_val_low.to(indices.device)

        y_hat_high = c_high[idx_high]                                  # (seq_v, d_eff)
        y_hat_low  = c_low[idx_low]                                    # (seq_v, head_dim - d_eff)

        y_hat = torch.cat([y_hat_high, y_hat_low], dim=-1)            # (seq_v, head_dim)

        # Un-rotate: V @ y_hat (y_hat is in eigenbasis) then rescale.
        reconstructed = (y_hat @ self.Pi.float()) * vec_norms.unsqueeze(-1)
        return reconstructed.half()

    # ------------------------------------------------------------------
    # Attention score estimation (override)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def attention_scores_pytorch(
        self, Q: torch.Tensor, compressed_k: dict
    ) -> torch.Tensor:
        """Asymmetric attention score estimator with selective QJL correction.

        **Differences from TurboQuantEngine.attention_scores_pytorch:**

        1. *Selective QJL*: only the top ``d_eff`` query components (in
           the eigenbasis) are used for the QJL correction, matching the
           ``d_eff``-dimensional sketch stored in ``qjl_signs``.
        2. *Correction scale*: divides by ``d_eff`` instead of ``d``
           because only ``d_eff`` QJL projections are used.  The full
           derivation gives:

               correction ≈ sqrt(π/2) · (1/d_eff) · |q_sem| · |r_sem|

           where ``q_sem`` and ``r_sem`` are the semantic projections of
           the query and key residual, respectively.

        The MSE term (term1) is identical to the parent class.

        Args:
            Q: (seq_q, head_dim) query vectors.
            compressed_k: dict as returned by ``compress_keys_pytorch``.

        Returns:
            (seq_q, seq_k) float32 attention logits (before softmax).
        """
        Q_f = Q.float()
        k_mse  = compressed_k["k_mse"].float()                        # (seq_k, head_dim)
        signs  = compressed_k["qjl_signs"].float()                    # (seq_k, head_dim)
        r_norms = compressed_k["residual_norms"].float()               # (seq_k,)

        # --- Term 1: MSE dot product (identical to TurboQuant) ---
        term1 = Q_f @ k_mse.T                                          # (seq_q, seq_k)

        # --- Term 2: Selective QJL correction (semantic regime only) ---
        # Project queries into eigenbasis and take only semantic coords.
        q_rotated  = Q_f @ self.PiT.float()                            # (seq_q, head_dim)
        q_semantic = q_rotated[:, :self.d_eff]                         # (seq_q, d_eff)

        # QJL inner product: (q_sem @ S_top^T) · signs_sem^T
        S_sel = self.S[:self.d_eff, :self.d_eff].float()               # (d_eff, d_eff)
        q_proj = q_semantic @ S_sel.T                                  # (seq_q, d_eff)

        signs_semantic = signs[:, :self.d_eff]                         # (seq_k, d_eff)
        qjl_ip = q_proj @ signs_semantic.T                             # (seq_q, seq_k)

        # Scale: sqrt(π/2) / d_eff · |r| per key
        term2 = self.correction_scale * qjl_ip * r_norms.unsqueeze(0)  # (seq_q, seq_k)

        return (term1 + term2) * self.scale

    # ------------------------------------------------------------------
    # Compression statistics
    # ------------------------------------------------------------------

    def compressed_size_bytes(self, seq_len: int) -> dict:
        """Compute actual compressed size with non-uniform bit allocation.

        Unlike ``TurboQuantEngine.compressed_size_bytes``, this accounts
        for the different bit widths used in the semantic and tail regimes.

        **Differences from TurboQuantEngine:**

        * Semantic regime: ``b_high`` bits (keys include 1 QJL sign bit).
        * Tail regime: ``b_low`` bits (no QJL).
        * ``avg_bits_per_coord`` is reported for comparison.

        Args:
            seq_len: Number of key/value vectors in the cache.

        Returns:
            dict with ``key_bytes``, ``val_bytes``, ``total_bytes``,
            ``fp16_bytes``, ``compression_ratio``, ``avg_bits_per_coord``.
        """
        d     = self.head_dim
        d_eff = self.d_eff

        # Keys: semantic regime = mse_bits_high + 1 QJL = b_high bits total
        key_sem_bits  = seq_len * d_eff * self.b_high
        key_tail_bits = seq_len * (d - d_eff) * self.b_low
        key_norm_bits = seq_len * 16 * 2           # vec_norm (FP16) + residual_norm (FP16)
        key_total = (key_sem_bits + key_tail_bits + key_norm_bits) / 8

        # Values: same regime split but no QJL
        val_sem_bits  = seq_len * d_eff * self.b_high
        val_tail_bits = seq_len * (d - d_eff) * self.b_low
        val_norm_bits = seq_len * 16               # vec_norm (FP16) only
        val_total = (val_sem_bits + val_tail_bits + val_norm_bits) / 8

        fp16_total = seq_len * d * 2 * 2           # K + V in FP16

        quant_bits = key_sem_bits + key_tail_bits + val_sem_bits + val_tail_bits
        avg_bpc = quant_bits / (seq_len * d * 2)

        return {
            "key_bytes":          key_total,
            "val_bytes":          val_total,
            "total_bytes":        key_total + val_total,
            "fp16_bytes":         fp16_total,
            "compression_ratio":  fp16_total / (key_total + val_total),
            "avg_bits_per_coord": avg_bpc,
        }

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_calibration(
        cls,
        calibration_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> "SpectralQuantEngine":
        """Load a SpectralQuantEngine from a saved calibration file.

        The calibration file must be a ``torch.save`` dict containing:
            ``eigenvectors``: [head_dim, head_dim] float tensor.
            ``eigenvalues``:  [head_dim] float tensor.
            ``d_eff``:        int or tensor.

        These are produced by ``EigenspectralCalibrator.save()`` in
        ``spectralquant.calibration``.

        Args:
            calibration_path: Path to ``.pt`` calibration file.
            device: Target device for tensors.
            **kwargs: Additional arguments forwarded to ``__init__``
                (e.g. ``total_bits``, ``b_high``, ``b_low``).

        Returns:
            Configured ``SpectralQuantEngine`` instance.
        """
        data = torch.load(calibration_path, map_location="cpu")
        return cls(
            eigenvectors=data["eigenvectors"],
            eigenvalues=data["eigenvalues"],
            d_eff=int(data["d_eff"]),
            device=device,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SpectralQuantEngine("
            f"head_dim={self.head_dim}, "
            f"d_eff={self.d_eff}, "
            f"total_bits={self.total_bits}, "
            f"b_high={self.b_high}, "
            f"b_low={self.b_low})"
        )
