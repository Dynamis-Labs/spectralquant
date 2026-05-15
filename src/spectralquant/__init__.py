"""
SpectralQuant: KV Cache Compression via Eigenspectral Structure.

This library improves upon TurboQuant (Google, ICLR 2026) by exploiting
the eigenspectral structure of attention key/value vectors for more efficient
KV cache compression in large language models.

Main components:
    - EigenspectralCalibrator: Calibrates per-head spectral structure
    - SpectralRotation / RandomRotation: Rotation transforms for KV vectors
    - NonUniformQuantizer: Non-uniform bit allocation via Lloyd-Max
    - SelectiveQJL / FullQJL: Quantized Johnson-Lindenstrauss projections
    - SpectralQuantEngine: Full compression pipeline
    - TurboQuantBaseline: Baseline (random rotation + uniform quant + full QJL)
"""

from typing import Optional

from spectralquant.calibration import EigenspectralCalibrator, HeadCalibrationData
from spectralquant.spectral_rotation import (
    BaseRotation,
    SpectralRotation,
    RandomRotation,
)
from spectralquant.nonuniform_quantization import (
    BitAllocator,
    LloydMaxQuantizer,
    NonUniformQuantizer,
    CompressedVector,
    WaterfillAllocation,
)
from spectralquant.selective_qjl import SelectiveQJL, FullQJL
# --- Engine resolution (V1-GAP-010 / docs §4.2) -------------------------
#
# v2 has TWO engines:
#
# 1. ``spectralquant.spectralquant.SpectralQuantEngine`` — pure-Python
#    pipeline that drives ``NonUniformQuantizer`` directly.  This is the
#    **canonical engine for v2 benchmarks** (it carries water-filling,
#    runs locally without Modal, and is what tests cover).
#
# 2. ``spectralquant.engine.SpectralQuantEngine`` — kernel-accelerated
#    subclass of ``turboquant_cutile.TurboQuantEngine``.  Only available
#    on Modal where ``turboquant_cutile`` is installed; intended for
#    speed/latency measurements that need the cuTile kernels.
#
# To make benchmark scripts unambiguous, the canonical engine is
# re-exported under the unqualified name ``SpectralQuantEngine`` and the
# kernel variant is exported as ``KernelSpectralQuantEngine``.  The old
# ``_LegacySpectralQuantEngine`` alias is kept for one release for
# backwards compatibility with internal callers.
from spectralquant.spectralquant import (
    SpectralQuantEngine,
    TurboQuantBaseline,
    EngineConfig,
)

# Lazy import for the cuTile engine — failures must not break the
# pure-Python import path.  The engine module itself defines stubs when
# turboquant_cutile is unavailable, but its top-level ``import`` of
# scipy/torch (and the cuTile bootstrap) can fail in stripped-down
# environments.  Treat any import-time error as "kernel engine
# unavailable" and surface a clear RuntimeError on first use.
try:
    from spectralquant.engine import SpectralQuantEngine as KernelSpectralQuantEngine
    _KERNEL_ENGINE_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as _exc:  # pragma: no cover — depends on environment
    _KERNEL_ENGINE_IMPORT_ERROR = _exc

    class KernelSpectralQuantEngine:  # type: ignore[no-redef]
        """Placeholder for the cuTile engine when import fails locally.

        Instantiating this class raises ``RuntimeError`` so that pure-Python
        utilities can still import :mod:`spectralquant` on a clean checkout
        without ``turboquant_cutile`` (or torch's cuda extensions).
        """

        _import_error = _KERNEL_ENGINE_IMPORT_ERROR

        def __init__(self, *args, **kwargs):  # noqa: D401
            raise RuntimeError(
                "KernelSpectralQuantEngine (spectralquant.engine.SpectralQuantEngine) "
                "is only available on the Modal image where turboquant_cutile is "
                f"installed.  Original import error: {self._import_error!r}"
            )

# Backwards-compatibility alias used by internal modules during the v1→v2
# transition.  Prefer the unqualified ``SpectralQuantEngine`` (canonical)
# or ``KernelSpectralQuantEngine`` (Modal/cuTile).
_LegacySpectralQuantEngine = SpectralQuantEngine
from spectralquant.metrics import (
    cosine_similarity,
    attention_output_cosine_sim,
    max_absolute_weight_error,
    weighted_mse,
    compression_ratio,
    inner_product_error,
)
from spectralquant.utils import (
    set_seed,
    get_model_config,
    load_calibration_data,
    save_results,
    load_results,
    Timer,
)

__version__ = "0.1.0"
__author__ = "SpectralQuant Authors"

__all__ = [
    # Calibration
    "EigenspectralCalibrator",
    "HeadCalibrationData",
    # Rotation
    "BaseRotation",
    "SpectralRotation",
    "RandomRotation",
    # Quantization
    "BitAllocator",
    "LloydMaxQuantizer",
    "NonUniformQuantizer",
    "CompressedVector",
    "WaterfillAllocation",
    # QJL
    "SelectiveQJL",
    "FullQJL",
    # Engine
    "SpectralQuantEngine",          # canonical pure-Python engine
    "KernelSpectralQuantEngine",    # Modal/cuTile-accelerated variant
    "TurboQuantBaseline",
    "EngineConfig",
    # Metrics
    "cosine_similarity",
    "attention_output_cosine_sim",
    "max_absolute_weight_error",
    "weighted_mse",
    "compression_ratio",
    "inner_product_error",
    # Utils
    "set_seed",
    "get_model_config",
    "load_calibration_data",
    "save_results",
    "load_results",
    "Timer",
]
