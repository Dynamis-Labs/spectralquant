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
)
from spectralquant.selective_qjl import SelectiveQJL, FullQJL
from spectralquant.spectralquant import SpectralQuantEngine as _LegacySpectralQuantEngine, TurboQuantBaseline, EngineConfig
from spectralquant.engine import SpectralQuantEngine  # real TurboQuant integration (subclasses TurboQuantEngine)
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
    # QJL
    "SelectiveQJL",
    "FullQJL",
    # Engine
    "SpectralQuantEngine",
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
