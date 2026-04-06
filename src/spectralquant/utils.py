"""
Utility functions and classes for SpectralQuant.

Includes:
- ``set_seed``: Reproducibility across torch, numpy, and Python random.
- ``get_model_config``: Extract architecture metadata from a HuggingFace model.
- ``load_calibration_data``: Load and tokenize a text dataset.
- ``save_results`` / ``load_results``: JSON-based result persistence.
- ``Timer``: Context manager for benchmarking code blocks.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all relevant libraries.

    Sets seeds for Python's :mod:`random`, :mod:`numpy` (if available),
    and :mod:`torch` (both CPU and all CUDA devices).

    Parameters
    ----------
    seed:
        Integer seed value.

    Examples
    --------
    >>> set_seed(42)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.debug("Random seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Model configuration extraction
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture metadata for a transformer model.

    Attributes
    ----------
    n_layers:
        Number of transformer layers.
    n_heads:
        Number of query attention heads per layer.
    head_dim:
        Dimension of each attention head vector.
    n_kv_heads:
        Number of key/value heads per layer (may differ from n_heads in GQA).
    hidden_size:
        Total hidden dimension (= n_heads * head_dim).
    model_type:
        Architecture family string (e.g. ``"qwen2"``, ``"llama"``).
    vocab_size:
        Vocabulary size.
    max_position_embeddings:
        Maximum sequence length.
    """

    n_layers: int
    n_heads: int
    head_dim: int
    n_kv_heads: int
    hidden_size: int
    model_type: str
    vocab_size: int
    max_position_embeddings: int


def get_model_config(model: nn.Module) -> Dict[str, Any]:
    """Extract architecture metadata from a pretrained HuggingFace model.

    Parameters
    ----------
    model:
        A pretrained transformer model with a ``.config`` attribute.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: ``n_layers``, ``n_heads``, ``head_dim``,
        ``n_kv_heads``, ``hidden_size``, ``model_type``, ``vocab_size``,
        ``max_position_embeddings``.

    Raises
    ------
    AttributeError
        If the model does not have a ``.config`` attribute.

    Examples
    --------
    >>> cfg = get_model_config(model)
    >>> print(cfg["head_dim"])
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise AttributeError(
            "Model does not have a .config attribute.  "
            "Ensure you are using a HuggingFace PreTrainedModel."
        )

    n_heads = getattr(cfg, "num_attention_heads", 1)
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    hidden_size = getattr(cfg, "hidden_size", n_heads * 64)
    head_dim = hidden_size // n_heads

    # Some models define head_dim explicitly
    head_dim = getattr(cfg, "head_dim", head_dim)

    return {
        "n_layers": getattr(cfg, "num_hidden_layers", 1),
        "n_heads": int(n_heads),
        "head_dim": int(head_dim),
        "n_kv_heads": int(n_kv_heads),
        "hidden_size": int(hidden_size),
        "model_type": getattr(cfg, "model_type", "unknown"),
        "vocab_size": getattr(cfg, "vocab_size", 0),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", 4096),
    }


# ---------------------------------------------------------------------------
# Calibration data loading
# ---------------------------------------------------------------------------

def load_calibration_data(
    dataset_name: str,
    n_samples: int,
    tokenizer: Any,
    max_length: int = 2048,
    dataset_split: str = "train",
    text_column: str = "text",
    seed: int = 42,
) -> List[str]:
    """Load and (lightly) preprocess text samples for calibration.

    Uses HuggingFace ``datasets`` to load the specified dataset, shuffles
    with ``seed``, and returns up to ``n_samples`` non-empty text strings.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset identifier (e.g. ``"wikitext"``,
        ``"wikitext-2-raw-v1"``).  For datasets with a config name, pass
        ``"dataset_name/config"`` — the function splits on ``"/"`` and uses
        the second part as the config.
    n_samples:
        Maximum number of text samples to return.
    tokenizer:
        HuggingFace tokenizer (used only for future pre-filtering if needed;
        the raw text strings are returned).
    max_length:
        Maximum token length (used for informational logging only here;
        truncation happens during calibration forward passes).
    dataset_split:
        Dataset split to load (default ``"train"``).
    text_column:
        Name of the text column in the dataset.
    seed:
        Shuffle seed.

    Returns
    -------
    List[str]
        List of up to ``n_samples`` non-empty text strings.

    Raises
    ------
    ImportError
        If ``datasets`` is not installed.

    Examples
    --------
    >>> texts = load_calibration_data("wikitext", 512, tokenizer)
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for load_calibration_data. "
            "Install it via `pip install datasets`."
        ) from exc

    logger.info(
        "Loading calibration dataset '%s' (split='%s', n_samples=%d).",
        dataset_name,
        dataset_split,
        n_samples,
    )

    # Handle "name/config" format
    parts = dataset_name.split("/", 1)
    if len(parts) == 2:
        ds = load_dataset(parts[0], parts[1], split=dataset_split)
    else:
        ds = load_dataset(dataset_name, split=dataset_split)

    # Shuffle and select
    ds = ds.shuffle(seed=seed)
    texts: List[str] = []
    for example in ds:
        text = example.get(text_column, "")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        if len(texts) >= n_samples:
            break

    logger.info("Loaded %d calibration samples.", len(texts))
    return texts


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(results_dict: Dict[str, Any], path: str) -> None:
    """Save a results dictionary to a JSON file with metadata.

    Automatically adds a ``"_metadata"`` key with timestamp and SpectralQuant
    version.  Non-JSON-serialisable values (e.g. tensors) are converted to
    lists or scalars.

    Parameters
    ----------
    results_dict:
        Dictionary of results to save.
    path:
        Output file path (should end in ``.json``).

    Examples
    --------
    >>> save_results({"mse": 0.003, "ratio": 8.0}, "results/experiment1.json")
    """
    import spectralquant

    output = dict(results_dict)
    output["_metadata"] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spectralquant_version": getattr(spectralquant, "__version__", "unknown"),
    }

    # Convert tensors and numpy arrays to plain Python types
    def _convert(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except ImportError:
            pass
        return obj

    output = _convert(output)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved results to %s.", path)


def load_results(path: str) -> Dict[str, Any]:
    """Load a results dictionary previously saved by :func:`save_results`.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    Dict[str, Any]
        Loaded results dictionary.

    Examples
    --------
    >>> results = load_results("results/experiment1.json")
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded results from %s.", path)
    return data


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for timing code blocks.

    Attributes
    ----------
    elapsed:
        Elapsed wall-clock time in seconds (set after ``__exit__``).
    label:
        Human-readable label shown in log messages.

    Examples
    --------
    >>> with Timer("forward pass") as t:
    ...     output = model(input_ids)
    >>> print(f"Elapsed: {t.elapsed:.3f}s")
    """

    def __init__(self, label: str = "block", log_level: int = logging.DEBUG) -> None:
        self.label = label
        self.log_level = log_level
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        logger.log(self.log_level, "Timer [%s] started.", self.label)
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger.log(
            self.log_level,
            "Timer [%s] finished: %.4f s.",
            self.label,
            self.elapsed,
        )

    def __repr__(self) -> str:
        return f"Timer(label={self.label!r}, elapsed={self.elapsed:.4f}s)"
