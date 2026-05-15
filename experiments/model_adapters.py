"""Model-family adapters for the v2 three-way benchmark.

The HuggingFace LLM ecosystem packs query/key/value into one of several
shapes per model family. Mistral, Llama, and Qwen2 each expose
``model.layers[i].self_attn.{q_proj,k_proj,v_proj}`` as separate
``nn.Linear`` modules, so a uniform adapter works for all three. Other
families (e.g. GPT-NeoX, Falcon) pack QKV into a single matrix and would
need a different adapter; we surface a clear error in that case.

The adapters are deliberately pure-Python and have no hard dependency on
``transformers`` — they accept any model object that exposes the expected
attribute path. This makes the adapter testable with fake ``nn.Module``
hierarchies, and lets ``run_three_way.py --dry-run`` import the adapter
without requiring transformers to be installed.

Discovery contract (per spec §13.4):

* ``find_attention_modules(model) -> list[(layer_idx, attn_module)]``
* ``get_qkv_projections(attn_module) -> (q_proj, k_proj, v_proj)``
* ``get_kv_dims(model_config, attn_module) -> (n_q, n_kv, head_dim)``

Each function raises ``UnsupportedArchitectureError`` with an actionable
message if the model does not match a known layout. Catching this error
and falling through to a slower, generic discovery is the caller's choice
— but the harness elects to fail loudly so the operator notices early.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UnsupportedArchitectureError(RuntimeError):
    """Raised when an HF model does not match a known v2 adapter layout."""


# --- Family detection ------------------------------------------------------

#: Known model_type strings -> family bucket. The bucket controls which
#: adapter layout we look for. Mistral and Llama share the layout; Qwen2 is
#: identical at the attention level; future families add new buckets here.
_FAMILY_BY_MODEL_TYPE = {
    "mistral": "llama_like",
    "llama":   "llama_like",
    "qwen2":   "llama_like",
    "qwen2_5": "llama_like",
    "qwen":    "llama_like",
}

#: Substrings in a HF repo id that imply a family when ``model_type`` is
#: missing (e.g. some custom configs strip it). Used as a fallback only.
_FAMILY_BY_REPO_HINT = (
    ("mistral",   "llama_like"),
    ("llama",     "llama_like"),
    ("qwen2.5",   "llama_like"),
    ("qwen2",     "llama_like"),
    ("qwen",      "llama_like"),
)


def detect_family(model_or_id: Any) -> str:
    """Return a family bucket for ``model_or_id``.

    ``model_or_id`` may be a model instance (with ``.config.model_type``)
    or a string repo id. Raises :class:`UnsupportedArchitectureError` if
    no rule matches.
    """
    if isinstance(model_or_id, str):
        rid = model_or_id.lower()
        for hint, family in _FAMILY_BY_REPO_HINT:
            if hint in rid:
                return family
        raise UnsupportedArchitectureError(
            f"No v2 adapter for model id '{model_or_id}'. "
            f"Supported families: {sorted(set(_FAMILY_BY_MODEL_TYPE.values()))}"
        )

    cfg = getattr(model_or_id, "config", None)
    mt = (getattr(cfg, "model_type", "") or "").lower()
    if mt in _FAMILY_BY_MODEL_TYPE:
        return _FAMILY_BY_MODEL_TYPE[mt]
    # Fall back to repo-id heuristic if config is missing.
    name_or_path = getattr(cfg, "_name_or_path", "") or ""
    rid = name_or_path.lower()
    for hint, family in _FAMILY_BY_REPO_HINT:
        if hint in rid:
            return family
    raise UnsupportedArchitectureError(
        f"No v2 adapter for model_type='{mt}' "
        f"(name_or_path='{name_or_path}'). "
        f"Add a family rule in experiments/model_adapters.py."
    )


# --- Layer / projection discovery ------------------------------------------


@dataclass
class AttentionLayer:
    """A discovered attention layer plus its projection modules."""
    layer_idx: int
    attn_module: Any
    q_proj: Any
    k_proj: Any
    v_proj: Any


def _llama_like_layers(model: Any) -> List[Tuple[int, Any]]:
    """Yield ``(layer_idx, attn_module)`` from a Mistral/Llama/Qwen2 model.

    Layout: ``model.model.layers[i].self_attn`` for every i.
    Raises if ``model.model.layers`` is missing.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise UnsupportedArchitectureError(
            "Expected llama-like model.model.layers, but model has no "
            "attribute 'model'."
        )
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise UnsupportedArchitectureError(
            "Expected llama-like model.model.layers, but model.model has "
            "no attribute 'layers'."
        )
    out: List[Tuple[int, Any]] = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise UnsupportedArchitectureError(
                f"Layer {i} has no .self_attn — adapter mismatch."
            )
        out.append((i, attn))
    if not out:
        raise UnsupportedArchitectureError(
            "model.model.layers iterated to zero entries."
        )
    return out


def _llama_like_qkv(attn_module: Any) -> Tuple[Any, Any, Any]:
    """Return (q_proj, k_proj, v_proj) from a llama-like ``self_attn``.

    Packed-QKV layouts (a single ``in_proj`` linear) are NOT supported
    here — they appear in GPT-NeoX/Falcon and need a separate adapter.
    """
    q = getattr(attn_module, "q_proj", None)
    k = getattr(attn_module, "k_proj", None)
    v = getattr(attn_module, "v_proj", None)
    if q is None or k is None or v is None:
        # Detect packed-QKV so the error is actionable.
        if any(getattr(attn_module, n, None) is not None
               for n in ("qkv_proj", "in_proj", "Wqkv", "c_attn")):
            raise UnsupportedArchitectureError(
                "Packed-QKV attention detected (one of {qkv_proj, in_proj, "
                "Wqkv, c_attn}). The v2 adapter currently only supports "
                "split q_proj/k_proj/v_proj layouts (Mistral, Llama, Qwen2)."
            )
        raise UnsupportedArchitectureError(
            "Attention module does not expose q_proj/k_proj/v_proj. "
            f"Available attrs: {sorted(a for a in dir(attn_module) if not a.startswith('_'))[:20]}"
        )
    return q, k, v


def find_attention_modules(model: Any) -> List[AttentionLayer]:
    """Return the full list of discovered attention layers + projections.

    Raises :class:`UnsupportedArchitectureError` if any step of the
    discovery fails.
    """
    family = detect_family(model)
    if family == "llama_like":
        out: List[AttentionLayer] = []
        for layer_idx, attn in _llama_like_layers(model):
            q, k, v = _llama_like_qkv(attn)
            out.append(AttentionLayer(layer_idx, attn, q, k, v))
        return out
    raise UnsupportedArchitectureError(
        f"Internal: family bucket '{family}' has no discovery rule."
    )


def get_qkv_projections(attn_module: Any) -> Tuple[Any, Any, Any]:
    """Convenience: return (q_proj, k_proj, v_proj) for a single attn module.

    Tries the llama-like layout (the only one currently supported).
    """
    return _llama_like_qkv(attn_module)


# --- Dim discovery ---------------------------------------------------------


def get_kv_dims(
    model_config: Any,
    attn_module: Optional[Any] = None,
) -> Tuple[int, int, int]:
    """Return (n_q_heads, n_kv_heads, head_dim).

    Resolution order:
      1. Explicit attention-module attrs (``num_heads``, ``num_key_value_heads``,
         ``head_dim``).
      2. ``model_config`` attrs of the same names.
      3. Computed: ``head_dim = hidden_size // num_attention_heads``.

    Raises :class:`UnsupportedArchitectureError` if neither path yields all
    three numbers.
    """
    def _from(obj: Any, name: str) -> Optional[int]:
        if obj is None:
            return None
        v = getattr(obj, name, None)
        return int(v) if isinstance(v, int) and not isinstance(v, bool) else None

    n_q = (
        _from(attn_module, "num_heads")
        or _from(model_config, "num_attention_heads")
    )
    n_kv = (
        _from(attn_module, "num_key_value_heads")
        or _from(model_config, "num_key_value_heads")
        or n_q
    )
    head_dim = (
        _from(attn_module, "head_dim")
        or _from(model_config, "head_dim")
    )
    if head_dim is None:
        hidden = (
            _from(attn_module, "hidden_size")
            or _from(model_config, "hidden_size")
        )
        if hidden is not None and n_q:
            head_dim = hidden // n_q

    if not (n_q and n_kv and head_dim):
        raise UnsupportedArchitectureError(
            "Could not derive (n_q_heads, n_kv_heads, head_dim) from "
            f"config={model_config!r} attn_module={attn_module!r}. "
            "Ensure num_attention_heads / num_key_value_heads / head_dim "
            "are set."
        )
    return int(n_q), int(n_kv), int(head_dim)


# --- Default layer sample lists (also exposed in run_three_way) ------------

#: Spec §13.4 sampling. Exposed here so tests can compare against the
#: harness-side copy without circular import.
SPEC_LAYER_SAMPLES = {
    "mistralai/Mistral-7B-v0.3": [0, 4, 8, 12, 16, 20, 24, 28],
    "Qwen/Qwen2.5-7B":           [0, 3, 6, 9, 12, 15, 18, 21],
}


def default_layer_sample(model_id: str, n: int, total_layers: int) -> List[int]:
    """Return ``n`` layer indices for a model id.

    If the model is in :data:`SPEC_LAYER_SAMPLES`, take its first ``n``
    entries. Otherwise spread ``n`` indices roughly evenly across
    ``total_layers``.
    """
    if model_id in SPEC_LAYER_SAMPLES:
        return list(SPEC_LAYER_SAMPLES[model_id][:n])
    if n <= 0:
        return []
    if total_layers <= 0:
        return list(range(n))
    n = min(n, total_layers)
    step = max(1, total_layers // n)
    return [min(total_layers - 1, i * step) for i in range(n)]
