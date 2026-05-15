"""Tests for ``experiments/model_adapters.py``.

The adapter is exercised against fake Mistral / Qwen-like module
hierarchies so we don't need the real transformers package or any model
weights. The tests pin:

* family detection from model_type and from repo-id heuristics;
* layer + projection discovery on llama-like layouts;
* dim resolution from config + attention module attrs;
* errors that should be raised on packed-QKV / unknown layouts.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _load_adapter_module():
    spec = importlib.util.spec_from_file_location(
        "experiments.model_adapters",
        EXPERIMENTS_DIR / "model_adapters.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register both as a submodule (so 'from experiments import model_adapters'
    # works inside run_three_way) and standalone for direct tests.
    pkg = types.ModuleType("experiments")
    pkg.__path__ = [str(EXPERIMENTS_DIR)]  # make 'experiments' a namespace pkg
    sys.modules.setdefault("experiments", pkg)
    sys.modules["experiments.model_adapters"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def adapters():
    return _load_adapter_module()


# --- Fakes -----------------------------------------------------------------


class _FakeLinear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features


class _FakeAttn:
    def __init__(
        self,
        hidden_size: int = 4096,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = n_heads
        self.num_key_value_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = _FakeLinear(hidden_size, n_heads * head_dim)
        self.k_proj = _FakeLinear(hidden_size, n_kv_heads * head_dim)
        self.v_proj = _FakeLinear(hidden_size, n_kv_heads * head_dim)


class _FakeLayer:
    def __init__(self, attn: _FakeAttn) -> None:
        self.self_attn = attn


class _FakeInner:
    def __init__(self, layers):
        self.layers = layers


class _FakeConfig:
    def __init__(
        self,
        model_type: str = "mistral",
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        hidden_size: int = 4096,
        name_or_path: str = "mistralai/Mistral-7B-v0.3",
        num_hidden_layers: int = 4,
    ) -> None:
        self.model_type = model_type
        self.num_attention_heads = n_heads
        self.num_key_value_heads = n_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self._name_or_path = name_or_path
        self.num_hidden_layers = num_hidden_layers


class _FakeModel:
    def __init__(
        self,
        n_layers: int = 4,
        attn_kwargs=None,
        config: _FakeConfig | None = None,
    ) -> None:
        attn_kwargs = attn_kwargs or {}
        layers = [_FakeLayer(_FakeAttn(**attn_kwargs)) for _ in range(n_layers)]
        self.model = _FakeInner(layers)
        self.config = config or _FakeConfig(num_hidden_layers=n_layers)


# --- Tests: family detection ----------------------------------------------


def test_detect_family_from_model_type_mistral(adapters):
    m = _FakeModel(config=_FakeConfig(model_type="mistral"))
    assert adapters.detect_family(m) == "llama_like"


def test_detect_family_from_model_type_qwen2(adapters):
    m = _FakeModel(
        config=_FakeConfig(model_type="qwen2", name_or_path="Qwen/Qwen2.5-7B"),
    )
    assert adapters.detect_family(m) == "llama_like"


def test_detect_family_from_repo_id_string(adapters):
    assert adapters.detect_family("mistralai/Mistral-7B-v0.3") == "llama_like"
    assert adapters.detect_family("Qwen/Qwen2.5-7B") == "llama_like"
    assert adapters.detect_family("meta-llama/Llama-3-8B") == "llama_like"


def test_detect_family_unknown_raises(adapters):
    with pytest.raises(adapters.UnsupportedArchitectureError):
        adapters.detect_family("openai/some-fictitious-model")


def test_detect_family_unknown_model_type_raises(adapters):
    m = _FakeModel(config=_FakeConfig(model_type="gpt_neox", name_or_path=""))
    with pytest.raises(adapters.UnsupportedArchitectureError):
        adapters.detect_family(m)


# --- Tests: layer + projection discovery -----------------------------------


def test_find_attention_modules_mistral(adapters):
    m = _FakeModel(n_layers=3)
    layers = adapters.find_attention_modules(m)
    assert [layer.layer_idx for layer in layers] == [0, 1, 2]
    for layer in layers:
        assert layer.q_proj is layer.attn_module.q_proj
        assert layer.k_proj is layer.attn_module.k_proj
        assert layer.v_proj is layer.attn_module.v_proj


def test_find_attention_modules_qwen(adapters):
    cfg = _FakeConfig(
        model_type="qwen2",
        n_heads=28, n_kv_heads=4, head_dim=128,
        hidden_size=28 * 128,
        name_or_path="Qwen/Qwen2.5-7B",
        num_hidden_layers=2,
    )
    m = _FakeModel(
        n_layers=2,
        attn_kwargs=dict(hidden_size=28 * 128, n_heads=28,
                         n_kv_heads=4, head_dim=128),
        config=cfg,
    )
    layers = adapters.find_attention_modules(m)
    assert [layer.layer_idx for layer in layers] == [0, 1]


def test_find_attention_modules_no_inner_raises(adapters):
    class NoInner:
        config = _FakeConfig()
    with pytest.raises(adapters.UnsupportedArchitectureError):
        adapters.find_attention_modules(NoInner())


def test_get_qkv_projections_packed_layout_raises(adapters):
    class PackedAttn:
        qkv_proj = _FakeLinear(4096, 4096 * 3)

    with pytest.raises(adapters.UnsupportedArchitectureError, match="Packed-QKV"):
        adapters.get_qkv_projections(PackedAttn())


def test_get_qkv_projections_missing_raises(adapters):
    class Empty:
        pass
    with pytest.raises(adapters.UnsupportedArchitectureError):
        adapters.get_qkv_projections(Empty())


# --- Tests: dim resolution -------------------------------------------------


def test_get_kv_dims_from_attn_module(adapters):
    m = _FakeModel()
    attn = m.model.layers[0].self_attn
    n_q, n_kv, hd = adapters.get_kv_dims(m.config, attn)
    assert (n_q, n_kv, hd) == (32, 8, 128)


def test_get_kv_dims_from_config_only(adapters):
    cfg = _FakeConfig()

    class Bare:
        pass

    n_q, n_kv, hd = adapters.get_kv_dims(cfg, Bare())
    assert (n_q, n_kv, hd) == (32, 8, 128)


def test_get_kv_dims_computed_from_hidden(adapters):
    cfg = _FakeConfig()
    cfg.head_dim = None  # type: ignore[assignment]
    delattr(cfg, "head_dim")

    class Bare:
        pass

    n_q, n_kv, hd = adapters.get_kv_dims(cfg, Bare())
    assert hd == cfg.hidden_size // cfg.num_attention_heads


def test_get_kv_dims_missing_raises(adapters):
    class EmptyCfg:
        pass

    with pytest.raises(adapters.UnsupportedArchitectureError):
        adapters.get_kv_dims(EmptyCfg(), None)


# --- Tests: default_layer_sample ------------------------------------------


def test_default_layer_sample_known(adapters):
    assert adapters.default_layer_sample(
        "mistralai/Mistral-7B-v0.3", 4, total_layers=32,
    ) == [0, 4, 8, 12]
    assert adapters.default_layer_sample(
        "Qwen/Qwen2.5-7B", 3, total_layers=28,
    ) == [0, 3, 6]


def test_default_layer_sample_generic(adapters):
    out = adapters.default_layer_sample("some/random-7B", 4, total_layers=32)
    assert len(out) == 4
    assert all(0 <= idx < 32 for idx in out)
    assert sorted(out) == out


# --- Smoke: end-to-end against a torch.nn-based fake -----------------------


def test_real_torch_module_layout(adapters):
    """Build a torch.nn-based hierarchy mirroring HF Mistral/Qwen and run
    discovery on it. Catches ``isinstance`` regressions that wouldn't show
    up against pure-Python fakes."""
    class RealAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 64
            self.num_heads = 8
            self.num_key_value_heads = 2
            self.head_dim = 8
            self.q_proj = nn.Linear(64, 8 * 8)
            self.k_proj = nn.Linear(64, 2 * 8)
            self.v_proj = nn.Linear(64, 2 * 8)

    class RealLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = RealAttn()

    class RealInner(nn.Module):
        def __init__(self, n_layers):
            super().__init__()
            self.layers = nn.ModuleList([RealLayer() for _ in range(n_layers)])

    class RealModel(nn.Module):
        def __init__(self, n_layers):
            super().__init__()
            self.model = RealInner(n_layers)
            self.config = _FakeConfig(
                model_type="llama",
                n_heads=8, n_kv_heads=2, head_dim=8, hidden_size=64,
                num_hidden_layers=n_layers,
            )

    m = RealModel(3)
    found = adapters.find_attention_modules(m)
    assert len(found) == 3
    assert all(hasattr(layer.q_proj, "weight") for layer in found)
