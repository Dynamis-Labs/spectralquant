import importlib
import inspect
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


log = logging.getLogger(__name__)


COVERAGE_PRESETS: Dict[str, Dict[str, Any]] = {
    "small": {
        "n_calib": 16,
        "n_eval": 8,
        "seq_len": 512,
        "layer_mode": "sampled",
        "head_mode": "sampled",
        "max_cached_vectors": 512,
    },
    "medium": {
        "n_calib": 64,
        "n_eval": 32,
        "seq_len": 1024,
        "layer_mode": "sampled",
        "head_mode": "sampled",
        "max_cached_vectors": 1024,
    },
    "large": {
        "n_calib": 128,
        "n_eval": 64,
        "seq_len": 2048,
        "layer_mode": "all",
        "head_mode": "all",
        "max_cached_vectors": 4096,
    },
}


def add_coverage_args(parser, *, default_preset: str = "medium") -> None:
    parser.add_argument(
        "--coverage-preset",
        choices=sorted(COVERAGE_PRESETS),
        default=default_preset,
        help="Coverage preset controlling sample count, seq length, and layer/head breadth.",
    )
    parser.add_argument("--n-calib", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-cached-vectors", type=int, default=None)
    parser.add_argument(
        "--layer-mode",
        choices=["sampled", "all"],
        default=None,
        help="Use 3 representative layers or all layers.",
    )
    parser.add_argument(
        "--head-mode",
        choices=["sampled", "all"],
        default=None,
        help="Use up to 4 KV heads or all KV heads.",
    )


def resolve_coverage_args(args) -> Dict[str, Any]:
    preset = COVERAGE_PRESETS[args.coverage_preset].copy()
    for key in ("n_calib", "n_eval", "seq_len", "max_cached_vectors", "layer_mode", "head_mode"):
        value = getattr(args, key, None)
        if value is not None:
            preset[key] = value
    preset["coverage_preset"] = args.coverage_preset
    return preset


def select_layer_indices(n_layers: int, mode: str) -> List[int]:
    if mode == "all":
        return list(range(n_layers))
    return sorted(set([n_layers // 4, n_layers // 2, 3 * n_layers // 4]))


def select_head_indices(n_kv: int, mode: str) -> List[int]:
    if mode == "all":
        return list(range(n_kv))
    return list(range(min(n_kv, 4)))


def causal_attention_output(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, head_dim: int) -> torch.Tensor:
    scale = 1.0 / math.sqrt(head_dim)
    scores = Q.float() @ K.float().T * scale
    q_len, k_len = scores.shape
    if q_len == k_len:
        mask = torch.triu(
            torch.full((q_len, k_len), float("-inf"), device=scores.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + mask
    return F.softmax(scores, dim=-1) @ V.float()


def noncausal_attention_output(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, head_dim: int) -> torch.Tensor:
    scale = 1.0 / math.sqrt(head_dim)
    scores = Q.float() @ K.float().T * scale
    return F.softmax(scores, dim=-1) @ V.float()


def _resolve_transformer_layers(model) -> Sequence[Any]:
    candidates = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("layers",),
    ]
    for path in candidates:
        cur = model
        try:
            for part in path:
                cur = getattr(cur, part)
            if len(cur):
                return cur
        except Exception:
            continue
    raise AttributeError("Could not locate transformer layers on model.")


def _resolve_attention_module(layer) -> Any:
    for attr in ("self_attn", "attention", "attn"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise AttributeError(f"Could not locate attention module on layer type {type(layer)!r}.")


def _shape_proj(proj: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    batch, seq_len, _ = proj.shape
    return proj.view(batch, seq_len, n_heads, head_dim).transpose(1, 2).contiguous()


def _capture_hidden_states(model, layer_indices: Iterable[int]):
    layers = _resolve_transformer_layers(model)
    captured: Dict[int, Dict[str, Any]] = {}
    handles = []

    for layer_idx in layer_indices:
        attn = _resolve_attention_module(layers[layer_idx])

        def _hook(module, args, kwargs, *, _layer_idx=layer_idx):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if hidden_states is None:
                return
            captured[_layer_idx] = {
                "hidden_states": hidden_states.detach(),
                "args": args,
                "kwargs": dict(kwargs),
            }

        handles.append(attn.register_forward_pre_hook(_hook, with_kwargs=True))

    return captured, handles


def _get_rotary_inputs(attn_module, hidden_states, n_kv, head_dim, capture):
    if "position_embeddings" in capture["kwargs"]:
        return capture["kwargs"]["position_embeddings"]

    position_ids = capture["kwargs"].get("position_ids")
    if not hasattr(attn_module, "rotary_emb"):
        raise ValueError("Attention module does not expose rotary embeddings.")

    value_states = _shape_proj(attn_module.v_proj(hidden_states), n_kv, head_dim)
    rotary = attn_module.rotary_emb
    try:
        return rotary(value_states, position_ids)
    except TypeError:
        return rotary(value_states, position_ids=position_ids)


def _apply_rotary(attn_module, query_states, key_states, rotary_inputs):
    if not isinstance(rotary_inputs, tuple) or len(rotary_inputs) != 2:
        raise ValueError("Expected rotary inputs to be a (cos, sin) tuple.")

    mod = importlib.import_module(attn_module.__class__.__module__)
    apply_fn = getattr(mod, "apply_rotary_pos_emb", None)
    if apply_fn is None:
        raise AttributeError(f"{mod.__name__} does not expose apply_rotary_pos_emb.")

    cos, sin = rotary_inputs
    try:
        return apply_fn(query_states, key_states, cos, sin, unsqueeze_dim=1)
    except TypeError:
        return apply_fn(query_states, key_states, cos, sin)


def compute_true_queries_for_layer(model, layer_idx: int, capture: Dict[str, Any], head_dim: int) -> torch.Tensor:
    layers = _resolve_transformer_layers(model)
    attn = _resolve_attention_module(layers[layer_idx])
    hidden_states = capture["hidden_states"].to(attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)

    cfg = model.config
    n_heads = int(getattr(cfg, "num_attention_heads"))
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))

    query_states = _shape_proj(attn.q_proj(hidden_states), n_heads, head_dim)
    key_states = _shape_proj(attn.k_proj(hidden_states), n_kv, head_dim)

    try:
        rotary_inputs = _get_rotary_inputs(attn, hidden_states, n_kv, head_dim, capture)
        query_states, _ = _apply_rotary(attn, query_states, key_states, rotary_inputs)
    except Exception as exc:
        log.warning("Layer %d query capture fell back to pre-RoPE projections: %s", layer_idx, exc)

    return query_states.detach().float().cpu()


def map_kv_head_to_query_head(kv_head_idx: int, n_heads: int, n_kv: int) -> int:
    if n_heads <= n_kv:
        return min(kv_head_idx, n_heads - 1)
    q_per_kv = max(n_heads // n_kv, 1)
    return min(kv_head_idx * q_per_kv, n_heads - 1)


def collect_eval_qkv(
    model,
    tokenizer,
    device: str,
    *,
    n_eval: int,
    seq_len: int,
    head_dim: int,
    layer_indices: Sequence[int],
    head_indices: Sequence[int],
    extract_kv_layer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    split_start: int = 0,
) -> Dict[str, Any]:
    from datasets import load_dataset

    cfg = model.config
    n_heads = int(getattr(cfg, "num_attention_heads"))
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    dataset_kwargs = {
        "path": dataset_name,
        "name": dataset_config,
        "split": f"{split}[{split_start}:{split_start + n_eval * 5}]",
    }
    try:
        ds = load_dataset(**dataset_kwargs)
    except Exception as exc:
        log.warning("Dataset load failed, retrying from local cache only: %s", exc)
        ds = load_dataset(**dataset_kwargs, download_mode="reuse_cache_if_exists")
    eval_queries = {(l, h): [] for l in layer_indices for h in head_indices}
    eval_keys = {(l, h): [] for l in layer_indices for h in head_indices}
    eval_vals = {(l, h): [] for l in layer_indices for h in head_indices}

    captured, handles = _capture_hidden_states(model, layer_indices)
    n_sequences = 0
    try:
        for item in ds:
            text = item.get("text", "")
            if len(text.strip()) < 100:
                continue

            enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
            if enc["input_ids"].shape[1] < 32:
                continue

            captured.clear()
            with torch.no_grad():
                out = model(**enc, use_cache=True)
            kv = out.past_key_values

            layer_queries: Dict[int, torch.Tensor] = {}
            for layer_idx in layer_indices:
                capture = captured.get(layer_idx)
                if capture is None:
                    continue
                layer_queries[layer_idx] = compute_true_queries_for_layer(model, layer_idx, capture, head_dim)

            for layer_idx in layer_indices:
                try:
                    k_l, v_l = extract_kv_layer(kv, layer_idx)
                    k_l = k_l.float().cpu()
                    v_l = v_l.float().cpu()
                except Exception:
                    continue

                q_l = layer_queries.get(layer_idx)
                if q_l is None:
                    continue

                for kv_head_idx in head_indices:
                    if kv_head_idx >= k_l.shape[1]:
                        continue
                    query_head_idx = map_kv_head_to_query_head(kv_head_idx, n_heads=n_heads, n_kv=n_kv)
                    if query_head_idx >= q_l.shape[1]:
                        continue
                    eval_queries[(layer_idx, kv_head_idx)].append(q_l[0, query_head_idx])
                    eval_keys[(layer_idx, kv_head_idx)].append(k_l[0, kv_head_idx])
                    eval_vals[(layer_idx, kv_head_idx)].append(v_l[0, kv_head_idx])

            n_sequences += 1
            if n_sequences >= n_eval:
                break
    finally:
        for handle in handles:
            handle.remove()

    return {
        "queries": eval_queries,
        "keys": eval_keys,
        "values": eval_vals,
        "n_sequences": n_sequences,
        "query_mode": "true_model_queries",
        "query_head_mapping": "first_query_head_per_kv_group",
        "attention_mode": "causal_self_attention",
        "seq_len": seq_len,
    }


def build_benchmark_note(key_bits: Any) -> str:
    return (
        f"{key_bits}-bit keys here means per-dimension scalar quantization in the rotated spectral basis, "
        "plus a separately stored vector norm. It is not a raw sign-only key in the original basis."
    )
