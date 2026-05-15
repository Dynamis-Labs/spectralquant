"""SpectralQuant v2 K/V projection-replay harness.

This module provides the smallest honest "compressed-method" evaluation
path that does not require a full HuggingFace cache replacement. It
attaches forward hooks to every transformer layer's ``k_proj`` and
``v_proj`` linear modules; each hook reshapes the projection output to
``(batch, n_kv_heads, seq, head_dim)``, runs the calibrated SpectralQuant
v2 engine's ``compress`` → ``decompress`` round-trip on it, and returns
the reshaped reconstruction in the original ``(batch, seq, n_kv_heads *
head_dim)`` layout. The HF model then sees the *compressed* K/V values
flowing through the rest of attention (RoPE, softmax, matmul) naturally.

Scope
-----

* **Real method**: the K/V values that downstream attention consumes are
  the engine's actual decompress output. Any quantization error in the
  v2 engine flows directly into the model's logits / generation /
  perplexity. There is no placeholder.
* **Pre-RoPE**: hooks fire at the linear-projection output, before
  rotary position embedding. This matches the v2 spec's
  ``key_space=pre_rope`` calibration assumption, so the rotation
  matrices V^T computed during calibration remain consistent with the
  data the engine sees at eval time.
* **Per-layer calibration**: ``EigenspectralCalibrator.calibrate`` is
  run once on the eval corpus (or a held-out subset) and produces
  per-(layer, head, type) eigenstatistics for *every* layer. The engine
  is fitted on the rotated K/V via the same captured tensors. If a
  layer is missing calibration data, the hook falls through unchanged
  (passes through FP16) — and that fact is reported in the per-layer
  coverage summary so a downstream artifact can flag partial coverage.
* **Honest about what we measure**: the result of running PPL / eval
  with these hooks attached is the perplexity *of the K/V-quantized
  model*. We deliberately do not claim it is end-to-end "TurboQuant
  vs. SpectralQuant" since this harness only exercises the K/V cache
  reconstruction path; weight quantization, attention-kernel
  optimization, etc. are out of scope and are not claimed.

Public API
----------

* :func:`build_calibrated_engine` — load model+tokenizer (caller
  provides), run calibration over a corpus, fit quantizers, return the
  ``(engine, coverage)`` pair.
* :func:`attach_replay_hooks` — register hooks on every layer's
  ``k_proj`` / ``v_proj``; returns a context-manager-style handle whose
  ``.remove()`` un-registers all hooks. The handle also exposes
  ``coverage_summary()``.
* :func:`make_turboquant_baseline` — same as above but for the
  ``TurboQuantBaseline`` engine; supports apples-to-apples comparison
  of v2 vs the in-repo TurboQuant baseline.

This module deliberately avoids importing ``transformers`` at import
time. Heavy imports happen inside the functions so a unit test can
monkey-patch a tiny fake model without pulling the HF stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coverage summary (what fraction of layers / heads got real compression)
# ---------------------------------------------------------------------------


@dataclass
class ReplayCoverage:
    """Summary of which layers/heads have real compressed-method coverage."""

    n_layers_total: int = 0
    n_layers_calibrated: int = 0
    n_layers_hooked: int = 0
    n_hook_calls: int = 0
    n_passthrough_calls: int = 0
    missing_layers: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_layers_total": int(self.n_layers_total),
            "n_layers_calibrated": int(self.n_layers_calibrated),
            "n_layers_hooked": int(self.n_layers_hooked),
            "n_hook_calls": int(self.n_hook_calls),
            "n_passthrough_calls": int(self.n_passthrough_calls),
            "missing_layers": list(self.missing_layers),
            "fraction_layers_real": (
                self.n_layers_hooked / max(1, self.n_layers_total)
            ),
        }


# ---------------------------------------------------------------------------
# Hook implementation
# ---------------------------------------------------------------------------


def _reshape_proj_to_heads(
    output: Any, n_kv_heads: int, head_dim: int,
) -> Tuple[Any, Tuple[int, ...]]:
    """Reshape projection output to (batch, n_kv_heads, seq, head_dim).

    Accepts (batch, seq, n_kv_heads*head_dim). Returns the reshaped
    tensor and the original shape so the inverse can run. Tensors only;
    no transformers dependency.
    """
    import torch  # noqa: F401  (caller has torch)
    orig_shape = tuple(output.shape)
    if len(orig_shape) != 3:
        raise ValueError(
            f"k_proj/v_proj output must be 3D (batch, seq, n_kv*head_dim); "
            f"got {orig_shape}"
        )
    bsz, seq, hidden = orig_shape
    expected = n_kv_heads * head_dim
    if hidden != expected:
        raise ValueError(
            f"hidden={hidden} != n_kv_heads*head_dim={expected}; "
            f"adapter mismatch?"
        )
    reshaped = output.reshape(bsz, seq, n_kv_heads, head_dim).permute(
        0, 2, 1, 3
    ).contiguous()
    return reshaped, orig_shape


def _reshape_heads_to_proj(reconstructed: Any, orig_shape: Tuple[int, ...]) -> Any:
    """Inverse of :func:`_reshape_proj_to_heads`."""
    bsz, seq, hidden = orig_shape
    return reconstructed.permute(0, 2, 1, 3).contiguous().reshape(bsz, seq, hidden)


def _round_trip_keys_v2(
    engine: Any, keys: Any, layer_idx: int,
) -> Any:
    """Compress + decompress keys using the v2 engine.

    The v2 ``SpectralQuantEngine`` does not expose a ``decompress_keys``
    helper; this function inlines the per-head compress→decompress→
    unrotate pipeline that v2's :meth:`compare_with_baseline` uses
    internally.
    """
    import torch
    compressed = engine.compress_keys(keys, layer_idx)
    head_hats: List[Any] = []
    for head_idx, cv in sorted(compressed.items()):
        quant = engine._get_quantizer(layer_idx, head_idx, "key")
        k_rot_hat = quant.decompress(cv)
        k_hat = engine._key_rotation.unrotate(k_rot_hat, layer_idx, head_idx)
        head_hats.append(k_hat)
    return torch.stack(head_hats, dim=1)


def _round_trip_values_v2(
    engine: Any, values: Any, layer_idx: int,
) -> Any:
    cv = engine.compress_values(values, layer_idx)
    return engine.decompress_values(cv, layer_idx)


def _round_trip_keys_baseline(
    engine: Any, keys: Any, layer_idx: int,
) -> Any:
    ck = engine.compress_keys(keys, layer_idx)
    return engine.decompress_keys(ck, layer_idx)


def _round_trip_values_baseline(
    engine: Any, values: Any, layer_idx: int,
) -> Any:
    cv = engine.compress_values(values, layer_idx)
    return engine.decompress_values(cv, layer_idx)


@dataclass
class ReplayHandle:
    """Returned by :func:`attach_replay_hooks` for cleanup + coverage."""

    handles: List[Any] = field(default_factory=list)
    coverage: ReplayCoverage = field(default_factory=ReplayCoverage)

    def remove(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def coverage_summary(self) -> Dict[str, Any]:
        return self.coverage.to_dict()


def attach_replay_hooks(
    model: Any,
    engine: Any,
    *,
    n_kv_heads: int,
    head_dim: int,
    method: str,
    calibrated_layers: Sequence[int],
) -> ReplayHandle:
    """Attach K/V projection-replay hooks for the requested method.

    Parameters
    ----------
    model: HuggingFace model (must expose ``.model.layers[i].self_attn``
           with ``k_proj`` and ``v_proj`` modules) or any object that
           passes :func:`experiments.model_adapters.find_attention_modules`.
    engine: A *fitted* SpectralQuantEngine (v2) or TurboQuantBaseline.
    n_kv_heads, head_dim: model dims (resolved by the adapter).
    method: ``"spectralquant_v2"`` or ``"turboquant"`` — chooses the
            round-trip shape (v2 engine doesn't expose decompress_keys,
            TurboQuantBaseline does, so we dispatch).
    calibrated_layers: layer indices for which the engine has calibration
                       and quantizers fitted; layers outside this set
                       fall through (FP16 passthrough) and are recorded
                       in the coverage summary.

    Returns
    -------
    ReplayHandle: call ``.remove()`` to detach, ``.coverage_summary()``
                  for an artifact-friendly dict.
    """
    from experiments import model_adapters

    layers = model_adapters.find_attention_modules(model)
    handle = ReplayHandle()
    handle.coverage.n_layers_total = len(layers)
    calibrated = set(int(i) for i in calibrated_layers)
    handle.coverage.n_layers_calibrated = len(calibrated)

    if method == "spectralquant_v2":
        rt_keys = _round_trip_keys_v2
        rt_vals = _round_trip_values_v2
    elif method == "turboquant":
        rt_keys = _round_trip_keys_baseline
        rt_vals = _round_trip_values_baseline
    else:
        raise ValueError(
            f"Unknown method {method!r}; supported: spectralquant_v2, turboquant"
        )

    def _make_hook(
        layer_idx: int, head_type: str, rt_fn: Callable[[Any, Any, int], Any],
    ) -> Callable[[Any, Any, Any], Any]:
        cov = handle.coverage

        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if layer_idx not in calibrated:
                cov.n_passthrough_calls += 1
                return output
            try:
                heads, orig_shape = _reshape_proj_to_heads(
                    output, n_kv_heads, head_dim,
                )
                # Dtype: rotate via float32 (engine math) and cast back.
                in_dtype = output.dtype
                heads_f32 = heads.float()
                hat = rt_fn(engine, heads_f32, layer_idx)
                cov.n_hook_calls += 1
                out = _reshape_heads_to_proj(hat, orig_shape).to(in_dtype)
                return out
            except Exception as exc:  # noqa: BLE001
                # Be loud but don't kill the forward pass; record passthrough.
                logger.warning(
                    "replay hook (%s) layer=%d failed: %s; falling back to FP16",
                    head_type, layer_idx, exc,
                )
                cov.n_passthrough_calls += 1
                return output

        return _hook

    n_hooked = 0
    for layer_meta in layers:
        layer_idx = int(layer_meta.layer_idx)
        attn = layer_meta.attn_module
        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        if k_proj is None or v_proj is None:
            handle.coverage.missing_layers.append(layer_idx)
            continue
        if layer_idx not in calibrated:
            handle.coverage.missing_layers.append(layer_idx)
            continue
        h_k = k_proj.register_forward_hook(
            _make_hook(layer_idx, "key", rt_keys)
        )
        h_v = v_proj.register_forward_hook(
            _make_hook(layer_idx, "value", rt_vals)
        )
        handle.handles.extend([h_k, h_v])
        n_hooked += 1
    handle.coverage.n_layers_hooked = n_hooked
    return handle


# ---------------------------------------------------------------------------
# Calibration helpers (real model path)
# ---------------------------------------------------------------------------


def _capture_rotated_kv(
    model: Any,
    tokenizer: Any,
    calibrator: Any,
    texts: Sequence[str],
    *,
    max_seq_tokens: int = 512,
) -> Dict[Tuple[int, int, str], Any]:
    """After calibration, capture pre-rotated K/V per (layer, head, type).

    Mirrors run_three_way's pattern: register transient hooks on each
    layer's k_proj/v_proj, run a forward pass over the calibration
    corpus, and stack the per-head rotated outputs into a tensor of
    shape ``(n_tokens, head_dim)`` for use by ``fit_quantizers``.
    """
    import torch
    from experiments import model_adapters
    from spectralquant.calibration import _get_kv_head_dims

    layers = model_adapters.find_attention_modules(model)
    cfg = getattr(model, "config", None)
    n_q_heads, n_kv_heads, head_dim = model_adapters.get_kv_dims(
        cfg, layers[0].attn_module
    )

    # Buckets: (layer_idx, head_idx, head_type) -> list of (n_tokens, head_dim)
    buckets: Dict[Tuple[int, int, str], List[Any]] = {}

    handles: List[Any] = []

    def _make_hook(layer_idx: int, head_type: str):
        def _hook(_mod, _inp, output):
            try:
                bsz, seq, hidden = output.shape
                if hidden != n_kv_heads * head_dim:
                    return
                heads = output.detach().reshape(
                    bsz, seq, n_kv_heads, head_dim
                ).permute(0, 2, 1, 3).contiguous().to("cpu", dtype=torch.float32)
                # Stack as (n_tokens, head_dim) per head, after rotation.
                for h in range(n_kv_heads):
                    flat = heads[:, h, :, :].reshape(-1, head_dim)
                    # Rotate using the calibrator's stored rotation.
                    hcd = calibrator.get(layer_idx, h, head_type)
                    if hcd is None:
                        continue
                    # eigenvectors are columns; project: x @ V (since V^T is the
                    # rotation that diagonalizes).  Match SpectralRotation.rotate.
                    V = hcd.eigenvectors  # (head_dim, head_dim)
                    rotated = flat @ V
                    buckets.setdefault(
                        (layer_idx, h, head_type), []
                    ).append(rotated)
            except Exception as exc:  # noqa: BLE001
                logger.debug("capture rotated kv hook err l=%d %s: %s",
                             layer_idx, head_type, exc)
        return _hook

    for layer_meta in layers:
        attn = layer_meta.attn_module
        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        if k_proj is not None:
            handles.append(k_proj.register_forward_hook(
                _make_hook(int(layer_meta.layer_idx), "key")
            ))
        if v_proj is not None:
            handles.append(v_proj.register_forward_hook(
                _make_hook(int(layer_meta.layer_idx), "value")
            ))

    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=max_seq_tokens,
                ).to(device)
                model(**inputs, use_cache=False, output_attentions=False)
    finally:
        for h in handles:
            h.remove()

    out: Dict[Tuple[int, int, str], Any] = {}
    for key, parts in buckets.items():
        out[key] = torch.cat(parts, dim=0)
    return out


@dataclass
class CalibratedEngineBundle:
    """Holds the artifacts needed to run the replay hooks."""

    engine: Any
    method: str
    calibrated_layers: List[int]
    n_kv_heads: int
    head_dim: int
    coverage: Dict[str, Any] = field(default_factory=dict)


def build_calibrated_engine(
    model: Any,
    tokenizer: Any,
    calibration_texts: Sequence[str],
    *,
    method: str = "spectralquant_v2",
    avg_bits: float = 3.0,
    use_water_fill: bool = True,
    n_calib_samples: int = 16,
    max_seq_tokens: int = 256,
    lloyd_max_iter: int = 200,
    progress: Optional[Callable[[str, str, Optional[Dict[str, Any]]], None]] = None,
) -> CalibratedEngineBundle:
    """Run calibration + quantizer fit and return a bundle ready for hooks.

    ``method`` ∈ {``"spectralquant_v2"``, ``"turboquant"``}. v2 uses the
    eigenspectral calibrator + non-uniform quantizers + water-fill bits;
    ``turboquant`` uses :class:`TurboQuantBaseline` with the same fitted
    rotations/codebooks.

    Parameters
    ----------
    lloyd_max_iter:
        Cap on Lloyd-Max iterations per per-head codebook fit. Lower values
        (e.g. 25–50) trade a little quantization quality for substantially
        faster calibration on CPU; the default ``200`` matches the existing
        ``EngineConfig.lloyd_max_iter`` and is the recommended setting for
        paper-valid runs.
    progress:
        Optional callable ``progress(stage, message, details)`` invoked at
        coarse milestones ("calib_eigh_start", "calib_capture_start",
        "calib_fit_start", "calib_fit_progress", "calib_fit_end"). It is
        called from the calling thread; the callback should be fast — the
        common implementation just forwards to a :class:`StatusWriter`.
    """
    from spectralquant import EngineConfig, SpectralQuantEngine, TurboQuantBaseline
    from spectralquant.calibration import EigenspectralCalibrator
    from experiments import model_adapters

    def _emit(stage: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        if progress is not None:
            try:
                progress(stage, message, details)
            except Exception:  # noqa: BLE001
                pass

    layers = model_adapters.find_attention_modules(model)
    cfg = getattr(model, "config", None)
    n_q_heads, n_kv_heads, head_dim = model_adapters.get_kv_dims(
        cfg, layers[0].attn_module
    )
    n_layers = len(layers)

    _emit(
        "calib_eigh_start",
        f"method={method} n_calib={n_calib_samples} n_layers={n_layers} "
        f"n_kv_heads={n_kv_heads} head_dim={head_dim}",
        {"n_calib": int(n_calib_samples), "n_layers": int(n_layers),
         "n_kv_heads": int(n_kv_heads), "head_dim": int(head_dim),
         "lloyd_max_iter": int(lloyd_max_iter)},
    )

    # 1. Calibration.
    calib = EigenspectralCalibrator(max_tokens_per_layer=4096)
    calib.calibrate(model, tokenizer, list(calibration_texts), n_samples=n_calib_samples)
    _emit(
        "calib_eigh_end",
        f"calibrated heads={sum(1 for _ in calib.iter_heads())}",
        {"n_heads_calibrated": int(sum(1 for _ in calib.iter_heads()))},
    )

    # 2. Fit quantizers from rotated K/V captured on the same corpus.
    _emit("calib_capture_start", f"capturing rotated K/V on n={min(n_calib_samples, len(calibration_texts))} texts")
    rotated_kv = _capture_rotated_kv(
        model, tokenizer, calib, list(calibration_texts)[:n_calib_samples],
        max_seq_tokens=max_seq_tokens,
    )
    calibrated_layers = sorted({k[0] for k in rotated_kv.keys()})
    _emit("calib_capture_end",
          f"captured layers={len(calibrated_layers)} entries={len(rotated_kv)}",
          {"n_calibrated_layers": len(calibrated_layers),
           "n_entries": len(rotated_kv)})

    if method == "spectralquant_v2":
        eng_cfg = EngineConfig(
            avg_bits=float(avg_bits),
            use_water_fill=bool(use_water_fill),
            wf_min_bits=0,
            lloyd_max_iter=int(lloyd_max_iter),
        )
        engine: Any = SpectralQuantEngine(calib, eng_cfg)
        _emit("calib_fit_start",
              f"fitting v2 quantizers entries={len(rotated_kv)} "
              f"lloyd_max_iter={lloyd_max_iter}",
              {"n_entries": len(rotated_kv),
               "lloyd_max_iter": int(lloyd_max_iter),
               "use_water_fill": bool(use_water_fill)})
        _fit_quantizers_with_progress(engine, rotated_kv, _emit)
    elif method == "turboquant":
        eng_cfg = EngineConfig(
            avg_bits=float(avg_bits),
            use_water_fill=False,
            lloyd_max_iter=int(lloyd_max_iter),
        )
        engine = TurboQuantBaseline(
            n_layers=n_layers, n_heads=n_kv_heads, head_dim=head_dim,
            config=eng_cfg,
        )
        # TurboQuantBaseline expects raw (un-rotated) K/V — the random
        # rotation lives inside the baseline. Re-capture without rotation.
        raw_kv = _capture_raw_kv(model, tokenizer, list(calibration_texts)[:n_calib_samples],
                                 n_kv_heads=n_kv_heads, head_dim=head_dim,
                                 max_seq_tokens=max_seq_tokens)
        calibrated_layers = sorted({k[0] for k in raw_kv.keys()})
        _emit("calib_fit_start",
              f"fitting turboquant quantizers entries={len(raw_kv)}",
              {"n_entries": len(raw_kv),
               "lloyd_max_iter": int(lloyd_max_iter)})
        engine.fit_quantizers(raw_kv)
        _emit("calib_fit_end",
              f"fit done; quantizers={len(getattr(engine, '_quantizers', {}))}",
              None)
    else:
        raise ValueError(
            f"method must be 'spectralquant_v2' or 'turboquant'; got {method!r}"
        )

    return CalibratedEngineBundle(
        engine=engine,
        method=method,
        calibrated_layers=calibrated_layers,
        n_kv_heads=int(n_kv_heads),
        head_dim=int(head_dim),
        coverage={
            "n_layers": int(n_layers),
            "n_calibrated": int(len(calibrated_layers)),
            "method": method,
            "avg_bits": float(avg_bits),
            "use_water_fill": bool(use_water_fill) if method == "spectralquant_v2" else False,
            "lloyd_max_iter": int(lloyd_max_iter),
            "n_calib_samples": int(n_calib_samples),
        },
    )


def _fit_quantizers_with_progress(
    engine: Any,
    rotated_kv: Dict[Tuple[int, int, str], Any],
    emit: Callable[[str, str, Optional[Dict[str, Any]]], None],
) -> None:
    """Fit v2 engine quantizers entry-by-entry, emitting progress.

    Mirrors ``SpectralQuantEngine.fit_quantizers`` but loops manually so
    the harness can emit progress between heads. Layout-compatible: same
    ``_quantizers`` dict is populated, ``_is_fitted`` is set at the end.
    """
    from spectralquant.nonuniform_quantization import NonUniformQuantizer

    # Sort so the progress order is deterministic and easy to read.
    keys = sorted(rotated_kv.keys())
    total = len(keys)
    last_layer = -1
    for i, key in enumerate(keys):
        layer_idx, head_idx, head_type = key
        data = rotated_kv[key]
        hcd = engine._calibrator.get(layer_idx, head_idx, head_type)
        if hcd is None:
            continue
        cfg = engine._config
        quant = NonUniformQuantizer(
            eigenvalues=hcd.eigenvalues,
            avg_bits=cfg.avg_bits,
            max_lloyd_iter=cfg.lloyd_max_iter,
            seed=cfg.lloyd_seed,
            use_water_fill=cfg.use_water_fill,
            wf_min_bits=cfg.wf_min_bits,
            wf_max_bits=cfg.wf_max_bits,
        ).fit(data, d_eff=hcd.d_eff)
        engine._quantizers[(layer_idx, head_idx, head_type)] = quant
        # Emit per-layer progress: one event when the layer changes.
        if layer_idx != last_layer:
            emit(
                "calib_fit_progress",
                f"layer={layer_idx} head={head_idx} type={head_type} "
                f"({i + 1}/{total})",
                {"layer_idx": int(layer_idx), "head_idx": int(head_idx),
                 "head_type": str(head_type), "fit_index": i + 1,
                 "fit_total": total},
            )
            last_layer = layer_idx
    engine._is_fitted = True
    emit(
        "calib_fit_end",
        f"fit done; quantizers={len(engine._quantizers)}",
        {"n_quantizers": int(len(engine._quantizers))},
    )


def _capture_raw_kv(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    n_kv_heads: int,
    head_dim: int,
    max_seq_tokens: int = 256,
) -> Dict[Tuple[int, int, str], Any]:
    """Capture un-rotated K/V vectors per (layer, head, type) for TurboQuant fit."""
    import torch
    from experiments import model_adapters

    layers = model_adapters.find_attention_modules(model)
    buckets: Dict[Tuple[int, int, str], List[Any]] = {}
    handles: List[Any] = []

    def _make_hook(layer_idx: int, head_type: str):
        def _hook(_mod, _inp, output):
            try:
                bsz, seq, hidden = output.shape
                if hidden != n_kv_heads * head_dim:
                    return
                heads = output.detach().reshape(
                    bsz, seq, n_kv_heads, head_dim
                ).permute(0, 2, 1, 3).contiguous().to("cpu", dtype=torch.float32)
                for h in range(n_kv_heads):
                    flat = heads[:, h, :, :].reshape(-1, head_dim)
                    buckets.setdefault((layer_idx, h, head_type), []).append(flat)
            except Exception:
                pass
        return _hook

    for layer_meta in layers:
        attn = layer_meta.attn_module
        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        if k_proj is not None:
            handles.append(k_proj.register_forward_hook(
                _make_hook(int(layer_meta.layer_idx), "key")
            ))
        if v_proj is not None:
            handles.append(v_proj.register_forward_hook(
                _make_hook(int(layer_meta.layer_idx), "value")
            ))

    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=max_seq_tokens,
                ).to(device)
                model(**inputs, use_cache=False, output_attentions=False)
    finally:
        for h in handles:
            h.remove()

    out: Dict[Tuple[int, int, str], Any] = {}
    for key, parts in buckets.items():
        out[key] = __import__("torch").cat(parts, dim=0)
    return out
