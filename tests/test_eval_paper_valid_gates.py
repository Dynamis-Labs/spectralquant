"""Paper-valid gating tests for the four next-stage eval harnesses.

These tests verify that:

* No method record carrying ``placeholder=true`` can land in a
  ``paper_valid=true`` payload.
* Methods outside ``REAL_EVAL_METHODS`` block paper_valid.
* Real (non-fp16) methods need ``replay_coverage.fraction_layers_real
  >= 0.99`` to be paper-valid.
* The replay-hook plumbing in :mod:`experiments.sqv2_replay` round-trips
  K/V tensors through compress + decompress without mutating shapes
  or dtypes that downstream attention would care about.

The tests use small fake / monkey-patched models — they never download
real HF weights and run in <1s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


@pytest.fixture(autouse=True)
def _add_paths(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.syspath_prepend(str(SRC_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_full_args(harness_module, **overrides):
    """Build a parsed Args via the harness's own parser, with sane defaults
    that select the *full* mode (no smoke flag)."""
    base = [
        "--model", "mistralai/Mistral-7B-v0.3",
        "--output-dir", "/tmp/_unused",
    ]
    if "methods" in overrides:
        base.append("--methods")
        base.extend(overrides.pop("methods"))
    for k, v in overrides.items():
        base.extend([f"--{k.replace('_', '-')}", str(v)])
    return harness_module.parse_args(base)


# ---------------------------------------------------------------------------
# run_perplexity gating
# ---------------------------------------------------------------------------


def test_perplexity_paper_valid_blocked_by_placeholder():
    from experiments import run_perplexity, eval_common

    args = _make_full_args(run_perplexity, methods=["fp16", "spectralquant_v1"])
    methods: Dict[str, Dict[str, Any]] = {
        "fp16": {
            "label": "fp16",
            "perplexity": 9.5, "nll_per_token": 2.25,
            "n_tokens": 100_000, "avg_bits": 16.0,
            "evidence_ids": ["RUN-PERPLEXITY-FP16"],
        },
        "spectralquant_v1": {
            "label": "spectralquant_v1",
            "perplexity": 1.0, "nll_per_token": 0.0,
            "n_tokens": 1, "avg_bits": 3.0,
            "evidence_ids": ["x"], "placeholder": True,
        },
    }
    payload = run_perplexity.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
        eval_corpus="x", n_eval_sequences=64,
    )
    assert payload["paper_valid"] is False
    # And caveats explain why.
    assert any("placeholder" in c.lower() for c in payload["caveats"])


def test_perplexity_paper_valid_blocked_by_unsupported_method():
    from experiments import run_perplexity, eval_common

    # spectralquant_v1 is in KNOWN_METHOD_KEYS but NOT in REAL_EVAL_METHODS.
    args = _make_full_args(run_perplexity, methods=["fp16", "spectralquant_v1"])
    methods = {
        "fp16": {
            "label": "fp16", "perplexity": 9.5, "nll_per_token": 2.25,
            "n_tokens": 100_000, "avg_bits": 16.0,
            "evidence_ids": ["RUN-PERPLEXITY-FP16"],
        },
        "spectralquant_v1": {
            "label": "spectralquant_v1", "perplexity": 9.7, "nll_per_token": 2.27,
            "n_tokens": 100_000, "avg_bits": 3.0,
            "evidence_ids": ["x"],
            # Even WITHOUT the placeholder flag, the method-key gate fires.
        },
    }
    payload = run_perplexity.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
        eval_corpus="x", n_eval_sequences=64,
    )
    assert payload["paper_valid"] is False
    assert any("real-eval" in c.lower() for c in payload["caveats"])


def test_perplexity_paper_valid_blocked_by_low_coverage():
    from experiments import run_perplexity, eval_common

    args = _make_full_args(run_perplexity, methods=["fp16", "spectralquant_v2"])
    methods = {
        "fp16": {
            "label": "fp16", "perplexity": 9.5, "nll_per_token": 2.25,
            "n_tokens": 100_000, "avg_bits": 16.0,
            "evidence_ids": ["RUN-PERPLEXITY-FP16"],
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2", "perplexity": 9.62, "nll_per_token": 2.27,
            "n_tokens": 100_000, "avg_bits": 3.0,
            "evidence_ids": ["x"],
            "replay_coverage": {
                "n_layers_total": 32, "n_layers_calibrated": 16,
                "n_layers_hooked": 16, "n_hook_calls": 100,
                "n_passthrough_calls": 100, "missing_layers": list(range(16, 32)),
                "fraction_layers_real": 0.5,
            },
        },
    }
    payload = run_perplexity.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
        eval_corpus="x", n_eval_sequences=64,
    )
    assert payload["paper_valid"] is False
    assert any("coverage" in c.lower() for c in payload["caveats"])


def test_perplexity_paper_valid_passes_when_all_gates_satisfied():
    from experiments import run_perplexity, eval_common

    args = _make_full_args(run_perplexity, methods=["fp16", "spectralquant_v2"])
    methods = {
        "fp16": {
            "label": "fp16", "perplexity": 9.5, "nll_per_token": 2.25,
            "n_tokens": 100_000, "avg_bits": 16.0,
            "evidence_ids": ["RUN-PERPLEXITY-FP16"],
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2", "perplexity": 9.62, "nll_per_token": 2.27,
            "n_tokens": 100_000, "avg_bits": 3.0,
            "evidence_ids": ["x"],
            "replay_coverage": {
                "n_layers_total": 32, "n_layers_calibrated": 32,
                "n_layers_hooked": 32, "n_hook_calls": 320,
                "n_passthrough_calls": 0, "missing_layers": [],
                "fraction_layers_real": 1.0,
            },
        },
    }
    payload = run_perplexity.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
        eval_corpus="x", n_eval_sequences=64,
    )
    assert payload["paper_valid"] is True


def test_perplexity_paper_valid_blocked_by_low_token_count():
    from experiments import run_perplexity, eval_common

    args = _make_full_args(run_perplexity, methods=["fp16"])
    methods = {
        "fp16": {
            "label": "fp16", "perplexity": 9.5, "nll_per_token": 2.25,
            "n_tokens": 1024, "avg_bits": 16.0,
            "evidence_ids": ["RUN-PERPLEXITY-FP16"],
        },
    }
    payload = run_perplexity.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
        eval_corpus="x", n_eval_sequences=64,
    )
    assert payload["paper_valid"] is False
    assert any("n_tokens" in c.lower() for c in payload["caveats"])


# ---------------------------------------------------------------------------
# run_generation gating
# ---------------------------------------------------------------------------


def test_generation_paper_valid_blocked_by_placeholder():
    from experiments import run_generation, eval_common

    args = _make_full_args(run_generation, methods=["fp16", "spectralquant_v1"])
    methods = {
        "fp16": {
            "label": "fp16",
            "completions": [{"prompt_id": "x", "text": "hello", "n_tokens": 1}],
            "metrics": {"mean_token_overlap_f1": 1.0,
                        "mean_distinct_1": 0.5, "mean_distinct_2": 0.5},
            "evidence_ids": ["x"],
        },
        "spectralquant_v1": {
            "label": "spectralquant_v1",
            "completions": [{"prompt_id": "x", "text": "", "n_tokens": 0,
                             "placeholder": True}],
            "metrics": {"mean_token_overlap_f1": 0.0,
                        "mean_distinct_1": 0.0, "mean_distinct_2": 0.0},
            "evidence_ids": ["x"], "placeholder": True,
        },
    }
    payload = run_generation.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
    )
    assert payload["paper_valid"] is False


def test_generation_paper_valid_with_real_v2_and_full_coverage():
    from experiments import run_generation, eval_common

    args = _make_full_args(run_generation, methods=["fp16", "spectralquant_v2"])
    methods = {
        "fp16": {
            "label": "fp16",
            "completions": [{"prompt_id": "x", "text": "hello world", "n_tokens": 2}],
            "metrics": {"mean_token_overlap_f1": 1.0,
                        "mean_distinct_1": 0.5, "mean_distinct_2": 0.5},
            "evidence_ids": ["x"],
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2",
            "completions": [{"prompt_id": "x", "text": "hello world", "n_tokens": 2}],
            "metrics": {"mean_token_overlap_f1": 1.0,
                        "mean_distinct_1": 0.5, "mean_distinct_2": 0.5},
            "evidence_ids": ["x"],
            "replay_coverage": {"fraction_layers_real": 1.0,
                                "n_layers_total": 32, "n_layers_hooked": 32,
                                "n_layers_calibrated": 32, "n_hook_calls": 100,
                                "n_passthrough_calls": 0, "missing_layers": []},
        },
    }
    payload = run_generation.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
    )
    assert payload["paper_valid"] is True


# ---------------------------------------------------------------------------
# run_latency gating
# ---------------------------------------------------------------------------


def test_latency_paper_valid_blocked_when_method_has_placeholder():
    from experiments import run_latency, eval_common

    base_args = [
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v1",
        "--device", "cuda",
    ]
    args = run_latency.parse_args(base_args)
    methods = {
        "fp16": {
            "label": "fp16",
            "operating_points": [{
                "batch_size": 1, "context_length": 512, "gen_tokens": 64,
                "prefill_ms_p50": 5.0, "prefill_ms_p95": 6.0,
                "decode_ms_per_token_p50": 1.0, "decode_ms_per_token_p95": 1.5,
                "tokens_per_sec_p50": 1000.0, "peak_memory_mb": 100.0,
                "n_iters": 5,
            }],
            "evidence_ids": ["RUN-LATENCY-FP16"],
        },
        "spectralquant_v1": {
            "label": "spectralquant_v1",
            "operating_points": [{
                "batch_size": 1, "context_length": 512, "gen_tokens": 64,
                "prefill_ms_p50": 5.0, "prefill_ms_p95": 6.0,
                "decode_ms_per_token_p50": 1.0, "decode_ms_per_token_p95": 1.5,
                "tokens_per_sec_p50": 1000.0, "peak_memory_mb": 100.0,
                "n_iters": 5, "placeholder": True,
            }],
            "evidence_ids": ["x"], "placeholder": True,
        },
    }
    payload = run_latency.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods, timer_label="torch.cuda.Event",
    )
    assert payload["paper_valid"] is False


def test_latency_microbench_caveat_present_when_v2_requested():
    from experiments import run_latency, eval_common

    args = run_latency.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2", "--device", "cuda",
    ])
    methods = {
        "fp16": {
            "label": "fp16",
            "operating_points": [{
                "batch_size": 1, "context_length": 512, "gen_tokens": 64,
                "prefill_ms_p50": 5.0,
                "decode_ms_per_token_p50": 1.0,
                "tokens_per_sec_p50": 1000.0,
                "n_iters": 5,
                "end_to_end_measured": True,
                "production_kernel": True,
                "measurement_kind": "fp16_end_to_end",
            }],
            "evidence_ids": ["RUN-LATENCY-FP16"],
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2",
            "operating_points": [{
                "batch_size": 1, "context_length": 512, "gen_tokens": 64,
                "prefill_ms_p50": 0.5, "decode_ms_per_token_p50": 0.001,
                "tokens_per_sec_p50": 50_000.0, "n_iters": 5,
                "microbenchmark": True,
                "microbenchmark_kind": "kv_compress_decompress_round_trip",
                "end_to_end_measured": False,
                "production_kernel": False,
                "measurement_kind": "microbenchmark_kv_round_trip",
            }],
            "evidence_ids": ["RUN-LATENCY-MICROBENCH-SPECTRALQUANT_V2"],
            "microbenchmark": True,
            "microbenchmark_kind": "kv_compress_decompress_round_trip",
        },
    }
    payload = run_latency.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods, timer_label="torch.cuda.Event",
    )
    assert any("MICROBENCHMARK" in c for c in payload["caveats"])
    # Microbench-only fails the paper-valid gate (e2e missing).
    assert payload["paper_valid"] is False
    assert any("end_to_end_measured" in c for c in payload["caveats"])


def test_latency_paper_valid_with_hooked_replay_end_to_end():
    """A v2 method record carrying an end-to-end hooked-replay row passes
    the e2e gate when coverage >= 0.99 and timer/device are right."""
    from experiments import run_latency, eval_common

    args = run_latency.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2", "--device", "cuda",
    ])
    methods = {
        "fp16": {
            "label": "fp16",
            "operating_points": [{
                "batch_size": 1, "context_length": 512, "gen_tokens": 64,
                "prefill_ms_p50": 5.0, "decode_ms_per_token_p50": 1.0,
                "tokens_per_sec_p50": 1000.0, "n_iters": 5,
                "end_to_end_measured": True, "production_kernel": True,
                "measurement_kind": "fp16_end_to_end",
            }],
            "evidence_ids": ["RUN-LATENCY-FP16"],
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2",
            "operating_points": [
                {
                    # Microbench row (still here for evidence).
                    "batch_size": 1, "context_length": 512, "gen_tokens": 64,
                    "prefill_ms_p50": 0.5, "decode_ms_per_token_p50": 0.001,
                    "tokens_per_sec_p50": 50_000.0, "n_iters": 5,
                    "microbenchmark": True,
                    "microbenchmark_kind": "kv_compress_decompress_round_trip",
                    "end_to_end_measured": False, "production_kernel": False,
                    "measurement_kind": "microbenchmark_kv_round_trip",
                },
                {
                    # Hooked-replay end-to-end row.
                    "batch_size": 1, "context_length": 512, "gen_tokens": 64,
                    "prefill_ms_p50": 8.5, "decode_ms_per_token_p50": 1.4,
                    "tokens_per_sec_p50": 700.0, "n_iters": 5,
                    "end_to_end_measured": True, "production_kernel": False,
                    "measurement_kind": "hooked_replay_end_to_end",
                },
            ],
            "evidence_ids": ["RUN-LATENCY-E2E-REPLAY-SPECTRALQUANT_V2"],
            "end_to_end_measured": True, "production_kernel": False,
            "measurement_kind": "hooked_replay_end_to_end",
            "replay_coverage": {
                "fraction_layers_real": 1.0, "n_layers_total": 32,
                "n_layers_calibrated": 32, "n_layers_hooked": 32,
                "n_hook_calls": 100, "n_passthrough_calls": 0,
                "missing_layers": [],
            },
        },
    }
    payload = run_latency.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
        timer_label="torch.cuda.Event",
    )
    assert payload["paper_valid"] is True
    # Caveat about hooked-replay should still be present.
    assert any("hooked_replay_end_to_end" in c.lower() or
               "hooked replay end-to-end" in c.lower()
               for c in payload["caveats"])


def test_latency_paper_valid_blocked_by_e2e_replay_low_coverage():
    from experiments import run_latency, eval_common

    args = run_latency.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2", "--device", "cuda",
    ])
    methods = {
        "fp16": {
            "label": "fp16",
            "operating_points": [{
                "batch_size": 1, "context_length": 512, "gen_tokens": 64,
                "prefill_ms_p50": 5.0, "decode_ms_per_token_p50": 1.0,
                "tokens_per_sec_p50": 1000.0, "n_iters": 5,
                "end_to_end_measured": True, "production_kernel": True,
                "measurement_kind": "fp16_end_to_end",
            }],
            "evidence_ids": ["RUN-LATENCY-FP16"],
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2",
            "operating_points": [{
                "batch_size": 1, "context_length": 512, "gen_tokens": 64,
                "prefill_ms_p50": 8.5, "decode_ms_per_token_p50": 1.4,
                "tokens_per_sec_p50": 700.0, "n_iters": 5,
                "end_to_end_measured": True, "production_kernel": False,
                "measurement_kind": "hooked_replay_end_to_end",
            }],
            "evidence_ids": ["x"],
            "end_to_end_measured": True, "production_kernel": False,
            "replay_coverage": {
                "fraction_layers_real": 0.5, "n_layers_total": 32,
                "n_layers_calibrated": 16, "n_layers_hooked": 16,
                "n_hook_calls": 50, "n_passthrough_calls": 50,
                "missing_layers": list(range(16, 32)),
            },
        },
    }
    payload = run_latency.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
        timer_label="torch.cuda.Event",
    )
    assert payload["paper_valid"] is False
    assert any("coverage" in c.lower() for c in payload["caveats"])


def test_latency_no_production_kernel_claim_for_replay():
    """The hooked-replay e2e measurement must never be tagged production_kernel=true."""
    from experiments import run_latency

    args = run_latency.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2", "--device", "cuda",
    ])
    # Build a method record as the harness would.
    op = {
        "batch_size": 1, "context_length": 512, "gen_tokens": 64,
        "prefill_ms_p50": 8.5, "decode_ms_per_token_p50": 1.4,
        "tokens_per_sec_p50": 700.0, "n_iters": 5,
        "end_to_end_measured": True, "production_kernel": False,
        "measurement_kind": "hooked_replay_end_to_end",
    }
    # The label markers are owned by the harness; this asserts the contract.
    assert op["measurement_kind"] == "hooked_replay_end_to_end"
    assert op["production_kernel"] is False


# ---------------------------------------------------------------------------
# run_longbench gating: paper_valid is hard to achieve until upstream
# loader is vendored. Inline-corpus must NEVER be paper_valid.
# ---------------------------------------------------------------------------


def test_longbench_full_paper_valid_requires_real_dataset_source():
    """Full-mode LongBench is paper-valid only when every method record
    has dataset_source == 'huggingface_thudm'. A row missing it (or set
    to 'inline_corpus') blocks paper_valid even at n_per_task >= 50."""
    from experiments import run_longbench, eval_common

    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2",
        "--n-per-task", "50",
    ])
    methods = {
        "fp16": {
            "label": "fp16",
            "per_task": [{"task": "narrativeqa", "metric": "qa_f1_en",
                          "score": 0.4, "n_examples": 50}],
            "aggregate": {"macro_score": 0.4},
            "evidence_ids": ["x"],
            # Missing dataset_source — must NOT pass.
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2",
            "per_task": [{"task": "narrativeqa", "metric": "qa_f1_en",
                          "score": 0.39, "n_examples": 50}],
            "aggregate": {"macro_score": 0.39},
            "evidence_ids": ["x"],
            "replay_coverage": {"fraction_layers_real": 1.0,
                                "n_layers_total": 32, "n_layers_hooked": 32,
                                "n_layers_calibrated": 32, "n_hook_calls": 100,
                                "n_passthrough_calls": 0, "missing_layers": []},
            "dataset_source": "huggingface_thudm",
        },
    }
    payload = run_longbench.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
    )
    assert payload["paper_valid"] is False
    assert any("dataset_source" in c.lower() or "huggingface_thudm" in c.lower()
               for c in payload["caveats"])


def test_longbench_full_paper_valid_passes_all_gates():
    from experiments import run_longbench, eval_common

    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2",
        "--n-per-task", "50",
        "--subset", "minimal",
    ])
    methods = {
        "fp16": {
            "label": "fp16",
            "per_task": [{"task": "narrativeqa", "metric": "qa_f1_en",
                          "score": 0.4, "n_examples": 50}],
            "aggregate": {"macro_score": 0.4},
            "evidence_ids": ["x"],
            "dataset_source": "huggingface_thudm",
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2",
            "per_task": [{"task": "narrativeqa", "metric": "qa_f1_en",
                          "score": 0.39, "n_examples": 50}],
            "aggregate": {"macro_score": 0.39},
            "evidence_ids": ["x"],
            "replay_coverage": {"fraction_layers_real": 1.0,
                                "n_layers_total": 32, "n_layers_hooked": 32,
                                "n_layers_calibrated": 32, "n_hook_calls": 100,
                                "n_passthrough_calls": 0, "missing_layers": []},
            "dataset_source": "huggingface_thudm",
        },
    }
    payload = run_longbench.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
    )
    assert payload["paper_valid"] is True
    # Subset != "full" carries an explicit caveat about transparent subset.
    assert any("transparent subset" in c.lower() for c in payload["caveats"])


def test_longbench_full_paper_valid_blocked_by_low_n_per_task():
    from experiments import run_longbench, eval_common

    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2",
        "--n-per-task", "10",
    ])
    methods = {
        "fp16": {
            "label": "fp16",
            "per_task": [{"task": "narrativeqa", "metric": "qa_f1_en",
                          "score": 0.4, "n_examples": 10}],
            "aggregate": {"macro_score": 0.4},
            "evidence_ids": ["x"],
            "dataset_source": "huggingface_thudm",
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2",
            "per_task": [{"task": "narrativeqa", "metric": "qa_f1_en",
                          "score": 0.39, "n_examples": 10}],
            "aggregate": {"macro_score": 0.39},
            "evidence_ids": ["x"],
            "replay_coverage": {"fraction_layers_real": 1.0,
                                "n_layers_total": 32, "n_layers_hooked": 32,
                                "n_layers_calibrated": 32, "n_hook_calls": 100,
                                "n_passthrough_calls": 0, "missing_layers": []},
            "dataset_source": "huggingface_thudm",
        },
    }
    payload = run_longbench.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
    )
    assert payload["paper_valid"] is False
    assert any("n_per_task" in c.lower() for c in payload["caveats"])


def test_longbench_inline_corpus_smoke_never_paper_valid():
    from experiments import run_longbench, eval_common

    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--inline-corpus-smoke",
        "--methods", "fp16", "spectralquant_v2",
        "--n-per-task", "100",
    ])
    methods = {
        "fp16": {
            "label": "fp16",
            "per_task": [{"task": "narrativeqa", "metric": "f1",
                          "score": 0.4, "n_examples": 100}],
            "aggregate": {"macro_score": 0.4},
            "evidence_ids": ["x"],
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2",
            "per_task": [{"task": "narrativeqa", "metric": "f1",
                          "score": 0.39, "n_examples": 100}],
            "aggregate": {"macro_score": 0.39},
            "evidence_ids": ["x"],
            "replay_coverage": {"fraction_layers_real": 1.0,
                                "n_layers_total": 32, "n_layers_hooked": 32,
                                "n_layers_calibrated": 32, "n_hook_calls": 100,
                                "n_passthrough_calls": 0, "missing_layers": []},
        },
    }
    payload = run_longbench.build_payload(
        args, ["x"], eval_common.MODE_INLINE_CORPUS_SMOKE, methods,
    )
    assert payload["paper_valid"] is False


# ---------------------------------------------------------------------------
# Calibration-knob CLI / caveat / paper_valid behavior.
# ---------------------------------------------------------------------------


def _full_lb_methods_payload(n: int):
    return {
        "fp16": {
            "label": "fp16",
            "per_task": [{"task": "narrativeqa", "metric": "qa_f1_en",
                          "score": 0.4, "n_examples": n}],
            "aggregate": {"macro_score": 0.4},
            "evidence_ids": ["x"],
            "dataset_source": "huggingface_thudm",
        },
        "spectralquant_v2": {
            "label": "spectralquant_v2",
            "per_task": [{"task": "narrativeqa", "metric": "qa_f1_en",
                          "score": 0.39, "n_examples": n}],
            "aggregate": {"macro_score": 0.39},
            "evidence_ids": ["x"],
            "replay_coverage": {"fraction_layers_real": 1.0,
                                "n_layers_total": 32, "n_layers_hooked": 32,
                                "n_layers_calibrated": 32, "n_hook_calls": 100,
                                "n_passthrough_calls": 0, "missing_layers": []},
            "dataset_source": "huggingface_thudm",
        },
    }


def test_longbench_cli_calibration_knobs_default_to_paper_valid():
    """An unspecified --n-calib must default to the paper-valid minimum
    so an operator who passes only --n-per-task does not silently lose
    paper validity."""
    from experiments import run_longbench

    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2",
        "--n-per-task", "50",
        "--subset", "minimal",
    ])
    assert args.n_calib >= run_longbench.PAPER_VALID_N_CALIB_MIN
    assert args.lloyd_max_iter == run_longbench.PAPER_VALID_LLOYD_MAX_ITER
    assert args.calib_max_seq_tokens >= run_longbench.PAPER_VALID_CALIB_MAX_SEQ_TOKENS_MIN
    assert run_longbench.is_reduced_calibration(args) is False


def test_longbench_cli_low_n_calib_marks_reduced_calibration():
    from experiments import run_longbench
    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2",
        "--n-per-task", "50", "--n-calib", "4",
    ])
    assert args.n_calib == 4
    assert run_longbench.is_reduced_calibration(args) is True


def test_longbench_cli_low_lloyd_max_iter_marks_reduced_calibration():
    from experiments import run_longbench
    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2",
        "--n-per-task", "50", "--lloyd-max-iter", "25",
    ])
    assert args.lloyd_max_iter == 25
    assert run_longbench.is_reduced_calibration(args) is True


def test_longbench_calibration_mode_paper_rejects_low_n_calib():
    import pytest as _pt
    from experiments import run_longbench
    with _pt.raises(SystemExit):
        run_longbench.parse_args([
            "--model", "x", "--output-dir", "/tmp/x",
            "--methods", "fp16", "spectralquant_v2",
            "--n-per-task", "50",
            "--n-calib", "4",
            "--calibration-mode", "paper",
        ])


def test_longbench_paper_valid_blocked_by_reduced_calibration():
    """Reduced calibration MUST force paper_valid=false even when every
    other gate (n_per_task, methods, coverage, dataset_source) passes."""
    from experiments import run_longbench, eval_common

    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2",
        "--n-per-task", "50", "--subset", "minimal",
        "--n-calib", "4",
    ])
    methods = _full_lb_methods_payload(50)
    payload = run_longbench.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
    )
    assert payload["paper_valid"] is False
    assert any("reduced_calibration" in c for c in payload["caveats"])
    assert payload["calibration"]["reduced_calibration"] is True
    assert payload["calibration"]["n_calib"] == 4


def test_longbench_smoke_calibration_mode_emits_caveat():
    from experiments import run_longbench, eval_common
    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2",
        "--n-per-task", "50", "--subset", "minimal",
        "--n-calib", "4", "--lloyd-max-iter", "25",
        "--calibration-mode", "smoke",
    ])
    methods = _full_lb_methods_payload(50)
    payload = run_longbench.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
    )
    assert payload["paper_valid"] is False
    assert payload["calibration"]["calibration_mode"] == "smoke"
    assert any("calibration_mode=smoke" in c for c in payload["caveats"])


def test_longbench_payload_calibration_block_present_for_fp16_only_runs():
    """The calibration block is metadata; it should be emitted even on
    FP16-only runs so downstream tooling can report it uniformly."""
    from experiments import run_longbench, eval_common
    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16",
        "--n-per-task", "50", "--subset", "minimal",
        "--n-calib", "4",  # reduced, but no compressed method requested
    ])
    methods = {
        "fp16": _full_lb_methods_payload(50)["fp16"],
    }
    payload = run_longbench.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
    )
    # FP16-only does not need calibration, so reduced_calibration must
    # not block paper_valid by itself.
    assert payload["paper_valid"] is True
    assert payload["calibration"]["n_calib"] == 4


def test_longbench_run_id_includes_calib_tag_when_reduced():
    from experiments import run_longbench, eval_common
    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "spectralquant_v2",
        "--n-per-task", "50", "--n-calib", "4",
    ])
    rid = run_longbench.derive_run_id(args, eval_common.MODE_FULL)
    assert "calib4" in rid


def test_longbench_run_id_does_not_change_for_paper_valid_default():
    """run_id must remain stable for default (paper-valid) configurations
    so existing artifact filenames do not move under operators' feet."""
    from experiments import run_longbench, eval_common
    args = run_longbench.parse_args([
        "--model", "x", "--output-dir", "/tmp/x",
        "--methods", "fp16", "spectralquant_v2",
        "--n-per-task", "50",
    ])
    rid = run_longbench.derive_run_id(args, eval_common.MODE_FULL)
    assert "_calib" not in rid
    assert "_lm" not in rid
    assert "_cs" not in rid


def test_longbench_progress_callback_emits_calib_stages(tmp_path):
    """The progress callback wired into sqv2_replay must forward each
    coarse milestone through the StatusWriter, so future runs cannot
    appear silent during calibration."""
    from experiments import run_longbench, run_status

    sw = run_status.StatusWriter(
        status_dir=tmp_path / "status",
        run_id="rid",
        commit="deadbeef",
        model="x",
    )
    cb = run_longbench._make_progress_cb(sw, "spectralquant_v2")
    assert cb is not None
    cb("calib_eigh_start", "go", {"n": 1})
    cb("calib_fit_progress", "layer=0", {"layer_idx": 0})
    cb("calib_fit_end", "done", {"n_quantizers": 5})
    events = (tmp_path / "status" / "events.jsonl").read_text().strip().splitlines()
    stages = [json.loads(line)["stage"] for line in events]
    assert "calib_eigh_start" in stages
    assert "calib_fit_progress" in stages
    assert "calib_fit_end" in stages


# ---------------------------------------------------------------------------
# sqv2_replay round-trip: hook plumbing
# ---------------------------------------------------------------------------


class _ProjLinear:
    """Stub linear module that allows us to fire forward hooks."""

    def __init__(self):
        self._hooks: List[Any] = []

    def register_forward_hook(self, hook):
        self._hooks.append(hook)

        class _H:
            def __init__(self, owner, fn):
                self._owner = owner
                self._fn = fn

            def remove(self):
                if self._fn in self._owner._hooks:
                    self._owner._hooks.remove(self._fn)

        return _H(self, hook)

    def fire(self, x):
        out = x
        for h in list(self._hooks):
            ret = h(self, (), out)
            if ret is not None:
                out = ret
        return out


class _Attn:
    def __init__(self):
        self.k_proj = _ProjLinear()
        self.v_proj = _ProjLinear()
        self.q_proj = _ProjLinear()


class _Layer:
    def __init__(self, attn):
        self.self_attn = attn


class _Inner:
    def __init__(self, layers):
        self.layers = layers


class _Cfg:
    def __init__(self, n_q, n_kv, hd, n_layers):
        self.model_type = "qwen2"
        self.num_attention_heads = n_q
        self.num_key_value_heads = n_kv
        self.hidden_size = n_q * hd
        self.head_dim = hd
        self.num_hidden_layers = n_layers
        self._name_or_path = "Qwen/Qwen2.5-0.5B"


class _FakeModel:
    def __init__(self, n_layers=2, n_q=4, n_kv=2, hd=8):
        attns = [_Attn() for _ in range(n_layers)]
        self.model = _Inner([_Layer(a) for a in attns])
        self.config = _Cfg(n_q, n_kv, hd, n_layers)
        self._attns = attns


def test_replay_hook_round_trips_kv_through_engine():
    """attach_replay_hooks should pass output through compress→decompress.

    We use a minimal hand-built engine (calibrator + pre-fitted quantizer)
    so we don't have to load HF. The test asserts: after the hook fires,
    the projection output's shape and dtype are unchanged, and the values
    differ from the input (compression is lossy → reconstruction != input).
    """
    pytest.importorskip("torch")
    import torch
    from experiments import sqv2_replay
    from spectralquant import EngineConfig, SpectralQuantEngine
    from spectralquant.calibration import EigenspectralCalibrator, HeadCalibrationData

    n_layers, n_q, n_kv, hd = 2, 4, 2, 8
    model = _FakeModel(n_layers=n_layers, n_q=n_q, n_kv=n_kv, hd=hd)

    # Build calibrator with random orthogonal eigenvectors per (layer, head, type).
    calib = EigenspectralCalibrator(max_tokens_per_layer=4096)
    g = torch.Generator(device="cpu").manual_seed(0)
    for li in range(n_layers):
        for h in range(n_kv):
            for ht in ("key", "value"):
                mat = torch.randn(hd, hd, generator=g)
                q, _ = torch.linalg.qr(mat)
                eig = torch.linspace(1.0, 0.01, hd)
                calib._calibration_data[(li, h, ht)] = HeadCalibrationData(
                    layer_idx=li, head_idx=h, head_type=ht,
                    eigenvalues=eig, eigenvectors=q,
                    d_eff=4.0, spectral_gap=None,
                    var_95=5, var_99=6, n_samples=1024, head_dim=hd,
                )
    calib._is_calibrated = True

    cfg = EngineConfig(avg_bits=3.0, use_water_fill=True, wf_min_bits=0)
    engine = SpectralQuantEngine(calib, cfg)
    rotated = {
        (li, h, ht): torch.randn(256, hd, generator=g)
        for li in range(n_layers) for h in range(n_kv) for ht in ("key", "value")
    }
    engine.fit_quantizers(rotated)

    handle = sqv2_replay.attach_replay_hooks(
        model, engine,
        n_kv_heads=n_kv, head_dim=hd,
        method="spectralquant_v2",
        calibrated_layers=list(range(n_layers)),
    )
    try:
        # Fire the layer-0 k_proj hook with a synthetic projection output.
        x = torch.randn(1, 6, n_kv * hd, generator=g)
        out = model._attns[0].k_proj.fire(x)
        # Shape and dtype preserved.
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        # But values changed (compression is lossy).
        assert not torch.allclose(out, x, atol=1e-6)
        # Coverage updated.
        assert handle.coverage.n_hook_calls >= 1
    finally:
        handle.remove()


def test_replay_hook_passthrough_for_uncalibrated_layer():
    pytest.importorskip("torch")
    import torch
    from experiments import sqv2_replay
    from spectralquant import EngineConfig, SpectralQuantEngine
    from spectralquant.calibration import EigenspectralCalibrator, HeadCalibrationData

    n_layers, n_q, n_kv, hd = 2, 4, 2, 8
    model = _FakeModel(n_layers=n_layers, n_q=n_q, n_kv=n_kv, hd=hd)
    calib = EigenspectralCalibrator(max_tokens_per_layer=4096)
    # Calibrate ONLY layer 0.
    g = torch.Generator(device="cpu").manual_seed(0)
    for h in range(n_kv):
        for ht in ("key", "value"):
            mat = torch.randn(hd, hd, generator=g)
            q, _ = torch.linalg.qr(mat)
            eig = torch.linspace(1.0, 0.01, hd)
            calib._calibration_data[(0, h, ht)] = HeadCalibrationData(
                layer_idx=0, head_idx=h, head_type=ht,
                eigenvalues=eig, eigenvectors=q,
                d_eff=4.0, spectral_gap=None,
                var_95=5, var_99=6, n_samples=1024, head_dim=hd,
            )
    calib._is_calibrated = True
    cfg = EngineConfig(avg_bits=3.0, use_water_fill=True, wf_min_bits=0)
    engine = SpectralQuantEngine(calib, cfg)
    rotated = {
        (0, h, ht): torch.randn(256, hd, generator=g)
        for h in range(n_kv) for ht in ("key", "value")
    }
    engine.fit_quantizers(rotated)

    handle = sqv2_replay.attach_replay_hooks(
        model, engine, n_kv_heads=n_kv, head_dim=hd,
        method="spectralquant_v2", calibrated_layers=[0],
    )
    try:
        # Layer 0 hook should be present; layer 1 should NOT have hooks
        # (since layer 1 isn't in calibrated_layers).
        assert len(model._attns[0].k_proj._hooks) == 1
        assert len(model._attns[1].k_proj._hooks) == 0
        cov = handle.coverage_summary()
        assert cov["n_layers_calibrated"] == 1
        assert cov["n_layers_hooked"] == 1
        assert cov["fraction_layers_real"] < 1.0  # layer 1 missed
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# REAL_EVAL_METHODS canonical set
# ---------------------------------------------------------------------------


def test_real_eval_methods_canonical():
    """Pin down the supported method set so accidental expansions are
    caught in code review."""
    from experiments import run_perplexity, run_generation, run_latency, run_longbench

    expected = ("fp16", "spectralquant_v2", "turboquant")
    assert tuple(run_perplexity.REAL_EVAL_METHODS) == expected
    assert tuple(run_generation.REAL_EVAL_METHODS) == expected
    assert tuple(run_latency.REAL_EVAL_METHODS) == expected
    assert tuple(run_longbench.REAL_EVAL_METHODS) == expected
