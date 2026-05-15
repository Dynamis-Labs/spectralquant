# SpectralQuant v2 — Full-Path Matrix Evidence Summary

This document is the canonical, repo-tracked summary of the four
*paper-valid full-path* three-way runs that have completed on Modal at
the time of writing. It is intentionally narrow: it records what has
actually been run on real HF weights against WikiText-103, and is not a
substitute for the full headline matrix described in
`docs/execution_audit_and_modal_runbook.md` §8.

Every row below is a single sliced benchmark configuration
(WikiText-103, n_calib=32, n_eval=8, 8 sampled layers, seed 42); none of
these runs is on its own a final paper claim. Multi-seed runs,
perplexity, latency, NIAH, LongBench, and the official-Google-TurboQuant
comparison are still pending — see §4 below.

The artifacts referenced here have been downloaded to a local
`modal_artifacts/` staging directory off-repo for inspection. Their
authoritative location is the Modal volume `spectralquant-v2-results`
under `/results/three_way/`. All four files validate against
`schemas/three_way_result.schema.json` (cross-resolved with
`schemas/accounting.schema.json`) using `experiments/run_three_way.py
::_validate_payload`.

## 1. Runs included

All four runs share the same harness commit and run configuration. The
commit is the audit anchor for this evidence summary.

| Field | Value |
|---|---|
| Repo | `niashwin/spectralquant-full` (private; renamed from `niashwin/spectralquant-v2`. The original-name value is the one stored in the JSON `repo` fields on disk and is preserved verbatim for audit traceability). |
| Commit | `abcb09197998cc027df688abceae5fb81cfcd31d` |
| Mode | `full` (real HF weights, real WikiText-103, not synthetic / not inline-corpus) |
| Seed | 42 |
| Calibration corpus | WikiText-103 |
| Eval corpus | WikiText-103 (disjoint from calibration) |
| `n_calib` | 32 |
| `n_eval` | 8 |
| `max_calib_tokens` | 384 |
| `n_layers_sample` | 8 |
| `qjl_projections` | 64 |
| Engine | `SpectralQuantEngine` (canonical pure-Python; spec §11.2) |
| Hardware | NVIDIA H200 on Modal |
| Software | `torch=2.11.0+cu130`, `transformers=5.7.0`, `datasets=4.8.5`, Python 3.12.1 |
| `paper_valid` | `true` for all four artifacts |

## 2. Artifact manifest

Each artifact JSON validates against
`schemas/three_way_result.schema.json` (with the accounting schema
cross-resolved from `schemas/accounting.schema.json`).

| Evidence ID | Model | Bits | Modal volume path | Local staging path | Run-internal `evidence_ids` |
|---|---|---:|---|---|---|
| `RUN-THREEWAY-MISTRAL-5BIT` | `mistralai/Mistral-7B-v0.3` | 5 | `spectralquant-v2-results:/results/three_way/Mistral-7B-v0.3_b5_calib32_eval8_seed42.json` | `modal_artifacts/mistral7b_full_b5/Mistral-7B-v0.3_b5_calib32_eval8_seed42.json` | `["RUN-THREEWAY-001"]` |
| `RUN-THREEWAY-MISTRAL-3BIT` | `mistralai/Mistral-7B-v0.3` | 3 | `spectralquant-v2-results:/results/three_way/Mistral-7B-v0.3_b3_calib32_eval8_seed42.json` | `modal_artifacts/mistral7b_full_b3/Mistral-7B-v0.3_b3_calib32_eval8_seed42.json` | `["RUN-THREEWAY-001"]` |
| `RUN-THREEWAY-MISTRAL-2BIT` | `mistralai/Mistral-7B-v0.3` | 2 | `spectralquant-v2-results:/results/three_way/Mistral-7B-v0.3_b2_calib32_eval8_seed42.json` | `modal_artifacts/mistral7b_full_b2/Mistral-7B-v0.3_b2_calib32_eval8_seed42.json` | `["RUN-THREEWAY-001"]` |
| `RUN-THREEWAY-QWEN-3BIT` | `Qwen/Qwen2.5-7B` | 3 | `spectralquant-v2-results:/results/three_way/Qwen2.5-7B_b3_calib32_eval8_seed42.json` | `modal_artifacts/qwen7b_full_b3/Qwen2.5-7B_b3_calib32_eval8_seed42.json` | `["RUN-THREEWAY-001"]` |

The mapping between repo-level evidence IDs (`RUN-THREEWAY-MISTRAL-3BIT`,
etc.) and the run-internal `evidence_ids[]` field (`RUN-THREEWAY-001`)
is:

- The run-internal field is a per-run namespace produced by
  `experiments/run_three_way.py`; it is the same string for every
  full-path run because the script does not yet stamp model/bit-specific
  IDs.
- The repo-level evidence IDs above are the ones cited in
  `docs/evidence_catalog.{md,json}` and in any v2 paper draft.

The on-Modal layer-sample lists are also recorded inside each JSON
under `config.layer_sample`:

- Mistral runs: `[0, 4, 8, 12, 16, 20, 24, 28]` (every 4th layer of 32).
- Qwen run: `[0, 3, 6, 9, 12, 15, 18, 21]` (every 3rd layer of 28).

## 3. Headline numbers (extracted from the JSONs)

Means and standard deviations are taken verbatim from each artifact's
`methods.<m>.attn_cosine_mean` / `attn_cosine_std`. Compression ratios
are taken verbatim from
`methods.<m>.compression_accounting.compression_ratio`. The TurboQuant
arm in every JSON carries `methods.turboquant.label = "local"` per
spec §13 and per V1-GAP-012.

| Model | Bits | TQ cos mean | TQ cos std | SQ v1 cos mean | SQ v1 cos std | SQ v2 cos mean | SQ v2 cos std | TQ ratio | SQ v2 ratio | Δ(SQ v2 − TQ) | Δ(SQ v2 − v1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mistral-7B-v0.3 | 5 | 0.6556 | 0.0886 | 0.9404 | 0.0398 | 0.9421 | 0.0390 | 3.0843 | 3.0820 | +0.2865 | +0.0017 |
| Mistral-7B-v0.3 | 3 | 0.6263 | 0.0920 | 0.9329 | 0.0369 | 0.9327 | 0.0370 | 5.0196 | 5.0135 | +0.3064 | −0.0002 |
| Mistral-7B-v0.3 | 2 | 0.6495 | 0.0712 | 0.9035 | 0.0422 | 0.9213 | 0.0381 | 7.3143 | 7.3012 | +0.2717 | +0.0177 |
| Qwen2.5-7B      | 3 | 0.3986 | 0.1491 | 0.7724 | 0.1055 | 0.7786 | 0.1104 | 5.0196 | 5.0135 | +0.3800 | +0.0062 |

Cosine values, bits, and ratios above are extracted verbatim from the
respective artifact JSONs. Δ columns are derived; if any downstream
table needs a Δ to higher precision than is shown here, regenerate from
the JSON and do not transcribe a rounded Δ from this table.

## 4. Interpretation guardrails

These four runs are *paper-valid harness rows for a sliced
configuration*. Treat them accordingly:

1. **Sliced full-path, not full sweep.** Each run uses 32 calibration
   sequences, 8 eval sequences, and 8 sampled layers per model. Per-head
   cosine min/max distributions, full-layer cosine distributions, and
   multi-seed CIs are not yet available; the sample sizes are too small
   to support a Wilcoxon p-value or a "universal architecture" claim.
2. **Single seed (42).** None of the multi-seed obligations from
   V1-GAP-001 are unblocked by these four runs.
3. **TurboQuant arm is the local reimplementation.** Every JSON sets
   `methods.turboquant.label = "local"`. V1-GAP-012 ("beats official
   Google TurboQuant") is **not** unblocked by these runs.
4. **Compression ratios are method-derived.** The ratios above come
   from `src/spectralquant/accounting.py::CompressionAccounting`. They
   are *not* the v1 paper's 5.95x at b=3; that headline (V1-GAP-014) is
   still flagged and any paper sentence quoting 5.95x must be backed by
   a reconciled `results/accounting_audit/` JSON, not by this evidence
   summary.
5. **No perplexity / NIAH / LongBench / latency claims unblocked.**
   These four artifacts contain attention-output cosine and compression
   accounting only. The corresponding v1 gaps (V1-GAP-003, 004b, 008, 009)
   remain in force.
6. **Inline-corpus and synthetic-smoke runs are excluded.** Files with
   `mode = "inline-corpus-smoke"` or `mode = "synthetic-smoke"` (or with
   `paper_valid = false`) must never be cited as evidence — see
   `docs/execution_audit_and_modal_runbook.md` §6.8a and
   `docs/modal_safety_protocol.md` §10.
7. **One-line summary for the paper.** "On Mistral-7B-v0.3 (b ∈ {2,3,5})
   and Qwen2.5-7B (b = 3), with WikiText-103 calibration (n=32) and
   disjoint eval (n=8) over 8 sampled layers, SpectralQuant v2 attention
   cosine exceeds the *local* TurboQuant baseline by ≈0.27–0.38 mean
   cosine; v2 matches v1 at b=3 and is +0.018 mean cosine over v1 at
   b=2 on Mistral-7B-v0.3." Anything stronger than this requires more
   runs.

## 5. How to reproduce the runs

The exact command line for each run was committed by
`experiments/run_three_way.py` into the artifact's top-level `command`
field. To re-launch the same configuration on Modal, see
`docs/execution_audit_and_modal_runbook.md` §7.6 step 4 (Mistral) and
step 5 (Qwen). The `--seed 42` flag plus the WikiText-103 disjoint-eval
contract are what make the runs comparable across bit widths.

The on-Modal results volume is `spectralquant-v2-results`. To pull the
JSONs back to local for inspection without rerunning Modal:

```bash
modal volume get spectralquant-v2-results /results/three_way/<filename> ./local_results/
```

## 6. Open work this evidence does not cover

| Item | Status | Why this evidence does not cover it |
|---|---|---|
| Multi-seed runs (≥ 5, ideally 10) on Mistral b=3 | Pending | Single seed only here. |
| Per-head min-cosine catastrophic-failure distribution | Pending | JSONs aggregate to per-layer mean only. |
| WikiText-2 / C4 perplexity at b=3 | Pending | These runs are attention-cosine only. |
| LongBench (n ≥ 50 / task) | Pending | Not in scope of `run_three_way.py`. |
| NIAH at 4k / 8k / 16k / 32k | Pending | Not in scope. |
| End-to-end decode latency, BS={1,4,8,16} | Pending | Latency claims still bound by V1-GAP-003. |
| Calibration time / amortization curves | Pending | Calibration-time claims still bound by V1-GAP-007. |
| Three-draw calibration stability | Pending | Replaces V1-GAP-005; not run yet. |
| Official Google TurboQuant comparison | Pending | V1-GAP-012 still in force. |
| Compression-accounting reconciliation | Pending | V1-GAP-014; needs `results/accounting_audit/` JSONs. |

## 7. Provenance and security

- The artifact JSONs do not contain any HF token, Modal token, or other
  secret. They record `software.{torch,transformers,datasets,python}`
  versions, model classes, and the gated-model revision SHAs only.
  Inspect any new artifact for accidental secret leakage before adding
  it to `docs/evidence_catalog.{md,json}`.
- The only credentials needed to *reproduce* the runs are the operator's
  Hugging Face token (for the gated `mistralai/Mistral-7B-v0.3` weights)
  and Modal CLI tokens. Both are managed via the procedures in
  `docs/modal_safety_protocol.md` §3 — never via CLI flags, never echoed
  to the result JSON, and never committed to the repo.
