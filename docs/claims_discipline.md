# SpectralQuant Claims Discipline

This document is the single rulebook for what the SpectralQuant repository and reports are allowed to say. It is derived from `docs/spectralquant_v2_technical_spec.md` (sections 6, 8, 16, and 18) and from the evidence audit in `docs/evidence_catalog.md`. It exists so that no figure caption, README sentence, or paper paragraph can quietly upgrade a measured local result into a universal claim.

## 0. Consolidated naming discipline (added 2026-05-01)

The public method is one program: **SpectralQuant**. Public-facing material — the consolidated unrestricted technical report at `paper_output_consolidated/spectralquant_unrestricted_paper.tex`, the README, figure captions, blog posts, slide decks, and any external description — must refer to **SpectralQuant**, not "SpectralQuant v1" or "SpectralQuant v2".

The labels `v1` and `v2` are repository-side development tags. They are allowed only in:

1. The development-history section (§11) of the consolidated paper, where the two evidence layers are documented as part of the program's provenance.
2. The traceability table that anchors empirical claims back to specific JSON artifacts.
3. The JSON method key `spectralquant_v2` and stable evidence-catalog identifiers (`V1-RESULT-*`, `V1-IMPL-*`, `V1-GAP-*`, `V2-SPEC-*`, `RUN-*`), preserved verbatim for tooling stability and audit cross-referencing.
4. Historical repository/directory/file paths (`paper_output_v2/`, `docs/spectralquant_v2_technical_spec.md`, `experiments/sqv2_replay.py`, the Modal volume name `spectralquant-v2-results`, and the historical repository name `niashwin/spectralquant-v2`), preserved as filesystem and provenance artifacts. The canonical public-facing repository is now `niashwin/spectralquant-full`; references in archived JSON `repo` fields and historical docs to `niashwin/spectralquant-v2` are intentional traceability and not to be edited.
5. This § 0 and `docs/consolidated_spectralquant_inventory.md`.

When two settings of the algorithm need to be distinguished within a single sentence, the canonical vocabulary is **uniform-allocation special case** (the configuration introduced in the initial evidence layer; corresponds to v1 in repository labels) and **water-filled allocation** (the configuration that became default through the rate–distortion derivation in the expanded evidence layer; corresponds to v2 in repository labels). The water-filled allocation is the default, and the uniform special case is the comparison baseline that isolates the contribution of the allocation rule from the contribution of the calibrated rotation.

The two evidence layers are named **initial evidence layer** (released through `Dynamis-Labs/spectralquant`; original public manuscript "3% Is All You Need", `paper_output/spectralquant.tex`) and **expanded evidence layer** (paper-valid next-stage runs in `results/v3/modal/` and the three-way matrix in `results/three_way/`). When neutral language is required the layers may be referred to simply as "the initial layer" and "the expanded layer".

Failure modes to watch for:

- "SpectralQuant v2 beats TurboQuant" — wrong; should be "SpectralQuant beats TurboQuant".
- "SpectralQuant v2 improves on SpectralQuant v1" — wrong; should be "the water-filled allocation improves on the uniform-allocation special case at b=2".
- "v2 is the new method" — wrong; the method is SpectralQuant; the water-filled allocation is the default configuration.

> Operational counterpart: `docs/execution_audit_and_modal_runbook.md` is the running execution and audit log (what is done, what remains, what tests have run, what to run on Modal, target reproduction numbers). Update it whenever a claim's safe/blocked status changes here.

The rule is simple: **a claim is safe only if every word of it is backed by a JSON artifact whose evidence ID appears in `docs/evidence_catalog.json`, and every gap that touches the claim has been resolved or demoted in writing.**

## 1. Safe headline (after Phase 6 reproduction)

The strongest sentence v2 may print, after the Phase 6 three-way runs complete and pass schema validation, is exactly this:

> On the tested Mistral-7B-v0.3 and Qwen2.5-7B attention-output benchmarks, SpectralQuant v2 Pareto-improves over the local TurboQuant baseline at aggressive KV-cache compression settings, with the largest gains at low bit budgets and heavier GQA.

Every word of that sentence is load-bearing:

- "tested … benchmarks" — restricts the claim to attention-output cosine on WikiText-103 calibration/eval.
- "local TurboQuant baseline" — this repo's reimplementation, not the official Google code (V1-GAP-012).
- "Pareto-improves" — both quality and compression ratio improve simultaneously, *as derived from* `src/spectralquant/accounting.py` (V1-GAP-014).
- "aggressive … low bit budgets" — Mistral 2/3-bit, Qwen 3-bit; not 5-bit and not "every operating point".
- "heavier GQA" — comparing Mistral 4:1 to Qwen 7:1 only; not a generalization across all GQA ratios.

## 2. Safe claims after target reproduction

After Phase 6 produces clean three-way JSONs and they pass `tests/test_result_schema.py`, v2 may claim:

| Dimension | Claim wording allowed |
|---|---|
| Attention-output cosine | "SQ v2 attention-output cosine exceeds the local TurboQuant baseline on Mistral-7B-v0.3 at 2/3/5-bit and Qwen2.5-7B at 3-bit." |
| v1 → v2 quality | "SQ v2 attention-output cosine exceeds SQ v1 on the same operating points." |
| Compression at fixed quality | "At matched cosine, SQ v2 reaches a higher compression ratio than the local TurboQuant baseline (numbers from `src/spectralquant/accounting.py`)." |
| Catastrophic head failure | "On low-bit Mistral, the local TurboQuant baseline shows per-head cosine outliers below [threshold] that SQ v2 does not, in the tested layer sample." (only if the per-head distribution is included in the JSON). |
| Calibration footprint | "Calibration uses n_calib=32 sequences at max_tokens=384 from WikiText-103 per spec §13.3." |

Each claim must carry the matching evidence IDs (e.g., `RUN-THREEWAY-MISTRAL-3BIT`) inline or in an appendix-style mapping (`docs/spectralquant_v2_technical_spec.md` §16).

## 3. Blocked claims

These claims are NOT permitted in v2 report drafts, even if they appear in the v1 README or v1 paper. Each is blocked by a specific gap or by missing experiments.

| Blocked claim | Why blocked | Unblock requirement |
|---|---|---|
| "Beats TurboQuant in every way possible" | Most dimensions unmeasured; v1 wording (spec §18). | Replace with the measured-dimensions table below. |
| "Beats the official TurboQuant implementation" | Local reimplementation only; phase 0 reproduction status unresolved. | V1-GAP-012 — run against official Google TurboQuant. |
| "Compression-neutral perplexity" | Identical 13-digit PPL across fp16/TQ/SQ in `v3_perplexity_v2.json` is suspicious. | V1-GAP-004b — re-run with longer context and per-step deltas. |
| "LongBench improvement" | n=5/task. | V1-GAP-008 — n ≥ 50 per task. |
| "NIAH 10/10" | NIAH artifacts limited. | V1-GAP-009 — re-run with documented NIAH protocol. |
| "10-seed CI; Wilcoxon p=0.031" | Only 5 seeds on disk. | V1-GAP-001 — run the missing 5 seeds OR re-state as 5 seeds and recompute statistics. |
| "4 models main results" | 3 Qwen rows in `all_models.json`. | V1-GAP-002 — restrict wording to 3 models OR add the 4th. |
| "SQ faster than TQ at 512 tokens" | Two latency files contradict. | V1-GAP-003 — pick canonical methodology and re-run. |
| "d_eff/head_dim ≈ 3–4% (universal)" | Two d_eff conventions; "universal" not measured. | V1-GAP-004 — restrict to "≈ 3–4% under the normalized-keys convention on the 5 tested models" and remove "universal". |
| "15-second calibration" | Closest on-disk timing is 31 s. | V1-GAP-007 — drop framing or re-time on documented hardware. |
| "Gemma 2-9B" as a measured architecture | Row is HF 403. | V1-GAP-011 — acquire access and re-run. |
| Any compression ratio not produced by `accounting.py` | Spec §10 derivation does not match 5.95×. | V1-GAP-014 — derive every ratio from stored bits via `accounting.py`. |
| "All architectures benefit" | Only 5 models tested with normalized-keys convention. | Add a broader model suite. |
| "End-to-end serving speedup" | No decode benchmark on disk. | Spec §13.7 — run the decode/throughput experiments. |
| "Production-ready" | No deployment, no kernel benchmarks beyond `results/kernel/`. | Spec §6 — explicitly forbidden until measured. |

## 4. Validation gates

Before any v2 commit can claim publication-readiness, these gates must pass.

### Gate G1 — Evidence catalog schema

`docs/evidence_catalog.json` must validate against `schemas/evidence_catalog.schema.json`. Enforced by `tests/test_result_schema.py` (this milestone).

### Gate G2 — Per-result schema

Every JSON in `results/three_way/`, `results/waterfill_ablation/`, `results/accounting_audit/` must validate against `schemas/three_way_result.schema.json` or `schemas/accounting.schema.json` as applicable.

### Gate G3 — Compression accounting parity

Every reported ratio is recomputed from stored bits via `src/spectralquant/accounting.py`. Hard-coded ratios are forbidden. Tests in `tests/test_accounting.py` per spec §10.

The audit mechanism is `accounting.check_headline_ratio(...)`, which compares a `CompressionAccounting` object's computed `compression_ratio` against a stated headline target and returns a `HeadlineRatioCheck` whose `matches` field is `False` (with a human-readable `diagnostic`) when the bits-derived value disagrees with the headline. The current foundation tests assert that the simple spec §10 SpectralQuant formula does NOT match the 5.95× headline at `(avg_bits=3, d_eff=3, head_dim=128)` — this is V1-GAP-014 made executable, not papered over. Any future engine path that stores its actual K/V bit components must use `accounting.spectralquant_accounting(...)` (the flexible builder) rather than the simple-formula helper, so the reported ratio matches the bits actually written.

### Gate G4 — v1 backward compatibility

`use_water_fill=False` reproduces v1 allocation byte-for-byte; v2's total semantic MSE bit budget equals v1's; v2's selective-QJL dimension equals v1's. Tests in `tests/test_engine_v2.py`.

### Gate G5 — Water-filling unit tests

`tests/test_waterfill.py` covers all 10 cases listed in spec §12.1.

### Gate G6 — Quantization tests

`tests/test_v2_quantization.py` covers all 8 cases in spec §12.2.

### Gate G7 — Calibration tests

`tests/test_calibration_v2.py` covers all 8 cases in spec §12.3.

### Gate G8 — TurboQuant baseline labeled

Every reference to "TurboQuant" in the v2 report or README is qualified as the local baseline implementation unless run against the official Google code. Enforced by review per spec §17 Pass 4.

### Gate G9 — Eight-pass reflection

Spec §17 — claim audit, math audit, accounting audit, baseline audit, reproducibility audit, statistical audit, reader audit, claims-discipline audit. Documented in the report appendix.

### Gate G10 — Negative results explained

Every "negative" file in `results/comparison/` (especially `comparison_results.json`, V1-GAP-002b) is either reconciled with the headline configuration or explicitly excluded with a written reason. No silent omissions.

## 5. Dimensional matrix (which claims are measurable when)

Adapted from spec §18. Update this table as gates flip.

| Dimension | Claimable after target Phase 6 runs? | Evidence required |
|---|---|---|
| Attention-output cosine on Mistral/Qwen tested settings | Yes, if reproduced | Three-way JSON + schema validation |
| Compression ratio at tested operating points | Yes, after accounting reconciliation | Accounting audit + tests |
| v2 over v1 quality | Yes, if reproduced | v1/v2 ablation JSON |
| Catastrophic low-bit failure reduction | Yes, if per-head distributions reproduce | Min/head histogram in JSON |
| Perplexity (FP16, v2, in-repo TQ via K/V replay) | After a Modal run with `paper_valid=true` | `RUN-PERPLEXITY-*` JSON validating against `schemas/perplexity.schema.json` with `mode=full`, every requested method in `REAL_EVAL_METHODS`, no placeholder records, replay coverage ≥ 0.99 for non-FP16, and `n_tokens >= 64_000`. The K/V projection-replay path lives in `experiments/sqv2_replay.py`. The `paper_valid=true` artifact establishes "PPL of model under K/V cache compression" — NOT "PPL of full v2 architecture against Google's official TurboQuant". |
| LongBench (transparent subset, full path) | After a Modal run with `paper_valid=true` | `RUN-LONGBENCH-*` JSON with `mode=full`, every method record carrying `dataset_source=huggingface_thudm`, `n_per_task ≥ 50`, every requested method in `REAL_EVAL_METHODS`, no placeholder records, replay coverage ≥ 0.99 for non-FP16. Subsets smaller than `full` are paper-valid for that *transparent subset of LongBench*; an explicit "transparent subset" caveat lands in the JSON. The full path uses `experiments/longbench_dataset.py` (HF `THUDM/LongBench` adapter) + `experiments/longbench_metrics.py` (in-repo re-implementations of the upstream metric registry). Inline-corpus runs are NEVER paper-valid (synthetic corpus). V1-GAP-008. |
| Real generation quality (FP16, v2, in-repo TQ) | After a Modal run with `paper_valid=true` | `RUN-GENERATION-*` JSON with `mode=full`, every requested method in `REAL_EVAL_METHODS`, no placeholder records, replay coverage ≥ 0.99 for non-FP16. Token-overlap F1 vs FP16, distinct-1/2 metrics, full completion text in the JSON. |
| End-to-end latency (FP16) + KV-microbenchmark latency (v2, in-repo TQ) + hooked-replay end-to-end latency (v2, in-repo TQ) | Yes — but only the hooked-replay end-to-end row may be reported as "end-to-end inference latency under K/V compression"; never as production speedup. | `RUN-LATENCY-*` JSON with `device=cuda`, `timer=torch.cuda.Event`, `paper_valid=true`, no placeholder rows, every non-FP16 method has at least one `end_to_end_measured=true` row, and replay coverage ≥ 0.99 on the hooked-replay rows. Each row exposes (a) `microbenchmark=true / microbenchmark_kind=kv_compress_decompress_round_trip` for the K/V round-trip rows, and (b) `end_to_end_measured=true / production_kernel=false / measurement_kind=hooked_replay_end_to_end` for the full forward+decode rows with replay hooks. Downstream reports must call the hooked-replay rows "hooked replay end-to-end latency" and **never** "production speedup" — the per-layer Python hooks add overhead. V1-GAP-003 (production-kernel v2 decode latency) is still in force pending a production-quality v2 kernel (HF `Cache` subclass or pre-attention rewrite). |
| Official TurboQuant superiority | No | Official-implementation comparison |
| All architectures | No | Broader model suite |

### 5.1 Compressed-method evaluation discipline

The four next-stage harnesses (perplexity, generation, longbench,
latency) accept a `--methods` flag that controls which methods get
evaluated. The harness enforces the following:

* `REAL_EVAL_METHODS` is `("fp16", "spectralquant_v2", "turboquant")`.
  Only these have a real evaluation path. Any method outside this set
  is recorded as a *placeholder* and prevents `paper_valid` from
  flipping to true.
* For non-FP16 methods, the K/V projection-replay path
  (`experiments/sqv2_replay.py`) must achieve
  `replay_coverage.fraction_layers_real >= 0.99`. Layers that fall
  back to FP16 passthrough are counted in
  `replay_coverage.n_passthrough_calls` and surface in the JSON. A
  partial-coverage artifact is by design **not** paper-valid.
* The `paper_valid` boolean in every artifact is the AND of all
  applicable gates (mode, method-set, no placeholders, coverage,
  per-family thresholds like PPL `n_tokens` or LongBench
  `n_per_task`).

A reader of the JSON does not need to consult external context to
decide if it is citable: `paper_valid=true` plus the relevant section
of the JSON's own `caveats` is sufficient.

### 5.2 Reporting language for compressed-method artifacts

When citing a `RUN-PERPLEXITY-*` or `RUN-GENERATION-*` artifact whose
non-FP16 methods came from the K/V replay path, downstream paper text
MUST:

1. Disclose that the v2 / TurboQuant rows are produced by K/V
   projection-replay hooks, not by a production-quality engine
   replacement.
2. Quote the `replay_coverage.fraction_layers_real` value if anything
   less than 1.0.
3. For latency, *never* compare a microbenchmark row to an end-to-end
   row in the same paragraph without an explicit clarifying sentence.
4. For latency, the hooked-replay end-to-end rows
   (`measurement_kind=hooked_replay_end_to_end`,
   `production_kernel=false`) measure real forward+decode wall-clock
   under K/V compression but include Python-level per-layer hook
   overhead. They MUST be reported as "hooked replay end-to-end
   latency", NOT "production speedup". A production-kernel claim
   requires a HF `Cache` subclass or pre-attention rewrite — neither
   exists in this repo at the time of writing.
5. For LongBench, the JSON's per-method `dataset_source` field
   distinguishes `huggingface_thudm` (real LongBench) from
   `inline_corpus` / `synthetic`. Only `huggingface_thudm` artifacts
   may be cited as LongBench evidence; `inline_corpus` artifacts are
   harness validation only. Subsets smaller than `full` carry an
   explicit "transparent subset" caveat — never headline a subset
   artifact as full LongBench.

**Schemas and harnesses for the four families above are landed at the
commit that added this row** (see `experiments/run_perplexity.py`,
`run_longbench.py`, `run_generation.py`, `run_latency.py`,
`scripts/launch_modal_eval.py`, and `tests/test_eval_harnesses.py`).
Until real Modal artifacts land, the only outputs these harnesses
write are `mode=synthetic_smoke` or `mode=inline_corpus_smoke` JSONs
with `paper_valid=false`. Those smoke artifacts must NOT be cited
as evidence in the report — see `docs/evidence_catalog.md` "v2 next-stage
evaluation harnesses (planned, not yet evidence)".

## 6. Author-time checklist

Before any v2 commit that touches `README.md`, `paper_output/`, or `docs/spectralquant_v2_technical_spec.md` figures/tables:

- [ ] Every empirical sentence cites at least one evidence ID.
- [ ] No claim sits in a "blocked" row of section 3 unless its unblock requirement is met.
- [ ] Every compression ratio is the output of `accounting.py`, not a typed string.
- [ ] Every `RUN-*` JSON cited validates against its schema.
- [ ] Every TurboQuant reference is qualified as "local" unless verified against Google's code.
- [ ] Negative-result files in `results/comparison/` are addressed.
- [ ] The eight-pass reflection (spec §17) has been performed and a brief signoff is committed.

When in doubt, prefer the narrower wording. The strategic message of v2 ("the right unit of compression is the measured spectrum") is strongest when the empirical claims around it are visibly disciplined.
