# SpectralQuant v2 — Four-Family Evidence Validation Summary

**Date.** 2026-04-30 (LongBench finalized 2026-04-30T22:04Z; this doc
last updated post-completion).
**Repo.** `niashwin/spectralquant-v2`
**Commits.**
- Three of the four next-stage families (perplexity, generation,
  latency) ran from `197bcfb4ad54a7d7bc9430a80695c62c145371fd`.
- The LongBench deterministic-subset run (canonical) ran from
  `1ecb578a0b0251f1a716469e51be4303c7191cd6` (12 h-capped
  relaunch after the earlier 6 h-capped attempt was kill-switched);
  it completed `paper_valid=true` in ≈ 5 h 38 min.
- The per-method partial-persistence patch + recovery merger were
  added in `1ecb578` (carried forward in subsequent doc-only commits).

This document is the canonical, repo-tracked summary of what is
**actually validated** across the four next-stage evidence families
described in `docs/execution_audit_and_modal_runbook.md` §7.7. Every
line below is checked against an on-disk JSON or partial shard; if it
is not, it is flagged as conjecture. Numbers are extracted verbatim
from the artifacts and must be re-extracted from the JSONs before any
downstream re-use.

The *configuration baseline* shared by the three completed families is
single-seed (42), single-model (Qwen2.5-7B), single-bit (b=3), single
GPU type (NVIDIA H200, Modal). Reading every single-config result as
"three runs at 7B Qwen on H200" is correct; reading any of them as
"validated across architectures / seeds / hardware" is not.

---

## 1. Perplexity (paper_valid)

- Artifact: `results/v3/modal/perplexity__Qwen2.5-7B__b3_seed42_n1024_tok1024_str512_fp16+spectralquant_v2+turboquant.json`
- Modal copy: `spectralquant-v2-results:/results/perplexity/perplexity__Qwen2.5-7B__b3_seed42_n1024_tok1024_str512_fp16+spectralquant_v2+turboquant.json`
- Script: `experiments/run_perplexity.py`
- Schema: `schemas/perplexity.schema.json` (validated by
  `eval_common.atomic_write_json` at write time).
- Dataset: `wikitext-103-raw-v1` validation, 1024 sequences × 1024
  tokens, stride=512, 104 224 tokens scored per method.
- Methods: fp16, spectralquant_v2 (b=3), turboquant (b=3).
- `paper_valid`: `true`. `caveats`: `[]`.

| Method | Avg bits | NLL/token | Perplexity | Replay coverage |
|---|---:|---:|---:|---:|
| fp16 | 16.0 | 1.870 | 6.4907 | n/a |
| spectralquant_v2 | 3.0 | 1.943 | **6.9773** | 1.00 (28/28) |
| turboquant | 3.0 | 7.625 | **2 048.5671** | 1.00 (28/28) |

**Reading.** SQv2 raises FP16 perplexity by 7.5 % at b=3, single seed,
single model. TurboQuant collapses to PPL=2 049 — a clear failure mode
of the in-repo TurboQuant baseline at this bit width on this model.
This conclusively unblocks Rung 1 in the paper's claim ladder for
perplexity vs the **in-repo** TurboQuant comparator. It does NOT
unblock multi-seed CIs, alternate architectures, or "official Google
TurboQuant".

**Limitations to disclose in any downstream sentence.**

1. Single seed, single model, single bit, single hardware.
2. The TurboQuant arm is the in-repo `methods.turboquant.label =
   "local"` — not the official Google or Blackwell-cuTile kernel.
3. 104 k tokens clears the `paper_valid_n_tokens_threshold = 64 000`
   from the JSON's `data` block, but n_tokens is the only sample-size
   number; CIs over multiple seeds / corpora are not reported.

---

## 2. Generation quality (paper_valid)

- Artifact: `results/v3/modal/generation__Qwen2.5-7B__b3_seed42_t0.00_new128_fp16+spectralquant_v2+turboquant.json`
- Modal copy: `spectralquant-v2-results:/results/generation/generation__Qwen2.5-7B__b3_seed42_t0.00_new128_fp16+spectralquant_v2+turboquant.json`
- Script: `experiments/run_generation.py`
- Schema: `schemas/generation.schema.json`.
- Decoding: `do_sample=False, temperature=0.0, top_p=1.0, top_k=0,
  max_new_tokens=128, seed=42`.
- Prompt set: 8 prompts spanning summarize / QA / code / math /
  long-context / reasoning. Prompts are stored verbatim inside
  `methods.<m>.completions[*].prompt`.
- `paper_valid`: `true`. `caveats`: `[]`.

| Method | mean_distinct_1 | mean_distinct_2 | mean_token_overlap_F1 vs FP16 self |
|---|---:|---:|---:|
| fp16 (self-overlap) | 0.581 | 0.768 | 1.000 |
| spectralquant_v2 (b=3) | 0.449 | 0.603 | **0.482** |
| turboquant (b=3) | 0.158 | 0.301 | **0.120** |

**Reading.** At b=3 on Qwen2.5-7B greedy, SpectralQuant v2 produces
output whose token-level overlap with the FP16 reference is ≈ 4× higher
than TurboQuant's. TurboQuant's mean_distinct_2 of 0.301 is the signature
of degenerate / repetitive output (tokens recycle into bigrams); SQv2's
0.603 is roughly 78 % of FP16's 0.768.

**Limitations.**

1. n=8 prompts. This is a qualitative diversity measurement, not a
   downstream-task accuracy claim.
2. No reference summarization / QA / coding-eval scores; that's
   LongBench's job.
3. Replay coverage is 1.0 (28/28 layers) for both compressed methods.

---

## 3. Latency (paper_valid, but with mandatory hooked-replay caveat)

- Artifact: `results/v3/modal/latency__Qwen2.5-7B__b3_seed42_bs1_ctx512x1024x2048_gen64_wm3_it10_fp16+spectralquant_v2+turboquant.json`
- Modal copy: `spectralquant-v2-results:/results/latency/latency__Qwen2.5-7B__b3_seed42_bs1_ctx512x1024x2048_gen64_wm3_it10_fp16+spectralquant_v2+turboquant.json`
- Script: `experiments/run_latency.py`
- Schema: `schemas/latency.schema.json`.
- Operating points: `batch_size=1, gen_tokens=64, ctx ∈ {512, 1024,
  2048}, warmup=3 iters, measured=10 iters`. Timer:
  `torch.cuda.Event`. Hardware: NVIDIA H200.
- `paper_valid`: `true`. `caveats` (verbatim from JSON):

  > spectralquant_v2 / turboquant rows include MICROBENCHMARK rows
  > (microbenchmark=true,
  > microbenchmark_kind=kv_compress_decompress_round_trip) and HOOKED
  > REPLAY END-TO-END rows (end_to_end_measured=true,
  > production_kernel=false,
  > measurement_kind=hooked_replay_end_to_end). The end-to-end rows
  > include Python-level per-layer hook overhead and are therefore NOT
  > a production-kernel speed claim. Compare them to fp16 only with
  > that caveat explicit.

| Ctx | fp16 decode (ms/tok p50) | SQv2 microbench | SQv2 hooked-replay e2e | TQ microbench | TQ hooked-replay e2e |
|---:|---:|---:|---:|---:|---:|
| 512 | 17.18 | 0.1188 | 664.85 | 0.00317 | 70.68 |
| 1024 | 17.66 | 0.0593 | 630.58 | 0.00158 | 70.72 |
| 2048 | 16.55 | 0.0290 | 630.59 | 0.00078 | 70.83 |

**Reading.**

- The K/V compress+decompress kernel is fast: SQv2 round-trip is
  0.0593 ms/tok @ ctx=1024, ≈ 297× the fp16 decode budget. TurboQuant's
  random-rotation round-trip is faster still (0.00158 ms/tok at the
  same ctx) — that is exactly what a data-oblivious method should be,
  and is the *cost SpectralQuant pays for calibration*.
- **The hooked-replay end-to-end rows are NOT a production-speed
  claim.** SQv2 hooked-replay decode at ctx=1024 is 630.6 ms/tok —
  about 36× slower than fp16, because each layer takes a Python-level
  per-token callback. Reporting "SQv2 is 36× slower than FP16" as a
  speed claim would be a misuse of the artifact.
- The peak-memory delta between fp16 and the compressed methods at
  these context lengths is < 30 MB; the cache-memory savings only
  start mattering at much longer context. This is consistent with
  V1-RESULT-001 (5.95× compression on the cache itself, not on total
  process memory).

**What this artifact unblocks.**

- "Microbenchmark: SpectralQuant v2 K/V compress+decompress kernel
  runs at 0.06 ms/tok @ ctx=1024 on Qwen2.5-7B / H200 / b=3."
- "Hooked-replay end-to-end forward+decode timing exists; it is not a
  production-speed claim, by construction."

**What this artifact does NOT unblock.**

- "SQv2 is faster than FP16 at decode." (Hooked replay is slower; a
  production-kernel implementation does not yet exist.)
- "SQv2 saves N% wall-clock at long context." (Only the K/V kernel is
  measured at the microbench level.)
- The V1-GAP-003 conflict between `neurips_latency_crossover.json` and
  `v3/v3_latency.json` is NOT resolved by this artifact; both remain
  in the gap table.

---

## 4. LongBench deterministic 5-task subset (paper_valid)

- **Status: COMPLETED, `paper_valid=true`, `caveats=["subset=deterministic: this artifact scores a *transparent subset* of LongBench tasks (['narrativeqa', 'qasper', 'hotpotqa', 'gov_report', 'trec']); do not headline it as full LongBench."]`.**
- Canonical artifact (local): `results/v3/modal/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant.json`
- Canonical artifact (Modal): `spectralquant-v2-results:/results/longbench/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant.json`
- Mirror + post-stop snapshot: `results/v3/modal/longbench_relaunch_2026-04-30/canonical/` and `results/v3/modal/longbench_relaunch_2026-04-30/snapshots/20260430T221642Z_post_stop/{status.json,events.jsonl,partial/method__<m>.json,partial/<m>__<task>.json}` (855 events; 3 method-record shards; 15 per-task progress shards).
- Modal run identity: app `ap-GHvmUwex1Hoav4BRrnTtVi`, function call `fc-01KQFKCFNP61M33NMBN7CQ4DQB`, app URL `https://modal.com/apps/sentra/main/ap-GHvmUwex1Hoav4BRrnTtVi`, app state at inspection `stopped` (clean).
- Wall-clock: started `2026-04-30T16:26:17Z` (harness), succeeded `2026-04-30T22:04:00Z` (harness `stage=success`), app stopped `2026-04-30T22:04:04Z`. ≈ 5 h 38 min on H200, ≈ 6 h 22 min remaining of the 12 h kill-switch.
- Code commit: `1ecb578a0b0251f1a716469e51be4303c7191cd6` (the patched harness with `_write_method_partial_record` and `--calibration-mode paper` hard-fail on silent calibration downgrade).
- Subset: `deterministic` = {narrativeqa, qasper, hotpotqa, gov_report, trec}. `n_per_task=50`, `max_input_tokens=8192`, `max_new_tokens=128`.
- Calibration knobs: `paper` mode, `n_calib=16`, `lloyd_max_iter=200`, `calib_max_seq_tokens=512` — all ≥ the `paper_valid_thresholds` baked into the JSON.
- Replay coverage: `fraction_layers_real = 1.0` for both compressed methods (28/28 layers calibrated, 0 passthrough) — clears the ≥ 0.99 paper-valid gate.

**Macro scores.**

| Method | macro_score | qa_f1 narrativeqa | qa_f1 qasper | qa_f1 hotpotqa | rouge_en gov_report | classification_em trec |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 0.1755 | 0.1361 | 0.1712 | 0.2713 | 0.1587 | 0.14 |
| spectralquant_v2 (b=3) | **0.2004** | **0.1648** | **0.3274** | **0.2735** | 0.1364 | 0.10 |
| turboquant (b=3) | 0.0044 | 0.0047 | 0.0000 | 0.0025 | 0.0149 | 0.00 |

**Reading.** SpectralQuant v2 macro-beats FP16 by +0.0250 absolute
(+14.2 % relative) on this deterministic subset at b=3 / single seed.
The win is driven mainly by qasper (+15.6 pts qa_f1, almost 2× FP16)
and a small narrativeqa gain (+2.9 pts); SQv2 loses on gov_report
(−2.2 pts ROUGE) and trec (−4 pts classification_em). TurboQuant b=3
collapses to ≈ 0 on every task — consistent with the perplexity
2 048.6 result and the generation-token-overlap 0.120 result already
documented in §1 and §2 above.

**Limitations to disclose.**

1. Subset, not full LongBench. Per `docs/claims_discipline.md` §5.2.5, the headline must say "5-task LongBench subset", not "LongBench". Full 21-task LongBench (`RUN-LONGBENCH-QWEN-FULL`) remains unblocked.
2. n=50 per task; no per-task error bars; single seed.
3. Single bit budget (b=3), single model (Qwen2.5-7B), single hardware class (NVIDIA H200 / Modal).
4. TurboQuant arm is the in-repo local re-implementation; V1-GAP-012 (official Google TurboQuant comparator) is not unblocked.
5. The 5 h 38 min wall-clock is dominated by SQv2's per-layer Python-callback hooked-replay overhead (≈ 3.66 h for SQv2's 5 tasks alone). This is the same overhead shown in §3's hooked-replay rows and is not a production-kernel claim.

**Recovery story (lesson preserved).** The first 6 h-capped attempt
on commit `6154175` (Modal app `ap-vTqL16w5Nmaw6s2oGc6czU`, launched
`2026-04-30T10:10Z`) was kill-switched without writing a canonical
JSON because the harness only wrote on a fully-successful
three-method finish. The relaunch above succeeded for two reasons:
(a) the kill-switch was raised to 12 h via
`SPECTRALQUANT_MODAL_TIMEOUT_LONGBENCH_SEC=43200`; (b) the harness was
patched to write per-method full-record shards as each method
finishes, so a future re-attempt that hits a wall-clock cap mid-run
remains recoverable via `scripts/merge_longbench_partials.py
--paper-valid`. Both invariants are now documented in
`docs/execution_audit_and_modal_runbook.md` §7.7 and the catalog.

---

## 5. Joint reading

All four completed families now paint a coherent picture for
SpectralQuant v2 at b=3 on Qwen2.5-7B versus the in-repo TurboQuant
baseline:

- **Perplexity.** SQv2 6.98 vs TQ 2 048.6 — a ≈ 294× gap.
- **Generation.** SQv2 token-overlap-F1 = 0.482 vs TQ 0.120 — 4×.
  Distinct-2 0.603 vs 0.301 — 2×.
- **Latency.** SQv2 K/V kernel is slower than TQ's by ≈ 38× at
  ctx=1024 (0.0593 vs 0.00158 ms/tok). This is the *price* SpectralQuant
  pays for calibration; the downstream-quality margins are the
  *return*.
- **LongBench (5-task deterministic subset, n=50/task).** SQv2 macro
  0.2004 vs FP16 0.1755 (+14.2 % relative) vs TQ 0.0044 (≈ 46×
  better than TQ). SQv2 also beats FP16 on macro at this operating
  point — driven mostly by qasper.

That cost / quality trade is the single cleanest reading of the v2
program. Anything stronger — multi-seed CI, multi-model, multi-bit,
production-kernel latency, official Google TurboQuant comparator,
full 21-task LongBench — is explicitly out of scope of this
snapshot.

---

## 6. What remains open after this snapshot

| Item | Status | Required artifact |
|---|---|---|
| Multi-seed CI on perplexity / generation | Not run | ≥ 5 seeds in JSONs, Wilcoxon p ≤ 0.05 |
| Multi-model architecture span on the next-stage harnesses | Partial (Qwen2.5-7B only across all four families) | Mistral-7B-v0.3 + LLaMA-3 8B (or similar) at b=3 |
| Multi-bit on the next-stage harnesses | Not run | b ∈ {2, 3, 5} on the same model under the same harnesses |
| Production-kernel v2 inference path | Not implemented | HF `Cache` subclass that does not use Python-level per-token hooks |
| Official Google TurboQuant comparator | Not run | TurboQuant from `scos_lab_turboquant` or `vangara2026turboquant_cutile`, run end-to-end |
| LongBench full 21-task | Not run | `RUN-LONGBENCH-QWEN-FULL` (deterministic 5-task subset is now done) |
| Calibration cost in wall-clock seconds (vs the v1 "15 s" headline) | V1-GAP-007, not resolved | An artifact in `results/calibration_cost/` with `wall_time_s` and tokens-per-s |
| Compression-accounting reconciliation (5.95× at b=3) | V1-GAP-014, not resolved | An artifact in `results/accounting_audit/` that derives 5.95× |

---

## 7. Provenance

- All four artifacts were produced by code on `niashwin/spectralquant-v2`
  at the commits cited in §0. The code is not branched from any other
  repo.
- The artifact JSONs do not contain HF tokens, Modal tokens, or any
  other secret. They record `software.{torch,transformers,datasets,python}`
  versions, model classes, and the gated-model revision SHAs only.
- The only credentials needed to *reproduce* the runs are the operator's
  HF token (for gated `mistralai/Mistral-7B-v0.3` weights — irrelevant
  for the Qwen2.5-7B runs above) and Modal CLI tokens. Both are managed
  via `docs/modal_safety_protocol.md` §3 — never via CLI flags, never
  echoed to the result JSON, and never committed to the repo.
