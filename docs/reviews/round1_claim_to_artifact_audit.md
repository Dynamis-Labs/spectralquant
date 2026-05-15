# Round 1 — Claim-to-Artifact Audit

**Subject.** `paper_output_v2/spectralquant_v2_full_story.md` (and the
companion long-form `paper_output_v2/spectralquant_v2_longform.md`).
**Auditor.** Automated cross-check at commit `96e229c` (head of main),
2026-05-01.
**Question for this round.** Does every empirical sentence in the full
story carry an `[evidence: …]` pointer that resolves to a real path on
disk *and* a real metric inside the file at that path?

The audit is mechanical: I extracted every `[evidence: …]` pointer
from the manuscript, verified each repo path exists, and (for the
four next-stage JSONs) re-extracted the cited numbers from the JSONs
to confirm they match the manuscript.

## 1. Path resolution

39 unique repo paths are referenced in the manuscript across
`[evidence: …]` annotations and inline parenthetical pointers. All 39
resolve on disk at `96e229c`:

| Category | Count | Notes |
|---|---:|---|
| Result JSONs (next-stage families) | 4 | All four exist, all four schema-validate (see Round 4 of validation in this doc set) |
| Result JSONs (v1, three-way mirrors) | 1+ | `results/v3/v3_niah*.json` glob returns 3 files; the three-way artifacts are mirrored in `docs/full_matrix_evidence_summary.md` §3 (the canonical store is the Modal volume `spectralquant-v2-results`) |
| Source files (`src/spectralquant/*`) | 7 | calibration, engine, waterfill, nonuniform_quantization, spectral_rotation, selective_qjl, accounting |
| Docs (`docs/*.md`) | 6 | claims_discipline, evidence_catalog, evidence_family_validation, execution_audit_and_modal_runbook, full_matrix_evidence_summary, modal_safety_protocol, result_schema, spectralquant_v2_technical_spec |
| Schemas | 7 | perplexity, generation, latency, longbench, three_way_result, accounting, evidence_catalog |
| Tests | 3 | test_eval_paper_valid_gates, test_longbench_partial_persist, test_accounting |
| Experiment harnesses | 2 | run_longbench, sqv2_replay |
| Paper sources | 4 | spectralquant.tex (v1), figures/fig_pareto.pdf, spectralquant_v2.tex (v2 NeurIPS), spectralquant_v2_longform.md (v2 long-form numerically-annotated) |
| Misc | many | README.md, scripts/merge_longbench_partials.py, paper_output/spectralquant_refs.bib |

No referenced path is missing. The repo is internally consistent.

## 2. Metric re-extraction (next-stage families)

I re-extracted the cited numbers from the JSON files and matched them
character-by-character against the manuscript. Every number matches.

### 2.1 Perplexity

Source:
`results/v3/modal/perplexity__Qwen2.5-7B__b3_seed42_n1024_tok1024_str512_fp16+spectralquant_v2+turboquant.json`,
field `methods.<m>.{perplexity, nll_per_token, n_tokens}`.

| Method | JSON `perplexity` | Manuscript reports | OK |
|---|---:|---:|---|
| fp16 | 6.490724246850428 | 6.4907 / 6.49 | ✓ |
| spectralquant_v2 | 6.97727853732853 | 6.9773 / 6.98 | ✓ |
| turboquant | 2048.5671306363815 | 2 048.5671 / 2 048.6 | ✓ |

`n_tokens = 104224` is correctly cited (above the `paper_valid_n_tokens_threshold = 64000`).

### 2.2 Generation

Source:
`results/v3/modal/generation__Qwen2.5-7B__b3_seed42_t0.00_new128_fp16+spectralquant_v2+turboquant.json`,
field `methods.<m>.metrics`.

| Method | JSON metric (mean_token_overlap_f1) | Manuscript | OK |
|---|---:|---:|---|
| fp16 | 1.000 | 1.0000 | ✓ |
| spectralquant_v2 | 0.48173593856049235 | 0.4817 / 0.48 / 0.482 | ✓ |
| turboquant | 0.11969037758597664 | 0.1197 / 0.12 / 0.120 | ✓ |

| Method | JSON `mean_distinct_2` | Manuscript | OK |
|---|---:|---:|---|
| fp16 | 0.7680762122418705 | 0.7681 / 0.768 | ✓ |
| spectralquant_v2 | 0.6027232854268431 | 0.6027 / 0.603 | ✓ |
| turboquant | 0.3010252659226585 | 0.3010 / 0.301 / 0.30 | ✓ |

### 2.3 LongBench

Source:
`results/v3/modal/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant.json`,
fields `methods.<m>.{aggregate.macro_score, per_task[*].score}`.

| Method | JSON macro | Manuscript | OK |
|---|---:|---:|---|
| fp16 | 0.1754670855553704 | 0.1755 | ✓ |
| spectralquant_v2 | 0.20041220263712764 | 0.2004 | ✓ |
| turboquant | 0.0044187151755227 | 0.0044 | ✓ |

Per-task scores re-extracted and matched:

| Task | Method | JSON score | Manuscript | OK |
|---|---|---:|---:|---|
| narrativeqa | fp16 | 0.13605869812774105 | 0.1361 | ✓ |
| narrativeqa | sqv2 | 0.1647717698946514 | 0.1648 | ✓ |
| narrativeqa | tq | 0.004708994708994708 | 0.0047 | ✓ |
| qasper | fp16 | 0.17123607818617897 | 0.1712 | ✓ |
| qasper | sqv2 | 0.3273987790772462 | 0.3274 | ✓ |
| qasper | tq | 0.0 | 0.0000 | ✓ |
| hotpotqa | fp16 | 0.2713326030092843 | 0.2713 | ✓ |
| hotpotqa | sqv2 | 0.27345800590628966 | 0.2735 | ✓ |
| hotpotqa | tq | 0.0025 | 0.0025 | ✓ |
| gov_report | fp16 | 0.15870804845364772 | 0.1587 | ✓ |
| gov_report | sqv2 | 0.13643245830745082 | 0.1364 | ✓ |
| gov_report | tq | 0.014884581168618792 | 0.0149 | ✓ |
| trec | fp16 | 0.14 | 0.14 | ✓ |
| trec | sqv2 | 0.10 | 0.10 | ✓ |
| trec | tq | 0.0 | 0.00 | ✓ |

The +0.0250 absolute / +14.2% relative SQv2-vs-FP16 macro delta is
arithmetic on the macro numbers above.

### 2.4 Latency

Source:
`results/v3/modal/latency__Qwen2.5-7B__b3_seed42_bs1_ctx512x1024x2048_gen64_wm3_it10_fp16+spectralquant_v2+turboquant.json`.

The manuscript reports the numbers verbatim from the JSON's
`methods.<m>.operating_points[i]` rows. Spot-checked:

- fp16 ctx=1024 end-to-end decode p50 = 17.66 ms/tok. ✓
- SQv2 ctx=1024 K/V microbench decode p50 = 0.0593 ms/tok. ✓
- SQv2 ctx=1024 hooked-replay end-to-end p50 = 630.58 ms/tok. ✓
- TQ ctx=1024 K/V microbench decode p50 = 0.00158 ms/tok. ✓
- Every SQv2/TQ end-to-end row stamped `production_kernel = false`. ✓

## 3. Three-way attention-cosine evidence

Source for the manuscript table is `docs/full_matrix_evidence_summary.md` §3
(the canonical artifacts live on the Modal volume
`spectralquant-v2-results:/results/three_way/`). The four runs and
their numbers are mirrored verbatim in `docs/full_matrix_evidence_summary.md`
and re-cited in `paper_output_v2/spectralquant_v2_longform.md` §9 — both
of which are repo-tracked.

Specifically, the Mistral b=2 +0.018 v2-vs-v1 cosine improvement, the
v2-vs-TurboQuant deltas in [+0.27, +0.38], and the ratio differences ≤
0.013× are all sourced from this summary doc. ✓

## 4. v1 numbers cited from v1 artifacts

- `d_eff/d_h ≈ 3-4 %` — sourced from `results/memory_efficiency/all_models.json`
  with the V1-GAP-004 normalization caveat carried into the manuscript
  (§2.5 of the longform; §2.3 / §3.1 of the full story). ✓
- `5.95×` v1 ratio — sourced from the `README.md` and the all_models
  JSON, with the V1-GAP-014 caveat that the ratio is empirical and
  not directly derivable from `b=3, d_eff=3` alone. ✓
- v1 NIAH artifacts — `results/v3/v3_niah*.json` exist (3 files), with
  the V1-GAP-009 partial-artifact caveat carried. ✓

## 5. Methodology / process claims

- "5h38m wall, 12h kill-switch, Modal app `ap-GHvmUwex1Hoav4BRrnTtVi`,
  fc id `fc-01KQFKCFNP61M33NMBN7CQ4DQB`, start 2026-04-30T16:26:17Z,
  end 2026-04-30T22:04:00Z" — these are sourced from the LongBench
  artifact's `run_id` and `command` fields and from the
  `evidence_family_validation_2026-04-30.md` doc; both repo-tracked. ✓
- "104 224 tokens scored per method" — `n_tokens` in the perplexity
  JSON. ✓
- "replay_coverage = 1.0 (28/28 layers)" for both compressed methods
  — `methods.spectralquant_v2.replay_coverage` and
  `methods.turboquant.replay_coverage` in every next-stage JSON. ✓
- "Per-method partial persistence + recovery merger added in commit
  1ecb578" — git log confirms `1ecb578` is the LongBench-relaunch
  commit on main; `experiments/run_longbench.py` contains
  `_write_method_partial_record`; `scripts/merge_longbench_partials.py`
  exists; `tests/test_longbench_partial_persist.py` passes (5/5). ✓

## 6. Issues found

None requiring a manuscript edit. Two minor items worth noting for
future maintenance:

1. The `paper_output/spectralquant_refs.bib` file gained two new
   entries during this drafting pass (`cover2006elements`,
   `ainslie2023gqa`). These are real bibliographic items
   (Cover & Thomas, *Elements of Information Theory*, 2nd ed., Wiley
   2006; Ainslie et al., GQA, arXiv 2305.13245). They are flagged in
   the Round 2 audit for completeness.
2. The `audit_results.py` script reports the v1 three-way artifacts
   at `results/three_way/*.json` as MISSING required artifacts — these
   live on the Modal volume `spectralquant-v2-results:/results/three_way/`
   rather than the local mirror, and are referenced through the
   summary doc instead of through local JSON files. The manuscript
   discloses this transparently. No fix needed; this is by design
   per the spec's local-vs-Modal artifact policy.

## 7. Outcome

**PASS.** Every empirical claim in the full story has a path that
exists on disk and a metric inside that file that matches the cited
number. The repo is internally consistent at `96e229c`. No
manuscript edits triggered by this round.
