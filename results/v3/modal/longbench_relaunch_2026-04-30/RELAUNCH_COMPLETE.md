# LongBench relaunch (commit 1ecb578) — completed 2026-04-30

## Outcome: SUCCESS, paper_valid=true

The 12h-capped relaunch from commit `1ecb578a0b0251f1a716469e51be4303c7191cd6`
finished cleanly within ~5h 38min, well inside the 12h kill-switch.

## Modal run

| Field | Value |
|---|---|
| App ID | `ap-GHvmUwex1Hoav4BRrnTtVi` |
| Function call | `fc-01KQFKCFNP61M33NMBN7CQ4DQB` |
| App URL | https://modal.com/apps/sentra/main/ap-GHvmUwex1Hoav4BRrnTtVi |
| Created | 2026-04-30T16:26:04Z |
| Run start (harness) | 2026-04-30T16:26:17Z |
| Run end (harness, stage=success) | 2026-04-30T22:04:00Z |
| App stopped | 2026-04-30T22:04:04Z |
| Wall-clock | ~5h 38m (20,263s harness, 20,280s app) |
| 12h cap | 43,200s (unused remainder ~22,920s ≈ 6h 22m) |
| Modal app state at inspection | `stopped` (clean) |

## Macro scores (Qwen2.5-7B, b=3, deterministic subset, n=50/task, 5 tasks)

| Method | macro_score | qa_f1 (narrative/qasper/hotpot) | rouge_en (gov_report) | classification_em (trec) |
|---|---|---|---|---|
| fp16             | 0.1755 | 0.1361 / 0.1712 / 0.2713 | 0.1587 | 0.14 |
| spectralquant_v2 | 0.2004 | 0.1648 / 0.3274 / 0.2735 | 0.1364 | 0.10 |
| turboquant       | 0.0044 | (collapsed)              | (collapsed) | (collapsed) |

`paper_valid: true` (n_calib=16, lloyd_max_iter=200, calib_max_seq_tokens=512).
spectralquant_v2 beats fp16 on macro score on this deterministic subset
(qasper +15.6 pts and hotpotqa +0.13 pt drove the gain). turboquant
collapses at b=3 on this LongBench config — consistent with the prior
turboquant generation-degradation result already in the paper.

## Artifacts preserved in repo

Canonical (paper-grade):

- `results/v3/modal/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant.json`
  (also kept under
  `results/v3/modal/longbench_relaunch_2026-04-30/canonical/...`)

Status / events / per-method shards:

- `results/v3/modal/longbench_relaunch_2026-04-30/snapshots/20260430T221642Z_post_stop/status.json`
- `results/v3/modal/longbench_relaunch_2026-04-30/snapshots/20260430T221642Z_post_stop/events.jsonl` (855 events)
- `results/v3/modal/longbench_relaunch_2026-04-30/snapshots/20260430T221642Z_post_stop/partial/method__{fp16,spectralquant_v2,turboquant}.json`
- `results/v3/modal/longbench_relaunch_2026-04-30/snapshots/20260430T221642Z_post_stop/partial/{fp16,spectralquant_v2,turboquant}__{narrativeqa,qasper,hotpotqa,gov_report,trec}.json` (15 per-task records)

The on-volume canonical at
`/longbench/longbench__Qwen2.5-7B__...json` was pulled into the repo at
the canonical results path and verified (6,058 B, identical bytes).

## Audit manifest update

`scripts/audit_results.py` `next_stage_evidence_families` now lists the
paper-valid LongBench canonical alongside the existing partial-status
entry, so future audit runs surface the completed evidence.

## Inspection-task posture (read this if relaunching)

A duplicate launch is **NOT safe** and **NOT needed**. The completed
canonical JSON exists on the Modal volume and in the repo. Any future
agent must (a) check
`results/v3/modal/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant.json`
first, then (b) check `modal app list` for app
`ap-GHvmUwex1Hoav4BRrnTtVi` (stopped) before deciding to relaunch.
