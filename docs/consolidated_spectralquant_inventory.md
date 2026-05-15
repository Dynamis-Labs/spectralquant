# Consolidated SpectralQuant Inventory

This document is the source-of-truth inventory for the consolidated SpectralQuant program. It maps every public artifact, code path, evidence file, and document across the three repositories that together carry the program's history. The consolidated unrestricted technical report at `paper_output_consolidated/spectralquant_unrestricted_paper.tex` is the canonical public-facing manuscript; everything here either feeds that manuscript or is preserved as historical provenance.

The public method is one program: **SpectralQuant**. The repository labels `v1` and `v2` are development tags that delimit the two evidence layers integrated by this consolidation.

## 1. Program scope: one method, two evidence layers

| Layer | Public name | Repository tag | Allocation rule | Contribution |
| --- | --- | --- | --- | --- |
| Initial evidence layer | SpectralQuant | `v1` (commit-history label) | Uniform-allocation special case inside the calibrated semantic subspace | Calibrated eigenbasis rotation; participation-ratio split; selective QJL correction; 5.95× compression on Qwen2.5-14B-Instruct at b≈3; deff/d_h ≈ 3–5% universality |
| Expanded evidence layer | SpectralQuant | `v2` (development cycle label) | Water-filled allocation derived from rate–distortion calculus | Greedy water-filling rule (waterfill.py); per-dim Lloyd–Max codebooks; CompressionAccounting module; four next-stage paper-valid evidence families on H200; three-way attention-cosine matrix at matched compression ratio; claims-discipline + audit-trail process |

The two layers share the calibration, rotation, and JL-correction stages. They differ only in the bit-allocation rule inside the semantic subspace; under matched compression ratio (within 0.013× absolute) the water-filled allocation matches the uniform allocation at b=3 and improves it by +0.018 mean cosine at b=2 on Mistral-7B-v0.3.

## 2. Repositories

| Role | URL | Visibility | Status |
| --- | --- | --- | --- |
| Canonical consolidation repository | https://github.com/niashwin/spectralquant-full | Private | **This repository.** Carries both evidence layers; consolidated paper lives here. Renamed from `niashwin/spectralquant-v2`; the historical name survives in archived JSON `repo` fields, the Modal volume name `spectralquant-v2-results`, and the development-history filesystem paths (e.g. `paper_output_v2/`, `docs/spectralquant_v2_technical_spec.md`) for audit traceability. |
| Original public release repository | https://github.com/Dynamis-Labs/spectralquant | Public | Read-only for this consolidation. Source of truth for the original SpectralQuant manuscript and the initial evidence layer's public release. Not modified by this work. |
| Original private development mirror | https://github.com/niashwin/spectralquant | Private | Read-only for this consolidation. Pre-public development snapshots of the initial evidence layer. Not modified by this work. |

This consolidation work modified only `niashwin/spectralquant-full` (the repository previously known as `niashwin/spectralquant-v2`).

## 3. Manuscript artifacts

| Path | Role |
| --- | --- |
| `paper_output_consolidated/spectralquant_unrestricted_paper.tex` | **Canonical consolidated unrestricted technical report.** Single SpectralQuant story. Use this for any public reference to the program. |
| `paper_output_consolidated/spectralquant_unrestricted_paper.pdf` | Compiled PDF of the consolidated report. |
| `paper_output_consolidated/figures/` | Six PDFs used by the consolidated report; copies of the expanded-layer figures with consolidated captions in the `.tex`. |
| `paper_output/spectralquant.tex` | Original public manuscript ("3% Is All You Need"). Initial evidence layer; preserved unchanged. |
| `paper_output/spectralquant.pdf` | Compiled original public manuscript. |
| `paper_output/spectralquant_refs.bib` | Shared bibliography. The consolidated report points to this file via `..\paper_output\spectralquant_refs`. |
| `paper_output/figures/` | Initial-evidence-layer figures (eigenvalue spectrum, memory savings, Pareto, scaling, seqlen). Preserved unchanged. |
| `paper_output_v2/spectralquant_unrestricted_paper.tex` | Pre-consolidation expanded-evidence draft. Superseded by the consolidated version; preserved for audit traceability. |
| `paper_output_v2/spectralquant_v2.tex` | Pre-consolidation NeurIPS-format expanded-evidence draft. Superseded; preserved. |
| `paper_output_v2/spectralquant_v2_full_story.md` | Narrative master used during the expanded-layer development cycle. |
| `paper_output_v2/spectralquant_v2_longform.md` | Numerically annotated long-form companion of the expanded-layer drafts. |
| `paper_output_v2/spectralquant_v2_supplement.tex` / `.pdf` | Supplement to the expanded-layer NeurIPS draft. |
| `paper_output_v2/figures/` | Expanded-evidence-layer figures (perplexity, generation, latency, longbench, attention cosine, pipeline). The consolidated report uses copies of these. |

## 4. Source code

The consolidation repository's `src/spectralquant/` is a strict superset of the original public repository's. The expanded layer added two modules; everything else is shared.

| Module | Layer | Role |
| --- | --- | --- |
| `src/spectralquant/calibration.py` | shared | Per-(layer, head) eigendecomposition; participation ratio; semantic/tail split |
| `src/spectralquant/spectral_rotation.py` | shared | Rotation by U_{ℓ,h} on cached keys |
| `src/spectralquant/selective_qjl.py` | shared | Selective Johnson–Lindenstrauss correction in the semantic subspace |
| `src/spectralquant/nonuniform_quantization.py` | shared (per-dim Lloyd–Max codebooks driven by the allocation in the expanded layer) | Lloyd–Max codebook fit |
| `src/spectralquant/spectralquant.py` | shared (deprecated; see `engine.py`) | Original engine |
| `src/spectralquant/engine.py` | expanded | Canonical inference engine for both allocation settings |
| `src/spectralquant/waterfill.py` | expanded only | Greedy integer water-filling rule (eq. for Algorithm 1 in the consolidated paper) |
| `src/spectralquant/accounting.py` | expanded only | `CompressionAccounting`: derives compression ratio from b and deff and per-layer overheads; cross-validated by `tests/test_accounting.py` |
| `src/spectralquant/metrics.py` | shared | Attention-output cosine, perplexity, distinct-n, F1 |
| `src/spectralquant/utils.py` | shared | Common utilities |

The original private development mirror `niashwin/spectralquant` additionally contains `src/spectralquant/kernel.py` and a long sequence of in-progress development commits (residual cache, cached codebooks, etc.); none of those are required to reproduce any number in the consolidated paper.

## 5. Evidence artifacts

### 5.1 Initial evidence layer (preserved verbatim)

| Path (in this repository) | Source | Use in consolidated paper |
| --- | --- | --- |
| `results/memory_efficiency/all_models.json` | initial layer | `V1-RESULT-001`: deff/d_h ≈ 3–5% universality across Qwen2.5-1.5B/7B/14B and Llama 3.1-8B (cited in §1, §2.3) |
| `results/eigenspectral/` | initial layer | Eigenvalue-spectrum analyses underlying §2.3 and Figure 1 (initial-layer figures kept in `paper_output/figures/`) |
| `results/comparison/`, `results/baseline_reproduction/` | initial layer | Cosine comparisons against TurboQuant on Qwen2.5-14B-Instruct at b≈3 (5.95× headline; 0.9485 vs 0.9226 cosine). Quoted in the abstract and §6 of the consolidated paper. |
| `results/aggressive/`, `results/calibration_stability/`, `results/comprehensive/`, `results/deff_sweep/`, `results/final/`, `results/kernel/`, `results/lowrank/`, `results/multiregime/`, `results/optimal_allocation/`, `results/push_095/`, `results/seqlen_sweep/`, `results/shaped_cache/`, `results/unnormalized/` | initial layer | Original-paper sweeps. Not all are cited in the consolidated paper; preserved for audit. |
| `results/neurips/` | initial layer | NeurIPS-form numbers from the original release. Provenance only. |

### 5.2 Expanded evidence layer (paper-valid next-stage runs)

| Path | Family | Evidence ID |
| --- | --- | --- |
| `results/v3/modal/perplexity__Qwen2.5-7B__b3_seed42_n1024_tok1024_str512_fp16+spectralquant_v2+turboquant.json` | Perplexity (WikiText-103, n_tokens = 104,224) | `RUN-PERPLEXITY-QWEN2.5-7B` |
| `results/v3/modal/generation__Qwen2.5-7B__b3_seed42_t0.00_new128_fp16+spectralquant_v2+turboquant.json` | Greedy generation (8 prompts; T=0; max_new=128) | `RUN-GENERATION-QWEN2.5-7B` |
| `results/v3/modal/latency__Qwen2.5-7B__b3_seed42_bs1_ctx512x1024x2048_gen64_wm3_it10_fp16+spectralquant_v2+turboquant.json` | Latency (CUDA-event; ctx ∈ {512,1024,2048}; microbench + hooked replay) | `RUN-LATENCY-QWEN2.5-7B` |
| `results/v3/modal/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant.json` | LongBench (deterministic 5-task subset; n=50/task) | `RUN-LONGBENCH-QWEN2.5-7B-DETERMINISTIC` |
| `results/three_way/` (Modal volume `spectralquant-v2-results:/results/three_way/`; volume name retained for provenance — predates the repo rename to `spectralquant-full`) | Three-way attention-cosine matrix; Mistral-7B-v0.3 b∈{2,3,5} + Qwen2.5-7B b=3 | `RUN-THREEWAY-MISTRAL-{2,3,5}BIT`, `RUN-THREEWAY-QWEN-3BIT` |
| `results/v3/v3_*.json` | Aggregate cross-arch / NIAH / latency / perplexity rollups from the expanded layer | reference only |

The JSON method key `spectralquant_v2` inside every file is the literal identifier under which SpectralQuant with the water-filled allocation is recorded. The key is preserved verbatim for downstream tooling stability; the public method is SpectralQuant.

## 6. Documentation

| Path | Role |
| --- | --- |
| `docs/consolidated_spectralquant_inventory.md` | **This document.** Source-of-truth inventory across all three repositories. |
| `docs/claims_discipline.md` | Single rulebook for what the program is allowed to claim publicly. Updated alongside this consolidation to spell out canonical SpectralQuant claim language. |
| `docs/spectralquant_v2_technical_spec.md` | Canonical SpectralQuant technical specification. Filename retains the historical "v2" tag; the technical content is the canonical specification. |
| `docs/evidence_catalog.md`, `docs/evidence_catalog.json` | Stable claim-identifier catalog (`V1-RESULT-*`, `V1-IMPL-*`, `V1-GAP-*`, `V2-SPEC-*`, `RUN-*`). The identifiers are historical; renaming would break audit cross-references. |
| `docs/full_matrix_evidence_summary.md` | §3 contains the verbatim three-way attention-cosine table feeding the consolidated paper's Table 4. |
| `docs/evidence_family_validation_2026-04-30.md` | Four-family validation summary on Modal H200. |
| `docs/execution_audit_and_modal_runbook.md` | Modal runbook; expanded-layer execution audit. |
| `docs/modal_safety_protocol.md` | Modal safety protocol (token handling, kill switches). |
| `docs/result_schema.md` | Result-JSON schema documentation; companion of `schemas/`. |
| `docs/reviews/round1_*.md` | Audit: claim-to-artifact (expanded layer). |
| `docs/reviews/round2_*.md` | Audit: citation. |
| `docs/reviews/round3_*.md` | Audit: narrative coherence. |
| `docs/reviews/round4_*.md` | Audit: style / human editorial. |
| `docs/reviews/round5_consolidated_naming_audit.md` | **New.** Audit: consolidated-naming discipline (created with this consolidation). |
| `docs/reviews/round6_mechanism_integration_audit.md` | **New.** Audit: integration of the April 2026 perplexity-mechanism supporting note into the consolidated paper. |
| `docs/supporting_notes/perplexity_mechanism_note_2026-04.md` | **New.** Sentra technical note (Vangara & Gopinath, April 2026) preserved verbatim with provenance; provides the four-mechanism narrative used by the consolidated paper's `Mechanism behind the perplexity gap` subsection. **Supporting-note-only**, not paper-valid; numeric Mistral-7B-v0.3 / WikiText-103 perplexity table is not used to back any paper claim. |

## 7. Schemas, scripts, tests

| Path | Role |
| --- | --- |
| `schemas/` | JSON Schema files validated by `tests/test_eval_paper_valid_gates.py` and `tests/test_longbench_partial_persist.py` |
| `scripts/launch_modal_eval.py` | Modal launcher for the four next-stage evidence families |
| `scripts/build_paper_figures.py` | Builds `paper_output_v2/figures/*.pdf` from the JSON artifacts. The consolidated report's `figures/` are copies of these. |
| `scripts/merge_longbench_partials.py` | Per-method partial persistence + recovery merger for LongBench |
| `experiments/sqv2_replay.py` | Calibration + replay harness for both allocation settings (filename retains the historical "sqv2" tag) |
| `experiments/run_longbench.py` | LongBench evaluation harness; partial-persistence aware |
| `tests/test_eval_paper_valid_gates.py` | 29 paper-validity gate assertions |
| `tests/test_longbench_partial_persist.py` | 5 persistence/merger assertions |
| `tests/test_accounting.py` | Cross-validates `CompressionAccounting` |
| `tests/test_result_schema.py` | Validates artifacts against the schemas |

## 8. Provenance commitments

* The consolidation repository `niashwin/spectralquant-full` (previously `niashwin/spectralquant-v2`; renamed 2026-05-01) is the only repository modified by the consolidation pass that produced the consolidated unrestricted technical report. The original public repository `Dynamis-Labs/spectralquant` and the original private development mirror `niashwin/spectralquant` are untouched.
* Every artifact in this repository's `results/` directory traces back to a specific evidence layer per §5.
* Every claim in `paper_output_consolidated/spectralquant_unrestricted_paper.tex` traces back to a JSON artifact and an evidence-catalog ID through the traceability table in that manuscript's §13.
* Historical labels (`v1`, `v2`, the `spectralquant_v2` JSON method key, `paper_output_v2/`, the historical repository name `niashwin/spectralquant-v2`, the Modal volume name `spectralquant-v2-results`, and `docs/spectralquant_v2_technical_spec.md`) are preserved in their original positions to keep the audit trail intact. The canonical public-facing repository name is now `niashwin/spectralquant-full`.

## 9. What this consolidation does NOT change

* Empirical numbers. Every reported number is read verbatim from the same JSON files used by the pre-consolidation drafts; no new experiments were run.
* Code paths. No source files in `src/`, `scripts/`, `experiments/`, `tests/`, or `schemas/` were modified.
* Result JSONs. No file under `results/` was modified.
* Bibliography. `paper_output/spectralquant_refs.bib` is reused unchanged.
* Caveat language. Every limitation and caveat from the pre-consolidation drafts is preserved verbatim in the consolidated paper.

What it does change is naming-discipline, narrative framing, and the structure of the public-facing technical report: SpectralQuant is presented as a single research program, with v1 / v2 surfacing only as repository labels in the development-history section, the JSON method key, and the stable evidence-catalog identifiers.
