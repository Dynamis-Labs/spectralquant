# SpectralQuant v1 + v2 Evidence Catalog

This catalog maps every empirical artifact already present in the v1 baseline (now imported into `niashwin/spectralquant-v2`) plus every paper-valid v2 Modal run to a stable evidence ID. The v2 implementation, the v2 report, and every v2 figure must reference these IDs rather than typing numbers freehand. The machine-readable form is `docs/evidence_catalog.json`; this Markdown file is its narrative companion and must stay synchronized with it.

The catalog is intentionally honest about what is missing or contradictory. Entries with caveats are not blocked from use — but the caveat must travel with the number into any downstream report.

## How to read this catalog

- `V1-PAPER-*` — paper source, compiled PDF, references, and published figures.
- `V1-README-*` — claim → script → JSON mapping in the v1 README.
- `V1-IMPL-*` — implementation modules under `src/spectralquant/`.
- `V1-EXP-*` — experiment scripts under `experiments/`.
- `V1-RESULT-*` — result JSONs (and one CSV) under `results/`.
- `V1-FIG-*` — figures (top-level `figures/` and `paper_output/figures/`).
- `V1-CONF-*` — config / build files.
- `V1-TEST-*` — existing tests.
- `V1-GAP-*` — known discrepancies. These block reuse of the affected claim until resolved.
- `V2-SPEC-001` — `docs/spectralquant_v2_technical_spec.md`, the authoritative v2 spec.
- `RUN-THREEWAY-*` — paper-valid v2 Modal run artifacts (sliced full-path WikiText-103 rows). The narrative summary lives in `docs/full_matrix_evidence_summary.md`. The JSONs live on the Modal volume `spectralquant-v2-results` under `/results/three_way/`; local copies sit in `modal_artifacts/` (off-repo).

Caveats use the convention: a numeric extraction with a caveat field MAY be cited as evidence ONLY if the caveat is preserved alongside it.

## Paper artifacts

| ID | Path | Supports | Notes |
|---|---|---|---|
| V1-PAPER-001 | `paper_output/spectralquant.tex` | v1 narrative, method, figure list | Authored claims; every empirical sentence must trace to a JSON. |
| V1-PAPER-002 | `paper_output/spectralquant.pdf` | compiled v1 report | Compiled output; not a primary source. |
| V1-PAPER-003 | `paper_output/spectralquant_refs.bib` | citations | — |
| V1-PAPER-004 | `paper_output/figures/` | published figures | Regenerate from JSON before reuse in v2. |
| V1-PAPER-005 | `paper_output/generate_figures.py` | figure generator | — |

## README claim mapping

| ID | Path | Supports | Notes |
|---|---|---|---|
| V1-README-001 | `README.md` | "Headline Results" + "Paper Claims → Code → Data" tables | Headline (0.9485 vs 0.9226, 5.95x vs 5.02x) maps to V1-RESULT-001 for Qwen2.5-14B-Instruct only; the two configs are at different `avg_bits` (3.19 vs 2.69). 2.2x latency claim conflicts with V1-GAP-003. |

## Implementation modules (V1-IMPL-*)

| ID | Path | Role | Notes |
|---|---|---|---|
| V1-IMPL-001 | `src/spectralquant/calibration.py` | per-layer/per-head collector, PCA, d_eff | Verify normalization convention before extending (V1-GAP-004). |
| V1-IMPL-002 | `src/spectralquant/spectral_rotation.py` | calibrated eigenbasis vs Haar | — |
| V1-IMPL-003 | `src/spectralquant/nonuniform_quantization.py` | Lloyd-Max + uniform-bit codebook | v2 will extend per-dim codebooks; v1 must remain reachable via `use_water_fill=False`. |
| V1-IMPL-004 | `src/spectralquant/selective_qjl.py` | semantic-only QJL residual | Preserve in v2. |
| V1-IMPL-005 | `src/spectralquant/engine.py` | engine subclass of TurboQuant | One of two coexisting engines (V1-GAP-010). |
| V1-IMPL-006 | `src/spectralquant/spectralquant.py` | legacy standalone engine | Other coexisting engine (V1-GAP-010). |
| V1-IMPL-007 | `src/spectralquant/metrics.py` | cosine, MSE, ratio helpers | — |
| V1-IMPL-008 | `src/spectralquant/utils.py` | seeds, model config, data loading | — |
| V1-IMPL-009 | `src/spectralquant/__init__.py` | public namespace | Audit exports while resolving V1-GAP-010. |

## Experiment scripts (V1-EXP-*)

| ID | Path | Produces | Notes |
|---|---|---|---|
| V1-EXP-001 | `experiments/run_memory_efficiency.py` | `results/memory_efficiency/all_models.json` | "4 models" claimed, 3 Qwen rows present (V1-GAP-002). |
| V1-EXP-002 | `experiments/neurips_models_asymmetry.py` | `results/neurips/neurips_{mistral,gemma,kv_asymmetry}.json` | Gemma row is a 403 (V1-GAP-011). |
| V1-EXP-003 | `experiments/neurips_seeds_latency.py` | `results/neurips/neurips_{10seed,latency_crossover,qwen7b_ppl}.json` | Filename says "10seed", file holds 5 seeds (V1-GAP-001). |
| V1-EXP-004 | `experiments/neurips_llama_full.py` | LongBench, NIAH, Llama PPL JSONs | LongBench n=5 (V1-GAP-008). |
| V1-EXP-005 | `experiments/lowrank_cossim_sweep.py` | `results/lowrank/lowrank_cossim_sweep.json` | — |
| V1-EXP-006 | `experiments/run_v3_perplexity_crossarch.py` | `results/v3/v3_{crossarch,perplexity}.json` | — |
| V1-EXP-007 | `experiments/run_v3_ppl_niah_v2.py` | `results/v3/v3_perplexity_v2.json`, `results/v3/v3_niah_llama_v2.json` | Identical PPL across methods (V1-GAP-004b). |
| V1-EXP-008 | `experiments/run_v3_deff_distshift_latency.py` | `results/v3/{v3_distribution_shift,v3_deff_sweep,v3_latency}.json` | Source of the conflicting latency file (V1-GAP-003). |
| V1-EXP-009 | `experiments/run_calibration_stability.py` | `results/calibration_stability/stability.json` | File self-declares as reconstructed from logs (V1-GAP-005). |
| V1-EXP-010 | `experiments/run_final_experiments.py` | `results/final/final_experiments.json` | — |
| V1-EXP-011 | `experiments/phase1_eigenspectral.py` | `results/eigenspectral/*` | gate1_status FAILED on disk (V1-GAP-006). |
| V1-EXP-012 | `experiments/phase0_setup.py` | `results/baseline_reproduction/` | TurboQuant local baseline reproduction status (V1-GAP-012). |
| V1-EXP-013 | `experiments/phase2_integration.py` | — | — |
| V1-EXP-014 | `experiments/phase3_exp1_attention_quality.py` | — | — |
| V1-EXP-015 | `experiments/phase3_exp2_ablation.py` | `results/comparison/{ablation_v2,full_ablation}.json` | TQ baselines diverge across files (V1-GAP-013). |
| V1-EXP-016 | `experiments/phase3_exp3_generation.py` | — | — |
| V1-EXP-017 | `experiments/phase3_exp4_benchmarks.py` | — | — |
| V1-EXP-018 | `experiments/phase3_exp5_vector_search.py` | — | — |
| V1-EXP-019 | `experiments/phase3_exp6_latency.py` | — | Cross-check before reuse (V1-GAP-003). |
| V1-EXP-020 | `experiments/phase3_exp7_calibration_cost.py` | — | "15-second calibration" provenance (V1-GAP-007). |
| V1-EXP-021 | `experiments/run_all_experiments.py` | — | Driver. |
| V1-EXP-022 | `experiments/multiregime_sweep.py` | `results/multiregime/` | — |
| V1-EXP-023 | `experiments/optimal_allocation_sweep.py` | `results/optimal_allocation/` | Closest v1 precursor to v2 water-filling. |
| V1-EXP-024 | `experiments/push_to_095.py` | `results/push_095/` | — |
| V1-EXP-025 | `experiments/shaped_cache_sweep.py` | `results/shaped_cache/`, `figures/shaped_cache_*.png` | — |

## Result JSONs with extracted numbers (V1-RESULT-*)

The following entries include numerical extractions copied verbatim from the on-disk JSON. See `docs/evidence_catalog.json` for the canonical extraction, which is what tests validate against.

### V1-RESULT-001 — `results/memory_efficiency/all_models.json`

Headline 14B (Qwen2.5-14B-Instruct, n_layers=48, n_kv_heads=8, head_dim=128, d_eff=4):

| Config | cos_sim | ratio | avg_bits |
|---|---:|---:|---:|
| TQ_3bit | 0.9226 | 5.020× | 3.1875 |
| TQ_2bit | 0.8249 | 7.314× | 2.1875 |
| SQ_noQJL_v3 | 0.9485 | 5.953× | 2.6875 |
| SQ_selQJL | 0.9147 | 5.919× | 2.7031 |

**Caveat:** The README's headline 14B comparison (TQ_3bit vs SQ_noQJL_v3) compares two configs with different `avg_bits`. The honest framing is "SpectralQuant achieves higher cosine at higher compression ratio (lower bits)", not "SpectralQuant beats TurboQuant at the same setting." Also, the file holds 3 Qwen rows; the README's "4 models" framing is incorrect (V1-GAP-002).

### V1-RESULT-002 — `results/neurips/neurips_kv_asymmetry.json`

| Model | d_eff_keys | d_eff_values | head_dim |
|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 3.95 | 40.34 | 128 |
| Qwen2.5-7B-Instruct | 4.30 | 52.15 | 128 |
| Qwen2.5-14B-Instruct | 4.17 | 52.11 | 128 |
| Llama-3.1-8B-Instruct | 3.64 | 44.21 | 128 |
| Mistral-7B-Instruct-v0.3 | 4.18 | 54.93 | 128 |

**Caveats:** (a) Gemma row is an HF 403 gated-repo error (V1-GAP-011). (b) These values use NORMALIZED keys; not comparable to the unnormalized 35.19 in V1-RESULT-008 (V1-GAP-004).

### V1-RESULT-003 — `results/neurips/neurips_latency_crossover.json`

Qwen2.5-1.5B-Instruct, 512 tokens: SQ per-step 0.2573 ms vs TQ per-step 0.5663 ms; SQ attention 0.0708 ms vs TQ attention 0.2266 ms; SQ compress 0.1865 ms vs TQ compress 0.3397 ms; SQ speedup over TQ 2.20×.

**Caveat:** Conflicts with V1-RESULT-007 at the same sequence length (V1-GAP-003).

### V1-RESULT-004 — `results/v3/v3_perplexity_v2.json`

fp16, TQ, and SQ all report `ppl = 9.509980534121931` to identical precision over 1023 tokens.

**Caveat:** Identical 13-digit equality is suspicious; likely a too-short / non-decoding eval (V1-GAP-004b).

### V1-RESULT-006 — `results/neurips/neurips_10seed.json`

Qwen2.5-1.5B-Instruct. Seeds in file: `[42, 123, 7, 2024, 31415]` (5 seeds). TQ mean 0.8409, SQ mean 0.8635, Wilcoxon p=0.03125 (n=5 paired observations — minimum achievable p).

**Caveat:** Filename and README claim 10 seeds; only 5 on disk (V1-GAP-001).

### V1-RESULT-007 — `results/v3/v3_latency.json`

512 tokens: TQ compression 0.399 ms, SQ compression 21.04 ms (≈53× slower); TQ attention 0.227 ms, SQ attention 0.051 ms.

**Caveat:** SQ compression cost contradicts V1-RESULT-003 (V1-GAP-003).

### V1-RESULT-008 — `results/eigenspectral/summary_statistics.json`

Unnormalized eigenspectral mean d_eff: keys 35.19, values 56.65 (head_dim=128).

**Caveat:** Unnormalized convention; do not compose with the normalized-keys numbers in V1-RESULT-002 (V1-GAP-004).

### V1-RESULT-009 — `results/eigenspectral/phase1_metadata.json`

Qwen/Qwen2.5-1.5B, n_seqs=50, seq_len=128, n_kv_heads=2, head_dim=128, **gate1_status="FAILED"**, wall_time_s=31.05.

**Caveats:** Gate failure on disk (V1-GAP-006); 31 s is the closest data point to README's "15-second calibration" claim (V1-GAP-007).

### V1-RESULT-011 — `results/calibration_stability/stability.json`

mean_cv 0.0394, max_cv 0.202, n_samples_per_split 50, per_split_mean_deff [3.93, 3.71, 3.61], total_heads 56, n_heads_cv_above_02 = 1.

**Caveat:** File contains the literal note "Reconstructed from B200 experiment log output (original file was truncated during download)" (V1-GAP-005).

### V1-RESULT-012 — `results/comparison/comparison_results.json`

Qwen/Qwen2.5-1.5B-Instruct, total_bits=3, n_samples=16, **TQ mean cosine 0.9703, SQ mean cosine 0.4187, mean improvement -0.5516, win_rate 0/16**.

**Caveat:** Strongly negative result — directly contradicts the headline "SQ wins per-head" story. Either a broken or different SQ build, or a real low-bit small-model weakness. Must be reconciled with `results/comparison/proper_comparison.json` before any v2 reuse (V1-GAP-002b).

### Other result JSONs (extractions deferred)

| ID | Path | Notes |
|---|---|---|
| V1-RESULT-005 | `results/neurips/neurips_qwen7b_ppl.json` | 7B PPL ≈ 7.51. |
| V1-RESULT-010 | `results/eigenspectral/deff_per_layer.csv` | CSV; verify columns. |
| V1-RESULT-013 | `results/comparison/proper_comparison.json` | Reconciles with V1-RESULT-012. |
| V1-RESULT-014 | `results/comparison/full_ablation.json` | — |
| V1-RESULT-015 | `results/comparison/ablation_v2.json` | "v2" here is v1-era ablation, NOT the SQ v2 work. |
| V1-RESULT-016 | `results/lowrank/lowrank_cossim_sweep.json` | r=4 → CosSim≈0.15 backing for value-failure claim. |
| V1-RESULT-017 | `results/final/final_experiments.json` | Config F. |
| V1-RESULT-018 | `results/v3/v3_crossarch.json` | Llama-only; Mistral / Gemma rows live in `neurips_*.json`. |
| V1-RESULT-019 | `results/v3/v3_distribution_shift.json` | wiki↔code splits. |
| V1-RESULT-020 | `results/v3/v3_longbench.json` | n=5/task — too small (V1-GAP-008). |
| V1-RESULT-021 | `results/v3/v3_niah_llama_v2.json` | NIAH limited (V1-GAP-009). |
| V1-RESULT-022 | `results/v3/v3_summary.json` | Aggregate. |
| V1-RESULT-023 | `results/all_results_summary.json` | Derived index. |

## Figures

| ID | Path | Source |
|---|---|---|
| V1-FIG-001 | `figures/shaped_cache_heatmap.png` | `gen_shaped_cache_figures.py` |
| V1-FIG-002 | `figures/shaped_cache_kv_sensitivity.png` | `gen_shaped_cache_figures.py` |
| V1-FIG-003 | `figures/shaped_cache_pareto.png` | `gen_shaped_cache_figures.py` |
| V1-FIG-004 | `paper_output/figures/fig_eigenvalue_spectrum.pdf` | `paper_output/generate_figures.py` |
| V1-FIG-005 | `paper_output/figures/fig_memory_savings.pdf` | `paper_output/generate_figures.py` |
| V1-FIG-006 | `paper_output/figures/fig_pareto.pdf` | `paper_output/generate_figures.py` |
| V1-FIG-007 | `paper_output/figures/fig_scaling.pdf` | `paper_output/generate_figures.py` |
| V1-FIG-008 | `paper_output/figures/fig_seqlen.pdf` | `paper_output/generate_figures.py` |

## Configs, build, and tests

| ID | Path | Notes |
|---|---|---|
| V1-CONF-001 | `configs/default.yaml` | — |
| V1-CONF-002 | `configs/quick.yaml` | — |
| V1-CONF-003 | `Makefile` | — |
| V1-CONF-004 | `pyproject.toml` | URL still points at `dynamis-labs/spectralquant`; v2 should retarget. |
| V1-TEST-001 | `tests/test_calibration.py` | — |
| V1-TEST-002 | `tests/test_quantization.py` | — |
| V1-TEST-003 | `tests/test_spectral_rotation.py` | — |
| V1-TEST-004 | `tests/test_end_to_end.py` | — |
| V1-TEST-005 | `tests/conftest.py` | — |

## v2 Modal run artifacts (RUN-THREEWAY-*)

These four entries are the paper-valid full-path three-way runs that have completed on Modal as of commit `abcb09197998cc027df688abceae5fb81cfcd31d`. They are sliced benchmark rows (WikiText-103, n_calib=32, n_eval=8, 8 sampled layers, single seed 42), not exhaustive paper claims. The full run configuration, headline numbers, and interpretation guardrails live in `docs/full_matrix_evidence_summary.md`. The canonical artifact path is on the Modal volume `spectralquant-v2-results` under `/results/three_way/`; local staging copies under `modal_artifacts/` are off-repo.

| ID | Model | Bits | TQ cos mean | SQ v1 cos mean | SQ v2 cos mean | TQ ratio | SQ v2 ratio | Modal volume path |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RUN-THREEWAY-MISTRAL-5BIT | mistralai/Mistral-7B-v0.3 | 5 | 0.6556 | 0.9404 | 0.9421 | 3.0843 | 3.0820 | `spectralquant-v2-results:/results/three_way/Mistral-7B-v0.3_b5_calib32_eval8_seed42.json` |
| RUN-THREEWAY-MISTRAL-3BIT | mistralai/Mistral-7B-v0.3 | 3 | 0.6263 | 0.9329 | 0.9327 | 5.0196 | 5.0135 | `spectralquant-v2-results:/results/three_way/Mistral-7B-v0.3_b3_calib32_eval8_seed42.json` |
| RUN-THREEWAY-MISTRAL-2BIT | mistralai/Mistral-7B-v0.3 | 2 | 0.6495 | 0.9035 | 0.9213 | 7.3143 | 7.3012 | `spectralquant-v2-results:/results/three_way/Mistral-7B-v0.3_b2_calib32_eval8_seed42.json` |
| RUN-THREEWAY-QWEN-3BIT    | Qwen/Qwen2.5-7B           | 3 | 0.3986 | 0.7724 | 0.7786 | 5.0196 | 5.0135 | `spectralquant-v2-results:/results/three_way/Qwen2.5-7B_b3_calib32_eval8_seed42.json` |

**Caveats common to all four entries:**

- TurboQuant arm is the **local** reimplementation (`methods.turboquant.label = "local"`); V1-GAP-012 still in force.
- Compression ratios are derived from `src/spectralquant/accounting.py`; they are **not** the v1 paper's 5.95x at b=3 and do **not** unblock V1-GAP-014.
- Single seed (42); V1-GAP-001 multi-seed obligation is **not** unblocked.
- Per-layer aggregates only; per-head min-cosine distributions are not in these artifacts.
- Inline-corpus and synthetic-smoke runs are excluded by construction (`mode = "full"`, `paper_valid = true` enforced before adding to this catalog).

## Discrepancies and gaps (V1-GAP-*)

These are blocking gates: a v2 claim that depends on a gap MUST either be demoted in `docs/claims_discipline.md` or be re-run before publication.

| ID | Summary | Blocks |
|---|---|---|
| V1-GAP-001 | "10-seed" framing with 5 seeds on disk in `neurips_10seed.json`. | Wilcoxon p=0.031 significance claim. |
| V1-GAP-002 | "4 models" framing with 3 Qwen rows in `all_models.json`. | "4 models" headline. |
| V1-GAP-002b | `comparison_results.json` shows SQ losing TQ on every head (3 bits, Qwen 1.5B). | "SQ wins per-head" claim. |
| V1-GAP-003 | Latency conflict: `neurips_latency_crossover.json` (SQ faster) vs `v3_latency.json` (SQ ~50x slower for compression). | "SQ faster than TQ at 512 tokens" headline. |
| V1-GAP-004 | Two d_eff conventions (normalized ≈ 4 vs unnormalized ≈ 35) coexist in different files. | "d_eff/head_dim ≈ 3–4%" claim without convention label. |
| V1-GAP-004b | Identical 13-digit PPL across fp16/TQ/SQ in `v3_perplexity_v2.json`. | "Compression-neutral PPL" claim. |
| V1-GAP-005 | `stability.json` self-declares as reconstructed from logs. | "CV=3.9%" as primary stability evidence. |
| V1-GAP-006 | `phase1_metadata.json` has gate1_status=FAILED. | Direct phase 1 d_eff citations. |
| V1-GAP-007 | "15-second calibration" claim vs 31 s on-disk timing. | "15-second calibration" headline. |
| V1-GAP-008 | LongBench n=5/task. | "LongBench improvement" claim. |
| V1-GAP-009 | NIAH artifacts limited / partly broken. | "NIAH 10/10" headline. |
| V1-GAP-010 | Two SpectralQuant engines coexist in the public namespace. | Reproducibility — different import paths exercise different code. |
| V1-GAP-011 | Gemma row is a 403, not a measurement. | "Gemma 2-9B" measured-architecture claim. |
| V1-GAP-012 | Local TurboQuant baseline reproduction marked failed in at least one phase 0 artifact. | "Beats official TurboQuant" claim. |
| V1-GAP-013 | TQ baselines disagree across `comparison/full_ablation.json` and `comparison/ablation_v2.json`. | Ablation rows reused as TQ baselines. |
| V1-GAP-014 | Compression-ratio formula in spec §10 does not yield 5.95x at b=3, d=3. | Any ratio not produced by `src/spectralquant/accounting.py`. |

## v2 Modal next-stage runs (RUN-PERPLEXITY/LATENCY/GENERATION/LONGBENCH-*)

The next-stage evidence families have been launched on Modal. The
artifacts below were pulled from the Modal volume
`spectralquant-v2-results` and copied into the repo under
`results/v3/modal/`; the canonical authoritative copies remain on
Modal. Numbers are extracted verbatim from the JSONs and must be
re-extracted from those files before any downstream re-use.

All four families ran from commit
`197bcfb4ad54a7d7bc9430a80695c62c145371fd` ("Close LongBench full-path
and end-to-end latency gaps") on a single NVIDIA H200 (Modal).

### RUN-PERPLEXITY-QWEN2.5-7B (paper_valid)

- Local path: `results/v3/modal/perplexity__Qwen2.5-7B__b3_seed42_n1024_tok1024_str512_fp16+spectralquant_v2+turboquant.json`
- Modal path: `spectralquant-v2-results:/results/perplexity/perplexity__Qwen2.5-7B__b3_seed42_n1024_tok1024_str512_fp16+spectralquant_v2+turboquant.json`
- Dataset: WikiText-103-raw-v1 validation, 1024 sequences × 1024 tokens, stride 512, 104,224 tokens scored per method.
- Methods: fp16 / spectralquant_v2 (b=3) / turboquant (b=3).
- Headline (verbatim from `methods.<m>.perplexity`): fp16 = 6.4907; SQv2 = 6.9773; TurboQuant = 2048.5671.
- Replay coverage on non-fp16 methods: 1.0 (28/28 layers calibrated, 0 passthrough).
- `paper_valid = true`, `caveats = []`.

### RUN-LATENCY-QWEN2.5-7B (paper_valid)

- Local path: `results/v3/modal/latency__Qwen2.5-7B__b3_seed42_bs1_ctx512x1024x2048_gen64_wm3_it10_fp16+spectralquant_v2+turboquant.json`
- Modal path: `spectralquant-v2-results:/results/latency/latency__Qwen2.5-7B__b3_seed42_bs1_ctx512x1024x2048_gen64_wm3_it10_fp16+spectralquant_v2+turboquant.json`
- Operating points: batch_size=1, gen_tokens=64, ctx ∈ {512, 1024, 2048}; warmup=3 iters, measured=10 iters; CUDA-event timer.
- Methods: fp16 (production_kernel=true), SQv2 / TurboQuant (microbenchmark=kv_compress_decompress_round_trip AND hooked_replay_end_to_end rows).
- Headline rows (decode ms/tok @ ctx=1024, p50): fp16 = 17.66; SQv2 hooked-replay e2e = 630.58; TurboQuant hooked-replay e2e = 70.72; SQv2 microbench = 0.0593; TurboQuant microbench = 0.0016.
- Headline rows (tokens/sec @ ctx=2048, p50): fp16 = 60.41; SQv2 hooked-replay = 1.59; TurboQuant hooked-replay = 14.12; SQv2 microbench = 34,526; TurboQuant microbench = 1,290,140.
- `paper_valid = true`. Caveat (verbatim from JSON): hooked-replay rows include Python-level per-layer hook overhead and are explicitly NOT a production-kernel claim. Microbench and end-to-end rows must never be compared directly.

### RUN-GENERATION-QWEN2.5-7B (paper_valid)

- Local path: `results/v3/modal/generation__Qwen2.5-7B__b3_seed42_t0.00_new128_fp16+spectralquant_v2+turboquant.json`
- Modal path: `spectralquant-v2-results:/results/generation/generation__Qwen2.5-7B__b3_seed42_t0.00_new128_fp16+spectralquant_v2+turboquant.json`
- Decoding: do_sample=False, temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=128, seed=42; 8 prompt set across summarize/QA/code/math/long-context/reasoning.
- Methods: fp16 / SQv2 (b=3) / TurboQuant (b=3).
- Headline (`methods.<m>.metrics`): mean_token_overlap_f1 vs FP16 self = 1.0; SQv2 = 0.4817; TurboQuant = 0.1197. mean_distinct_2: fp16 = 0.768, SQv2 = 0.603, TurboQuant = 0.301 (TurboQuant degenerates into repetitive output).
- Replay coverage: 1.0 (28/28 layers calibrated) for both compressed methods.
- `paper_valid = true`, `caveats = []`.

### RUN-LONGBENCH-QWEN2.5-7B-DETERMINISTIC (paper_valid)

- **Status: COMPLETED, `paper_valid=true`.** A 12 h-capped relaunch
  from commit `1ecb578a0b0251f1a716469e51be4303c7191cd6` started
  `2026-04-30T16:26:17Z` (Modal app `ap-GHvmUwex1Hoav4BRrnTtVi`,
  function call `fc-01KQFKCFNP61M33NMBN7CQ4DQB`) and finished
  `2026-04-30T22:04:00Z` — wall-clock ≈ 5 h 38 min, well inside the
  12 h kill-switch (`SPECTRALQUANT_MODAL_TIMEOUT_LONGBENCH_SEC=43200`).
  The earlier 6 h-capped attempt at `2026-04-30T10:10Z` (commit
  `6154175`) was kill-switched without writing a canonical JSON; the
  patched harness from commit `1ecb578` writes per-method full-record
  shards as each method finishes, so a future re-launch is recoverable
  even if it again hits a wall-clock cap.
- Local canonical: `results/v3/modal/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant.json` (also mirrored at `results/v3/modal/longbench_relaunch_2026-04-30/canonical/`).
- Modal canonical: `spectralquant-v2-results:/results/longbench/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant.json`.
- Status / events / shards (post-stop snapshot 2026-04-30T22:16Z):
  `results/v3/modal/longbench_relaunch_2026-04-30/snapshots/20260430T221642Z_post_stop/{status.json,events.jsonl,partial/method__<m>.json,partial/<m>__<task>.json}` (855 events; 3 method-record shards; 15 per-task progress shards).
- Configuration: subset=deterministic (5 tasks: narrativeqa, qasper, hotpotqa, gov_report, trec); `n_per_task=50`, `max_input_tokens=8192`, `max_new_tokens=128`; methods fp16 + spectralquant_v2 + turboquant; calibration knobs `paper` mode, `n_calib=16`, `lloyd_max_iter=200`, `calib_max_seq_tokens=512` (all clear the `paper_valid_thresholds`).
- Replay coverage: `fraction_layers_real = 1.0` for both compressed methods (28/28 layers calibrated, 0 passthrough hook calls) — clears the `≥ 0.99` paper-valid gate.
- Hardware: NVIDIA H200, CUDA 13.0, torch 2.11.0+cu130, transformers 5.7.0, datasets 4.8.5, python 3.12.1.

**Macro scores (verbatim from `methods.<m>.aggregate.macro_score`).**

| Method | macro_score |
|---|---:|
| fp16 | **0.1755** |
| spectralquant_v2 (b=3) | **0.2004** |
| turboquant (b=3) | **0.0044** |

**Per-task scores (verbatim from `methods.<m>.per_task[*]`; `score_0_100` shown).**

| Task | metric | n | fp16 | spectralquant_v2 | turboquant |
|---|---|---:|---:|---:|---:|
| narrativeqa | qa_f1_en | 50 | 13.61 | 16.48 | 0.47 |
| qasper | qa_f1_en | 50 | 17.12 | **32.74** | 0.00 |
| hotpotqa | qa_f1_en | 50 | 27.13 | 27.35 | 0.25 |
| gov_report | rouge_en | 50 | 15.87 | 13.64 | 1.49 |
| trec | classification_em | 50 | 14.00 | 10.00 | 0.00 |

**Reading.** SpectralQuant v2 macro-beats FP16 (+0.0250 absolute = +14.2 % relative) on this deterministic subset, driven mainly by qasper (+15.6 pts qa_f1) and a small narrativeqa gain (+2.9 pts). Spectralquant v2 loses on gov_report (−2.2 pts ROUGE) and trec (−4.0 pts classification_em). TurboQuant b=3 collapses to ≈ 0 on every task — consistent with the perplexity 2 049 result and the generation-degradation result already in the catalog (`RUN-PERPLEXITY-QWEN2.5-7B`, `RUN-GENERATION-QWEN2.5-7B`).

**Caveats (mandatory, must travel with any cited number).**

1. `caveats[]` in JSON: *"subset=deterministic: this artifact scores a *transparent subset* of LongBench tasks (['narrativeqa', 'qasper', 'hotpotqa', 'gov_report', 'trec']); do not headline it as full LongBench."* — verbatim per `docs/claims_discipline.md` §5.2.5.
2. Single seed (42), single model (Qwen2.5-7B), single bit budget (b=3), single hardware class (NVIDIA H200 / Modal). No multi-seed CI; the +0.0250 macro delta has no error bar.
3. n=50 per task; 5 tasks of LongBench's 21. Full 21-task LongBench (`RUN-LONGBENCH-QWEN-FULL`) remains unblocked.
4. TurboQuant arm is the **in-repo local re-implementation** (`methods.turboquant.label = "turboquant"`); V1-GAP-012 (official Google TurboQuant) is **not** unblocked.
5. The K/V hooked-replay path costs ≈ 3.66 h of wall-clock for the SQv2 5-task subset on H200 — that is the same per-layer-Python-callback overhead surfaced in `RUN-LATENCY-QWEN2.5-7B` and is **not** a production-kernel claim.

## v2 next-stage evaluation harnesses (planned, not yet evidence)

The four next-stage evidence families ship as harnesses under
`experiments/run_perplexity.py`, `experiments/run_longbench.py`,
`experiments/run_generation.py`, `experiments/run_latency.py`, with
matching schemas under `schemas/{perplexity,longbench,generation,latency}.schema.json`
and a unified Modal launcher at `scripts/launch_modal_eval.py`. The
runbook entry for the exact commands is `docs/execution_audit_and_modal_runbook.md` §7.7.

These IDs are RESERVED — they will be populated only after real Modal
runs land schema-valid JSON on the volume. **No `RUN-*` ID below has
a result file yet.**

| Reserved ID | Family | Planned config | Required to mark done |
|---|---|---|---|
| RUN-PERPLEXITY-QWEN-3BIT  | perplexity | Qwen2.5-7B b=3 fp16, n=64, 1024-tok stride 512 | JSON validates against `perplexity.schema.json`, `paper_valid=true`. |
| RUN-PERPLEXITY-MISTRAL-{2,3,5}BIT | perplexity | Mistral-7B-v0.3 each bits-budget | same as above per row. |
| RUN-LONGBENCH-QWEN-DETERMINISTIC | longbench | Qwen2.5-7B `subset=deterministic`, `n_per_task=50`, full path on `THUDM/LongBench` HF dataset + vendored metrics in `experiments/longbench_metrics.py` | `paper_valid=true`, every method record `dataset_source=huggingface_thudm`. Subsets smaller than `full` carry an explicit "transparent subset" caveat — never headlined as full LongBench. |
| RUN-LONGBENCH-QWEN-FULL | longbench | Qwen2.5-7B `subset=full`, `n_per_task=50` | `paper_valid=true`; full 21-task LongBench on the HF dataset. |
| RUN-GENERATION-QWEN  | generation | Qwen2.5-7B greedy on 8-prompt set | JSON validates; FP16 reference completions present. |
| RUN-LATENCY-QWEN-FP16 | latency | Qwen2.5-7B, ctx ∈ {512,1024,2048}, gen=64, ≥10 measured iters, FP16 only | `device=cuda`, `timer=torch.cuda.Event`, `paper_valid=true`. |
| RUN-LATENCY-QWEN-MICROBENCH-V2 | latency | v2 / TurboQuant K/V compress+decompress microbenchmark | per-row `microbenchmark=true / microbenchmark_kind=kv_compress_decompress_round_trip`. NOT end-to-end inference latency. |
| RUN-LATENCY-QWEN-E2E-REPLAY-V2 | latency | v2 / TurboQuant **hooked-replay end-to-end** forward+decode | per-row `end_to_end_measured=true / production_kernel=false / measurement_kind=hooked_replay_end_to_end`. Reportable as "hooked replay end-to-end latency", NOT "production speedup". |

**Discipline.** Until each ID's JSON is on disk and validates, the
corresponding row in any v2 report draft must say "not yet measured"
and cite the row above. Synthetic-smoke and inline-corpus-smoke
artifacts always carry `paper_valid=false` and a caveat — they exist
for harness validation only and must NOT be cited as evidence.

The four schemas are syntactically validated by `tests/test_result_schema.py`.
The harnesses themselves are exercised by `tests/test_eval_harnesses.py`.

## Supporting notes (interpretive, NOT paper-valid)

Supporting notes are interpretive / mechanistic write-ups whose claims may
inform the manuscript's narrative but are **not** schema-validated artifacts.
They are registered here so any reuse travels with the supporting-note caveat.

| ID | Path | Origin | Claims supported in the consolidated paper | Numbers in note that are NOT paper-valid |
|---|---|---|---|---|
| `SUPPORT-NOTE-PERPLEXITY-MECHANISM-2026-04` | `docs/supporting_notes/perplexity_mechanism_note_2026-04.md` | Sentra technical note, Vangara & Gopinath, April 2026 (preserved verbatim with provenance header) | The four-mechanism decomposition (calibrated rotation; two-regime allocation; selective QJL; water-filling), and the qualitative shape of the data-oblivious-arm collapse at low bit budgets. Used in the consolidated paper's `Mechanism behind the perplexity gap` subsection (§Interpretation). | The Mistral-7B-v0.3 / WikiText-103 perplexity table (12.22 / 12.36 / 12.33 / 12.29 at b=5; 14.88 / 13.24 / 13.14 at b=3; 82.13 / 25.13 / 26.71 at b=2). No schema-validated `results/v3/modal/perplexity__Mistral-7B-v0.3_*.json` exists at the time of capture. The "15-second calibration" is qualitative; paper-valid amortization is V1-GAP-007. The "ESM-2 / ViT" universality sentence is narrowed to the LLM models present in V1-RESULT-001 when used in the paper. |

## v2 obligations

For Phase 1 (this milestone), the catalog must be:

1. Complete — every v1 artifact has an entry.
2. Honest — every gap and contradiction is recorded.
3. Schema-validated — `tests/test_result_schema.py` validates `docs/evidence_catalog.json` against `schemas/evidence_catalog.schema.json`.

Subsequent v2 phases will:

- Resolve gaps by re-running, demoting, or excluding.
- Add `RUN-THREEWAY-*`, `RUN-WATERFILL-*`, and `RUN-ACCOUNTING-*` IDs in `results/three_way/`, `results/waterfill_ablation/`, and `results/accounting_audit/`.
- Require every empirical sentence in the v2 report to cite at least one ID from this catalog.
