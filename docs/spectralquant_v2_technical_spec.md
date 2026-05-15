# SpectralQuant v2 Technical Specification

**Working title:** 3% Is All You Need, Revisited  
**Subtitle:** Water-filling the semantic subspace for production-model KV-cache compression  
**Authors:** Anirudh B. Vangara and Ashwin Gopinath  
**Status:** Execution specification and technical-report blueprint  
**Target repository:** `niashwin/spectralquant-v2` private repository  
**Source baseline:** `Dynamis-Labs/spectralquant` must remain untouched  
**Current intended artifact:** one consolidated Markdown specification that can become both the v2 implementation plan and the v2 technical report scaffold  

## 1. Purpose

This document specifies the full SpectralQuant v2 project from scratch. It consolidates the original SpectralQuant v1 story, the new SpectralQuant v2 water-filling extension, the implementation plan, the unit-testing plan, the benchmarking plan, the evidence-catalog plan, and the technical-report narrative into one executable document. The intended outcome is a private `niashwin/spectralquant-v2` repository that starts from the existing `Dynamis-Labs/spectralquant` codebase, implements v2 cleanly, reproduces every claim from Anirudh’s April 2026 production-model report, and produces a publishable technical report with evidence traceability for every figure, table, and sentence-level empirical claim.

The central story is: KV-cache memory is not isotropic noise. It contains a tiny, structured, meaning-bearing subspace, and once that structure is measured, the compression budget should be spent according to the spectrum rather than uniformly across all coordinates. TurboQuant is the best-known data-oblivious baseline for this style of vector quantization; it uses random rotation, scalar quantization, and QJL residual correction to address MSE and inner-product distortion. SpectralQuant v1 replaces random rotation with calibrated spectral rotation and uses selective QJL only where the semantic signal lives. SpectralQuant v2 goes one step further: even inside the semantic subspace, only a few directions dominate, so v2 water-fills the bit budget across semantic eigen-directions.

The working paper frame is “3% Is All You Need, Revisited.” In v1, the message was that roughly 3% of the head dimension can carry most of the useful key signal. In v2, the message becomes sharper: not only is the useful subspace tiny, but the budget inside that subspace should be allocated according to the eigenvalue spectrum. The goal is to show that this is not a cosmetic improvement over TurboQuant, but a principled, measurable, reproducible advantage on production-scale models.

## 2. Repository Policy

No work should be done in `Dynamis-Labs/spectralquant`. The original repository is the v1 baseline and evidence source. The v2 implementation should occur only in this private repository:

```text
niashwin/spectralquant-v2
```

The first commit should be an untouched import of the original repository. This document should be the first intentional v2 change. Implementation should proceed only after this specification is committed.

## 3. One-Paragraph Technical Abstract

SpectralQuant v2 is a calibrated KV-cache compression method for transformer inference. It begins with the SpectralQuant v1 observation that key vectors have low effective dimension after per-layer, per-KV-head covariance analysis. Instead of applying a data-oblivious random rotation and full-dimensional QJL residual correction, SpectralQuant rotates keys into their calibrated eigenbasis, preserves a small semantic subspace, quantizes the tail cheaply, and applies QJL only to the semantic residual. SpectralQuant v2 adds a water-filling allocator over the semantic eigenvalues: each semantic dimension receives an integer number of MSE quantization bits according to the marginal rule \( \lambda_i / 4^{b_i} \), preserving the same semantic bit budget while improving reconstruction quality. The target result is a three-way comparison against TurboQuant and SpectralQuant v1 on Mistral-7B-v0.3 and Qwen2.5-7B, showing that v2 improves attention-output cosine most strongly at aggressive bit budgets and heavy-GQA architectures.

## 4. Background and Motivation

Autoregressive transformer inference stores key and value tensors from prior tokens in the KV cache, allowing later decoding steps to reuse previous attention state rather than recomputing it from scratch. KV-cache memory grows with sequence length, number of layers, number of heads, and head dimension, which makes cache management central to long-context and high-throughput serving.

TurboQuant is the direct baseline because it addresses both mean-squared vector distortion and biased inner-product estimation through a two-stage method: an MSE quantizer followed by a 1-bit Quantized Johnson-Lindenstrauss residual transform. The QJL line of work is important because attention depends on inner products, not only on reconstructing vectors under MSE. KVQuant is also relevant background because it demonstrates that KV-cache distributions are structured enough to benefit from pre-RoPE key quantization, non-uniform quantization, and custom kernels.

SpectralQuant’s premise is more specific than generic KV-cache quantization. It claims that the key cache has a tiny effective semantic subspace and that compression should exploit this measured spectral structure. The v1 repository already contains a paper artifact, core implementation modules, experiment scripts, figures, and result JSONs supporting the first version of this claim in `paper_output/spectralquant.tex`, `paper_output/spectralquant.pdf`, `src/spectralquant/`, `experiments/`, and `results/`.

## 5. The Consolidated Story

### 5.1 The v1 story

The v1 story is: random rotations are robust, but they are deliberately blind. TurboQuant uses data-oblivious structure to make coordinates well-behaved in expectation, then repairs inner-product bias with a full-dimensional QJL residual. SpectralQuant v1 says this leaves performance on the table for real LLM key caches because real keys are not isotropic. They have concentrated eigen-spectra, and the correct rotation is not random; it is the calibrated eigenbasis of each layer and KV head.

In v1, each layer/head calibration collects key vectors, normalizes each vector, computes the covariance matrix, eigendecomposes it, and estimates the participation-ratio effective dimension. The first \(d_{\text{eff}}\) eigen-directions are treated as the semantic subspace. The rest are treated as tail dimensions. The semantic subspace receives high-fidelity quantization and selective QJL residual correction. The tail receives cheaper MSE quantization and no QJL residual correction.

The v1 repository evidence map identifies the main implementation modules as `calibration.py`, `spectral_rotation.py`, `nonuniform_quantization.py`, `selective_qjl.py`, `engine.py`, `spectralquant.py`, `metrics.py`, and `utils.py`. It also identifies v1 paper and result artifacts under `paper_output/`, `results/memory_efficiency/`, `results/neurips/`, `results/v3/`, `results/eigenspectral/`, and `results/comparison/`.

### 5.2 The v2 story

The v2 story is: v1 correctly identifies the semantic subspace, but v1 still treats the semantic dimensions too uniformly. If the semantic eigenvalues look like:

```text
lambda_0 = 0.80
lambda_1 = 0.12
lambda_2 = 0.05
lambda_3 = 0.02
```

then allocating the same number of MSE bits to every semantic dimension is suboptimal. The dominant dimension carries far more variance and should receive more resolution. The least dominant semantic dimension can often tolerate fewer bits without materially changing the attention output. SpectralQuant v2 keeps the v1 calibrated eigenbasis, the v1 semantic/tail split, and the v1 selective QJL logic, but replaces uniform semantic MSE bit allocation with greedy water-filling.

The v2 message is therefore not that v1 was wrong. The message is that v1 discovered the semantic subspace, and v2 learns how to spend bits inside it.

### 5.3 The paper thesis

The technical report should make one thesis unavoidable:

```text
KV-cache compression should be spectrum-aware.
```

TurboQuant is data-oblivious and full-dimensional. SpectralQuant v1 is spectrum-aware at the subspace level. SpectralQuant v2 is spectrum-aware at the within-subspace allocation level. The report should show this progression clearly and avoid redundant explanations.

## 6. Target Claims

The v2 project should aim to support the following empirical claims only after reproduction:

| Claim | Evidence required | Status before v2 repo work |
|---|---|---|
| SQ v2 beats the local TurboQuant baseline on every tested Mistral and Qwen operating point | `results/three_way/*.json` plus generated summary table | Target claim from Anirudh report |
| SQ v2 improves most at lower bit budgets | Mistral 5-bit, 3-bit, 2-bit sweep | Target claim from Anirudh report |
| SQ v2 improves more on heavier GQA models | Mistral 4:1 GQA vs Qwen 7:1 GQA comparison | Target claim from Anirudh report |
| TurboQuant has catastrophic low-bit head failures under this local baseline | Per-head/per-layer min-cosine distributions | Target claim from Anirudh report |
| Water-filling is backward-compatible with v1 | Unit tests and `use_water_fill=False` reproducing v1 behavior | Must be implemented |
| Water-filling preserves the same semantic bit budget as v1 | Compression-accounting tests | Must be implemented |
| All tables in the report trace to JSON artifacts | Evidence catalog and JSON schema validation | Must be implemented |

The report should not claim full production readiness, official TurboQuant superiority, perplexity improvement, LongBench improvement, or universal architecture generalization until those experiments are run.

## 7. Existing v1 Evidence Catalog

The v2 repo must begin by cataloging evidence already present in the v1 repo. The catalog should be stored as:

```text
docs/evidence_catalog.md
docs/evidence_catalog.json
```

The catalog must include at least the following entries:

| Evidence ID | Existing path | Supports | Notes and caveats |
|---|---|---|---|
| V1-PAPER-001 | `paper_output/spectralquant.tex` | v1 narrative, method, figures, paper structure | 1540-line LaTeX source identified in repo evidence map |
| V1-PAPER-002 | `paper_output/spectralquant.pdf` | compiled v1 report | Must be treated as compiled output, not source of truth |
| V1-README-001 | `README.md` lines around canonical claim/script/JSON table | canonical v1 claim mapping | Must be checked after clone for exact line numbers |
| V1-IMPL-001 | `src/spectralquant/calibration.py` | PCA, \(d_{\text{eff}}\), collector hooks | Need inspect exact APIs before v2 changes |
| V1-IMPL-002 | `src/spectralquant/spectral_rotation.py` | eigenbasis vs Haar/random rotation | Needed for v2 spectral basis |
| V1-IMPL-003 | `src/spectralquant/nonuniform_quantization.py` | Lloyd-Max and bit allocator | v2 likely extends this for per-dim codebooks |
| V1-IMPL-004 | `src/spectralquant/selective_qjl.py` | selective QJL implementation | v2 should preserve this logic |
| V1-IMPL-005 | `src/spectralquant/engine.py` | current engine class | Evidence map says this subclasses `TurboQuantEngine` |
| V1-IMPL-006 | `src/spectralquant/spectralquant.py` | legacy standalone engine | Evidence map flags duplicate engine class risk |
| V1-RESULT-001 | `results/memory_efficiency/all_models.json` | Qwen 14B headline memory/cosine numbers | Evidence map reports TQ 0.9226 vs SQ 0.9485 and 5.02x to 5.95x |
| V1-RESULT-002 | `results/neurips/neurips_kv_asymmetry.json` | key/value effective dimension asymmetry | Evidence map reports Qwen 1.5B key \(d_{\text{eff}}\) 3.95 and value \(d_{\text{eff}}\) 40.34 |
| V1-RESULT-003 | `results/neurips/neurips_latency_crossover.json` | latency speedups | Evidence map reports SQ 0.257 ms vs TQ 0.566 ms at 512 tokens, but flags contradictions |
| V1-RESULT-004 | `results/v3/v3_perplexity*.json` or related files | perplexity parity | Evidence map reports fp16/tq/sq identical PPL for certain Qwen/Llama runs |
| V1-RESULT-005 | `results/calibration_stability/stability.json` | calibration stability | Evidence map says file is reconstructed from logs, so use cautiously |
| V1-GAP-001 | `results/neurips/neurips_10seed.json` | seed CI | Evidence map says only 5 seeds appear on disk despite “10-seed” language |
| V1-GAP-002 | `results/comparison/*.json` | negative ablations | Evidence map flags negative results that must be explained or excluded explicitly |

The evidence catalog should never hide discrepancies. Instead, it should mark them as validation gates. The v2 report is stronger if it shows disciplined evidence handling rather than selectively quoting only positive results.

## 8. Known v1 Issues That v2 Must Fix

The v2 repo must explicitly address these issues from the evidence map:

1. The v1 repo appears to contain a “10-seed CI” claim with only 5 seeds on disk.
2. A headline “4 models” table may only contain 3 Qwen sizes in `all_models.json`.
3. Latency evidence conflicts across `neurips_latency_crossover.json` and `v3_latency.json`.
4. \(d_{\text{eff}}\) methodology differs between normalized per-head results and unnormalized eigenspectral summaries.
5. TurboQuant baseline reproduction appears marked as failed in one artifact.
6. “15-second calibration” may not match the closest on-disk timing number.
7. NIAH evidence appears limited and some earlier artifacts may be broken.
8. LongBench evidence is too small and potentially signal-free.
9. Negative comparison artifacts exist and must be explained.
10. Two `SpectralQuantEngine` classes may exist in the public namespace.

The v2 report should not inherit any unresolved ambiguity from v1. It should either reproduce a claim cleanly, demote it to background, or remove it from the headline.

## 9. SpectralQuant v2 Algorithm

### 9.1 Calibration

For every layer \( \ell \) and KV head \( h \), collect key vectors:

\[
X_{\ell,h} \in \mathbb{R}^{n \times D}
\]

Each row must be normalized:

\[
\tilde{x}_j = \frac{x_j}{\|x_j\|_2 + \epsilon}
\]

Compute covariance:

\[
C_{\ell,h} = \frac{1}{n} \tilde{X}_{\ell,h}^{\top} \tilde{X}_{\ell,h}
\]

Eigendecompose:

\[
C_{\ell,h} = V_{\ell,h} \Lambda_{\ell,h} V_{\ell,h}^{\top}
\]

Sort eigenvalues in descending order:

\[
\lambda_0 \geq \lambda_1 \geq \dots \geq \lambda_{D-1}
\]

Compute participation ratio:

\[
d_{\text{eff,float}} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2 + \epsilon}
\]

Clamp the integer effective dimension:

\[
d_{\text{eff}} = \mathrm{clamp}(\lceil d_{\text{eff,float}} \rceil, 2, D - 2)
\]

The implementation must store both the raw floating estimate and the integer \(d_{\text{eff}}\).

### 9.2 v1 semantic split

Rotate a normalized key vector:

\[
z = V_{\ell,h}^{\top} k
\]

Split:

```text
semantic = z[0:d_eff]
tail     = z[d_eff:D]
```

v1 quantizes the semantic region with uniform high-fidelity MSE bits and the tail with cheaper bits. v1 applies QJL only to the semantic residual. This behavior must remain exactly reproducible with:

```python
use_water_fill = False
```

### 9.3 v2 water-filling

v2 takes the semantic eigenvalues:

```text
lambda_sem = eigenvalues[:d_eff]
```

and allocates a fixed total semantic MSE bit budget:

```text
B_semantic = b_high * d_eff
```

Instead of:

```text
[b_high, b_high, ..., b_high]
```

v2 produces:

```text
[b_0, b_1, ..., b_{d_eff-1}]
```

subject to:

\[
\sum_i b_i = B_{\text{semantic}}
\]

The greedy marginal allocation rule is:

\[
i^\* = \arg\max_i \frac{\lambda_i}{4^{b_i}}
\]

After selecting \(i^\*\):

\[
b_{i^\*} \leftarrow b_{i^\*} + 1
\]

Repeat until all semantic bits are allocated.

### 9.4 Water-filling pseudocode

```python
def allocate_waterfill_bits(
    eigenvalues,
    total_bits,
    min_bits=0,
    max_bits=None,
    eps=1e-12,
):
    eig = np.asarray(eigenvalues, dtype=np.float64)
    if eig.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    if len(eig) == 0:
        raise ValueError("eigenvalues must be non-empty")
    if np.any(eig < 0):
        raise ValueError("eigenvalues must be non-negative")

    d = len(eig)
    if total_bits < d * min_bits:
        raise ValueError("total_bits cannot satisfy min_bits")
    if max_bits is not None and total_bits > d * max_bits:
        raise ValueError("total_bits exceeds max_bits")

    eig_safe = np.maximum(eig, eps)
    bits = np.full(d, min_bits, dtype=np.int64)
    remaining = total_bits - d * min_bits

    for _ in range(remaining):
        scores = eig_safe / (4.0 ** bits)
        if max_bits is not None:
            scores = np.where(bits >= max_bits, -np.inf, scores)
        i = int(np.argmax(scores))
        bits[i] += 1

    return bits
```

Tie-breaking must be deterministic. The default should select the lowest-index dimension when scores tie.

### 9.5 Per-dimension codebooks

Each semantic dimension receives its own Lloyd-Max codebook:

```text
bits_i = waterfill_bits[i]
sigma_i = sqrt(lambda_i)
codebook_i = LloydMaxNormal(bits=bits_i, mean=0, std=sigma_i)
```

The report language “fit to \(N(0, \sqrt{\lambda_i})\)” must be implemented carefully. The distribution should be parameterized as:

```text
mean = 0
variance = lambda_i
standard deviation = sqrt(lambda_i)
```

The v2 implementation must record each head’s allocation vector, eigenvalues, and codebook metadata in the run output.

## 10. Compression Accounting

Compression accounting is a blocking validation gate. The v2 report claims ratios such as:

```text
TurboQuant @ 3 bits: 5.02x
SpectralQuant @ 3 bits: 5.95x
```

The simple appendix formula in the pasted report gives:

```text
TurboQuant:
  K = (b - 1) * 128 + 128 + 32 = 128b + 32
  V = 128b + 16
  average slot = 128b + 24
  ratio = 2048 / (128b + 24)
```

and:

```text
SpectralQuant:
  K = 128b + d + 32
  V = 128b + 16
  average slot = 128b + (d + 48) / 2
  ratio = 2048 / (128b + (d + 48) / 2)
```

However, the second formula does not yield 5.95x for \(b=3, d=3\). Therefore, the implementation must not hard-code the report’s ratios without deriving them from the actual stored bits. A dedicated accounting module must be added:

```text
src/spectralquant/accounting.py
```

Required API:

```python
@dataclass
class CompressionAccounting:
    method: str
    avg_bits_arg: int
    head_dim: int
    d_eff: int | None
    k_mse_bits: float
    k_qjl_bits: float
    k_norm_bits: float
    v_mse_bits: float
    v_norm_bits: float
    total_k_bits: float
    total_v_bits: float
    average_slot_bits: float
    fp16_slot_bits: float
    compression_ratio: float
    formula_version: str
```

Required tests:

```text
tests/test_accounting.py
```

The test suite must include:

1. TurboQuant 3-bit accounting equals approximately 5.02x under the documented formula.
2. TurboQuant 5-bit accounting equals approximately 3.08x under the documented formula.
3. SpectralQuant accounting matches the actual implemented bit layout.
4. v1 and v2 have equal compression ratios when only water-filling changes semantic allocation.
5. The code fails loudly if a headline ratio is requested but cannot be derived.

## 11. Implementation Architecture

### 11.1 New modules

```text
src/spectralquant/waterfill.py
src/spectralquant/accounting.py
src/spectralquant/three_way.py
```

### 11.2 Existing modules to extend

```text
src/spectralquant/nonuniform_quantization.py
src/spectralquant/engine.py
src/spectralquant/calibration.py
src/spectralquant/selective_qjl.py
src/spectralquant/metrics.py
```

### 11.3 New experiment scripts

```text
experiments/run_three_way.py
experiments/run_waterfill_ablation.py
experiments/run_deff_stats.py
experiments/run_compression_accounting_audit.py
experiments/plot_three_way.py
```

### 11.4 New docs

```text
docs/spectralquant_v2_technical_spec.md
docs/reproduction.md
docs/evidence_catalog.md
docs/claims_discipline.md
docs/result_schema.md
```

### 11.5 New results directories

```text
results/three_way/
results/waterfill_ablation/
results/deff_stats/
results/accounting_audit/
results/report_figures/
```

## 12. Unit Testing Specification

### 12.1 Water-filling tests

Create:

```text
tests/test_waterfill.py
```

Required tests:

| Test | Purpose |
|---|---|
| `test_allocation_sums_to_budget` | Ensures \( \sum_i b_i = B \) |
| `test_equal_eigenvalues_uniform_allocation` | Ensures symmetry behaves sensibly |
| `test_concentrated_spectrum_allocates_to_first_dim` | Ensures dominant eigenvalue receives more bits |
| `test_min_bits_respected` | Ensures lower bounds work |
| `test_max_bits_respected` | Ensures upper bounds work |
| `test_invalid_negative_eigenvalues_raise` | Prevents silent invalid inputs |
| `test_zero_eigenvalues_do_not_nan` | Ensures numerical stability |
| `test_deterministic_tie_breaking` | Ensures reproducibility |
| `test_numpy_and_torch_inputs_match` | Ensures input backend consistency |
| `test_input_not_mutated` | Prevents side effects |

### 12.2 Quantization tests

Create:

```text
tests/test_v2_quantization.py
```

Required tests:

1. Per-dim codebook count equals \(d_{\text{eff}}\).
2. Per-dim codebook bit widths match `waterfill_bits`.
3. Codebook standard deviations equal \(\sqrt{\lambda_i}\) within tolerance.
4. Encoding and decoding preserve shape.
5. Uniform allocation reproduces v1 semantic quantization when water-filling is disabled.
6. No dimension receives an invalid bit width.
7. Quantization handles \(d_{\text{eff}}=2\).
8. Quantization handles \(d_{\text{eff}}=D-2\).

### 12.3 Calibration tests

Create:

```text
tests/test_calibration_v2.py
```

Required tests:

1. Key row normalization produces unit norms.
2. Covariance matrix is symmetric.
3. Eigenvalues are sorted descending.
4. Eigenvectors are orthonormal within tolerance.
5. \(d_{\text{eff}}\) is clamped to `[2, D - 2]`.
6. Calibration artifact can be saved and loaded.
7. Pre-RoPE hook mode is explicitly recorded.
8. Layer/head metadata is correct.

### 12.4 Engine tests

Create:

```text
tests/test_engine_v2.py
```

Required tests:

1. `use_water_fill=False` reproduces v1 allocation exactly.
2. `use_water_fill=True` changes allocation when eigenvalues are non-uniform.
3. v1 and v2 have equal total semantic MSE bits.
4. v1 and v2 have equal selective QJL dimensions.
5. Compression and decompression preserve tensor shapes.
6. Attention scoring returns finite logits.
7. Causal masking is applied outside or inside attention scoring consistently.
8. The engine does not silently mix normalized and unnormalized keys.

### 12.5 TurboQuant baseline tests

Create:

```text
tests/test_turboquant_baseline.py
```

Required tests:

1. Random orthogonal matrix is orthonormal.
2. Seeded rotation is deterministic.
3. Full-dimensional QJL signs have dimension \(D\).
4. TurboQuant 3-bit accounting returns approximately 5.02x under the chosen formula.
5. TurboQuant attention scoring produces finite logits.
6. TurboQuant baseline is explicitly labeled as local if not official Google code.

### 12.6 JSON schema tests

Create:

```text
tests/test_result_schema.py
```

Every result JSON must validate against:

```text
schemas/three_way_result.schema.json
schemas/accounting.schema.json
schemas/evidence_catalog.schema.json
```

The schema must require:

1. model name,
2. model architecture metadata,
3. hardware metadata,
4. software metadata,
5. dataset metadata,
6. calibration metadata,
7. method metrics,
8. compression accounting,
9. per-layer table,
10. evidence IDs,
11. repo commit hash,
12. timestamp,
13. command-line invocation.

## 13. Benchmarking Specification

### 13.1 Primary benchmark

The primary benchmark is a three-way attention-output cosine comparison:

```text
TurboQuant
SpectralQuant v1
SpectralQuant v2
```

For each method, compare:

```text
attn_fp16 = softmax(Q K^T / sqrt(D)) V
attn_compressed = softmax(method.attention_scores(Q, compress(K)) / sqrt(D)) decompress(V)
quality = mean tokenwise cosine(attn_fp16, attn_compressed)
```

### 13.2 Models

Required initial models:

| Model | Layers | Q heads | KV heads | Head dim | GQA |
|---|---:|---:|---:|---:|---:|
| `mistralai/Mistral-7B-v0.3` | 32 | 32 | 8 | 128 | 4:1 |
| `Qwen/Qwen2.5-7B` | 28 | 28 | 4 | 128 | 7:1 |

### 13.3 Dataset

Use:

```text
WikiText-103
```

Calibration:

```text
n_calib = 32
max_tokens = 384
```

Evaluation:

```text
n_eval = 8
disjoint from calibration
```

### 13.4 Layer sampling

Mistral:

```text
0, 4, 8, 12, 16, 20, 24, 28
```

Qwen:

```text
0, 3, 6, 9, 12, 15, 18, 21
```

### 13.5 Required runs

```bash
for B in 2 3 5; do
  python3 experiments/run_three_way.py \
    --model mistralai/Mistral-7B-v0.3 \
    --avg-bits $B \
    --n-calib 32 \
    --n-eval 8 \
    --n-layers-sample 8 \
    --output-dir results/three_way
done
```

```bash
python3 experiments/run_three_way.py \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 \
  --n-calib 32 \
  --n-eval 8 \
  --n-layers-sample 8 \
  --output-dir results/three_way
```

### 13.6 Target result table

These are target reproduction values from Anirudh’s report, not hard-coded outputs:

| Model | Bits | TQ Compression | SQ v2 Compression | TQ Attn Cos | SQ v2 Attn Cos | SQ v2 minus TQ | SQ v2 minus v1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mistral-7B-v0.3 | 5 | 3.08x | 3.07x | 0.9904 | 0.9938 | +0.34 pp | -0.001 pp |
| Mistral-7B-v0.3 | 3 | 5.02x | 5.95x | 0.8975 | 0.9374 | +3.98 pp | +0.10 pp |
| Mistral-7B-v0.3 | 2 | 8.0x | 9.5x | 0.7108 | 0.8206 | +10.98 pp | +0.54 pp |
| Qwen2.5-7B | 3 | 5.02x | 5.95x | 0.7465 | 0.8427 | +9.62 pp | +0.78 pp |

### 13.7 Additional benchmarks before publication

Before any public claim that v2 is broadly superior, add:

1. Perplexity on WikiText-2 and C4.
2. NIAH at 4k, 8k, 16k, and 32k.
3. LongBench with at least 50 examples per task.
4. RULER if feasible.
5. End-to-end decoding throughput at batch sizes 1, 4, 8, and 16.
6. Latency breakdown for compression, score computation, decompression, and total decode.
7. Calibration time and amortization curves.
8. Official TurboQuant comparison if code is available.

## 14. Result Schema

Every benchmark output must include:

```json
{
  "run_id": "mistral-7b-v0.3_bits3_seed42",
  "timestamp": "2026-04-29T00:00:00Z",
  "repo": "niashwin/spectralquant-v2",
  "commit": "...",
  "command": "...",
  "model": {
    "name": "mistralai/Mistral-7B-v0.3",
    "layers": 32,
    "q_heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "gqa_ratio": 4
  },
  "hardware": {
    "gpu": "NVIDIA H200",
    "cuda": "12.4"
  },
  "software": {
    "python": "...",
    "torch": "2.6",
    "transformers": "5.6",
    "datasets": "4.8"
  },
  "data": {
    "calibration_corpus": "WikiText-103",
    "n_calib": 32,
    "eval_corpus": "WikiText-103",
    "n_eval": 8,
    "max_calib_tokens": 384,
    "disjoint_eval": true
  },
  "calibration": {
    "normalize_keys": true,
    "key_space": "pre_rope",
    "d_eff_method": "participation_ratio",
    "d_eff_rounding": "ceil",
    "d_eff_min": 2,
    "d_eff_max": 126,
    "d_eff_stats": {
      "mean": 2.76,
      "min": 2,
      "max": 7
    }
  },
  "methods": {
    "turboquant": {},
    "spectralquant_v1": {},
    "spectralquant_v2": {}
  },
  "evidence_ids": [
    "RUN-THREEWAY-001"
  ]
}
```

## 15. Technical Report Structure

The final report should be a single coherent paper, not a pile of repeated v1 and v2 sections. The proposed structure is:

1. Abstract
2. Introduction: 3% is all you need, revisited
3. Background: KV cache, TurboQuant, QJL, and spectral structure
4. SpectralQuant v1: finding the semantic subspace
5. SpectralQuant v2: water-filling the semantic subspace
6. Experimental methodology
7. Results
8. Why water-filling helps
9. Failure modes and caveats
10. Reproducibility
11. Claims discipline
12. Conclusion

The report should include the following core sentence:

```text
SpectralQuant v1 showed that most key-cache signal lives in a tiny effective subspace; SpectralQuant v2 shows that even within that subspace, the spectrum matters.
```

## 16. Evidence Rules for the Report

Every empirical sentence in the final report must be backed by one of:

1. A JSON result file in `results/`.
2. A script in `experiments/` that generated the JSON.
3. A schema validation log.
4. A test result.
5. A figure generated from a JSON file.
6. A specific external paper citation.

The report should include evidence tags in comments or an appendix-style mapping:

```text
Claim: SQ v2 improves Mistral 3-bit attention cosine by +3.98 pp over TQ.
Evidence:
  results/three_way/mistral-7b-v0.3_bits3_seed42.json
  experiments/run_three_way.py
  docs/evidence_catalog.json
```

No manually typed result table should be allowed in the final paper unless it is generated from a JSON artifact.

## 17. Eight-Pass Reflection Protocol

Before any public-facing report is released, perform eight review passes:

### Pass 1: Claim audit

List every empirical claim. Verify that each claim has an evidence ID. Remove or weaken any claim without evidence.

### Pass 2: Math audit

Check every formula for dimensional consistency, notation consistency, and implementation correspondence.

### Pass 3: Compression-accounting audit

Recompute every compression ratio from stored bit counts. Resolve the 5.95x discrepancy before publication.

### Pass 4: Baseline audit

Verify that TurboQuant is described accurately and that the local implementation is not presented as the official Google implementation unless that is true.

### Pass 5: Reproducibility audit

Run every command in `docs/reproduction.md` from a clean environment. Confirm that expected files are generated.

### Pass 6: Statistical audit

Check seed counts, sample counts, confidence intervals, and whether min/max claims are robust or anecdotal.

### Pass 7: Reader audit

Rewrite the introduction and method explanation so a technically strong but non-specialist reader can understand why KV caches matter, why QJL matters, why spectral rotation matters, and why water-filling matters.

### Pass 8: Claims-discipline audit

Separate safe claims from aspirational claims. Ensure the report never says “superior in every way possible” unless every relevant dimension has been measured.

## 18. Claims Discipline

The intended strategic message is strong:

```text
Spectrum-aware KV-cache compression is more efficient than data-oblivious compression in the tested production-model settings.
```

The wording “superior to TurboQuant in every way possible” must be converted into measurable claims:

| Dimension | Can claim after current target runs? | Required evidence |
|---|---|---|
| Attention-output cosine on Mistral/Qwen tested settings | Yes, if reproduced | Three-way JSON |
| Compression ratio at tested operating points | Yes, after accounting reconciliation | Accounting audit |
| v2 over v1 quality | Yes, if reproduced | v1/v2 ablation |
| Catastrophic low-bit failure reduction | Yes, if per-head distributions reproduce | Min/head histogram |
| Perplexity | No | PPL runs |
| LongBench | No | LongBench n ≥ 50 |
| Real generation quality | No | Generation eval |
| End-to-end latency | No | Decode benchmark |
| Official TurboQuant superiority | No | Official implementation comparison |
| All architectures | No | Broader model suite |

The safe headline after the target reproduction is:

```text
On the tested Mistral-7B-v0.3 and Qwen2.5-7B attention-output benchmarks, SpectralQuant v2 Pareto-improves over the local TurboQuant baseline at aggressive KV-cache compression settings, with the largest gains at low bit budgets and heavier GQA.
```

## 19. Execution Plan

### Phase 0: Create private repo

1. Verify `niashwin/spectralquant-v2` does not exist.
2. Clone `Dynamis-Labs/spectralquant`.
3. Remove origin.
4. Create private GitHub repo.
5. Push untouched baseline.
6. Add this spec.

Exit criteria:

```text
Original repo untouched.
Private repo exists.
Initial baseline commit exists.
Spec commit exists.
```

### Phase 1: Evidence catalog

1. Catalog all v1 claims.
2. Catalog all v1 scripts.
3. Catalog all v1 JSONs.
4. Catalog all v1 figures.
5. Mark discrepancies.

Exit criteria:

```text
docs/evidence_catalog.md exists.
docs/evidence_catalog.json validates.
Every v1 headline claim is mapped or demoted.
```

### Phase 2: Unit-test scaffolding

1. Add test files.
2. Add schemas.
3. Add CI command.
4. Ensure baseline tests run.

Exit criteria:

```text
pytest runs.
Existing failures are documented.
New tests initially fail where v2 is not implemented.
```

### Phase 3: Water-filling

1. Implement `waterfill.py`.
2. Implement tests.
3. Validate allocation behavior.

Exit criteria:

```text
All waterfill tests pass.
```

### Phase 4: Per-dim codebooks

1. Add variable-bit semantic codebooks.
2. Preserve v1 codepath.
3. Add quantization tests.

Exit criteria:

```text
v1 unchanged.
v2 codebooks correct.
```

### Phase 5: Engine integration

1. Add `use_water_fill`.
2. Add allocation metadata.
3. Add accounting metadata.
4. Add attention scoring validation.

Exit criteria:

```text
Synthetic v1/v2 integration tests pass.
```

### Phase 6: Three-way benchmark

1. Implement `run_three_way.py`.
2. Run smoke test.
3. Run Mistral 3-bit.
4. Run Qwen 3-bit.
5. Run Mistral 2/5-bit.

Exit criteria:

```text
All target JSONs generated.
Schema validation passes.
```

### Phase 7: Report generation

1. Generate tables from JSON.
2. Generate plots from JSON.
3. Populate technical report.
4. Run eight-pass reflection.

Exit criteria:

```text
No manually typed empirical table remains.
Every claim has an evidence ID.
```

## 20. Definition of Done

The private v2 milestone is complete when:

1. `niashwin/spectralquant-v2` exists privately.
2. The original repo is untouched.
3. This spec is committed.
4. Water-filling unit tests pass.
5. Accounting tests pass.
6. TurboQuant local baseline is validated or clearly caveated.
7. Mistral 2/3/5-bit three-way runs complete.
8. Qwen 3-bit three-way run completes.
9. All result JSONs validate.
10. All report tables are generated from JSON.
11. Compression ratios are reconciled.
12. The eight-pass reflection protocol is complete.
13. The technical report can honestly say what has been measured and what has not.

## 21. Final Implementation North Star

SpectralQuant v2 should make one technical point with overwhelming clarity:

```text
The right unit of KV-cache compression is not the coordinate, and not even the semantic subspace as a uniform block. The right unit is the measured spectrum of each layer and KV head.
```

That is the scientific extension from v1 to v2. It is also the strongest way to make the report feel like a real continuation rather than a parameter tweak.
