<p align="center">
  <h1 align="center">SpectralQuant</h1>
  <p align="center">
    <strong>3% Is All You Need: Breaking TurboQuant's Compression Limit via Spectral Structure</strong>
  </p>
  <p align="center">
    <a href="paper_output/spectralquant.pdf"><img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper PDF"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/PyTorch-2.2%2B-orange" alt="PyTorch 2.2+">
  </p>
</p>

> **Paper submitted to arXiv.** The arXiv link will be updated here once available.
> In the meantime, the full paper is included in this repository: [`paper_output/spectralquant.pdf`](paper_output/spectralquant.pdf)

---

## Overview

SpectralQuant is a KV cache compression method for large language model inference. It improves on TurboQuant (Zandieh et al., ICLR 2026) by exploiting a universal structural property: across **six models in four architecture families**, KV cache key vectors concentrate signal in only **3–4% of the head dimension**.

By identifying these dimensions through a one-time **15-second calibration** and removing error correction on the remaining 96–97% noise dimensions, SpectralQuant achieves **better quality and better compression** simultaneously.

### Headline Results

| | SpectralQuant | TurboQuant | Improvement |
|---|---|---|---|
| **Cosine similarity** (Qwen 2.5-14B) | 0.9485 | 0.9226 | +2.59 pp |
| **Compression ratio** | 5.95× | 5.02× | +18.6% |
| **Latency** (512 tokens) | 0.257 ms/step | 0.566 ms/step | 2.2× faster |
| **Perplexity** (Qwen 7B, 1024 tok) | 7.51 | 7.51 | Compression-neutral |

### Key Findings

1. **Universal low-rank structure.** d_eff/head_dim ≈ 3–4% across Qwen (1.5B, 7B, 14B), Llama 3.1-8B, Mistral 7B, and Gemma 2-9B — the ratio is constant across head dimensions, model sizes, and architecture families.

2. **Statistically significant.** 10-seed CI on Qwen 2.5-1.5B: SQ mean=0.8635 ± 0.0024 vs TQ mean=0.8409 ± 0.0046, Wilcoxon p=0.031.

3. **Faster at all sequence lengths.** SQ is faster than TQ at 512, 1024, and 2048 tokens. No latency penalty for calibration-aware compression.

4. **KV spectral asymmetry.** Keys: d_eff ≈ 4. Values: d_eff ≈ 40–55 (10–15× larger). This explains why low-rank compression fails for values while SQ succeeds.

### Representation Note

Throughout these experiments, "`1-bit keys`" should be read narrowly: one bit per rotated key dimension in the calibrated spectral basis, plus a separately stored vector norm. It is not a raw sign-only key in the original basis.

### Small-Model Sanity Checks

To check how much of the reported `1-bit keys` quality depends on the benchmark itself, I ran a reduced local MPS comparison on three small open models using the same compression configs but two different evaluation modes.

The two modes were:

1. `Proxy Q=K`
   - queries are replaced with keys (`Q = K`)
   - attention is evaluated with a non-causal self-attention proxy
   - this is the more favorable setup

2. `Real Q + causal`
   - queries are captured from the actual attention module during a forward pass
   - the same compressed keys/values are evaluated with causal attention
   - this is closer to the intended decoding setting

Important scope note:
- this is not a pure `Q=K` ablation, because the comparison changes two variables at once:
  - query source: proxy `Q=K` vs true captured `Q`
  - attention pattern: non-causal vs causal
- the result is still useful because it measures the gap between an optimistic proxy and a more realistic decoding-style evaluation
- but it does not by itself identify how much of the gap comes from `Q=K` specifically versus the causal mask

What stayed fixed between the two runs:
- same model
- same reduced coverage
- same sampled layers and heads
- same compression config
- same key/value reconstruction path

Coverage used for all rows below:
- `n_calib=8`
- `n_eval=4`
- `seq_len=512`
- `max_cached_vectors=256`
- sampled layers / sampled KV heads

Interpretation:
- `Proxy Q=K` and `Real Q + causal` have the same key/value reconstruction numbers for a given config.
- The gap is therefore caused by the evaluation setup, not by a different quantizer.
- A negative delta means the favorable proxy made the config look better than it does under real queries.

For the representative `1bit-K_6bit-V` setting, replacing the proxy with true captured queries and causal attention lowers attention cosine substantially:

| Model | Config | Proxy `Q=K` | Real Q + causal | Delta |
|---|---|---:|---:|---:|
| Qwen 2.5 1.5B | `1bit-K_6bit-V` | 0.7451 | 0.5250 | -0.2201 |
| TinyLlama 1.1B | `1bit-K_6bit-V` | 0.8644 | 0.4954 | -0.3690 |
| Qwen 2.5 0.5B | `1bit-K_6bit-V` | 0.8676 | 0.6070 | -0.2606 |

Files:
- `results/push_095/push095_Qwen-1.5B-proxy-small.json`
- `results/push_095/push095_Qwen-1.5B-realq-small.json`
- `results/push_095/push095_TinyLlama-1.1B-proxy-small.json`
- `results/push_095/push095_TinyLlama-1.1B-realq-small.json`
- `results/push_095/push095_Qwen-0.5B-proxy-small.json`
- `results/push_095/push095_Qwen-0.5B-realq-small.json`
- `results/push_095/qwen1p5b_1bit_claim_check_small.md`

Recommended follow-up for a cleaner attribution:
- run the full `2 x 2` evaluation matrix:
  - `Q=K + non-causal`
  - `Q=K + causal`
  - `Real Q + non-causal`
  - `Real Q + causal`
- that would isolate whether the optimism is driven more by proxy queries, by removing the causal mask, or by both
- the `Real Q + non-causal` setting has now been run for `Qwen 2.5 1.5B`
  - file: `results/push_095/push095_Qwen-1.5B-realq-noncausal-small.json`
  - for `1bit-K_6bit-V`, the three observed values are:
    - `Q=K + non-causal`: `0.7451`
    - `Real Q + non-causal`: `0.4124`
    - `Real Q + causal`: `0.5250`
  - on this reduced run, most of the optimism comes from the `Q=K` proxy rather than from removing the causal mask
  - this means the current evidence supports the narrower statement "proxy queries are the main confound here"; the `Q=K + causal` corner is still missing if a full `2 x 2` attribution is needed

---

## Quick Start

```bash
git clone https://github.com/dynamis-labs/spectralquant.git
cd spectralquant
pip install -e ".[dev]"

# Clone TurboQuant baseline
mkdir -p baseline
git clone https://github.com/DevTechJr/turboquant_cutile.git baseline/turboquant_cutile

# Run main experiment (quick mode)
PYTHONPATH=src python experiments/run_memory_efficiency.py --quick
```

### Full Reproduction

```bash
# Core experiments
PYTHONPATH=src python experiments/neurips_models_asymmetry.py  # Mistral + Gemma + KV asymmetry
PYTHONPATH=src python experiments/neurips_seeds_latency.py     # 10-seed CI + latency crossover
PYTHONPATH=src python experiments/neurips_llama_full.py        # LongBench on Llama (requires HF_TOKEN)
PYTHONPATH=src python experiments/lowrank_cossim_sweep.py      # Low-rank sweep
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.2.0
- CUDA GPU (experiments ran on NVIDIA B200)

### Random Seeds

All experiments use seed 42 as default. The 10-seed CI test uses seeds: 42, 123, 7, 2024, 31415, 99, 1337, 8675309, 271828, 314159.

---

## Paper Claims → Code → Data

Every number in the paper traces to a script and a result file in this repository.

| Paper Section | Claim | Script | Result File |
|---|---|---|---|
| Abstract | SQ 0.9485 vs TQ 0.9226 on 14B (+2.59 pp) | `run_memory_efficiency.py` | `results/memory_efficiency/all_models.json` |
| Abstract | 5.95× vs 5.02× compression | Analytical (bit accounting) | Same |
| Abstract | PPL=9.51 (Qwen 1.5B) | `run_v3_ppl_niah_v2.py` | `results/v3/v3_perplexity_v2.json` |
| Abstract | PPL=7.51 (Qwen 7B) | `neurips_seeds_latency.py` | `results/neurips/neurips_qwen7b_ppl.json` |
| Abstract | NIAH 10/10 (Llama) | `run_v3_ppl_niah_v2.py` | `results/v3/v3_niah_llama_v2.json` |
| Table 1 | d_eff/head_dim ≈ 3–4% (6 models) | `neurips_models_asymmetry.py` | `results/neurips/neurips_*.json` |
| Table 3 | Main results (4 models) | `run_memory_efficiency.py` | `results/memory_efficiency/all_models.json` |
| §Stats | Wilcoxon p=0.031, 10-seed CI | `neurips_seeds_latency.py` | `results/neurips/neurips_10seed.json` |
| §Cross-arch | Llama +1.74 pp, Mistral +1.21 pp, Gemma +0.72 pp | `neurips_models_asymmetry.py` | `results/neurips/neurips_*.json` + `results/v3/v3_crossarch.json` |
| §Dist shift | +2.1 to +3.6 pp across domains | `run_v3_deff_distshift_latency.py` | `results/v3/v3_distribution_shift.json` |
| §Latency | SQ faster at all seq lengths | `neurips_seeds_latency.py` | `results/neurips/neurips_latency_crossover.json` |
| §KV asymmetry | d_eff_keys≈4, d_eff_vals≈40–55 | `neurips_models_asymmetry.py` | `results/neurips/neurips_kv_asymmetry.json` |
| §Low-rank | Values fail at r=4 (CosSim=0.15) | `lowrank_cossim_sweep.py` | `results/lowrank/lowrank_cossim_sweep.json` |
| §Calibration | CV=3.9% | `run_calibration_stability.py` | `results/calibration_stability/stability.json` |
| Ablation | Config G = 0.8741 | `run_final_experiments.py` | `results/final/final_experiments.json` |
| §LongBench | Preliminary n=5 | `neurips_llama_full.py` | `results/v3/v3_longbench.json` |

---

## Repository Structure

```
spectralquant/
├── src/spectralquant/           Core library (9 modules)
│   ├── calibration.py           Eigenspectral calibration (PCA, d_eff, κ)
│   ├── spectral_rotation.py     Spectral rotation vs random rotation baseline
│   ├── nonuniform_quantization.py  Lloyd-Max with per-regime codebooks
│   ├── selective_qjl.py         QJL correction on signal dims only
│   ├── engine.py                SpectralQuantEngine (subclasses TurboQuantEngine)
│   ├── spectralquant.py         Full standalone pipeline
│   ├── metrics.py               Cosine similarity, MSE, compression ratio
│   └── utils.py                 Seeds, model config, data loading
│
├── experiments/                 21 experiment scripts (see table above)
│
├── results/                     Raw experimental data (44 JSON files)
│   ├── memory_efficiency/       Main results: 4 models × TQ vs SQ
│   ├── neurips/                 10-seed CI, Gemma, Mistral, KV asymmetry, latency
│   ├── v3/                      Cross-arch, perplexity, NIAH, LongBench, d_eff
│   ├── final/                   Ablation table (Config F)
│   ├── calibration_stability/   Calibration stability (CV=3.9%)
│   ├── lowrank/                 Low-rank projection sweep (r=2..64)
│   ├── eigenspectral/           Phase 1 calibration (d_eff per layer, summary stats)
│   ├── baseline_reproduction/   Phase 0 baseline reproduction targets
│   ├── comparison/              Head-to-head TQ vs SQ with per-head statistics
│   ├── comprehensive/           Multi-model sweep across d_eff methods
│   ├── aggressive/              Aggressive compression variant metrics
│   ├── deff_sweep/              d_eff method comparison (participation ratio vs cumvar)
│   ├── kernel/                  Kernel benchmark timing
│   ├── seqlen_sweep/            Sequence length sweep (128–2048 tokens)
│   └── unnormalized/            Normalized vs unnormalized quantization
│
├── paper_output/                Paper source and figures
│   ├── spectralquant.tex        LaTeX source
│   ├── spectralquant_refs.bib   Bibliography
│   ├── spectralquant.pdf        Compiled PDF
│   ├── generate_figures.py      Figure generation script
│   └── figures/                 Publication figures (PDF + PNG)
│
├── tests/                       Test suite (5 files)
├── configs/                     Experiment configs (default + quick)
├── scripts/                     Setup and runner scripts
├── pyproject.toml               Package metadata
├── Makefile                     Build targets
└── LICENSE                      MIT
```

---

## Experiment Scripts

| Script | Description | Output |
|---|---|---|
| `neurips_models_asymmetry.py` | Mistral 7B + Gemma 2-9B + KV asymmetry (5 models) | `results/neurips/neurips_mistral.json`, `neurips_gemma.json`, `neurips_kv_asymmetry.json` |
| `neurips_seeds_latency.py` | 10-seed CI + latency crossover + Qwen 7B PPL | `results/neurips/neurips_10seed.json`, `neurips_latency_crossover.json`, `neurips_qwen7b_ppl.json` |
| `neurips_llama_full.py` | LongBench (n=5, 6 subtasks) + NIAH on Llama 3.1-8B | `results/v3/v3_longbench.json`, `v3_niah_llama_v2.json` |
| `lowrank_cossim_sweep.py` | Low-rank SVD projection sweep (r=2..64) | `results/lowrank/lowrank_cossim_sweep.json` |
| `run_memory_efficiency.py` | Main results: 4 models × 9 configs | `results/memory_efficiency/all_models.json` |
| `run_v3_perplexity_crossarch.py` | Cross-architecture + 5-seed CI | `results/v3/v3_crossarch.json` |
| `run_v3_ppl_niah_v2.py` | Perplexity + NIAH (Llama) | `results/v3/v3_perplexity_v2.json`, `v3_niah_llama_v2.json` |
| `run_v3_deff_distshift_latency.py` | d_eff sweep + distribution shift + latency | `results/v3/v3_distribution_shift.json`, `v3_deff_sweep.json` |
| `run_final_experiments.py` | Config F ablation | `results/final/final_experiments.json` |
| `run_calibration_stability.py` | Calibration stability (CV=3.9%) | `results/calibration_stability/stability.json` |

---

## Attribution

**TurboQuant** — Zandieh, Daliri, Hadian, and Mirrokni (Google Research / Google DeepMind / NYU).
Paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026.
We use the community implementation by Anirudh Bharadwaj Vangara: [DevTechJr/turboquant_cutile](https://github.com/DevTechJr/turboquant_cutile).

**The Price of Meaning** — Barman, Starenky, Bodnar, Narasimhan, and Gopinath (Sentra).
Paper: [arXiv:2603.27116](https://arxiv.org/abs/2603.27116).
The eigenspectral analysis in SpectralQuant builds on the observation from this work that semantic memory systems exhibit universal low-rank structure in their representations.

---

## Citation

```bibtex
@article{gopinath2026spectralquant,
  title={3\% Is All You Need: Breaking {TurboQuant}'s Compression Limit
         via Spectral Structure},
  author={Gopinath, Ashwin},
  year={2026},
  note={Sentra; MIT Department of Mechanical Engineering}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
