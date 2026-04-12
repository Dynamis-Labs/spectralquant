# WHT vs PCA vs PolarQuant

This note summarizes the final cleaned-up version of Experiment 4 after fixing the calibration inconsistencies in the earlier drafts.

The final comparison is a three-way benchmark:

- `PCA`
- `WHT-only`
- `WHT PolarQuant-style`

The important cleanup is that `WHT-only` is now calibrated from **normalized** vectors, which makes it comparable to the PolarQuant branch.

## Goal

The point of this experiment is to separate three different questions that were previously mixed together:

1. What happens if we replace the learned PCA basis with a fixed WHT basis?
2. What happens if we keep WHT but use per-coordinate empirical calibration?
3. What happens if we keep WHT and instead use PolarQuant-style shared theoretical codebooks?

This gives a cleaner decomposition than the earlier two-way experiments.

## Variants

### PCA

- learn per-head K/V covariance from calibration data
- eigendecompose covariance
- use the learned eigenbasis
- build per-coordinate codebooks from PCA eigenvalues
- quantize normalized vectors and store norms separately

This is the SpectralQuant-style path.

### WHT-only

- use a fixed randomized Hadamard basis `D1 @ H @ D2`
- normalize each vector before rotation
- estimate empirical variance for each rotated coordinate from normalized calibration data
- build a separate codebook for each rotated coordinate using those empirical `sigma_i`
- quantize normalized vectors and store norms separately

This is the clean “WHT plus calibrated per-coordinate scalar quantization” branch.

### WHT PolarQuant-style

- use the same fixed randomized Hadamard basis
- normalize each vector before rotation
- use shared codebooks derived from the PolarQuant theory:
  - `1-bit`: `±sqrt(2 / (pi d))`
  - `2-bit`: fixed closed-form levels scaled by `1 / sqrt(d)`
  - `3+ bit`: Lloyd-Max on `N(0, 1/d)`
- apply rotated-domain norm correction before inverse rotation
- store norms separately

This is the practical TurboQuant-style branch.

## Why The Earlier Runs Were Misleading

Earlier versions mixed together several inconsistencies:

- some runs compared proxy `Q=K` against real-`Q` causal
- some compared different models
- older “WHT” runs did not clearly separate `WHT-only` from `WHT PolarQuant-style`
- `WHT-only` originally used an incorrect flat `sigma = 1`
- `WHT-only` was later calibrated from raw vectors instead of normalized vectors

Those issues made the intermediate conclusions unstable.

The results below are from the final corrected version, where:

- `WHT-only` calibration is done on normalized data
- `WHT PolarQuant-style` uses the theoretical `1/d` scaling
- all three variants are produced by the same script in one run

## Commands

Proxy run:

```bash
.venv/bin/python experiments/codex_wht_vs_pca.py \
  --device mps \
  --mode main \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --short-name Qwen-1.5B
```

Real-`Q` causal run:

```bash
.venv/bin/python experiments/codex_wht_vs_pca.py \
  --device mps \
  --mode main \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --short-name Qwen-1.5B \
  --evaluation-mode real_q_causal
```

## Raw Results

Proxy:

- `results/codex_wht_vs_pca/codex_wht_vs_pca_proxy_threeway_Qwen-1.5B.json`
- `results/codex_wht_vs_pca/codex_wht_vs_pca_proxy_threeway_Qwen-1.5B.md`

Real-`Q` causal:

- `results/codex_wht_vs_pca/codex_wht_vs_pca_realq_causal_threeway_Qwen-1.5B.json`
- `results/codex_wht_vs_pca/codex_wht_vs_pca_realq_causal_threeway_Qwen-1.5B.md`

## Final Results

### Proxy setup

Setup:

- `Q = K`
- non-causal attention
- model: `Qwen2.5-1.5B`

| Variant / Config | Attn Cos | KL | Key Cos | Value Cos |
|---|---:|---:|---:|---:|
| `PCA 1bit-K_6bit-V` | 0.7569 | 5.5865 | 0.9022 | 0.9695 |
| `WHT-only 1bit-K_6bit-V` | 0.8141 | 1.1236 | 0.8634 | 0.9996 |
| `WHT PolarQuant-style 1bit-K_6bit-V` | 0.9251 | 0.4098 | 0.8084 | 0.9996 |
| `PCA 1bit-K_4bit-V` | 0.6793 | 5.5865 | 0.9022 | 0.8709 |
| `WHT-only 1bit-K_4bit-V` | 0.8126 | 1.1236 | 0.8634 | 0.9956 |
| `WHT PolarQuant-style 1bit-K_4bit-V` | 0.9225 | 0.4098 | 0.8084 | 0.9955 |
| `PCA 2bit-K_6bit-V` | 0.7622 | 4.7245 | 0.9022 | 0.9695 |
| `WHT-only 2bit-K_6bit-V` | 0.9004 | 1.2201 | 0.9455 | 0.9996 |
| `WHT PolarQuant-style 2bit-K_6bit-V` | 0.9745 | 0.1114 | 0.9439 | 0.9996 |

### Real-`Q` causal setup

Setup:

- real captured `Q`
- causal attention
- model: `Qwen2.5-1.5B`

| Variant / Config | Attn Cos | KL | Key Cos | Value Cos |
|---|---:|---:|---:|---:|
| `PCA 1bit-K_6bit-V` | 0.5148 | 6.7447 | 0.9022 | 0.9695 |
| `WHT-only 1bit-K_6bit-V` | 0.7727 | 0.4972 | 0.8634 | 0.9996 |
| `WHT PolarQuant-style 1bit-K_6bit-V` | 0.6769 | 0.8792 | 0.8084 | 0.9996 |
| `PCA 1bit-K_4bit-V` | 0.4884 | 6.7447 | 0.9022 | 0.8709 |
| `WHT-only 1bit-K_4bit-V` | 0.7725 | 0.4972 | 0.8634 | 0.9956 |
| `WHT PolarQuant-style 1bit-K_4bit-V` | 0.6753 | 0.8792 | 0.8084 | 0.9955 |
| `PCA 2bit-K_6bit-V` | 0.5290 | 5.9388 | 0.9022 | 0.9695 |
| `WHT-only 2bit-K_6bit-V` | 0.8993 | 0.3452 | 0.9455 | 0.9996 |
| `WHT PolarQuant-style 2bit-K_6bit-V` | 0.8964 | 0.2739 | 0.9439 | 0.9996 |

## Interpretation

### 1. Is the learned PCA basis necessary?

No.

In both proxy and real-`Q` causal setups, the fixed WHT branches are better than PCA once the calibration is made internally consistent.

So the favorable `1-bit key` effect is not a PCA-only artifact.

### 2. Does WHT-only converge toward PolarQuant-style behavior?

Yes, much more than in the earlier buggy runs.

Once `WHT-only` estimates `sigma_i` from normalized rotated vectors, it moves into the same regime as the PolarQuant branch. This matches the theoretical expectation that the rotated normalized coordinates should have variance close to `1/d`.

There is still a difference:

- `WHT-only` uses empirical per-coordinate `sigma_i`
- `WHT PolarQuant-style` uses shared theoretical `1 / sqrt(d)`

But they are now clearly much closer than before.

### 3. Which branch is best?

Proxy:

- `WHT PolarQuant-style` > `WHT-only` > `PCA`

Real-`Q` causal:

- `WHT-only` > `WHT PolarQuant-style` > `PCA` on the 1-bit points
- `WHT-only` and `WHT PolarQuant-style` are essentially tied at `2bit-K_6bit-V`

So the final story is not “PolarQuant always dominates WHT-only.”

It is:

- both WHT branches are strong once calibrated consistently
- the basis change away from PCA matters
- the choice between empirical per-coordinate `sigma_i` and shared theoretical `1/sqrt(d)` is a second-order effect compared with the earlier calibration bugs

### 4. What do these results suggest?

They suggest that the strong performance is primarily coming from:

- normalizing the vectors
- using a randomized orthogonal WHT basis
- quantizing in that rotated normalized space

Once that is done correctly, both:

- empirical WHT calibration
- theoretical PolarQuant codebooks

work well.

## Bottom Line

The final cleaned-up experiment supports three conclusions:

1. The strong `1-bit key` result does not depend on the learned PCA basis.
2. The earlier contradictory WHT results were mostly caused by calibration inconsistencies.
3. After fixing those inconsistencies, WHT-based quantization is clearly competitive with or better than PCA, and the empirical-calibrated WHT and PolarQuant-style WHT branches land in the same general regime.
