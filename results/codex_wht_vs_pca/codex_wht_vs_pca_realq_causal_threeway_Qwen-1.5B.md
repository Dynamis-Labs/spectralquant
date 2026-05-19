# Three-Way Real-Q Causal Result

Source:

- `codex_wht_vs_pca_realq_causal_threeway_Qwen-1.5B.json`

Setup:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- real captured `Q`
- causal attention
- sampled layers: `7, 14, 21`
- sampled heads: `0, 1`

Variants:

- `PCA`
  Learned PCA basis plus per-dimension calibrated codebooks.
- `WHT-only`
  Fixed WHT basis plus the same PCA-like per-dimension codebook structure.
- `WHT PolarQuant-style`
  Fixed WHT basis plus shared `1/d`-based PolarQuant centroids and rotated-domain norm correction.

## Results

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

Runtime:

- `506.96` seconds on MPS

## Reading

After fixing `WHT-only` so it estimates per-coordinate WHT-space variances from normalized data, the realistic decomposition changed substantially:

- `WHT-only` now beats `PCA` by a large margin.
- `WHT-only` is also slightly better than `WHT PolarQuant-style` on the 1-bit settings.
- at `2bit-K_6bit-V`, `WHT-only` and `WHT PolarQuant-style` are essentially tied, with PolarQuant slightly better on KL and `WHT-only` slightly better on attention cosine.

So on this final realistic run, the ranking is:

- `WHT-only` > `WHT PolarQuant-style` > `PCA` for the 1-bit points
- `WHT-only ~= WHT PolarQuant-style` > `PCA` for `2bit-K_6bit-V`
