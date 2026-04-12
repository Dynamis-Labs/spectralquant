# Three-Way Proxy Result

Source:

- `codex_wht_vs_pca_proxy_threeway_Qwen-1.5B.json`

Setup:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- `Q = K`
- non-causal attention
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
| `PCA 1bit-K_6bit-V` | 0.7569 | 5.5865 | 0.9022 | 0.9695 |
| `WHT-only 1bit-K_6bit-V` | 0.8141 | 1.1236 | 0.8634 | 0.9996 |
| `WHT PolarQuant-style 1bit-K_6bit-V` | 0.9251 | 0.4098 | 0.8084 | 0.9996 |
| `PCA 1bit-K_4bit-V` | 0.6793 | 5.5865 | 0.9022 | 0.8709 |
| `WHT-only 1bit-K_4bit-V` | 0.8126 | 1.1236 | 0.8634 | 0.9956 |
| `WHT PolarQuant-style 1bit-K_4bit-V` | 0.9225 | 0.4098 | 0.8084 | 0.9955 |
| `PCA 2bit-K_6bit-V` | 0.7622 | 4.7245 | 0.9022 | 0.9695 |
| `WHT-only 2bit-K_6bit-V` | 0.9004 | 1.2201 | 0.9455 | 0.9996 |
| `WHT PolarQuant-style 2bit-K_6bit-V` | 0.9745 | 0.1114 | 0.9439 | 0.9996 |

Runtime:

- `609.50` seconds on MPS

## Reading

After fixing `WHT-only` so it estimates per-coordinate WHT-space variances from normalized data, the proxy decomposition is much cleaner:

- `PCA -> WHT-only` already improves the result substantially.
- `WHT-only -> WHT PolarQuant-style` still improves it further, but by less than before.

So on this corrected run:

- changing the basis plus calibrated WHT-space quantization helps a lot
- the PolarQuant-style shared `1/d` quantizer still helps on top of that

The ranking is:

- `WHT PolarQuant-style` > `WHT-only` > `PCA`
