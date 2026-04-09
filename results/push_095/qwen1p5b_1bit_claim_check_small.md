# Qwen 1.5B 1-bit Key Claim Check

This note compares the original favorable benchmark setup against the patched benchmark on `Qwen/Qwen2.5-1.5B-Instruct`.

Result files:
- `results/push_095/push095_Qwen-1.5B-proxy-small.json`
- `results/push_095/push095_Qwen-1.5B-realq-small.json`

## What Changed Versus The Initial Setup

The original favorable setup was approximated as:
- queries replaced by keys: `Q = K`
- non-causal self-attention proxy
- attention quality measured on the proxy setup

The patched setup used here changes that to:
- true model queries captured from the attention module
- causal attention instead of the symmetric proxy
- identical sampled coverage for both runs:
  - `n_calib=8`
  - `n_eval=4`
  - `seq_len=512`
  - `max_cached_vectors=256`
  - sampled layers `[7, 14, 21]`
  - sampled KV heads `[0, 1]`

Important representation note:
- "`1-bit keys`" here still means one bit per rotated key dimension in the calibrated basis, plus a stored norm.
- It is not raw sign-only keys in the original basis.

## Main Outcome

The key/value reconstruction metrics barely move between the two setups, because compression is unchanged.
What moves is the attention-output cosine once the benchmark uses real queries and causal attention.

For every tested 1-bit-key configuration, the patched benchmark is about `0.19` to `0.22` lower in attention cosine than the proxy benchmark.

## Side-By-Side Results

| Config | Bits | Compression | Proxy attn cos | Real-Q attn cos | Delta |
|---|---:|---:|---:|---:|---:|
| `1bit-K_3bit-V` | 544 | 7.53x | 0.6496 | 0.4364 | -0.2133 |
| `1bit-K_4bit-V` | 672 | 6.10x | 0.6718 | 0.4797 | -0.1921 |
| `1bit-K_5bit-V` | 800 | 5.12x | 0.7148 | 0.5087 | -0.2060 |
| `1bit-K_6bit-V` | 928 | 4.41x | 0.7451 | 0.5250 | -0.2201 |
| `uniform-3bit` | 800 | 5.12x | 0.6673 | 0.4580 | -0.2093 |

## Interpretation

- The proxy benchmark materially overstates how good the 1-bit-key setting looks.
- The drop is not caused by worse key reconstruction:
  - `key_cos_sim` is the same in proxy and real-Q runs for each config.
  - `val_cos_sim` is also the same.
- The difference comes from the benchmark itself, not the quantizer.

## Extra Ablation: Real Q + Non-Causal

An additional follow-up run was added for the missing `Real Q + non-causal` corner:
- `results/push_095/push095_Qwen-1.5B-realq-noncausal-small.json`

For the representative `1bit-K_6bit-V` setting:

| Setting | Attention cosine |
|---|---:|
| `Q=K + non-causal` | 0.7451 |
| `Real Q + non-causal` | 0.4124 |
| `Real Q + causal` | 0.5250 |

What this shows on the reduced Qwen 1.5B run:
- most of the optimistic gap comes from using `Q = K` instead of true captured queries
- the causal mask is not the source of the main drop on this specific run
- the remaining missing corner is `Q=K + causal` if a full `2 x 2` attribution is needed

## Narrower Claim Supported By This Run

A narrower claim that is supported by this reduced Qwen 1.5B run is:

- 1-bit rotated keys can still be competitive under a more realistic benchmark.
- But the original proxy setup overstated absolute attention quality by roughly `0.2` cosine points.
- On this run, `1bit-K_5bit-V` still beats `uniform-3bit` at the same `800`-bit budget:
  - `0.5087` vs `0.4580`
- So the claim is not "1-bit keys are universally great as measured originally".
- The safer claim is "1-bit rotated keys remain viable, but the original `Q=K` proxy made them look substantially better than they do under true model queries."
