# Supporting note: perplexity-gap mechanism (Sentra technical note, April 2026)

## Provenance and status

- **Original title (preserved verbatim):** *Why SpectralQuant v2 beats TurboQuant on perplexity — the mechanism*
- **Authors:** Anirudh B. Vangara, Ashwin Gopinath
- **Origin:** Sentra Technical Note, April 2026 (student / lab-internal write-up by A. B. Vangara, with co-author A. Gopinath; reflects expanded-evidence-layer development discussions)
- **Captured into this repository:** 2026-05-01, as a supporting artifact accompanying the consolidated SpectralQuant manuscript at `paper_output_consolidated/spectralquant_unrestricted_paper.tex`.
- **Public-method framing:** in this repository the program is *one method, SpectralQuant, two evidence layers* (initial / uniform-allocation special case, and expanded / water-filled allocation). The note's "v1 / v2" terminology is preserved in the original-text section below as written, but every consolidated-paper citation of this note uses the single-method framing (`uniform-allocation special case` and `water-filled allocation`).
- **Evidence status:** **supporting-note only — not paper-valid.**
  The Mistral-7B-v0.3 WikiText-103 perplexity numbers in the table below are the note's own observations as recorded by the authors during expanded-layer development. They are **not** present as schema-registered JSON under `results/v3/modal/perplexity__Mistral-7B*.json` at the time of capture; a `find . -name 'perplexity__Mistral-7B*.json'` against this commit returns no matches, and `grep` for the specific perplexity values (12.22, 12.36, 12.33, 12.29, 14.88, 82.13 in the perplexity context, 13.14, 13.24, 25.13, 26.71) against `results/` returns no perplexity-context hits. The note is therefore registered as an *interpretive / mechanistic* supporting document for the consolidated manuscript, and its tabulated Mistral perplexity numbers must not be reproduced in any paper-valid table without a corresponding schema-validated artifact. Where the consolidated paper draws on the note, it cites the qualitative mechanism (calibrated rotation, two-regime allocation, selective QJL, water-filling) — not the unsupported numeric perplexity values.

## How the consolidated paper uses this note

The note's four-mechanism decomposition is incorporated into the consolidated technical report's Background and Interpretation sections (see in particular the new §"Mechanism behind the perplexity gap" subsection of *Interpretation*). Numeric values cited in the consolidated paper are taken only from schema-valid artifacts under `results/v3/modal/`; mechanistic claims that are math (rate–distortion / Lloyd–Max / Shannon–Berger) or already-cited literature are kept as such; per-layer compounding is presented as a plausible explanatory account of observed metrics, not as a directly measured phenomenon (no per-layer-residual-stream attribution artifact exists in this repository).

## Original text (verbatim, as captured 2026-05-01)

> # Why SpectralQuant v2 beats TurboQuant on perplexity — the mechanism
>
> A breakdown of the four algorithmic reasons behind the measured perplexity gap, and why those gains compound across transformer layers.
>
> Anirudh B. Vangara, Ashwin Gopinath — Sentra Technical Note, April 2026
>
> ## What we measured
>
> Mistral-7B-v0.3 on WikiText-103 perplexity (lower = better):
>
> | avg_bits | Compression | FP16 | TurboQuant | SpectralQuant v1 | SpectralQuant v2 |
> |---|---|---|---|---|---|
> | 5 | 3.07× | 12.22 | 12.36 (+1.14%) | 12.33 (+0.88%) | 12.29 (+0.55%) |
> | 3 | 5.95× | 12.22 | 14.88 (+21.78%) | 13.24 (+8.36%) | 13.14 (+7.48%) |
> | 2 | 9.5× | 12.22 | 82.13 (+571.9%) | 25.13 (+105.6%) | 26.71 (+118.5%) |
>
> At 5.95× compression (the production sweet spot), TurboQuant adds 22% perplexity overhead vs FP16 — SpectralQuant v2 adds only 7.5%. Why?
>
> ## The fundamental insight everything else flows from
>
> KV cache key vectors are not random — they concentrate their information in a few dimensions.
>
> If you take all the K vectors a Mistral 7B head produces during inference and look at where the variance lives, you see something like this:
>
> Eigenvalue spectrum of K covariance (typical Mistral head):
>   Dim 0:    λ₀ = 0.50  ← almost half the variance
>   Dim 1:    λ₁ = 0.20
>   Dim 2:    λ₂ = 0.08
>   Dim 3:    λ₃ = 0.04
>   Dims 4–127: ~0.001 each (basically noise)
>
> About 80–90% of the useful information lives in 3–5 dimensions out of 128. This is the empirical observation that defines d_eff (the effective dimensionality). It's an accident of how transformer attention learns to work, but it's a robust accident — true across every model we've measured (Mistral, Qwen, ESM-2, ViT).
>
> TurboQuant is data-oblivious and doesn't know this. SpectralQuant exploits it directly.
>
> ## Mechanism 1 — Calibrated rotation puts the information in known places
>
> TurboQuant rotates K with a random orthogonal matrix. After the rotation, the variance is roughly spread evenly across all 128 dimensions (the random rotation smooths the spectrum). It then quantizes every dimension with the same Lloyd-Max codebook calibrated to assume uniform variance.
>
> This is mathematically optimal if you don't know anything about your data. But we do know — and after random rotation, the actual per-dim variances still differ; the random rotation just moves the eigenvalues around to unpredictable coordinates.
>
> SpectralQuant rotates K with the eigenvectors of the K covariance (computed once during a 15-second calibration). After this rotation, dimension 0 has variance exactly λ₀ (largest), dimension 1 has λ₁, etc. The variance is maximally concentrated in the top coordinates and provably zero-correlated across coordinates (this is what eigendecomposition does).
>
> Concrete impact: for the same total bit budget, TQ wastes bits on 124 nearly-identical low-variance coordinates while underspending on the few high-variance ones. SQ knows which dims to spend on.
>
> ## Mechanism 2 — Two-regime quantization spends bits where it matters
>
> Because SQ knows which dims have variance, it uses a two-regime split:
> - Semantic regime (top d_eff ≈ 4 dims): quantize with b_high bits per dim (more precision)
> - Tail regime (remaining ~124 dims): quantize with b_low bits per dim (less precision)
>
> TurboQuant has no concept of regimes — every dim gets the same bits.
>
> At avg_bits=3 with d_eff=4 and head_dim=128:
> - TQ: 3 bits × 128 dims = 384 bits per K vector, evenly spent
> - SQ: ~4 bits × 4 semantic dims + ~3 bits × 124 tail dims ≈ 388 bits, but the 4 most-important dims get 33% more precision than uniform allocation would give them
>
> For attention quality this is huge — Q · K is dominated by contributions from the high-variance dims, so getting them right matters a lot more than getting tail dims right.
>
> ## Mechanism 3 — Selective QJL is a double win (bits AND noise)
>
> The QJL correction adds an unbiased estimate of the residual error to attention scores. It's needed because pure scalar quantization (Lloyd-Max) is biased for inner products.
>
> TurboQuant applies QJL on all 128 dims. Costs 128 extra sign bits per K vector. The correction term has variance proportional to head_dim (sqrt(pi/2) / d factor in the score formula).
>
> SpectralQuant applies QJL on only the d_eff ≈ 4 semantic dims. Costs 4 extra sign bits. The correction's variance is proportional to d_eff, not d — so the correction itself is 30x less noisy for Mistral's head_dim=128.
>
> Why is this OK to skip on the tail? Because tail variance is essentially zero — the residual K - K_mse_reconstructed on tail dims is also essentially zero. There's nothing to correct.
>
> So SQ saves bits AND adds less noise to attention scores. Both improve the final answer.
>
> ## Mechanism 4 — Water-filling refines this within the semantic regime
>
> Even within the top 4 semantic dims, the eigenvalues vary: λ0=0.50, λ1=0.20, λ2=0.08, λ3=0.04. The uniform-allocation special case gives every semantic dim the same b_high bits — so dim 3 gets the same precision as dim 0, which is wasteful.
>
> The water-filled allocation uses greedy water-filling (a 1948 result from Shannon-Berger rate-distortion theory): allocate the next bit to whichever dim has the largest marginal distortion reduction λ_i / 4^b_i. Result: maybe [5, 3, 2, 2] instead of [3, 3, 3, 3]. Same total bits, redistributed optimally per the math.
>
> ## Why this compounds dramatically through 32 layers
>
> A single layer of Mistral 7B does:
> hidden_next = hidden_prev + attention(hidden_prev) + MLP(...)
>
> If attention's output is even 0.5 pp closer to FP16, that's a slightly cleaner contribution to hidden_next. The next layer reads from hidden_next, computes its own attention, and adds its own (slightly cleaner) output. Across 32 layers, errors compound multiplicatively through the residual stream.
>
> This is why a 0.5 pp per-layer attention cosine improvement (the 5-bit case) can produce a 52% reduction in perplexity overhead — the per-layer effect amplifies through depth.
>
> It's also why TurboQuant collapses at 2 bits: when per-layer attention noise crosses some threshold, the residual stream accumulates so much noise that hidden states become essentially random. The model's logits become near-uniform → perplexity blows up to 82 (basically random next-token prediction).
>
> SQ doesn't cross that threshold at 2 bits because:
> - Calibrated rotation puts most variance in 4 dims that are well-quantized
> - Selective QJL keeps attention score noise low
> - Water-fill gets the dominant dim into 4–5-bit precision even when the budget is "2 bits avg"
>
> So SQ degrades gracefully instead of catastrophically.
>
> ## The 4 mechanisms in one table
>
> | Mechanism | TurboQuant | SpectralQuant | Why it helps perplexity |
> |---|---|---|---|
> | Rotation basis | Random | Calibrated eigenvectors | Information is concentrated where you can find it |
> | Bit allocation by regime | Uniform across all 128 dims | More on top 4, less on tail 124 | Bits go to high-information coords |
> | QJL correction breadth | All 128 dims (128 sign bits) | Only d_eff ≈ 4 dims (4 sign bits) | Saves bits AND reduces score noise 30x |
> | Bit allocation within semantic regime | N/A (no regime) | Water-filled by eigenvalue | Optimal allocation by Shannon-Berger |
>
> Each mechanism contributes a slice of the per-layer attention improvement. Combined and compounded across 32 layers, you get the perplexity numbers measured: TQ adds 22% perplexity at 3 bits, SQ adds 7.5%.
>
> ## Why the gap widens at lower bits
>
> | avg_bits | TQ has to quantize each dim with | SQ can quantize the dominant dim with |
> |---|---|---|
> | 5 | 32 levels (plenty) | 64+ levels via water-fill |
> | 3 | 8 levels (tight) | 16–32 levels via water-fill |
> | 2 | 4 levels (broken) | 16–32 levels via water-fill (rest of budget on tail) |
>
> At 5 bits, TQ has enough precision per dim that randomness doesn't hurt much. At 2 bits, TQ has only 4 levels per dim — not enough to distinguish meaningful K values. SQ doesn't suffer because its budget is concentrated where it matters: the dominant dim still gets enough precision to be useful, and tail dims weren't carrying meaningful information anyway.
>
> This is why the SQ-vs-TQ delta measured grows from +0.6 pp at 5 bits → +4 pp at 3 bits → +11 pp at 2 bits on attention cosine (and +0.6% → +14% → +454% perplexity overhead reduction). The mechanism is the same — small per-layer wins compound — but tighter budgets reveal the gap more dramatically because TQ's data-oblivious uniform allocation runs out of headroom faster than SQ's calibrated allocation.
>
> ## TL;DR
>
> TurboQuant is data-oblivious — it spreads quantization bits evenly across all 128 dimensions because it doesn't know which ones are important. SpectralQuant calibrates once to discover that ~4 dimensions hold ~80% of the information, then concentrates its bit budget on those dims while skipping QJL correction on the noisy tail. This per-layer attention quality improvement compounds across 32 transformer layers, turning small per-head wins into large perplexity wins — and at extreme compression (2 bits), it's the difference between a working model and a broken one.

## Caveats applied when integrating into the consolidated paper

1. **Naming.** "v1 / v2" framing in the note is mapped to one-method language in the consolidated paper: *uniform-allocation special case* (= initial-evidence-layer SpectralQuant) and *water-filled allocation* (= expanded-evidence-layer SpectralQuant). v1/v2 labels appear in the consolidated paper only in the development-history and traceability sections.
2. **Mistral perplexity table.** Treated as supporting-note observation, not paper-valid evidence. The consolidated paper does **not** quote the 12.22 / 12.36 / 12.33 / 12.29 / 14.88 / 82.13 / 25.13 / 26.71 numbers; it cites the *qualitative shape* (TurboQuant collapses at low bit widths on this model; SpectralQuant degrades gracefully) as a registered mechanistic claim sourced from this note, with the explicit caveat that no schema-valid Mistral perplexity artifact exists in this repository at the time of writing.
3. **Calibration cost.** The note's "15-second calibration" is qualitative; the paper-valid amortization-curve evidence is `V1-GAP-007` (open). The consolidated paper says "calibration takes wall-clock seconds per (layer, head); amortizes over the cache lifetime" rather than quoting a single number from the note.
4. **80–90% information in 3–5 dimensions.** Anchored in the paper to `results/memory_efficiency/all_models.json` (V1-RESULT-001, deff/d_h ≈ 3–5%) rather than to the note. The note's sentence is presented as an authors' summary of that universality result, not a separate empirical claim.
5. **Layer-wise compounding.** Presented in the paper as a *plausible explanatory account* of the observed perplexity gap consistent with the per-layer attention-cosine improvements (RUN-ATTNCOS-MISTRAL-7B-V0.3, RUN-ATTNCOS-QWEN2.5-7B); not as a directly measured per-layer error-propagation result. No per-layer-residual-stream attribution artifact exists in `results/`.
6. **Cross-model universality of d_eff.** The note's "Mistral, Qwen, ESM-2, ViT" sentence is narrowed in the paper to the LLM models for which `results/memory_efficiency/all_models.json` carries paper-valid evidence; ESM-2 / ViT references are not treated as paper evidence.
