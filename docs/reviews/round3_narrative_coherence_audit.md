# Round 3 — Narrative Coherence Audit

**Subject.** `paper_output_v2/spectralquant_v2_full_story.md`.
**Auditor.** Cross-section read at commit `96e229c`, 2026-05-01.
**Question for this round.** Do the sections flow cleanly from one
to the next? Are caveats carried consistently across sections? Are
there any internal contradictions?

## 1. Section flow

The 18 sections follow this arc:

1. **Hook → empirical thesis** (§0): one-page summary of every
   conclusive claim with its evidence pointer; explicit boundary
   between what the evidence supports and what it does not.
2. **Why care** (§1): KV-cache memory is the dominant inference
   cost at long context, sized with concrete numbers; structure
   wins because hardware does not save us.
3. **Background** (§2): attention, KV cache, GQA, RoPE, KV
   quantization, spectral methods, water-filling, Lloyd–Max,
   TurboQuant, metrics. Written for the new reader; every concept
   we will use later is introduced here.
4. **v1 story** (§3): bias-variance argument, algorithm,
   engine, what v1 measured, what v1 left on the table. Sets up
   v2 as a focused fix to a specific gap.
5. **v2 derivation and engine** (§4): optimization, water-filling
   solution, greedy integer variant, per-dim Lloyd–Max codebooks,
   why ratio is held identical to v1, engine architecture, replay,
   calibration cost.
6. **Experimental protocol** (§5): the shared configuration of the
   four next-stage families, the gates, the schemas.
7. **Evidence presentations** (§6 PPL, §7 Generation, §8 LongBench,
   §9 Latency, §10 Three-way cosine): each section follows the same
   "Configuration → Numbers (verbatim) → Reading → Caveats" template.
8. **Compression accounting cross-check** (§11): why ratios are
   trustworthy across v1 and v2.
9. **Interpretation** (§12): the three-line story
   ("structure beats budget"; "fixed compression ratio";
   "TurboQuant collapses") tied directly back to numbers from §6–10.
10. **Limitations** (§13), **traceability** (§14), **defensible
    claim wording** (§15): the cleanup needed to use this document
    as a reference.
11. **Reproducibility** (§16), **conflict of interest** (§17),
    **document evidence trail** (§18): the meta layer.

The arc is: *why care → primer → v1 (what stuck and what didn't) → v2
(derivation) → measured evidence → interpretation → caveats →
defensible language → reproduce*. Each step is the natural next step
from the previous one, and §12 (Interpretation) is the explicit
synthesis that ties §6–10 back to §0's thesis.

## 2. Caveat propagation

The four caveats listed in the user's brief are carried verbatim
through the document at every place they apply:

### 2.1 Single seed

| Section | Carried? |
|---|---|
| §0 (thesis) | "single-seed, single-bit budget for the four next-stage families, single-model for those families" |
| §5 (protocol) | "Seed 42" listed in shared knobs table |
| §6 (PPL) | §6.4: "Single seed; multi-seed CIs not unblocked" |
| §7 (generation) | §7.4: "8 prompts is small; no multi-seed" |
| §8 (LongBench) | §8.6: "Single seed (42)" |
| §10 (three-way) | §10.4: "Single seed" |
| §13 (limitations) | §13.1: top of list |
| §15 (claim wording) | "single seed (42)" embedded in headline sentence |

### 2.2 Single model / single bit budget

| Section | Carried? |
|---|---|
| §0 | "single-model for those families (Qwen2.5-7B)" |
| §5 | Shared knobs table: bit budget = 3 |
| §6, §7, §8 | All scoped to Qwen2.5-7B b=3 |
| §13 | §13.5: "Multi-architecture under the next-stage harness" listed |
| §15 | Embedded |

### 2.3 In-repo TurboQuant comparator (not official Google)

| Section | Carried? |
|---|---|
| §0 | "uses the in-repo TurboQuant comparator rather than the official Google or Blackwell kernels" |
| §2.6 | Background explicitly defines the in-repo vs. community vs. official distinction |
| §5 | "(local)" suffix on TurboQuant rows |
| §6 | "in-repo TurboQuant" in §6.3 reading |
| §7 | "in-repo TurboQuant" in §7.3 reading |
| §8 | "in-repo TurboQuant" + §8.6 limitations |
| §10 | §10.4: "TurboQuant arm is the local (in-repo) baseline" |
| §13 | §13.3: explicit V1-GAP-012 / V1-GAP-013 |
| §15 | "in-repo TurboQuant comparator" in headline |

### 2.4 Deterministic 5-task LongBench subset (not full 21-task)

| Section | Carried? |
|---|---|
| §0 | "deterministic 5-task LongBench subset rather than the full 21-task suite" |
| §8 (entire section) | Section heading itself is "deterministic 5-task subset"; §8.5 mandatory subset caveat |
| §13 | §13.4: "Headline must say '5-task LongBench subset'; full 21-task remains unblocked" |
| §15 | "deterministic 5-task LongBench subset" in headline |

### 2.5 No production-kernel speedup claim

| Section | Carried? |
|---|---|
| §0 | "hooked-replay end-to-end latency that is 4–40× slower than FP16 because per-layer Python forward hooks dominate the wall-clock — `production_kernel = false` is stamped on every JSON" |
| §4.6 | Engine section: hooks add overhead; production-kernel claim is open work |
| §9 (entire section) | §9.1: two measurement kinds; §9.3: "this is *not* a production-kernel speed claim"; §9.5: "Nothing about wall-clock serving speedup" |
| §13 | §13.2: "Production-kernel latency" listed |
| §15 | "(production_kernel = false), and a production-kernel speedup claim is open work" |

### 2.6 Pre-existing test failures

| Section | Carried? |
|---|---|
| §13 | §13.8: "Six pre-existing calibration/quantization test failures … do not affect any number reported here" |

All five caveats are carried consistently. None of them are
introduced and then dropped.

## 3. Internal contradictions

I scanned for contradictions on the load-bearing numbers:

- **PPL.** 6.49 / 6.98 / 2 048.6 in §0, §6.2, §6.3, §12, §15. All
  match. No contradiction.
- **Generation token-F1.** 0.48 / 0.12 in §0, §7.2, §7.3, §12, §15.
  All match.
- **LongBench macro.** 0.1755 / 0.2004 / 0.0044 in §0, §8.2, §8.3,
  §12, §15. All match.
- **Compression ratio at b=3.** 5.33× (16/3) in §0, §1.2, §6.2 (avg
  bits column), §9.4, §15. All match.
- **Replay coverage.** 1.0 (28/28) in §0, §4.6, §6.2, §8.1. All
  match.
- **Three-way attention cosine.** Δ ∈ [+0.27, +0.38] vs TurboQuant,
  +0.018 v2-vs-v1 at b=2 Mistral in §0, §10.2, §12, §15. All match.
- **Hooked-replay latency.** 630 ms/tok at ctx=1024 in §0, §4.6, §9.2,
  §9.3. All match. K/V microbench 0.06 ms/tok at ctx=1024 in §0,
  §9.2, §9.3, §15. All match.
- **Bit ratio language.** §1.2 says "16/b reduction"; §6.2 reports
  "Avg bits 3.0" for the compressed methods; §15 says "5.33× smaller
  than FP16 (16/3)". All consistent with b=3.

No internal contradiction found.

## 4. Tension between FP16-comparison sentences

§8.3 says SpectralQuant v2 "macro-beats FP16 on this subset by
+0.0250 absolute (+14.2 % relative)". §8.6 then carefully notes
"single seed", "n=50/task", "no per-task error bars", "the +0.0250
macro delta has no CI". §15 carries the same wording: "macro-beats
FP16 on a deterministic 5-task LongBench subset (0.2004 vs 0.1755)".

This is the section pair I worried most about (it is the closest the
manuscript gets to a "we beat FP16" claim, which is a strong
statement that would be embarrassing if pushed without caveat).
The carry is consistent: the FP16 macro-beat is reported as observed,
the n=50 / single-seed caveat is carried into the next paragraph in
§8.3 itself ("the 0.156 absolute gap is suspect on its own"), and §15
preserves the subset tag in the defensible-claim wording.

§12.5 also explicitly says "FP16 is the right comparator for 'good
enough', not for 'wins'. Beating FP16 on a 5-task subset at b = 3 on
a single seed is a clean directional signal. We do not claim it as a
universal LongBench result." This is exactly the right framing and it
is *consistent* with the rest of the document.

## 5. Tension between latency narratives

Three latency stories coexist in the manuscript:

1. **Memory-only** (§0, §1.2, §9.4, §15): the cache itself is 5.33×
   smaller; this is structural and unchanged by v2.
2. **K/V microbenchmark** (§9.2, §9.3, §9.4): the compress/decompress
   round-trip is 0.06 ms/tok at ctx=1024 on H200 — i.e. the kernel
   is not a showstopper, even though TurboQuant's microbenchmark is
   faster.
3. **Hooked-replay end-to-end** (§4.6, §9.2, §9.3, §9.5): SQv2 at
   ctx=1024 is 630 ms/tok vs FP16 17.66; this is a Python overhead
   measurement, not a kernel measurement, and is stamped
   `production_kernel = false`.

The manuscript handles this tension well in §9.3 by labelling both
measurement kinds and explaining why each one tells a different
story. §9.5 ("What latency does *not* prove") is specifically where
the worry "users might confuse 0.06 ms/tok for an end-to-end speedup
claim" gets blocked. §15 ("Latency, narrow") preserves the
distinction in the defensible claim wording.

This three-way framing is correct and consistently applied.

## 6. Tension between v1 5.95× ratio and the v2 ratios

§3.3 cites v1's 5.95× headline at b ≈ 3 on Qwen2.5-14B-Instruct.
§4.5 says v1 and v2 ratios are within 0.013× of each other on the
four three-way runs (Mistral b=2/3/5, Qwen-7B b=3) — none of which is
the 14B model. §11 explicitly reconciles: "the headline v1 number
(5.95× at b ≈ 3 on Qwen2.5-14B-Instruct) is not unblocked by these
v2 runs; it is preserved in the v1 evidence catalog with the
V1-GAP-014 caveat that the spec §10 formula does not yield 5.95×
from b = 3, $d_{\mathrm{eff}}$ = 3 alone".

This is the correct handling: v1's 5.95× headline is documented as a
historical empirical number with its own caveat; the v2 work runs at
ratios in [3.08, 7.30] (§10.2), at compression-ratio parity with v1
within 0.013×, and does not attempt to "rederive" 5.95×. No
contradiction.

## 7. Sentra-fundraising framing

§0 ("structure beats budget"), §1.3 ("why structure rather than
scale"), §12 (interpretation), and §15 (defensible claim wording)
form the through-line on the Sentra-narrative side. The phrasing is
consistent — *structure beats budget at fixed compression ratio* —
and §15 is the exact one-page extract of "what may appear in
fundraising materials". The framing does not strengthen the empirical
claims; it stays inside the boundaries set by §13's limitations list.

## 8. Issues found

None. The narrative is internally coherent, the caveats are carried
across every section that needs them, the FP16-comparison and
latency-three-way tensions are handled explicitly rather than papered
over, and the v1-vs-v2 ratio relationship is documented cleanly.

## 9. Outcome

**PASS.** The full story flows from motivation through derivation,
evidence, interpretation, and limitations without contradiction. The
five mandatory caveats (single seed, single bit / single model,
in-repo TurboQuant comparator, deterministic-subset LongBench, no
production-kernel speedup) appear in every section that touches them
and in the defensible-claim wording in §15.
