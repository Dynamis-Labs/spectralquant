# Round 6 — Mechanism integration audit

Purpose: verify that the integration of the April 2026 Sentra technical note ("Why SpectralQuant v2 beats TurboQuant on perplexity — the mechanism", Vangara & Gopinath) into the consolidated unrestricted technical report (`paper_output_consolidated/spectralquant_unrestricted_paper.tex`) (i) preserves the supporting note as a registered repository artifact with provenance, (ii) restates the note's content in the single-method "SpectralQuant" voice (no public `v1`/`v2` framing), (iii) does not introduce any unsupported numeric claim, and (iv) routes every mechanism statement through either derived math, prior literature, or already-paper-valid evidence catalog entries.

Audit date: 2026-05-01.

Scope:
- `paper_output_consolidated/spectralquant_unrestricted_paper.tex` (specifically the new §"Mechanism behind the perplexity gap" subsection of §Interpretation, the additions to §Limitations and to the §Claim-to-artifact traceability table).
- `docs/supporting_notes/perplexity_mechanism_note_2026-04.md` (newly created).
- `docs/evidence_catalog.md` (new "Supporting notes" subsection).
- `docs/consolidated_spectralquant_inventory.md` (new pointers in §6 Documentation).

Method: each new sentence in the consolidated paper that originates from the supporting note was placed into one of four buckets:

1. **Math.** Standard rate–distortion or quantization-theory result already cited in the manuscript. No new evidence required.
2. **Cited literature.** Already-cited prior work (`zandieh2024qjl`, `zandieh2025turboquant`, `cover2006elements`, `gao2015linear`).
3. **Already paper-valid.** Anchored in an existing evidence-catalog entry under `docs/evidence_catalog.md`.
4. **Supporting-note only.** Qualitative claim that originates in the note, not in a schema-validated artifact. Must carry a supporting-note caveat.

A claim that drifts into bucket (4) without an explicit caveat is a fail.

## 1. Per-claim audit of §"Mechanism behind the perplexity gap"

| Mechanism / sentence in §Interpretation | Bucket | Anchor |
| --- | --- | --- |
| (M1) Calibrated rotation puts variance in known coordinates; data-oblivious rotation is worst-case-near-optimal over R, not over the data. | 2 (cited lit) + 3 (paper-valid) | `zandieh2025turboquant`; V1-RESULT-001 (`results/memory_efficiency/all_models.json`) for d_eff/d_h ≈ 3–5%. |
| Σ = U Λ U^T diagonalization is exact; cross-coord correlation = 0 by construction. | 1 (math) | Standard linear algebra; already in Background §2.3. |
| (M2) Two-regime allocation; semantic dimensions get more per-dim precision than uniform across all 128 dims. | 1 (math) + 3 | Algorithm 1 (Section "Method"); Section §"Method" notation block. |
| Attention scores are dominated by high-variance directions of K, so per-dim precision in semantic subspace translates to attention-output cosine. | 1 (math) | Quadratic-form interpretation of <Q,K> after rotation; standard. |
| Three-way attention cosine: Δ cos +0.27 to +0.38 vs in-repository TQ; +0.018 at b=2 (Mistral) for water-filled vs uniform. | 3 (paper-valid) | V2-RESULT-ATTNCOS-MATRIX (`docs/full_matrix_evidence_summary.md` §3); RUN-THREEWAY-MISTRAL-{2,3,5}BIT, RUN-THREEWAY-QWEN-3BIT. |
| (M3) Selective QJL: side-info cost reduced by d_h/d_eff factor; estimator variance reduced by same factor. | 2 (cited lit) + 1 (math) | `zandieh2024qjl`; JL variance ∝ projection dimensionality is standard. The "30×" factor is the d_h/d_eff ratio (128/4) reported as "roughly", not as a measured number — qualifier preserved. |
| Skipping QJL on tail is justified because tail variance is at JL noise floor. | 3 (paper-valid) | V1-RESULT-001 + Background §2.3 (eigenvalue-spectrum analysis). |
| (M4) Water-filling is integer projection of classical Gaussian rate–distortion solution; allocates greedily by λ_i · 4^(-b_i). | 1 (math) + 2 (cited lit) | Eq. (5) (rate–distortion problem), Algorithm 1 (greedy integer projection); `cover2006elements`, `gao2015linear`. |
| At high b allocation is close to uniform; at low b the constraint binds and water-filling concentrates precision on dominant directions. | 1 (math) | Direct property of the water-filling solution; matches the matched-compression-ratio table. |
| Per-layer headwind shows up as a perplexity gap; depth-wise compounding. | **4 (supporting-note-only)** | Explicitly flagged in the paper as "plausible explanatory account, not a directly measured per-layer error-propagation result"; consistent with the per-layer attention-cosine matrix; no per-layer residual-stream attribution artifact in `results/`. **Caveat present.** |
| Order-of-magnitude separation between SpectralQuant (perplexity within 7.5% of FP16) and the in-repository TQ arm (perplexity ≈ 2 049). | 3 (paper-valid) | RUN-PERPLEXITY-QWEN2.5-7B (`results/v3/modal/perplexity__Qwen2.5-7B__b3_seed42_*.json`). |
| Mistral / WikiText-103 perplexity table shape (TQ collapses at low b; SpectralQuant degrades gracefully). | **4 (supporting-note-only)** | Explicitly flagged: "registered as supporting-note observation, not paper-valid"; pointer to `docs/supporting_notes/perplexity_mechanism_note_2026-04.md` ("Evidence status"); the b=2 attention-cosine result on Mistral (RUN-THREEWAY-MISTRAL-2BIT) is the closest paper-valid analog. **Caveat present.** |
| "Why the gap widens at lower bit budgets" — Δ cosine grows from < 0.03 at b=5 to 0.11 at b=2. | 3 (paper-valid) | V2-RESULT-ATTNCOS-MATRIX. |

**Verdict.** Every bucket-(4) sentence in the new subsection carries an explicit supporting-note caveat in the same paragraph. No bucket-(4) sentence is allowed to anchor a numeric claim in any table, figure, or abstract sentence.

## 2. Single-method framing audit (no public v1 / v2 in mechanism subsection)

A `grep '\bv1\b|\bv2\b|SpectralQuant v1|SpectralQuant v2'` over the new subsection in `paper_output_consolidated/spectralquant_unrestricted_paper.tex` returns only:

- The reference to `RUN-THREEWAY-MISTRAL-{2,3,5}BIT` and `RUN-THREEWAY-QWEN-3BIT` (Category (ii) repository-label / audit-stability — same justification as in Round 5).
- The path `docs/supporting_notes/perplexity_mechanism_note_2026-04.md` (Category (ii) filesystem path).
- The phrase "uniform-allocation special case" / "water-filled allocation" used throughout the new subsection.

Public-narrative hits ("SpectralQuant v2 beats…"): **none in the new subsection**. The note's original title is preserved verbatim only inside the supporting-note artifact, where the provenance header explicitly states that the consolidated paper restates the content in single-method language.

## 3. Caveat preservation

| Pre-existing caveat | Status in the additions |
| --- | --- |
| Single seed (42), single bit budget (b=3) for the four next-stage families | UNCHANGED — the new subsection cites the same artifacts. |
| In-repository TurboQuant (not official Google TurboQuant) for every TQ comparison | UNCHANGED — the new subsection refers to the "data-oblivious arm" / "in-repository TQ", consistent with §Limitations item 3. |
| LongBench is the deterministic 5-task subset, not full LongBench | UNCHANGED — the mechanism subsection does not reframe the LongBench claim. |
| 5.33× memory factor at b=3 unchanged by expanded layer | UNCHANGED — no new memory claim. |
| d_eff/d_h ≈ 3–5% across tested models (V1-RESULT-001) | PRESERVED — the supporting note's "80–90% of the useful information lives in 3–5 dimensions" is anchored to V1-RESULT-001 and the wording in the consolidated paper does not paraphrase the note's "across every model we've measured (Mistral, Qwen, ESM-2, ViT)" sentence (ESM-2 / ViT not in V1-RESULT-001). |
| 15-second calibration provenance (V1-GAP-007) | PRESERVED — the consolidated paper retains the existing Limitations item; the supporting note's "15-second calibration" phrase is **not** quoted in the consolidated paper's new subsection. |

## 4. New artifact registration

- `docs/supporting_notes/perplexity_mechanism_note_2026-04.md` is created with provenance header (origin, authors, capture date, original title preserved verbatim, evidence-status flag).
- `docs/evidence_catalog.md` adds a "Supporting notes" subsection registering `SUPPORT-NOTE-PERPLEXITY-MECHANISM-2026-04` and listing exactly which numeric values from the note are NOT paper-valid.
- `docs/consolidated_spectralquant_inventory.md` §6 Documentation adds pointers to the supporting note and to this audit.
- `paper_output_consolidated/spectralquant_unrestricted_paper.tex` traceability table adds a row anchoring the mechanism narrative to the math, the cited literature, and (where unavoidable) the supporting note with its caveat.

## 5. Sign-off

The integration follows Round 5's single-method discipline and `docs/claims_discipline.md`. No paper-valid claim is weakened; no new paper-valid claim is added without anchor. Mechanism narrative is traceable to math + cited literature + already-registered evidence; the two unavoidably-supporting-note-only items (depth-wise compounding picture, Mistral perplexity table shape) carry explicit caveats and are not used to back any numeric value in any table, figure, or abstract sentence.

Round 6 closes with no open action items beyond the existing `V1-GAP-*` set already tracked in `docs/evidence_catalog.md`.
