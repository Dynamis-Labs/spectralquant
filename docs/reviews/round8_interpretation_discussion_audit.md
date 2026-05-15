# Round 8 audit — Interpretation paragraphs and Discussion / Future-directions section

**Scope.** This round covers the May 2026 expansion of
`paper_output_consolidated/spectralquant_unrestricted_paper.tex` that
adds per-result Interpretation paragraphs to each Results subsection,
tightens the existing §Interpretation lead with a unifying SNR /
effective-subspace framing, and inserts a new §Discussion and
forward-looking implications section before the development-history
section. This is the last major content expansion before NeurIPS-format
preparation.

The audit categorizes every new claim along four axes:

- **direct evidence** — the claim restates or quotes a number from a
  schema-validated artifact already in `results/v3/modal/` or
  `docs/full_matrix_evidence_summary.md` §3, or from a value already
  cited in an earlier section of the manuscript.
- **derived math** — the claim follows by standard rate--distortion or
  linear-algebra reasoning from a primitive in §Background or §Method
  and from a measured covariance object.
- **interpretation consistent with evidence** — the claim is a
  mechanistic reading of a measured number that is consistent with the
  evidence but is not itself an additional measurement (e.g. layerwise
  accumulation, SNR threshold language, regularization-as-noise).
- **speculation / future work** — the claim is forward-looking and is
  marked as such in the manuscript; this category is concentrated in
  §Discussion.

The rule for §Discussion is that every paragraph carries one of the
four labels in-text, and that no §Discussion claim is used to back any
empirical sentence in §Results, §Interpretation, or the abstract.

---

## §Results — per-result Interpretation paragraphs

### Perplexity (§5.2)
- **direct evidence.** The reported perplexities $6.49$ (FP16),
  $6.98$ (SQ), $2{,}049$ (TQ) and the cosine $0.40$ on Qwen2.5-7B at
  $b\!=\!3$ are quoted from §5.1 (headline table) and §5.6
  (three-way matrix), themselves quoted from
  `results/v3/modal/perplexity__Qwen2.5-7B__b3_*.json` and
  `docs/full_matrix_evidence_summary.md` §3.
- **derived math.** "Calibrated rotation places almost all of the
  per-(layer, head) key variance into the $\deff$ semantic
  coordinates" follows from $\Sigma=U\Lambda U^\top$ and the
  participation-ratio definition (§Background eq. 1).
- **interpretation consistent with evidence.** The "SNR readout"
  framing and "per-(layer, head) noise accumulates through depth"
  framing. Marked in-text as "We treat the layerwise-accumulation
  picture as a *plausible mechanistic account* consistent with the
  per-layer attention-cosine matrix; we do not claim a directly
  measured per-layer error-propagation result."
- **speculation.** None in the perplexity Interpretation paragraph.

### Generation (§5.3)
- **direct evidence.** F1 $0.482$ / $0.120$ and distinct-2
  $0.603$ / $0.301$ are quoted from §5.1 and §5.3, from
  `results/v3/modal/generation__Qwen2.5-7B__b3_*.json`.
- **derived math.** "Greedy decoding selects the argmax" is a
  definitional statement.
- **interpretation consistent with evidence.** The mapping from F1 to
  per-step distribution fidelity, the mapping from distinct-$n$ to
  distribution shape, and the degenerate-looping account of TQ's
  collapse. The paragraph explicitly says generation "corroborates
  rather than independently confirms" perplexity — both are
  downstream of the same SNR.
- **speculation.** None.

### LongBench (§5.4)
- **direct evidence.** The five per-task numbers and the macro-beats
  numbers are quoted verbatim from Table 2 / §5.4, themselves quoted
  from
  `results/v3/modal/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_*.json`.
- **derived math.** "QA tasks hinge on attention being able to
  retrieve a small number of evidence spans" is a property of the
  task, not a measurement.
- **interpretation consistent with evidence.** The per-task
  decomposition (qasper / narrativeqa / hotpotqa benefit because
  high-variance memory subspace is preserved; gov_report / trec lag
  because they depend on lower-variance distributional properties).
  The "compression-as-regularization" reading of why FP16 is beaten
  on QA is registered explicitly as "plausible but not directly
  measured here."
- **speculation.** Implicit: the qualitative prediction that a
  full-suite multi-seed run "could realistically show the macro at or
  below FP16" is forward-looking. It is registered with "We report
  the $+14.2\%$ relative number as a directional signal, not as a
  universal LongBench claim."

### Latency (§5.5)
- **direct evidence.** The kernel microbench number
  ($0.06\,\mathrm{ms/token}$ at ctx 1024) and the hooked-replay
  numbers are quoted from §5.5 and the source JSON.
- **derived math.** None new.
- **interpretation consistent with evidence.** "Quality-valid before
  systems-valid"; the diagnosis that hook overhead dominates rather
  than the kernel; the production-kernel preconditions (Cache
  subclass, fused decompression, preserved memory bandwidth). These
  are restatements of the existing §5.5 diagnostic framing in causal
  language.
- **speculation.** "Until that systems work lands, the latency
  evidence in this manuscript supports only the kernel-level claim;
  serving-time speedups remain explicit open work" — registered in
  Limitations §7.2.

### Three-way attention cosine (§5.6)
- **direct evidence.** All quoted deltas ($+0.27$ to $+0.38$ vs TQ;
  $\le +0.006$ at $b\!\in\!\{3,5\}$, $+0.018$ at $b\!=\!2$ on
  Mistral-7B-v0.3; $0.626$ vs $0.738$ at $b\!=\!2$) are sourced from
  Table 3 / §Mechanism.
- **derived math.** Greedy water-filling matching the Gaussian
  rate--distortion solution at fixed total bits is a direct restatement
  of §Method eq. (rd) and §Background §2.4.
- **interpretation consistent with evidence.** "Closest mechanism-
  level evidence in this manuscript"; small per-layer cosine deltas
  compounding through depth; the existence of a narrow attention-
  cosine threshold below which downstream metrics fall off a cliff.
  Compounding is consistent with the per-layer cosine matrix; the
  existence of a sharp threshold is consistent with the order-of-
  magnitude separation between TQ and SQ on perplexity but is not
  itself a measurement of the threshold.
- **speculation.** None — paragraph stays in interpretation territory.

### Compression accounting (§5.7)
- **direct evidence.** The "within $0.013\times$" matched-ratio
  statement and the $\approx\!3$ bits/element numbers are quoted from
  §5.6 / §5.7.
- **derived math.** None new.
- **interpretation consistent with evidence.** The framing that
  matched ratios make the rest of the paper a causal-attribution
  argument, and that the headline $5.95\times$ initial-layer number
  must include codebook bytes / JL / metadata / per-head asymmetry
  rather than a $16/b$ back-of-envelope calculation.
- **speculation.** None.

---

## §Interpretation lead synthesis

The new lead paragraph collects the SNR / effective-subspace /
layerwise-accumulation framing and labels every result family in one
sentence each. All claims are restatements of existing §Results
content with a unifying vocabulary. No new empirical claims are
introduced; the existing five numbered paragraphs (1)–(5) are kept
verbatim and the §Mechanism subsection is unchanged. Category:
**interpretation consistent with evidence**.

---

## §Discussion and forward-looking implications (new section)

This is the only section in the manuscript whose primary purpose is
forward-looking. The opening paragraph carries an explicit four-label
key (evidence-backed / derived / interpretation-consistent /
speculation) and the section header is *Discussion and forward-looking
implications*; downstream subsections each include an in-paragraph
label.

### §8.1 Distribution-aware inference
- **evidence-backed:** "calibrated rotation already buys $+0.27$ to
  $+0.38$ attention cosine over a data-oblivious one at matched
  compression ratio (§5.6)."
- **derived math:** "the eigenbasis is the variance-maximizing
  rotation under a Gaussian rate--distortion problem (§2.4)."
- **speculative:** "the same calibration object can be reused across
  non-quantization compressors (low-rank cache projections, H2O-style
  eviction policies, attention-sparsity kernels)" — labelled as such.
- **speculative:** the practical near-term Hugging Face Cache
  subclass is described conditionally on the production-kernel
  precondition; it is not claimed as an existing artifact.

### §8.2 Training-time implications
All three subsubsections (SNR-aware regularization, explicit
allocation of representational bandwidth, compression-as-
regularization) are explicitly labelled **speculative; future work**.
The compression-as-regularization paragraph specifically warns "treat
the LongBench macro-beats-FP16 number as a directional signal, not as
a license for a regularization claim."

### §8.3 Domain-specific compression and effective subspaces
- **speculative:** biological-sequence transformers (ESM-2,
  AlphaMissense, Evo); clinical / legal / scientific corpora;
  calibration as a measurement tool. Each subsubsection is in-text
  labelled "Future work" or "Speculative". The supporting note's
  reference to ESM-2 and ViT is registered, but no schema-validated
  artifact is claimed.

### §8.4 Interpretability
- **mathematically derived / evidence-backed:** $\deff$ as a scalar
  summary, eigenvalues as a per-coordinate ranking, eigenvectors as
  explicit directions.
- **speculative:** mechanistic-interpretability probes and
  cross-layer compositionality. Both labelled in-text.

### §8.5 Why $\deff$ is small
- **speculative:** softmax-attention bottleneck and GQA forcing-
  function hypotheses. Both labelled in-text. Closing sentence: "These
  are research questions, not claims."

### §8.6 Systems roadmap
- **evidence-backed / forward-looking:** the six numbered priorities
  (production cache subclass, fused decompression, multi-seed /
  multi-arch, full LongBench, official Google TQ, low-level kernels).
  Each item is already named in §Limitations §7; this subsection
  organizes them into a sequence with the SNR / effective-subspace
  framing. No new claims about completed work.

---

## Citation additions

The Discussion section introduces 11 new bibliography keys, all added
verbatim to `paper_output/spectralquant_refs.bib`:

- `zhang2023h2o` — Heavy-Hitter Oracle, NeurIPS 2023.
- `lin2022esm2` — ESM-2 protein LM, Science 2023.
- `cheng2023alphamissense` — AlphaMissense, Science 2023.
- `nguyen2024evo` — Evo genomic LM, Science 2024.
- `srivastava2014dropout` — Dropout, JMLR 2014.
- `fedus2022switch` — Switch Transformers, JMLR 2022.
- `tenney2019bert` — BERT Rediscovers the Classical NLP Pipeline,
  ACL 2019.
- `elhage2022superposition` — Toy Models of Superposition, Anthropic.
- `katharopoulos2020lineartransformers` — Linear Transformers, ICML
  2020.
- `gu2023mamba` — Mamba, arXiv 2023.
- `shazeer2019mqa` — Multi-query attention, arXiv 2019.

All entries are public refs with venue / DOI / arXiv where available;
none are added to back an empirical claim. They appear only inside
**speculative** paragraphs in §Discussion as related work for the
proposed research directions.

---

## Boundary discipline

The audit confirms:

1. No empirical sentence in §Abstract, §Introduction, §Results, or
   §Interpretation depends on any §Discussion claim.
2. The §Mechanism subsection (§5.8) was not modified by this round; it
   remains the boundary between the paper's evidence layer and its
   research-direction layer.
3. The supporting-note Mistral perplexity table is unchanged in status:
   referenced in §5.6 / §Mechanism and §Discussion §8.3 only as
   supporting-note observation. No paper-valid claim depends on it.
4. The single-program / two-evidence-layers framing is unchanged. The
   Discussion section uses "calibrated rotation," "water-filled
   allocation," "uniform-allocation special case," and "initial /
   expanded evidence layer" only; v1 / v2 product framing does not
   reappear.
5. Every paragraph in §Discussion that proposes future measurements
   names the artifact that would unblock the corresponding claim, in
   line with `docs/claims_discipline.md`.

## Result

The May 2026 round-8 expansion is internally consistent with the
schema-validated evidence base and with the prior seven audit rounds.
The manuscript is ready for the NeurIPS-format conversion pass.
