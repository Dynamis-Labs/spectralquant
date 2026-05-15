# Round 9 — Paper framing and per-result interpretation audit

Date: 2026-05-01
Scope: `paper_output_consolidated/spectralquant_unrestricted_paper.tex` (and its bibfile `paper_output/spectralquant_refs.bib`).

This audit records the round-9 changes made in response to user feedback that
(a) the manuscript was framing itself administratively rather than academically,
and (b) only the perplexity result had a mechanism-depth interpretation while
the other result families were left under-interpreted.

## 1. Summary of user feedback addressed

1. Do not present the paper as a "technical report with end-to-end repository
   traceability". Drop that subtitle and that framing identity.
2. Remove "May 2026" from the title block.
3. Remove rename history from the main paper (i.e. `niashwin/spectralquant-v2`
   → `niashwin/spectralquant-full`, references to "original public artifact",
   "consolidation repository", etc.). Keep such admin only in docs.
4. Remove "unrestricted-length consolidated technical report" / "consolidated
   technical report" framing. Read like a paper introducing SpectralQuant.
5. Provide mechanism-level interpretation for *every* main result family at
   roughly the same depth as the existing perplexity-mechanism subsection.
6. Keep Discussion / forward implications / Limitations / Reproducibility as
   normal academic-paper sections, not as administrative sections.

## 2. Specific framing language removed

The following strings were removed or replaced in `spectralquant_unrestricted_paper.tex`:

| Removed / replaced | Replacement |
|---|---|
| Subtitle: "Consolidated Technical Report with End-to-End Repository Traceability" | Removed; title is now the algorithm name only. |
| `\date{May 2026 \quad\textbar\quad Consolidation repository: ... (renamed from niashwin/spectralquant-v2) \quad\textbar\quad Original public artifact: ...}` and the trailing "This is the unrestricted-length consolidated technical report" sentence | `\date{}` (suppressed). |
| Header right-side "consolidated technical report" decoration | Just the page number. |
| Comment block at top: "consolidated unrestricted-length technical report" with `v1`/`v2` repository-label preamble | Trimmed to a normal source-of-truth comment. |
| Abstract paragraph: "The empirical evidence integrated in this report comes from two development layers of the same research program. The *initial evidence layer* ... labelled `v1` in the consolidation repository's git history ... The *expanded evidence layer*, internally tagged `v2` during development ..." | "We report two complementary bodies of evidence on the same algorithm. A first round ... A second round ..." (no rename history, no `v1`/`v2`, no consolidation language). |
| Abstract close: "Every empirical sentence in this manuscript carries an explicit pointer into the `niashwin/spectralquant-full` repository ... The Sentra-relevant reading is unchanged ..." | Trimmed to "The take-away is unchanged across every experiment: structure beats budget." |
| Post-abstract block citing "consolidated unrestricted-length technical report ... earlier development snapshots ... audits supporting this text are in ..." | Removed entirely. |
| Intro contribution 2 title: "Empirical evaluation across two evidence layers." | "An empirical evaluation in two parts." |
| Intro paragraph wording: "The initial evidence layer ... The expanded evidence layer ... Every expanded-layer artifact is schema-validated ..." | "A first round ... A second round ... Every second-round artifact is schema-validated ..." |
| Manuscript-organization paragraph mentioning Section `sec:devhistory` and the traceability section's purpose ("the claim-to-artifact traceability table that anchors every empirical sentence in this manuscript to a repository path across the two evidence layers") | Single sentence mentioning the reproducibility recipe and the traceability map without admin framing. |
| Entire `\section{Development history and repository labels}` (subsections: "Initial public release ...", "Expanded evidence layer ...", "When v1 / v2 labels appear") | Deleted from the paper. (Admin/provenance is preserved in `docs/` not in the manuscript.) |
| Reproducibility "Repository layout" bullets that named "renamed from `niashwin/spectralquant-v2`", "Historical manuscript snapshots (kept for provenance) ...", and full audit-list | Trimmed to: repository, spec, manuscript, bibliography, Modal runbook. |
| Reproducibility "Three-way attention cosine" bullet's parenthetical "(volume name retained for provenance; no longer reflects the renamed repository)" | Removed. |
| Conclusion sentence: "The consolidation repository `niashwin/spectralquant-full` (renamed from `niashwin/spectralquant-v2`), which preserves the original public release as its initial evidence layer and the expanded paper-valid runs as its second evidence layer, is sufficient to reproduce ..." | "The repository `niashwin/spectralquant-full` is sufficient to reproduce ..." |
| Conflict-of-interest sentence: "the consolidation repository `niashwin/spectralquant-full` (renamed from `niashwin/spectralquant-v2`; carries both evidence layers and incorporates the original public release at `Dynamis-Labs/spectralquant`)" | "the repository `niashwin/spectralquant-full`" |
| Discussion paragraph "The calibration objects to support this analysis already exist in the consolidation repository" | "... already exist in the repository" |

The `v1`/`v2` evidence-catalog identifiers (`V1-RESULT-*`, `V2-SPEC-*`) and
JSON method keys (`spectralquant_v2`) are preserved in artifact pointers and
in `\evid{...}` annotations because they are stable identifiers minted on disk;
the manuscript no longer makes a feature of them. The phrases "first round"
and "second round" replace "initial evidence layer" / "expanded evidence
layer" in a few in-line spots; some downstream references to "expanded layer"
remain in the Discussion / Method context paragraphs where the contrast with
the first round is informational rather than administrative.

## 3. Result-family interpretation subsections added or expanded

The Interpretation section was restructured. The previous structure had
`\section{Interpretation}` with a single `\subsection{Mechanism behind the
perplexity gap}` plus a list of synthesis paragraphs. The new structure is:

- `\subsection{Synthesis}` — the existing five-paragraph synthesis (1)–(5),
  plus the "Sentra-relevant reading" paragraph, all retained.
- `\subsection{Mechanism behind the perplexity gap}` — existing, retained
  intact (M1)–(M4) + depth-compounding + low-bit-widening + claim-discipline.
- `\subsection{Mechanism behind the attention-output cosine matrix}` — **new**.
  Three paragraphs: (C1) per-layer SNR set by post-rotation per-coordinate
  allocation; (C2) water-filling matters most at low $b$ at fixed compression
  ratio (rate–distortion derivation); (C3) what cosine can and cannot predict
  (sign and ordering, not magnitude), with caveats on multi-seed and
  full-depth coverage.
- `\subsection{Mechanism behind generation quality}` — **new**.
  Three paragraphs: (G1) F1 reads attention-score SNR through the argmax;
  (G2) distinct-$n$ reads distribution shape (cites
  `holtzman2019curious`); (G3) cumulative drift compounds the per-step
  perturbation. Caveat on small prompt set.
- `\subsection{Mechanism behind the LongBench task split}` — **new**.
  Three paragraphs: (L1) retrieval/QA tasks live in the high-variance memory
  subspace (cites `olsson2022context` for the mechanistic-interpretability
  evidence); (L2) distribution-style tasks rely on tail coordinates that
  quantization attenuates; (L3) why macro $>$ FP16 must be read cautiously
  (qasper-driven, compression-as-regularization speculation, subset bias).
- `\subsection{Mechanism behind the latency split}` — **new**.
  Three paragraphs: (T1) microbench measures the kernel; (T2) hooked replay
  measures Python overhead, not the kernel; (T3) what systems integration
  must change (Cache subclass, fused decompression, metadata residency).
- `\subsection{Mechanism behind the compression accounting}` — **new**.
  Three paragraphs: (A1) why matched ratio matters for causal attribution;
  (A2) why the simple $16/b$ story is not the right accounting (codebook
  bytes, JL projection, calibration metadata, per-(layer,head) $\deff$);
  (A3) what this changes for downstream comparisons.
- `\subsection{Mechanism behind the participation-ratio universality}` —
  **new**. Three paragraphs: (D1) why $\deff$ is small (rank-selectivity
  of attention; cites `olsson2022context`); (D2) why a small calibration
  set suffices ($\approx$15 s, rank-$\deff$ identifiability); (D3) caveats
  on the empirical universality (no proof; multi-architecture coverage open).

Each new subsection follows the same shape as the existing perplexity-mechanism
subsection: numbered mechanism statements with explicit caveats and explicit
boundaries on what is and isn't claimed. Speculative content is flagged
in-line ("speculative", "consistent with", "directly measured ... is not").

## 4. Bibliography additions

Two entries added to `paper_output/spectralquant_refs.bib`:

- `olsson2022context` — Olsson et al., "In-context Learning and Induction
  Heads", Transformer Circuits Thread (2022). Used as the mechanistic-
  interpretability citation for retrieval-style attention heads in the
  LongBench mechanism subsection and for the rank-selectivity argument in
  the participation-ratio mechanism subsection.
- `holtzman2019curious` — Holtzman et al., "The Curious Case of Neural Text
  Degeneration", ICLR 2020. Used as the degenerate-decoding reference in
  the generation-quality mechanism subsection.

Both pass `bibtex` cleanly and resolve to numeric citations in the bbl.

## 5. Sections kept (unchanged in content, preserved per user feedback)

- Background, Related work, Method (algorithm derivation, water-filling,
  Lloyd–Max codebooks, engine architecture), Experimental protocol, Results
  (all four next-stage families + three-way matrix + compression accounting
  cross-check) — all unchanged.
- Limitations and remaining work — unchanged.
- Discussion and forward-looking implications — unchanged framing per user
  feedback that "Discussion, forward implications, limitations, remaining
  work, and reproducibility are fine in principle".
- Reproducibility — preserved as a normal reproducibility section. Trimmed
  to drop rename-history bullets and historical-snapshot bullets, kept the
  artifact JSON listing and Modal launch commands.
- Claim-to-artifact traceability — preserved (the tabular form is normal
  paper-style artifact pointer documentation, not "repository audit"
  framing).
- Conclusion / Conflict of interest / Acknowledgments — kept; rename-history
  language removed.

## 6. Caveats preserved

The discipline-of-caveats around the empirical evidence is unchanged:

- single seed (42), single model (Qwen2.5-7B) for the four next-stage families;
- in-repository TurboQuant comparator, not the official Google or Blackwell-
  cuTile kernel;
- deterministic five-task LongBench subset, not full 21-task LongBench;
- `production_kernel = false` on every \SQ{} and \TQ{} latency row;
- microbench-vs-hooked-replay distinction preserved verbatim;
- Mistral perplexity supporting-note observation registered as
  supporting-note-only;
- depth-wise compounding picture explicitly labelled "plausible explanatory
  account, not directly measured";
- LongBench macro $>$ FP16 explicitly labelled "directional signal, not a
  universal LongBench claim";
- compression-as-regularization labelled speculative;
- the new mechanism subsections each carry their own caveat sentence on
  multi-seed / multi-architecture / full-depth coverage being open work.

## 7. Compile status

After bibtex + 2 pdflatex passes:

- 40 pages (was 37 before round 9).
- Zero overfull boxes.
- Zero undefined references.
- Zero undefined citations (the new `holtzman2019curious` and
  `olsson2022context` resolve cleanly).
- No TODO / TBD / FIXME tokens in the source.

Output: `paper_output_consolidated/spectralquant_unrestricted_paper.pdf`.

## 8. Verdict

The manuscript now reads like an academic paper introducing SpectralQuant:

- Title is the algorithm only.
- Date suppressed.
- Abstract presents the algorithm and the two complementary bodies of
  evidence without rename history or "consolidated technical report" framing.
- Introduction's contributions are the algorithm, the two-part empirical
  evaluation, and the discipline of caveats — not "repository traceability".
- Every main result family has a mechanism-level interpretation subsection at
  roughly the same depth as the original perplexity mechanism, and each
  subsection labels speculation explicitly.
- Reproducibility, Discussion, Limitations are present as normal sections
  rather than as administrative sections.

This addresses the user feedback in full.
