# Round 5 — Consolidated Naming Audit

Purpose: confirm that the consolidated unrestricted technical report (`paper_output_consolidated/spectralquant_unrestricted_paper.tex`) presents SpectralQuant as a single research program, that historical `v1`/`v2` labels appear only where audit-stability requires them, and that the consolidated framing does not weaken any caveat from prior audit rounds.

Audit date: 2026-05-01.
Scope: `paper_output_consolidated/spectralquant_unrestricted_paper.tex`, `paper_output_consolidated/spectralquant_unrestricted_paper.pdf`, `docs/consolidated_spectralquant_inventory.md`, `docs/claims_discipline.md` consolidated section, `paper_output_consolidated/figures/` captions in source.
Method: a grep over `\bv1\b|\bv2\b|SpectralQuant v1|SpectralQuant v2` in the consolidated `.tex`, with each hit categorized as either (i) public-narrative, (ii) provenance/repository-label, or (iii) source comment / macro definition. Public-narrative hits are forbidden.

## 1. Inventory of v1 / v2 occurrences in the consolidated manuscript

| Line | Surrounding text (excerpt) | Category | Verdict |
| --- | --- | --- | --- |
| 9–11 | Source-file header comment block explaining consolidation policy | (iii) source comment | OK |
| 77–82 | `\SQv` / `\SQone` macro definition comment | (iii) source comment | OK |
| 173–181 | Abstract: "*the **initial evidence layer**, released in `Dynamis-Labs/spectralquant` and labelled `v1` in the consolidation repository's git history*…" / "*the **expanded evidence layer**, internally tagged `v2` during development…*" | (ii) provenance/repository-label | OK — the body of the abstract uses neutral language ("initial evidence layer", "expanded evidence layer") and only mentions `v1`/`v2` once each, attributed explicitly to git-history labels. |
| 1190 | Three-way table caption: "*the configuration used in the initial evidence layer (\citep{kvtc2025}) and tagged `v1` in the consolidation repository's commit history*" | (ii) repository-label | OK |
| 1406 | Development-history section §11.1: "*the consolidation repository's commit history tags it as 'v1'*" | (ii) repository-label, by design | OK |
| 1416 | Development-history section §11.2: "*labelled 'v2' internally*" | (ii) repository-label, by design | OK |
| 1435–1462 | Section heading and body of §11.3 "When v1 / v2 labels appear": explicitly enumerates the five places where v1/v2 may appear (JSON method key, source paths, repo URL, evidence-catalog IDs, dev-history section) | (ii) provenance, by design | OK |
| 1477–1487 | Reproducibility section: discloses that the `paper_output_v2/` directory contains pre-consolidation drafts, and that `docs/spectralquant_v2_technical_spec.md` retains its historical filename | (ii) provenance | OK |
| 1650 | Traceability table row: "*Uniform/water-filled ratios within 0.013×*" — note this row was rewritten away from "v1/v2 ratios" for the consolidation | (ii) corrected | OK |
| every JSON path containing `spectralquant_v2` (figs, captions, repro section, traceability table) | Literal path in repository / JSON method key | (ii) repository-label | OK — the JSON method key is preserved verbatim per §11.3; renaming the key would break downstream tooling without a traceable benefit. |
| `niashwin/spectralquant-v2`, `paper_output_v2/`, `paper_output_consolidated/`, `sqv2_replay.py`, `spectralquant_v2_technical_spec.md` | Repository, directory, file paths | (ii) repository-label | OK — all are stable filesystem paths, unchanged by the consolidation. |
| every `V1-RESULT-*`, `V1-IMPL-*`, `V1-GAP-*`, `V2-SPEC-*` identifier | Stable evidence-catalog IDs | (ii) audit-stability | OK — renaming would break audit cross-references and existing citations in `docs/evidence_catalog.{md,json}`. |

No public-narrative occurrence remains.

## 2. Title, abstract, contributions

* Title: `SpectralQuant: Calibrated Eigenbasis Rotation and Water-Filled Bit Allocation for KV-Cache Compression`. The title does not contain `v1` or `v2`.
* Abstract: introduces a single program SpectralQuant. The two evidence layers are named ("initial evidence layer", "expanded evidence layer"); their git-history tags `v1` / `v2` appear once each, in apposition to the descriptive name. No empirical claim is qualified by `SpectralQuant v2`.
* Contributions list (§1): both contributions attribute results to SpectralQuant, with the comparison framed as "uniform-allocation special case" vs "water-filled allocation". The phrase "SpectralQuant v2" does not appear in the contributions list. The phrase "SpectralQuant v1" does not appear in the contributions list.

## 3. Caveat preservation

Each caveat from prior audit rounds was checked for round-trip preservation:

| Caveat (prior round) | Present in consolidated paper | Location |
| --- | --- | --- |
| Single seed (42), single bit budget (b=3) for the four next-stage families | YES | §1 abstract, §1 last paragraph, §8.1 |
| In-repository TurboQuant comparator (not Google reference, not Blackwell-cuTile) | YES | §1 third contribution, §5.3, §8.3 |
| Deterministic 5-task LongBench subset, not full 21-task LongBench | YES | §1 abstract, §6.4, §7.5, §8.4 |
| `production_kernel = false` on every non-FP16 latency row | YES | §1 third contribution, §6.5, §8.2 |
| FP16-vs-replay gap dominated by Python hooks, not kernel | YES | §6.5, §8.2 |
| Original-paper 5.95× headline preserved with V1-GAP-014 caveat | YES | §6.7 |
| Pre-existing test failures unrelated to next-stage families | YES | §8.8 |
| Two engines in namespace (V1-GAP-010) | YES | §8.7 |

No caveat was softened or omitted by the consolidation.

## 4. Figure / table captions

| Figure / table | Caption mentions | Verdict |
| --- | --- | --- |
| Fig. 1 (pipeline) | "the SpectralQuant calibration pipeline" | OK |
| Fig. 2 (perplexity) | FP16, SpectralQuant, in-repository TQ; clarifies SpectralQuant is the water-filled configuration | OK |
| Fig. 3 (generation) | SpectralQuant, in-repository TQ | OK |
| Fig. 4 (longbench) | SpectralQuant, in-repository TQ | OK |
| Fig. 5 (latency) | SpectralQuant, TQ | OK |
| Fig. 6 (attention cosine) | "SpectralQuant strictly beats … the water-filled allocation matches the uniform-allocation special case" | OK — this caption is the only one where the within-program comparison is on display, and it uses the neutral "water-filled / uniform" vocabulary. |
| Tab. 1 (headline) | FP16 / SpectralQuant / TurboQuant local | OK |
| Tab. 2 (longbench) | FP16 / SpectralQuant / TurboQuant local | OK |
| Tab. 3 (latency) | SpectralQuant rows; no v1/v2 | OK |
| Tab. 4 (three-way) | columns "Unif. cos μ" and "SpectralQuant cos μ"; caption explains the historical mapping | OK |

## 5. Bibliography and citations

* `kvtc2025` is the original SpectralQuant manuscript; cited as the source of the initial evidence layer. No invented citations introduced by the consolidation.
* All bibtex keys present in the consolidated `.tex` resolve in `paper_output/spectralquant_refs.bib` (verified by clean `bibtex` run with zero warnings).
* The consolidation introduces no new citations; the bibliography is reused unchanged.

## 6. Style / human-prose check

The consolidation rewrote roughly 30 paragraphs across the abstract, contributions, methodology, results-interpretation, limitations, and reproducibility sections. Each rewrite was checked for:

* Avoidance of generic AI-writing tells: hedging connectives ("notably", "importantly", "it is worth noting"), bullet-point fanfare, decorative parallelism. No new instances introduced.
* Human technical voice: short declarative sentences, active verbs, named subjects. The rewritten paragraphs continue the prior rounds' style.
* Domain-specific grounding: every comparative claim ("strictly improves", "matches", "collapses to") names the operating point, the bit width, and the source artifact.

## 7. Verdict

The consolidated unrestricted technical report presents SpectralQuant as a single research program. Public-narrative `v1`/`v2` references have been eliminated; the labels survive only in (i) source-file comments, (ii) the development-history section by explicit design, (iii) repository / directory / filename paths, (iv) the JSON method-key `spectralquant_v2` preserved for downstream tooling stability, and (v) stable evidence-catalog identifiers. Every empirical caveat from prior audit rounds is preserved.

The consolidation does not unblock any new claim, does not weaken any caveat, and does not modify any code, JSON artifact, schema, or test. It is a naming-discipline and narrative-framing pass on top of the existing evidence base.
