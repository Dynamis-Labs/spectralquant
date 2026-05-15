# SpectralQuant v2 — Manuscript Audit Trail

This directory captures the four audit rounds run against
`paper_output_v2/spectralquant_v2_full_story.md` on 2026-05-01 at
commit `96e229c` (head of main).

| Round | File | Question | Outcome |
|---:|---|---|---|
| 1 | `round1_claim_to_artifact_audit.md` | Does every empirical claim have a path + metric + schema, all resolving on disk? | PASS — 39/39 paths exist; every cited metric matches the JSON. |
| 2 | `round2_citation_audit.md` | Does every background / related-work claim resolve to a real bibliography entry? Any invented references? | PASS — 16/16 keys defined; two new entries (`cover2006elements`, `ainslie2023gqa`) added for water-filling and GQA. |
| 3 | `round3_narrative_coherence_audit.md` | Do the sections flow? Are caveats carried? Any internal contradictions? | PASS — flow OK; 5 mandatory caveats carried in every section that touches them; no contradictions found on the load-bearing numbers. |
| 4 | `round4_style_human_editorial_audit.md` | Does the prose read as precise human writing rather than generic AI cadence? | PASS — pattern scanner finds 0 instances of common AI tics; sentence opening variety normal; no inflated claims; no boilerplate summaries. |

Re-run order if the manuscript changes:

1. Re-run Round 1's path-existence + metric-extraction script.
2. Re-run Round 2's citation regex against the bib.
3. Re-read Round 3's caveat-propagation matrix.
4. Re-run Round 4's pattern scanner (Appendix in the file).

Each audit document is self-contained and can be re-executed in
isolation.
