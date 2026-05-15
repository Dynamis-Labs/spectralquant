# Round 7 — Repository Rename Audit

**Date.** 2026-05-01
**Trigger.** GitHub repository renamed from `niashwin/spectralquant-v2` to
`niashwin/spectralquant-full`. The renamed repository is now the canonical
private full SpectralQuant repository.
**Scope.** Sweep the repository for occurrences of the old name and decide,
per occurrence, whether to update the canonical reference or preserve the
historical token.

## 1. Decision rule

Replace `niashwin/spectralquant-v2` (and the URL form
`https://github.com/niashwin/spectralquant-v2`) with the canonical
`niashwin/spectralquant-full` everywhere it functions as a *public-facing
canonical pointer to the consolidation repository*. Preserve it verbatim
where it functions as one of the following stable identifiers, all of
which are documented in §0 of `docs/claims_discipline.md` and §8 of
`docs/consolidated_spectralquant_inventory.md`:

1. **Frozen JSON `repo` fields** in already-shipped artifact files under
   `results/v3/modal/`. The artifact JSON is part of the audit trail and
   must remain bit-for-bit reproducible.
2. **JSON method keys** (`spectralquant_v2`) inside JSON file names and
   schema files. These are stable downstream tooling identifiers.
3. **Modal infrastructure names** — the volume `spectralquant-v2-results`
   and the app names `spectralquant-v2-eval` and `spectralquant-v2`
   (in `scripts/launch_modal_*.py`). Renaming these would orphan all
   existing artifacts on the Modal side.
4. **Historical filesystem paths** under `paper_output_v2/`,
   `docs/spectralquant_v2_technical_spec.md`, and
   `experiments/sqv2_replay.py` — these are dated work products preserved
   as filesystem provenance.
5. **Historical narrative documents** that describe the development
   cycle as it happened. Rewriting them retroactively would corrupt the
   audit trail. Examples: `docs/execution_audit_and_modal_runbook.md`,
   `docs/evidence_family_validation_2026-04-30.md`,
   `docs/evidence_catalog.{md,json}`,
   `docs/reviews/round1_*.md`, `docs/reviews/round5_*.md`.
6. **Historical paper drafts** in `paper_output_v2/` — pre-consolidation
   drafts kept for audit traceability per the inventory.
7. **Test fixture names** in `tests/test_safe_local_cleanup.py`
   (`spectralquant-v2-active`) — these are local-path strings that the
   test harness fabricates inside a tmp dir and never reaches GitHub.
8. **Evidence-catalog identifiers** (`V1-RESULT-*`, `V1-IMPL-*`,
   `V1-GAP-*`, `V2-SPEC-*`, `RUN-*`) — stable claim IDs preserved
   verbatim per the consolidated naming discipline.

## 2. Files updated to the canonical name

| File | Change |
|---|---|
| `README.md` | Added "About this repository" section announcing the canonical name `niashwin/spectralquant-full`, the rename from `niashwin/spectralquant-v2`, and the list of historical labels intentionally retained. |
| `paper_output_consolidated/spectralquant_unrestricted_paper.tex` | Updated `\repofile` and `\rfile` URL bases to `niashwin/spectralquant-full`. Updated date line, abstract, dev-history section, reproducibility section, and conflict-of-interest section to reference the canonical name and to record the rename. Modal-volume sentence in §Repro and §three-way table caption now annotate the volume name as a retained provenance identifier. |
| `paper_output_consolidated/spectralquant_unrestricted_paper.pdf` | Recompiled (28 pages, 668 KB). Zero undefined references, zero undefined citations, zero overfull boxes. |
| `docs/consolidated_spectralquant_inventory.md` | Repository row in §2 now lists `niashwin/spectralquant-full` as canonical, with a dependent clause documenting the rename and the retained historical labels. §5.2 three-way row now annotates the Modal volume name as a retained identifier. §8 provenance commitments now name `niashwin/spectralquant-full` and explicitly enumerate the retained historical labels. |
| `docs/claims_discipline.md` | §0 item 4 rewritten: historical filesystem and infra labels (including the historical repo name `niashwin/spectralquant-v2`, the Modal volume `spectralquant-v2-results`, `paper_output_v2/`, `docs/spectralquant_v2_technical_spec.md`, `experiments/sqv2_replay.py`) are preserved as provenance. Canonical public-facing repository is `niashwin/spectralquant-full`. |
| `docs/result_schema.md` | `repo` field documentation now lists `niashwin/spectralquant-full` as the current value and notes the pre-rename value is also valid for traceability. |
| `docs/full_matrix_evidence_summary.md` | Repo row in §2 audit table now reads `niashwin/spectralquant-full`, with a parenthetical note that the value stored in the JSON `repo` fields on disk remains the pre-rename string. |
| `experiments/eval_common.py` | `REPO_SLUG` constant (used to stamp every new artifact's `repo` field) bumped to `niashwin/spectralquant-full`. |
| `experiments/run_three_way.py` | Same `REPO_SLUG` bump. |
| `scripts/merge_longbench_partials.py` | `repo` literal bumped to `niashwin/spectralquant-full`. |

## 3. References intentionally preserved (sample)

| File | Why preserved |
|---|---|
| `results/v3/modal/*.json` (5 files) | Frozen artifacts; their `repo` and method-key fields are part of the audit anchor. |
| `results/v3/modal/longbench_relaunch_2026-04-30/...` | Frozen relaunch artifacts. |
| `schemas/*.schema.json` (7 files) | Method-key enums and example values reference `spectralquant_v2` as a stable downstream key. |
| `scripts/audit_results.py` | Path constants point at frozen JSON files whose names embed `spectralquant_v2`. |
| `scripts/launch_modal_eval.py`, `scripts/launch_modal_three_way.py` | Modal volume/app names; renaming would orphan existing artifacts. |
| `paper_output_v2/*.tex`, `paper_output_v2/*.md` | Pre-consolidation paper drafts; preserved as provenance per inventory §3. |
| `docs/spectralquant_v2_technical_spec.md` | Historical spec; filename retains the `v2` tag per inventory §6. |
| `docs/evidence_catalog.{md,json}` | Stable claim catalog; ID space and historical text preserved. |
| `docs/evidence_family_validation_2026-04-30.md` | Dated historical validation summary. |
| `docs/execution_audit_and_modal_runbook.md` | Historical execution log; rewriting would corrupt audit trail. |
| `docs/modal_safety_protocol.md` | Historical safety protocol; references the volume name. |
| `docs/reviews/round1_*.md`, `docs/reviews/round5_*.md` | Prior audits; historical record. |
| `tests/test_safe_local_cleanup.py` | Local fixture path strings under tmp; not a GitHub reference. |

## 4. Compile / lint / test checks

- LaTeX recompile of the canonical paper succeeded after three pdflatex
  passes plus bibtex. **0 undefined references**, **0 undefined
  citations**, **0 overfull boxes**. The 62 underfull boxes are cosmetic
  spacing in itemized lists and are unchanged from the pre-rename PDF.
- Final PDF: `paper_output_consolidated/spectralquant_unrestricted_paper.pdf`,
  28 pages, 668 KB.

## 5. Survivors after this audit

After this audit, every remaining occurrence of `spectralquant-v2` /
`niashwin/spectralquant-v2` / `spectralquant_v2` falls into one of the
preserved categories above. The grep audit produces 36 files in total;
all 36 are accounted for, and the per-file rationale is in §2 (updated)
or §3 (preserved). The canonical public-facing references in the README,
the consolidated paper TeX/PDF, the inventory, the claims-discipline
document, and the result-schema documentation now consistently use
`niashwin/spectralquant-full`.
