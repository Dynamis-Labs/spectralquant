# SpectralQuant v2 — Execution Audit and Modal Runbook

**Purpose.** This document is a single, repo-tracked execution and audit log
for SpectralQuant v2. It records exactly what has been done in the
`niashwin/spectralquant-v2` private repo, what remains to be done, what tests
have been run, what tests still need to be run, what experiments must run on
Modal, and how the technical report will be written from evidence rather than
from intuition. It is intentionally long and practical: an unattended agent or
a new collaborator should be able to read this file once and continue the work
without further context.

This file is the operational companion to:

- `docs/spectralquant_v2_technical_spec.md` — authoritative algorithm and plan.
- `docs/evidence_catalog.md` and `docs/evidence_catalog.json` — v1 evidence
  audit and stable IDs (V1-PAPER-*, V1-IMPL-*, V1-RESULT-*, V1-GAP-*, etc.).
- `docs/claims_discipline.md` — wording rules for the report.
- `docs/result_schema.md` — JSON-schema spec for result artifacts.
- `docs/modal_safety_protocol.md` — credential handling, license verification,
  GPU choice, resumable result paths, atomic JSON writes, frequent artifact
  sync, timeouts, retries, and preflight checks. **Read this before any
  Modal run that loads weights or spends GPU credit.**

If those documents disagree with this one, the spec, claims discipline, and
schemas take precedence; this audit must be brought into line with them.

Today's date for this audit: **2026-04-30**. Anchor any "since" or "before"
dates to that.

**Audit update (2026-04-30, anchored at commit `abcb09197998cc027df688abceae5fb81cfcd31d`).**
The four-row Phase-6 headline matrix in §8 has been run on Modal:
`RUN-THREEWAY-MISTRAL-{2,3,5}BIT` and `RUN-THREEWAY-QWEN-3BIT`. The
narrative summary, headline numbers, and interpretation guardrails are
in `docs/full_matrix_evidence_summary.md`. The four artifacts are
catalogued under `RUN-THREEWAY-*` in `docs/evidence_catalog.{md,json}`
and pass `experiments/run_three_way.py::_validate_payload` against
`schemas/three_way_result.schema.json`.

---

## 1. Current repo status (as of this commit)

### 1.1 Repository

- Remote: `https://github.com/niashwin/spectralquant-v2` (private; pushed via
  `git-agent-proxy.perplexity.ai` for agent runs).
- Branch: `main` only. No long-lived feature branches.
- License: `LICENSE` is the v1 MIT license, inherited from the
  `Dynamis-Labs/spectralquant` baseline.
- Original `Dynamis-Labs/spectralquant` repo is untouched per the policy in
  `docs/spectralquant_v2_technical_spec.md` §2.

### 1.2 Recent commits (oldest first within v2 work)

```
8d5a431  SpectralQuant: 3% Is All You Need                         (v1 baseline import)
1254311  Add shaped cache sweep experiment, results, and figures   (v1)
d5dd891  Add multiregime, optimal allocation, and push-to-0.95 experiments  (v1)
1a9df75  Add SpectralQuant v2 technical specification              (v2 starts here)
21a90f2  Phase 1 + minimal Phase 2 scaffolding from v2 spec
564ef9f  Add waterfill and accounting foundation modules with unit tests
dad8ec5  Wire water-filling into NonUniformQuantizer behind use_water_fill flag
21c1c9b  Add pytest pythonpath=src so tests run from clean checkout
```

Commits `1a9df75` onwards constitute v2 work. Commits before `1a9df75` are the
v1 baseline import and must be treated as historical evidence (the
`V1-*` entries in `docs/evidence_catalog.json`).

### 1.3 Documentation in the repo

| Path | Role |
|---|---|
| `docs/spectralquant_v2_technical_spec.md` | Authoritative algorithm + execution plan (V2-SPEC-001). |
| `docs/evidence_catalog.md` / `.json` | V1 evidence catalog with stable IDs and gaps. |
| `docs/claims_discipline.md` | What the v2 report is and is not allowed to say. |
| `docs/result_schema.md` | Human-readable spec for the JSON schemas under `schemas/`. |
| `docs/execution_audit_and_modal_runbook.md` | **This file** — execution and audit log. |
| `docs/modal_safety_protocol.md` | Credential handling, license checks, preflight, atomic writes, retries — read before any Modal run. |
| `scripts/preflight_modal.py` | Read-only preflight checks (Python, git, env vars, Modal CLI, disk). Never echoes secret values. |
| `scripts/audit_results.py` | Read-only audit of expected vs. present result artifacts. |
| `scripts/safe_local_cleanup.py` | Dry-run-by-default cache cleanup with explicit per-category flags; refuses to delete the active repo. |
| `.env.example` | Names-only template for local environment variables; real `.env` is gitignored. |
| `README.md` | Inherited from v1; not yet rewritten for v2 messaging. |

### 1.4 JSON schemas

| Path | Validates |
|---|---|
| `schemas/evidence_catalog.schema.json` | `docs/evidence_catalog.json`. |
| `schemas/three_way_result.schema.json` | (future) `results/three_way/*.json`. |
| `schemas/accounting.schema.json` | (future) `results/accounting_audit/*.json`. |

`tests/test_result_schema.py` currently validates the evidence catalog JSON
and exercises the three-way and accounting schemas with hand-crafted minimal
documents. The schemas are wired up; what is missing is the actual benchmark
JSONs they will eventually validate.

### 1.5 Source modules

```
src/spectralquant/__init__.py
src/spectralquant/accounting.py            (v2; new in 564ef9f)
src/spectralquant/calibration.py           (v1, untouched in v2)
src/spectralquant/engine.py                (v1; subclasses turboquant_cutile.TurboQuantEngine)
src/spectralquant/metrics.py               (v1)
src/spectralquant/nonuniform_quantization.py  (v2-extended; use_water_fill flag added in dad8ec5)
src/spectralquant/selective_qjl.py         (v1)
src/spectralquant/spectral_rotation.py     (v1)
src/spectralquant/spectralquant.py         (v1; legacy engine + TurboQuantBaseline)
src/spectralquant/utils.py                 (v1)
src/spectralquant/waterfill.py             (v2; new in 564ef9f)
```

`src/spectralquant/__init__.py` exports both engines under unambiguous
names (decision recorded for **V1-GAP-010** below):

- **`SpectralQuantEngine`** — re-exported from `spectralquant.spectralquant`.
  Pure-Python pipeline that drives `NonUniformQuantizer` directly. **This is
  the canonical engine for v2 benchmarks**: it carries water-filling, runs
  locally without Modal, and is what the v2 unit-test suite covers.
- **`KernelSpectralQuantEngine`** — re-exported from `spectralquant.engine`.
  cuTile-accelerated subclass of `turboquant_cutile.TurboQuantEngine`,
  intended for kernel/latency measurements on Modal where
  `turboquant_cutile` is installed. The import is wrapped in a guarded
  `try/except` so a clean checkout without the cuTile baseline still
  imports the package; constructing the engine outside Modal raises a
  clear `RuntimeError` describing the original import error.
- `_LegacySpectralQuantEngine` is a transitional alias that points at the
  canonical `SpectralQuantEngine`. It will be removed once internal
  callers migrate.

This decision resolves **V1-GAP-010** in v2: benchmark scripts can write
`SpectralQuantEngine` (pure-Python, water-fill-aware) for quality
benchmarks and `KernelSpectralQuantEngine` for kernel-path latency
benchmarks, and the result JSON's `software.engine` field can record the
canonical class name unambiguously.

### 1.6 Tests

```
tests/conftest.py                  shared fixtures
tests/test_accounting.py           v2 — accounting math
tests/test_calibration.py          v1 — heavy; needs torch
tests/test_end_to_end.py           v1 — heavy
tests/test_quantization.py         v1 — quantizer
tests/test_result_schema.py        v2 — schema wiring
tests/test_spectral_rotation.py    v1 — rotation
tests/test_v2_quantization.py      v2 — water-filling integration
tests/test_waterfill.py            v2 — pure-numpy water-filling
```

`pyproject.toml` has `pythonpath = ["src"]` (commit `21c1c9b`) so
`pytest` runs from a clean checkout without `pip install -e .`.

---

## 2. What has been implemented

The list below is concrete and traceable to commits.

### 2.1 Evidence catalog and claims discipline (commits up to `21a90f2`)

- `docs/evidence_catalog.md` and `docs/evidence_catalog.json` enumerate every
  v1 paper artifact, README claim, implementation module, experiment script,
  result JSON, figure, config, build file, and test, with stable IDs of the
  form `V1-PAPER-001`, `V1-IMPL-001`, `V1-RESULT-001`, `V1-FIG-001`,
  `V1-GAP-001`, etc.
- All 14 known v1 gaps (V1-GAP-001 through V1-GAP-014) are documented with
  the specific claim each one blocks. These are listed in §4 below.
- `docs/claims_discipline.md` translates the spec's §6 / §16 / §18 rules into
  a concrete list of safe claims, blocked claims, and required evidence IDs.
  No v2 figure caption or paper sentence can quietly upgrade a measured local
  result into a universal claim without the catalog and discipline files
  approving the wording.
- `docs/result_schema.md` documents the JSON schemas in `schemas/`.

### 2.2 Result schemas (commit `21a90f2`)

- `schemas/evidence_catalog.schema.json` — JSON Schema 2020-12 validation for
  the catalog. Required fields include `schema_version`, `repo`, `entries`,
  `gaps`. Each entry has `id`, `kind`, `path`, `supports`, `caveats`.
- `schemas/three_way_result.schema.json` — placeholder schema for
  `results/three_way/*.json`. Required top-level fields: `run_id`,
  `timestamp`, `repo`, `commit`, `command`, `model`, `hardware`, `software`,
  `data`, `calibration`, `methods`, `evidence_ids`.
- `schemas/accounting.schema.json` — placeholder schema for accounting
  artifacts mirroring `CompressionAccounting` in `src/spectralquant/accounting.py`.

`tests/test_result_schema.py` validates `docs/evidence_catalog.json` and
exercises the other two schemas against minimal hand-crafted documents.

### 2.3 Water-filling utilities (commit `564ef9f`)

`src/spectralquant/waterfill.py` (230 lines) is a pure-numpy module independent
of torch and the rest of the engine. Public API:

```python
from spectralquant.waterfill import (
    allocate_waterfill_bits,
    waterfill_metadata,
    FORMULA_VERSION,            # "waterfill-v1"
)
```

`allocate_waterfill_bits(eigenvalues, total_bits, min_bits=0, max_bits=None)`
implements the spec §9.3 / §9.4 greedy rule
`i* = argmax_i lambda_i / 4 ** b_i`, with deterministic lowest-index
tie-breaking. Inputs may be numpy arrays, Python sequences, or torch tensors.
Inputs are not mutated; outputs are int64 numpy arrays summing to `total_bits`.

`waterfill_metadata(eigenvalues, bits)` returns a JSON-safe dict with the
allocation, formula version, and the per-dimension lambda / bits pair for use
by `WaterfillAllocation` (see §2.4) and the eventual three-way JSON.

### 2.4 Compression accounting (commit `564ef9f`)

`src/spectralquant/accounting.py` (389 lines) computes per-method bit layouts
and ratios from explicit components rather than from headline targets.
Mirroring `schemas/accounting.schema.json`:

```python
@dataclass
class CompressionAccounting:
    method: str
    avg_bits_arg: int
    head_dim: int
    d_eff: Optional[int]
    k_mse_bits: float
    k_qjl_bits: float
    k_norm_bits: float
    v_mse_bits: float
    v_norm_bits: float
    total_k_bits: float
    total_v_bits: float
    average_slot_bits: float
    fp16_slot_bits: float
    compression_ratio: float
    formula_version: str
    waterfill_allocation: Optional[List[int]]
    notes: Optional[str]
```

Two "formula versions" coexist:

- `turboquant-spec-v1` — TurboQuant slot accounting from spec §10.
  Test-verified to recover ≈ 5.02x at b=3 and ≈ 3.08x at b=5.
- `spectralquant-spec-v1` and `spectralquant-flex-v1` — the two SpectralQuant
  variants. The simple formula in spec §10 does **not** yield 5.95x at
  `b=3, d_eff=3`. `check_headline_ratio` exposes that discrepancy
  (V1-GAP-014) rather than papering over it.

This separation is what eventually lets the Phase 6 three-way runs report
ratios derived from real bit layouts, not from the v1 paper's appendix
shortcut.

### 2.5 v2 quantization integration (commit `dad8ec5`)

`src/spectralquant/nonuniform_quantization.py` now exposes a `use_water_fill`
flag on `NonUniformQuantizer`:

- `use_water_fill=False` (default) reproduces v1 uniform semantic MSE bit
  allocation byte-for-byte. v1 calibration / d_eff / selective QJL semantics
  are unchanged.
- `use_water_fill=True` calls `allocate_waterfill_bits` to spread the same
  semantic bit budget over the semantic eigenvalues, producing per-dimension
  Lloyd-Max codebooks parameterized by `mean=0, variance=lambda_i`.

The allocation, eigenvalues, and formula-version metadata are stored on the
quantizer as a `WaterfillAllocation` dataclass and exposed via the
`waterfill_allocation` property so they can be persisted into the future
three-way JSON. The unit test
`tests/test_v2_quantization.py::test_uniform_when_water_fill_disabled`
fails loudly if the v1 codepath is changed in passing.

### 2.6 Test harness fix (commit `21c1c9b`)

`pyproject.toml` adds:

```
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
pythonpath = ["src"]
```

This makes `pytest tests/test_waterfill.py tests/test_accounting.py
tests/test_v2_quantization.py tests/test_result_schema.py -q` succeed from a
clean checkout without `pip install -e .` or manual `PYTHONPATH=src`.

---

## 3. What has been run

### 3.1 Tests run (and current expected outcome)

The focused, dependency-free test set is the v2 contract surface:

```
pytest tests/test_waterfill.py \
       tests/test_accounting.py \
       tests/test_v2_quantization.py \
       tests/test_result_schema.py \
       -q
```

Expected outcome on a clean checkout at the audit commit: **85 passed in
~3 s** (Python 3.12 + numpy + jsonschema; no torch needed).

Breakdown:

- `tests/test_waterfill.py` — 34 tests covering sum-to-budget, equal /
  concentrated spectra, min/max bit constraints, negative-eigenvalue refusal,
  zero-eigenvalue stability, deterministic tie-breaking, numpy / torch input
  parity, no-mutation guarantees, metadata round-trip.
- `tests/test_accounting.py` — 23 tests covering TurboQuant 3-bit ≈ 5.02x and
  5-bit ≈ 3.08x, SpectralQuant flexible / spec accounting, headline-ratio
  guard, JSON schema round-trip, formula-version handling.
- `tests/test_v2_quantization.py` — 17 tests covering uniform vs water-fill
  parity, per-dim codebook count == d_eff, codebook sigma == sqrt(lambda_i),
  shape preservation, edge cases d_eff=2 and d_eff=D-2, no-NaN allocation,
  metadata persistence.
- `tests/test_result_schema.py` — 11 tests validating `evidence_catalog.json`
  and exercising the other two schemas with hand-crafted minimal payloads.

### 3.2 Tests that have **not** been run in this environment

These either need torch / a model checkpoint / heavy I/O and are not part of
the focused contract test set:

- `tests/test_calibration.py` — calibration plumbing; needs torch.
- `tests/test_end_to_end.py` — end-to-end engine; needs torch and Modal-side
  `turboquant_cutile`.
- `tests/test_quantization.py` — pre-existing v1 Lloyd-Max tests; needs
  torch.
- `tests/test_spectral_rotation.py` — v1 rotation tests; needs torch.

Status of these on Modal is not currently recorded in this repo. They should
be run as part of the Phase 0 reproduction (§5.3) and their outcome appended
here.

### 3.3 Experiments / model runs

#### 3.3.1 Completed paper-valid Modal runs (full-path)

The headline four-row matrix in §8 has been run on Modal at commit
`abcb09197998cc027df688abceae5fb81cfcd31d`. The four artifacts are
catalogued under `RUN-THREEWAY-*` in `docs/evidence_catalog.{md,json}`,
summarized in `docs/full_matrix_evidence_summary.md`, and live on the
Modal volume `spectralquant-v2-results` under `/results/three_way/`.

| Evidence ID | Model | Bits | Modal volume path | `paper_valid` |
|---|---|---:|---|---|
| RUN-THREEWAY-MISTRAL-5BIT | mistralai/Mistral-7B-v0.3 | 5 | `/results/three_way/Mistral-7B-v0.3_b5_calib32_eval8_seed42.json` | true |
| RUN-THREEWAY-MISTRAL-3BIT | mistralai/Mistral-7B-v0.3 | 3 | `/results/three_way/Mistral-7B-v0.3_b3_calib32_eval8_seed42.json` | true |
| RUN-THREEWAY-MISTRAL-2BIT | mistralai/Mistral-7B-v0.3 | 2 | `/results/three_way/Mistral-7B-v0.3_b2_calib32_eval8_seed42.json` | true |
| RUN-THREEWAY-QWEN-3BIT    | Qwen/Qwen2.5-7B           | 3 | `/results/three_way/Qwen2.5-7B_b3_calib32_eval8_seed42.json`     | true |

All four artifacts validate against
`schemas/three_way_result.schema.json` (with
`schemas/accounting.schema.json` cross-resolved) and pass the harness
guardrails: `mode = "full"`, `paper_valid = true`,
`data.calibration_corpus = data.eval_corpus = "WikiText-103"`,
`data.disjoint_eval = true`, `data.n_calib = 32`, `data.n_eval = 8`,
`config.n_layers_sample = 8`, `config.seed = 42`,
`config.max_calib_tokens = 384`, GPU `NVIDIA H200`.

#### 3.3.2 Modal runs and v1 experiments that have **not** been run

- `experiments/run_waterfill_ablation.py` — not yet implemented; no JSONs
  under `results/waterfill_ablation/`.
- `experiments/run_deff_stats.py` — not yet implemented; no JSONs under
  `results/deff_stats/`.
- `experiments/run_compression_accounting_audit.py` — not yet
  implemented; no JSONs under `results/accounting_audit/` (V1-GAP-014
  remains in force).
- `experiments/plot_three_way.py` — not yet implemented; no figures
  under `paper_output/v2/figures/`.
- Multi-seed runs (≥ 5, ideally 10) on Mistral b=3 — single seed only at
  this commit (V1-GAP-001 still in force).
- Calibration stability v2 (three-draw protocol) — not yet run
  (V1-GAP-005 still in force).
- The v1 experiments under `experiments/` (e.g. `phase3_exp1_attention_quality.py`,
  `run_v3_perplexity_crossarch.py`, `run_calibration_stability.py`) have not
  been re-run; their existing JSONs under `results/` are v1 evidence with
  caveats per `docs/evidence_catalog.md`.

---

## 4. Known caveats and unresolved issues

These are the v1 gaps inherited into v2 plus the v2-specific issues that have
surfaced during scaffolding. None of them blocks the `pytest` baseline, but
each one blocks at least one paper-level claim.

### 4.1 v1 gaps still in force

These are kept verbatim from `docs/evidence_catalog.md` so the audit and the
catalog cannot drift:

| ID | Issue | Blocks |
|---|---|---|
| V1-GAP-001 | "10-seed" framing with only 5 seeds in `neurips_10seed.json`. | Wilcoxon p=0.031 significance claim. |
| V1-GAP-002 | "4 models" framing with 3 Qwen rows in `all_models.json`. | "4 models" headline. |
| V1-GAP-002b | `comparison_results.json` shows SQ losing TQ on every head (3 bits, Qwen 1.5B). | "SQ wins per-head" claim. |
| V1-GAP-003 | Latency conflict: `neurips_latency_crossover.json` vs `v3_latency.json`. | "SQ faster than TQ at 512 tokens" headline. |
| V1-GAP-004 | Two d_eff conventions (normalized ≈ 4 vs unnormalized ≈ 35). | "d_eff/head_dim ≈ 3–4%" universal claim. |
| V1-GAP-004b | Identical 13-digit PPL across fp16/TQ/SQ in `v3_perplexity_v2.json`. | "Compression-neutral PPL" claim. |
| V1-GAP-005 | `stability.json` self-declares as reconstructed from logs. | "CV=3.9%" stability headline. |
| V1-GAP-006 | `phase1_metadata.json` has `gate1_status=FAILED`. | Direct phase 1 d_eff citations. |
| V1-GAP-007 | "15-second calibration" claim vs 31 s on disk. | "15-second calibration" headline. |
| V1-GAP-008 | LongBench n=5/task. | "LongBench improvement" claim. |
| V1-GAP-009 | NIAH artifacts limited / partly broken. | "NIAH 10/10" claim. |
| V1-GAP-010 | Two `SpectralQuantEngine` classes coexist in the public namespace. | Reproducibility — different imports run different code paths. |
| V1-GAP-011 | Gemma row is a 403, not a measurement. | "Gemma 2-9B measured" claim. |
| V1-GAP-012 | Local TurboQuant baseline reproduction marked failed in at least one phase 0 artifact. | "Beats official TurboQuant" claim. |
| V1-GAP-013 | TQ baselines disagree across `comparison/full_ablation.json` and `comparison/ablation_v2.json`. | Re-using ablations as TQ baselines. |
| V1-GAP-014 | Compression-ratio formula in spec §10 does not yield 5.95x at b=3, d=3. | Any ratio not produced by `src/spectralquant/accounting.py`. |

### 4.2 v2-specific caveats surfaced during scaffolding

1. **Engine duplication resolved (V1-GAP-010).** `__init__.py` now
   exports the canonical pure-Python engine as `SpectralQuantEngine`
   and the cuTile-accelerated subclass as `KernelSpectralQuantEngine`,
   with `_LegacySpectralQuantEngine` aliased to the canonical class
   for backwards compatibility. Both engines accept `use_water_fill`,
   `wf_min_bits`, and `wf_max_bits` and surface the same
   `WaterfillAllocation` metadata for the result JSONs (spec §11.2).
   The cuTile import path is guarded so clean checkouts without
   `turboquant_cutile` still import the package; constructing the
   kernel engine off-Modal raises a clear `RuntimeError`.
2. **Modal-only baseline.** `engine.py` imports
   `turboquant_cutile` (host + codebook + constants) from the
   `baseline/turboquant_cutile/` directory or from a Modal-side install. On
   any environment that does not have the cutile package, `_StubEngine`
   raises at runtime. This means **no end-to-end benchmark can run locally**;
   Phase 6 is a Modal-only execution.
3. **Local TurboQuant baseline ≠ official Google code.** Per V1-GAP-012, the
   v2 report is only allowed to claim improvements over the *local* baseline
   until the official Google TurboQuant comparison is run. Every JSON written
   by Phase 6 must label its TurboQuant arm as "local" in `methods.turboquant.label`.
4. **v1 evidence discrepancies persist.** The 5.95x compression ratio in the
   pasted v1 report does not derive from the simple appendix formula at
   `b=3, d=3`. `accounting.check_headline_ratio` flags this; the fix lives
   in the eventual `experiments/run_compression_accounting_audit.py` and a
   reconciled paper appendix.
5. **`pyproject.toml` build backend.** `setuptools.backends.legacy:build` is
   non-standard for new repos. It currently works for `pip install -e .` on
   the Modal image, but should be revisited if the package needs to be wheel-built.
6. **Tests requiring torch / cutile are skipped here.** `test_calibration.py`,
   `test_end_to_end.py`, `test_quantization.py`,
   `test_spectral_rotation.py` need a torch install; `test_end_to_end.py` also
   needs the cutile baseline. None has been validated in this environment;
   their status will be appended after Phase 0 on Modal.

---

## 5. Detailed work remaining, by phase

The phases follow `docs/spectralquant_v2_technical_spec.md` §19. Phases 0–2
are mostly complete on the v2 side; the remaining work is concentrated in
phases 5–7.

### 5.1 Phase 0 — Repo and Modal environment

| Item | Status | Owner | Artifact |
|---|---|---|---|
| Private `niashwin/spectralquant-v2` repo exists, untouched baseline imported | Done | infra | first commit `8d5a431` |
| v2 spec committed | Done | author | `1a9df75` |
| `scripts/setup_b200.sh` describes Modal H100/H200/B200 image | Done | infra | `scripts/setup_b200.sh` |
| Modal image with `turboquant_cutile` available | **TODO** | infra | Modal image build log |
| Hugging Face access for `mistralai/Mistral-7B-v0.3` (gated) | **TODO** | infra | HF token, license accepted |

### 5.2 Phase 1 — Evidence catalog

| Item | Status | Owner | Artifact |
|---|---|---|---|
| Catalog v1 paper, README, modules, experiments, results, figures, configs, tests | Done | code-agent | `docs/evidence_catalog.md`, `.json` |
| Catalog all v1 gaps (V1-GAP-001..014) | Done | code-agent | catalog §"Discrepancies and gaps" |
| Schema validation of the catalog JSON | Done | code-agent | `tests/test_result_schema.py` |
| Re-extract any v1 number used in the v2 paper from JSON, not from PDF | **TODO before paper draft** | paper-agent | per-claim cite list |

### 5.3 Phase 2 — Unit-test scaffolding

| Item | Status | Owner | Artifact |
|---|---|---|---|
| `tests/test_waterfill.py` | Done | code-agent | 34 tests passing |
| `tests/test_accounting.py` | Done | code-agent | 23 tests passing |
| `tests/test_v2_quantization.py` | Done | code-agent | 17 tests passing |
| `tests/test_result_schema.py` | Done | code-agent | 11 tests passing |
| `tests/test_calibration_v2.py` | **TODO** | code-agent | spec §12.3 |
| `tests/test_engine_v2.py` | **TODO** | code-agent | spec §12.4 |
| `tests/test_turboquant_baseline.py` | **TODO** | code-agent | spec §12.5 |
| Re-run v1 tests on Modal and record outcomes here | **TODO** | code-agent | this doc §3.2 |

### 5.4 Phase 3 — Water-filling

| Item | Status |
|---|---|
| `src/spectralquant/waterfill.py` implemented | Done (commit `564ef9f`) |
| Unit tests for greedy allocator | Done (`test_waterfill.py`) |
| Hooked into `WaterfillAllocation` dataclass | Done |

### 5.5 Phase 4 — Per-dim codebooks

| Item | Status |
|---|---|
| Variable-bit semantic codebooks in `NonUniformQuantizer` | Done (commit `dad8ec5`) |
| v1 codepath preserved when `use_water_fill=False` | Done, asserted by `test_v2_quantization.py` |
| Codebook sigma == sqrt(lambda_i) | Done, asserted by tests |
| Bias of unused dimensions vs. d_eff = D-2 edge | Done (edge-case tests) |

### 5.6 Phase 5 — Engine integration (partially done)

| Item | Status | Notes |
|---|---|---|
| `use_water_fill` plumbed through `NonUniformQuantizer` | Done | quantizer-level (commit `dad8ec5`) |
| `EngineConfig` carries `use_water_fill` / `wf_min_bits` / `wf_max_bits` | Done | canonical pure-Python engine; default `False` keeps v1 behaviour |
| `KernelSpectralQuantEngine` (cuTile subclass) accepts the same flags | Done | metadata recorded; CUDA per-dim execution still TODO (kernel path uses regime-uniform codebook) |
| Allocation / accounting metadata included in compressed-key dict | Done | both engines now expose `use_water_fill` and `semantic_bits_per_dim` (canonical: `CompressedVector.semantic_bits_per_dim`; kernel: `compress_keys_pytorch`/`compress_values_pytorch` dicts) |
| Engine duplication resolved (V1-GAP-010 / §4.2) | Done | canonical = pure-Python `SpectralQuantEngine`; kernel = `KernelSpectralQuantEngine`; `_LegacySpectralQuantEngine` aliased |
| `tests/test_engine_v2.py` (eight tests in spec §12.4) | Done | 23 tests covering canonical-engine identity, EngineConfig validation, v1/v2 budget parity, shapes, finite logits, masking, normalization, JSON-safe metadata, and clean-checkout imports |

### 5.7 Phase 6 — Three-way benchmark on Modal

| Item | Status | Notes |
|---|---|---|
| `experiments/run_three_way.py` skeleton (CLI, dry-run, synthetic-smoke) | Done | argparse contract matches spec §13.5; supports `--dry-run`, `--synthetic-smoke`, `--skip-if-exists`/`--resume`, `--force`, atomic JSON write, schema validation pre-rename |
| `experiments/run_three_way.py` full HF-model path | Done | runs end-to-end on Modal; produced the four `RUN-THREEWAY-*` artifacts at commit `abcb09197998cc027df688abceae5fb81cfcd31d` |
| `experiments/run_waterfill_ablation.py` | **TODO** | per-spec §11.3 |
| `experiments/run_deff_stats.py` | **TODO** | needed for the d_eff distribution figure |
| `experiments/run_compression_accounting_audit.py` | **TODO** | reconciles V1-GAP-014 |
| `experiments/plot_three_way.py` | **TODO** | reads JSONs only; no models |
| Local synthetic smoke on Mistral 3-bit, n_calib=4, n_eval=2 | Done | `--synthetic-smoke` writes a schema-valid JSON without downloading any model |
| Modal smoke run on Mistral 3-bit | Done | preceded the full sweep; absorbed into the persistent HF cache under `/results/hf_cache/` |
| Mistral-7B-v0.3 at b=2,3,5 | Done | RUN-THREEWAY-MISTRAL-{2,3,5}BIT; sliced full-path (n_calib=32, n_eval=8, 8 layers, seed 42) |
| Qwen2.5-7B at b=3 | Done | RUN-THREEWAY-QWEN-3BIT; same sliced configuration |
| All result JSONs validate against `schemas/three_way_result.schema.json` | Done | all four full-path JSONs validate (cross-resolved with `accounting.schema.json`) via `experiments/run_three_way.py::_validate_payload` |
| Compression-accounting audit JSONs written | **TODO** | `results/accounting_audit/*.json` |
| Multi-seed (≥ 5) on Mistral b=3 | **TODO (Modal-only)** | required to unblock V1-GAP-001 |
| Per-head min-cosine distribution stored in result JSON | **TODO** | current schema records per-layer aggregates only |

### 5.8 Phase 7 — Report generation

| Item | Status | Notes |
|---|---|---|
| Tables generated from `results/three_way/*.json`, not typed | **TODO** | `experiments/plot_three_way.py` writes `paper_output/v2/tables/*.tex` |
| Figures generated from JSON | **TODO** | per spec §11.5 → `results/report_figures/` |
| `paper_output/v2/spectralquant_v2.tex` populated section-by-section | **TODO** | sections enumerated in §10 below |
| Eight-pass reflection protocol completed | **TODO** | spec §17, this doc §11 |
| `docs/reproduction.md` rewritten so a clean Modal pull reproduces every JSON | **TODO** | spec §11.4 |

---

## 6. Detailed test plan

For every test group: **purpose**, **files**, **command**, **pass criteria**,
**current status**.

### 6.1 Water-filling unit tests

- **Purpose.** Lock the greedy allocator behavior in spec §9.3-9.4 so that
  any later refactor preserves the property `sum(b_i) == total_bits` and
  the `i* = argmax_i lambda_i / 4 ** b_i` rule.
- **Files.** `tests/test_waterfill.py`, exercising
  `src/spectralquant/waterfill.py`.
- **Command.** `pytest tests/test_waterfill.py -q`.
- **Pass criteria.** All 34 tests green; uniform allocation when eigenvalues
  are equal; concentrated allocation when one eigenvalue dominates; deterministic
  tie-breaking; numpy / torch / list inputs all return identical int64 numpy
  output; original input objects remain unmutated; negative eigenvalues raise;
  zero eigenvalues do not produce NaN.
- **Status.** Passing as of this commit.

### 6.2 Accounting unit tests

- **Purpose.** Pin TurboQuant ≈ 5.02x at b=3 and ≈ 3.08x at b=5 to the
  spec §10 formula; assert that `check_headline_ratio` raises on the
  V1-GAP-014 5.95x discrepancy; round-trip `CompressionAccounting` through
  the JSON schema.
- **Files.** `tests/test_accounting.py`, exercising
  `src/spectralquant/accounting.py` and `schemas/accounting.schema.json`.
- **Command.** `pytest tests/test_accounting.py -q`.
- **Pass criteria.** All 23 tests green. Ratios derived from stored bits, not
  hard-coded.
- **Status.** Passing.

### 6.3 v2 quantization integration tests

- **Purpose.** Lock the `use_water_fill=False` ⇒ v1 invariant; assert that
  `use_water_fill=True` produces per-dim codebooks of the correct count, bit
  width, and sigma.
- **Files.** `tests/test_v2_quantization.py`, exercising
  `src/spectralquant/nonuniform_quantization.py` (and indirectly
  `waterfill.py`).
- **Command.** `pytest tests/test_v2_quantization.py -q`.
- **Pass criteria.** All 17 tests green; v1 vs v2 produce identical bits when
  the spectrum is uniform; v2 reallocates as expected when one eigenvalue
  dominates; edge cases d_eff=2 and d_eff=D-2 do not crash.
- **Status.** Passing.

### 6.4 Result-schema tests

- **Purpose.** Catch any drift between docs and schemas; ensure that every
  result file ever written under `results/three_way/` will carry
  `model`, `hardware`, `software`, `data`, `calibration`, `methods`,
  `evidence_ids`, `repo`, `commit`, `command`, and `timestamp`.
- **Files.** `tests/test_result_schema.py`, exercising the three schemas in
  `schemas/`.
- **Command.** `pytest tests/test_result_schema.py -q`.
- **Pass criteria.** Hand-crafted minimal documents validate; the on-disk
  `docs/evidence_catalog.json` validates.
- **Status.** Passing. Will be extended to validate every Phase-6 JSON
  glob-by-glob.

### 6.5 Calibration tests (TODO — `tests/test_calibration_v2.py`)

- **Purpose.** Lock spec §9.1 invariants — row normalization unit norms,
  symmetric covariance, sorted eigenvalues, orthonormal eigenvectors, d_eff
  clamped to `[2, D-2]`, save/load round-trip, pre-RoPE hook recorded,
  layer/head metadata correct.
- **Files (planned).** `tests/test_calibration_v2.py` exercising
  `src/spectralquant/calibration.py`.
- **Command.** `pytest tests/test_calibration_v2.py -q`.
- **Pass criteria.** All eight tests in spec §12.3 pass on a synthetic key
  cache (no model required).
- **Status.** Not yet written.

### 6.6 Engine tests (TODO — `tests/test_engine_v2.py`)

- **Purpose.** Lock the engine-level invariants in spec §12.4: v1 vs v2 equal
  total semantic MSE bits and selective QJL dimensions; shape preservation;
  finite logits; consistent causal masking; no silent mixing of normalized vs
  unnormalized keys; engine duplication resolved.
- **Files (planned).** `tests/test_engine_v2.py`, exercising the canonical
  engine after Phase 5 consolidation.
- **Command.** `pytest tests/test_engine_v2.py -q`.
- **Pass criteria.** All eight tests in spec §12.4 pass on a synthetic
  attention head (no model required where possible; otherwise tiny smoke
  model with `monkeypatch`).
- **Status.** Not yet written.

### 6.7 TurboQuant baseline tests (TODO — `tests/test_turboquant_baseline.py`)

- **Purpose.** Lock spec §12.5: random orthogonal matrix is orthonormal,
  seed-deterministic, full-dim QJL signs have dim D, 3-bit accounting ≈ 5.02x,
  attention scoring returns finite logits, baseline labeled "local" everywhere.
- **Files (planned).** `tests/test_turboquant_baseline.py`.
- **Command.** `pytest tests/test_turboquant_baseline.py -q`.
- **Pass criteria.** All six tests pass; both engines (cutile-based and
  legacy) tested or one explicitly deprecated.
- **Status.** Not yet written.

### 6.8 Smoke test

- **Purpose.** Catch a Phase 6 misconfiguration (wrong tokenizer, wrong
  layer-sample list, missing HF license) before paying for a full sweep.
- **Files.** `experiments/run_three_way.py` with `--dry-run` /
  `--synthetic-smoke` (local, no model download) and `--smoke` (Modal,
  alias for `--synthetic-smoke`).
- **Local commands.**

  ```bash
  # 1. Dry-run: validate args, print plan + output path, write nothing.
  python3 experiments/run_three_way.py \
    --model mistralai/Mistral-7B-v0.3 \
    --avg-bits 3 --n-calib 4 --n-eval 2 --n-layers-sample 2 \
    --output-dir results/three_way_smoke --dry-run

  # 2. Synthetic smoke: full three-way pipeline on tiny synthetic Q/K/V,
  #    writes a schema-valid JSON under results/three_way_smoke/.
  python3 experiments/run_three_way.py \
    --model mistralai/Mistral-7B-v0.3 \
    --avg-bits 3 --n-calib 4 --n-eval 2 --n-layers-sample 2 \
    --output-dir results/three_way_smoke --synthetic-smoke
  ```
- **Modal smoke command.** Use the launcher with `--smoke`:

  ```bash
  python3 scripts/launch_modal_three_way.py --dry-run \
    --model mistralai/Mistral-7B-v0.3 --avg-bits 3 --smoke
  # then drop --dry-run to actually launch on Modal
  ```

  The launcher constructs `python3 experiments/run_three_way.py …
  --synthetic-smoke` inside the Modal container.

- **Pass criteria.** Local: dry-run / synthetic-smoke return exit code 0
  and emit a payload that validates against
  `schemas/three_way_result.schema.json`. Modal: a schema-valid JSON is
  written under `results/three_way_smoke/`; cosine values are finite and
  sane; total wall-clock under 5 minutes on a single H200.
- **Status.** Local dry-run + synthetic-smoke modes implemented and
  covered by `tests/test_run_three_way.py`. Full HF model path implemented
  in `experiments/run_three_way.py::run_full_hf` and exercised via the
  Modal launcher; needs `transformers` and `datasets` installed (they are
  pip-installed in the Modal image but not locally).

### 6.8a Inline-corpus smoke (harness validation only)

- **Purpose.** Validate the harness end-to-end on a real (small) HF model
  when the WikiText / `datasets.load_dataset` download is hanging or
  unavailable. Exercises the model load → adapter discovery → hooks →
  calibration → quantization → eval pipeline without any dataset
  download.
- **Files.** `experiments/run_three_way.py --inline-corpus-smoke`
  (forwarded by `scripts/launch_modal_three_way.py --inline-corpus-smoke`
  and by `main_entry`'s `inline_corpus_smoke=True` parameter on Modal).
- **Local command.**

  ```bash
  # CPU is allowed; the inline corpus is tiny but the model still loads.
  python3 experiments/run_three_way.py \
    --model Qwen/Qwen2.5-0.5B \
    --avg-bits 3 --n-calib 4 --n-eval 2 --n-layers-sample 2 \
    --output-dir results/three_way_smoke \
    --device cuda --dtype float16 --inline-corpus-smoke
  ```

- **Modal command.**

  ```bash
  modal run -d scripts/launch_modal_three_way.py::main_entry \
    --model Qwen/Qwen2.5-0.5B --avg-bits 3 \
    --n-calib 4 --n-eval 2 --n-layers-sample 2 \
    --inline-corpus-smoke
  ```

- **Output.** A schema-valid JSON prefixed `inline_corpus_smoke__…json`
  with `mode: "inline-corpus-smoke"`, `paper_valid: false`,
  `evidence_ids: ["RUN-THREEWAY-INLINESMOKE-001"]`, and
  `data.calibration_corpus = data.eval_corpus = "inline_smoke"`.
- **Status events.** `dataset_inline_start` / `dataset_inline_end` are
  emitted in place of `dataset_load_start` / `dataset_load_end` so the
  status stream unambiguously identifies inline-corpus runs.
- **Important.** Inline-corpus runs are **harness validation only** and
  must never be cited as benchmark evidence. Full sweeps require
  WikiText (or the configured benchmark corpus). The metadata fields
  above (`paper_valid: false`, `calibration_corpus: "inline_smoke"`,
  evidence id ending in `INLINESMOKE`) are designed so downstream tooling
  can filter inline-corpus runs out of evidence catalogues.

### 6.9 Calibration stability tests (TODO)

- **Purpose.** Quantify how stable d_eff and waterfill allocations are across
  three independent calibration draws. Replaces V1-GAP-005's reconstructed
  stability JSON.
- **Files (planned).** `experiments/run_calibration_stability_v2.py` and
  per-run JSON in `results/calibration_stability_v2/`.
- **Pass criteria.** d_eff CV across draws < 10% on Mistral-7B-v0.3; waterfill
  Hamming distance between draws documented in the JSON.
- **Status.** Not yet written.

### 6.10 Benchmark reproducibility test (TODO)

- **Purpose.** Re-running `experiments/run_three_way.py --seed 42` twice on
  Modal produces bit-identical attention-output cosines.
- **Files (planned).** `tests/test_benchmark_reproducibility.py` runs the
  smoke variant twice and diffs the JSON minus timestamps.
- **Pass criteria.** Diff is empty after stripping `timestamp` and any
  hardware string.
- **Status.** Not yet written.

### 6.11 Paper / evidence tests (TODO)

- **Purpose.** Every empirical sentence in `paper_output/v2/spectralquant_v2.tex`
  carries an evidence ID that exists in `docs/evidence_catalog.json` AND that
  the cited number matches the JSON.
- **Files (planned).** `tests/test_paper_claims.py`.
- **Pass criteria.** No unmapped `\evidence{...}` macros; every cited number
  is within tolerance of the file it cites.
- **Status.** Not yet written.

---

## 7. Modal run plan

This is the operational runbook for Phase 6. It assumes the Modal CLI is
authenticated and that the repo has been pushed to `niashwin/spectralquant-v2`.

### 7.1 Environment

- Hardware: NVIDIA H200 (preferred) or H100 SXM. Single-GPU is sufficient for
  spec §13.2 models.
- Image: per `scripts/setup_b200.sh`. Required deps: `torch>=2.6`,
  `transformers>=5.6` (Mistral-7B-v0.3 needs `transformers>=4.40` plus
  `tokenizers` upgrades), `datasets>=4.8`, `numpy`, `scipy`, `jsonschema>=4.18`,
  `tqdm`, `rich`, `pandas`, `matplotlib`, `seaborn`, and the in-repo
  `baseline/turboquant_cutile/` package.
- Python: 3.10, 3.11, or 3.12 — `pyproject.toml` declares all three.

### 7.2 Hugging Face access (note for operators)

- `mistralai/Mistral-7B-v0.3` is gated. The Modal secret `HF_TOKEN` must be a
  token of an account that has accepted the Mistral license. Without this,
  `transformers.AutoModelForCausalLM.from_pretrained` fails with `403`.
- `Qwen/Qwen2.5-7B` is publicly downloadable but rate-limited.
- WikiText-103 is on `datasets`; no token required.
- Record the licenses-accepted state and the exact tokenizer revision in the
  output JSON's `software` field so the run is reproducible.

### 7.3 Models

| Model | Layers | Q heads | KV heads | Head dim | GQA | Notes |
|---|---:|---:|---:|---:|---:|---|
| `mistralai/Mistral-7B-v0.3` | 32 | 32 | 8 | 128 | 4:1 | gated; HF token required |
| `Qwen/Qwen2.5-7B` | 28 | 28 | 4 | 128 | 7:1 | public |

### 7.4 Datasets

- WikiText-103 via `datasets.load_dataset("wikitext", "wikitext-103-raw-v1")`.
- Calibration: `n_calib = 32`, `max_calib_tokens = 384`.
- Evaluation: `n_eval = 8`, disjoint from calibration (different document
  starts; record `seed` and `data.disjoint_eval=true` in the JSON).

### 7.5 Layer sampling

- Mistral: `[0, 4, 8, 12, 16, 20, 24, 28]`.
- Qwen: `[0, 3, 6, 9, 12, 15, 18, 21]`.

### 7.6 Commands to run on Modal

**Always read `docs/modal_safety_protocol.md` before launching.** The
commands below are the launch order; they assume credentials are provided
via environment variables or Modal secrets — never via CLI flags or
committed files. The Modal-side wrapper is
`scripts/launch_modal_three_way.py` — it constructs the exact
`run_three_way.py` argv, declares the GPU/timeouts/secrets, and mounts
the durable result volume.

Step 0 — preflight (read-only, never starts a run):

```bash
# Loads .env locally; on Modal, secrets are injected by the function decorator.
python3 scripts/preflight_modal.py --strict
```

The script checks Python version, git cleanliness, presence of `HF_TOKEN`
(without printing the value), Modal CLI availability, disk space, and that
no secret-looking path is tracked by git.

Step 0b — if `disk_space` is tight (or you expect a large model download),
free local cache space with the safe cleanup helper **before** launching.
Never use a broad `rm -rf` over `/tmp` or `~/.cache` — that pattern has, in
the past, deleted an active local clone alongside the intended caches. The
helper refuses to touch the active repo root and requires explicit
per-category flags plus `--yes`:

```bash
# Dry-run (default): show what would be deleted, change nothing.
python3 scripts/safe_local_cleanup.py \
  --delete-hf-cache --delete-playwright-cache --delete-temp-clones

# Apply once the plan looks right:
python3 scripts/safe_local_cleanup.py \
  --delete-hf-cache --delete-playwright-cache --delete-temp-clones --yes
```

See `docs/modal_safety_protocol.md` §9 for the full cleanup contract
(allow-listed categories, refusal rules, the active-repo guard).

Step 1 — local dry-run of the launcher (no Modal call, no compute):

```bash
python3 scripts/launch_modal_three_way.py --dry-run \
  --model mistralai/Mistral-7B-v0.3 --avg-bits 3 --seed 42
```

Step 2 — local synthetic-smoke (no model download, no Modal):

```bash
python3 experiments/run_three_way.py \
  --model mistralai/Mistral-7B-v0.3 \
  --avg-bits 3 --n-calib 4 --n-eval 2 --n-layers-sample 2 \
  --output-dir results/three_way_smoke \
  --synthetic-smoke
```

Step 3 — Modal smoke test (cheap; catches most misconfigurations):

```bash
# Verify the Modal app spec without launching:
python3 scripts/launch_modal_three_way.py --print-modal-app \
  --model mistralai/Mistral-7B-v0.3 --avg-bits 3

# Launch the synthetic-smoke pipeline inside the Modal container
# (attached / foreground — fine for synthetic smoke since it runs in
# under a few minutes):
python3 scripts/launch_modal_three_way.py \
  --model mistralai/Mistral-7B-v0.3 \
  --avg-bits 3 --n-calib 4 --n-eval 2 --n-layers-sample 2 \
  --output-dir /results/three_way_smoke \
  --smoke --gpu H200 --timeout-sec 900
```

Step 3b — Detached real-model smoke (HF download + small calibration):

The Python wrapper above will be killed if the local client hits a tool
or shell timeout, leaving the GPU run orphaned and the result JSON
unwritten. For any run that loads real HF weights, use Modal's native
detached mode so the remote app survives a local disconnect:

```bash
modal run -d scripts/launch_modal_three_way.py::main_entry -- \
  --model mistralai/Mistral-7B-v0.3 \
  --avg-bits 3 \
  --n-calib 4 --n-eval 2 --n-layers-sample 2 \
  --output-dir /results/three_way_smoke \
  --smoke
```

Internally, `main_entry` calls `run_one.spawn(...)` rather than
`.remote(...)`. This is mandatory for detached mode: a `.remote()` call
inside a detached app is *cancelled* when the local caller drops its
connection, with this Modal warning in stderr:

```
remote() and .map() calls in detached apps may be canceled when the
local caller disconnects. Use .spawn() for detached or background work.
```

We hit exactly this failure mode previously: the local tool boundary
(10 min) disconnected the client mid-eval, and although the *app* kept
running, the GPU function call itself was cancelled before it could
write the result JSON. Calibration artifacts landed on the volume but
the final result did not. Switching to `spawn()` fixes that.

On submission, the launcher prints (sanitized):

```
[launch_modal_three_way] forwarding local commit <sha12> via SPECTRALQUANT_GIT_COMMIT
[launch_modal_three_way] launching (entrypoint): mistralai/Mistral-7B-v0.3 b=3 seed=42 smoke=True
[launch_modal_three_way] spawned remote call: call_id=fc-XXXXXXXXXXXXXXXX
[launch_modal_three_way] expected_output_path: /results/three_way_smoke/synthetic_smoke__Mistral-7B-v0.3_b3_calib4_eval2_seed42.json
[launch_modal_three_way] status_path: /results/status/synthetic_smoke__Mistral-7B-v0.3_b3_calib4_eval2_seed42/status.json
[launch_modal_three_way] poll status: modal volume get spectralquant-v2-results /results/status/synthetic_smoke__Mistral-7B-v0.3_b3_calib4_eval2_seed42/status.json -
[launch_modal_three_way] poll: modal volume ls spectralquant-v2-results /results/three_way_smoke/synthetic_smoke__Mistral-7B-v0.3_b3_calib4_eval2_seed42.json
[launch_modal_three_way] result: python3 -c "import modal; print(modal.FunctionCall.from_id('fc-XXXXXXXXXXXXXXXX').get(timeout=0))"
```

Capture `call_id=`, `expected_output_path:`, and `status_path:` —
those three lines are everything you need to retrieve the run after the
local client exits.

Polling (status JSON → live progress; volume → final artifact;
FunctionCall → structured result; logs → last-resort triage):

```bash
# (a) Status JSON polling — emits during the run, including failures.
#     Heartbeat is rewritten atomically every ~30 s of subprocess output.
modal volume get spectralquant-v2-results \
  /results/status/<run_id>/status.json -

# Full event history (one line per stage transition):
modal volume get spectralquant-v2-results \
  /results/status/<run_id>/events.jsonl -

# List all in-flight runs at once:
modal volume ls spectralquant-v2-results /results/status/

# (b) Volume polling — the final result artifact appears at the printed path.
modal volume ls   spectralquant-v2-results /results/three_way_smoke
modal volume get  spectralquant-v2-results \
  /results/three_way_smoke/synthetic_smoke__Mistral-7B-v0.3_b3_calib4_eval2_seed42.json \
  ./local_results/

# (c) FunctionCall — fetches the structured ``run_one`` return value.
python3 -c "import modal; print(modal.FunctionCall.from_id('<call_id>').get(timeout=0))"  # peek
python3 -c "import modal; print(modal.FunctionCall.from_id('<call_id>').get())"            # block

# (d) App-level controls (last resort — logs are often unavailable).
modal app list                          # running apps + states
modal app logs <app-id>                 # tail (Ctrl-C does not stop the app)
modal app stop <app-id>                 # only if you must force-stop
```

**Sweep gate.** No full sweep is launched until the HF smoke writes a
`status.json` whose `stage` is `success`. If the smoke instead writes
`stage: failure`, read the `error` and `traceback` fields from
`status.json`, fix the failure mode, and rerun the smoke before
spending a full-sweep GPU minute. The persistent HF cache under
`/results/hf_cache/` (see `docs/modal_safety_protocol.md` §6d) makes
the smoke's model download a one-time cost across retries.

Recovery rules (mirrors `docs/modal_safety_protocol.md` §7.0a):

1. If the local client dies but the run finished, the JSON is on the
   Modal volume — pull with `modal volume get spectralquant-v2-results
   /three_way_smoke/<file>.json ./`. The exact filename is the
   `expected_output_path` printed at submission time.
2. A `.tmp_*` leftover in the output dir means a crash mid-write —
   triage from `modal app logs`, then clean with
   `python3 scripts/audit_results.py --delete-stale-tmp`.
3. Never re-run with `--force` until you have confirmed the failure
   mode; the default `--skip-if-exists` will silently no-op if a stale
   artifact is present.
4. If `FunctionCall.from_id(<call_id>).get()` raises
   `modal.exception.ExecutionError`, the call was cancelled (e.g. the
   app was stopped). Check `modal app logs` and inspect the volume for
   partial state before re-submitting.

Step 4 — full Mistral sweep (resumable: `--skip-if-exists` is default):

```bash
for B in 2 3 5; do
  python3 scripts/launch_modal_three_way.py \
    --model mistralai/Mistral-7B-v0.3 \
    --avg-bits "$B" \
    --n-calib 32 --n-eval 8 --n-layers-sample 8 \
    --output-dir /results/three_way \
    --calibration-dir /results/calibration \
    --save-calibration \
    --gpu H200 --timeout-sec 5400 \
    --seed 42
done
```

The first b=B run for a given (model, seed, n_calib) writes
`/results/calibration/{model}_calib32_tok384_seed42.pt`; the next two
runs reload it via `--load-calibration` so eigendecomposition is paid
only once per model.

Step 5 — Qwen 3-bit:

```bash
python3 scripts/launch_modal_three_way.py \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 \
  --n-calib 32 --n-eval 8 --n-layers-sample 8 \
  --output-dir /results/three_way \
  --calibration-dir /results/calibration \
  --save-calibration \
  --gpu H200 --timeout-sec 5400 \
  --seed 42
```

Or run the full controlled matrix from a JSON config (one config at a
time, resumable, with explicit `RunConfig` keys validated at load time):

```bash
python3 scripts/launch_modal_three_way.py \
  --matrix-config docs/sweep_matrix.json \
  --gpu H200 --timeout-sec 5400
```

Step 4 — compression-accounting audit (no model load):

```bash
python3 experiments/run_compression_accounting_audit.py \
  --output-dir results/accounting_audit
```

Step 5 — post-run audit and validation:

```bash
python3 scripts/audit_results.py            # human table; --strict to enforce
python3 experiments/plot_three_way.py \
  --input results/three_way \
  --output paper_output/v2/figures
pytest tests/test_result_schema.py -q
```

**Credentials.** The runner expects:

- `HF_TOKEN` — Hugging Face token of an account that has accepted the
  Mistral-7B-v0.3 license. Inject via Modal secret (`modal.Secret.from_name(
  "hf-token")`) or via local `.env` (gitignored). Never via a CLI flag and
  never echoed in logs.
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` — only for the Modal CLI itself.
  Prefer `~/.modal.toml` (created by `modal token new`) over env vars; do
  not pass to scripts.

`.env.example` (committed) lists names only; copy to `.env` (gitignored)
to fill in values locally.

### 7.7 Expected outputs

For each successful three-way run:

```
results/three_way/<model_short>_bits<B>_seed42.json
```

containing per-spec §14: `run_id`, `timestamp`, `repo`, `commit`, `command`,
`model{...}`, `hardware{...}`, `software{...}`, `data{...}`,
`calibration{normalize_keys, key_space, d_eff_method, d_eff_rounding,
d_eff_min, d_eff_max, d_eff_stats{mean,min,max}}`,
`methods.{turboquant,spectralquant_v1,spectralquant_v2}.{cosine_mean,
cosine_std,cosine_min,cosine_max,per_layer[],compression_accounting{...},
waterfill_allocation[]}`, and `evidence_ids[]`.

### 7.8 Resource and time expectations (qualitative)

These are operator-side qualitative estimates, not measured. Every run that
completes must overwrite this with measured numbers.

- Calibration on Mistral-7B-v0.3 with `n_calib=32, max_tokens=384` is roughly
  a few minutes on H200; v1's "15-second" framing is contested
  (V1-GAP-007).
- The full 3-way attention-output sweep over 8 sampled layers, 4 KV heads each,
  `n_eval=8` is wall-clock-dominated by model load, not by the quantizer.
- The full Mistral 2/3/5-bit + Qwen 3-bit matrix is expected to fit comfortably
  in a single Modal job hour, but should be timed per-run and recorded in the
  JSON's `timing` field once it exists.

### 7.9 Artifacts to save

For every Modal run, save into `results/three_way/`:

- The per-run JSON (validates against `schemas/three_way_result.schema.json`).
- A copy of the standard-out / standard-err log, named
  `<run_id>.stdout.log`, for the `command` field's provenance.
- The `accounting_audit/<run_id>.json` produced from the same compressed
  state.
- Any plotted figure under `paper_output/v2/figures/` must be regeneratable
  from the JSON via `experiments/plot_three_way.py`.

---

## 7.7. Next-stage evaluation harnesses (perplexity, LongBench, generation, latency)

**Status as of 2026-04-30T22:04Z: all four families have at least one
`paper_valid=true` artifact on Modal + in-repo.** Perplexity,
generation, and latency landed on commit
`197bcfb4ad54a7d7bc9430a80695c62c145371fd`; LongBench
deterministic-subset landed on commit
`1ecb578a0b0251f1a716469e51be4303c7191cd6` after a 12 h-capped relaunch
that replaced an earlier 6 h-capped attempt that had been kill-switched
without writing a canonical JSON. Macro / headline numbers are
extracted in `docs/evidence_family_validation_2026-04-30.md`; the catalog
entries are `RUN-PERPLEXITY-QWEN2.5-7B`, `RUN-GENERATION-QWEN2.5-7B`,
`RUN-LATENCY-QWEN2.5-7B`, and `RUN-LONGBENCH-QWEN2.5-7B-DETERMINISTIC`
in `docs/evidence_catalog.md`.

**Recovery lesson preserved (post-2026-04-30 LongBench incident).**
The original LongBench launch (commit `6154175`, app
`ap-vTqL16w5Nmaw6s2oGc6czU`, started `2026-04-30T10:10Z`) hit the 6 h
`FAMILY_TIMEOUT_SEC[longbench] = 21600` Modal kill-switch with two
methods complete and TurboQuant trec at 23/50; because the harness
only wrote the canonical JSON on a fully-successful three-method
finish, **6 h of compute were unrecoverable**. The fix shipped in
commit `1ecb578` is two-part:

1. The harness now writes a per-method full-record shard
   `partial/method__<m>.json` immediately after each method's
   evaluation finishes (`experiments/run_longbench.py`,
   `_write_method_partial_record`). A future kill-switched run with
   ≥ 1 method complete is recoverable via
   `scripts/merge_longbench_partials.py --paper-valid` over the
   shard set.
2. The launcher honours `SPECTRALQUANT_MODAL_TIMEOUT_LONGBENCH_SEC`,
   so an operator who knows the SQv2 hooked-replay path costs
   ≈ 3.66 h on its own can raise the cap to 12 h or more without code
   changes. The 2026-04-30 relaunch used `=43200` and finished in
   ≈ 5 h 38 min, leaving ≈ 6 h 22 min of headroom.

The general invariants below remain.

The four next-stage evidence families ship as separate Modal-friendly
harnesses under `experiments/`, with a unified launcher
`scripts/launch_modal_eval.py` that selects the family via `--family`.
Each harness mirrors the operational discipline of
`experiments/run_three_way.py`: `--dry-run` / `--synthetic-smoke` /
`--inline-corpus-smoke`, atomic JSON writes with schema validation,
status artifacts at `/results/status/<family>/<run_id>/`, sanitized
command strings, and `paper_valid=false` for every non-`full` run.

**Important — these families are NOT YET unblocked for paper-valid use.**
Until the SpectralQuant engine is wired into HuggingFace generation
end-to-end, only the FP16 reference rows are real measurements; non-FP16
method rows are placeholders that emit a caveat into the JSON. The
relevant gates are still:

- V1-GAP-004b — perplexity. The harness records FP16 PPL but cannot
  yet record TQ / SQ v1 / SQ v2 PPL because the engine wraps a
  per-(layer, head) attention hook, not a full causal-LM forward pass.
- V1-GAP-008 — LongBench. The full path is now implemented. It loads
  the upstream `THUDM/LongBench` HF dataset via
  `experiments/longbench_dataset.py` (vendored prompt templates +
  middle-truncation), generates greedy completions at the upstream
  per-task `max_new_tokens`, and scores each task with a transparent
  in-repo re-implementation of the LongBench metric registry
  (`experiments/longbench_metrics.py`: token F1, ROUGE-L, classification
  EM, retrieval EM, count EM, code edit similarity).

  Loader resilience: `THUDM/LongBench` ships as a script-based dataset
  (`LongBench.py` + `data.zip` on the Hub). Newer `datasets` releases
  reject scripts entirely with `RuntimeError: Dataset scripts are no
  longer supported, but found LongBench.py`. The loader first tries
  `datasets.load_dataset(..., trust_remote_code=True)` for older
  installs, then falls back to downloading `data.zip` directly via
  `huggingface_hub.hf_hub_download` and parsing the per-task
  `data/<config>.jsonl` files. The Modal image now includes
  `huggingface_hub>=0.20`, and the extracted archive is cached under
  `$HF_DATASETS_CACHE/longbench_thudm_data/`, which the launcher
  points at the persistent volume — first run pays the ~114 MB
  download once, retries reuse the cache.
  `paper_valid=true` requires (a) `mode=full` with the HF dataset
  actually loaded (every method record carries
  `dataset_source=huggingface_thudm`), (b) `n_per_task ≥ 50`,
  (c) every requested method in `REAL_EVAL_METHODS`, (d) no placeholder
  records, and (e) replay coverage ≥ 0.99 for non-FP16 methods.
  Subsets smaller than `full` are valid evidence for *that subset*,
  but the artifact carries an explicit "transparent subset of LongBench"
  caveat so a downstream report cannot headline it as full LongBench.
  `--inline-corpus-smoke` and `--synthetic-smoke` are still available
  for harness validation and remain `paper_valid=false`.
- Real generation quality — schema, judge-free metrics, and the
  K/V projection-replay path are now wired in. FP16,
  `spectralquant_v2`, and the in-repo `turboquant` baseline produce
  real completions; methods outside this set remain placeholders that
  block `paper_valid`. `paper_valid=true` requires every requested
  method to be real, no placeholder records, and replay coverage ≥
  0.99 for non-FP16.
- End-to-end latency — FP16 is end-to-end timing on the HF model.
  `spectralquant_v2` and `turboquant` now produce **two** distinct
  timing rows per operating point:
  1. **Microbenchmark** (`microbenchmark=true`,
     `microbenchmark_kind=kv_compress_decompress_round_trip`,
     `end_to_end_measured=false`): times only the engine's K/V
     compress→decompress round-trip on a synthetic K/V tensor. Useful
     for kernel-level signal but **not** end-to-end inference latency.
  2. **Hooked replay end-to-end** (`end_to_end_measured=true`,
     `production_kernel=false`,
     `measurement_kind=hooked_replay_end_to_end`): the full HF forward
     + decode loop with K/V projection-replay hooks attached, so every
     layer's K/V passes through compress→decompress before the rest of
     attention. This is *real* end-to-end inference timing under K/V
     cache compression, but the per-layer hooks add Python-level
     overhead. Downstream reports must call this "hooked replay
     end-to-end latency", **not** "production speedup". The rows are
     reported separately from FP16 and never headlined as a head-to-head
     speed claim.

  A paper-valid latency artifact requires `--device cuda`, the
  `torch.cuda.Event` timer, no placeholder rows, every non-FP16 method
  must have at least one `end_to_end_measured=true` row (microbenchmark
  alone is insufficient), and the replay coverage on the hooked-replay
  rows must be ≥ 0.99. See `experiments/run_latency.py::build_payload`
  for the exact gate. The two CLI flags `--include-microbench` /
  `--no-microbench` and `--include-end-to-end-replay` /
  `--no-end-to-end-replay` toggle each path independently.

The K/V projection-replay path lives in
`experiments/sqv2_replay.py`. It calibrates the SpectralQuant v2
engine (or `TurboQuantBaseline`) on the eval corpus, attaches
`forward_hook`s to every layer's `k_proj` and `v_proj`, and replaces
the projection output with `compress(x)` → `decompress(...)`. The HF
model then sees compressed K/V flowing through the rest of attention
naturally. The hook records per-layer coverage in
`replay_coverage.fraction_layers_real`; a layer that fails the
round-trip falls back to FP16 passthrough and is counted in
`n_passthrough_calls`.

This is intentionally a narrower claim than "end-to-end TurboQuant vs
SpectralQuant v2 architecture comparison": weight quantization, kernel
optimizations, and Google's official TurboQuant kernels are out of
scope. What we *do* claim is the perplexity / generation quality / KV
microbenchmark latency of the model under the named K/V cache
compression method.

### 7.7.1. Files

| Path | Role |
|---|---|
| `experiments/eval_common.py` | Shared utilities: provenance, atomic write, sanitizer, run-id, base payload, inline corpus, default prompt set. |
| `experiments/run_perplexity.py` | Perplexity harness. Validates against `schemas/perplexity.schema.json`. |
| `experiments/run_longbench.py` | LongBench harness scaffold. Validates against `schemas/longbench.schema.json`. |
| `experiments/run_generation.py` | Real-generation quality harness. Validates against `schemas/generation.schema.json`. |
| `experiments/run_latency.py` | End-to-end latency harness. Validates against `schemas/latency.schema.json`. |
| `experiments/sqv2_replay.py` | K/V projection-replay engine (compress+decompress hooks). Used by run_perplexity, run_generation, run_longbench, and run_latency for real `spectralquant_v2` / `turboquant` evaluation. |
| `scripts/launch_modal_eval.py` | Unified Modal launcher (foreground + detached). |
| `tests/test_eval_harnesses.py` | Schema, run-id, smoke-mode, dry-run, paper-valid-discipline tests. |
| `tests/test_eval_paper_valid_gates.py` | Paper-valid gating tests: placeholder records, unsupported methods, low coverage, and inline-corpus all block `paper_valid=true`. |

### 7.7.2. Local smoke commands (no GPU, no model download)

These commands write schema-valid JSON in seconds. Use them to verify
the result-plumbing before submitting any Modal job.

```bash
# Perplexity
PYTHONPATH=src python experiments/run_perplexity.py \
  --model mistralai/Mistral-7B-v0.3 --output-dir /tmp/sq2_smoke \
  --synthetic-smoke --avg-bits 3 --methods fp16

# LongBench
PYTHONPATH=src python experiments/run_longbench.py \
  --model Qwen/Qwen2.5-7B --output-dir /tmp/sq2_smoke \
  --synthetic-smoke --subset minimal --n-per-task 5 --methods fp16

# Generation
PYTHONPATH=src python experiments/run_generation.py \
  --model mistralai/Mistral-7B-v0.3 --output-dir /tmp/sq2_smoke \
  --synthetic-smoke --methods fp16

# Latency
PYTHONPATH=src python experiments/run_latency.py \
  --model mistralai/Mistral-7B-v0.3 --output-dir /tmp/sq2_smoke \
  --synthetic-smoke --methods fp16 --batch-sizes 1 --context-lengths 512,1024
```

### 7.7.3. Modal commands (operator-launched; not run by subagent)

The unified launcher accepts a `--family`, plus the harness's CLI
flags after `--extra`. Both the synchronous (`launch_modal_eval.py
--family ...`) and the detached (`modal run -d ...::main_entry`)
forms are supported.

**Tiny first-run smoke for each family** (Modal, FP16-only, small
context, low cost):

```bash
# Perplexity, FP16 only, n_eval_sequences=8, max_eval_tokens=512.
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family perplexity \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 \
  --extra="--n-eval-sequences 8 --max-eval-tokens 512 --stride 256"

# LongBench, FP16 only, minimal subset, n_per_task=5.
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family longbench \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 \
  --extra="--subset minimal --n-per-task 5 --max-input-tokens 4096 --max-new-tokens 64 --inline-corpus-smoke"
# NOTE: full path is not yet implemented; --inline-corpus-smoke runs
#   FP16 against deterministic in-memory tasks for harness validation.

# Generation, FP16 greedy decode, deterministic prompts.
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family generation \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 \
  --extra="--max-new-tokens 64 --temperature 0.0"

# Latency, FP16, 2 operating points only.
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family latency \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 \
  --extra="--batch-sizes 1 --context-lengths 512,1024 --gen-tokens 32 --warmup-iters 2 --measured-iters 5"
```

**Full target runs (after each smoke passes).** Start with one
representative config per family before fanning out to the full
matrix; the suggested starting config matches the paper-valid
three-way matrix already on disk:

```bash
# Perplexity — Qwen2.5-7B b=3 (matches RUN-THREEWAY-QWEN-3BIT model+bits)
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family perplexity \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 \
  --extra="--n-eval-sequences 64 --max-eval-tokens 1024 --stride 512"

# Perplexity — Mistral-7B-v0.3 b=2/3/5 (one bits-budget at a time)
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family perplexity \
  --model mistralai/Mistral-7B-v0.3 \
  --avg-bits 3 --methods fp16 \
  --extra="--n-eval-sequences 64 --max-eval-tokens 1024 --stride 512"

# LongBench — Qwen2.5-7B subset=deterministic n_per_task=50 (FP16 only)
#   Paper-valid threshold for V1-GAP-008 is n>=50/task.
#   This now uses the real THUDM/LongBench HF dataset + vendored metrics.
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family longbench \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 \
  --extra="--subset deterministic --n-per-task 50 --max-input-tokens 8192 --max-new-tokens 128"

# Generation — Qwen2.5-7B greedy on the default 8-prompt set
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family generation \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 \
  --extra="--max-new-tokens 128 --temperature 0.0"

# Latency — Qwen2.5-7B, 3 context lengths, 5 measured iters per OP.
#   Use --device cuda; set --measured-iters 10+ for tighter p50/p95.
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family latency \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 \
  --extra="--batch-sizes 1 --context-lengths 512,1024,2048 --gen-tokens 64 --warmup-iters 3 --measured-iters 10"
```

**Compressed-method runs (real K/V replay).** These are the new
paper-valid candidate runs that the K/V projection-replay path
produces. Each requires real GPU and the model on disk; smoke-test
the FP16-only command above first to verify the harness pipe.

```bash
# Perplexity — fp16 + SpectralQuant v2 (real compressed) + TurboQuant baseline.
#   Increase --n-eval-sequences and --max-eval-tokens until n_tokens
#   exceeds the paper-valid threshold (default 64k tokens).
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family perplexity \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 spectralquant_v2 turboquant \
  --extra="--n-eval-sequences 256 --max-eval-tokens 1024 --stride 512"

# Generation — fp16 reference + SpectralQuant v2 + TurboQuant on the
# default 8-prompt set, deterministic greedy decode.
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family generation \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 spectralquant_v2 turboquant \
  --extra="--max-new-tokens 128 --temperature 0.0"

# Latency — fp16 end-to-end + BOTH microbenchmark and hooked-replay
# end-to-end for v2 / TurboQuant. The microbenchmark rows are kernel-
# level signal; the hooked-replay rows are real forward+decode timing
# with K/V compression hooks (production_kernel=false). Downstream
# reports must call the v2/TQ rows "hooked replay end-to-end latency",
# not "production speedup".
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family latency \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 spectralquant_v2 turboquant \
  --extra="--batch-sizes 1 --context-lengths 512,1024,2048 --gen-tokens 64 --warmup-iters 3 --measured-iters 10 --device cuda --include-microbench --include-end-to-end-replay"

# LongBench — full path on the real THUDM/LongBench HF dataset.
# Paper-valid for the requested subset when n_per_task >= 50, methods
# are in REAL_EVAL_METHODS, replay coverage >= 0.99 for non-FP16, AND
# calibration knobs meet the paper-valid bar (n_calib>=16,
# lloyd_max_iter==200, calib_max_seq_tokens>=512). Subsets smaller than
# "full" carry an explicit "transparent subset" caveat — do NOT headline
# as full LongBench.
#
# Recommended SAFE relaunch (post-2026-04-30 timeout incident): pass
# --calibration-mode paper so the harness hard-fails on any silent
# downgrade of the calibration knobs, plus a generous timeout.
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family longbench \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 spectralquant_v2 turboquant \
  --extra="--subset deterministic --n-per-task 50 --max-input-tokens 8192 --max-new-tokens 128 --calibration-mode paper --n-calib 16 --lloyd-max-iter 200 --calib-max-seq-tokens 512"

# LongBench — full 21-task path (long-running). Use only after the
# subset=deterministic command above succeeds.
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family longbench \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 spectralquant_v2 turboquant \
  --extra="--subset full --n-per-task 50 --max-input-tokens 8192 --max-new-tokens 128"

# LongBench — inline-corpus harness validation (NEVER paper_valid;
# synthetic corpus). Useful for plumbing checks pre-flight.
modal run -d scripts/launch_modal_eval.py::main_entry -- \
  --family longbench \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods fp16 spectralquant_v2 turboquant \
  --extra="--subset minimal --n-per-task 50 --max-input-tokens 4096 --max-new-tokens 64 --inline-corpus-smoke"
```

### 7.7.4. Output paths on the Modal volume

Each harness writes to its family directory under `/results/`:

| Family | Output dir | Status dir |
|---|---|---|
| perplexity | `/results/perplexity/` | `/results/status/perplexity/<run_id>/` |
| longbench | `/results/longbench/` | `/results/status/longbench/<run_id>/` |
| generation | `/results/generation/` | `/results/status/generation/<run_id>/` |
| latency | `/results/latency/` | `/results/status/latency/<run_id>/` |

Filenames are deterministic (see `eval_common.derive_run_id`) so
`--skip-if-exists` is safe across retries. To force a re-run pass
`--force` (mapped through the launcher).

### 7.7.4a. LongBench calibration tractability and the safe-relaunch policy

**Incident (2026-04-30).** A LongBench `deterministic`/`spectralquant_v2`
run on Modal loaded `THUDM/LongBench` correctly, finished FP16
generation, and then went silent at status
`stage=calibration_start, message='spectralquant_v2 n_calib=16'`. The
process was CPU-bound inside `experiments/sqv2_replay.build_calibrated_engine`
(per-(layer, head) eigh, CPU rotated K/V capture, Lloyd-Max codebook
fits up to 200 iterations). The Modal call was cancelled; no result
JSON was written.

**Mitigations now in repo.**

1. **Observable progress.** `experiments/sqv2_replay.build_calibrated_engine`
   accepts a `progress` callback and emits coarse milestones:
   `calib_eigh_start` / `calib_eigh_end`, `calib_capture_start` /
   `calib_capture_end`, `calib_fit_start`, one `calib_fit_progress` event
   per `(layer, head, type)` triple, and `calib_fit_end`. The LongBench
   harness wires this into the `StatusWriter` so future runs cannot go
   silent during calibration. The new stages are listed in
   `experiments/run_status.KNOWN_STAGES`.
2. **CLI knobs for tractability.** `experiments/run_longbench.py` now
   exposes `--n-calib`, `--calib-max-seq-tokens`, and `--lloyd-max-iter`,
   plus a `--calibration-mode` flag that takes `auto`, `paper`, or
   `smoke`. Defaults: `n_calib = max(16, 4 * n_tasks)`,
   `calib_max_seq_tokens = 512`, `lloyd_max_iter = 200`. These match the
   previous hard-coded values, so existing paper-valid runs are
   unaffected; an operator must explicitly opt into reduced calibration.
3. **Reduced-calibration policy (paper validity).** Any run with
   `n_calib < 16` OR `lloyd_max_iter < 200` OR `calib_max_seq_tokens <
   512` AND a non-FP16 method is requested is tagged
   `calibration.reduced_calibration = true` in the result JSON, an
   explicit "reduced_calibration: …" caveat is appended, and
   `paper_valid` is forced to `false` even if every other gate passes.
   `--calibration-mode paper` makes any of those knobs a hard CLI error
   so a paper-mode launch cannot silently degrade. `--calibration-mode
   smoke` is the bounded paper-candidate path: it produces a schema-valid,
   non-paper-valid artifact intended for harness validation. The harness
   does NOT silently weaken paper-valid gates.
4. **Run-id discipline.** When (and only when) calibration knobs deviate
   from the paper-valid defaults, `derive_run_id` appends
   `_calib<N>_lm<I>_cs<T>` so smoke and paper-valid artifacts cannot
   collide on disk. Filenames for default (paper-valid) runs are
   unchanged.

**Safe relaunch — recommended commands.** The launcher's
`main_entry` is a Modal `@local_entrypoint` and accepts flags
*directly* after the entrypoint reference; the standalone `--`
separator (used by raw `modal run`) is unnecessary and previously
failed with `missing --model`. The launcher passes pass-through
harness arguments via the single `--extra` string.

The Modal kill-switch is now resolved per-family by
`scripts/launch_modal_eval.py` (see `FAMILY_TIMEOUT_SEC`); the
longbench default is `21600s` (6 h). To pin it explicitly at module
load time set
`SPECTRALQUANT_MODAL_TIMEOUT_LONGBENCH_SEC=<seconds>` in the
launching shell — the value is read once at app build time, so the
detached call is bounded by that ceiling for its entire lifetime.
A larger ceiling does NOT increase Modal cost; you are billed for
actual GPU minutes. The cost ceiling on a paper-valid run is set by
the harness flags (`--n-per-task`, `--max-new-tokens`, `--n-calib`,
`--lloyd-max-iter`), not by the kill-switch.

```bash
# (a) Full paper-valid relaunch on `deterministic` subset.
#     --calibration-mode paper hard-fails on any downgrade. The 6 h
#     Modal timeout is the new family default; no env var required.
modal run -d scripts/launch_modal_eval.py::main_entry \
  --family longbench \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods "fp16,spectralquant_v2,turboquant" \
  --force \
  --extra "--subset deterministic --n-per-task 50 --max-input-tokens 8192 --max-new-tokens 128 --calibration-mode paper --n-calib 16 --lloyd-max-iter 200 --calib-max-seq-tokens 512"

# (a') Same as (a) but pinning a longer kill-switch (e.g. for a 13B
#     model rerun). Only the env var path can change the timeout for
#     a detached `main_entry` launch — the CLI's `--timeout-sec` flag
#     is consumed by the standalone `python3 scripts/launch_modal_eval.py`
#     entrypoint, not by `main_entry`.
SPECTRALQUANT_MODAL_TIMEOUT_LONGBENCH_SEC=43200 \
  modal run -d scripts/launch_modal_eval.py::main_entry \
  --family longbench \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods "fp16,spectralquant_v2,turboquant" \
  --force \
  --extra "--subset deterministic --n-per-task 50 --max-input-tokens 8192 --max-new-tokens 128 --calibration-mode paper --n-calib 16 --lloyd-max-iter 200 --calib-max-seq-tokens 512"

# (b) Bounded paper-candidate / smoke (NOT paper-valid; emits caveat).
#     Use to confirm the engine is observable end-to-end before paying
#     for the full calibration. Shorter Lloyd-Max + tighter token cap
#     is the single biggest CPU-bound knob; the run will write events
#     for every (layer, head, type) triple so you can tell it is alive.
modal run -d scripts/launch_modal_eval.py::main_entry \
  --family longbench \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods "fp16,spectralquant_v2,turboquant" \
  --force \
  --extra "--subset minimal --n-per-task 50 --max-input-tokens 4096 --max-new-tokens 64 --calibration-mode smoke --n-calib 4 --lloyd-max-iter 25 --calib-max-seq-tokens 256"
```

Polling cadence (status JSON is rewritten on every emit). Note the
per-family directory: `launch_modal_eval.py` writes status under
`/results/status/<family>/<run_id>/`, so for LongBench the `<family>`
prefix is `longbench`.

```bash
modal volume get spectralquant-v2-results \
  /results/status/longbench/<run_id>/status.json -        # latest snapshot
modal volume get spectralquant-v2-results \
  /results/status/longbench/<run_id>/events.jsonl -       # full history

# Listing all live longbench runs (the standalone CLI prints the exact
# directory in its spawn output as ``poll status: modal volume ls ...``):
modal volume ls spectralquant-v2-results /results/status/longbench/
```

A run that *appears* silent should be inspected with `events.jsonl`
first: a healthy v2 calibration emits one `calib_fit_progress` event
per layer (so 32 events for Mistral-7B-v0.3, 28 for Qwen2.5-7B). If the
file plateaus on `calib_fit_start` for many minutes the run is likely
CPU-bound and should be re-launched with reduced knobs (option (b)).

### 7.7.4b. LongBench generation-stage stall policy

**Incident (2026-04-30, post-calibration).** A LongBench `paper`/
`spectralquant_v2` Modal run (call `fc-01KQEQT92W82WXJN3JEEH0Y6RR`,
app `ap-dQv7icBKxLZLVupCr3DRB3`) successfully passed dataset loading,
FP16 generation, and the newly instrumented v2 calibration (28
`calib_fit_progress` events + `calibration_end`). It then entered
status `stage=eval_progress message='spectralquant_v2 task=narrativeqa'`
and emitted nothing further during the SQ-v2 replay generation loop —
the harness only emitted one `eval_progress` event *per task*, with no
per-example heartbeat and no on-disk record of partial completions, so
the parent observer could not distinguish a slow `model.generate` from
a stuck one and could not tell which example index had been reached.

**Mitigations now in repo.**

1. **Per-example progress.** `experiments/run_longbench.py` now wraps
   every `_generate_for_task` call in a `GenerationProgress` object
   that emits `eval_task_start`, one `eval_progress` event per example
   (configurable via `progress_every_n`), and `eval_task_end`. Each
   event's `details` block carries `method`, `task`, `completed`,
   `total`, and `elapsed_s`, so a poller can compute throughput and
   ETA without parsing log text. The new stages are listed in
   `experiments/run_status.KNOWN_STAGES`.
2. **Safe partial checkpoints.** After every example the harness
   atomically rewrites a tiny per-(method,task) JSON shard at
   `<status_dir>/partial/<method>__<task>.json` containing
   `completed`, `total`, `elapsed_s`, `status` ∈ {`running`,`ended`},
   `paper_valid: false`, `partial: true`, and `kind:
   "longbench_partial_task"`. A run-level snapshot
   `<status_dir>/partial/partial_status.json` is rewritten at every
   method/task transition with `methods_planned`, `tasks_planned`,
   `completed_methods`, `current_method`, `current_task`, also tagged
   `paper_valid: false, partial: true`. These artifacts are NEVER
   schema-validated against `schemas/longbench.schema.json` and are
   never written to the canonical result path; they live next to
   `events.jsonl` in the status tree only. An interrupted run leaves
   explicit evidence of where it stopped.
3. **Paper-valid discipline.** Partial outputs do not weaken any
   paper-valid gate. The harness still produces the canonical result
   JSON only via `eval_common.atomic_write_json` at the end of a
   successful run, with `paper_valid` derived from the same
   `n_per_task ≥ 50`, real-dataset, replay-coverage, and
   reduced-calibration gates as before. No example is ever silently
   skipped — `_generate_for_task` always iterates the full row list.
4. **No automatic resume.** A relaunch is required if the run is
   cancelled. The partial JSON tells you where to look but the
   harness does NOT consume it on the next launch — exact resume
   would require persisting per-example completions plus the
   calibrated engine state, which is not safe to do without a
   per-checkpoint hash of the engine and the inputs. The recommended
   recovery is a fresh launch with the *same* CLI flags.

**Polling for the generation stage (exact paths).** As above, the
launcher places the status tree under
`/results/status/<family>/<run_id>/`; for LongBench the `<family>`
component is literally `longbench`. The previously-documented
`/results/status/<run_id>/...` form is a historical artifact of the
three-way launcher (`launch_modal_three_way.py`), where there was no
family separator. Always inspect the launcher's spawn log for the
exact directory.

```bash
# Latest snapshot (single status event).
modal volume get spectralquant-v2-results \
  /results/status/longbench/<run_id>/status.json -

# Full event history. Healthy generation emits one eval_progress per
# example: 50 events per task at n_per_task=50.
modal volume get spectralquant-v2-results \
  /results/status/longbench/<run_id>/events.jsonl -

# Per-(method,task) partial completions. Rewritten on every example.
modal volume get spectralquant-v2-results \
  /results/status/longbench/<run_id>/partial/<method>__<task>.json -

# Run-level partial snapshot. Rewritten at every method/task transition.
modal volume get spectralquant-v2-results \
  /results/status/longbench/<run_id>/partial/partial_status.json -
```

**Stall classification.**

| Symptom in `events.jsonl` | Likely cause | Action |
| ------------------------- | ------------ | ------ |
| Last event is `calibration_start` for many minutes, no `calib_fit_progress` | CPU-bound calibration | Cancel; relaunch with `--calibration-mode smoke --lloyd-max-iter 25 --calib-max-seq-tokens 256` to confirm health, then full mode |
| `calib_fit_progress` plateaus mid-fit | Single-layer calibration is slow | Wait; one event per (layer, head, type) is expected |
| `eval_task_start` emitted but no `eval_progress` for many minutes on the *first* example | First-token compile / KV-cache warmup | Wait at least 5 min; re-check `partial/<method>__<task>.json` |
| `eval_progress` plateaus at completed `< total` | Single-example `model.generate` stuck | Cancel; relaunch with same flags. The partial JSON tells you which example index stalled, useful for upstream issue triage |
| No events at all for many minutes after `start` | Modal worker not yet picked up the call | Wait; `modal app logs <app_id>` will tell you if the worker is dead |

A run that emits `eval_task_start` for `(method, task)` but never
emits an `eval_task_end` for that pair, and whose `partial/<method>__<task>.json`
shows `completed < total` and `status: "running"`, was interrupted
mid-task. The partial JSON is the authoritative record of where it
stopped; the canonical result JSON does not exist for that run.

### 7.7.4c. Modal kill-switch timeout for paper-valid LongBench

**Incident (2026-04-30, post-generation-stall fix).** A LongBench
`paper`/`spectralquant_v2`/`turboquant`/`fp16` Modal run (call
`fc-01KQEQT92W82WXJN3JEEH0Y6RR`, commit `98e559f`) on `Qwen/Qwen2.5-7B`
finished FP16 generation, completed SQ-v2 calibration with 28
`calib_fit_progress` events and `calibration_end`, and advanced into
`spectralquant_v2 task=qasper` — and was then killed by Modal at the
default 5400 s function timeout. The run was healthy; it simply
needed more wall-clock than the kill-switch allowed.

**Mitigation now in repo.** `scripts/launch_modal_eval.py` exposes a
per-family timeout map plus environment-variable and CLI overrides:

- `FAMILY_TIMEOUT_SEC = {"perplexity": 5400, "generation": 5400,`
  `"latency": 3600, "longbench": 21600}`. The longbench cap of 6 h is
  derived from the observed runtime envelope: ≈10 min calibration +
  21 tasks × 50 examples × seconds-per-example with input ≤8 192
  tokens / output ≤128 tokens, on a 7B model.
- `--timeout-sec <int>` on the standalone CLI (`python3
  scripts/launch_modal_eval.py …`).
- Environment variables consumed at module import time (used by
  `modal run -d … main_entry`):
    - `SPECTRALQUANT_MODAL_TIMEOUT_SEC` (global override).
    - `SPECTRALQUANT_MODAL_TIMEOUT_<FAMILY>_SEC` (per-family, takes
      precedence over the global override).
- A hard ceiling of `MAX_TIMEOUT_SEC = 86400` (24 h). Anything larger
  raises a `ValueError` rather than silently passing through.
- The `main_entry` log line on each spawn now prints the resolved
  Modal kill-switch and its source so the operator can verify the
  ceiling before walking away.

**What this does NOT change.** Cost. Modal bills on actual GPU
minutes, not on the kill-switch cap. Raising the timeout is purely a
ceiling that prevents a healthy long-running paper-valid LongBench
job from being killed prematurely. The cost ceiling is set entirely
by the harness CLI flags (`--n-per-task`, `--max-new-tokens`,
`--n-calib`, `--lloyd-max-iter`, `--calib-max-seq-tokens`) plus
`--methods`. **None of those defaults were weakened**; the
paper-valid gates in `experiments/run_longbench.py` remain
authoritative.

**Recommended relaunch (current commit).**

```bash
# Paper-valid LongBench, deterministic subset, 6 h Modal kill-switch.
modal run -d scripts/launch_modal_eval.py::main_entry \
  --family longbench \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 --methods "fp16,spectralquant_v2,turboquant" \
  --force \
  --extra "--subset deterministic --n-per-task 50 --max-input-tokens 8192 --max-new-tokens 128 --calibration-mode paper --n-calib 16 --lloyd-max-iter 200 --calib-max-seq-tokens 512"
```

If the relaunch is for a model that took >6 h previously, raise the
ceiling explicitly:

```bash
SPECTRALQUANT_MODAL_TIMEOUT_LONGBENCH_SEC=43200 \
  modal run -d scripts/launch_modal_eval.py::main_entry \
  --family longbench \
  …  # same flags as above
```

Polling continues to use the paths in §7.7.4a/§7.7.4b; the spawn log
now also includes the resolved timeout so a stalled run can be
distinguished from a kill-switched one by the gap between
`status.json` mtime and the timeout boundary.

### 7.7.4cc. Per-method partial persistence and the recovery merger

**Why this exists.** `experiments/run_longbench.py` in earlier commits
held every method's result record (fp16, spectralquant_v2, turboquant)
in memory until *all three* methods had finished, and then wrote a
single canonical schema-validated JSON with `paper_valid=true`. If the
final method timed out — for example, TurboQuant generation hitting the
6 h Modal kill-switch after fp16 + SQv2 had already each cost an hour
or more — the in-memory records of the earlier methods were lost; the
on-disk per-(method, task) progress shards under
`status/longbench/.../partial/{method}__{task}.json` only carried
completion counts and `last_completion_chars`, NOT the actual
predictions or per-task scores. The completed-but-discarded SQv2 data
in such a run was the most expensive single piece of compute in the
v2 program and represented a real loss-of-work risk.

**Patch.** `_write_method_partial_record(status, method=, record=)`
in `experiments/run_longbench.py` writes the in-memory `methods[m]`
record to a per-method shard
`<status_dir>/partial/method__<m>.json` immediately after each
method's full evaluation finishes. Shards always carry
`paper_valid: false, partial: true, kind:
"longbench_partial_method_record"`; the canonical schema-validated
JSON is still produced only at the end of a successful three-method
run.

**Recovery.** `scripts/merge_longbench_partials.py` recombines the
per-method shards into a single JSON. The merger is conservative by
default:

- `paper_valid=false, partial=true, mode=full_partial` if not all of
  `{fp16, spectralquant_v2, turboquant}` are present.
- `paper_valid=true, partial=false, mode=full` ONLY if all three
  methods are present AND the operator passes `--paper-valid`.

The merged JSON does NOT go through the canonical longbench schema
validator — it is explicitly a recovery artifact. Any paper sentence
cited from a merged file MUST disclose that the file was merged from
per-method shards from a partially-killed run.

```bash
# Pull the partial shards locally:
modal volume get spectralquant-v2-results \
  /status/longbench/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant/partial/ \
  /tmp/lb_partial/ --force

# Recombine. Without --paper-valid, the merged JSON will be tagged
# paper_valid=false, partial=true, mode=full_partial:
python3 scripts/merge_longbench_partials.py \
  --partial-dir /tmp/lb_partial/partial \
  --run-id longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant \
  --model Qwen/Qwen2.5-7B \
  --avg-bits 3 \
  --output results/v3/modal/longbench__merged.json
```

Tests for the helper + merger live in
`tests/test_longbench_partial_persist.py`.

**Important property.** The patch only protects future runs (those
with a status directory mounted under
`spectralquant-v2-results:/results/status/longbench/...`). The
already-running job at app `ap-vTqL16w5Nmaw6s2oGc6czU`
(`commit 6154175`, launched `2026-04-30T10:10Z`) does NOT run the
patched harness; if its TurboQuant phase times out, the SQv2 method
record will be lost and a TurboQuant-only retry will need to re-run
SQv2 too. Future LongBench runs from the patched commit do not have
this risk.

### 7.7.4d. Validated evidence-family snapshot (2026-04-30)

As of commit `615417531da9e282a002698c17576e73bfd857a4` and the
following Modal pulls, three of the four next-stage evidence families
have **paper-valid** Modal artifacts on Qwen2.5-7B at b=3 (single seed
42, NVIDIA H200). The fourth (LongBench deterministic subset) is a
**live job in progress** with on-disk partial shards.

| Family | Status | Local path | Modal volume path | Headline |
|---|---|---|---|---|
| Perplexity | paper_valid=true | `results/v3/modal/perplexity__Qwen2.5-7B__b3_seed42_n1024_tok1024_str512_fp16+spectralquant_v2+turboquant.json` | `spectralquant-v2-results:/results/perplexity/...` | fp16=6.49, SQv2=6.98, TQ=2048.6 (104k tokens, wikitext-103-raw-v1 validation) |
| Generation | paper_valid=true | `results/v3/modal/generation__Qwen2.5-7B__b3_seed42_t0.00_new128_fp16+spectralquant_v2+turboquant.json` | `spectralquant-v2-results:/results/generation/...` | token_overlap_F1 vs FP16 self: SQv2=0.482, TQ=0.120; distinct-2: 0.768/0.603/0.301 |
| Latency | paper_valid=true | `results/v3/modal/latency__Qwen2.5-7B__b3_seed42_bs1_ctx512x1024x2048_gen64_wm3_it10_fp16+spectralquant_v2+turboquant.json` | `spectralquant-v2-results:/results/latency/...` | fp16 decode 17.66 ms/tok @ ctx=1024; SQv2 hooked-replay 630.6 ms/tok; SQv2 microbench 0.0593 ms/tok (×297 vs e2e) |
| LongBench (deterministic subset, 5 tasks × 50 ex) | in_progress | `results/v3/modal/longbench_partial/` | `spectralquant-v2-results:/results/status/longbench/...` | fp16 5/5 done; SQv2 narrativeqa done, qasper 20/50; TurboQuant pending. Final scored JSON not yet written. |

The replay-coverage `fraction_layers_real` for the non-fp16 methods is
1.0 (28/28 layers calibrated, 0 passthrough) on perplexity and
generation, satisfying the ≥0.99 paper_valid gate.

The hooked-replay end-to-end latency rows for SQv2 are **300×–10000×
slower than fp16** because each forward pass now runs Python-level
hooks per layer. This is the expected microbench/end-to-end gap and is
labeled in the JSON with `production_kernel=false`. The microbenchmark
rows show the K/V compress+decompress kernel itself runs at
0.06 ms/tok @ ctx=1024 for SQv2 — which is what a HF `Cache`-subclass
engine should approach in production.

### 7.7.5. What still needs to happen before any of these claims become safe

- The K/V projection-replay path (`experiments/sqv2_replay.py`) now
  produces *real* compressed-method evaluation for `spectralquant_v2`
  and the in-repo `turboquant` baseline across PPL, generation,
  LongBench (full HF dataset path), KV-microbenchmark latency, AND
  hooked-replay end-to-end latency. What it does NOT cover: weight
  quantization, official Google TurboQuant kernels, and a
  production-kernel v2 implementation.
- The full LongBench path uses `experiments/longbench_dataset.py` (HF
  `THUDM/LongBench` adapter with vendored prompt templates +
  middle-truncation) and `experiments/longbench_metrics.py` (in-repo
  re-implementations of the upstream metrics: token F1, ROUGE-L,
  classification EM, retrieval EM, count EM, code edit similarity).
  Subsets smaller than `full` produce paper-valid evidence for that
  *transparent subset*, with an explicit caveat in the JSON; do not
  headline a subset as full LongBench.
- For latency under v2, the harness now produces both the
  microbenchmark row AND a hooked-replay end-to-end row (see
  `measurement_kind=hooked_replay_end_to_end`). The end-to-end row is
  *real* end-to-end inference timing with K/V compression, but it
  carries `production_kernel=false` because the per-layer Python hooks
  add overhead. To produce a `production_kernel=true` row the v2
  engine still needs to land as a HF `Cache` subclass or a
  pre-attention rewrite that replaces the K/V cache without per-token
  Python overhead. Until that exists, downstream reports must call the
  v2 / TQ rows "hooked replay end-to-end latency", not "production
  speedup".
- For PPL, the gate requires `n_tokens >= 64_000` per the
  `paper_valid_n_tokens_threshold` field in the JSON's `data` block;
  the smoke commands intentionally fall well below this so they can
  run quickly. Real runs should configure `--n-eval-sequences` /
  `--max-eval-tokens` to comfortably clear the threshold.
- Restate every downstream paper sentence about the four families
  with the standard caveat (number of tokens, stride, seed,
  warmup/measured iters as applicable, and for non-FP16 rows: the
  K/V-only scope, the replay coverage fraction, and for latency the
  microbenchmark scope). See `docs/claims_discipline.md`.

---

## 8. Exact benchmark matrix

The tested grid for the Phase-6 paper run is intentionally small and listed
explicitly. Status, evidence ID, and the headline TQ vs SQ v2 attention-cosine
gap are sourced from `docs/full_matrix_evidence_summary.md` and the underlying
artifact JSONs.

| # | Model | avg_bits | n_calib | n_eval | n_layers_sample | seeds | Status | Evidence ID | TQ cos mean | SQ v2 cos mean |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|
| 1 | Mistral-7B-v0.3 | 5 | 32 | 8 | 8 | {42} | Done @ abcb091 | RUN-THREEWAY-MISTRAL-5BIT | 0.6556 | 0.9421 |
| 2 | Mistral-7B-v0.3 | 3 | 32 | 8 | 8 | {42} | Done @ abcb091 | RUN-THREEWAY-MISTRAL-3BIT | 0.6263 | 0.9327 |
| 3 | Mistral-7B-v0.3 | 2 | 32 | 8 | 8 | {42} | Done @ abcb091 | RUN-THREEWAY-MISTRAL-2BIT | 0.6495 | 0.9213 |
| 4 | Qwen2.5-7B      | 3 | 32 | 8 | 8 | {42} | Done @ abcb091 | RUN-THREEWAY-QWEN-3BIT    | 0.3986 | 0.7786 |

These four rows are paper-valid harness rows for the sliced configuration
(WikiText-103, n_calib=32, n_eval=8, 8 sampled layers, single seed). They are
not on their own a final paper claim — see
`docs/full_matrix_evidence_summary.md` §4 for the interpretation guardrails
that the paper draft must respect.

Extensions reserved for after the headline matrix is reproduced:

| # | Extension | Why | Status |
|---|---|---|---|
| 5 | WikiText-2 / C4 perplexity at b=3 | Unblock V1-GAP-004b "compression-neutral PPL" claim. | TODO |
| 6 | NIAH at 4k / 8k / 16k / 32k | Unblock V1-GAP-009 "NIAH 10/10". | TODO |
| 7 | LongBench n ≥ 50 / task | Unblock V1-GAP-008 "LongBench improvement". | TODO |
| 8 | RULER if feasible | Long-context corroboration. | TODO |
| 9 | End-to-end decode latency, BS={1,4,8,16} | Unblock V1-GAP-003 latency claim. | TODO |
| 10 | Per-stage latency breakdown (compress / score / decompress / decode) | Same. | TODO |
| 11 | Calibration time and amortization curves | Unblock V1-GAP-007 "15-second calibration". | TODO |
| 12 | Three independent calibration seeds | Calibration stability replaces V1-GAP-005. | TODO |
| 13 | ≥ 5 (preferably 10) seeds for the Mistral 3-bit row | Unblock V1-GAP-001 statistics. | TODO |
| 14 | Official Google TurboQuant comparison | Unblock V1-GAP-012. | TODO |

---

## 9. Target numbers from Anirudh's report (TARGETS, not verified)

Every number in this section is a **target reproduction value**, not a result
produced by this repo. They are reproduced here exclusively so that the
Phase-6 runbook has something to compare against. They must not be quoted in
any v2 figure caption, README sentence, or paper paragraph until they have
been re-derived from a JSON under `results/three_way/` that validates against
`schemas/three_way_result.schema.json`.

### 9.1 Headline three-way table (target)

| Model | Bits | TQ Compression | SQ v2 Compression | TQ Attn Cos | SQ v2 Attn Cos | SQ v2 − TQ | SQ v2 − v1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mistral-7B-v0.3 | 5 | 3.08x | 3.07x | 0.9904 | 0.9938 | +0.34 pp | −0.001 pp |
| Mistral-7B-v0.3 | 3 | 5.02x | 5.95x | 0.8975 | 0.9374 | +3.98 pp | +0.10 pp |
| Mistral-7B-v0.3 | 2 | 8.00x | 9.50x | 0.7108 | 0.8206 | +10.98 pp | +0.54 pp |
| Qwen2.5-7B | 3 | 5.02x | 5.95x | 0.7465 | 0.8427 | +9.62 pp | +0.78 pp |

### 9.2 Mistral attention-output cosine summary (target)

| Bits | TQ mean | TQ std | TQ min | TQ max | SQ v2 mean | SQ v2 std | SQ v2 min | SQ v2 max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.9904 | (target) | (target) | (target) | 0.9938 | (target) | (target) | (target) |
| 3 | 0.8975 | (target) | (target) | (target) | 0.9374 | (target) | (target) | (target) |
| 2 | 0.7108 | (target) | (target) | (target) | 0.8206 | (target) | (target) | (target) |

The std/min/max columns are filled from the report on first encounter;
otherwise they are written as `(target)` so an operator cannot accidentally
quote a number that did not exist in the source.

### 9.3 Qwen attention-output cosine summary (target)

| Bits | TQ mean | TQ std | TQ min | TQ max | SQ v2 mean | SQ v2 std | SQ v2 min | SQ v2 max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.7465 | (target) | (target) | (target) | 0.8427 | (target) | (target) | (target) |

### 9.4 Per-layer breakdown (target)

The Phase-6 JSON must include `methods.<m>.per_layer[]` with `layer_idx`,
`cosine_mean`, `cosine_std`, `cosine_min`, `cosine_max`. Anirudh's report
target is roughly that v2 strictly Pareto-improves over the local TQ baseline
on every sampled layer of Mistral-7B-v0.3 at b=3, and that the gap widens at
b=2. These are TARGETS until a JSON exists.

### 9.5 d_eff statistics (target)

| Model | d_eff mean | d_eff min | d_eff max | d_eff/head_dim |
|---|---:|---:|---:|---:|
| Mistral-7B-v0.3 | ≈ 2.76 | 2 | 7 | ≈ 2–6 % |
| Qwen2.5-7B | (target) | (target) | (target) | ≈ 3–4 % |

The "≈ 3–4 %" framing is bound by V1-GAP-004 (two d_eff conventions
coexist). Any v2 paper sentence must specify the convention
(*normalized keys, participation ratio, ceil*).

### 9.6 Pareto table (target)

A target Pareto curve for Mistral-7B-v0.3 should show, at matched attention
cosine, that v2 reaches a higher compression ratio than the local TQ baseline.
Until the JSONs exist, the Pareto entries are TARGETS only.

---

## 10. Paper / report plan

### 10.1 Title and authors

- **Title (working).** *3% Is All You Need, Revisited: Water-filling the
  Semantic Subspace for Production-Model KV-Cache Compression*.
- **Authors.** Anirudh B. Vangara and Ashwin Gopinath.

### 10.2 Thesis (one sentence)

> KV-cache compression should be spectrum-aware: SpectralQuant v1 finds the
> tiny semantic subspace, and SpectralQuant v2 shows that the spectrum inside
> that subspace governs how the bit budget should be spent.

### 10.3 Section outline

1. Abstract.
2. Introduction — *3% is all you need, revisited*.
3. Background — KV cache, TurboQuant, QJL, spectral structure.
4. SpectralQuant v1 — finding the semantic subspace.
5. SpectralQuant v2 — water-filling the semantic subspace (algorithm, math,
   pseudocode, and the per-dim codebook construction).
6. Experimental methodology — models, datasets, calibration, layer sampling,
   metrics.
7. Results — three-way headline, Pareto, per-layer, d_eff stats.
8. Why water-filling helps — variance accounting; toy two-dimensional plot.
9. Failure modes and caveats — every V1-GAP that bears on a paper claim.
10. Reproducibility — `docs/reproduction.md` walkthrough, `experiments/run_three_way.py`.
11. Claims discipline — direct pointer to `docs/claims_discipline.md`.
12. Conclusion.

### 10.4 Figures and tables to generate

| ID | Source script | Source JSON(s) | Figure / table file |
|---|---|---|---|
| F1 | `plot_three_way.py` | `results/three_way/*.json` | `paper_output/v2/figures/headline_threeway.pdf` |
| F2 | `plot_three_way.py` | same | `paper_output/v2/figures/per_layer_cosine.pdf` |
| F3 | `plot_deff_stats.py` | `results/deff_stats/*.json` | `paper_output/v2/figures/deff_distribution.pdf` |
| F4 | `plot_three_way.py --pareto` | same | `paper_output/v2/figures/pareto.pdf` |
| F5 | `plot_waterfill_intuition.py` | `results/waterfill_ablation/*.json` | `paper_output/v2/figures/waterfill_intuition.pdf` |
| T1 | `make_table_headline.py` | `results/three_way/*.json` | `paper_output/v2/tables/headline.tex` |
| T2 | `make_table_perlayer.py` | same | `paper_output/v2/tables/per_layer_mistral_b3.tex` |
| T3 | `make_table_accounting.py` | `results/accounting_audit/*.json` | `paper_output/v2/tables/accounting.tex` |

No figure or table is permitted in the LaTeX source unless its source JSON
exists in the repo.

### 10.5 Claim-to-evidence mapping (excerpt)

| Claim | Evidence IDs |
|---|---|
| "v2 attention-output cosine exceeds the local TurboQuant baseline on Mistral-7B-v0.3 at 2/3/5-bit." | `RUN-THREEWAY-MISTRAL-2BIT`, `RUN-THREEWAY-MISTRAL-3BIT`, `RUN-THREEWAY-MISTRAL-5BIT` |
| "v2 attention-output cosine exceeds the local TurboQuant baseline on Qwen2.5-7B at 3-bit." | `RUN-THREEWAY-QWEN-3BIT` |
| "v2 exceeds v1 at the same operating points." | same JSONs |
| "At matched cosine, v2 reaches a higher compression ratio than the local TQ baseline." | `RUN-ACCOUNTING-AUDIT-001` |
| "Calibration uses n_calib=32, max_tokens=384 from WikiText-103." | spec §13.3 + JSONs |
| "v1 → v2 is backward-compatible (`use_water_fill=False`)." | `tests/test_v2_quantization.py::test_uniform_when_water_fill_disabled` |

The full mapping must be appended to the paper as an appendix table generated
from `docs/evidence_catalog.json`.

### 10.6 What goes in the paper vs. what is held back

**In the paper (after Phase 6 + accounting reconciliation):**

- The headline three-way table for the four operating points in §8.
- Per-layer cosine distributions for Mistral b=3 and b=2.
- d_eff distribution under the *normalized keys, participation ratio, ceil*
  convention only.
- Compression ratios derived from `accounting.py`, with the formula version
  spelled out.
- Backward compatibility statement bounded by the unit tests.
- Failure modes section that lists every V1-GAP that the paper does not lift.

**Held back until further runs (§8 rows 5–14):**

- Perplexity and LongBench / NIAH / RULER claims.
- End-to-end latency / throughput claims.
- "Universal" architecture claim.
- "Beats official Google TurboQuant" claim.
- Multi-seed CI and Wilcoxon p-values until ≥ 5 (ideally ≥ 10) Mistral 3-bit
  seeds exist.

### 10.7 Citation / evidence rules

- Every empirical sentence cites at least one evidence ID.
- A number lifted from a JSON file must match the JSON within tolerance
  (float-comparison precision documented in `docs/result_schema.md`).
- The TurboQuant baseline is always introduced as "the local TurboQuant
  reimplementation" until V1-GAP-012 is resolved.
- The d_eff convention is stated every time d_eff is mentioned.
- "TurboQuant has catastrophic low-bit head failures" is allowed only if the
  per-head min-cosine distribution is included in the supporting JSON.

### 10.8 Eight reflection passes (per spec §17)

1. **Claim audit.** Every empirical sentence has an evidence ID.
2. **Math audit.** Every formula's notation is consistent with `accounting.py`
   and `waterfill.py`; spec §10's appendix formula is annotated as the source
   of V1-GAP-014.
3. **Compression-accounting audit.** Every ratio derives from
   `accounting.CompressionAccounting`; the 5.95x is reconciled or footnoted.
4. **Baseline audit.** TurboQuant labeled "local" everywhere; official
   comparison removed unless V1-GAP-012 resolves.
5. **Reproducibility audit.** Run every command in `docs/reproduction.md`
   from a fresh Modal image; record outcomes in this audit doc.
6. **Statistical audit.** Seed counts, sample counts, and CIs honest;
   min/max claims tied to the per-layer / per-head JSON arrays.
7. **Reader audit.** Method explanation readable by a strong non-specialist.
8. **Claims-discipline audit.** Cross-check every paper sentence against
   `docs/claims_discipline.md` §1–§3.

The outcome of each pass is appended back into this audit document in §11.

---

## 11. Final checklist

Compact view of every blocker. `Status` is one of `done`, `in-progress`,
`pending`. `Blocker` calls out what is preventing `pending` items.

| Item | Owner / agent | File / artifact | Command | Status | Blocker |
|---|---|---|---|---|---|
| v2 spec committed | author | `docs/spectralquant_v2_technical_spec.md` | n/a | done | — |
| Evidence catalog (md + json) | code-agent | `docs/evidence_catalog.{md,json}` | n/a | done | — |
| Claims discipline | code-agent | `docs/claims_discipline.md` | n/a | done | — |
| Result schema doc | code-agent | `docs/result_schema.md` | n/a | done | — |
| JSON schemas | code-agent | `schemas/*.json` | n/a | done | — |
| Schema test | code-agent | `tests/test_result_schema.py` | `pytest tests/test_result_schema.py -q` | done | — |
| Water-filling module | code-agent | `src/spectralquant/waterfill.py` | n/a | done | — |
| Water-filling tests | code-agent | `tests/test_waterfill.py` | `pytest tests/test_waterfill.py -q` | done | — |
| Accounting module