# SpectralQuant v2 Result Schemas

This document is the human-readable specification of the JSON schemas that v2 result files must validate against. The machine-readable forms live in:

```
schemas/evidence_catalog.schema.json
schemas/three_way_result.schema.json
schemas/accounting.schema.json
schemas/perplexity.schema.json
schemas/longbench.schema.json
schemas/generation.schema.json
schemas/latency.schema.json
```

The last four schemas correspond to the next-stage evaluation harnesses
under `experiments/run_{perplexity,longbench,generation,latency}.py`
(see `docs/execution_audit_and_modal_runbook.md` §7.7). Each result JSON
they produce includes a top-level `family` field naming the schema,
`mode` ∈ {`full`, `synthetic_smoke`, `inline_corpus_smoke`, `dry_run`},
and a `paper_valid` boolean. Smoke and dry-run modes are NEVER
paper-valid; the invariant is enforced by `tests/test_eval_harnesses.py`.

Each schema is JSON Schema draft 2020-12. Schemas are intentionally practical: they capture the structural invariants the report depends on (model identity, calibration metadata, compression accounting, evidence linkage) without freezing every field that might be added in later phases. Required fields must be present and well-typed; additional informational fields are allowed unless explicitly noted.

The validation entrypoint is `tests/test_result_schema.py`. The test currently validates `docs/evidence_catalog.json`; future phases extend it to validate every JSON written under `results/three_way/`, `results/waterfill_ablation/`, and `results/accounting_audit/`.

The operational plan for producing those JSONs (Modal commands, model matrix, target reproduction numbers, and per-run artifacts) lives in `docs/execution_audit_and_modal_runbook.md`.

## 1. `evidence_catalog.schema.json`

Validates `docs/evidence_catalog.json`.

**Required top-level fields:**

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | Currently `"1"`. |
| `repo` | string | The catalog's owning repository. |
| `entries` | array | One entry per evidence artifact; cross-referenced by stable IDs. |
| `gaps` | array | One entry per known v1 gap; referenced from `caveats`. |

**`entries[]` required fields:**

| Field | Type | Notes |
|---|---|---|
| `id` | string | Pattern `^V1-(PAPER|README|IMPL|EXP|RESULT|FIG|CONF|TEST)-\d{3}[a-z]?$` or `V2-SPEC-001`. Stable across rewrites. |
| `kind` | string | One of `paper_source`, `paper_compiled`, `paper_bib`, `paper_figures_dir`, `figure_generator`, `readme_claims_table`, `code_module`, `experiment_script`, `result_json`, `figure`, `config`, `build`, `test`, `spec`. |
| `path` | string | Repo-relative path. |
| `supports` | array of string | What claims/artifacts this entry backs. |
| `caveats` | array of string | Known limits. Empty array is allowed; missing field is not. |

**`entries[]` optional fields:**

| Field | Type | Notes |
|---|---|---|
| `produces` | array of string | For experiment scripts, the on-disk artifacts produced. |
| `produced_by` | string | For figures, the script that produced them. |
| `extracted` | object | For result JSONs, numeric values copied from the on-disk file (the canonical citation). |
| `files` | array of string | For directory-kind entries, the contained files. |

**`gaps[]` required fields:**

| Field | Type | Notes |
|---|---|---|
| `id` | string | Pattern `^V1-GAP-\d{3}[a-z]?$`. |
| `summary` | string | One-paragraph plain description. |
| `blocks_v2_claims` | array of string | Specific v2 claims this gap blocks. |
| `resolution_options` | array of string | What v2 may do to resolve. |

## 2. `three_way_result.schema.json`

Validates the JSON produced by `experiments/run_three_way.py` (Phase 6, not implemented yet) and any other file that purports to record the three-way comparison.

**Required top-level fields (high level):**

- `run_id` (string) — must match the filename stem.
- `timestamp` (string, ISO 8601 UTC).
- `repo` (string) — origin URL or `niashwin/spectralquant-full` (older artifacts may carry the pre-rename value `niashwin/spectralquant-v2`; both are valid for traceability).
- `commit` (string) — full or short git SHA.
- `command` (string) — exact CLI invocation used.
- `model` (object) — `name`, `layers`, `q_heads`, `kv_heads`, `head_dim`, `gqa_ratio`.
- `hardware` (object) — `gpu`, `cuda`.
- `software` (object) — `python`, `torch`, `transformers`, `datasets`.
- `data` (object) — calibration corpus, n_calib, eval corpus, n_eval, max_calib_tokens, disjoint_eval flag.
- `calibration` (object) — `normalize_keys`, `key_space` (`pre_rope` or `post_rope`), `d_eff_method`, `d_eff_rounding`, `d_eff_min`, `d_eff_max`, optional `d_eff_stats`.
- `methods` (object) — keyed by `turboquant`, `spectralquant_v1`, `spectralquant_v2`. Each method object must include `attn_cosine_mean`, `compression_accounting` (object validating against `accounting.schema.json`), `per_layer` (array), and `evidence_ids` (array of strings).
- `evidence_ids` (array of strings) — top-level evidence IDs (e.g., `RUN-THREEWAY-MISTRAL-3BIT`).

**Notes on the `methods` map:**

- All three methods must be present in every produced file. A run that benchmarks only one method is not a "three-way result".
- Method-level `compression_accounting` must validate against `accounting.schema.json` (next section).
- `per_layer` array elements must include `layer_index`, `d_eff`, and per-method `attn_cosine_mean`. Other fields (per-head min, kappa, etc.) are allowed.

**Notes on `model.gqa_ratio`:** an integer (e.g., 4 for Mistral-7B-v0.3 with Q=32 / KV=8), validated against `q_heads / kv_heads`.

## 3. `accounting.schema.json`

Validates the per-method `compression_accounting` object inside a three-way result, and any standalone JSON written by `experiments/run_compression_accounting_audit.py` (Phase 5+, not implemented yet).

This schema mirrors the `CompressionAccounting` dataclass in spec §10:

| Field | Type | Notes |
|---|---|---|
| `method` | string | One of `turboquant`, `spectralquant_v1`, `spectralquant_v2`. |
| `avg_bits_arg` | integer | The bit budget passed on the CLI (2, 3, or 5 for the Phase 6 sweep). |
| `head_dim` | integer | Typically 128. |
| `d_eff` | integer or null | Null for TurboQuant (no semantic split). |
| `k_mse_bits` | number | Bits per coordinate spent on MSE quantization of K. |
| `k_qjl_bits` | number | Bits spent on QJL residual for K. |
| `k_norm_bits` | number | Bits stored for the K norm side-channel. |
| `v_mse_bits` | number | Bits per coordinate spent on MSE quantization of V. |
| `v_norm_bits` | number | Bits stored for the V norm side-channel. |
| `total_k_bits` | number | Total bits per K slot derived from the breakdown. |
| `total_v_bits` | number | Total bits per V slot derived from the breakdown. |
| `average_slot_bits` | number | Per the spec, `(total_k_bits + total_v_bits) / 2`. |
| `fp16_slot_bits` | number | Reference (typically 2048 for `head_dim`=128, K+V combined per slot). |
| `compression_ratio` | number | `fp16_slot_bits / average_slot_bits`. |
| `formula_version` | string | E.g., `v2.spec.s10`. Pinned to the spec section that defined the formula. |

**Invariants enforced by the schema:**

- All numeric fields ≥ 0.
- `total_k_bits` ≥ `k_mse_bits + k_qjl_bits + k_norm_bits` (the schema cannot enforce equality without arithmetic; that's a test, not a schema check).
- `compression_ratio` is the output of the accounting module — never a typed-in string. v1's "5.95×" was the original reason for adding this gate (V1-GAP-014).

The schema is deliberately permissive about which `bits` fields are nonzero per method:

- `turboquant`: `d_eff` is null; `k_qjl_bits` and `k_norm_bits` may be nonzero (full-dimensional QJL).
- `spectralquant_v1`: `d_eff` is an int ≥ 2; QJL is selective (`k_qjl_bits` typically smaller than TQ's).
- `spectralquant_v2`: same as v1 plus a per-method optional `waterfill_allocation` array (sum equals `b_high * d_eff`).

`waterfill_allocation` is not a required field at the schema level (the schema must validate v1 and TQ runs that do not set it), but Phase 5+ tests will require it whenever `method == spectralquant_v2` and `use_water_fill == true`.

## 4. Why these three schemas (and only these)

- `evidence_catalog.schema.json` is required NOW (Phase 1) to lock the catalog before v2 implementation begins.
- `three_way_result.schema.json` is required to land BEFORE Phase 6 produces JSONs (otherwise the runs cannot be schema-gated).
- `accounting.schema.json` is required to land BEFORE Phase 5 compression-accounting tests (spec §10).

Other potential schemas (waterfill ablation result, deff stats result, NIAH result) are deferred until the matching experiment scripts exist. Adding them now would be premature.

## 5. Test wiring

`tests/test_result_schema.py` performs three checks on every run:

1. Each schema file in `schemas/` is itself valid JSON.
2. `docs/evidence_catalog.json` parses and validates against `schemas/evidence_catalog.schema.json`.
3. If `jsonschema` is unavailable, tests skip with a clear message. Schema syntactic validity is still checked — that part needs only the standard library.

Phase 5–6 commits will extend the same test file to walk `results/three_way/*.json` and `results/accounting_audit/*.json` and assert each one validates.

## 6. Updating the schemas

Schemas are versioned by the `schema_version` field where present (currently only the evidence catalog has one). Backward-incompatible changes must:

1. Bump `schema_version`.
2. Update existing JSONs to match.
3. Update this document.
4. Update `tests/test_result_schema.py` to either continue accepting the old version (during a transition) or reject it.

Adding optional fields is not a breaking change.
