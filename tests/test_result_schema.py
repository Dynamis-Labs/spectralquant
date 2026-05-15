"""Schema validation tests for SpectralQuant v2.

Phase 1 scope (this file):

1. Each schema file in `schemas/` is itself valid JSON.
2. `docs/evidence_catalog.json` parses and validates against
   `schemas/evidence_catalog.schema.json`.
3. The schemas themselves are valid JSON Schema documents (when `jsonschema`
   is installed).

When `jsonschema` is not available, tests that need it skip with a clear
message. The pure-stdlib parsing checks still run, so a syntactic regression
in any schema or in the catalog is caught regardless of dependency state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "schemas"
DOCS_DIR = REPO_ROOT / "docs"

EVIDENCE_CATALOG_PATH = DOCS_DIR / "evidence_catalog.json"
EVIDENCE_SCHEMA_PATH = SCHEMAS_DIR / "evidence_catalog.schema.json"
ACCOUNTING_SCHEMA_PATH = SCHEMAS_DIR / "accounting.schema.json"
THREE_WAY_SCHEMA_PATH = SCHEMAS_DIR / "three_way_result.schema.json"
PERPLEXITY_SCHEMA_PATH = SCHEMAS_DIR / "perplexity.schema.json"
LONGBENCH_SCHEMA_PATH = SCHEMAS_DIR / "longbench.schema.json"
GENERATION_SCHEMA_PATH = SCHEMAS_DIR / "generation.schema.json"
LATENCY_SCHEMA_PATH = SCHEMAS_DIR / "latency.schema.json"

ALL_SCHEMAS = [
    EVIDENCE_SCHEMA_PATH,
    ACCOUNTING_SCHEMA_PATH,
    THREE_WAY_SCHEMA_PATH,
    PERPLEXITY_SCHEMA_PATH,
    LONGBENCH_SCHEMA_PATH,
    GENERATION_SCHEMA_PATH,
    LATENCY_SCHEMA_PATH,
]


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_schemas_directory_exists() -> None:
    assert SCHEMAS_DIR.is_dir(), f"missing schemas directory: {SCHEMAS_DIR}"


@pytest.mark.parametrize("schema_path", ALL_SCHEMAS, ids=lambda p: p.name)
def test_schema_file_is_valid_json(schema_path: Path) -> None:
    """Every schema file must parse as JSON. No jsonschema dependency needed."""
    assert schema_path.is_file(), f"missing schema file: {schema_path}"
    obj = _load_json(schema_path)
    assert isinstance(obj, dict), f"{schema_path.name} must be a JSON object"
    assert obj.get("$schema"), f"{schema_path.name} must declare $schema"
    assert obj.get("title"), f"{schema_path.name} must declare a title"


def test_evidence_catalog_file_is_valid_json() -> None:
    assert EVIDENCE_CATALOG_PATH.is_file(), (
        f"missing evidence catalog: {EVIDENCE_CATALOG_PATH}"
    )
    obj = _load_json(EVIDENCE_CATALOG_PATH)
    assert isinstance(obj, dict)
    assert "entries" in obj and isinstance(obj["entries"], list) and obj["entries"]


def test_evidence_catalog_ids_are_unique() -> None:
    """Independent of jsonschema: catalog entry IDs and gap IDs must each be unique."""
    catalog = _load_json(EVIDENCE_CATALOG_PATH)
    entry_ids = [e["id"] for e in catalog["entries"]]
    gap_ids = [g["id"] for g in catalog.get("gaps", [])]
    assert len(entry_ids) == len(set(entry_ids)), (
        f"duplicate entry ids: {sorted({i for i in entry_ids if entry_ids.count(i) > 1})}"
    )
    assert len(gap_ids) == len(set(gap_ids)), (
        f"duplicate gap ids: {sorted({i for i in gap_ids if gap_ids.count(i) > 1})}"
    )


def test_evidence_catalog_paths_exist() -> None:
    """Every cataloged in-repo path must exist on disk.

    Off-repo artifacts (Modal volume entries of the form ``<volume>:/path``,
    URLs, or anything otherwise marked as ``modal_run_artifact``) are skipped:
    they live on a Modal volume, not in the working tree, and the assertion
    only governs the in-repo evidence.
    """
    catalog = _load_json(EVIDENCE_CATALOG_PATH)
    missing: list[str] = []
    for entry in catalog["entries"]:
        path = entry["path"]
        # Off-repo artifact references (Modal volume paths use a "vol:/path"
        # prefix; URLs are obvious; modal_run_artifact entries are off-repo
        # by construction).
        if entry.get("kind") == "modal_run_artifact":
            continue
        if path.startswith(("http://", "https://")):
            continue
        if ":" in path.split("/", 1)[0]:
            continue
        p = REPO_ROOT / path
        if not p.exists():
            missing.append(f"{entry['id']} -> {path}")
    assert not missing, "cataloged paths missing on disk:\n  " + "\n  ".join(missing)


# --- jsonschema-backed validation ---------------------------------------------------

jsonschema = pytest.importorskip(
    "jsonschema",
    reason=(
        "jsonschema is not installed in this environment. "
        "Install via `pip install jsonschema` (it is also listed under the "
        "[dev] optional dependencies in pyproject.toml)."
    ),
)


def test_evidence_schema_is_valid_jsonschema() -> None:
    schema = _load_json(EVIDENCE_SCHEMA_PATH)
    # Will raise SchemaError if the schema itself is malformed.
    jsonschema.Draft202012Validator.check_schema(schema)


def test_accounting_schema_is_valid_jsonschema() -> None:
    schema = _load_json(ACCOUNTING_SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)


def test_three_way_schema_is_valid_jsonschema() -> None:
    schema = _load_json(THREE_WAY_SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)


def test_evidence_catalog_validates_against_schema() -> None:
    schema = _load_json(EVIDENCE_SCHEMA_PATH)
    catalog = _load_json(EVIDENCE_CATALOG_PATH)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(catalog), key=lambda e: list(e.absolute_path))
    if errors:
        msg = "\n".join(
            f"  - at {list(e.absolute_path) or '<root>'}: {e.message}" for e in errors
        )
        pytest.fail(f"evidence_catalog.json does not validate:\n{msg}")
