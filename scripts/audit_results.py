#!/usr/bin/env python3
"""Audit which expected SpectralQuant v2 result artifacts are present on disk.

Read-only: never starts any experiment. Reads a manifest of expected
artifact paths and prints a present/missing summary.

The manifest is a JSON file. By default this script uses the built-in
manifest derived from `docs/execution_audit_and_modal_runbook.md` §8
(the headline benchmark matrix) and §10.4 (figures and tables).

Usage:
    python3 scripts/audit_results.py
    python3 scripts/audit_results.py --manifest docs/expected_artifacts.json
    python3 scripts/audit_results.py --json
    python3 scripts/audit_results.py --strict   # exit 1 if any missing

Exit codes:
    0  audit ran (no missing required artifacts in --strict)
    1  --strict was set and at least one required artifact is missing
    2  invocation error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Default manifest derived from docs/execution_audit_and_modal_runbook.md.
# Paths are relative to the repo root. "required" means the artifact is
# part of the headline matrix and absence blocks the v2 paper draft.
DEFAULT_MANIFEST: dict[str, list[dict]] = {
    "three_way_runs": [
        # Filenames mirror experiments/run_three_way.py's _output_path() helper:
        # "<model_basename>_b<bits>_calib<n_calib>_eval<n_eval>_seed<seed>.json".
        # The corresponding catalogued artifacts (off-repo, on the Modal volume
        # ``spectralquant-v2-results``) carry evidence IDs RUN-THREEWAY-* in
        # docs/evidence_catalog.{md,json}; see docs/full_matrix_evidence_summary.md.
        {
            "path": "results/three_way/Mistral-7B-v0.3_b5_calib32_eval8_seed42.json",
            "required": True,
            "label": "Mistral-7B-v0.3 b=5 (RUN-THREEWAY-MISTRAL-5BIT)",
        },
        {
            "path": "results/three_way/Mistral-7B-v0.3_b3_calib32_eval8_seed42.json",
            "required": True,
            "label": "Mistral-7B-v0.3 b=3 (RUN-THREEWAY-MISTRAL-3BIT)",
        },
        {
            "path": "results/three_way/Mistral-7B-v0.3_b2_calib32_eval8_seed42.json",
            "required": True,
            "label": "Mistral-7B-v0.3 b=2 (RUN-THREEWAY-MISTRAL-2BIT)",
        },
        {
            "path": "results/three_way/Qwen2.5-7B_b3_calib32_eval8_seed42.json",
            "required": True,
            "label": "Qwen2.5-7B b=3 (RUN-THREEWAY-QWEN-3BIT)",
        },
    ],
    "accounting_audit": [
        {
            "path": "results/accounting_audit/turboquant_b3.json",
            "required": True,
            "label": "TQ accounting b=3",
        },
        {
            "path": "results/accounting_audit/spectralquant_v2_b3_d3.json",
            "required": True,
            "label": "SQ v2 accounting b=3 d_eff=3",
        },
    ],
    "deff_stats": [
        {
            "path": "results/deff_stats/mistral7b_v03.json",
            "required": False,
            "label": "Mistral d_eff distribution",
        },
        {
            "path": "results/deff_stats/qwen25_7b.json",
            "required": False,
            "label": "Qwen d_eff distribution",
        },
    ],
    "waterfill_ablation": [
        {
            "path": "results/waterfill_ablation/mistral7b_v03_b3.json",
            "required": False,
            "label": "Waterfill ablation Mistral b=3",
        },
    ],
    "calibration_stability": [
        {
            "path": "results/calibration_stability_v2/mistral7b_v03.json",
            "required": False,
            "label": "Calibration stability Mistral",
        },
    ],
    "figures": [
        {
            "path": "paper_output/v2/figures/headline_threeway.pdf",
            "required": False,
            "label": "F1 headline three-way",
        },
        {
            "path": "paper_output/v2/figures/per_layer_cosine.pdf",
            "required": False,
            "label": "F2 per-layer cosine",
        },
        {
            "path": "paper_output/v2/figures/deff_distribution.pdf",
            "required": False,
            "label": "F3 d_eff distribution",
        },
        {
            "path": "paper_output/v2/figures/pareto.pdf",
            "required": False,
            "label": "F4 Pareto",
        },
        {
            "path": "paper_output/v2/figures/waterfill_intuition.pdf",
            "required": False,
            "label": "F5 waterfill intuition",
        },
    ],
    "tables": [
        {
            "path": "paper_output/v2/tables/headline.tex",
            "required": False,
            "label": "T1 headline table",
        },
        {
            "path": "paper_output/v2/tables/per_layer_mistral_b3.tex",
            "required": False,
            "label": "T2 per-layer table",
        },
        {
            "path": "paper_output/v2/tables/accounting.tex",
            "required": False,
            "label": "T3 accounting table",
        },
    ],
    "next_stage_evidence_families": [
        # Paper-valid Modal artifacts pulled into the repo for the four
        # next-stage families. Authoritative copy lives on the Modal
        # volume ``spectralquant-v2-results``; the local copy is the
        # one this audit checks. See
        # docs/evidence_family_validation_2026-04-30.md.
        {
            "path": "results/v3/modal/perplexity__Qwen2.5-7B__b3_seed42_n1024_tok1024_str512_fp16+spectralquant_v2+turboquant.json",
            "required": False,
            "label": "Perplexity Qwen2.5-7B b=3 seed=42 (RUN-PERPLEXITY-QWEN2.5-7B)",
        },
        {
            "path": "results/v3/modal/generation__Qwen2.5-7B__b3_seed42_t0.00_new128_fp16+spectralquant_v2+turboquant.json",
            "required": False,
            "label": "Generation Qwen2.5-7B b=3 seed=42 (RUN-GENERATION-QWEN2.5-7B)",
        },
        {
            "path": "results/v3/modal/latency__Qwen2.5-7B__b3_seed42_bs1_ctx512x1024x2048_gen64_wm3_it10_fp16+spectralquant_v2+turboquant.json",
            "required": False,
            "label": "Latency Qwen2.5-7B b=3 seed=42 (RUN-LATENCY-QWEN2.5-7B)",
        },
        {
            "path": "results/v3/modal/longbench_partial/status.json",
            "required": False,
            "label": "LongBench Qwen2.5-7B b=3 seed=42 partial run-level status (RUN-LONGBENCH-QWEN2.5-7B-DETERMINISTIC-PARTIAL)",
        },
        {
            "path": "results/v3/modal/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant.json",
            "required": False,
            "label": "LongBench Qwen2.5-7B b=3 seed=42 deterministic n=50 paper-valid canonical (RUN-LONGBENCH-QWEN2.5-7B-DETERMINISTIC)",
        },
    ],
    "validation_docs": [
        {
            "path": "docs/evidence_family_validation_2026-04-30.md",
            "required": False,
            "label": "Four-family evidence validation snapshot (2026-04-30)",
        },
        {
            "path": "scripts/merge_longbench_partials.py",
            "required": False,
            "label": "LongBench per-method shard merger (recovery from partial runs)",
        },
    ],
}


#: Result subdirectories to scan for stale ``.tmp_*`` files when
#: ``--scan-tmp`` is on. ``atomic_write_json`` in run_three_way.py creates
#: a ``.tmp_*`` file in the same directory as the target JSON and renames
#: it on success; if the process is killed mid-write (Modal timeout, OOM,
#: Ctrl-C) the tempfile can be left behind. We report — but never delete
#: by default — so the operator can decide whether the run is worth
#: salvaging.
DEFAULT_TMP_SCAN_DIRS: tuple[str, ...] = (
    "results/three_way",
    "results/calibration",
    "results/calibration_stability_v2",
    "results/accounting_audit",
    "results/deff_stats",
    "results/waterfill_ablation",
)

#: Glob pattern matching tempfiles created by ``tempfile.mkstemp(prefix=".tmp_", ...)``
#: in ``experiments/run_three_way.py::atomic_write_json``.
TMP_GLOB = ".tmp_*"


@dataclass
class StaleTmpFile:
    """A leftover tempfile from a crashed atomic write."""
    path: str             # path relative to repo root
    size_bytes: int
    mtime: float          # unix epoch seconds


@dataclass
class Item:
    group: str
    path: str
    required: bool
    label: str
    present: bool = False
    size_bytes: int = 0


def scan_stale_tmp_files(
    repo_root: Path,
    subdirs: "tuple[str, ...] | list[str]" = DEFAULT_TMP_SCAN_DIRS,
) -> list[StaleTmpFile]:
    """Return any leftover ``.tmp_*`` files under each result subdir.

    These are produced by ``atomic_write_json`` when the process dies
    between ``tempfile.mkstemp`` and ``os.replace``. They are zero-byte
    or partially-written and never become part of a result set, but
    they consume volume space and (more importantly) are a *signal*
    that a run crashed mid-write — the operator should know.

    Read-only: no files are removed. Pair with ``--delete-stale-tmp``
    on the CLI to clean them up after review.
    """
    found: list[StaleTmpFile] = []
    for sub in subdirs:
        full = (repo_root / sub).resolve()
        if not full.is_dir():
            continue
        # Use rglob so nested run directories on the Modal volume are
        # also caught.
        for p in full.rglob(TMP_GLOB):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            try:
                rel = str(p.relative_to(repo_root))
            except ValueError:
                rel = str(p)
            found.append(
                StaleTmpFile(
                    path=rel,
                    size_bytes=int(st.st_size),
                    mtime=float(st.st_mtime),
                )
            )
    found.sort(key=lambda s: s.path)
    return found


@dataclass
class Audit:
    items: list[Item] = field(default_factory=list)
    stale_tmp: list[StaleTmpFile] = field(default_factory=list)

    def add(self, group: str, entry: dict, repo_root: Path) -> None:
        path = entry.get("path")
        if not path:
            raise ValueError(f"manifest entry in group {group!r} is missing 'path': {entry}")
        full = repo_root / path
        present = full.is_file()
        size = full.stat().st_size if present else 0
        self.items.append(
            Item(
                group=group,
                path=path,
                required=bool(entry.get("required", False)),
                label=str(entry.get("label", path)),
                present=present,
                size_bytes=size,
            )
        )

    def missing_required(self) -> list[Item]:
        return [i for i in self.items if i.required and not i.present]

    def render_table(self) -> str:
        rows: list[str] = []
        groups: dict[str, list[Item]] = {}
        for item in self.items:
            groups.setdefault(item.group, []).append(item)
        for group, items in groups.items():
            rows.append(f"## {group}")
            for it in items:
                tag = "[PRESENT]" if it.present else "[MISSING]"
                req = "(required)" if it.required else "(optional)"
                size = f" {it.size_bytes:,} B" if it.present else ""
                rows.append(f"  {tag} {req:11s} {it.path}  — {it.label}{size}")
            rows.append("")
        if self.stale_tmp:
            rows.append("## stale_tmp_files")
            rows.append(
                "  WARNING: leftover atomic-write tempfiles found. "
                "Each indicates a run that crashed before os.replace. "
                "Review (and delete with --delete-stale-tmp) after triage."
            )
            for s in self.stale_tmp:
                rows.append(
                    f"  [STALE]   {s.path}  size={s.size_bytes:,}B "
                    f"mtime={int(s.mtime)}"
                )
            rows.append("")
        return "\n".join(rows)

    def summary(self) -> str:
        n = len(self.items)
        n_present = sum(1 for i in self.items if i.present)
        n_required = sum(1 for i in self.items if i.required)
        n_missing_req = len(self.missing_required())
        stale_note = (
            f"; {len(self.stale_tmp)} stale .tmp_* file(s)"
            if self.stale_tmp else ""
        )
        return (
            f"summary: {n_present}/{n} artifacts present; "
            f"{n_required - n_missing_req}/{n_required} required artifacts present; "
            f"{n_missing_req} required missing" + stale_note
        )

    def to_dict(self) -> dict:
        return {
            "items": [
                {
                    "group": i.group,
                    "path": i.path,
                    "required": i.required,
                    "label": i.label,
                    "present": i.present,
                    "size_bytes": i.size_bytes,
                }
                for i in self.items
            ],
            "stale_tmp_files": [
                {
                    "path": s.path,
                    "size_bytes": s.size_bytes,
                    "mtime": s.mtime,
                }
                for s in self.stale_tmp
            ],
            "summary": {
                "total": len(self.items),
                "present": sum(1 for i in self.items if i.present),
                "required": sum(1 for i in self.items if i.required),
                "missing_required": [i.path for i in self.missing_required()],
                "stale_tmp_count": len(self.stale_tmp),
            },
        }


def load_manifest(path: str | None) -> dict[str, list[dict]]:
    if path is None:
        return DEFAULT_MANIFEST
    p = Path(path)
    with p.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"manifest at {path} must be a JSON object of group -> [entries]")
    return data


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return start.resolve()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit expected SpectralQuant v2 result artifacts.",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="path to a JSON manifest; defaults to the built-in manifest",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of a human table",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any required artifact is missing",
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help="override the repo root (defaults to script's repo root)",
    )
    p.add_argument(
        "--scan-tmp",
        action="store_true",
        default=True,
        dest="scan_tmp",
        help=(
            "scan result subdirectories for stale .tmp_* files left "
            "behind by crashed atomic writes (default on; report-only)"
        ),
    )
    p.add_argument(
        "--no-scan-tmp",
        action="store_false",
        dest="scan_tmp",
        help="disable the stale-tempfile scan",
    )
    p.add_argument(
        "--delete-stale-tmp",
        action="store_true",
        dest="delete_stale_tmp",
        help=(
            "DELETE the stale .tmp_* files found by --scan-tmp. "
            "Off by default — review the report first. Only removes "
            "regular files matching .tmp_* under known result dirs."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else find_repo_root(Path(__file__).parent)
    )

    try:
        manifest = load_manifest(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"failed to load manifest: {exc}", file=sys.stderr)
        return 2

    audit = Audit()
    for group, entries in manifest.items():
        if not isinstance(entries, list):
            print(f"manifest group {group!r} must be a list of entries", file=sys.stderr)
            return 2
        for entry in entries:
            audit.add(group, entry, repo_root)

    if args.scan_tmp:
        audit.stale_tmp = scan_stale_tmp_files(repo_root)
        if args.delete_stale_tmp and audit.stale_tmp:
            removed = 0
            for s in audit.stale_tmp:
                full = repo_root / s.path
                # Defensive: only delete regular files whose basename
                # matches the .tmp_* pattern, and only under repo_root.
                try:
                    full_resolved = full.resolve()
                except OSError:
                    continue
                if not full_resolved.is_file():
                    continue
                if not full_resolved.name.startswith(".tmp_"):
                    continue
                try:
                    full_resolved.relative_to(repo_root.resolve())
                except ValueError:
                    # Outside repo root — refuse.
                    continue
                try:
                    os.unlink(full_resolved)
                    removed += 1
                except OSError as exc:
                    print(
                        f"failed to delete {full_resolved}: {exc}",
                        file=sys.stderr,
                    )
            print(
                f"[audit_results] deleted {removed} stale .tmp_* file(s)",
                file=sys.stderr,
            )

    if args.json:
        print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    else:
        print(audit.render_table())
        print(audit.summary())

    if args.strict and audit.missing_required():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
