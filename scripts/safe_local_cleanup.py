#!/usr/bin/env python3
"""Safe local cleanup helper for SpectralQuant v2 workspaces.

Purpose
-------
Free local disk before launching a Modal sweep without using broad ``rm -rf``
globs. The script is intentionally conservative:

* Dry-run is the default. Nothing is deleted unless an explicit ``--delete-*``
  flag is passed AND ``--yes`` confirms the action.
* Only a fixed allow-list of caches/artifacts may be removed. Each must be
  selected by its own flag.
* The script refuses to delete any path that is, contains, or is contained by
  the active repository root (resolved via ``git rev-parse --show-toplevel``,
  with the script's own location as a fallback). This is the guard added after
  a broad cleanup command accidentally deleted an active local clone.

It never reads or echoes credentials, never touches ``.env`` files, never
exits non-zero in dry-run mode, and never recurses into the repo root.

Examples
--------
    # Just show disk usage and what *would* be deleted with each flag:
    python3 scripts/safe_local_cleanup.py

    # Actually delete the Hugging Face cache:
    python3 scripts/safe_local_cleanup.py --delete-hf-cache --yes

    # Delete temporary clones in /tmp (excluding the active repo path):
    python3 scripts/safe_local_cleanup.py --delete-temp-clones --yes

Exit codes
----------
    0  success (dry-run always returns 0 unless argv is malformed)
    1  one or more requested deletions were refused (e.g. would touch the repo)
    2  invocation error (bad CLI args)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Default candidate paths per category. These are well-known local caches that
# can be regenerated without losing source-controlled work.
HF_CACHE_DIRS: tuple[str, ...] = (
    "~/.cache/huggingface",
    "~/.cache/huggingface/hub",
    "~/.cache/huggingface/datasets",
)
PLAYWRIGHT_CACHE_DIRS: tuple[str, ...] = (
    "~/.cache/ms-playwright",
    "~/Library/Caches/ms-playwright",
)
TEMP_CLONE_PARENTS: tuple[str, ...] = (
    "/tmp",
    "/var/tmp",
)
TEMP_CLONE_PREFIXES: tuple[str, ...] = (
    "spectralquant",
    "claude_code_clone_",
    "tmp_clone_",
)
TORCH_CACHE_DIRS: tuple[str, ...] = (
    "~/.cache/torch",
    "~/.cache/torch_extensions",
)
PIP_CACHE_DIRS: tuple[str, ...] = (
    "~/.cache/pip",
)


@dataclass
class Plan:
    """An ordered list of (label, absolute_path) deletion candidates."""

    entries: list[tuple[str, Path]] = field(default_factory=list)
    refusals: list[tuple[str, Path, str]] = field(default_factory=list)

    def add(self, label: str, path: Path) -> None:
        self.entries.append((label, path))

    def refuse(self, label: str, path: Path, reason: str) -> None:
        self.refusals.append((label, path, reason))


# ---------- safety primitives ----------


def detect_repo_root(start: Path | None = None) -> Path:
    """Return the resolved repo root.

    Uses ``git rev-parse --show-toplevel`` first; falls back to the parent of
    this script if git isn't available or the script is run outside a repo.
    The result is always an absolute, resolved path (no symlinks).
    """
    here = (start or Path(__file__).resolve()).parent
    try:
        out = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        out = None
    if out and out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip()).resolve()
    # Fallback: assume scripts/ lives at <repo>/scripts/
    return here.parent.resolve() if here.name == "scripts" else here.resolve()


def is_under(path: Path, parent: Path) -> bool:
    """True if ``path`` is ``parent`` or lives anywhere beneath it.

    Both paths are resolved before comparison to defeat symlink games.
    """
    p = path.resolve()
    q = parent.resolve()
    if p == q:
        return True
    try:
        p.relative_to(q)
        return True
    except ValueError:
        return False


def path_conflicts_with_repo(candidate: Path, repo_root: Path) -> str | None:
    """Return a refusal reason if deleting ``candidate`` would touch the repo.

    A candidate conflicts if:
      * it equals the repo root,
      * it is an ancestor of the repo root (would remove the repo as a side-effect),
      * it is contained inside the repo root.

    Returns ``None`` when the candidate is safe to consider further.
    """
    cand = candidate.resolve()
    root = repo_root.resolve()
    if cand == root:
        return f"refuses to delete the active repo root ({root})"
    if is_under(root, cand):
        return f"refuses to delete {cand} because it contains the active repo root ({root})"
    if is_under(cand, root):
        return f"refuses to delete {cand} because it is inside the active repo ({root})"
    return None


def safe_to_delete(candidate: Path, repo_root: Path) -> tuple[bool, str]:
    """Combined safety check for a deletion candidate.

    Returns ``(ok, reason_if_not_ok)``. The reason is empty when ok.
    """
    if not candidate.exists():
        return False, f"path does not exist: {candidate}"
    reason = path_conflicts_with_repo(candidate, repo_root)
    if reason:
        return False, reason
    # Refuse deleting top-level system paths even if no repo overlap.
    forbidden = {Path("/"), Path.home(), Path("/tmp"), Path("/var/tmp"), Path("/usr"), Path("/etc")}
    cand = candidate.resolve()
    if cand in {p.resolve() for p in forbidden if p.exists()}:
        return False, f"refuses to delete protected system path: {cand}"
    return True, ""


# ---------- discovery ----------


def expand(paths: Iterable[str]) -> list[Path]:
    return [Path(os.path.expanduser(p)) for p in paths]


def discover_temp_clones(
    parents: Iterable[str],
    prefixes: Iterable[str],
    repo_root: Path,
) -> list[Path]:
    """Find temp clone directories under ``parents`` matching ``prefixes``.

    The active repo path is always excluded by exact resolved-path comparison.
    """
    found: list[Path] = []
    root_resolved = repo_root.resolve()
    for parent_str in parents:
        parent = Path(parent_str)
        if not parent.is_dir():
            continue
        try:
            children = list(parent.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            name = child.name
            if not any(name.startswith(pref) for pref in prefixes):
                continue
            try:
                if child.resolve() == root_resolved:
                    continue
            except OSError:
                continue
            found.append(child)
    return found


# ---------- reporting ----------


def report_disk_usage(targets: Iterable[Path]) -> str:
    lines: list[str] = []
    for target in targets:
        probe = target if target.exists() else target.parent
        if not probe.exists():
            lines.append(f"  {target}: (parent missing)")
            continue
        try:
            usage = shutil.disk_usage(probe)
        except OSError as exc:
            lines.append(f"  {target}: disk_usage failed ({exc})")
            continue
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        lines.append(f"  {probe}: free={free_gb:.1f} GB / total={total_gb:.1f} GB")
    return "\n".join(lines)


def dir_size_bytes(path: Path) -> int:
    """Best-effort recursive size; never raises."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path, followlinks=False):
            for f in files:
                fp = Path(root) / f
                try:
                    total += fp.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def render_plan(plan: Plan, dry_run: bool) -> str:
    lines: list[str] = []
    if plan.entries:
        header = "[dry-run] Would delete:" if dry_run else "Will delete:"
        lines.append(header)
        for label, path in plan.entries:
            size_mb = dir_size_bytes(path) / (1024 * 1024)
            lines.append(f"  - [{label}] {path} (~{size_mb:.1f} MiB)")
    else:
        lines.append("(no deletion candidates selected)")
    if plan.refusals:
        lines.append("")
        lines.append("Refused (will NOT be touched):")
        for label, path, reason in plan.refusals:
            lines.append(f"  - [{label}] {path}: {reason}")
    return "\n".join(lines)


# ---------- plan construction ----------


def build_plan(
    *,
    repo_root: Path,
    delete_hf: bool,
    delete_playwright: bool,
    delete_temp_clones: bool,
    delete_torch_cache: bool,
    delete_pip_cache: bool,
    extra_temp_parents: Iterable[str] = (),
    extra_temp_prefixes: Iterable[str] = (),
) -> Plan:
    plan = Plan()

    def add_candidates(label: str, candidates: Iterable[Path]) -> None:
        for cand in candidates:
            ok, reason = safe_to_delete(cand, repo_root)
            if ok:
                plan.add(label, cand.resolve())
            elif reason and not reason.startswith("path does not exist"):
                plan.refuse(label, cand, reason)

    if delete_hf:
        add_candidates("hf_cache", expand(HF_CACHE_DIRS))
    if delete_playwright:
        add_candidates("playwright_cache", expand(PLAYWRIGHT_CACHE_DIRS))
    if delete_torch_cache:
        add_candidates("torch_cache", expand(TORCH_CACHE_DIRS))
    if delete_pip_cache:
        add_candidates("pip_cache", expand(PIP_CACHE_DIRS))
    if delete_temp_clones:
        parents = list(TEMP_CLONE_PARENTS) + list(extra_temp_parents)
        prefixes = list(TEMP_CLONE_PREFIXES) + list(extra_temp_prefixes)
        clones = discover_temp_clones(parents, prefixes, repo_root)
        add_candidates("temp_clone", clones)

    # Deduplicate while preserving order. Also drop any entry whose path is
    # nested inside another planned entry: deleting the ancestor will remove
    # the descendant, and keeping both causes a spurious "path does not exist"
    # error when the second deletion runs. (Seen with ~/.cache/huggingface and
    # ~/.cache/huggingface/hub being listed together.)
    seen: set[Path] = set()
    deduped: list[tuple[str, Path]] = []
    for label, path in plan.entries:
        if path in seen:
            continue
        seen.add(path)
        deduped.append((label, path))

    # Sort by path depth so ancestors come first, then drop descendants of
    # any already-kept path. Stable so original relative ordering of unrelated
    # entries is preserved as much as possible; we then restore original order
    # for the kept set at the end.
    original_order = {path: i for i, (_, path) in enumerate(deduped)}
    by_depth = sorted(deduped, key=lambda lp: len(lp[1].parts))
    kept_paths: list[Path] = []
    kept_entries: list[tuple[str, Path]] = []
    for label, path in by_depth:
        if any(is_under(path, ancestor) and path != ancestor for ancestor in kept_paths):
            continue
        kept_paths.append(path)
        kept_entries.append((label, path))
    kept_entries.sort(key=lambda lp: original_order[lp[1]])
    plan.entries = kept_entries
    return plan


# ---------- execution ----------


def execute_plan(plan: Plan, repo_root: Path) -> tuple[int, list[str]]:
    """Delete planned entries. Returns (num_deleted, errors)."""
    deleted = 0
    errors: list[str] = []
    for label, path in plan.entries:
        # Re-check just before deletion — defense in depth against TOCTOU.
        ok, reason = safe_to_delete(path, repo_root)
        if not ok:
            errors.append(f"[{label}] {path}: {reason}")
            continue
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            deleted += 1
        except OSError as exc:
            errors.append(f"[{label}] {path}: {exc}")
    return deleted, errors


# ---------- CLI ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Safe local cleanup for SpectralQuant v2. "
            "Dry-run by default. Refuses to delete the active repo."
        ),
    )
    p.add_argument(
        "--delete-hf-cache",
        action="store_true",
        help="delete the local Hugging Face cache (~/.cache/huggingface)",
    )
    p.add_argument(
        "--delete-playwright-cache",
        action="store_true",
        help="delete the local Playwright browsers cache",
    )
    p.add_argument(
        "--delete-temp-clones",
        action="store_true",
        help="delete known temp-clone directories under /tmp and /var/tmp "
        "(active repo path is always excluded)",
    )
    p.add_argument(
        "--delete-torch-cache",
        action="store_true",
        help="delete the torch and torch_extensions caches",
    )
    p.add_argument(
        "--delete-pip-cache",
        action="store_true",
        help="delete the pip wheel/build cache",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="actually perform deletions; without this flag the run is dry",
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help="override the repo root used for the safety guard "
        "(defaults to git toplevel of this script)",
    )
    p.add_argument(
        "--extra-temp-parent",
        action="append",
        default=[],
        help="extra parent directory to scan for temp clones (repeatable)",
    )
    p.add_argument(
        "--extra-temp-prefix",
        action="append",
        default=[],
        help="extra directory-name prefix to consider a temp clone (repeatable)",
    )
    p.add_argument(
        "--min-disk-gb",
        type=float,
        default=0.0,
        help="exit non-zero if free disk at the repo root is below this threshold",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else detect_repo_root()
    print(f"repo root (protected): {repo_root}")
    print()
    print("Disk usage:")
    print(report_disk_usage([repo_root, Path.home(), Path("/tmp")]))
    print()

    any_flag = any(
        [
            args.delete_hf_cache,
            args.delete_playwright_cache,
            args.delete_temp_clones,
            args.delete_torch_cache,
            args.delete_pip_cache,
        ]
    )
    if not any_flag:
        print(
            "No --delete-* flag passed; nothing to plan. "
            "Pass one or more of: --delete-hf-cache, --delete-playwright-cache, "
            "--delete-temp-clones, --delete-torch-cache, --delete-pip-cache."
        )

    plan = build_plan(
        repo_root=repo_root,
        delete_hf=args.delete_hf_cache,
        delete_playwright=args.delete_playwright_cache,
        delete_temp_clones=args.delete_temp_clones,
        delete_torch_cache=args.delete_torch_cache,
        delete_pip_cache=args.delete_pip_cache,
        extra_temp_parents=args.extra_temp_parent,
        extra_temp_prefixes=args.extra_temp_prefix,
    )

    print(render_plan(plan, dry_run=not args.yes))
    print()

    exit_code = 0
    if args.yes and plan.entries:
        deleted, errors = execute_plan(plan, repo_root)
        print(f"Deleted {deleted} path(s).")
        if errors:
            print("Errors:")
            for line in errors:
                print(f"  - {line}")
            exit_code = 1
    elif args.yes and not plan.entries:
        print("Nothing to delete.")
    else:
        print("[dry-run] Pass --yes to actually delete; nothing was changed.")

    if args.min_disk_gb > 0.0:
        try:
            usage = shutil.disk_usage(repo_root)
            free_gb = usage.free / (1024**3)
            if free_gb < args.min_disk_gb:
                print(
                    f"FAIL: free disk at repo root is {free_gb:.1f} GB "
                    f"(< {args.min_disk_gb:.1f} GB)",
                    file=sys.stderr,
                )
                exit_code = max(exit_code, 1)
        except OSError as exc:
            print(f"warn: could not check repo-root disk usage: {exc}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
