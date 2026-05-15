#!/usr/bin/env python3
"""Preflight checks for SpectralQuant v2 Modal runs.

Read-only: never starts a model run, never downloads a weight, never
spends GPU credit. Reports the *presence* of credentials but never their
values.

Exit codes:
    0  all required checks passed (warnings allowed)
    1  one or more required checks failed
    2  invocation error (bad CLI args, etc.)

Usage:
    python3 scripts/preflight_modal.py
    python3 scripts/preflight_modal.py --strict        # warnings -> failures
    python3 scripts/preflight_modal.py --allow-dirty   # tolerate dirty tree
    python3 scripts/preflight_modal.py --output-dir results/three_way
    python3 scripts/preflight_modal.py --min-disk-gb 10
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REQUIRED_ENV_VARS = (
    "HF_TOKEN",
)
OPTIONAL_ENV_VARS = (
    "HUGGING_FACE_HUB_TOKEN",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "WANDB_API_KEY",
)
SECRET_PATH_PATTERNS = (
    re.compile(r"(^|/)\.env(\..*)?$"),
    re.compile(r"(^|/)modal_token($|[._-])"),
    re.compile(r"(^|/)hf_token($|[._-])"),
    re.compile(r"(^|/)secrets($|/)"),
    re.compile(r"\.pem$"),
    re.compile(r"(^|/)id_rsa($|\.)"),
)
DEFAULT_RESULT_DIRS = (
    "results/three_way",
    "results/three_way_smoke",
    "results/accounting_audit",
    "results/waterfill_ablation",
    "results/deff_stats",
    "results/calibration_stability_v2",
    "results/report_figures",
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    severity: str = "error"  # "error" or "warn"


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, message: str, severity: str = "error") -> None:
        self.results.append(CheckResult(name, ok, message, severity))

    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if (not r.ok) and r.severity == "error"]

    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if (not r.ok) and r.severity == "warn"]

    def render(self) -> str:
        lines: list[str] = []
        for r in self.results:
            tag = "[ok]   " if r.ok else (
                "[FAIL] " if r.severity == "error" else "[warn] "
            )
            lines.append(f"{tag}{r.name}: {r.message}")
        return "\n".join(lines)


# ---------- individual checks ----------


def check_python_version(report: Report) -> None:
    major, minor = sys.version_info[:2]
    ok = (major == 3) and (10 <= minor <= 12)
    report.add(
        "python_version",
        ok,
        f"Python {major}.{minor} (need 3.10–3.12)",
        severity="error" if not ok else "error",
    )


def _run_git(args: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 127, "git not found"
    return out.returncode, (out.stdout + out.stderr).strip()


def check_git_state(report: Report, allow_dirty: bool) -> None:
    rc, _ = _run_git(["rev-parse", "--is-inside-work-tree"])
    if rc != 0:
        report.add("git_repo", False, "not inside a git repo")
        return
    report.add("git_repo", True, "inside a git working tree")

    rc, branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if rc == 0:
        report.add("git_branch", True, f"branch={branch}")

    rc, status = _run_git(["status", "--porcelain"])
    if rc != 0:
        report.add("git_clean", False, "git status failed")
        return
    if status:
        # Filter out only safe-to-ignore items (untracked files allowed by .gitignore
        # already wouldn't show; anything here is real). Be conservative.
        msg = f"working tree has changes ({len(status.splitlines())} entries)"
        if allow_dirty:
            report.add("git_clean", True, f"{msg} (allowed by --allow-dirty)", severity="warn")
        else:
            report.add("git_clean", False, msg)
    else:
        report.add("git_clean", True, "working tree clean")


def check_env_var_presence(report: Report) -> None:
    """Report presence/absence of secret env vars without printing values."""
    for var in REQUIRED_ENV_VARS:
        val = os.environ.get(var)
        if val:
            report.add(
                f"env_{var}",
                True,
                f"{var} is set (length={len(val)})",
            )
        else:
            report.add(
                f"env_{var}",
                False,
                f"{var} is not set (required for gated HF model access)",
            )

    for var in OPTIONAL_ENV_VARS:
        val = os.environ.get(var)
        if val:
            report.add(
                f"env_{var}",
                True,
                f"{var} is set (length={len(val)})",
                severity="warn",
            )
        else:
            report.add(
                f"env_{var}",
                True,
                f"{var} is not set (optional)",
                severity="warn",
            )


def check_modal_cli(report: Report) -> None:
    path = shutil.which("modal")
    if path:
        try:
            out = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            ver = (out.stdout + out.stderr).strip().splitlines()[0:1]
            ver_str = ver[0] if ver else "version unknown"
        except Exception as exc:  # noqa: BLE001
            ver_str = f"version probe failed: {exc.__class__.__name__}"
        report.add("modal_cli", True, f"modal CLI present ({ver_str})", severity="warn")
    else:
        report.add(
            "modal_cli",
            True,
            "modal CLI not on PATH (ok for local preflight; install on Modal worker)",
            severity="warn",
        )


def check_hf_token_silently(report: Report) -> None:
    """Confirm HF_TOKEN format looks plausible without printing it.

    Real validation requires a network call; we only check shape here.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        report.add(
            "hf_token_shape",
            False,
            "neither HF_TOKEN nor HUGGING_FACE_HUB_TOKEN is set",
        )
        return
    looks_plausible = len(token) >= 20 and not token.startswith(" ") and not token.endswith(" ")
    report.add(
        "hf_token_shape",
        looks_plausible,
        f"token shape ok (length={len(token)})" if looks_plausible
        else "token looks too short or has surrounding whitespace",
    )


def _iter_tracked_files() -> Iterable[str]:
    rc, out = _run_git(["ls-files"])
    if rc != 0:
        return ()
    return out.splitlines()


def check_no_secret_paths_tracked(report: Report) -> None:
    matches: list[str] = []
    for path in _iter_tracked_files():
        for pat in SECRET_PATH_PATTERNS:
            if pat.search(path):
                # .env.example is allowed
                if path.endswith(".env.example"):
                    continue
                matches.append(path)
                break
    if matches:
        report.add(
            "no_secret_paths_tracked",
            False,
            f"git tracks paths matching secret patterns: {matches}",
        )
    else:
        report.add(
            "no_secret_paths_tracked",
            True,
            "no tracked file matches secret-path patterns",
        )


def check_disk_space(report: Report, output_dir: str, min_gb: float) -> None:
    target = Path(output_dir)
    probe = target if target.exists() else target.parent if target.parent.exists() else Path(".")
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        report.add("disk_space", False, f"disk_usage failed at {probe}: {exc}")
        return
    free_gb = usage.free / (1024**3)
    ok = free_gb >= min_gb
    report.add(
        "disk_space",
        ok,
        f"free={free_gb:.1f} GB at {probe} (need >= {min_gb:.1f} GB)",
    )


def check_result_dirs(report: Report, dirs: Iterable[str]) -> None:
    missing: list[str] = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            missing.append(d)
    if missing:
        report.add(
            "result_dirs",
            True,
            f"will be created on first write: {missing}",
            severity="warn",
        )
    else:
        report.add("result_dirs", True, "all expected result dirs exist")


# ---------- main ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Read-only preflight checks for SpectralQuant v2 Modal runs.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as failures",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="tolerate uncommitted changes in the working tree",
    )
    p.add_argument(
        "--output-dir",
        default="results/three_way",
        help="result output directory to check disk space for",
    )
    p.add_argument(
        "--min-disk-gb",
        type=float,
        default=5.0,
        help="minimum free disk space in GB (default 5)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = Report()

    check_python_version(report)
    check_git_state(report, allow_dirty=args.allow_dirty)
    check_env_var_presence(report)
    check_hf_token_silently(report)
    check_modal_cli(report)
    check_no_secret_paths_tracked(report)
    check_disk_space(report, args.output_dir, args.min_disk_gb)
    check_result_dirs(report, DEFAULT_RESULT_DIRS)

    print(report.render())
    print()

    errors = report.errors()
    warnings = report.warnings()
    if errors:
        print(f"FAIL: {len(errors)} required check(s) failed", file=sys.stderr)
        return 1
    if warnings and args.strict:
        print(f"FAIL (--strict): {len(warnings)} warning(s)", file=sys.stderr)
        return 1
    print(f"OK: {len(report.results)} check(s); {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
