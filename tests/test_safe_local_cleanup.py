"""Tests for scripts/safe_local_cleanup.py.

These tests:
  * never delete anything outside ``tmp_path``;
  * verify the active-repo guard rejects the repo root and any path that
    contains or is contained by it;
  * verify dry-run is the default and nothing is changed without ``--yes``;
  * verify temp-clone discovery excludes the active repo path exactly.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "safe_local_cleanup.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cleanup():
    return _load_module("safe_local_cleanup", SCRIPT_PATH)


# --- safety primitives ----------------------------------------------------


def test_path_conflicts_refuses_repo_root(cleanup, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    reason = cleanup.path_conflicts_with_repo(repo, repo)
    assert reason is not None
    assert "active repo root" in reason


def test_path_conflicts_refuses_ancestor_of_repo(cleanup, tmp_path):
    repo = tmp_path / "outer" / "inner_repo"
    repo.mkdir(parents=True)
    parent = repo.parent  # outer/
    reason = cleanup.path_conflicts_with_repo(parent, repo)
    assert reason is not None
    assert "contains the active repo root" in reason


def test_path_conflicts_refuses_path_inside_repo(cleanup, tmp_path):
    repo = tmp_path / "repo"
    sub = repo / "experiments" / "scratch"
    sub.mkdir(parents=True)
    reason = cleanup.path_conflicts_with_repo(sub, repo)
    assert reason is not None
    assert "inside the active repo" in reason


def test_path_conflicts_allows_unrelated_path(cleanup, tmp_path):
    repo = tmp_path / "repo"
    other = tmp_path / "other_dir"
    repo.mkdir()
    other.mkdir()
    assert cleanup.path_conflicts_with_repo(other, repo) is None


def test_safe_to_delete_rejects_missing_path(cleanup, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    missing = tmp_path / "does_not_exist"
    ok, reason = cleanup.safe_to_delete(missing, repo)
    assert not ok
    assert "does not exist" in reason


def test_safe_to_delete_rejects_repo_root(cleanup, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    ok, reason = cleanup.safe_to_delete(repo, repo)
    assert not ok
    assert "repo" in reason


def test_safe_to_delete_accepts_unrelated_dir(cleanup, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    other = tmp_path / "cache"
    other.mkdir()
    ok, reason = cleanup.safe_to_delete(other, repo)
    assert ok
    assert reason == ""


def test_is_under_handles_self_and_descendant(cleanup, tmp_path):
    a = tmp_path / "a"
    b = a / "b"
    b.mkdir(parents=True)
    assert cleanup.is_under(a, a)
    assert cleanup.is_under(b, a)
    assert not cleanup.is_under(a, b)


# --- temp-clone discovery -------------------------------------------------


def test_discover_temp_clones_excludes_active_repo(cleanup, tmp_path):
    parent = tmp_path / "tmp"
    parent.mkdir()
    active_repo = parent / "spectralquant-v2-active"
    other_clone = parent / "spectralquant-old"
    unrelated = parent / "node_modules"
    for d in (active_repo, other_clone, unrelated):
        d.mkdir()

    found = cleanup.discover_temp_clones(
        parents=[str(parent)],
        prefixes=("spectralquant",),
        repo_root=active_repo,
    )
    found_resolved = {p.resolve() for p in found}
    assert active_repo.resolve() not in found_resolved
    assert other_clone.resolve() in found_resolved
    assert unrelated.resolve() not in found_resolved


def test_discover_temp_clones_handles_missing_parent(cleanup, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    found = cleanup.discover_temp_clones(
        parents=[str(tmp_path / "nope")],
        prefixes=("spectralquant",),
        repo_root=repo,
    )
    assert found == []


# --- plan construction ----------------------------------------------------


def test_build_plan_with_no_flags_is_empty(cleanup, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = cleanup.build_plan(
        repo_root=repo,
        delete_hf=False,
        delete_playwright=False,
        delete_temp_clones=False,
        delete_torch_cache=False,
        delete_pip_cache=False,
    )
    assert plan.entries == []
    assert plan.refusals == []


def test_build_plan_temp_clones_excludes_repo(cleanup, tmp_path):
    repo = tmp_path / "tmp" / "spectralquant-v2-active"
    other = tmp_path / "tmp" / "spectralquant-old"
    repo.mkdir(parents=True)
    other.mkdir()

    plan = cleanup.build_plan(
        repo_root=repo,
        delete_hf=False,
        delete_playwright=False,
        delete_temp_clones=True,
        delete_torch_cache=False,
        delete_pip_cache=False,
        extra_temp_parents=[str(tmp_path / "tmp")],
        extra_temp_prefixes=("spectralquant",),
    )
    paths = {p for _, p in plan.entries}
    assert repo.resolve() not in paths
    assert other.resolve() in paths


def test_build_plan_dedupes_nested_child_under_parent(cleanup, tmp_path, monkeypatch):
    """If an HF cache parent and its 'hub' child are both candidates, only the
    parent should remain in the plan — keeping both caused the second rmtree
    to fail because the first had already removed it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    fake_home = tmp_path / "home"
    hf_root = fake_home / ".cache" / "huggingface"
    hf_hub = hf_root / "hub"
    hf_datasets = hf_root / "datasets"
    hf_hub.mkdir(parents=True)
    hf_datasets.mkdir(parents=True)
    (hf_hub / "blob").write_text("x")
    (hf_datasets / "blob").write_text("x")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    plan = cleanup.build_plan(
        repo_root=repo,
        delete_hf=True,
        delete_playwright=False,
        delete_temp_clones=False,
        delete_torch_cache=False,
        delete_pip_cache=False,
    )
    paths = [p for _, p in plan.entries]
    assert hf_root.resolve() in paths
    assert hf_hub.resolve() not in paths
    assert hf_datasets.resolve() not in paths


def test_execute_plan_succeeds_after_dedup(cleanup, tmp_path, monkeypatch):
    """After dedup, executing the plan must not fail with 'path does not exist'
    even though the original HF defaults list both parent and children.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    fake_home = tmp_path / "home"
    hf_root = fake_home / ".cache" / "huggingface"
    (hf_root / "hub").mkdir(parents=True)
    (hf_root / "datasets").mkdir(parents=True)
    (hf_root / "hub" / "blob").write_text("x")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    plan = cleanup.build_plan(
        repo_root=repo,
        delete_hf=True,
        delete_playwright=False,
        delete_temp_clones=False,
        delete_torch_cache=False,
        delete_pip_cache=False,
    )
    deleted, errors = cleanup.execute_plan(plan, repo)
    assert errors == []
    assert deleted == 1
    assert not hf_root.exists()


def test_build_plan_dedup_preserves_unrelated_entries(cleanup, tmp_path, monkeypatch):
    """Dedup must only drop nested descendants; unrelated paths stay."""
    repo = tmp_path / "repo"
    repo.mkdir()
    fake_home = tmp_path / "home"
    hf_root = fake_home / ".cache" / "huggingface"
    torch_root = fake_home / ".cache" / "torch"
    pip_root = fake_home / ".cache" / "pip"
    for d in (hf_root, hf_root / "hub", torch_root, pip_root):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    plan = cleanup.build_plan(
        repo_root=repo,
        delete_hf=True,
        delete_playwright=False,
        delete_temp_clones=False,
        delete_torch_cache=True,
        delete_pip_cache=True,
    )
    paths = {p for _, p in plan.entries}
    assert hf_root.resolve() in paths
    assert torch_root.resolve() in paths
    assert pip_root.resolve() in paths
    # The hub child is a descendant of hf_root and should be dropped:
    assert (hf_root / "hub").resolve() not in paths


def test_execute_plan_actually_deletes(cleanup, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = tmp_path / "doomed"
    target.mkdir()
    (target / "file.txt").write_text("x")

    plan = cleanup.Plan()
    plan.add("test", target)
    deleted, errors = cleanup.execute_plan(plan, repo)
    assert deleted == 1
    assert errors == []
    assert not target.exists()


def test_execute_plan_refuses_repo_at_runtime(cleanup, tmp_path):
    """Even if a plan smuggles in the repo path, execute_plan re-checks."""
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = cleanup.Plan()
    plan.add("evil", repo)
    deleted, errors = cleanup.execute_plan(plan, repo)
    assert deleted == 0
    assert len(errors) == 1
    assert repo.exists()  # untouched
    assert "repo" in errors[0]


# --- CLI / dry-run --------------------------------------------------------


def test_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "safe local cleanup" in result.stdout.lower()


def test_dry_run_default_does_not_delete(tmp_path):
    """Without --yes, nothing should be removed even if a flag is passed."""
    parent = tmp_path / "tmp"
    parent.mkdir()
    repo = parent / "spectralquant-v2-active"
    other = parent / "spectralquant-old"
    repo.mkdir()
    other.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--delete-temp-clones",
            "--repo-root",
            str(repo),
            "--extra-temp-parent",
            str(parent),
            "--extra-temp-prefix",
            "spectralquant",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "[dry-run]" in result.stdout
    # Both directories should still exist:
    assert repo.exists()
    assert other.exists()


def test_yes_flag_actually_deletes_temp_clone(tmp_path):
    parent = tmp_path / "tmp"
    parent.mkdir()
    repo = parent / "spectralquant-v2-active"
    other = parent / "spectralquant-old"
    repo.mkdir()
    other.mkdir()
    (other / "marker").write_text("delete me")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--delete-temp-clones",
            "--yes",
            "--repo-root",
            str(repo),
            "--extra-temp-parent",
            str(parent),
            "--extra-temp-prefix",
            "spectralquant",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert repo.exists(), "active repo path must not be deleted"
    assert not other.exists(), "non-active temp clone should be deleted"


def test_no_flags_runs_disk_report_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(repo)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Disk usage:" in result.stdout
    assert "No --delete-* flag passed" in result.stdout


def test_min_disk_gb_can_force_failure(tmp_path):
    """Setting an absurdly high --min-disk-gb should make the script exit 1."""
    repo = tmp_path / "repo"
    repo.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--repo-root",
            str(repo),
            "--min-disk-gb",
            "999999999",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stderr


def test_does_not_print_secret_values(monkeypatch, tmp_path):
    """The cleanup script must never echo HF_TOKEN-shaped values."""
    sentinel = "DO_NOT_LEAK_VALUE_123_ABCDEFGHIJKLMNOP"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    repo = tmp_path / "repo"
    repo.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(repo)],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert sentinel not in combined
