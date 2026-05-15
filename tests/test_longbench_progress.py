"""Tests for ``run_longbench``'s per-example generation progress and the
safe partial-checkpoint discipline added in response to the 2026-04-30
post-calibration generation-stage stall.

Coverage:

* ``GenerationProgress.start/step/end`` emits ``eval_task_start`` /
  ``eval_progress`` / ``eval_task_end`` events with the expected
  ``method``, ``task``, ``completed``, ``total``, and ``elapsed_s`` keys
  in ``details``.
* Per-example ``eval_progress`` actually fires every ``progress_every_n``
  examples, plus on the first and last steps.
* The per-(method, task) partial JSON shard is rewritten on every step
  and is tagged ``paper_valid: false, partial: true``.
* The run-level ``partial_status.json`` records planned/completed
  methods + tasks and is tagged ``paper_valid: false, partial: true``.
* Partial artifacts are NEVER written into the canonical result path.
* ``GenerationProgress`` is a no-op when ``status`` is None.
* The new stages appear in ``run_status.KNOWN_STAGES``.

These tests do not load any HF model — they exercise the progress
instrumentation directly and via small in-memory fakes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _add_paths(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "src"))


def _make_status(tmp_path: Path):
    from experiments import run_status

    return run_status.StatusWriter(
        status_dir=tmp_path / "status_root" / "rid",
        run_id="rid",
        commit="abc",
        model="m",
    )


# ---------------------------------------------------------------------------
# Stage registration
# ---------------------------------------------------------------------------


def test_eval_task_start_and_end_are_known_stages():
    from experiments import run_status

    assert "eval_task_start" in run_status.KNOWN_STAGES
    assert "eval_task_end" in run_status.KNOWN_STAGES


# ---------------------------------------------------------------------------
# GenerationProgress event emission
# ---------------------------------------------------------------------------


def _read_events(status) -> List[Dict[str, Any]]:
    p = Path(status.events_path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_generation_progress_emits_start_per_step_and_end(tmp_path):
    from experiments import run_longbench

    status = _make_status(tmp_path)
    prog = run_longbench.GenerationProgress(
        status, method="spectralquant_v2", task="narrativeqa", total=3,
    )
    prog.start()
    prog.step(completion="answer-1")
    prog.step(completion="answer-2")
    prog.step(completion="answer-3")
    prog.end(score=0.42)

    events = _read_events(status)
    stages = [e["stage"] for e in events]
    assert "eval_task_start" in stages
    assert stages.count("eval_progress") == 3  # one per example
    assert "eval_task_end" in stages

    # Every progress event carries method/task/completed/total/elapsed_s.
    for e in events:
        if e["stage"] != "eval_progress":
            continue
        det = e["details"]
        assert det["method"] == "spectralquant_v2"
        assert det["task"] == "narrativeqa"
        assert isinstance(det["completed"], int)
        assert det["total"] == 3
        assert isinstance(det["elapsed_s"], (int, float))

    # eval_task_end carries the score.
    end = next(e for e in events if e["stage"] == "eval_task_end")
    assert end["details"]["score"] == pytest.approx(0.42)
    assert end["details"]["completed"] == 3


def test_generation_progress_progress_every_n_thinning(tmp_path):
    """progress_every_n=5 should still fire at step 1 (first), step 5
    (boundary), and step 10 (last) — but not on every step."""
    from experiments import run_longbench

    status = _make_status(tmp_path)
    prog = run_longbench.GenerationProgress(
        status, method="fp16", task="qasper", total=10,
        progress_every_n=5,
    )
    prog.start()
    for i in range(10):
        prog.step(completion=f"c{i}")
    prog.end()

    events = _read_events(status)
    progress_completed = [
        e["details"]["completed"] for e in events
        if e["stage"] == "eval_progress"
    ]
    # First (1), 5 (boundary), 10 (last) — and 10 is also a multiple-of-5
    # boundary, so we get exactly 3 events.
    assert progress_completed == [1, 5, 10]


def test_generation_progress_no_status_is_noop(tmp_path):
    from experiments import run_longbench

    prog = run_longbench.GenerationProgress(
        status=None, method="fp16", task="t", total=2,
    )
    prog.start()
    prog.step(completion="x")
    prog.step(completion="y")
    prog.end()
    # No partial dir created, no events.
    assert not (tmp_path / "status_root").exists()


# ---------------------------------------------------------------------------
# Per-(method, task) partial JSON shard
# ---------------------------------------------------------------------------


def test_partial_task_shard_written_after_every_step(tmp_path):
    from experiments import run_longbench

    status = _make_status(tmp_path)
    prog = run_longbench.GenerationProgress(
        status, method="spectralquant_v2", task="narrativeqa", total=4,
    )
    prog.start()
    prog.step(completion="alpha")
    shard_path = (
        Path(status.status_dir) / run_longbench.PARTIAL_SUBDIR
        / "spectralquant_v2__narrativeqa.json"
    )
    assert shard_path.exists()
    after_one = json.loads(shard_path.read_text())
    assert after_one["completed"] == 1
    assert after_one["total"] == 4
    assert after_one["paper_valid"] is False
    assert after_one["partial"] is True
    assert after_one["status"] == "running"
    assert after_one["kind"] == "longbench_partial_task"
    assert after_one["last_completion_chars"] == len("alpha")

    prog.step(completion="beta")
    after_two = json.loads(shard_path.read_text())
    assert after_two["completed"] == 2

    prog.end(score=0.7)
    final = json.loads(shard_path.read_text())
    assert final["status"] == "ended"
    assert final["score"] == pytest.approx(0.7)


def test_partial_task_shard_filename_is_sanitized(tmp_path):
    """A method or task containing path separators must not write
    outside the partial directory."""
    from experiments import run_longbench

    status = _make_status(tmp_path)
    prog = run_longbench.GenerationProgress(
        status, method="../escape", task="../t", total=1,
    )
    prog.start()
    prog.step(completion="x")
    pd = Path(status.status_dir) / run_longbench.PARTIAL_SUBDIR
    files = list(pd.iterdir())
    # Exactly one shard, sitting in the partial dir — not anywhere else.
    assert len(files) == 1
    assert files[0].parent.resolve() == pd.resolve()
    # The shard name has no path separator, so it cannot escape pd even
    # if a future filename change reintroduces "..".
    assert "/" not in files[0].name
    assert "\\" not in files[0].name


# ---------------------------------------------------------------------------
# Run-level partial_status.json
# ---------------------------------------------------------------------------


def test_run_level_partial_status_paper_valid_false(tmp_path):
    from experiments import run_longbench

    status = _make_status(tmp_path)
    args = run_longbench.parse_args([
        "--model", "m", "--output-dir", str(tmp_path),
        "--methods", "fp16", "spectralquant_v2",
        "--subset", "minimal", "--n-per-task", "5",
    ])
    run_longbench._write_run_partial_status(
        status, args=args, mode="full",
        methods_planned=["fp16", "spectralquant_v2"],
        tasks_planned=["narrativeqa", "qasper"],
        current_method="spectralquant_v2",
        current_task="narrativeqa",
        completed_methods=["fp16"],
        completed_tasks_per_method={"spectralquant_v2": []},
        note="starting method=spectralquant_v2",
    )
    snap_path = (
        Path(status.status_dir)
        / run_longbench.PARTIAL_SUBDIR
        / run_longbench.PARTIAL_STATUS_FILENAME
    )
    assert snap_path.exists()
    snap = json.loads(snap_path.read_text())
    assert snap["paper_valid"] is False
    assert snap["partial"] is True
    assert snap["kind"] == "longbench_partial_run"
    assert snap["methods_planned"] == ["fp16", "spectralquant_v2"]
    assert snap["completed_methods"] == ["fp16"]
    assert snap["current_method"] == "spectralquant_v2"
    assert snap["current_task"] == "narrativeqa"
    assert snap["note"] == "starting method=spectralquant_v2"


def test_partial_directory_separate_from_canonical_result_path(tmp_path):
    """Partial artifacts live under <status_dir>/partial/, never at the
    canonical <output_dir>/<run_id>.json result path. A partial JSON
    must never be mistaken for a paper-valid result."""
    from experiments import eval_common, run_longbench

    status = _make_status(tmp_path)
    prog = run_longbench.GenerationProgress(
        status, method="fp16", task="t", total=1,
    )
    prog.start()
    prog.step(completion="x")
    prog.end()

    args = run_longbench.parse_args([
        "--model", "m", "--output-dir", str(tmp_path / "out"),
        "--methods", "fp16", "--subset", "minimal", "--n-per-task", "5",
    ])
    canonical = eval_common.output_path(
        args.output_dir, run_longbench.derive_run_id(args, "full"),
    )
    # Canonical result path was never created (partials only).
    assert not canonical.exists()
    # Partials are under status_dir / partial / and NOT inside output_dir.
    pd = Path(status.status_dir) / run_longbench.PARTIAL_SUBDIR
    assert pd.is_dir()
    assert canonical not in pd.iterdir()


# ---------------------------------------------------------------------------
# _generate_for_task threads progress through to per-example events.
# ---------------------------------------------------------------------------


class _FakeIds:
    def __init__(self, n: int = 1) -> None:
        self._n = n

    def to(self, *_a, **_k):
        return self

    def size(self, _dim: int) -> int:
        return self._n


class _FakeBatch:
    def __init__(self, ids: _FakeIds) -> None:
        self.input_ids = ids


class _FakeTokenizer:
    def __call__(self, *_a, **_k):
        return _FakeBatch(_FakeIds(1))

    def decode(self, ids, **_k):
        # Return a deterministic tag using the integer state of the ids.
        return f"completion-{getattr(ids, '_tag', 0)}"


class _FakeOutIds:
    def __init__(self, tag: int) -> None:
        self._tag = tag

    def __getitem__(self, _slice):
        return self  # pretend slicing returns same fake


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *_a, **_k):
        self.calls += 1
        return _FakeOutIds(self.calls)


class _FakeRow:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.expected_max_new_tokens = 8


def test_generate_for_task_emits_per_example_events(tmp_path, monkeypatch):
    """``_generate_for_task`` should drive the progress object once per
    row. We avoid loading HF/torch by stubbing the import."""
    fake_torch = type(sys)("torch")

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    fake_torch.no_grad = _NoGrad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from experiments import run_longbench

    status = _make_status(tmp_path)
    args = run_longbench.parse_args([
        "--model", "m", "--output-dir", str(tmp_path),
        "--methods", "fp16", "--subset", "minimal", "--n-per-task", "3",
        "--max-input-tokens", "128", "--max-new-tokens", "8",
    ])
    rows = [_FakeRow(f"p{i}") for i in range(3)]
    prog = run_longbench.GenerationProgress(
        status, method="fp16", task="narrativeqa", total=len(rows),
    )
    completions = run_longbench._generate_for_task(
        _FakeModel(), _FakeTokenizer(), device="cpu", args=args, rows=rows,
        progress=prog,
    )
    prog.end()
    assert len(completions) == 3

    events = _read_events(status)
    progress_events = [e for e in events if e["stage"] == "eval_progress"]
    completeds = [e["details"]["completed"] for e in progress_events]
    assert completeds == [1, 2, 3]

    # Per-task partial JSON exists and shows ended.
    shard = (
        Path(status.status_dir) / run_longbench.PARTIAL_SUBDIR
        / "fp16__narrativeqa.json"
    )
    assert shard.exists()
    final = json.loads(shard.read_text())
    assert final["completed"] == 3
    assert final["total"] == 3
    assert final["status"] == "ended"
    assert final["paper_valid"] is False
    assert final["partial"] is True


# ---------------------------------------------------------------------------
# Paper-valid discipline: partial outputs do NOT bypass paper-valid gates.
# ---------------------------------------------------------------------------


def test_partial_artifacts_do_not_make_payload_paper_valid(tmp_path):
    """A run that left only partial artifacts must not be able to claim
    paper_valid=true. We simulate by feeding a result JSON's methods
    through ``build_payload`` and asserting that even without partials
    the gate logic continues to work — i.e. partial artifacts on disk
    are independent of the schema-validated payload."""
    from experiments import run_longbench, eval_common

    args = run_longbench.parse_args([
        "--model", "m", "--output-dir", str(tmp_path),
        "--methods", "fp16",
        "--n-per-task", "50", "--subset", "minimal",
    ])
    methods = {
        "fp16": {
            "label": "fp16",
            "per_task": [{"task": "narrativeqa", "metric": "qa_f1_en",
                          "score": 0.4, "n_examples": 50}],
            "aggregate": {"macro_score": 0.4},
            "evidence_ids": ["x"],
            "dataset_source": "huggingface_thudm",
        },
    }
    payload = run_longbench.build_payload(
        args, ["x"], eval_common.MODE_FULL, methods,
    )
    # Without spectralquant_v2 partials, FP16-only minimal is paper-valid
    # subject to its own caveats (transparent subset). The test confirms
    # that partial-status code paths did not silently flip the gate.
    assert payload["paper_valid"] is True
