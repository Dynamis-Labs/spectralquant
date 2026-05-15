#!/usr/bin/env python3
"""LongBench evaluation harness.

LongBench (https://github.com/THUDM/LongBench) is a long-context
benchmark suite covering multi-document QA, summarization, few-shot
learning, code completion, and synthetic tasks. The v1 paper reported
LongBench scores at ``n=5`` per task — V1-GAP-008 in the evidence
catalog. The next-generation v2 evidence requires ``n_per_task >= 50``
to be paper-valid.

Modes
-----

* ``--dry-run`` — validate args, print plan, write nothing.
* ``--synthetic-smoke`` — fabricate per-task scores for the configured
  task list. Output ``paper_valid=false``, ``mode=synthetic_smoke``.
* ``--inline-corpus-smoke`` — load HF model, generate completions on
  the inline corpus, then score with token-F1. Output ``paper_valid=false``.
* (no flag) — full path. Loads ``THUDM/LongBench`` from HF datasets,
  formats per-task prompts using upstream templates, runs greedy
  decode at the upstream per-task ``max_new_tokens``, and scores using
  the per-task LongBench metrics vendored in
  :mod:`experiments.longbench_metrics`. Output ``paper_valid=true``
  iff (a) ``n_per_task >= 50``, (b) every requested method is in
  ``REAL_EVAL_METHODS``, (c) no placeholder records, (d) replay
  coverage >= 0.99 for non-FP16, (e) the actual upstream HF dataset
  was loaded (``methods.<m>.dataset_source == "huggingface_thudm"``).

Subset selection
----------------

* ``minimal`` — three short-form English tasks.
* ``deterministic`` — five tasks covering each LongBench category.
* ``full`` — all 21 LongBench tasks. Long-running; use with care.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import eval_common  # noqa: E402
from experiments import run_status  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "schemas" / "longbench.schema.json"
FAMILY = "longbench"

#: Methods for which the harness can produce real (non-placeholder)
#: per-task scores when the upstream LongBench dataset is loaded.
#: ``--inline-corpus-smoke`` exercises this path on a synthetic corpus
#: only; the full LongBench path uses HF datasets THUDM/LongBench.
REAL_EVAL_METHODS: Tuple[str, ...] = ("fp16", "spectralquant_v2", "turboquant")


# Subsets: deliberately small first for runtime safety on Modal.
SUBSETS: Dict[str, List[str]] = {
    "minimal": [
        "narrativeqa",
        "qasper",
        "hotpotqa",
    ],
    "deterministic": [
        "narrativeqa",
        "qasper",
        "hotpotqa",
        "gov_report",
        "trec",
    ],
    "full": [
        "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
        "hotpotqa", "2wikimqa", "musique", "dureader",
        "gov_report", "qmsum", "multi_news", "vcsum",
        "trec", "triviaqa", "samsum", "lsht",
        "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
        "lcc", "repobench-p",
    ],
}


# Per-task default metric (kept here for synthetic smoke records). The
# *real* per-task metric name comes from longbench_metrics.TASK_METRIC_NAMES
# whenever the full path runs, so a downstream consumer can always trust
# the per_task[*].metric field.
TASK_METRIC: Dict[str, str] = {
    "narrativeqa": "qa_f1_en",
    "qasper": "qa_f1_en",
    "multifieldqa_en": "qa_f1_en",
    "multifieldqa_zh": "qa_f1_zh",
    "hotpotqa": "qa_f1_en",
    "2wikimqa": "qa_f1_en",
    "musique": "qa_f1_en",
    "dureader": "rouge_zh",
    "gov_report": "rouge_en",
    "qmsum": "rouge_en",
    "multi_news": "rouge_en",
    "vcsum": "rouge_zh",
    "trec": "classification_em",
    "triviaqa": "qa_f1_en",
    "samsum": "rouge_en",
    "lsht": "classification_em",
    "passage_count": "count_em",
    "passage_retrieval_en": "retrieval_em",
    "passage_retrieval_zh": "retrieval_em",
    "lcc": "edit_sim",
    "repobench-p": "edit_sim",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    model: str
    output_dir: Path
    seed: int
    device: str
    dtype: str
    dry_run: bool
    synthetic_smoke: bool
    inline_corpus_smoke: bool
    force: bool
    skip_if_exists: bool
    git_commit_override: Optional[str]
    status_dir: Optional[Path]
    avg_bits: int
    methods: Tuple[str, ...]
    subset: str
    n_per_task: int
    max_input_tokens: int
    max_new_tokens: int
    use_e_split: bool
    n_calib: int
    calib_max_seq_tokens: int
    lloyd_max_iter: int
    calibration_mode: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_longbench.py",
        description=(
            "SpectralQuant v2 LongBench evaluation harness. "
            "Supports --dry-run, --synthetic-smoke, --inline-corpus-smoke "
            "for safe local use; the full path is intended for Modal."
        ),
    )
    eval_common.add_common_eval_args(p)
    p.add_argument("--avg-bits", type=int, default=3, dest="avg_bits")
    p.add_argument("--methods", nargs="+", default=["fp16"],
                   choices=list(eval_common.KNOWN_METHOD_KEYS))
    p.add_argument("--subset", type=str, default="minimal",
                   choices=list(SUBSETS.keys()))
    p.add_argument("--n-per-task", type=int, default=50, dest="n_per_task",
                   help=("Examples per task. Paper-valid runs require >=50 "
                         "(V1-GAP-008)."))
    p.add_argument("--max-input-tokens", type=int, default=8192,
                   dest="max_input_tokens")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   dest="max_new_tokens",
                   help=("Cap on max_new_tokens. The harness still uses "
                         "the upstream per-task max-new value when smaller."))
    p.add_argument("--use-e-split", action="store_true", dest="use_e_split",
                   help=("Load the LongBench-E (English-only equal-length) "
                         "splits instead of the default. Off by default."))
    # ----------------------------------------------------------------
    # Calibration knobs (tractability for compressed methods).
    # The previous LongBench harness wired n_calib_samples = len(calib_texts)
    # which silently scaled with the eval set; we now expose it explicitly
    # so an operator can dial calibration cost down to keep a CPU-bound
    # path from going silent for hours.
    # ----------------------------------------------------------------
    p.add_argument("--n-calib", type=int, default=None, dest="n_calib",
                   help=("Number of calibration prompts to use for "
                         "spectralquant_v2 / turboquant calibration. "
                         "Defaults to min(16, 4*len(tasks)). Lower values "
                         "speed up calibration but trigger an explicit "
                         "'reduced calibration' caveat in the result JSON."))
    p.add_argument("--calib-max-seq-tokens", type=int, default=512,
                   dest="calib_max_seq_tokens",
                   help=("Per-prompt token cap during calibration K/V "
                         "capture (default 512). Lowering this is the "
                         "single biggest knob for CPU-bound runs."))
    p.add_argument("--lloyd-max-iter", type=int, default=200,
                   dest="lloyd_max_iter",
                   help=("Cap on Lloyd-Max iterations per per-head codebook "
                         "fit (default 200, matching EngineConfig). 25-50 "
                         "is a useful smoke value; anything <200 marks the "
                         "run as 'reduced calibration'."))
    p.add_argument("--calibration-mode", type=str, default="auto",
                   choices=("auto", "paper", "smoke"),
                   dest="calibration_mode",
                   help=("'paper': enforce paper-valid calibration knobs "
                         "(n_calib>=16, lloyd_max_iter==200, calib_max_seq_tokens"
                         ">=512). 'smoke': bounded paper-candidate config "
                         "explicitly tagged reduced_calibration=true and "
                         "paper_valid=false. 'auto' (default): infer from "
                         "the supplied knobs."))
    return p


#: Paper-valid calibration thresholds. Anything below ANY of these is
#: classified as a "reduced calibration" run and forces paper_valid=False
#: with an explicit caveat. These are the conservative defaults that
#: preceded the LongBench timeout incident — see
#: ``docs/execution_audit_and_modal_runbook.md`` §7.7.5 for the policy.
PAPER_VALID_N_CALIB_MIN: int = 16
PAPER_VALID_LLOYD_MAX_ITER: int = 200
PAPER_VALID_CALIB_MAX_SEQ_TOKENS_MIN: int = 512


def _default_n_calib(subset: str) -> int:
    """Return a reasonable default ``n_calib`` for a subset.

    Mirrors the legacy hard-coded ``[:16]`` cap that the K/V replay path
    used before the knob existed. We default to the paper-valid minimum
    (``PAPER_VALID_N_CALIB_MIN`` = 16) so an unspecified ``--n-calib``
    does not silently break paper validity. Operators who want a smoke
    run pass ``--n-calib 4 --calibration-mode smoke`` explicitly.
    """
    n_tasks = len(SUBSETS.get(subset, []))
    return max(PAPER_VALID_N_CALIB_MIN, 4 * max(1, n_tasks))


def is_reduced_calibration(args: "Args") -> bool:
    """Return True if any calibration knob is below the paper-valid bar."""
    if args.n_calib < PAPER_VALID_N_CALIB_MIN:
        return True
    if args.lloyd_max_iter < PAPER_VALID_LLOYD_MAX_ITER:
        return True
    if args.calib_max_seq_tokens < PAPER_VALID_CALIB_MAX_SEQ_TOKENS_MIN:
        return True
    return False


def parse_args(argv: Optional[Sequence[str]] = None) -> Args:
    p = build_parser()
    ns = p.parse_args(argv)
    if ns.synthetic_smoke and ns.inline_corpus_smoke:
        p.error("--synthetic-smoke and --inline-corpus-smoke are mutually exclusive")
    if ns.dry_run:
        ns.synthetic_smoke = False
        ns.inline_corpus_smoke = False
    if ns.force and ns.skip_if_exists:
        ns.skip_if_exists = False
    if ns.n_per_task < 1:
        p.error("--n-per-task must be >= 1")

    # n_calib: derive a default from the subset if not supplied.
    if ns.n_calib is None:
        ns.n_calib = _default_n_calib(str(ns.subset))
    if ns.n_calib < 1:
        p.error("--n-calib must be >= 1")
    if ns.calib_max_seq_tokens < 16:
        p.error("--calib-max-seq-tokens must be >= 16")
    if ns.lloyd_max_iter < 1:
        p.error("--lloyd-max-iter must be >= 1")

    if ns.calibration_mode == "paper":
        # Hard-fail rather than silently degrade a paper-mode launch.
        if ns.n_calib < PAPER_VALID_N_CALIB_MIN:
            p.error(
                f"--calibration-mode paper requires --n-calib >= "
                f"{PAPER_VALID_N_CALIB_MIN}, got {ns.n_calib}"
            )
        if ns.lloyd_max_iter < PAPER_VALID_LLOYD_MAX_ITER:
            p.error(
                f"--calibration-mode paper requires --lloyd-max-iter == "
                f"{PAPER_VALID_LLOYD_MAX_ITER}, got {ns.lloyd_max_iter}"
            )
        if ns.calib_max_seq_tokens < PAPER_VALID_CALIB_MAX_SEQ_TOKENS_MIN:
            p.error(
                f"--calibration-mode paper requires --calib-max-seq-tokens "
                f">= {PAPER_VALID_CALIB_MAX_SEQ_TOKENS_MIN}, got "
                f"{ns.calib_max_seq_tokens}"
            )

    return Args(
        model=ns.model,
        output_dir=Path(ns.output_dir),
        seed=int(ns.seed),
        device=str(ns.device),
        dtype=str(ns.dtype),
        dry_run=bool(ns.dry_run),
        synthetic_smoke=bool(ns.synthetic_smoke),
        inline_corpus_smoke=bool(ns.inline_corpus_smoke),
        force=bool(ns.force),
        skip_if_exists=bool(ns.skip_if_exists),
        git_commit_override=ns.git_commit_override,
        status_dir=(Path(ns.status_dir) if ns.status_dir else None),
        avg_bits=int(ns.avg_bits),
        methods=tuple(ns.methods),
        subset=str(ns.subset),
        n_per_task=int(ns.n_per_task),
        max_input_tokens=int(ns.max_input_tokens),
        max_new_tokens=int(ns.max_new_tokens),
        use_e_split=bool(ns.use_e_split),
        n_calib=int(ns.n_calib),
        calib_max_seq_tokens=int(ns.calib_max_seq_tokens),
        lloyd_max_iter=int(ns.lloyd_max_iter),
        calibration_mode=str(ns.calibration_mode),
    )


# ---------------------------------------------------------------------------
# Calibration progress callback
# ---------------------------------------------------------------------------


def _make_progress_cb(
    status: Optional[run_status.StatusWriter], method: str,
):
    """Return an emitter usable as ``sqv2_replay.build_calibrated_engine``'s
    ``progress`` callback. If ``status`` is ``None``, returns ``None`` so
    the calibrator skips the work of building progress payloads."""
    if status is None:
        return None

    def _cb(stage: str, message: str, details: Optional[Dict[str, Any]]) -> None:
        try:
            status.emit(
                stage,
                message=f"{method} {message}" if message else None,
                details=(dict(details) | {"method": method}) if details else {"method": method},
            )
        except Exception:  # noqa: BLE001
            pass

    return _cb


# ---------------------------------------------------------------------------
# Per-task generation progress + partial checkpoints
# ---------------------------------------------------------------------------
#
# The 2026-04-30 LongBench stall on Modal showed two distinct silent
# regions: calibration (now instrumented per-(layer,head,type) by
# experiments/sqv2_replay) and the post-calibration generation loop —
# `eval_progress task=<t>` was emitted *once* per task, with no per-
# example heartbeat, no partial completions on disk, and no way to tell a
# stuck generate() apart from a slow one. ``GenerationProgress`` fills
# both gaps:
#
#   * it emits an ``eval_progress`` status event per example (or every
#     ``progress_every_n``-th example) with ``method``, ``task``,
#     ``completed``, ``total``, and ``elapsed_s``; and
#   * after every example it atomically rewrites a tiny per-(method,task)
#     partial JSON under ``<status_dir>/partial/`` so an interrupted run
#     leaves explicit evidence of where it stopped.
#
# Partial checkpoints are NEVER paper-valid; they are tagged
# ``paper_valid: false`` and ``partial: true`` and live next to the
# events stream, not at the canonical result path. The full result JSON
# is still produced only by the canonical schema-validated atomic write
# at the end of a successful run.

#: Filename of the run-level partial-status snapshot written next to the
#: per-(method,task) shards.
PARTIAL_STATUS_FILENAME = "partial_status.json"

#: Subdirectory (under the run's ``status_dir``) used for partial
#: per-(method,task) completions.
PARTIAL_SUBDIR = "partial"


def _atomic_write_partial(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic JSON write for partial / status artifacts.

    Distinct from :func:`eval_common.atomic_write_json` so it can never
    accidentally validate against the family schema (a partial completion
    is not a valid LongBench result). Same tempfile + ``os.replace`` +
    ``fsync`` discipline so an interrupted write never appears at the
    canonical path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_partial_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _partial_dir_for(status: Optional[run_status.StatusWriter]) -> Optional[Path]:
    """Return the partial-checkpoint directory for ``status`` or None.

    Lives under ``status.status_dir / PARTIAL_SUBDIR`` so a poller can
    look in one well-known place regardless of family/run_id.
    """
    if status is None:
        return None
    return Path(status.status_dir) / PARTIAL_SUBDIR


class GenerationProgress:
    """Per-(method, task) generation observability + partial checkpointing.

    Wraps a :class:`run_status.StatusWriter` so the LongBench harness can
    emit ``eval_task_start`` / ``eval_progress`` / ``eval_task_end``
    events with explicit ``method``, ``task``, ``completed``, ``total``,
    ``elapsed_s`` details, and write a small JSON checkpoint after every
    example. Safe when ``status`` is ``None`` (then it is a no-op except
    for accumulating completion counts in memory).

    ``progress_every_n`` controls how often status events are emitted:
    ``1`` emits one per example (default), ``5`` once every five
    examples; the *partial JSON checkpoint* is rewritten on every example
    regardless, since it's the only on-disk record of where a stalled
    run stopped. The first example, the last example, and the boundary
    that crosses ``progress_every_n`` always emit.
    """

    def __init__(
        self,
        status: Optional[run_status.StatusWriter],
        method: str,
        task: str,
        total: int,
        *,
        progress_every_n: int = 1,
    ) -> None:
        self.status = status
        self.method = str(method)
        self.task = str(task)
        self.total = int(total)
        self.progress_every_n = max(1, int(progress_every_n))
        self.completed: int = 0
        self._start_ts: float = time.monotonic()
        self._partial_dir = _partial_dir_for(status)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._start_ts

    def _details(self, **extra: Any) -> Dict[str, Any]:
        details: Dict[str, Any] = {
            "method": self.method,
            "task": self.task,
            "completed": int(self.completed),
            "total": int(self.total),
            "elapsed_s": round(self.elapsed_s, 3),
        }
        details.update(extra)
        return details

    def start(self) -> None:
        if self.status is None:
            return
        try:
            self.status.emit(
                "eval_task_start",
                message=f"{self.method} task={self.task} total={self.total}",
                details=self._details(),
            )
        except Exception:  # noqa: BLE001
            pass
        self._write_checkpoint(completion=None, status="running")

    def step(self, completion: Optional[str] = None) -> None:
        """Record one finished example and (maybe) emit a progress event.

        ``completion`` is intentionally NOT persisted in the partial JSON
        — it can be very large (LongBench summarization) and is sensitive
        to the same redaction rules as model output. We only persist
        counts, timing, and the most recent completion's length, which is
        enough to identify *where* a stalled run stopped without leaking
        bulk model output.
        """
        self.completed += 1
        last_len = len(completion) if isinstance(completion, str) else None

        # Emit periodic progress, plus always at the boundaries.
        is_first = self.completed == 1
        is_last = self.completed >= self.total
        is_periodic = (self.completed % self.progress_every_n) == 0
        if self.status is not None and (is_first or is_last or is_periodic):
            try:
                self.status.emit(
                    "eval_progress",
                    message=(
                        f"{self.method} task={self.task} "
                        f"{self.completed}/{self.total} "
                        f"elapsed_s={round(self.elapsed_s, 1)}"
                    ),
                    details=self._details(last_completion_chars=last_len),
                )
            except Exception:  # noqa: BLE001
                pass

        # Always rewrite the partial checkpoint — this is the only on-disk
        # record a poller can see when a run goes silent mid-task.
        self._write_checkpoint(
            completion=last_len, status="running",
        )

    def end(self, *, score: Optional[float] = None) -> None:
        if self.status is not None:
            try:
                self.status.emit(
                    "eval_task_end",
                    message=(
                        f"{self.method} task={self.task} "
                        f"{self.completed}/{self.total} "
                        f"elapsed_s={round(self.elapsed_s, 1)}"
                        + (f" score={score:.4f}" if score is not None else "")
                    ),
                    details=self._details(score=score),
                )
            except Exception:  # noqa: BLE001
                pass
        self._write_checkpoint(
            completion=None, status="ended", score=score,
        )

    def _write_checkpoint(
        self,
        *,
        completion: Optional[int],
        status: str,
        score: Optional[float] = None,
    ) -> None:
        if self._partial_dir is None:
            return
        # Per-(method, task) shard. Filename is sanitized so we never
        # interpolate user input into a path.
        m = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.method)
        t = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.task)
        shard = self._partial_dir / f"{m}__{t}.json"
        payload: Dict[str, Any] = {
            "method": self.method,
            "task": self.task,
            "completed": int(self.completed),
            "total": int(self.total),
            "elapsed_s": round(self.elapsed_s, 3),
            "status": status,
            "paper_valid": False,
            "partial": True,
            "kind": "longbench_partial_task",
        }
        if completion is not None:
            payload["last_completion_chars"] = int(completion)
        if score is not None:
            payload["score"] = float(score)
        try:
            _atomic_write_partial(shard, payload)
        except OSError:
            pass


def _write_method_partial_record(
    status: Optional[run_status.StatusWriter],
    *,
    method: str,
    record: Dict[str, Any],
) -> None:
    """Persist the in-memory per-method evaluation record to a partial JSON.

    Called immediately after a method's full evaluation finishes successfully.
    The shard is written to ``<status_dir>/partial/method__<method>.json`` so
    that an interrupted run (e.g. a later method timing out) does not lose
    expensive completed-method results. These shards are tagged
    ``paper_valid: false, partial: true`` because they are not the canonical
    schema-validated three-method JSON; merging them into a paper-valid
    artifact requires the full set of methods or an explicit recombination
    step.

    Safe to call with ``status=None`` — then it is a no-op.
    """
    pd = _partial_dir_for(status)
    if pd is None or status is None:
        return
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in method)
    shard = pd / f"method__{safe}.json"
    payload = {
        "kind": "longbench_partial_method_record",
        "method": method,
        "paper_valid": False,
        "partial": True,
        "record": record,
    }
    try:
        _atomic_write_partial(shard, payload)
    except OSError:
        pass


def _write_run_partial_status(
    status: Optional[run_status.StatusWriter],
    *,
    args: "Args",
    mode: str,
    methods_planned: Sequence[str],
    tasks_planned: Sequence[str],
    current_method: Optional[str] = None,
    current_task: Optional[str] = None,
    completed_methods: Sequence[str] = (),
    completed_tasks_per_method: Optional[Dict[str, List[str]]] = None,
    note: Optional[str] = None,
) -> None:
    """Write a small run-level snapshot of where a run currently is.

    Lives at ``<status_dir>/partial/partial_status.json`` and is rewritten
    at every method/task transition. Always tagged
    ``paper_valid: False, partial: True``. Safe to call with
    ``status=None`` — then it is a no-op.
    """
    pd = _partial_dir_for(status)
    if pd is None or status is None:
        return
    payload = {
        "kind": "longbench_partial_run",
        "run_id": status.run_id,
        "mode": mode,
        "model": status.model,
        "subset": args.subset,
        "n_per_task": int(args.n_per_task),
        "methods_planned": list(methods_planned),
        "tasks_planned": list(tasks_planned),
        "current_method": current_method,
        "current_task": current_task,
        "completed_methods": list(completed_methods),
        "completed_tasks_per_method": {
            k: list(v) for k, v in (completed_tasks_per_method or {}).items()
        },
        "paper_valid": False,
        "partial": True,
    }
    if note is not None:
        payload["note"] = str(note)
    try:
        _atomic_write_partial(pd / PARTIAL_STATUS_FILENAME, payload)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Run id
# ---------------------------------------------------------------------------


def derive_run_id(args: Args, mode: str) -> str:
    methods_tag = "+".join(sorted(args.methods))
    e_tag = "_E" if args.use_e_split else ""
    # Include a calib tag only when knobs deviate from the paper-valid
    # defaults, so existing paper-valid run_ids do not change.
    needs_compressed = any(m != "fp16" for m in args.methods)
    calib_tag = ""
    if needs_compressed and is_reduced_calibration(args):
        calib_tag = (
            f"_calib{args.n_calib}_lm{args.lloyd_max_iter}_cs{args.calib_max_seq_tokens}"
        )
    suffix = (
        f"b{args.avg_bits}_seed{args.seed}_subset{args.subset}{e_tag}_"
        f"n{args.n_per_task}_in{args.max_input_tokens}_out{args.max_new_tokens}_"
        f"{methods_tag}{calib_tag}"
    )
    return eval_common.derive_run_id(FAMILY, args.model, suffix=suffix, mode=mode)


# ---------------------------------------------------------------------------
# Synthetic-smoke pipeline
# ---------------------------------------------------------------------------


def _synthetic_method_record(
    label: str, tasks: Sequence[str], *, n_per_task: int, base_score: float,
) -> Dict[str, Any]:
    """Deterministic toy LongBench scores per task."""
    per_task = []
    total = 0.0
    for i, t in enumerate(tasks):
        score = max(0.0, min(1.0, base_score + 0.01 * ((i % 3) - 1)))
        per_task.append({
            "task": t,
            "metric": TASK_METRIC.get(t, "qa_f1_en"),
            "score": float(score),
            "n_examples": int(n_per_task),
        })
        total += score
    macro = total / max(1, len(tasks))
    return {
        "label": label,
        "per_task": per_task,
        "aggregate": {"macro_score": float(macro)},
        "evidence_ids": [f"RUN-LONGBENCH-SMOKE-{label.upper()}"],
        "dataset_source": "synthetic",
    }


def run_synthetic_smoke(args: Args) -> Dict[str, Dict[str, Any]]:
    tasks = SUBSETS[args.subset]
    base_for = {
        "fp16": 0.41,
        "spectralquant_v2": 0.39,
        "spectralquant_v1": 0.37,
        "turboquant": 0.32,
        "official_turboquant": 0.33,
    }
    out: Dict[str, Dict[str, Any]] = {}
    for m in args.methods:
        out[m] = _synthetic_method_record(
            m, tasks,
            n_per_task=args.n_per_task,
            base_score=base_for.get(m, 0.30),
        )
    return out


# ---------------------------------------------------------------------------
# Inline-corpus pipeline (full HF model, deterministic in-memory tasks)
# ---------------------------------------------------------------------------


def _build_inline_longbench_examples(
    tasks: Sequence[str], n_per_task: int,
) -> Dict[str, List[Dict[str, str]]]:
    base_corpus = list(eval_common.INLINE_CORPUS)
    out: Dict[str, List[Dict[str, str]]] = {}
    for t in tasks:
        rows: List[Dict[str, str]] = []
        for i in range(n_per_task):
            ctx = base_corpus[i % len(base_corpus)]
            rows.append({
                "input": (
                    f"Read the passage and answer the question.\n\n"
                    f"Passage: {ctx}\n\n"
                    f"Question: What is the topic of this passage? "
                    f"Answer in two words. [task={t} idx={i}]"
                ),
                "answer": "compression research",
            })
        out[t] = rows
    return out


def _token_overlap_f1(pred: str, ref: str) -> float:
    p_tokens = pred.lower().split()
    r_tokens = ref.lower().split()
    if not p_tokens or not r_tokens:
        return 0.0
    common: Dict[str, int] = {}
    p_count: Dict[str, int] = {}
    r_count: Dict[str, int] = {}
    for tok in p_tokens:
        p_count[tok] = p_count.get(tok, 0) + 1
    for tok in r_tokens:
        r_count[tok] = r_count.get(tok, 0) + 1
    for tok, c in p_count.items():
        common[tok] = min(c, r_count.get(tok, 0))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p = overlap / sum(p_count.values())
    r = overlap / sum(r_count.values())
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _evaluate_fp16_inline(
    args: Args,
    examples_per_task: Dict[str, List[Dict[str, str]]],
    status: Optional[run_status.StatusWriter],
) -> Dict[str, Any]:
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if status is not None:
        status.emit("model_load_start", message=args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype]
    )
    device = torch.device(args.device if args.device != "cpu" else "cpu")
    model.to(device)
    model.eval()
    if status is not None:
        status.emit("model_load_end")

    per_task = []
    total_score = 0.0
    with torch.no_grad():
        for t, rows in examples_per_task.items():
            sliced = rows[: args.n_per_task]
            prog = GenerationProgress(status, "fp16", t, total=len(sliced))
            prog.start()
            scores: List[float] = []
            for r in sliced:
                ids = tok(
                    r["input"], return_tensors="pt",
                    truncation=True, max_length=args.max_input_tokens,
                ).input_ids.to(device)
                out_ids = model.generate(
                    ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                )
                completion = tok.decode(
                    out_ids[0, ids.size(1):], skip_special_tokens=True,
                )
                scores.append(_token_overlap_f1(completion, r["answer"]))
                prog.step(completion=completion)
            mean_score = sum(scores) / max(1, len(scores))
            per_task.append({
                "task": t,
                "metric": TASK_METRIC.get(t, "qa_f1_en"),
                "score": float(mean_score),
                "n_examples": int(len(scores)),
            })
            total_score += mean_score
            prog.end(score=mean_score)
            if status is not None:
                status.emit("eval_progress",
                            message=f"task={t} score={mean_score:.4f}")

    macro = total_score / max(1, len(per_task))
    return {
        "label": "fp16",
        "per_task": per_task,
        "aggregate": {"macro_score": float(macro)},
        "evidence_ids": ["RUN-LONGBENCH-INLINE-FP16"],
        "dataset_source": "inline_corpus",
    }


def _placeholder_method_record(
    method: str, tasks: Sequence[str], n_per_task: int,
) -> Tuple[Dict[str, Any], str]:
    per_task = []
    for t in tasks:
        per_task.append({
            "task": t,
            "metric": TASK_METRIC.get(t, "qa_f1_en"),
            "score": 0.0,
            "n_examples": int(n_per_task),
        })
    record = {
        "label": method,
        "per_task": per_task,
        "aggregate": {"macro_score": 0.0},
        "evidence_ids": [f"RUN-LONGBENCH-PLACEHOLDER-{method.upper()}"],
        "placeholder": True,
        "dataset_source": "none",
    }
    caveat = (
        f"method={method} is unsupported by this harness on this corpus."
    )
    return record, caveat


def _evaluate_compressed_inline(
    method: str,
    args: Args,
    examples_per_task: Dict[str, List[Dict[str, str]]],
    status: Optional[run_status.StatusWriter],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Real K/V replay path for inline-corpus LongBench scoring."""
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from experiments import sqv2_replay  # noqa: E402

    if status is not None:
        status.emit("model_load_start", message=f"{method} {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype]
    )
    device = torch.device(args.device if args.device != "cpu" else "cpu")
    model.to(device)
    model.eval()

    calib_texts: List[str] = []
    for t, rows in examples_per_task.items():
        for r in rows[:4]:
            calib_texts.append(r["input"])
    calib_texts = calib_texts[: args.n_calib]
    progress_cb = _make_progress_cb(status, method)
    if status is not None:
        status.emit(
            "calibration_start",
            message=(f"{method} n_calib={len(calib_texts)} "
                     f"lloyd_max_iter={args.lloyd_max_iter} "
                     f"max_seq_tokens={args.calib_max_seq_tokens}"),
        )
    bundle = sqv2_replay.build_calibrated_engine(
        model, tok, calib_texts,
        method=method, avg_bits=float(args.avg_bits),
        use_water_fill=(method == "spectralquant_v2"),
        n_calib_samples=len(calib_texts),
        max_seq_tokens=min(args.max_input_tokens, args.calib_max_seq_tokens),
        lloyd_max_iter=int(args.lloyd_max_iter),
        progress=progress_cb,
    )
    if status is not None:
        status.emit(
            "calibration_end",
            message=f"{method} calibrated_layers={len(bundle.calibrated_layers)}",
        )
    handle = sqv2_replay.attach_replay_hooks(
        model, bundle.engine,
        n_kv_heads=bundle.n_kv_heads, head_dim=bundle.head_dim,
        method=method, calibrated_layers=bundle.calibrated_layers,
    )

    per_task = []
    total_score = 0.0
    try:
        with torch.no_grad():
            for t, rows in examples_per_task.items():
                sliced = rows[: args.n_per_task]
                prog = GenerationProgress(status, method, t, total=len(sliced))
                prog.start()
                scores: List[float] = []
                for r in sliced:
                    ids = tok(
                        r["input"], return_tensors="pt", truncation=True,
                        max_length=args.max_input_tokens,
                    ).input_ids.to(device)
                    out_ids = model.generate(
                        ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                    )
                    completion = tok.decode(
                        out_ids[0, ids.size(1):], skip_special_tokens=True,
                    )
                    scores.append(_token_overlap_f1(completion, r["answer"]))
                    prog.step(completion=completion)
                mean_score = sum(scores) / max(1, len(scores))
                per_task.append({
                    "task": t,
                    "metric": TASK_METRIC.get(t, "qa_f1_en"),
                    "score": float(mean_score),
                    "n_examples": int(len(scores)),
                })
                total_score += mean_score
                prog.end(score=mean_score)
                if status is not None:
                    status.emit("eval_progress",
                                message=f"{method} task={t} score={mean_score:.4f}")
    finally:
        handle.remove()

    macro = total_score / max(1, len(per_task))
    coverage = handle.coverage_summary()
    rec = {
        "label": method,
        "per_task": per_task,
        "aggregate": {"macro_score": float(macro)},
        "evidence_ids": [f"RUN-LONGBENCH-INLINE-{method.upper()}"],
        "replay_coverage": coverage,
        "dataset_source": "inline_corpus",
    }
    diag = {"method": method, "coverage": coverage,
            "calibration": bundle.coverage}
    return rec, diag


# ---------------------------------------------------------------------------
# Full path: real LongBench from HF datasets + per-task official metrics
# ---------------------------------------------------------------------------


def _load_full_examples(
    args: Args,
    tokenizer: Any,
    status: Optional[run_status.StatusWriter],
) -> Dict[str, List[Any]]:
    """Load LongBench rows for the requested subset.

    Uses :mod:`experiments.longbench_dataset`, which wraps
    ``datasets.load_dataset("THUDM/LongBench", task)``.
    """
    from experiments import longbench_dataset

    tasks = SUBSETS[args.subset]
    out: Dict[str, List[Any]] = {}
    for t in tasks:
        if status is not None:
            status.emit("dataset_load_start",
                        message=f"task={t} n_per_task={args.n_per_task}")
        rows = longbench_dataset.load_longbench_task(
            t,
            n_per_task=args.n_per_task,
            max_input_tokens=args.max_input_tokens,
            tokenizer=tokenizer,
            seed=args.seed,
            use_e_split=args.use_e_split,
        )
        out[t] = rows
        if status is not None:
            status.emit("dataset_load_end",
                        message=f"task={t} loaded={len(rows)}")
    return out


def _generate_for_task(
    model: Any, tokenizer: Any, device: Any, args: Args, rows: List[Any],
    *,
    progress: Optional[GenerationProgress] = None,
) -> List[str]:
    """Decode ``rows`` for one task and emit per-example progress.

    The generation loop has two failure modes that are indistinguishable
    in the parent process: a slow per-example ``model.generate`` and a
    truly stuck one. The ``progress`` object converts both into observable
    state — a periodic ``eval_progress`` event with completed/total +
    elapsed_s, and an on-disk per-task partial checkpoint after every
    example. Without this, an interrupted run leaves no evidence of where
    it stopped (V1 stall mode, see runbook §7.7.4b).
    """
    import torch  # type: ignore[import-not-found]

    if progress is not None:
        progress.start()

    completions: List[str] = []
    with torch.no_grad():
        for row in rows:
            ids = tokenizer(
                row.prompt, return_tensors="pt", truncation=True,
                max_length=args.max_input_tokens,
            ).input_ids.to(device)
            mn = min(args.max_new_tokens, row.expected_max_new_tokens)
            out_ids = model.generate(
                ids, max_new_tokens=mn, do_sample=False,
            )
            completion = tokenizer.decode(
                out_ids[0, ids.size(1):], skip_special_tokens=True,
            ).strip()
            completions.append(completion)
            if progress is not None:
                progress.step(completion=completion)
    return completions


def _score_method_full(
    method: str,
    examples_per_task: Dict[str, List[Any]],
    completions_per_task: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Apply the per-task LongBench metric to every (rows, completions)."""
    from experiments import longbench_metrics

    per_task: List[Dict[str, Any]] = []
    total = 0.0
    for t, rows in examples_per_task.items():
        comps = completions_per_task.get(t, [])
        answers = [r.answers for r in rows]
        all_classes = [r.all_classes for r in rows]
        scored = longbench_metrics.score_task(
            t, comps, answers, all_classes_per_example=all_classes,
        )
        # Convert from [0, 100] (LongBench convention) to [0, 1] for the JSON.
        score_01 = float(scored["score_0_100"]) / 100.0
        per_task.append({
            "task": t,
            "metric": scored["metric"],
            "score": score_01,
            "score_0_100": float(scored["score_0_100"]),
            "n_examples": int(scored["n_examples"]),
        })
        total += score_01
    macro = total / max(1, len(per_task))
    return {
        "label": method,
        "per_task": per_task,
        "aggregate": {"macro_score": float(macro)},
        "evidence_ids": [f"RUN-LONGBENCH-{method.upper()}"],
    }


def _evaluate_full_fp16(
    args: Args,
    examples_per_task: Dict[str, List[Any]],
    status: Optional[run_status.StatusWriter],
) -> Dict[str, Any]:
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if status is not None:
        status.emit("model_load_start", message=f"fp16 {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype]
    )
    device = torch.device(args.device if args.device != "cpu" else "cpu")
    model.to(device)
    model.eval()
    if status is not None:
        status.emit("model_load_end")

    completions_per_task: Dict[str, List[str]] = {}
    completed_tasks: List[str] = []
    for t, rows in examples_per_task.items():
        if status is not None:
            status.emit("eval_progress", message=f"fp16 generating task={t}")
        _write_run_partial_status(
            status, args=args, mode=eval_common.MODE_FULL,
            methods_planned=list(args.methods),
            tasks_planned=list(examples_per_task.keys()),
            current_method="fp16", current_task=t,
            completed_methods=[],
            completed_tasks_per_method={"fp16": list(completed_tasks)},
        )
        prog = GenerationProgress(status, "fp16", t, total=len(rows))
        completions_per_task[t] = _generate_for_task(
            model, tok, device, args, rows, progress=prog,
        )
        prog.end()
        completed_tasks.append(t)

    rec = _score_method_full("fp16", examples_per_task, completions_per_task)
    rec["dataset_source"] = "huggingface_thudm"
    return rec


def _evaluate_full_compressed(
    method: str,
    args: Args,
    examples_per_task: Dict[str, List[Any]],
    status: Optional[run_status.StatusWriter],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from experiments import sqv2_replay

    if status is not None:
        status.emit("model_load_start", message=f"{method} {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype]
    )
    device = torch.device(args.device if args.device != "cpu" else "cpu")
    model.to(device)
    model.eval()

    # Calibrate on a held-out chunk: use first 4 prompts per task, capped at n_calib.
    calib_texts: List[str] = []
    for t, rows in examples_per_task.items():
        for r in rows[:4]:
            calib_texts.append(r.prompt)
    calib_texts = calib_texts[: args.n_calib]
    if status is not None:
        status.emit(
            "calibration_start",
            message=(
                f"{method} n_calib={len(calib_texts)} "
                f"lloyd_max_iter={args.lloyd_max_iter} "
                f"calib_max_seq_tokens={args.calib_max_seq_tokens}"
            ),
        )
    progress_cb = _make_progress_cb(status, method)
    bundle = sqv2_replay.build_calibrated_engine(
        model, tok, calib_texts,
        method=method, avg_bits=float(args.avg_bits),
        use_water_fill=(method == "spectralquant_v2"),
        n_calib_samples=len(calib_texts),
        max_seq_tokens=min(args.max_input_tokens, args.calib_max_seq_tokens),
        lloyd_max_iter=int(args.lloyd_max_iter),
        progress=progress_cb,
    )
    handle = sqv2_replay.attach_replay_hooks(
        model, bundle.engine,
        n_kv_heads=bundle.n_kv_heads, head_dim=bundle.head_dim,
        method=method, calibrated_layers=bundle.calibrated_layers,
    )
    if status is not None:
        status.emit("calibration_end",
                    message=f"calibrated_layers={len(bundle.calibrated_layers)}")

    completions_per_task: Dict[str, List[str]] = {}
    completed_tasks: List[str] = []
    try:
        for t, rows in examples_per_task.items():
            if status is not None:
                status.emit("eval_progress", message=f"{method} task={t}")
            _write_run_partial_status(
                status, args=args, mode=eval_common.MODE_FULL,
                methods_planned=list(args.methods),
                tasks_planned=list(examples_per_task.keys()),
                current_method=method, current_task=t,
                completed_methods=[],
                completed_tasks_per_method={method: list(completed_tasks)},
            )
            prog = GenerationProgress(status, method, t, total=len(rows))
            completions_per_task[t] = _generate_for_task(
                model, tok, device, args, rows, progress=prog,
            )
            prog.end()
            completed_tasks.append(t)
    finally:
        handle.remove()

    coverage = handle.coverage_summary()
    rec = _score_method_full(method, examples_per_task, completions_per_task)
    rec["dataset_source"] = "huggingface_thudm"
    rec["replay_coverage"] = coverage
    diag = {"method": method, "coverage": coverage,
            "calibration": bundle.coverage}
    return rec, diag


# ---------------------------------------------------------------------------
# Build payload + main
# ---------------------------------------------------------------------------


def build_payload(
    args: Args,
    argv: Sequence[str],
    mode: str,
    methods: Dict[str, Dict[str, Any]],
    *,
    extra_caveats: Sequence[str] = (),
) -> Dict[str, Any]:
    methods_real = all(m in REAL_EVAL_METHODS for m in args.methods)
    no_placeholder = not any(rec.get("placeholder") for rec in methods.values())
    coverage_ok = True
    for m, rec in methods.items():
        if m == "fp16":
            continue
        cov = rec.get("replay_coverage") or {}
        if cov.get("fraction_layers_real", 0.0) < 0.99:
            coverage_ok = False
            break
    # In the full path every method record must have dataset_source ==
    # "huggingface_thudm". Inline-corpus runs are NEVER paper_valid.
    real_dataset = all(
        rec.get("dataset_source") == "huggingface_thudm"
        for rec in methods.values()
    )
    # Reduced-calibration runs are explicitly NOT paper-valid even if
    # every other gate passes. The harness must never silently "weaken"
    # paper-valid gates: an operator who set lower n_calib/lloyd_max_iter/
    # calib_max_seq_tokens to keep a CPU-bound run tractable gets a clearly
    # caveated, schema-valid artifact, but paper_valid stays false.
    reduced_calibration = is_reduced_calibration(args)
    needs_compressed_calibration = any(
        m in REAL_EVAL_METHODS and m != "fp16" for m in args.methods
    )
    paper_valid = (
        mode == eval_common.MODE_FULL
        and args.n_per_task >= 50
        and methods_real
        and no_placeholder
        and coverage_ok
        and real_dataset
        and "fp16" in methods
        and not (needs_compressed_calibration and reduced_calibration)
    )
    payload = eval_common.base_payload(
        family=FAMILY,
        run_id=derive_run_id(args, mode),
        argv=argv,
        model_name=args.model,
        mode=mode,
        paper_valid=paper_valid,
        device=args.device,
        git_commit_override=args.git_commit_override,
    )
    payload["subset"] = {
        "name": args.subset,
        "n_per_task": int(args.n_per_task),
        "max_input_tokens": int(args.max_input_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "use_e_split": bool(args.use_e_split),
    }
    payload["calibration"] = {
        "n_calib": int(args.n_calib),
        "calib_max_seq_tokens": int(args.calib_max_seq_tokens),
        "lloyd_max_iter": int(args.lloyd_max_iter),
        "calibration_mode": str(args.calibration_mode),
        "reduced_calibration": bool(reduced_calibration),
        "paper_valid_thresholds": {
            "n_calib_min": int(PAPER_VALID_N_CALIB_MIN),
            "lloyd_max_iter_min": int(PAPER_VALID_LLOYD_MAX_ITER),
            "calib_max_seq_tokens_min": int(PAPER_VALID_CALIB_MAX_SEQ_TOKENS_MIN),
        },
    }
    payload["tasks"] = list(SUBSETS[args.subset])
    payload["methods"] = methods
    payload["evidence_ids"] = [
        f"RUN-LONGBENCH-{eval_common.model_short(args.model).upper()}-{args.subset.upper()}"
    ]
    caveats: List[str] = []
    if not paper_valid:
        if mode != eval_common.MODE_FULL:
            caveats.append(
                f"mode={mode}: not paper-valid evidence."
            )
        if args.n_per_task < 50:
            caveats.append(
                f"n_per_task={args.n_per_task} < 50: LongBench requires "
                f"n>=50/task per V1-GAP-008."
            )
        if not methods_real:
            caveats.append(
                f"At least one requested method is outside the real-eval "
                f"set {list(REAL_EVAL_METHODS)}; paper_valid=false."
            )
        if not no_placeholder:
            caveats.append(
                "At least one method record carries placeholder=true; "
                "paper_valid=false."
            )
        if not coverage_ok:
            caveats.append(
                "Non-FP16 method has replay_coverage < 0.99; partial "
                "coverage cannot be paper-valid."
            )
        if not real_dataset:
            caveats.append(
                "At least one method record has dataset_source != "
                "'huggingface_thudm'; only HF THUDM/LongBench can be "
                "paper-valid."
            )
    if mode == eval_common.MODE_INLINE_CORPUS_SMOKE:
        caveats.append(
            "Inline-corpus LongBench runs use a synthetic corpus and are "
            "harness-validation only. The full path uses HF datasets "
            "THUDM/LongBench with the upstream task templates and "
            "metrics; see longbench_dataset.py and longbench_metrics.py."
        )
    if mode == eval_common.MODE_FULL and args.subset != "full":
        caveats.append(
            f"subset={args.subset}: this artifact scores a *transparent "
            f"subset* of LongBench tasks ({list(SUBSETS[args.subset])}); "
            f"do not headline it as full LongBench."
        )
    if reduced_calibration and needs_compressed_calibration:
        knobs = []
        if args.n_calib < PAPER_VALID_N_CALIB_MIN:
            knobs.append(
                f"n_calib={args.n_calib}<{PAPER_VALID_N_CALIB_MIN}"
            )
        if args.lloyd_max_iter < PAPER_VALID_LLOYD_MAX_ITER:
            knobs.append(
                f"lloyd_max_iter={args.lloyd_max_iter}<{PAPER_VALID_LLOYD_MAX_ITER}"
            )
        if args.calib_max_seq_tokens < PAPER_VALID_CALIB_MAX_SEQ_TOKENS_MIN:
            knobs.append(
                f"calib_max_seq_tokens={args.calib_max_seq_tokens}<"
                f"{PAPER_VALID_CALIB_MAX_SEQ_TOKENS_MIN}"
            )
        caveats.append(
            "reduced_calibration: " + ", ".join(knobs) +
            ". Calibration was bounded for tractability; paper_valid is "
            "false even if every other gate passes. Re-run with "
            "--calibration-mode paper to clear this caveat."
        )
    if args.calibration_mode == "smoke" and needs_compressed_calibration:
        caveats.append(
            "calibration_mode=smoke: bounded paper-candidate configuration "
            "intended for harness validation and CPU-bound runs. Schema-valid "
            "but never paper_valid. Treat scores as plumbing checks only."
        )
    caveats.extend(extra_caveats)
    payload["caveats"] = caveats
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    mode = eval_common.resolve_mode(args)

    run_id = derive_run_id(args, mode)
    out_path = eval_common.output_path(args.output_dir, run_id)

    print(f"[run_longbench] {mode}: model={args.model} subset={args.subset}")
    print(f"[run_longbench] {mode}: output={out_path}")

    if out_path.exists() and args.skip_if_exists and not args.force:
        print(f"[run_longbench] {mode}: skip (exists): {out_path}")
        return 0

    if mode == eval_common.MODE_DRY_RUN:
        print(f"[run_longbench] dry-run: would write {out_path}")
        return 0

    status: Optional[run_status.StatusWriter] = None
    if args.status_dir is not None:
        sd = Path(args.status_dir) / run_id
        status = run_status.StatusWriter(
            status_dir=sd,
            run_id=run_id,
            commit=eval_common.git_commit(override=args.git_commit_override),
            model=args.model,
        )
        status.emit("start", message=f"family={FAMILY} mode={mode}")

    if mode == eval_common.MODE_SYNTHETIC_SMOKE:
        if status is not None:
            status.emit("eval_start", message="synthetic_smoke")
        methods = run_synthetic_smoke(args)
        eval_common.assert_method_keys(methods)
        payload = build_payload(args, raw_argv, mode, methods)
        eval_common.atomic_write_json(out_path, payload, schema_path=SCHEMA_PATH)
        print(f"[run_longbench] synthetic-smoke: wrote {out_path}")
        if status is not None:
            status.emit("success", message=str(out_path))
        return 0

    if mode == eval_common.MODE_INLINE_CORPUS_SMOKE:
        try:
            tasks = SUBSETS[args.subset]
            examples = _build_inline_longbench_examples(tasks, args.n_per_task)
            methods: Dict[str, Dict[str, Any]] = {}
            extra_caveats: List[str] = []
            for m in args.methods:
                if m == "fp16":
                    methods[m] = _evaluate_fp16_inline(args, examples, status)
                elif m in REAL_EVAL_METHODS:
                    rec, _diag = _evaluate_compressed_inline(
                        m, args, examples, status,
                    )
                    methods[m] = rec
                else:
                    rec, caveat = _placeholder_method_record(m, tasks, args.n_per_task)
                    methods[m] = rec
                    extra_caveats.append(caveat)
            eval_common.assert_method_keys(methods)
            payload = build_payload(
                args, raw_argv, mode, methods, extra_caveats=extra_caveats,
            )
            eval_common.atomic_write_json(out_path, payload, schema_path=SCHEMA_PATH)
            print(f"[run_longbench] inline-corpus-smoke: wrote {out_path}")
            if status is not None:
                status.emit("success", message=str(out_path))
            return 0
        except BaseException as exc:
            if status is not None:
                status.emit_failure(exc)
            raise

    # ---- Full path: HF datasets THUDM/LongBench + vendored metrics. ----
    try:
        from transformers import (  # type: ignore[import-not-found]
            AutoTokenizer,
        )
        # Load tokenizer once for prompt-truncation; the per-method
        # _evaluate_full_* functions reload it (cheap, cached).
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        examples_per_task = _load_full_examples(args, tok, status)

        methods: Dict[str, Dict[str, Any]] = {}
        extra_caveats: List[str] = []
        tasks = SUBSETS[args.subset]
        completed_methods: List[str] = []
        for m in args.methods:
            _write_run_partial_status(
                status, args=args, mode=mode,
                methods_planned=list(args.methods),
                tasks_planned=list(examples_per_task.keys()),
                current_method=m, current_task=None,
                completed_methods=list(completed_methods),
                note=f"starting method={m}",
            )
            if m == "fp16":
                methods[m] = _evaluate_full_fp16(args, examples_per_task, status)
            elif m in REAL_EVAL_METHODS:
                rec, _diag = _evaluate_full_compressed(
                    m, args, examples_per_task, status,
                )
                methods[m] = rec
            else:
                rec, caveat = _placeholder_method_record(
                    m, tasks, args.n_per_task,
                )
                methods[m] = rec
                extra_caveats.append(caveat)
            _write_method_partial_record(status, method=m, record=methods[m])
            completed_methods.append(m)
            _write_run_partial_status(
                status, args=args, mode=mode,
                methods_planned=list(args.methods),
                tasks_planned=list(examples_per_task.keys()),
                current_method=None, current_task=None,
                completed_methods=list(completed_methods),
                note=f"finished method={m}",
            )

        eval_common.assert_method_keys(methods)
        payload = build_payload(
            args, raw_argv, mode, methods, extra_caveats=extra_caveats,
        )
        eval_common.atomic_write_json(out_path, payload, schema_path=SCHEMA_PATH)
        print(f"[run_longbench] full: wrote {out_path}")
        if status is not None:
            status.emit("success", message=str(out_path))
        return 0
    except BaseException as exc:
        if status is not None:
            status.emit_failure(exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
