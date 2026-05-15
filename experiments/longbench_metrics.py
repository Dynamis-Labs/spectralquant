"""Transparent in-repo implementations of LongBench task metrics.

LongBench (https://github.com/THUDM/LongBench) ships a set of per-task
metrics in ``LongBench/metrics.py``. To produce a paper-valid LongBench
artifact without taking a hard runtime dependency on the upstream repo
we vendor a *minimal, well-commented* re-implementation of the metrics
that are needed to score the public ``THUDM/LongBench`` HF dataset.

Scope
-----

* We reimplement: ``qa_f1_score`` (English / Chinese), ``rouge_score``,
  ``classification_score``, ``retrieval_score`` (English / Chinese),
  ``count_score``, and ``code_sim_score``.
* We do NOT reimplement the official LLM-judge "longbench_e" metric.
  The dataset rows we score below are all rule-based (string F1 / ROUGE
  / exact match / code edit-distance) so a transparent re-implementation
  is faithful to what the upstream repo computes.
* Where Chinese tokenization is involved (qa_f1_zh, retrieval_score_zh,
  rouge for ``vcsum``), we use ``jieba`` if it is importable, falling
  back to character-level tokenization otherwise. The fallback is
  recorded in the metric's caveat.
* ROUGE: we use the ``rouge`` package's ``Rouge`` class if present,
  otherwise a faithful in-repo re-implementation of ROUGE-L F-measure
  (LCS-based) so the result is computable on any Python environment.

These metrics return a float in ``[0, 100]`` to match the upstream
convention; the harness divides by 100 before storing macro_score so
the JSON's score field is in ``[0, 1]`` (consistent with the inline
corpus path and downstream tooling).

Reference: ``LongBench/metrics.py`` upstream — same per-task switch is
re-implemented in :func:`score_task` below.
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _normalize_answer_en(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_en(s: str) -> List[str]:
    return _normalize_answer_en(s).split()


_JIEBA_AVAILABLE: Optional[bool] = None


def _have_jieba() -> bool:
    global _JIEBA_AVAILABLE
    if _JIEBA_AVAILABLE is None:
        try:
            import jieba  # type: ignore  # noqa: F401
            _JIEBA_AVAILABLE = True
        except Exception:
            _JIEBA_AVAILABLE = False
    return _JIEBA_AVAILABLE


def _tokenize_zh(s: str) -> List[str]:
    """Chinese tokenization. Prefers jieba; falls back to per-character."""
    s = s.lower().strip()
    s = re.sub(r"\s+", "", s)
    if _have_jieba():
        import jieba  # type: ignore
        return [t for t in jieba.lcut(s) if t.strip()]
    return [ch for ch in s if ch.strip()]


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


def _f1_from_tokens(p_tokens: Sequence[str], r_tokens: Sequence[str]) -> float:
    if not p_tokens or not r_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(r_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p = overlap / len(p_tokens)
    r = overlap / len(r_tokens)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def qa_f1_en(prediction: str, ground_truth: str) -> float:
    """SQuAD-style English token F1 in [0, 100]."""
    return 100.0 * _f1_from_tokens(
        _tokenize_en(prediction), _tokenize_en(ground_truth),
    )


def qa_f1_zh(prediction: str, ground_truth: str) -> float:
    return 100.0 * _f1_from_tokens(
        _tokenize_zh(prediction), _tokenize_zh(ground_truth),
    )


def _rouge_l_lcs_f(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    """ROUGE-L F-measure from token LCS, in [0, 1]."""
    n, m = len(pred_tokens), len(ref_tokens)
    if n == 0 or m == 0:
        return 0.0
    # DP for LCS length.
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[m]
    if lcs == 0:
        return 0.0
    p = lcs / n
    r = lcs / m
    if p + r == 0:
        return 0.0
    # ROUGE-L F (beta=1).
    return 2 * p * r / (p + r)


def rouge_score_en(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1, English. Returns [0, 100]."""
    try:
        from rouge import Rouge  # type: ignore
        r = Rouge()
        scores = r.get_scores(prediction.strip() or " ",
                              ground_truth.strip() or " ")
        f = scores[0]["rouge-l"]["f"]
        return 100.0 * float(f)
    except Exception as exc:  # noqa: BLE001
        logger.debug("rouge package unavailable (%s); using LCS fallback", exc)
        return 100.0 * _rouge_l_lcs_f(
            _tokenize_en(prediction), _tokenize_en(ground_truth),
        )


def rouge_score_zh(prediction: str, ground_truth: str) -> float:
    return 100.0 * _rouge_l_lcs_f(
        _tokenize_zh(prediction), _tokenize_zh(ground_truth),
    )


def classification_score(
    prediction: str, ground_truth: str, *, all_classes: Sequence[str],
) -> float:
    """Multi-class classification: 1 iff exactly one class label appears
    in the prediction *and* it matches the ground truth.

    Mirrors LongBench upstream: rejects predictions that match multiple
    candidate labels (ambiguous) by returning 0.
    """
    if not all_classes:
        return 100.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
    pred = prediction.lower()
    em_match = []
    for c in all_classes:
        if c.lower() in pred:
            em_match.append(c)
    if len(em_match) != 1:
        return 0.0
    return 100.0 if em_match[0].lower() == ground_truth.lower() else 0.0


def retrieval_score_en(prediction: str, ground_truth: str) -> float:
    """passage_retrieval_en: ground truth is "Paragraph K"; we extract
    the integer the model emitted and compare. Upstream uses a regex.
    """
    pattern = re.compile(r"Paragraph\s+(\d+)", re.IGNORECASE)
    m_p = pattern.search(prediction)
    m_g = pattern.search(ground_truth)
    if not m_p or not m_g:
        return 0.0
    return 100.0 if m_p.group(1) == m_g.group(1) else 0.0


def retrieval_score_zh(prediction: str, ground_truth: str) -> float:
    """passage_retrieval_zh — Chinese paragraph retrieval. Looks for
    ``段落X`` (paragraph X) in both sides."""
    pattern = re.compile(r"段落\s*(\d+)")
    m_p = pattern.search(prediction)
    m_g = pattern.search(ground_truth)
    if not m_p or not m_g:
        return 0.0
    return 100.0 if m_p.group(1) == m_g.group(1) else 0.0


def count_score(prediction: str, ground_truth: str) -> float:
    """passage_count: exact-match on the integer count."""
    pat = re.compile(r"-?\d+")
    m_p = pat.search(prediction)
    if not m_p:
        return 0.0
    try:
        gt_int = int(ground_truth.strip())
    except Exception:
        return 0.0
    try:
        pred_int = int(m_p.group(0))
    except Exception:
        return 0.0
    return 100.0 if pred_int == gt_int else 0.0


def code_sim_score(prediction: str, ground_truth: str) -> float:
    """LCC / repobench-p similarity: token-level edit similarity.

    Upstream uses :func:`difflib.SequenceMatcher.ratio`. We do the same.
    """
    import difflib
    pred = prediction.split("\n")[0] if prediction else ""
    gt = ground_truth.split("\n")[0] if ground_truth else ""
    if not pred and not gt:
        return 100.0
    return 100.0 * difflib.SequenceMatcher(None, pred, gt).ratio()


# ---------------------------------------------------------------------------
# Per-task dispatch (mirrors upstream LongBench task → metric mapping)
# ---------------------------------------------------------------------------


_TASK_SCORERS: Dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_en,
    "qasper": qa_f1_en,
    "multifieldqa_en": qa_f1_en,
    "hotpotqa": qa_f1_en,
    "2wikimqa": qa_f1_en,
    "musique": qa_f1_en,
    "triviaqa": qa_f1_en,
    "multifieldqa_zh": qa_f1_zh,
    "dureader": rouge_score_zh,
    "vcsum": rouge_score_zh,
    "gov_report": rouge_score_en,
    "qmsum": rouge_score_en,
    "multi_news": rouge_score_en,
    "samsum": rouge_score_en,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score_en,
    "passage_retrieval_zh": retrieval_score_zh,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    # trec / lsht use classification_score with a per-row class list.
    "trec": classification_score,
    "lsht": classification_score,
}


#: Canonical metric *name* per task (for the JSON's per-task ``metric``
#: field). Mirrors the upstream metric naming so a downstream consumer
#: can match against the published LongBench table.
TASK_METRIC_NAMES: Dict[str, str] = {
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


def score_example(
    task: str,
    prediction: str,
    answers: Sequence[str],
    *,
    all_classes: Optional[Sequence[str]] = None,
) -> float:
    """Score one example. Takes the *max* over all listed gold answers.

    Returns the score in ``[0, 100]`` (LongBench convention).
    """
    if task not in _TASK_SCORERS:
        raise KeyError(f"task {task!r} not in LongBench metric registry")
    fn = _TASK_SCORERS[task]
    if not answers:
        return 0.0
    if task in ("trec", "lsht"):
        # Classification: take max over each gold label.
        ac = list(all_classes or [])
        scores = [fn(prediction, gt, all_classes=ac) for gt in answers]
        return float(max(scores)) if scores else 0.0
    scores = [fn(prediction, gt) for gt in answers]
    return float(max(scores)) if scores else 0.0


def score_task(
    task: str,
    predictions: Sequence[str],
    answers_per_example: Sequence[Sequence[str]],
    *,
    all_classes_per_example: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Score a list of predictions for one task.

    Returns ``{"score_0_100": float, "n_examples": int, "metric": str}``.
    """
    if len(predictions) != len(answers_per_example):
        raise ValueError(
            f"predictions ({len(predictions)}) != answers ({len(answers_per_example)})"
        )
    n = len(predictions)
    if n == 0:
        return {
            "score_0_100": 0.0, "n_examples": 0,
            "metric": TASK_METRIC_NAMES.get(task, "unknown"),
        }
    total = 0.0
    for i in range(n):
        ac = (all_classes_per_example[i] if all_classes_per_example
              else None)
        total += score_example(task, predictions[i],
                               answers_per_example[i], all_classes=ac)
    return {
        "score_0_100": float(total / n),
        "n_examples": int(n),
        "metric": TASK_METRIC_NAMES.get(task, "unknown"),
    }
