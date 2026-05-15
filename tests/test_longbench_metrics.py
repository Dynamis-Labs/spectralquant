"""Tests for the vendored LongBench task metrics.

These tests verify our re-implementations of the upstream LongBench
metrics return values consistent with the reference behavior on small
deterministic inputs. They avoid network access — the
``longbench_dataset.load_longbench_task`` HF loader has its own
contract; here we only test the metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


@pytest.fixture(autouse=True)
def _add_paths(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.syspath_prepend(str(SRC_DIR))


def test_qa_f1_en_exact_match_is_100():
    from experiments import longbench_metrics
    assert longbench_metrics.qa_f1_en("Paris", "Paris") == 100.0


def test_qa_f1_en_no_overlap_is_0():
    from experiments import longbench_metrics
    assert longbench_metrics.qa_f1_en("foo bar", "baz qux") == 0.0


def test_qa_f1_en_partial_overlap_in_range():
    from experiments import longbench_metrics
    s = longbench_metrics.qa_f1_en("the answer is paris", "paris")
    assert 0.0 < s < 100.0


def test_qa_f1_en_strips_articles_and_punctuation():
    from experiments import longbench_metrics
    s1 = longbench_metrics.qa_f1_en("The Paris.", "Paris")
    assert s1 == 100.0  # SQuAD-normalize drops "the" + period


def test_classification_score_single_match():
    from experiments import longbench_metrics
    s = longbench_metrics.classification_score(
        "The answer is HUM:ind",
        "HUM:ind",
        all_classes=["HUM:ind", "LOC:city", "NUM:date"],
    )
    assert s == 100.0


def test_classification_score_ambiguous_returns_zero():
    from experiments import longbench_metrics
    s = longbench_metrics.classification_score(
        "The class is HUM:ind or LOC:city",
        "HUM:ind",
        all_classes=["HUM:ind", "LOC:city", "NUM:date"],
    )
    assert s == 0.0  # multiple labels matched -> 0


def test_retrieval_score_en_correct():
    from experiments import longbench_metrics
    s = longbench_metrics.retrieval_score_en(
        "Paragraph 17", "Paragraph 17",
    )
    assert s == 100.0


def test_retrieval_score_en_wrong_index():
    from experiments import longbench_metrics
    s = longbench_metrics.retrieval_score_en(
        "Paragraph 5", "Paragraph 17",
    )
    assert s == 0.0


def test_count_score_exact():
    from experiments import longbench_metrics
    assert longbench_metrics.count_score("The answer is 7", "7") == 100.0


def test_count_score_wrong():
    from experiments import longbench_metrics
    assert longbench_metrics.count_score("12", "7") == 0.0


def test_code_sim_score_identical():
    from experiments import longbench_metrics
    s = longbench_metrics.code_sim_score(
        "x = 1\nfoo()", "x = 1\nbar()",
    )
    assert s == 100.0  # only first line is compared


def test_rouge_l_lcs_f_self_is_one():
    from experiments import longbench_metrics
    f = longbench_metrics._rouge_l_lcs_f(
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
    )
    assert f == 1.0


def test_score_task_dispatches_correctly():
    from experiments import longbench_metrics
    out = longbench_metrics.score_task(
        "narrativeqa",
        ["paris is the answer"],
        [["paris"]],
    )
    assert out["metric"] == "qa_f1_en"
    assert 0.0 < out["score_0_100"] <= 100.0
    assert out["n_examples"] == 1


def test_score_task_takes_max_over_gold_answers():
    from experiments import longbench_metrics
    out = longbench_metrics.score_task(
        "hotpotqa",
        ["new york"],
        [["new york", "paris"]],  # max should pick new york -> 100
    )
    assert out["score_0_100"] == 100.0


def test_task_metric_names_complete_for_known_subsets():
    from experiments import longbench_metrics, run_longbench
    for tasks in run_longbench.SUBSETS.values():
        for t in tasks:
            assert t in longbench_metrics.TASK_METRIC_NAMES, (
                f"task {t!r} missing from metric registry"
            )
