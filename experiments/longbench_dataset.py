"""LongBench dataset loader — HF datasets ``THUDM/LongBench`` adapter.

This module wraps :func:`datasets.load_dataset` so the harness can
treat the upstream LongBench tasks as a uniform list of
``(prompt_text, gold_answers, all_classes)`` rows. The upstream
dataset stores per-task config splits (``narrativeqa``, ``qasper``,
``hotpotqa``, …) under the ``THUDM/LongBench`` HF id; each row has
``input``, ``context``, ``answers``, ``length``, ``dataset``,
``language``, ``all_classes``, ``_id``.

We:

* Build the prompt by formatting the upstream task template
  (``input + context``) with the per-task header used in
  LongBench/config/dataset2prompt.json. To avoid a hard dependency on
  the upstream JSON file, we vendor the canonical templates inline.
* Truncate prompts to ``max_input_tokens`` using a tokenizer-aware
  policy (mirrors upstream's truncation rule: keep the *middle* of the
  context out and concatenate head + tail when too long).
* Return deterministic example slices for ``--n-per-task`` so reruns
  reproduce.

If ``datasets`` cannot reach HuggingFace, this module raises a
clear error; the caller (``run_longbench``) translates that into a
``paper_valid=false`` artifact + non-zero return code.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Upstream HF dataset id and the path of the bundled data archive.
_LONGBENCH_REPO_ID = "THUDM/LongBench"
_LONGBENCH_DATA_ZIP = "data.zip"

#: Tasks that exist as ``<task>_e`` configs in LongBench-E (English-only,
#: uniform-length distribution variant). The full LongBench dataset has
#: 21 tasks; only a subset has an ``_e`` mirror.
LONGBENCH_E_TASKS: Tuple[str, ...] = (
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
)

# Process-wide lock so concurrent task loads don't redundantly download
# or extract the same data.zip.
_extract_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Per-task prompt templates and answer-length budgets (vendored from
# THUDM/LongBench/config/dataset2prompt.json + dataset2maxlen.json so
# this loader is self-contained).
# ---------------------------------------------------------------------------

DATASET2PROMPT: Dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\nNow, answer the question based on the story as "
        "concisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        "article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nArticle: {context}\n\nAnswer the question based on the "
        "above article as concisely as you can, using a single phrase or sentence "
        "if possible. If the question cannot be answered based on the information "
        "in the article, write \"unanswerable\". If the question is a yes/no "
        "question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer "
        "the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "multifieldqa_zh": (
        "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文字回答下面的问题，"
        "只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答："
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\n"
        "Answer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\n"
        "Answer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\n"
        "Answer:"
    ),
    "dureader": (
        "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。"
        "\n\n问题：{input}\n回答："
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary "
        "of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of "
        "the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or "
        "instruction. Answer the query in one or more sentences.\n\nTranscript:\n"
        "{context}\n\nNow, answer the query based on the above meeting transcript "
        "in one or more sentences.\n\nQuery: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all "
        "news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the "
        "news.\n\nSummary:"
    ),
    "vcsum": (
        "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n"
        "会议总结："
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples "
        "of questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer "
        "and do not output any other words. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may "
        "be duplicates. Please carefully read these paragraphs and determine how "
        "many unique paragraphs there are after removing duplicates. In other "
        "words, how many non-repeating paragraphs are there in total?\n\n{context}"
        "\n\nPlease enter the final count of unique paragraphs after removing "
        "duplicates. The output format should only contain the number, such as 1, "
        "2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n{context}\n\nThe "
        "following is an abstract.\n\n{input}\n\nPlease enter the number of the "
        "paragraph that the abstract is from. The answer format must be like "
        "\"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "passage_retrieval_zh": (
        "以下是若干段落文字，以及一个摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n"
        "\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是："
    ),
    "lcc": (
        "Please complete the code given below. \n{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line of "
        "code:\n"
    ),
}

# Max generation length per task (mirrors upstream ``dataset2maxlen.json``).
DATASET2MAXLEN: Dict[str, int] = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "multifieldqa_zh": 64, "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "dureader": 128, "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "vcsum": 512, "trec": 64, "triviaqa": 32, "samsum": 128, "lsht": 64,
    "passage_count": 32, "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32, "lcc": 64, "repobench-p": 64,
}


@dataclass
class LongBenchExample:
    task: str
    prompt: str
    answers: List[str]
    all_classes: List[str]
    length: int
    example_id: str
    expected_max_new_tokens: int


def _truncate_middle(
    text: str, tokenizer: Any, max_tokens: int,
) -> str:
    """LongBench-style middle truncation: keep head + tail tokens, drop the
    middle. Implements the upstream truncation policy.
    """
    if max_tokens <= 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    head = max_tokens // 2
    tail = max_tokens - head
    keep = ids[:head] + ids[-tail:]
    try:
        return tokenizer.decode(keep, skip_special_tokens=True)
    except Exception:
        return text  # tokenizer doesn't support decode round-trip


def _longbench_extract_root(env: Optional[Dict[str, str]] = None) -> Path:
    """Pick a stable directory for the extracted ``data.zip`` payload.

    Prefers ``HF_DATASETS_CACHE`` (which the Modal launcher points at the
    persistent volume via :func:`run_status.configure_persistent_hf_cache`),
    falling back to ``HF_HOME``, ``XDG_CACHE_HOME``, then ``~/.cache``.
    """
    src = env if env is not None else os.environ
    base: Optional[str] = None
    for key in ("HF_DATASETS_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        v = src.get(key)
        if v:
            base = v
            break
    if base is None:
        base = str(Path.home() / ".cache")
    return Path(base) / "longbench_thudm_data"


def _hf_hub_download_data_zip() -> str:
    """Download ``data.zip`` from ``THUDM/LongBench`` via huggingface_hub.

    Returns the local path to the cached zip. Raises a clear RuntimeError
    if huggingface_hub is unavailable or the download fails.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to load THUDM/LongBench under "
            "current `datasets` versions (which no longer support dataset "
            "scripts). `pip install huggingface_hub>=0.20`."
        ) from exc
    try:
        return hf_hub_download(
            repo_id=_LONGBENCH_REPO_ID,
            filename=_LONGBENCH_DATA_ZIP,
            repo_type="dataset",
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to download {_LONGBENCH_REPO_ID}/{_LONGBENCH_DATA_ZIP} "
            f"via huggingface_hub: {exc}. Check internet access / HF token "
            f"/ cache permissions."
        ) from exc


def _ensure_extracted(extract_root: Path) -> Path:
    """Download (if needed) and extract ``data.zip`` exactly once.

    Idempotent and safe under multi-threaded callers via a process-wide
    lock. Returns the directory containing the per-task ``*.jsonl`` files
    (i.e. the ``data/`` folder inside the archive).
    """
    extract_root.mkdir(parents=True, exist_ok=True)
    sentinel = extract_root / ".extracted_ok"
    data_dir = extract_root / "data"
    with _extract_lock:
        if sentinel.exists() and data_dir.is_dir():
            return data_dir
        zip_path = _hf_hub_download_data_zip()
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # Defence-in-depth: reject zip entries that escape the
                # extraction root via absolute paths or "..".
                for member in zf.namelist():
                    norm = os.path.normpath(member)
                    if norm.startswith("..") or os.path.isabs(norm):
                        raise RuntimeError(
                            f"refusing to extract suspicious zip member: {member!r}"
                        )
                zf.extractall(extract_root)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                f"THUDM/LongBench data.zip at {zip_path} is corrupted: {exc}"
            ) from exc
        if not data_dir.is_dir():
            raise RuntimeError(
                f"Expected `data/` directory inside data.zip; not found "
                f"under {extract_root}."
            )
        sentinel.write_text("ok")
    return data_dir


def _read_task_jsonl(data_dir: Path, config_name: str) -> List[Dict[str, Any]]:
    """Parse one task's JSONL file into a list of dict rows.

    ``config_name`` is the LongBench config (e.g. ``narrativeqa`` or
    ``narrativeqa_e``) and matches the upstream ``data/<config>.jsonl``
    file name.
    """
    path = data_dir / f"{config_name}.jsonl"
    if not path.is_file():
        raise RuntimeError(
            f"LongBench task file not found: {path}. Available: "
            f"{sorted(p.name for p in data_dir.glob('*.jsonl'))[:8]}..."
        )
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def _load_via_datasets(
    config_name: str,
) -> Optional[Sequence[Dict[str, Any]]]:
    """Try ``datasets.load_dataset`` with ``trust_remote_code=True``.

    Returns ``None`` on any failure that looks like the script-based
    dataset path is unsupported (so the caller can fall back to the HF
    Hub data.zip path). Raises only on completely unexpected import
    errors.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - datasets is in the image
        raise RuntimeError(
            "datasets is not installed in this environment."
        ) from exc
    try:
        ds = load_dataset(
            _LONGBENCH_REPO_ID, config_name, split="test",
            trust_remote_code=True,
        )
        return ds
    except TypeError:
        # Older `datasets` does not accept trust_remote_code; retry plain.
        try:
            from datasets import load_dataset as _ld
            return _ld(_LONGBENCH_REPO_ID, config_name, split="test")
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "datasets.load_dataset (no trust_remote_code) failed for "
                "%s; falling back to HF Hub data.zip path: %s",
                config_name, exc,
            )
            return None
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "datasets.load_dataset failed for %s: %s; falling back to HF "
            "Hub data.zip path.", config_name, exc,
        )
        return None


def _load_rows_for_config(config_name: str) -> Sequence[Dict[str, Any]]:
    """Return raw LongBench rows for ``config_name``.

    Tries ``datasets.load_dataset`` first (fast path that uses the HF
    datasets cache) and falls back to downloading and parsing
    ``data.zip`` directly when the installed ``datasets`` no longer
    supports dataset loading scripts (the failure mode at HEAD is
    ``RuntimeError: Dataset scripts are no longer supported``).
    """
    rows = _load_via_datasets(config_name)
    if rows is not None:
        return rows
    extract_root = _longbench_extract_root()
    data_dir = _ensure_extracted(extract_root)
    return _read_task_jsonl(data_dir, config_name)


def load_longbench_task(
    task: str,
    *,
    n_per_task: int,
    max_input_tokens: int,
    tokenizer: Any,
    seed: int = 42,
    use_e_split: bool = False,
) -> List[LongBenchExample]:
    """Load and prepare ``n_per_task`` rows from the upstream LongBench
    HF dataset, truncated to fit ``max_input_tokens``.

    ``use_e_split`` is reserved for the english-only LongBench-E variant
    (``THUDM/LongBench`` config name ``<task>_e``); only tasks listed in
    :data:`LONGBENCH_E_TASKS` have an ``_e`` split upstream.
    """
    if task not in DATASET2PROMPT:
        raise KeyError(f"task {task!r} not in LongBench task registry")
    if use_e_split and task not in LONGBENCH_E_TASKS:
        raise KeyError(
            f"task {task!r} has no ``_e`` split in LongBench-E; "
            f"valid: {sorted(LONGBENCH_E_TASKS)}"
        )

    config_name = task + ("_e" if use_e_split else "")
    try:
        ds = _load_rows_for_config(config_name)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load THUDM/LongBench:{config_name}: {exc}. "
            f"Check internet access / HF token / cache permissions."
        ) from exc

    # Deterministic slice (no shuffle): take the first n_per_task rows.
    n_take = min(n_per_task, len(ds))
    out: List[LongBenchExample] = []
    template = DATASET2PROMPT[task]
    max_new = DATASET2MAXLEN[task]
    for i in range(n_take):
        row = ds[i]
        # Truncate the *context* (not the question) so the question stays intact.
        ctx = row.get("context") or ""
        inp = row.get("input") or ""
        # Reserve some tokens for the input/question and prompt frame.
        reserve = 256
        ctx_truncated = _truncate_middle(
            ctx, tokenizer, max(64, max_input_tokens - reserve),
        )
        prompt = template.format(context=ctx_truncated, input=inp)
        # Final clamp to keep total prompt <= max_input_tokens.
        try:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) > max_input_tokens:
                ids = ids[:max_input_tokens]
                prompt = tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            pass
        ans = row.get("answers") or []
        if isinstance(ans, str):
            ans = [ans]
        out.append(LongBenchExample(
            task=task,
            prompt=prompt,
            answers=[str(a) for a in ans],
            all_classes=[str(c) for c in (row.get("all_classes") or [])],
            length=int(row.get("length") or 0),
            example_id=str(row.get("_id") or f"{task}-{i:04d}"),
            expected_max_new_tokens=int(max_new),
        ))
    return out
