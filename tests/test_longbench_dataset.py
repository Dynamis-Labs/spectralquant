"""Offline tests for the LongBench dataset adapter.

These tests don't reach HuggingFace — they verify the prompt-template
registry, max-len registry, the middle-truncation helper, and the
``data.zip`` fallback path that replaces the script-based dataset loader
that newer ``datasets`` releases reject. The full HF download path is
exercised end-to-end on Modal by the ``run_longbench`` integration.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


@pytest.fixture(autouse=True)
def _add_paths(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.syspath_prepend(str(SRC_DIR))


def test_dataset2prompt_covers_full_subset():
    from experiments import longbench_dataset, run_longbench
    for t in run_longbench.SUBSETS["full"]:
        assert t in longbench_dataset.DATASET2PROMPT, f"{t} missing prompt"
        assert "{context}" in longbench_dataset.DATASET2PROMPT[t]
        assert t in longbench_dataset.DATASET2MAXLEN


def test_truncate_middle_keeps_short_unchanged():
    from experiments import longbench_dataset

    class _Tok:
        def encode(self, s, add_special_tokens=False):
            return list(range(len(s.split())))

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"w{i}" for i in ids)

    text = "one two three four five"
    out = longbench_dataset._truncate_middle(text, _Tok(), 100)
    assert out == text


def test_truncate_middle_drops_middle_when_too_long():
    from experiments import longbench_dataset

    class _Tok:
        def encode(self, s, add_special_tokens=False):
            return list(range(100))  # always 100 tokens

        def decode(self, ids, skip_special_tokens=True):
            return f"head_tail_len{len(ids)}"

    out = longbench_dataset._truncate_middle("anything", _Tok(), 30)
    # Should produce a head+tail of total 30 tokens.
    assert "len30" in out


def test_load_longbench_task_rejects_unknown():
    from experiments import longbench_dataset

    class _T:
        def encode(self, s, add_special_tokens=False):
            return [0]

        def decode(self, ids, skip_special_tokens=True):
            return ""

    with pytest.raises(KeyError):
        longbench_dataset.load_longbench_task(
            "not_a_real_task",
            n_per_task=1, max_input_tokens=512, tokenizer=_T(),
        )


# ---------------------------------------------------------------------------
# data.zip fallback — guards against the failure mode where the installed
# ``datasets`` version no longer supports script-based datasets and
# rejects ``load_dataset("THUDM/LongBench", ...)`` outright.
# ---------------------------------------------------------------------------


class _CharTokenizer:
    """Minimal tokenizer-like used to drive the truncation/encoding path."""

    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 256 for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if 32 <= i < 127)


def _build_fake_data_zip(zip_path: Path, task: str) -> None:
    """Build a stand-in for THUDM/LongBench's ``data.zip`` with one task."""
    rows = [
        {
            "input": f"Q{i}",
            "context": " ".join(f"tok{j}" for j in range(40)),
            "answers": [f"ans{i}"],
            "length": 40,
            "dataset": task,
            "language": "en",
            "all_classes": [],
            "_id": f"{task}-{i:04d}",
        }
        for i in range(3)
    ]
    payload = "\n".join(json.dumps(r) for r in rows) + "\n"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"data/{task}.jsonl", payload)


def test_load_via_datasets_returns_none_on_script_rejection(monkeypatch):
    """If `datasets.load_dataset` raises (as it does on >=3.0 for scripted
    datasets like THUDM/LongBench), the helper must downgrade gracefully
    by returning None so the caller can fall back to the data.zip path.
    """
    from experiments import longbench_dataset

    def _raise(*a, **kw):
        # Mimic the real failure mode reported on Modal:
        raise RuntimeError(
            "Dataset scripts are no longer supported, but found LongBench.py"
        )

    import types as _types

    fake_datasets = _types.SimpleNamespace(load_dataset=_raise)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    assert longbench_dataset._load_via_datasets("narrativeqa") is None


def test_load_longbench_task_falls_back_to_data_zip(monkeypatch, tmp_path):
    """End-to-end fallback: when `datasets` rejects scripts, the loader
    must download `data.zip` via huggingface_hub, extract it, and yield
    real LongBenchExample rows.
    """
    from experiments import longbench_dataset

    # 1. Force the datasets fast-path to fail like the Modal incident.
    import types as _types

    def _raise_load_dataset(*a, **kw):
        raise RuntimeError(
            "Dataset scripts are no longer supported, but found LongBench.py"
        )

    fake_datasets = _types.SimpleNamespace(load_dataset=_raise_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    # 2. Stub `huggingface_hub.hf_hub_download` so no network is touched.
    fake_zip = tmp_path / "data.zip"
    _build_fake_data_zip(fake_zip, "narrativeqa")

    def _stub_hf_hub_download(repo_id, filename, repo_type=None, **kw):
        assert repo_id == "THUDM/LongBench"
        assert filename == "data.zip"
        return str(fake_zip)

    fake_hub = _types.SimpleNamespace(hf_hub_download=_stub_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    # 3. Point the extraction cache at tmp_path (don't pollute ~/.cache).
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    rows = longbench_dataset.load_longbench_task(
        "narrativeqa",
        n_per_task=2,
        max_input_tokens=512,
        tokenizer=_CharTokenizer(),
    )
    assert len(rows) == 2
    assert rows[0].task == "narrativeqa"
    assert rows[0].answers == ["ans0"]
    assert rows[0].example_id == "narrativeqa-0000"
    assert rows[0].expected_max_new_tokens == longbench_dataset.DATASET2MAXLEN[
        "narrativeqa"
    ]
    # Sentinel must exist after first extraction and a second call must
    # not re-download (verified by replacing the stub with a tripwire).
    sentinel = tmp_path / "cache" / "longbench_thudm_data" / ".extracted_ok"
    assert sentinel.exists()

    def _trip(*a, **kw):
        raise AssertionError("hf_hub_download called twice (cache miss)")

    fake_hub.hf_hub_download = _trip
    rows2 = longbench_dataset.load_longbench_task(
        "narrativeqa",
        n_per_task=1,
        max_input_tokens=512,
        tokenizer=_CharTokenizer(),
    )
    assert len(rows2) == 1


def test_data_zip_fallback_rejects_zip_slip(monkeypatch, tmp_path):
    """A malicious data.zip with a `..` traversal entry must be rejected
    before any file is written outside the extraction root.
    """
    from experiments import longbench_dataset

    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "owned")

    def _stub_download(*a, **kw):
        return str(bad_zip)

    import types as _types

    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        _types.SimpleNamespace(hf_hub_download=_stub_download),
    )

    extract_root = tmp_path / "cache" / "longbench_thudm_data"
    with pytest.raises(RuntimeError, match="suspicious zip member"):
        longbench_dataset._ensure_extracted(extract_root)
    # Nothing should have been written outside extract_root.
    assert not (tmp_path / "escape.txt").exists()


def test_load_longbench_e_split_validates_known_task():
    from experiments import longbench_dataset

    # _e split is valid for hotpotqa per upstream LongBench-E.
    assert "hotpotqa" in longbench_dataset.LONGBENCH_E_TASKS
    # _e split does not exist for narrativeqa upstream; the loader must
    # reject the request rather than emit a 404 / silent miss on Modal.
    with pytest.raises(KeyError):
        longbench_dataset.load_longbench_task(
            "narrativeqa",
            n_per_task=1, max_input_tokens=512,
            tokenizer=_CharTokenizer(),
            use_e_split=True,
        )
