import pytest
import tiktoken

import reasoning_core
import reasoning_core.template as template
from reasoning_core import score_answer
from reasoning_core.template import Config, Problem, Task


class _DummyTokenizer:
    def encode(self, text):
        return list(text)


@pytest.fixture(autouse=True)
def _patch_tokenizer(monkeypatch):
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: _DummyTokenizer())


class DefaultVersionTask(Task):
    task_name = "default_version_task"

    def __init__(self):
        super().__init__(config=Config())

    def generate(self):
        return Problem(metadata={"instance": "Return Alpha Beta."}, answer="Alpha Beta")

    def prompt(self, metadata):
        return metadata["instance"]

    def score_answer(self, answer, entry):
        return float(str(answer).strip() == str(entry.answer).strip())


class SparseVersionTask(Task):
    task_name = "sparse_version_task"
    score_answer_version = 1
    score_answer_history = {}

    def __init__(self):
        super().__init__(config=Config())

    def generate(self):
        return Problem(metadata={"instance": "Normalize spacing."}, answer="Alpha Beta")

    def prompt(self, metadata):
        return metadata["instance"]

    def score_answer(self, answer, entry):
        normalize = lambda text: "".join(str(text).split()).lower()
        return float(normalize(answer) == normalize(entry.answer))


def _register_test_task(task_cls):
    reasoning_core.DATASETS[task_cls.task_name] = task_cls
    reasoning_core.scorers[task_cls.task_name] = lambda answer, entry: task_cls.score_answer_for_entry(answer, entry, object())


def test_generated_examples_record_default_version_and_commit(monkeypatch):
    monkeypatch.setattr(template, "current_git_commit", lambda: "deadbeef")
    task = DefaultVersionTask()
    example = task.generate_example()

    assert example.metadata["_score_answer"] == {
        "version": 0,
        "hash": DefaultVersionTask.score_answer_hash(DefaultVersionTask.score_answer),
        "commit": "deadbeef",
    }


def test_global_score_answer_loads_legacy_scorer_from_file(tmp_path):
    legacy_path = tmp_path / "legacy_sparse_version_task.py"
    legacy_path.write_text(
        "def score_answer(self, answer, entry):\n"
        "    return float(str(answer).strip().lower() == str(entry.answer).strip().lower())\n"
    )
    SparseVersionTask.score_answer_history = {0: {"file": str(legacy_path)}}
    _register_test_task(SparseVersionTask)

    entry = Problem(
        metadata={
            "_task": "sparse_version_task",
            "_score_answer": {"version": 0},
        },
        answer="Alpha Beta",
    )

    assert score_answer("alphabeta", entry) == 0.0
    assert score_answer("alpha beta", entry) == 1.0


def test_unknown_legacy_version_requires_sparse_history_registration():
    SparseVersionTask.score_answer_history = {}
    _register_test_task(SparseVersionTask)

    entry = Problem(
        metadata={
            "_task": "sparse_version_task",
            "_score_answer": {"version": 0},
        },
        answer="Alpha Beta",
    )

    with pytest.raises(KeyError, match="score_answer_history"):
        score_answer("alpha beta", entry)
