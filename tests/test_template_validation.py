import ctypes

import pytest

from reasoning_core import template
from reasoning_core.template import Entry, Task, timeout_retry


class ConstantLabelTask(Task):
    summary = "Generate distinct indexed prompts with one constant Boolean label for validation tests."

    def __init__(self):
        super().__init__()
        self.index = 0

    def generate_entry(self):
        self.index += 1
        return Entry({"index": self.index}, "True")

    def render_prompt(self, metadata):
        return f"Example {metadata['index']}"

    def score_answer(self, answer, entry):
        return float(str(answer) == entry.answer)


class NonJsonMetadataTask(ConstantLabelTask):
    def generate_entry(self):
        self.index += 1
        return Entry({"index": self.index, "bad": object()}, "True")


class MissingSummaryTask(Task):
    def generate_entry(self):
        raise AssertionError("summary validation must run before generation")


def test_validation_does_not_treat_repeated_labels_as_other_answers():
    task = ConstantLabelTask()
    rows = [Entry({"index": i}, "True") for i in range(4)]
    for row in rows:
        row.prompt = task.render_prompt(row.metadata)

    task._check_validation_examples(rows[0], rows[1:], n_samples=3)


def test_validation_rejects_non_json_metadata():
    with pytest.raises(TypeError, match="JSON serializable"):
        NonJsonMetadataTask().validate(n_samples=1)


def test_validation_requires_a_packed_one_line_summary():
    with pytest.raises(AssertionError, match="one-line coverage spec"):
        MissingSummaryTask().validate(n_samples=1)


def test_timeout_retry_recovers_from_ctypes_wrapped_signal(monkeypatch):
    calls = 0
    monkeypatch.setattr(template.time, "sleep", lambda _: None)

    @timeout_retry(seconds=1, attempts=2)
    def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ctypes.ArgumentError("argument 1: TimeoutException")
        return "ok"

    assert operation() == "ok"
    assert calls == 2
