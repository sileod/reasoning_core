import ctypes

import pytest

from reasoning_core import template
from reasoning_core.template import Entry, Task, TimeoutException, timeout_retry


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


class RejectingTask(ConstantLabelTask):
    def __init__(self, candidate):
        super().__init__()
        self.candidate = candidate

    def generate_entry(self):
        self.index += 1
        return self.candidate

    def render_prompt(self, metadata):
        return "oversized"


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


def test_timeout_is_an_exception_not_a_base_exception_only():
    assert issubclass(TimeoutException, TimeoutError)


def test_timeout_retry_does_not_catch_keyboard_interrupt():
    @timeout_retry(seconds=1, attempts=2)
    def operation():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        operation()


def test_entry_iteration_obeys_mapping_contract():
    entry = Entry({"x": 1}, "yes")

    assert list(entry) == list(entry.to_dict())
    assert dict(entry) == entry.to_dict()


@pytest.mark.parametrize("candidate", [None, Entry({}, "answer")])
def test_generate_example_raises_after_rejection_exhaustion(candidate):
    task = RejectingTask(candidate)
    task.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: list(str(text))})()

    with pytest.raises(RuntimeError, match="failed to generate an admissible example after 1000 attempts"):
        task.generate_example(max_tokens=1)

    assert task.index == 1_000
