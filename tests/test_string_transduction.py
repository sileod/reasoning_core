import random

from reasoning_core import list_tasks
from reasoning_core.template import Entry, edict
from reasoning_core.tasks.string_transduction import StringTransduction, StringTransductionConfig


def test_string_transduction_registers_and_scores():
    assert "string_transduction" in list_tasks()
    assert "diff_prediction" not in list_tasks()
    task = StringTransduction()
    for _ in range(20):
        problem = task.generate_example(max_tokens=0)
        assert problem.answer
        assert task.score_answer(problem.answer, problem) == 1


def test_string_transduction_scores_near_misses_by_edit_similarity():
    task = StringTransduction()
    entry = Entry(edict(), "abcd")

    assert task.score_answer("abcd", entry) == 1.0
    assert task.score_answer("abxd", entry) == 0.75
    assert task.score_answer("", entry) == 0.0


def test_string_transduction_edit_mode():
    task = StringTransduction(StringTransductionConfig(edit_rate=1.0))
    problem = task.generate_example(max_tokens=0)
    assert problem.metadata.mode == "edit"
    assert problem.metadata.edits
    assert "Edits:" in problem.prompt


def test_string_transduction_excludes_spaces_when_requested(monkeypatch):
    task = StringTransduction(StringTransductionConfig(edit_rate=0.0, exclude_spaces=1.0))
    draws = iter([0.5, 0.0, 0.0])
    monkeypatch.setattr(random, "random", lambda: next(draws))
    monkeypatch.setattr(random, "randint", lambda _a, _b: 4)
    monkeypatch.setattr(random, "sample", lambda population, k: list(population[:k]))
    monkeypatch.setattr(task, "_program", lambda _alphabet: [("sort ascending", lambda s: "".join(sorted(s)))])

    entry = task.generate_entry()

    assert " " not in entry.answer
    assert f"String: {entry.metadata.source}" in task.render_prompt(entry.metadata)
    assert "excluding spaces" in task.render_prompt(entry.metadata)
    assert '"' not in entry.answer
    assert task.score_answer(entry.answer, entry) == 1


def test_string_transduction_omits_redundant_space_instruction(monkeypatch):
    task = StringTransduction(StringTransductionConfig(edit_rate=0.0, exclude_spaces=1.0))
    draws = iter([0.5, 0.0])
    monkeypatch.setattr(random, "random", lambda: next(draws))
    monkeypatch.setattr(random, "randint", lambda _a, _b: 4)
    monkeypatch.setattr(random, "sample", lambda population, k: list(population[:k]))
    monkeypatch.setattr(task, "_program", lambda _alphabet: [("keep only a and b", lambda s: "".join(c for c in s if c in "ab"))])

    entry = task.generate_entry()

    assert " " not in entry.answer
    assert not entry.metadata.exclude_spaces
    assert "excluding spaces" not in task.render_prompt(entry.metadata)


def test_string_transduction_respects_max_noop_rate(monkeypatch):
    task = StringTransduction(
        StringTransductionConfig(length=4, edit_rate=0.0, max_noop_rate=0.0)
    )
    programs = iter([
        [("identity", lambda s: s)],
        [("caesar shift by 1", lambda s: "".join(chr(ord(c) + 1) for c in s))],
    ])
    letters = iter("abababab")
    monkeypatch.setattr(random, "random", lambda: 0.5)
    monkeypatch.setattr(random, "choice", lambda _alphabet: next(letters))
    monkeypatch.setattr(task, "_program", lambda _alphabet: next(programs))

    entry = task.generate_entry()

    assert entry.metadata.ops == ["caesar shift by 1"]
    assert entry.metadata.noop_rate == 0.0
    assert entry.answer == "bcbc"
