import random

import pytest

from reasoning_core.tasks.generated.wave10.instruction_priority.instruction_priority import (
    Priority, PriorityConfig, _resolve, _RANK)


def test_generate_roundtrip():
    random.seed(123)
    task = Priority()
    for level in range(6):
        task.config.set_level(level)
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0


def test_score_junk():
    random.seed(7)
    task = Priority()
    x = task.generate_example()
    assert task.score_answer("", x) < 1.0
    assert task.score_answer("abc", x) < 1.0


def test_resolve_highest_authority():
    assert _resolve([("employee", 1), ("director", 2)]) == 2
    assert _resolve([("regulation", 3), ("ceo", 4)]) == 3


def test_resolve_recency_tie():
    assert _resolve([("manager", 1), ("manager", 2)]) == 2


def test_difficulty_changes():
    task = Priority()
    task.config.set_level(0)
    n0 = task.config.n_instructions
    task.config.set_level(5)
    n5 = task.config.n_instructions
    assert n5 > n0


def test_answer_is_stated_value():
    random.seed(42)
    task = Priority()
    for level in range(6):
        task.config.set_level(level)
        x = task.generate_example()
        assert x.answer in x.metadata.final.values()


def test_meta_present():
    from reasoning_core.tasks.generated.wave10.instruction_priority import (
        instruction_priority as m)
    assert m.TASK_META["hypothesis"] == "ASTRA0-01"
