import random

from reasoning_core.tasks.generated.wave11.incremental_interpretation.incremental_interpretation import (
    IncrementalInterpretation,
    _feasible,
)


def test_gold_scores_1():
    task = IncrementalInterpretation()
    for _ in range(200):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_difficulty_changes():
    c = IncrementalInterpretation.config_cls()
    c.set_level(0)
    v0 = c.max_val
    c.set_level(6)
    assert c.max_val > v0
    assert c.n_fragments > 3


def test_answer_consistent_with_prefix():
    task = IncrementalInterpretation()
    for _ in range(100):
        e = task.generate_example()
        max_val = e.metadata.max_val
        gold = [int(v) for v in e.answer.split()]
        if gold == [0]:
            assert _feasible(max_val, e.metadata.constraints) == set()
        else:
            assert sorted(gold) == sorted(_feasible(max_val, e.metadata.constraints))


def test_garbage_scores_0():
    task = IncrementalInterpretation()
    e = task.generate_example()
    assert task.score_answer("banana", e) == 0.0
    assert task.score_answer("", e) == 0.0


def test_wrong_answer_scores_0():
    task = IncrementalInterpretation()
    for _ in range(100):
        e = task.generate_example()
        gold = [int(v) for v in e.answer.split()]
        wrong = "0" if gold != [0] else "1 2"
        assert task.score_answer(wrong, e) == 0.0
