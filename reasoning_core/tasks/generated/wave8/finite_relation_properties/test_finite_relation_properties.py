import random

from reasoning_core.tasks.generated.wave8.finite_relation_properties.finite_relation_properties import (
    FiniteRelationProperties,
)


def test_gold_scores_one():
    random.seed(1)
    task = FiniteRelationProperties()
    for _ in range(20):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_junk_scores_zero():
    random.seed(2)
    task = FiniteRelationProperties()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("transitive", e) == (1.0 if e.answer == "transitive" else 0.0)


def test_difficulty_changes():
    task = FiniteRelationProperties()
    c = task.config_cls()
    c.set_level(0)
    n0 = c.n
    c.set_level(5)
    n5 = c.n
    assert n5 > n0
