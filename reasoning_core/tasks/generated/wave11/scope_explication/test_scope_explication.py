import random

from reasoning_core.template import Task
from reasoning_core.tasks.generated.wave11.scope_explication.scope_explication import (
    ScopeExplication, ScopeExplicationConfig, _norm, _make_clause,
)


def _task(level=0):
    return ScopeExplication(config=ScopeExplicationConfig(), _level=level)


def test_gold_scores_one():
    for level in (0, 1, 2, 5, 6):
        t = _task(level)
        for _ in range(20):
            ex = t.generate_example(level=level)
            assert t.score_answer(ex.answer, ex) == 1.0


def test_junk_and_empty_score_zero():
    t = _task()
    ex = t.generate_example(level=0)
    assert t.score_answer("", ex) == 0.0
    assert t.score_answer("asdf qwerty random", ex) == 0.0


def test_both_readings_distinct():
    seen = set()
    for _ in range(200):
        s, h, a = _make_clause()
        seen.add(a)
        assert s and h and a
    assert len(seen) > 5


def test_difficulty_increases():
    base = ScopeExplicationConfig()
    base.set_level(0)
    l0 = base.n_clauses
    hi = ScopeExplicationConfig()
    hi.set_level(6)
    assert hi.n_clauses > l0


def test_deterministic_under_seed():
    t = _task()
    random.seed(974528455)
    ex1 = t.generate_example(level=3)
    random.seed(974528455)
    ex2 = t.generate_example(level=3)
    assert ex1.answer == ex2.answer
    assert ex1.metadata["payload"]["sentence"] == ex2.metadata["payload"]["sentence"]


def test_score_lenient_separators():
    t = _task()
    ex = t.generate_example(level=0)
    gold = ex.answer
    altered = gold.replace(" and ", "; ")
    assert t.score_answer(altered, ex) == 1.0
