from fractions import Fraction
import random
import os

from reasoning_core.tasks.generated.wave9.probability_tree_marginal.probability_tree_marginal import (
    ProbabilityTree, _parse_frac)


def _make(level):
    t = ProbabilityTree()
    t.config.set_level(level)
    return t


def test_generate_and_score():
    for level in range(7):
        obj = _make(level)
        entry = obj.generate_example()
        assert 0 < _parse_frac(entry.answer) <= 1
        assert obj.score_answer(entry.answer, entry) == 1.0


def test_random_wrong_not_one():
    obj = _make(3)
    entry = obj.generate_example()
    assert obj.score_answer("", entry) == 0.0
    assert obj.score_answer("garbage", entry) == 0.0
    assert obj.score_answer("0", entry) < 1.0


def test_distractors_not_gold():
    obj = _make(4)
    entry = obj.generate_example()
    gold = _parse_frac(entry.answer)
    for cand in obj.distractor_candidates(entry):
        c = _parse_frac(cand)
        assert c != gold


def test_difficulty_changes():
    base = _make(0)
    high = _make(6)
    assert high.config.steps >= base.config.steps
    assert base.generate_example() is not None
    assert high.generate_example() is not None


def test_domain_probability():
    random.seed(1234)
    for _ in range(10):
        obj = _make(5)
        entry = obj.generate_example()
        a = _parse_frac(entry.answer)
        assert a > 0
        assert a <= 1
