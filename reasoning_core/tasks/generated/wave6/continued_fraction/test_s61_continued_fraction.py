import random
from fractions import Fraction

from reasoning_core.tasks.generated.wave6.s61_continued_fraction.s61_continued_fraction import (
    ContinuedFraction, ContinuedFractionConfig, _list_to_fraction)


def test_gold_scores_one():
    random.seed(42)
    task = ContinuedFraction()
    for _ in range(20):
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_reconstructs_input():
    random.seed(7)
    task = ContinuedFraction()
    for _ in range(50):
        entry = task.generate_example()
        terms = [int(p) for p in entry.answer.split(",")]
        assert _list_to_fraction(terms) == Fraction(entry.metadata["frac"])


def test_no_trailing_one():
    random.seed(11)
    task = ContinuedFraction()
    for _ in range(50):
        entry = task.generate_example()
        terms = [int(p) for p in entry.answer.split(",")]
        if len(terms) >= 2:
            assert terms[-1] != 1


def test_junk_scores_zero():
    task = ContinuedFraction()
    entry = task.generate_example()
    for bad in ["", "abc", "1.5", "foo, bar", "3, 7, 15, 1"]:
        assert task.score_answer(bad, entry) < 1.0


def test_trailing_one_rejected():
    random.seed(3)
    task = ContinuedFraction()
    entry = task.generate_example()
    terms = [int(p) for p in entry.answer.split(",")]
    if len(terms) >= 2:
        bad = ", ".join(str(t) for t in terms)
        assert task.score_answer(bad + ", 1", entry) < 1.0


def test_levels_generate():
    for level in range(7):
        config = ContinuedFractionConfig()
        config.set_level(level)
        task = ContinuedFraction(config=config)
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0
