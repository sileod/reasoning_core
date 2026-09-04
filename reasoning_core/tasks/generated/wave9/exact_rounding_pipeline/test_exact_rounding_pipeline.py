import math
import random
from fractions import Fraction

from reasoning_core.tasks.generated.wave9.exact_rounding_pipeline.exact_rounding_pipeline import (
    ExactRoundingPipeline, _round_frac, _frac_text)


def test_round_frac_rules():
    half = Fraction(1, 2)
    assert _round_frac(Fraction(1, 2), 0, "floor") == 0
    assert _round_frac(Fraction(1, 2), 0, "ceiling") == 1
    assert _round_frac(Fraction(1, 2), 0, "truncate") == 0
    assert _round_frac(Fraction(1, 2), 0, "half-even") == 0
    assert _round_frac(Fraction(3, 2), 0, "half-even") == 2
    assert _round_frac(Fraction(5, 2), 0, "half-even") == 2
    assert _round_frac(Fraction(3, 2), 0, "truncate") == 1
    assert _round_frac(Fraction(-3, 2), 0, "truncate") == -1


def test_round_frac_places():
    assert _round_frac(Fraction(123, 100), 1, "floor") == Fraction(12, 10)
    assert _round_frac(Fraction(129, 100), 1, "ceiling") == Fraction(13, 10)
    assert _round_frac(Fraction(125, 100), 1, "half-even") == Fraction(12, 10)
    assert _round_frac(Fraction(135, 100), 1, "half-even") == Fraction(14, 10)


def test_gold_scores():
    random.seed(123)
    task = ExactRoundingPipeline()
    for _ in range(20):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_garbage():
    random.seed(7)
    task = ExactRoundingPipeline()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("junk", e) == 0.0
    assert task.score_answer("3.7", e) == 0.0


def test_levels():
    for level in range(7):
        task = ExactRoundingPipeline()
        task.config.set_level(level)
        for _ in range(10):
            e = task.generate_example()
            assert e.answer.lstrip("-").isdigit()


def test_answers_vary():
    random.seed(99)
    task = ExactRoundingPipeline()
    answers = {task.generate_example().answer for _ in range(200)}
    assert len(answers) > 60
