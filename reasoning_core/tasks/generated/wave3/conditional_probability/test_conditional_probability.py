from fractions import Fraction

from reasoning_core.tasks.generated.wave3.s35_conditional_probability.conditional_probability import (
    ConditionalProbability,
    _parse_frac,
)


def _make_task(level=0):
    import random
    random.seed(12345)
    t = ConditionalProbability()
    t.config.set_level(level)
    return t


def test_gold_scores_one():
    import random
    random.seed(7)
    t = ConditionalProbability()
    for _ in range(50):
        ex = t.generate_example()
        assert t.score_answer(ex.answer, ex) == 1.0


def test_each_level_generates():
    import random
    random.seed(99)
    t = ConditionalProbability()
    for level in range(7):
        t.config.set_level(level)
        ex = t.generate_example()
        assert t.score_answer(ex.answer, ex) == 1.0


def test_bad_answers_score_zero():
    import random
    random.seed(3)
    t = ConditionalProbability()
    ex = t.generate_example()
    assert t.score_answer("", ex) == 0.0
    assert t.score_answer("garbage", ex) == 0.0
    assert t.score_answer("0/0", ex) == 0.0


def test_parse_frac():
    assert _parse_frac("3/4") == Fraction(3, 4)
    assert _parse_frac(" 1 / 2 ") == Fraction(1, 2)
    assert _parse_frac("1.5") is None
    assert _parse_frac("x/y") is None


def test_answer_lowest_terms():
    import random
    random.seed(11)
    t = ConditionalProbability()
    for _ in range(50):
        ex = t.generate_example()
        fr = Fraction(ex.answer)
        assert fr.numerator % fr.denominator != 0 or fr.denominator == 1
        assert 0 <= fr <= 1
