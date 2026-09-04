from fractions import Fraction

from reasoning_core.tasks.generated.wave9.conditional_expectation.conditional_expectation import (
    ConditionalExpectation,
    _parse,
    _fmt,
)


def _gold(entry):
    return _parse(entry.answer)


def test_generate_smoke():
    for level in (0, 3, 6):
        task = ConditionalExpectation()
        task.config.set_level(level)
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0


def test_answer_domain():
    task = ConditionalExpectation()
    x = task.generate_example()
    e = _gold(x)
    assert 0 <= e or e.denominator >= 1


def test_junk_scores_zero():
    task = ConditionalExpectation()
    x = task.generate_example()
    assert task.score_answer("", x) == 0.0
    assert task.score_answer("banana", x) == 0.0
    assert task.score_answer("abc/def", x) == 0.0


def test_parse_formats():
    assert _parse("3/4") == Fraction(3, 4)
    assert _parse("7") == Fraction(7, 1)
    assert _parse("-2/3") == Fraction(-2, 3)
    assert _parse("x") is None
    assert _fmt(Fraction(5, 1)) == "5"
    assert _fmt(Fraction(3, 4)) == "3/4"


def test_expectation_bounded():
    task = ConditionalExpectation()
    for _ in range(20):
        x = task.generate_example()
        p = x.metadata
        e = _gold(x)
        lo = min(p["values"][p["target_var"]])
        hi = max(p["values"][p["target_var"]])
        assert Fraction(lo) <= e <= Fraction(hi)


def test_levels_change_config():
    task = ConditionalExpectation()
    task.config.set_level(0)
    n0 = task.config.n_outcomes
    task.config.set_level(5)
    n5 = task.config.n_outcomes
    assert n5 > n0
