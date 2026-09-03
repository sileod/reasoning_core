import random
from fractions import Fraction

import pytest

from reasoning_core.tasks.generated.wave4.s48_layered_tariffs.s48_layered_tariffs import (
    LayeredTariffs, _compute,
)


def _ent(task, level=None):
    if level is not None:
        cfg = task.config_cls()
        cfg.set_level(level)
        task.config = cfg
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0
    assert task.score_answer("", x) < 1.0
    assert task.score_answer("not a number", x) < 1.0
    return x


def test_runs_all_levels():
    t = LayeredTariffs()
    for level in range(7):
        cfg = t.config_cls()
        cfg.set_level(level)
        t.config = cfg
        x = t.generate_example()
        assert t.score_answer(x.answer, x) == 1.0


def test_prompt_mentions_answer_format():
    x = _ent(LayeredTariffs())
    assert "numerator" in x.metadata.scenario
    assert "n/d" in x.metadata.scenario


def test_answers_vary():
    t = LayeredTariffs()
    answers = set()
    for _ in range(20):
        x = t.generate_example()
        answers.add(x.answer)
    assert len(answers) > 5


def test_compute_exact():
    b = [(0, 500, Fraction(1, 10)), (500, 1000, Fraction(1, 5))]
    r = _compute(600, b, 1000, Fraction(1, 4), 50, Fraction(3, 2))
    assert r.denominator <= 2000


def test_domain_positive():
    t = LayeredTariffs()
    for _ in range(30):
        x = t.generate_example()
        num, den = x.answer.split("/")
        assert int(num) > 0
        assert int(den) > 0
