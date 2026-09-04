"""Tests for the poly_euclid_algorithm trial task."""

import random

from reasoning_core.tasks.generated.wave9.polynomial_euclidean_algorithm.poly_euclid_algorithm import (
    PolyEuclid,
    _monic_str,
    _gcd_poly_mod,
    _gcd_poly_int,
)


def test_generation_scores_all_levels():
    random.seed(4112812501)
    t = PolyEuclid()
    for lvl in range(7):
        t.config.set_level(lvl)
        for _ in range(8):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0


def test_gold_divides_and_is_monic():
    random.seed(7)
    t = PolyEuclid()
    for lvl in (0, 2, 5):
        t.config.set_level(lvl)
        for _ in range(20):
            e = t.generate_example()
            assert e.metadata.answer == e.answer


def test_garbage_scores_zero():
    random.seed(11)
    t = PolyEuclid()
    e = t.generate_example()
    assert t.score_answer("", e) == 0.0
    assert t.score_answer("junk", e) == 0.0
    assert t.score_answer("2*x", e) == 0.0


def test_answer_not_on_surface():
    random.seed(13)
    t = PolyEuclid()
    for _ in range(30):
        e = t.generate_example()
        assert e.answer not in e.metadata.f
        assert e.answer not in e.metadata.g


def test_monic_helper():
    assert _monic_str("2*x + 4") == "x + 2"
    assert _monic_str("3*x**2 - 6") == "x**2 - 2"


def test_difficulty_changes_config():
    t = PolyEuclid()
    c0 = t.config.to_dict()
    t.config.set_level(3)
    assert t.config.to_dict() != c0


def test_summary_one_line():
    s = PolyEuclid.summary
    assert isinstance(s, str) and "\n" not in s and "\r" not in s
