import random
from fractions import Fraction

from reasoning_core.tasks.generated.wave5.s54_polynomial_gcd.s54_polynomial_gcd import (
    PolynomialGcd, PolynomialGcdConfig, _gcd_poly, _monic
)


def test_generate_scores_self():
    random.seed(4028055723)
    t = PolynomialGcd()
    for _ in range(20):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_coprime_answer_one():
    random.seed(7)
    t = PolynomialGcd()
    for _ in range(20):
        e = t.generate_entry()
        if e.answer == '1':
            a = [Fraction(c) for c in e.metadata.a_coeffs]
            b = [Fraction(c) for c in e.metadata.b_coeffs]
            assert len(_monic(_gcd_poly(a, b))[0]) == 1
        else:
            assert len(_monic(_gcd_poly(
                [Fraction(c) for c in e.metadata.a_coeffs],
                [Fraction(c) for c in e.metadata.b_coeffs]))[0]) > 1


def _div_rem(p, d):
    p = [Fraction(c) for c in p]
    d = [Fraction(c) for c in d]
    rem = p[:]
    while len(rem) >= len(d) and rem != [0] * len(rem):
        lead = rem[-1] / d[-1]
        shift = len(rem) - len(d)
        for i in range(len(d)):
            rem[i + shift] -= lead * d[i]
        while rem and rem[-1] == 0:
            rem.pop()
    return rem or [Fraction(0)]


def test_gcd_is_monic_divides_both():
    random.seed(11)
    t = PolynomialGcd()
    for _ in range(30):
        e = t.generate_entry()
        g = [Fraction(c) for c in e.metadata.gcd]
        a = [Fraction(c) for c in e.metadata.a_coeffs]
        b = [Fraction(c) for c in e.metadata.b_coeffs]
        assert g[-1] == 1
        for p in (a, b):
            assert _div_rem(p, g) == [Fraction(0)]


def test_wrong_answers_score_zero():
    random.seed(3)
    t = PolynomialGcd()
    e = t.generate_example()
    assert t.score_answer('x - 1', e) == 0.0
    assert t.score_answer('', e) == 0.0


def test_difficulty_changes_config():
    cfg = PolynomialGcdConfig()
    base = (cfg.degree, cfg.coeff_range)
    cfg2 = PolynomialGcdConfig()
    cfg2.apply_difficulty(5)
    assert base != (cfg2.degree, cfg2.coeff_range)
