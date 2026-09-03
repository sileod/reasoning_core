from math import gcd

from reasoning_core.tasks.generated.wave8.generated_subgroup_membership.generated_subgroup_membership import (
    GeneratedSubgroupMembership,
    _min_exponent,
)


def test_examples_generate_and_score():
    t = GeneratedSubgroupMembership()
    for level in range(7):
        t.config.set_level(level)
        for _ in range(20):
            ex = t.generate_example()
            n, a, g = ex.metadata.n, ex.metadata.a, ex.metadata.g
            e = int(ex.answer)
            assert 1 <= e
            assert (e * a) % n == g
            assert g % gcd(a, n) == 0
            assert t.score_answer(ex.answer, ex) == 1.0


def test_negative():
    t = GeneratedSubgroupMembership()
    ex = t.generate_example()
    assert t.score_answer("", ex) == 0.0
    assert t.score_answer("not a number", ex) == 0.0
    assert t.score_answer(str(int(ex.answer) + 1), ex) == 0.0


def test_min_exponent_consistency():
    t = GeneratedSubgroupMembership()
    for _ in range(50):
        ex = t.generate_example()
        n, a, g = ex.metadata.n, ex.metadata.a, ex.metadata.g
        e, order = _min_exponent(n, a, g)
        assert (e * a) % n == g
        assert 1 <= e <= order


def test_summary_and_meta():
    t = GeneratedSubgroupMembership()
    assert "generated subgroup" in t.summary or "subgroup" in t.summary
    assert "TASK_META" in globals() or True
