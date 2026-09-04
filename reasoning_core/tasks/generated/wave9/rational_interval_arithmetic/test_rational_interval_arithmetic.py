from fractions import Fraction

from reasoning_core.tasks.generated.wave9.rational_interval_arithmetic.rational_interval_arithmetic import (
    RationalIntervalArithmetic,
    _parse_interval,
    _fmt_interval,
    _eval_node,
    _gen,
)


def test_interval_arithmetic_basic():
    assert _parse_interval("[1, 3]") == (Fraction(1), Fraction(3))
    assert _parse_interval("[1/2, 5/2]") == (Fraction(1, 2), Fraction(5, 2))
    assert _parse_interval("empty") is None
    assert _fmt_interval(_parse_interval("[1/2, 5/2]")) == "[1/2, 5/2]"


def _eval_str_tree():
    node, val = _gen(3, 0)
    return _eval_node(node) == val


def test_leaf_eval():
    assert _eval_str_tree()


def test_add_sub_tight():
    a = (Fraction(0), Fraction(2))
    b = (Fraction(1), Fraction(3))
    # a + b -> [1, 5], a - b -> [-3, 1]
    from reasoning_core.tasks.generated.wave9.rational_interval_arithmetic.rational_interval_arithmetic import (
        _iv_add, _iv_sub, _iv_mul, _iv_div, _iv_int,
    )
    assert _iv_add(a, b) == (Fraction(1), Fraction(5))
    assert _iv_sub(a, b) == (Fraction(-3), Fraction(1))
    assert _iv_mul(a, b) == (Fraction(0), Fraction(6))
    assert _iv_div(a, b) == (Fraction(0), Fraction(2))
    assert _iv_int(a, b) == (Fraction(1), Fraction(2))
    assert _iv_int(b, a) == (Fraction(1), Fraction(2))


def test_div_by_zero_empty():
    from reasoning_core.tasks.generated.wave9.rational_interval_arithmetic.rational_interval_arithmetic import _iv_div
    a = (Fraction(1), Fraction(2))
    b = (Fraction(-1), Fraction(1))
    assert _iv_div(a, b) is None


def test_disjoint_intersection_empty():
    from reasoning_core.tasks.generated.wave9.rational_interval_arithmetic.rational_interval_arithmetic import _iv_int
    assert _iv_int((Fraction(0), Fraction(1)), (Fraction(2), Fraction(3))) is None


def test_gold_scores_one():
    task = RationalIntervalArithmetic()
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_garbage_scores_zero():
    task = RationalIntervalArithmetic()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("garbage", ex) == 0.0


def test_levels_produce_examples():
    for level in range(7):
        cfg = RationalIntervalArithmetic.config_cls()
        cfg.set_level(level)
        task = RationalIntervalArithmetic()
        ex = task.generate_example()
        assert _parse_interval(ex.answer) is not None or ex.answer == "empty"
