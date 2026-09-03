import random

from reasoning_core.tasks.generated.wave3.s33_signed_digit_arithmetic.signed_digit_arithmetic import (
    SignedDigitArithmetic,
    _from_balanced_ternary,
    _to_balanced_ternary,
)


def _gold_denotes_stated_value(entry):
    if entry.metadata.operation == "convert":
        return _from_balanced_ternary(entry.answer) == entry.metadata.given
    a = _from_balanced_ternary(entry.metadata.a)
    b = _from_balanced_ternary(entry.metadata.b)
    return _from_balanced_ternary(entry.answer) == a + b


def test_gold_is_denotable():
    task = SignedDigitArithmetic()
    for level in (0, 3, 6):
        task.config.set_level(level)
        for _ in range(100):
            entry = task.generate_example()
            assert _gold_denotes_stated_value(entry), (entry.metadata, entry.answer)


def test_balanced_ternary_roundtrip():
    for _ in range(200):
        value = random.randint(-1000, 1000)
        assert _from_balanced_ternary(_to_balanced_ternary(value)) == value


def test_zero():
    assert _to_balanced_ternary(0) == "0"
    assert _from_balanced_ternary("0") == 0


def test_known_values():
    assert _to_balanced_ternary(1) == "1"
    assert _to_balanced_ternary(2) == "1T"
    assert _to_balanced_ternary(3) == "10"
    assert _to_balanced_ternary(4) == "11"
    assert _to_balanced_ternary(5) == "1TT"
    assert _to_balanced_ternary(-1) == "T"
    assert _to_balanced_ternary(-2) == "T1"
    assert _to_balanced_ternary(-3) == "T0"
    assert _to_balanced_ternary(-4) == "TT"


def test_generate_and_score():
    task = SignedDigitArithmetic()
    for _ in range(50):
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0
        assert task.score_answer("", entry) == 0.0
        assert task.score_answer("garbage", entry) == 0.0
        assert task.score_answer(None, entry) == 0.0


def test_difficulty_changes():
    config = SignedDigitArithmetic.config_cls()
    base = config.n_digits
    config.set_level(3)
    assert config.n_digits > base
