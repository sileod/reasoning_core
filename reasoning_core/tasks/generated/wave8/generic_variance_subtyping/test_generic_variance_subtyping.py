import random

from reasoning_core.tasks.generated.wave8.generic_variance_subtyping.generic_variance_subtyping import (
    VarianceSubtyping,
    compute_witness,
    MARKERS,
)


def _fresh_task(level=0):
    random.seed(12345)
    t = VarianceSubtyping()
    t.config.set_level(level)
    return t


def test_gold_scores_one():
    random.seed(99)
    t = VarianceSubtyping()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(200):
            ex = t.generate_example()
            assert t.score_answer(ex.answer, ex) == 1.0


def test_answer_is_valid_marker_set():
    random.seed(7)
    t = VarianceSubtyping()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(100):
            ex = t.generate_example()
            if ex.answer == "none":
                continue
            for part in ex.answer.split(","):
                assert part.strip() in MARKERS


def test_junk_and_empty_fail():
    random.seed(3)
    t = VarianceSubtyping()
    for level in (0, 2, 5):
        t.config.set_level(level)
        ex = t.generate_example()
        assert t.score_answer("", ex) < 1.0
        assert t.score_answer("garbage!!", ex) < 1.0
        assert t.score_answer("42", ex) < 1.0


def test_order_invariant_scoring():
    random.seed(11)
    t = VarianceSubtyping()
    ex = t.generate_example()
    if ex.answer == "none":
        assert t.score_answer("none", ex) == 1.0
        assert t.score_answer("Fruit", ex) < 1.0
    else:
        parts = ex.answer.split(", ")
        swapped = ", ".join(reversed(parts))
        assert t.score_answer(swapped, ex) == 1.0
        wrong = parts[0] if len(parts) > 1 else "none"
        assert t.score_answer(wrong, ex) < 1.0


def test_witness_semantics():
    a = frozenset(["Fruit", "Color"])
    b = frozenset(["Fruit"])
    assert compute_witness(a, b, "+") == ["Color"]
    assert compute_witness(a, b, "-") == []
    assert compute_witness(a, b, "o") == ["Color"]
    assert compute_witness(a, b, "*") == []


def test_difficulty_changes():
    t = VarianceSubtyping()
    t.config.set_level(0)
    low = int(t.config.operands)
    t.config.set_level(5)
    high = int(t.config.operands)
    assert high >= low
    assert t.config.join_p >= 0


def test_answer_distribution_wide():
    random.seed(2024)
    t = VarianceSubtyping()
    t.config.set_level(3)
    answers = {}
    for _ in range(200):
        ex = t.generate_example()
        answers[ex.answer] = answers.get(ex.answer, 0) + 1
    top = max(answers.values()) / sum(answers.values())
    assert len(answers) >= 20
    assert top <= 0.30
