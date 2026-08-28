import random
from reasoning_core.tasks.generated.wave0.n11_interval_abstract_interpretation.interval_ai import (
    IntervalAI,
    _interp_interval,
)

random.seed(2110083903)


def test_all_levels_generate_and_score():
    for level in (0, 1, 2, 3, 4, 5, 6):
        t = IntervalAI()
        t.config.set_level(level)
        x = t.generate_example()
        assert t.score_answer(x.answer, x) == 1.0
        assert t.score_answer(x.answer.upper(), x) == 1.0
        assert x.answer in ("exact", "sound")


def test_wrong_answer_scores_zero():
    t = IntervalAI()
    x = t.generate_example()
    wrong = "exact" if x.answer == "sound" else "sound"
    assert t.score_answer(wrong, x) == 0.0
    assert t.score_answer("foo", x) == 0.0
    assert t.score_answer("", x) == 0.0
    assert t.score_answer(None, x) == 0.0


def test_abstract_is_sound_over_approximation():
    # For any generated instance, soundness must hold: abstract covers actual.
    for _ in range(50):
        t = IntervalAI()
        t.config.set_level(2)
        m = t.generate_example().metadata
        lo, hi = m.abstract_range
        alo, ahi = m.actual_range
        assert lo <= alo and hi >= ahi, (lo, hi, alo, ahi)


def test_exact_answer_corresponds():
    t = IntervalAI()
    for _ in range(50):
        x = t.generate_example()
        m = x.metadata
        if x.answer == "exact":
            assert m.abstract_range == m.actual_range


def test_both_answers_occur():
    random.seed(42)
    t = IntervalAI()
    seen = set()
    for _ in range(40):
        t.config.set_level(0)
        seen.add(t.generate_example().answer)
    assert "exact" in seen and "sound" in seen


def test_difficulty_changes_config():
    t = IntervalAI()
    base = t.config.n_stmts
    t.config.set_level(5)
    assert t.config.n_stmts > base


def test_deduplication_key_stable():
    t = IntervalAI()
    a = t.generate_example()
    b = t.generate_example()
    assert a.metadata["_deduplication_key"] != b.metadata["_deduplication_key"]
