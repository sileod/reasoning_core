import random

from reasoning_core.tasks.generated.wave9.abstract_interval_analysis.abstract_interval_analysis import (
    AbstractIntervalAnalysis,
)


def test_roundtrip_all_levels():
    t = AbstractIntervalAnalysis()
    for L in (0, 2, 5):
        t.config.set_level(L)
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_wrong_answers():
    t = AbstractIntervalAnalysis()
    for L in (0, 2, 5):
        t.config.set_level(L)
        e = t.generate_example()
        assert t.score_answer("garbage", e) == 0.0
        assert t.score_answer("", e) == 0.0
        assert t.score_answer("()", e) == 0.0
        assert t.score_answer(None, e) == 0.0
        assert t.score_answer("(5)", e) == 0.0


def test_answer_varies():
    t = AbstractIntervalAnalysis()
    answers = set()
    for _ in range(60):
        t.config.set_level(3)
        answers.add(t.generate_example().answer)
    assert len(answers) > 8


def test_answer_distribution_spread():
    t = AbstractIntervalAnalysis()
    answers = []
    for L in (0, 2, 3, 5):
        t.config.set_level(L)
        for _ in range(20):
            answers.append(t.generate_example().answer)
    top = max(set(answers), key=answers.count)
    assert answers.count(top) / len(answers) < 0.35


def test_levels_all_generate():
    t = AbstractIntervalAnalysis()
    for L in (0, 1, 2, 3, 4, 5, 6):
        t.config.set_level(L)
        for _ in range(5):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0


def test_intervals_well_formed():
    from reasoning_core.tasks.generated.wave9.abstract_interval_analysis.abstract_interval_analysis import (
        _parse_interval,
    )

    assert _parse_interval("(3, +inf)") == (3.0, float("inf"))
    assert _parse_interval("(-inf, +inf)") == (float("-inf"), float("inf"))
    assert _parse_interval("(5, 9)") == (5.0, 9.0)
    assert _parse_interval("(9, 5)") is None
    assert _parse_interval("junk") is None
