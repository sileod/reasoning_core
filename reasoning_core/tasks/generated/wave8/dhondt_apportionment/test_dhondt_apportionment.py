import random

from reasoning_core.tasks.generated.wave8.dhondt_apportionment.dhondt_apportionment import (
    DhondtApportionment,
    _dhondt,
    _parse_answer,
)


def test_gold_scores_one():
    random.seed(123)
    t = DhondtApportionment()
    for _ in range(20):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_dhondt_sum():
    random.seed(7)
    t = DhondtApportionment()
    for _ in range(20):
        e = t.generate_example()
        assert sum(_parse_answer(e.answer)) == e.metadata.seats


def test_levels():
    t = DhondtApportionment()
    for level in range(7):
        random.seed(5 + level)
        t.config.set_level(level)
        for _ in range(5):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0


def test_junk_scores_zero():
    t = DhondtApportionment()
    random.seed(1)
    e = t.generate_example()
    assert t.score_answer("", e) == 0.0
    assert t.score_answer("junk", e) == 0.0
    assert t.score_answer("abc", e) == 0.0


def test_dhondt_known():
    assert _dhondt([100, 80, 30, 20], 7) == [3, 3, 1, 0]


def test_all_levels_vary():
    answers = set()
    t = DhondtApportionment()
    for level in range(6):
        random.seed(100 + level)
        t.config.set_level(level)
        for _ in range(10):
            e = t.generate_example()
            answers.add(e.answer)
    assert len(answers) > 10
