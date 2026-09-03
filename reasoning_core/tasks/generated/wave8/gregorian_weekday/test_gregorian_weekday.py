from datetime import date

from reasoning_core.tasks.generated.wave8.gregorian_weekday.gregorian_weekday import (
    GregorianWeekday,
    WEEKDAYS,
)


def test_gold_scores_one():
    task = GregorianWeekday()
    for _ in range(20):
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_answer_matches_datetime():
    task = GregorianWeekday()
    for _ in range(20):
        entry = task.generate_example()
        y, m, d = entry.metadata.year, entry.metadata.month, entry.metadata.day
        assert WEEKDAYS[date(y, m, d).weekday()] == entry.answer


def test_junk_and_empty_not_one():
    task = GregorianWeekday()
    entry = task.generate_example()
    assert task.score_answer("", entry) < 1.0
    assert task.score_answer("garbage", entry) < 1.0
    assert task.score_answer(123, entry) < 1.0


def test_levels_produce_different_years():
    task = GregorianWeekday()
    seen = set()
    for level in (0, 3, 6):
        task.config.set_level(level)
        for _ in range(5):
            entry = task.generate_example()
            seen.add(entry.metadata.year)
    assert len(seen) > 1
