from datetime import datetime

from reasoning_core.tasks.generated.wave8.cron_next_fire.cron_next_fire import (
    CronNextFire,
    _next_fire,
)


def test_gold_scores_one():
    task = CronNextFire()
    for _ in range(20):
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0
        assert datetime.strptime(x.answer, "%Y-%m-%d %H:%M") > datetime.strptime(
            x.metadata.payload["reference timestamp"], "%Y-%m-%d %H:%M"
        )


def test_junk_wrong():
    task = CronNextFire()
    x = task.generate_example()
    assert task.score_answer("", x) < 1.0
    assert task.score_answer("garbage", x) < 1.0
    assert task.score_answer("0000-00-00 00:00", x) < 1.0


def test_next_fire_basic():
    fields = (30, 14, 15, 3)  # minute 30, hour 14, day 15, month 3
    ref = datetime(2030, 3, 15, 14, 29)
    nxt = _next_fire(fields, ref)
    assert nxt == datetime(2030, 3, 15, 14, 30)
    ref2 = datetime(2030, 3, 15, 14, 30)
    assert _next_fire(fields, ref2) == datetime(2031, 3, 15, 14, 30)


def test_wildcard_all_any():
    fields = (None, None, None, None)
    ref = datetime(2025, 6, 1, 12, 0)
    assert _next_fire(fields, ref) == datetime(2025, 6, 1, 12, 1)


def test_month_day_validity_never_none():
    task = CronNextFire()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(15):
            x = task.generate_example()
            assert x is not None
