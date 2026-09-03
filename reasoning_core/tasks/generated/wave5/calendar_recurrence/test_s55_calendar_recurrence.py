import importlib.util
from datetime import date, timedelta
from pathlib import Path

import pytest

MOD_FILE = Path(__file__).with_name("s55_calendar_recurrence.py")
SPEC = importlib.util.spec_from_file_location("s55_calendar_recurrence", MOD_FILE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _task(level):
    t = MOD.CalendarRecurrence()
    cfg = t.config_cls()
    cfg.set_level(level)
    t.config = cfg
    return t


def test_families_cover_all_rules():
    seen = set()
    for _ in range(120):
        ex = _task(3).generate_example()
        seen.add(ex.metadata.family)
        assert MOD._score(ex.answer, ex) == 1.0
    assert seen == {"weekday", "interval", "monthend"}


def test_gold_scores_one_and_format():
    for level in (0, 2, 5):
        ex = _task(level).generate_example()
        assert MOD._score(ex.answer, ex) == 1.0
        parts = ex.answer.split("-")
        assert len(parts) == 3
        assert all(len(p) == len(("0000", "00", "00")[i]) for i, p in enumerate(parts))
        date(*map(int, parts))


def test_junk_scores_zero():
    ex = _task(2).generate_example()
    assert MOD._score("", ex) == 0.0
    assert MOD._score("not a date", ex) == 0.0
    assert MOD._score(ex.answer + "x", ex) == 0.0


def test_nth_answer_is_valid_occurrence():
    for _ in range(80):
        ex = _task(4).generate_example()
        assert MOD._score(ex.answer, ex) == 1.0
        meta = ex.metadata
        if meta.family == "interval":
            start = date(*map(int, meta.start_iso.split("-")))
            expected = [MOD._business(start + timedelta(days=i * meta.D)).isoformat()
                        for i in range(meta.n)]
            assert ex.answer == expected[-1]
        else:
            expected = [MOD._month_date(yy, mm, tuple(meta.spec)).isoformat()
                        for i in range(meta.n)
                        for yy, mm in [MOD._add_months(meta.year, meta.month, i * meta.K)]]
            assert ex.answer == expected[-1]


def test_answers_vary_across_examples():
    ans = {_task(3).generate_example().answer for _ in range(40)}
    assert len(ans) > 15


def test_rendered_prompt_carries_all_clauses():
    ex = _task(3).generate_example()
    p = MOD.CalendarRecurrence().render_prompt(ex.metadata)
    assert "YYYY-MM-DD" in p
    assert "occurrence" in p


def test_interval_skip_moves_to_monday():
    from datetime import date as _d
    shifted = MOD._business(_d(2021, 6, 12))  # Saturday
    assert shifted == _d(2021, 6, 14)
