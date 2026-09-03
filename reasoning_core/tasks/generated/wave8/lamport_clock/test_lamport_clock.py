import random

import pytest

from reasoning_core.tasks.generated.wave8.lamport_clock.lamport_clock import (
    LamportClock,
    LamportClockConfig,
    _compute_timestamps,
)


def _gold_from_metadata(md):
    ts = md.timestamps
    assert md.n_processes == len(ts)
    idx = md.all_events.index(md.query)
    proc = idx // md.n_columns
    col = idx % md.n_columns
    return ts[proc][col]


def test_gold_scores_one_default():
    random.seed(1)
    task = LamportClock()
    for _ in range(20):
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0
        assert _gold_from_metadata(x.metadata) == int(x.answer)


def test_gold_scores_one_all_levels():
    random.seed(7)
    task = LamportClock()
    for level in range(7):
        x = task.generate_example(level=level)
        assert task.score_answer(x.answer, x) == 1.0
        assert _gold_from_metadata(x.metadata) == int(x.answer)


def test_junk_and_wrong_answers_fail():
    random.seed(3)
    task = LamportClock()
    x = task.generate_example()
    assert task.score_answer("", x) < 1.0
    assert task.score_answer("abc", x) < 1.0
    assert task.score_answer(str(int(x.answer) + 1), x) < 1.0
    assert task.score_answer("   -7  ", x) < 1.0


def test_answer_is_boosted_not_local_index():
    random.seed(11)
    task = LamportClock()
    seen = set()
    for _ in range(120):
        x = task.generate_example()
        md = x.metadata
        idx = md.all_events.index(md.query)
        col = idx % md.n_columns
        assert int(x.answer) != col + 1
        seen.add(x.answer)
    assert len(seen) > 1


def test_answer_domain_is_positive():
    random.seed(23)
    task = LamportClock()
    for _ in range(60):
        x = task.generate_example()
        assert int(x.answer) >= 1


def test_difficulty_changes_config():
    c0 = LamportClockConfig()
    c0.set_level(0)
    c6 = LamportClockConfig()
    c6.set_level(6)
    assert c0.level == 0 and c6.level == 6
    assert c0.columns >= 2 and c0.num_messages >= 1
    assert c6.columns > c0.columns
    assert c6.num_messages >= c0.num_messages


def test_all_levels_generate_quickly():
    random.seed(99)
    task = LamportClock()
    for level in range(7):
        for _ in range(3):
            x = task.generate_example(level=level)
            assert task.score_answer(x.answer, x) == 1.0
            md = x.metadata
            idx = md.all_events.index(md.query)
            col = idx % md.n_columns
            assert int(x.answer) >= 1
            assert int(x.answer) != col + 1



def test_compute_timestamps_reference():
    n_proc, cols = 3, 4
    messages = [(0, 2, 1, 0)]
    ts = _compute_timestamps(n_proc, cols, messages)
    assert ts[1][0] == 4
    assert ts[1][1] == 5
    assert ts[0][2] == 3


def test_compute_timestamps_acyclic_check():
    n_proc, cols = 2, 2
    messages = [(0, 1, 1, 0), (1, 1, 0, 0)]
    assert _compute_timestamps(n_proc, cols, messages) is None
