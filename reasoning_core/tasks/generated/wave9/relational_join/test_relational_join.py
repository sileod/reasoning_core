import random
import pytest

from reasoning_core.tasks.generated.wave9.relational_join_execution.relational_join import (
    RelationalJoin, _score_inner, _score_left, _score_semi,
    _score_anti,
)


def test_roundtrip_scores_gold():
    random.seed(1)
    task = RelationalJoin()
    for level in (0, 2, 5):
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0
            assert int(ex.answer) >= 0


def test_junk_not_scored():
    random.seed(2)
    task = RelationalJoin()
    ex = task.generate_example()
    assert task.score_answer("", ex) < 1.0
    assert task.score_answer("garbage", ex) < 1.0
    assert task.score_answer("12.5", ex) < 1.0


def test_manual_join_math():
    la = [0, 1, 0]
    ra = [2, 3, 4]
    lb = [0, 1]
    rb = [5, 6]
    k = 3
    assert _score_inner(la, ra, lb, rb, k) == (3 * 2 + 5) + (3 * 3 + 6) + (3 * 4 + 5)
    assert _score_left(la, ra, lb, rb, k) == (3 * 2 + 5) + (3 * 3 + 6) + (3 * 4 + 5)
    la2 = [0, 2, 1]
    ra2 = [2, 3, 4]
    assert _score_left(la2, ra, lb, rb, k) == (3 * 2 + 5) + (3 * 4) + (3 * 3 + 6)
    assert _score_semi(la2, ra, lb, rb, k) == 3 * 2 + 3 * 4
    assert _score_anti(la2, ra, lb, rb, k) == 3 * 3
