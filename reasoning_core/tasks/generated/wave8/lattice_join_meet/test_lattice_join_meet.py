import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))

import random

random.seed(991003721)

from reasoning_core.tasks.generated.wave8.lattice_join_meet.lattice_join_meet import (
    LatticeJoinMeet,
    LatticeJoinMeetConfig,
)


def test_generate_and_score():
    task = LatticeJoinMeet()
    cfg = LatticeJoinMeetConfig()
    for level in (0, 2, 5):
        cfg.set_level(level)
        task.config = cfg
        seen = 0
        for _ in range(50):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0
            assert _valid_domain(ex)
            seen += 1
        assert seen == 50


def _valid_domain(ex):
    j, m = (int(x) for x in ex.answer.split())
    assert j >= m
    assert j, m >= 1
    assert m == ex.metadata.meet
    assert j == ex.metadata.join
    assert ex.metadata.n % j == 0
    assert ex.metadata.n % m == 0
    return True


def test_answer_format():
    task = LatticeJoinMeet()
    ex = task.generate_example()
    join, meet = ex.answer.split()
    assert int(join) >= int(meet)


def test_junk_scoring():
    task = LatticeJoinMeet()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("garbage", ex) == 0.0
    assert task.score_answer("1", ex) == 0.0
    assert task.score_answer("a b", ex) == 0.0
