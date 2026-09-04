import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from reasoning_core.tasks.generated.wave9.round_robin_scheduling.round_robin_scheduling import (
    RoundRobinScheduling,
    _simulate,
)


def test_gold_scores_one():
    random.seed(1)
    t = RoundRobinScheduling()
    for _ in range(50):
        e = t.generate_entry()
        assert t.score_answer(e.answer, e) == 1.0


def test_simulate_consistency_with_query():
    random.seed(2)
    t = RoundRobinScheduling()
    for _ in range(30):
        e = t.generate_entry()
        p = e.metadata.payload
        running, waiting, completion = _simulate(
            p["n"], p["quantum"], list(p["arrivals"]), list(p["bursts"])
        )
        trg = e.metadata.target
        if e.metadata.chosen == "executions":
            assert int(e.answer) == running[trg]
        elif e.metadata.chosen == "waiting":
            assert int(e.answer) == waiting[trg]
        else:
            assert int(e.answer) == completion[trg]


def test_junk_and_empty_score_zero():
    t = RoundRobinScheduling()
    e = t.generate_example()
    assert t.score_answer("", e) == 0.0
    assert t.score_answer("abc", e) == 0.0
    assert t.score_answer("-3", e) == 0.0


def test_answer_nonnegative():
    random.seed(3)
    t = RoundRobinScheduling()
    for level in range(7):
        t.config.set_level(level)
        for _ in range(10):
            e = t.generate_entry()
            assert int(e.answer) >= 0
