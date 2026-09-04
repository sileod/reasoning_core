import random

from reasoning_core.template import Entry
from reasoning_core.tasks.generated.wave9.dataflow_liveness.dataflow_liveness import (
    DataflowLiveness,
)


def test_gold_scores_one():
    random.seed(12345)
    task = DataflowLiveness()
    for _ in range(200):
        e = task.generate_entry()
        assert task.score_answer(e.answer, e) == 1.0


def test_junk_scores_zero():
    random.seed(999)
    task = DataflowLiveness()
    e = task.generate_entry()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("garbage", e) == 0.0
    assert task.score_answer(None, e) == 0.0


def test_answer_format_and_length():
    random.seed(42)
    task = DataflowLiveness()
    for _ in range(50):
        e = task.generate_entry()
        parts = e.answer.split()
        assert len(parts) == e.metadata.n_blocks
        for p in parts:
            assert p.startswith("{") and p.endswith("}")
            inner = p[1:-1]
            if inner:
                for v in inner.split(","):
                    assert v in "abcdefghijklmnopqrstuvwxyz"


def test_difficulty_changes():
    task = DataflowLiveness()
    c0 = task.config_cls()
    c0.set_level(0)
    c6 = task.config_cls()
    c6.set_level(6)
    assert c6.n_blocks >= c0.n_blocks


def test_all_levels_generate():
    task = DataflowLiveness()
    for level in range(7):
        c = task.config_cls()
        c.set_level(level)
        task2 = DataflowLiveness()
        task2.config = c
        for _ in range(10):
            e = task2.generate_entry()
            assert task2.score_answer(e.answer, e) == 1.0
