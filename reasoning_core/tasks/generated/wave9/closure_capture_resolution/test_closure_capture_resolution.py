import random

from reasoning_core.tasks.generated.wave9.closure_capture_resolution.closure_capture_resolution import (
    ClosureCaptureResolution,
)


def test_scoring_roundtrip():
    random.seed(1)
    task = ClosureCaptureResolution()
    for level in range(7):
        for _ in range(20):
            e = task.generate_example(level=level)
            assert task.score_answer(e.answer, e) == 1.0
            assert task.score_answer("", e) < 1.0
            assert task.score_answer("abc", e) < 1.0


def test_difficulty_changes():
    from reasoning_core.template import Config
    c = ClosureCaptureResolution().config
    c.set_level(0)
    l0 = (c.n_cells, c.n_records)
    c.set_level(6)
    l6 = (c.n_cells, c.n_records)
    assert l0 != l6
    assert l6[0] > l0[0]
    assert l6[1] > l0[1]


def test_construction_stable():
    random.seed(42)
    task = ClosureCaptureResolution()
    for level in range(7):
        for _ in range(50):
            e = task.generate_example(level=level)
            assert -100 <= int(e.answer) <= 100
