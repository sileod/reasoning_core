import random

from reasoning_core.tasks.generated.wave10.request_revision.request_revision import (
    RequestRevision,
)


def test_gold_scores_one():
    for _ in range(200):
        task = RequestRevision()
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_junk_scores_zero():
    task = RequestRevision()
    e = task.generate_example()
    assert task.score_answer("banana", e) == 0.0
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("999999", e) == 0.0


def test_difficulty_changes():
    t = RequestRevision()
    c0 = t.config_cls()
    t.config_cls().apply_difficulty(0)
    c6 = t.config_cls()
    c6.apply_difficulty(6)
    assert c6.n_ops > c0.n_ops


def test_determinism_under_seed():
    random.seed(123)
    a = [RequestRevision().generate_example().answer for _ in range(20)]
    random.seed(123)
    b = [RequestRevision().generate_example().answer for _ in range(20)]
    assert a == b


def test_answer_domain():
    task = RequestRevision()
    for level in (0, 1, 3, 6):
        cfg = task.config_cls()
        cfg.apply_difficulty(level)
        t = RequestRevision()
        for _ in range(50):
            e = t.generate_example()
            if e.answer == "none":
                continue
            vals = [int(x) for x in e.answer.split(",")]
            assert all(1 <= v <= 12 for v in vals)
            assert len(vals) == len(set(vals))
