import random

from reasoning_core.tasks.generated.wave9.ledger_reconciliation.ledger_reconciliation import (
    LedgerReconciliation,
)


def test_gold_scores_one():
    task = LedgerReconciliation()
    random.seed(12345)
    for _ in range(50):
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0


def test_garbage_scores_zero():
    task = LedgerReconciliation()
    random.seed(99)
    for _ in range(20):
        x = task.generate_example()
        assert task.score_answer("garbage", x) == 0.0
        assert task.score_answer("", x) == 0.0


def test_levels_generate():
    task = LedgerReconciliation()
    for level in (0, 1, 2, 3, 4, 5, 6):
        cfg = task.config
        cfg.set_level(level)
        task.config = cfg
        for _ in range(3):
            x = task.generate_example()
            assert x.answer


def test_difficulty_changes():
    task = LedgerReconciliation()
    c0 = task.config
    c0.set_level(0)
    n0 = c0.n_ops
    c1 = task.config
    c1.set_level(5)
    assert c1.n_ops > n0
