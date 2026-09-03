import random

from reasoning_core.tasks.generated.wave8.vector_clock_order.vector_clock_order import (
    VectorClockOrder,
    _classify,
)


def test_gold_scores_one():
    task = VectorClockOrder()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_junk_and_empty_not_correct():
    task = VectorClockOrder()
    x = task.generate_example()
    assert task.score_answer("", x) == 0.0
    assert task.score_answer("garbage", x) == 0.0


def test_classify_agrees_with_gold():
    task = VectorClockOrder()
    for _ in range(200):
        x = task.generate_example()
        assert _classify(x.metadata.u, x.metadata.v) == x.answer


def test_all_relations_appear():
    random.seed(123)
    task = VectorClockOrder()
    seen = set()
    for _ in range(200):
        x = task.generate_example()
        seen.add(x.answer)
    assert seen == {"equal", "before", "after", "concurrent"}


def test_balance_is_reasonable():
    random.seed(7)
    task = VectorClockOrder()
    counts = {}
    for _ in range(400):
        x = task.generate_example()
        counts[x.answer] = counts.get(x.answer, 0) + 1
    top = max(counts.values()) / 400.0
    assert top < 0.40


def test_difficulty_changes():
    cfg = VectorClockOrder.config_cls()
    cfg.set_level(0)
    c0 = (cfg.length, cfg.vmax)
    cfg2 = VectorClockOrder.config_cls()
    cfg2.set_level(5)
    assert (cfg2.length, cfg2.vmax) != c0
