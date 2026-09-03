import random
import sys

from reasoning_core.tasks.generated.wave8.open_addressing_slot.open_addressing_slot import (
    OpenAddressingSlot,
    _query_slot,
    _insert_slots,
)


def test_gold_scores_one():
    random.seed(64225588)
    task = OpenAddressingSlot()
    for _ in range(50):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_garbage_scores_zero():
    random.seed(123)
    task = OpenAddressingSlot()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("abc", ex) == 0.0
    assert task.score_answer("99999", ex) == 0.0


def test_answer_deterministic_construction():
    insert_keys = [5, 8, 13]
    step = 1
    size = 11
    result = _insert_slots(insert_keys, step, size)
    assert result is not None
    slots, occupied = result
    for key, cell in slots.items():
        assert occupied[cell]
    assert len(slots) == len(insert_keys)


def test_answer_within_domain():
    random.seed(7)
    task = OpenAddressingSlot()
    for level in range(7):
        task.config.set_level(level)
        ex = task.generate_example()
        ans = int(ex.answer)
        assert 0 <= ans < ex.metadata.size


def test_difficulty_changes():
    random.seed(1)
    task = OpenAddressingSlot()
    task.config.set_level(0)
    s0 = task.config.size
    task.config.set_level(5)
    s5 = task.config.size
    assert s5 > s0
