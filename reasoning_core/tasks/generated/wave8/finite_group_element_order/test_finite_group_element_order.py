import random

from reasoning_core.tasks.generated.wave8.finite_group_element_order.finite_group_element_order import (
    FiniteGroupElementOrder,
    parse_order,
)


def test_gold_answer_scores_one():
    task = FiniteGroupElementOrder()
    for level in (0, 1, 2, 3, 4, 5, 6):
        task.config = type(task.config)()
        task.config.set_level(level)
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_wrong_answers():
    task = FiniteGroupElementOrder()
    entry = task.generate_example()
    gold = parse_order(entry.answer)
    wrong = gold + 1 if gold != 1 else gold + 1
    assert task.score_answer(str(wrong), entry) == 0.0


def test_junk_scores_zero():
    task = FiniteGroupElementOrder()
    entry = task.generate_example()
    assert task.score_answer("", entry) == 0.0
    assert task.score_answer("not a number", entry) == 0.0
    assert task.score_answer(None, entry) == 0.0


def test_answers_vary():
    task = FiniteGroupElementOrder()
    seen = set()
    for _ in range(40):
        entry = task.generate_example()
        seen.add(parse_order(entry.answer))
    assert len(seen) > 1


def test_answer_in_domain():
    task = FiniteGroupElementOrder()
    for _ in range(40):
        entry = task.generate_example()
        order = parse_order(entry.answer)
        assert order is not None and order >= 1
