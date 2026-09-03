import random

from reasoning_core.tasks.generated.wave8.round_robin_completion.round_robin_completion import (
    RoundRobinCompletion,
    compute_completion_order,
)


def test_gold_scoring():
    t = RoundRobinCompletion()
    random.seed(1)
    for _ in range(50):
        x = t.generate_example()
        assert t.score_answer(x.answer, x) == 1.0


def test_garbage_scoring():
    t = RoundRobinCompletion()
    random.seed(2)
    for _ in range(50):
        x = t.generate_example()
        assert t.score_answer("", x) == 0.0
        assert t.score_answer("abc,def", x) == 0.0


def test_completion_check_is_exact():
    assert compute_completion_order([0, 0], [3, 3], 2) == [1, 2]
    assert compute_completion_order([0, 1], [4, 2], 2) == [2, 1]
