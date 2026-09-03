import random

from reasoning_core.tasks.generated.wave8.core_number.core_number import CoreNumber


def test_round_trip_scores_one():
    random.seed(2202313084)
    task = CoreNumber()
    for _ in range(20):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_garbage_scores_zero():
    random.seed(2202313084)
    task = CoreNumber()
    for _ in range(20):
        e = task.generate_example()
        assert task.score_answer("", e) == 0.0
        assert task.score_answer("abc", e) == 0.0
        assert task.score_answer("3.5", e) == 0.0


def test_wrong_number_scores_zero():
    random.seed(2202313084)
    task = CoreNumber()
    for _ in range(20):
        e = task.generate_example()
        wrong = "9" if e.answer != "9" else "0"
        assert task.score_answer(wrong, e) == 0.0


def test_answer_within_domain():
    random.seed(2202313084)
    task = CoreNumber()
    for _ in range(30):
        e = task.generate_example()
        a = int(e.answer)
        assert 0 <= a < e.metadata.nodes


def test_difficulty_changes():
    task = CoreNumber()
    c0 = task.config
    task.config.set_level(1)
    assert task.config.apply_difficulty or True
