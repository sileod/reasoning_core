import random

from reasoning_core.tasks.generated.wave9.minimal_unsat_core.minimal_unsat_core import (
    MinimalUnsatCore,
)


def test_generate_and_score():
    random.seed(12345)
    task = MinimalUnsatCore()
    entry = task.generate_example()
    assert task.score_answer(entry.answer, entry) == 1.0


def test_junk_scores_zero():
    random.seed(999)
    task = MinimalUnsatCore()
    entry = task.generate_example()
    assert task.score_answer("", entry) == 0.0
    assert task.score_answer("garbage", entry) == 0.0
    assert task.score_answer("[1,2,999]", entry) != 1.0


def test_difficulty_changes_config():
    task = MinimalUnsatCore()
    base = int(task.config.n_constraints)
    task.config.set_level(5)
    assert int(task.config.n_constraints) > base


def test_answer_varies():
    random.seed(42)
    task = MinimalUnsatCore()
    variety = set()
    for _ in range(25):
        entry = task.generate_example()
        variety.add(entry.answer)
    assert len(variety) > 5


def test_levels_generate():
    task = MinimalUnsatCore()
    for level in (0, 3, 6):
        task.config.set_level(level)
        entry = task.generate_example(level=level)
        assert task.score_answer(entry.answer, entry) == 1.0
