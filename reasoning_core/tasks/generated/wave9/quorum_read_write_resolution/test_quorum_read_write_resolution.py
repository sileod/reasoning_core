import random

from reasoning_core.tasks.generated.wave9.quorum_read_write_resolution.quorum_read_write_resolution import (
    QuorumReadWriteResolution,
    _score_answer,
)


def test_gold_scores_1():
    random.seed(1175081579)
    task = QuorumReadWriteResolution()
    for level in (0, 1, 2, 3, 4, 5, 6):
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_example()
            assert _score_answer(ex.answer, ex) == 1.0
            assert ex.answer.isdigit() or ex.answer.lstrip("-").isdigit()


def test_junk_scores_0():
    random.seed(1)
    task = QuorumReadWriteResolution()
    task.config.set_level(3)
    ex = task.generate_example()
    assert _score_answer("", ex) == 0.0
    assert _score_answer("abc", ex) == 0.0
    assert _score_answer("1.5", ex) == 0.0


def test_difficulty_changes_config():
    task = QuorumReadWriteResolution()
    task.config.set_level(0)
    n0 = task.config.n_ops
    task.config.set_level(6)
    n6 = task.config.n_ops
    assert n6 > n0


def test_answer_is_valid_value():
    random.seed(7)
    task = QuorumReadWriteResolution()
    for level in (0, 2, 5):
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_example()
            assert 0 <= int(ex.answer) < task.config.value_range


def test_reproducible_seeded():
    random.seed(123)
    t1 = QuorumReadWriteResolution(); t1.config.set_level(3)
    e1 = t1.generate_example()
    random.seed(123)
    t2 = QuorumReadWriteResolution(); t2.config.set_level(3)
    e2 = t2.generate_example()
    assert e1.answer == e2.answer
    assert e1.metadata.ops == e2.metadata.ops
    assert e1.metadata.queried_read == e2.metadata.queried_read
