import random

from reasoning_core.tasks.generated.wave6.s60_dependency_batches.dependency_batches import (
    DependencyBatches,
    DependencyBatchesConfig,
)


def test_gold_scores_one():
    random.seed(1)
    task = DependencyBatches()
    task.config = DependencyBatchesConfig()
    for _ in range(20):
        e = task.generate_entry()
        assert task.score_answer(e.answer, e) == 1.0


def test_rounds_partition_and_layer():
    random.seed(2)
    task = DependencyBatches()
    task.config = DependencyBatchesConfig()
    for _ in range(20):
        e = task.generate_entry()
        rounds = [tuple(r) for r in e.metadata.rounds]
        assert rounds is not None
        names = {n for r in rounds for n in r}
        assert names == set(e.metadata.prereqs.keys())
        round_idx = {n: i for i, r in enumerate(rounds) for n in r}
        for b, deps in e.metadata.prereqs.items():
            for a in deps:
                assert round_idx[a] < round_idx[b]


def test_empty_answer_not_gold():
    random.seed(3)
    task = DependencyBatches()
    task.config = DependencyBatchesConfig()
    e = task.generate_entry()
    assert task.score_answer("", e) < 1.0
    assert task.score_answer("garbage", e) < 1.0


def test_difficulty_scales():
    c = DependencyBatchesConfig()
    c.set_level(1)
    assert c.n_jobs > DependencyBatchesConfig().n_jobs or c.depth > DependencyBatchesConfig().depth


def test_generate_high_level():
    random.seed(4)
    task = DependencyBatches()
    for level in (0, 3, 6):
        c = DependencyBatchesConfig()
        c.set_level(level)
        task.config = c
        e = task.generate_entry()
        assert e.answer
        assert task.score_answer(e.answer, e) == 1.0
