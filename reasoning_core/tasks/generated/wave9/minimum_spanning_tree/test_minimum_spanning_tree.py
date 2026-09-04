import random

from reasoning_core.tasks.generated.wave9.minimum_spanning_tree.minimum_spanning_tree import (
    MSTConfig,
    MinimumSpanningTree,
    _kruskal,
)


def test_score_gold():
    random.seed(7)
    task = MinimumSpanningTree()
    for _ in range(20):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_score_wrong():
    random.seed(11)
    task = MinimumSpanningTree()
    for _ in range(10):
        e = task.generate_example()
        assert task.score_answer("0", e) in (0.0, 1.0)
        assert task.score_answer("", e) == 0.0
        assert task.score_answer("abc", e) == 0.0


def test_difficulty_changes():
    cfg = MSTConfig()
    base_n = cfg.n_vertices
    cfg.set_level(3)
    assert cfg.n_vertices > base_n


def test_kruskal_matches_answer():
    random.seed(3)
    task = MinimumSpanningTree()
    for _ in range(10):
        e = task.generate_example()
        total, _ = _kruskal(e.metadata.n_vertices, [tuple(x) for x in e.metadata.edges])
        assert total == int(e.answer)
