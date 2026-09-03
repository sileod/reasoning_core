import random

from reasoning_core.tasks.generated.wave8.topological_generation.topological_generation import (
    TopologicalGeneration,
    _rounds_removal,
)


def test_round_removal_full():
    n = 4
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    rounds, total, rounds_of = _rounds_removal(n, edges)
    assert total == n
    assert rounds >= 1
    assert rounds_of[0] == 1


def test_generate_and_score_all_levels():
    random.seed(12345)
    task = TopologicalGeneration()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0
            assert int(ex.answer) >= 1


def test_answer_domain():
    random.seed(7)
    task = TopologicalGeneration()
    seen = set()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(30):
            ex = task.generate_example()
            r = int(ex.answer)
            assert r >= 1
            assert r <= task.config.depth
            seen.add(r)
    assert len(seen) >= 2


def test_junk_answer_zero():
    random.seed(7)
    task = TopologicalGeneration()
    task.config.set_level(3)
    ex = task.generate_example()
    assert task.score_answer("abc", ex) == 0.0
    assert task.score_answer("", ex) == 0.0
    gold = int(ex.answer)
    assert task.score_answer(str(gold + 1000), ex) == 0.0


def test_round_equals_longest_chain():
    random.seed(99)
    task = TopologicalGeneration()
    task.config.set_level(1)
    for _ in range(50):
        ex = task.generate_example()
        edges = [tuple(e) for e in ex.metadata.edges]
        n = ex.metadata.n_nodes
        _, total, rounds_of = _rounds_removal(n, edges)
        assert total == n
        assert rounds_of[ex.metadata.target] == int(ex.answer)
