import random

from reasoning_core.tasks.generated.wave9.articulation_bridges.articulation_bridges import (
    ArticulationBridges,
    _tarjan,
    _cc_without,
)


def _run_low_level(level):
    random.seed(0)
    task = ArticulationBridges()
    task.config.set_level(level)
    return [task.generate_example() for _ in range(5)]


def test_gold_scores_1():
    random.seed(0)
    task = ArticulationBridges()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(30):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0


def test_junk_scores_0():
    random.seed(0)
    task = ArticulationBridges()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_example()
            assert task.score_answer("", ex) == 0.0
            assert task.score_answer(None, ex) == 0.0
            assert task.score_answer("bogus", ex) == 0.0


def test_articulation_verified():
    for level in range(7):
        for ex in _run_low_level(level):
            if ex.metadata.mode == 0:
                _, bridges = _tarjan(_adj(ex), ex.metadata.n)
                got = set(int(x) for x in ex.answer.split(","))
                ap, _ = _tarjan(_adj(ex), ex.metadata.n)
                assert got == ap


def test_bridge_verified():
    for level in range(7):
        for ex in _run_low_level(level):
            if ex.metadata.mode == 1:
                got = set(tuple(int(y) for y in b.split("-")) for b in ex.answer.split(";"))
                _, bridges = _tarjan(_adj(ex), ex.metadata.n)
                expected = set(tuple(sorted(e)) for e in bridges)
                assert got == expected


def test_components_verified():
    for level in range(7):
        for ex in _run_low_level(level):
            if ex.metadata.mode == 2:
                n = ex.metadata.n
                adj = _adj(ex)
                assert int(ex.answer) == _cc_without(adj, n, ex.metadata.target)


def _adj(ex):
    n = ex.metadata.n
    adj = [[] for _ in range(n)]
    for u, v in ex.metadata.edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def test_answer_variety():
    random.seed(1)
    task = ArticulationBridges()
    task.config.set_level(3)
    answers = set()
    for _ in range(100):
        ex = task.generate_example()
        answers.add(ex.answer)
    assert len(answers) > 20
