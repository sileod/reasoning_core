import random

import networkx as nx

from reasoning_core.tasks.generated.wave9.strongly_connected_components.strongly_connected_components import (
    StronglyConnectedComponents,
)


def test_gold_scores_and_is_valid_scc():
    random.seed(1)
    task = StronglyConnectedComponents()
    for level in [0, 2, 5]:
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_example()
            assert task.score_answer(ex.answer, ex) == 1.0
            meta = ex.metadata
            g = nx.DiGraph()
            g.add_nodes_from(range(meta.n))
            g.add_edges_from(meta.edges)
            sccs = sorted((sorted(c) for c in nx.strongly_connected_components(g)),
                          key=lambda c: c[0])
            if meta.mode == "component":
                query = meta.query
                comp = next(c for c in sccs if query in c)
                assert set(ex.answer.split(",")) == set(str(x) for x in comp)
            else:
                assert ex.answer == ";".join("[" + ",".join(map(str, c)) + "]" for c in sccs)


def test_wrong_answers_score_zero():
    random.seed(2)
    task = StronglyConnectedComponents()
    for level in [0, 2, 5]:
        task.config.set_level(level)
        for _ in range(10):
            ex = task.generate_example()
            assert task.score_answer("", ex) < 1.0
            assert task.score_answer("999,888,777", ex) < 1.0
            assert task.score_answer("not an answer", ex) < 1.0


def test_difficulty_changes_config():
    task = StronglyConnectedComponents()
    task.config.set_level(0)
    n0 = task.config.n_nodes
    task.config.set_level(5)
    assert task.config.n_nodes > n0
