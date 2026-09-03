import os
import random
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))

from articulation_vertices import ArticulationVertices, _check_articulations


def _task_for(level, seed):
    t = ArticulationVertices()
    t.config.set_level(level)
    random.seed(seed)
    return t


def test_gold_scores_1():
    for level in range(7):
        t = _task_for(level, 123 + level)
        for _ in range(30):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0


def test_wrong_answers_score_0():
    random.seed(7)
    t = ArticulationVertices()
    for _ in range(50):
        e = t.generate_example()
        assert t.score_answer("", e) == 0.0
        assert t.score_answer("junk", e) == 0.0
        assert t.score_answer("[]", e) == 0.0 or e.answer == "[]"


def test_verifier_matches_networkx():
    G = nx.cycle_graph(5)
    assert _check_articulations(G, list(nx.articulation_points(G))) is None
    G2 = nx.complete_graph(4)
    assert _check_articulations(G2, set()) is None


def test_generation_matches_verifier():
    for level in [0, 3, 6]:
        t = _task_for(level, 5)
        for _ in range(30):
            e = t.generate_example()
            G = nx.Graph()
            G.add_nodes_from(range(e.metadata.n_nodes))
            G.add_edges_from(e.metadata.edges)
            assert _check_articulations(G, list(nx.articulation_points(G))) is None
            arts = sorted(nx.articulation_points(G))
            assert "[" + ", ".join(map(str, arts)) + "]" == e.answer


def test_summary_nonempty():
    assert len(ArticulationVertices().summary) > 20
