import random
import networkx as nx

from reasoning_core.tasks.generated.wave0.n08_bipartite_matching.bipartite_matching import BipartiteMatching


def _max_card(edges, left, right):
    g = nx.Graph()
    for l, r in edges:
        g.add_edge(l, left + r)
    return len(nx.max_weight_matching(g, maxcardinality=True))


def _expected_card(graph):
    edges = {e for e in graph["edges"]}
    if isinstance(next(iter(edges)), tuple) and len(next(iter(edges))) == 3:
        edges = {(l, r) for l, r, _ in edges}
    return _max_card(edges, len(graph["left"]), len(graph["right"]))


def test_validate_all_levels():
    random.seed(2431745573)
    for L in (0, 2, 5):
        t = BipartiteMatching()
        t.config.set_level(L)
        assert t.validate()


def test_score_matches_networkx():
    random.seed(7)
    t = BipartiteMatching()
    for L in (0, 1, 2, 3, 5):
        t2 = BipartiteMatching()
        t2.config.set_level(L)
        ex = t2.generate_example()
        graph = ex.metadata.payload["graph"]
        expected = _expected_card(graph)
        assert t2.score_answer(str(expected), ex) == 1.0
        assert ex.answer == str(expected)


def test_wrong_answers_rejected():
    random.seed(11)
    t = BipartiteMatching()
    t.config.set_level(2)
    ex = t.generate_example()
    assert t.score_answer("999", ex) == 0.0
    assert t.score_answer("banana", ex) == 0.0
    assert t.score_answer(None, ex) == 0.0
