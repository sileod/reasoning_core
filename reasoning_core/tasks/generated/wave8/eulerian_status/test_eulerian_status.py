import random

import networkx as nx

from reasoning_core.tasks.generated.wave8.eulerian_status.eulerian_status import (
    EulerianStatus,
    EulerianStatusConfig,
    _build_graph,
)


def _decode(ans):
    parts = ans.split()
    if parts[0] == "none":
        return ("none", None, int(parts[1]))
    return (parts[0], int(parts[1]), None)


def test_three_statuses_present():
    random.seed(12345)
    task = EulerianStatus()
    seen = set()
    for _ in range(200):
        e = task.generate_example(level=3)
        seen.add(e.answer.split()[0])
    assert seen == {"circuit", "open", "none"}


def test_answers_match_graph():
    random.seed(999)
    task = EulerianStatus()
    for _ in range(120):
        e = task.generate_example(level=4)
        status, start, k = _decode(e.answer)
        G = nx.Graph()
        G.add_nodes_from(range(e.metadata.n_verts))
        G.add_edges_from(tuple(x) for x in e.metadata.edges)
        assert nx.is_connected(G)
        odd = sorted(v for v in G.nodes if G.degree(v) % 2 == 1)
        if status == "circuit":
            assert start == min(G.nodes) and len(odd) == 0
        elif status == "open":
            assert start == odd[0] and len(odd) == 2
        else:
            assert k == len(odd) and len(odd) >= 4
        assert task.score_answer(e.answer, e) == 1.0
        assert task.score_answer("", e) < 1.0
        assert task.score_answer("garbage", e) < 1.0


def test_difficulty_scaling():
    c0 = EulerianStatusConfig()
    c6 = EulerianStatusConfig()
    c6.set_level(6)
    assert c6.n_verts > c0.n_verts
