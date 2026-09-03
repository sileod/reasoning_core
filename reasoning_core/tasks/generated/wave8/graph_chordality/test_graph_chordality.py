import random
import networkx as nx
from reasoning_core.template import Task
from reasoning_core.tasks.generated.wave8.graph_chordality.graph_chordality import (
    GraphChordality,
    _longest_chordless_cycle,
)


def test_gold_scores_1():
    random.seed(42)
    task = GraphChordality()
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_answer_matches_graph():
    random.seed(7)
    task = GraphChordality()
    for L in range(3):
        task.config.set_level(L)
    task.config.set_level(0)
    for _ in range(20):
        ex = task.generate_entry()
        g = nx.Graph()
        g.add_nodes_from(ex.metadata["nodes"])
        g.add_edges_from(ex.metadata["edges"])
        if nx.is_chordal(g):
            assert ex.answer == "true"
        else:
            assert ex.answer.startswith("false ")
            l = int(ex.answer.split()[1])
            assert l == ex.metadata["longest_chordless_cycle"]
            assert ex.metadata["longest_chordless_cycle"] >= 4


def test_garbage_scores_0():
    random.seed(3)
    task = GraphChordality()
    ex = task.generate_example()
    assert task.score_answer("", ex) < 1.0
    assert task.score_answer("garbage", ex) < 1.0


def test_longest_chordless_cycle_matches_nx():
    random.seed(11)
    for _ in range(15):
        g = nx.gnp_random_graph(random.randint(4, 8), 0.5, seed=random)
        mine = _longest_chordless_cycle(g)
        if nx.is_chordal(g):
            assert mine == 0
        else:
            assert mine >= 4
