import random

import networkx as nx

from reasoning_core.tasks.graph_operations import BaseGraphTask, GraphPathfinding


def test_graph_rendering_uses_global_random_state():
    graph = nx.DiGraph([(0, 1), (1, 2)])

    random.seed(17)
    BaseGraphTask._render_graph(None, graph)
    actual_state = random.getstate()

    random.seed(17)
    random.choice(range(8))
    assert actual_state == random.getstate()


def test_pathfinding_pair_weights_grow_with_level(monkeypatch):
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    task = GraphPathfinding()
    captured = {}

    def choose(population, weights, k):
        captured["pairs"] = population
        captured["weights"] = weights
        return [population[-1]]

    monkeypatch.setattr(random, "choices", choose)
    task.config.level = 4

    assert task._reachable_pair(graph) == (2, 3, 1)
    by_pair = dict(zip(captured["pairs"], captured["weights"]))
    assert len(by_pair) == 6
    assert all(weight > 0 for weight in by_pair.values())
    assert by_pair[(0, 3, 3)] == 9
    assert by_pair[(0, 1, 1)] == 1


def test_graph_size_keeps_moderate_level_growth():
    sizes = []
    for level in range(7):
        task = GraphPathfinding()
        task.config.apply_difficulty(level)
        sizes.append(task.config.num_nodes)

    assert sizes[0] == 6
    assert sizes[-1] < 50
    assert sizes == sorted(sizes)


def test_pathfinding_keeps_no_path_instruction_for_negative_cases():
    task = GraphPathfinding()
    prompt = task.render_prompt({
        "weighted": False,
        "mention_no_path": True,
        "mention_ties": True,
        "start_node": 0,
        "end_node": 1,
        "payload": {"graph": "Directed Edges:"},
    })

    assert "Break ties lexicographically." in prompt
    assert "`None` if no path exists" in prompt


def test_pathfinding_detects_only_optimal_ties():
    task = GraphPathfinding()
    graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)])
    nx.set_edge_attributes(graph, 1, "weight")

    assert not task._has_optimal_tie(graph, 0, 3)
    graph.remove_edge(0, 3)
    assert task._has_optimal_tie(graph, 0, 3)
