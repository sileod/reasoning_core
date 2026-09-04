import random

from reasoning_core.tasks.generated.wave9.topological_layering.topological_layering import (
    TopologicalLayering,
    _removal_order,
    _is_dag,
)


def _parse_answer(answer):
    return [int(x) for x in answer.split(",")]


def test_gold_scores_one():
    task = TopologicalLayering()
    for _ in range(200):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_order_is_valid_topological_order():
    task = TopologicalLayering()
    for _ in range(200):
        e = task.generate_example()
        metadata = e.metadata
        if "round" in metadata.payload:
            continue
        order = _parse_answer(e.answer)
        assert sorted(order) == list(
            range(len(metadata.payload["nodes"].split()))
        )


def test_round_removed_nodes_reproduce_order():
    task = TopologicalLayering()
    for _ in range(200):
        e = task.generate_example()
        metadata = e.metadata
        if "round" not in metadata.payload:
            continue
        n = len(metadata.payload["nodes"].split())
        edges = []
        for line in metadata.payload["edges"].split("\n"):
            u, v = line.split(" -> ")
            edges.append((int(u), int(v)))
        tie = "max" if "largest" in metadata.payload["tie"] else "min"
        order = _removal_order(n, edges, tie)
        assert order is not None
        pp = int(metadata.payload["nodes_per_round"])
        r = int(metadata.payload["round"])
        low = (r - 1) * pp
        high = min(n, low + pp)
        expected = sorted(order[low:high])
        assert expected == _parse_answer(e.answer)


def test_wrong_answer_scores_zero():
    task = TopologicalLayering()
    for _ in range(50):
        e = task.generate_example()
        wrong = e.answer + ",9"
        assert task.score_answer(wrong, e) == 0.0
        assert task.score_answer("", e) == 0.0
        assert task.score_answer("not a list", e) == 0.0


def test_difficulty_changes_config():
    task = TopologicalLayering()
    base = task.config.n_nodes
    task.config.set_level(5)
    assert task.config.n_nodes > base


def test_generation_survives_all_levels():
    task = TopologicalLayering()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(5):
            e = task.generate_example()
            assert task.score_answer(e.answer, e) == 1.0
