import networkx as nx

from reasoning_core.tasks.generated.wave9.maximum_flow.maximum_flow import MaximumFlow


def test_gold_scores_one():
    task = MaximumFlow()
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_garbage_scores_zero():
    task = MaximumFlow()
    ex = task.generate_example()
    for junk in ["", "abc", "-1", "999999999999", "None"]:
        assert task.score_answer(junk, ex) < 1.0


def test_answer_matches_networkx():
    task = MaximumFlow()
    ex = task.generate_example()
    graph = ex.metadata.payload["graph"]
    g = nx.DiGraph()
    for (u, v, c) in graph["edges"]:
        g.add_edge(u, v, capacity=c)
    val = nx.maximum_flow_value(g, graph["source"], graph["sink"])
    assert int(val) == int(ex.answer)


def test_levels_change():
    task = MaximumFlow()
    n0 = task.config.n_nodes
    c0 = task.config.max_cap
    task.config.set_level(1)
    assert task.config.n_nodes >= n0 and task.config.max_cap >= c0
    assert (task.config.n_nodes != n0 or task.config.max_cap != c0
            or abs(task.config.p - 0.5) > 0)


def test_varied_answers():
    task = MaximumFlow()
    task.config.set_level(4)
    answers = set()
    for _ in range(30):
        answers.add(task.generate_example().answer)
    assert len(answers) >= 5
