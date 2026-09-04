import random

from reasoning_core.tasks.generated.wave9.bipartite_matching.bipartite_matching import (
    BipartiteMatching,
    _has_augmenting_path,
    _hopcroft_karp,
)


def test_gold_scores_one():
    random.seed(0)
    task = BipartiteMatching()
    for _ in range(50):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_junk_not_one():
    random.seed(1)
    task = BipartiteMatching()
    ex = task.generate_example()
    assert task.score_answer("", ex) < 1.0
    assert task.score_answer("import x", ex) < 1.0
    assert task.score_answer("reajrjrje9595!", ex) < 1.0
    assert task.score_answer(None, ex) < 1.0


def test_matching_is_maximum():
    random.seed(2)
    task = BipartiteMatching()
    for level in (0, 3, 6):
        ex = task.generate_example(level=level)
        nL, nR = ex.metadata.left, ex.metadata.right
        edges = ex.metadata.edges.splitlines()
        adj = []
        for line in edges:
            body = line.split(":", 1)[1]
            adj.append([int(t.replace("R", "")) for t in body.replace(",", " ").split() if t.startswith("R")])
        pairL, pairR = _hopcroft_karp(nL, nR, adj)
        assert not _has_augmenting_path(nL, nR, adj, pairL, pairR), "HK must be maximum"


def test_difficulty_changes():
    task = BipartiteMatching()
    base = task.config.max_left
    task.config.set_level(5)
    assert task.config.max_left > base


def test_answer_in_domain():
    random.seed(3)
    task = BipartiteMatching()
    for _ in range(50):
        ex = task.generate_example()
        if ex.answer == "None":
            continue
        idx = int(ex.answer)
        assert 0 <= idx < ex.metadata.right


def test_match_consistency_with_gold():
    random.seed(4)
    task = BipartiteMatching()
    for _ in range(50):
        ex = task.generate_example()
        q = ex.metadata.query
        if ex.answer == "None":
            continue
        partner = int(ex.answer)
        edges = ex.metadata.edges.splitlines()
        line = edges[q].split(":", 1)[1]
        right_ids = [int(t.replace("R", "")) for t in line.replace(",", " ").split() if t.startswith("R")]
        assert partner in right_ids, "answer partner must be an edge of the queried vertex"
