import random

from reasoning_core.tasks.generated.wave9.transitive_reduction.transitive_reduction import (
    TransitiveReduction,
    transitive_reduction,
    reachability_edges,
    edges_to_answer,
)


def test_summary_present():
    assert isinstance(TransitiveReduction.summary, str) and TransitiveReduction.summary


def test_default_generates_and_scores():
    random.seed(0)
    t = TransitiveReduction()
    for _ in range(20):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_edge_list_roundtrip():
    random.seed(1)
    t = TransitiveReduction()
    for _ in range(30):
        e = t.generate_example()
        if "query" in e.metadata:
            continue
        ans = e.answer
        assert ans.count(";") >= 0
        edges = {tuple(x.strip().split(" -> ")) for x in ans.split(";")}
        assert len(edges) >= 1


def _label_of(s):
    return int(s[1:])


def test_transitive_reduction_preserves_reachability():
    random.seed(2)
    t = TransitiveReduction()
    for _ in range(30):
        e = t.generate_example()
        n = e.metadata["n"]
        edges = _parse_edges(e.metadata["edges"], n)
        red = transitive_reduction(edges, n)
        assert reachability_edges(edges, n) == reachability_edges(list(red), n)


def _parse_edges(s, n):
    out = []
    toks = s.split()
    assert len(toks) % 3 == 0
    for i in range(0, len(toks), 3):
        a, arrow, b = toks[i], toks[i + 1], toks[i + 2]
        assert arrow == "->"
        out.append((int(a[1:]), int(b[1:])))
    return out


def test_garbage_scores_zero():
    t = TransitiveReduction()
    e = t.generate_example()
    assert t.score_answer("", e) == 0.0
    assert t.score_answer("garbage", e) < 1.0


def test_difficulty_changes_config():
    t = TransitiveReduction()
    base = t.config.n_nodes
    t.config.set_level(6)
    assert t.config.n_nodes > base


def test_query_mode_scores_exact():
    random.seed(3)
    t = TransitiveReduction()
    seen = set()
    for _ in range(400):
        e = t.generate_example()
        if "query" in e.metadata:
            seen.add(e.answer)
            assert t.score_answer(e.answer, e) == 1.0
            assert t.score_answer("nope", e) == 0.0
    assert seen <= {"yes", "no"}


def test_both_query_answers_appear():
    random.seed(4)
    t = TransitiveReduction()
    yes = no = 0
    for _ in range(800):
        e = t.generate_example()
        if "query" in e.metadata:
            if e.answer == "yes":
                yes += 1
            else:
                no += 1
    assert yes > 0 and no > 0
