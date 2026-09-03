import random

from reasoning_core.tasks.generated.wave8.poset_cover_query.poset_cover_query import (
    PosetCoverQuery,
    _build_poset,
)


def _covers_of(edge_set, names):
    adj = {x: [] for x in names}
    for (u, v) in edge_set:
        adj[u].append(v)
    reach = {}
    for x in names:
        seen = set()
        stack = [x]
        while stack:
            node = stack.pop()
            for nxt in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        reach[x] = seen
    covers = {}
    for x in names:
        cv = []
        for y in reach[x]:
            mids = [z for z in reach[x] if z != y and y in reach[z]]
            if not mids:
                cv.append(y)
        covers[x] = sorted(cv)
    return covers


def test_generate_roundtrip():
    random.seed(1)
    task = PosetCoverQuery()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_cover_math():
    random.seed(2)
    names = ["a", "b", "c", "d"]
    edge_set, covers = _build_poset(names, 1)
    covers2 = _covers_of(edge_set, names)
    assert covers == covers2
    # covers are strict upper neighbors: no intermediate
    for x, cv in covers.items():
        for y in cv:
            assert y in covers2[x]
            for z in names:
                assert not (z != y and y in _covers_of(edge_set, names)[z] and z in _covers_of(edge_set, names)[x])


def test_score_rejects_junk():
    random.seed(3)
    task = PosetCoverQuery()
    x = task.generate_example()
    assert task.score_answer("", x) == 0.0
    assert task.score_answer("garbage", x) == 0.0


def test_level_changes():
    task = PosetCoverQuery()
    task.config.set_level(0)
    n0 = int(task.config.n_elements)
    task.config.set_level(6)
    n6 = int(task.config.n_elements)
    assert n6 >= n0
