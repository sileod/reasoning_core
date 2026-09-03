import random

from reasoning_core.tasks.generated.wave8.fenwick_prefix_nodes.fenwick_prefix_nodes import (
    FenwickPrefixNodes,
    _fenwick_path,
)


def test_gold_paths():
    assert _fenwick_path(1) == [1]
    assert _fenwick_path(12) == [12, 8]
    assert _fenwick_path(16) == [16]
    assert _fenwick_path(13) == [13, 12, 8]
    assert _fenwick_path(0) == []


def test_generate_score_roundtrip():
    random.seed(123)
    task = FenwickPrefixNodes()
    for _ in range(50):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_answer_valid_domain():
    random.seed(7)
    task = FenwickPrefixNodes()
    for _ in range(50):
        e = task.generate_example()
        vals = [int(x) for x in e.answer.split(",")]
        assert vals == sorted(vals, reverse=True)
        assert all(v > 0 for v in vals)
        # must start at the given index
        assert vals[0] == e.metadata.payload["index"]


def test_score_rejects_junk():
    task = FenwickPrefixNodes()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("abc", e) == 0.0
    assert task.score_answer("999", e) == 0.0
    assert task.score_answer("0", e) == 0.0


def test_gold_match_selfcheck():
    # every path's successive differences equal the lowbit
    for i in range(1, 65):
        path = _fenwick_path(i)
        for a, b in zip(path, path[1:]):
            assert a - b == a & -a
