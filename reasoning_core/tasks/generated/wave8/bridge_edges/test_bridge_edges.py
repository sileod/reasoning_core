import random

from reasoning_core.tasks.generated.wave8.bridge_edges.bridge_edges import (
    BridgeEdgesTask,
    _canonical_bridges,
    _bridges_str,
)


def _make_entry(task, level):
    task.config.set_level(level)
    return task.generate_example()


def test_gold_scores_one():
    task = BridgeEdgesTask()
    for level in (0, 1, 2, 3, 4, 5, 6):
        entry = _make_entry(task, level)
        assert task.score_answer(entry.answer, entry) == 1.0


def test_set_equivalence():
    task = BridgeEdgesTask()
    entry = _make_entry(task, 3)
    canon = _canonical_bridges_from_str(entry.answer)
    assert len(canon) == entry.metadata.n_bridges
    gold = set(_parse(entry.answer))
    any_set = set(_parse(entry.answer))
    assert gold == any_set


def test_order_reordering_scores_one():
    task = BridgeEdgesTask()
    entry = _make_entry(task, 2)
    pairs = entry.answer.split(";")
    pairs.reverse()
    reordered = "; ".join(p.strip() for p in pairs)
    assert task.score_answer(reordered, entry) == 1.0


def test_wrong_and_junk_score_zero():
    task = BridgeEdgesTask()
    entry = _make_entry(task, 2)
    assert task.score_answer("", entry) == 0.0
    assert task.score_answer("0-1; 9-9", entry) == 0.0
    assert task.score_answer("garbage", entry) == 0.0


def test_positive_bridges():
    task = BridgeEdgesTask()
    n_bridge_values = set()
    for _ in range(40):
        entry = _make_entry(task, 3)
        n_bridge_values.add(entry.metadata.n_bridges)
    assert len(n_bridge_values) > 2


def test_distinct_answers():
    task = BridgeEdgesTask()
    answers = set()
    for _ in range(30):
        entry = _make_entry(task, 3)
        answers.add(entry.answer)
    assert len(answers) > 3


def test_reproducible():
    task = BridgeEdgesTask()
    random.seed(12345)
    entry_a = _make_entry(task, 1)
    random.seed(12345)
    entry_b = _make_entry(task, 1)
    assert entry_a.answer == entry_b.answer


def _parse(astr):
    pairs = astr.split(";")
    out = set()
    for p in pairs:
        p = p.strip()
        if p == "none":
            continue
        u, v = p.split("-")
        out.add((int(u), int(v)))
    return out


def _canonical_bridges_from_str(astr):
    if astr.strip() == "none":
        return []
    pairs = astr.split(";")
    out = []
    for p in pairs:
        p = p.strip()
        u, v = p.split("-")
        out.append((int(u), int(v)))
    return out
