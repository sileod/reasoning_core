import random

import networkx as nx

from reasoning_core.tasks.generated.wave8.conflict_serializability.conflict_serializability import (
    ConflictSerializability,
    _serial_order,
)


def _check_answer(ex):
    """Verify the gold answer matches the precedence graph exactly."""
    prec = ex.metadata.precedence_edges
    n_trans = ex.metadata.n_trans
    if ex.answer == "NO":
        order = _serial_order(prec, n_trans)
        assert order is None, "claimed NO but graph has unique serial order"
    else:
        assert ex.answer.startswith("YES:")
        order = tuple(int(x) for x in ex.answer.split(":", 1)[1].split(","))
        assert len(order) == n_trans
        assert sorted(order) == list(range(1, n_trans + 1))
        # unique topo order must equal the claimed serial order
        uniq = _serial_order(prec, n_trans)
        assert uniq == order, f"mismatch: {uniq} vs {order}"


def test_gold_scores_one():
    t = ConflictSerializability()
    for _ in range(80):
        ex = t.generate_example()
        assert t.score_answer(ex.answer, ex) == 1.0


def test_answer_matches_precedence_graph():
    t = ConflictSerializability()
    for _ in range(80):
        ex = t.generate_example()
        _check_answer(ex)


def test_junk_scores_zero():
    t = ConflictSerializability()
    for _ in range(20):
        ex = t.generate_example()
        assert t.score_answer("", ex) == 0.0
        assert t.score_answer("jarble", ex) == 0.0
        assert t.score_answer("YES:1", ex) == 0.0


def test_both_labels_appear():
    t = ConflictSerializability()
    random.seed(1)
    seen_no = seen_yes = 0
    for _ in range(60):
        ex = t.generate_example()
        if ex.answer == "NO":
            seen_no += 1
        else:
            seen_yes += 1
    assert seen_no > 0 and seen_yes > 0


def test_levels_generate():
    for level in (0, 3, 6):
        cfg = ConflictSerializability.config_cls()
        cfg.set_level(level)
        t = ConflictSerializability(config=cfg)
        for _ in range(5):
            ex = t.generate_example()
            assert t.score_answer(ex.answer, ex) == 1.0
