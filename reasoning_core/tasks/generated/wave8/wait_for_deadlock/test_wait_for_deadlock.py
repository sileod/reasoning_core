import random

from reasoning_core.tasks.generated.wave8.wait_for_deadlock.wait_for_deadlock import (
    WaitForDeadlock, _find_deadlocked,
)


def test_gold_answer_scores_one():
    random.seed(5)
    t = WaitForDeadlock()
    for _ in range(20):
        e = t.generate_entry()
        assert t.score_answer(e.answer, e) == 1.0
        assert t.score_answer(" " + e.answer + " ", e) == 1.0


def test_junk_scores_zero():
    random.seed(6)
    t = WaitForDeadlock()
    for _ in range(20):
        e = t.generate_entry()
        assert t.score_answer("", e) == 0.0
        assert t.score_answer("garbage", e) == 0.0
        assert t.score_answer(None, e) == 0.0


def test_answer_matches_finder():
    random.seed(7)
    t = WaitForDeadlock()
    for _ in range(50):
        e = t.generate_entry()
        dead = _find_deadlocked(e.metadata.n_txns, e.metadata.edges)
        expected = ",".join(f"T{d}" for d in sorted(dead)) or "none"
        assert e.answer == expected


def test_answer_in_domain():
    random.seed(8)
    t = WaitForDeadlock()
    for _ in range(50):
        e = t.generate_entry()
        n = len(e.metadata.payload["transactions"])
        for label in e.metadata.dead:
            assert 0 <= label <= n - 1


def test_difficulty_changes():
    t = WaitForDeadlock()
    c0 = t.config_cls()
    c0.set_level(0)
    c6 = t.config_cls()
    c6.set_level(6)
    assert c6.n_txns > c0.n_txns


def test_finder_detects_known_cycle():
    # T0 -> T1 -> T2 -> T0 forms a cycle.
    assert _find_deadlocked(3, [(0, 1), (1, 2), (2, 0)]) == {0, 1, 2}
    # Acyclic path.
    assert _find_deadlocked(3, [(0, 1), (1, 2)]) == set()
