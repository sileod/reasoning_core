import random

from reasoning_core.tasks.generated.wave9.deadlock_detection.deadlock_detection import (
    DeadlockDetection,
    DeadlockDetectionConfig,
    _greedy,
    _can_complete,
    _normalize,
)


def test_gold_scores_one_at_multiple_levels():
    task = DeadlockDetection()
    for level in (0, 2, 5):
        task.config.set_level(level)
        for _ in range(10):
            e = task.generate_example()
            assert task.score_answer(e.answer, e) == 1.0
            assert task.score_answer("", e) == 0.0
            assert task.score_answer("garbage", e) == 0.0


def test_both_flavors_appear():
    task = DeadlockDetection()
    flavors = {"safe": False, "deadlock": False}
    for _ in range(80):
        e = task.generate_example()
        flavors[e.metadata.flavor] = True
        assert "," in e.answer
    assert flavors["safe"] and flavors["deadlock"]


def test_answers_vary():
    task = DeadlockDetection()
    for level in (0, 3, 6):
        task.config.set_level(level)
        ans = {task.generate_example().answer for _ in range(25)}
        assert len(ans) >= 8


def test_greedy_matches_scc_check_on_generated():
    task = DeadlockDetection()
    for _ in range(40):
        e = task.generate_example()
        n = e.metadata.n
        holder = e.metadata.holder
        requests = e.metadata.requests
        ordered, deadlocked = _greedy(n, holder, requests)
        completable = set(_can_complete(n, holder, requests))
        assert set(deadlocked) == set(range(n)) - completable
        if e.metadata.flavor == "safe":
            assert len(ordered) == n
        else:
            assert deadlocked


def test_difficulty_changes():
    cfg = DeadlockDetectionConfig()
    cfg.apply_difficulty(0)
    n0 = cfg.n_processes
    cfg.set_level(6)
    assert cfg.n_processes > n0


def test_normalize_trims():
    assert _normalize(" 1 , 3 , 5 ") == "1 , 3 , 5"
