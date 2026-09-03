import random

from reasoning_core.tasks.generated.wave6.s59_interval_stabbing.interval_stabbing import (
    IntervalStabbing,
    _greedy_stab,
    _brute_min,
)


def test_gold_scores_one():
    random.seed(7)
    task = IntervalStabbing()
    for _ in range(50):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_greedy_matches_brute():
    random.seed(11)
    for _ in range(200):
        n = random.randint(1, 10)
        intervals = []
        for _i in range(n):
            lo = random.randint(0, 20)
            hi = lo + random.randint(1, 6)
            intervals.append((lo, hi))
        assert len(_greedy_stab(intervals)) == len(_brute_min(intervals))


def test_junk_scores_zero():
    task = IntervalStabbing()
    e = task.generate_example()
    assert task.score_answer("", e) < 1.0
    assert task.score_answer("garbage", e) < 1.0
    assert task.score_answer("1,2,3,99", e) < 1.0


def test_non_increasing_rejected():
    task = IntervalStabbing()
    e = task.generate_example()
    parts = [int(p) for p in e.answer.split(",")]
    bad = ",".join(str(p) for p in reversed(parts))
    assert task.score_answer(bad, e) < 1.0


def test_difficulty_changes():
    c = IntervalStabbing_config()
    base = c.n
    c.apply_difficulty(5)
    assert c.n >= base


def test_answer_is_absolute_and_valid():
    random.seed(3)
    for _ in range(60):
        t = IntervalStabbing()
        cfg = IntervalStabbing_config()
        cfg.seed = random.randrange(2 ** 32)
        cfg.set_level(random.randint(0, 6))
        t.config = cfg
        e = t.generate_entry()
        gold = [int(p) for p in e.answer.split(",")]
        assert sorted(gold) == gold
        intervals = e.metadata.intervals
        assert all(any(a <= p <= b for p in gold) for a, b in intervals)


def IntervalStabbing_config():
    return IntervalStabbing().config_cls()
