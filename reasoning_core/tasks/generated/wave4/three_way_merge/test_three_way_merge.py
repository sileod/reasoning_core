import random

from reasoning_core.tasks.generated.wave4.s43_three_way_merge.three_way_merge import (
    ThreeWayMerge,
    ThreeWayMergeConfig,
    build_merged,
    apply_ops,
    score_answer,
)


def test_gold_scores_1():
    t = ThreeWayMerge()
    for _ in range(200):
        e = t.generate_example()
        assert score_answer(e.answer, e) == 1.0


def test_junk_scores_less():
    t = ThreeWayMerge()
    for _ in range(50):
        e = t.generate_example()
        assert score_answer("garbage", e) < 1.0
        assert score_answer("", e) < 1.0


def test_difficulty_changes():
    c = ThreeWayMergeConfig()
    c.set_level(0)
    n0 = int(c.n_lines)
    c.set_level(5)
    n5 = int(c.n_lines)
    assert n5 > n0


def test_clean_merge_applies_both():
    base = ["a", "b", "c"]
    a = [(0, "chg", "A"), (2, "ins", "I")]
    b = [(1, "del", None)]
    merged, conflict = build_merged(base, a, b)
    assert conflict is None
    assert merged == ["A", "I", "c"]


def test_conflict_detected():
    base = ["a", "b", "c"]
    a = [(0, "chg", "A")]
    b = [(0, "del", None)]
    merged, conflict = build_merged(base, a, b)
    assert merged is None
    assert conflict == 0


def test_variety_of_answers():
    t = ThreeWayMerge()
    random.seed(12345)
    t.config.set_level(0)
    seen = set()
    for _ in range(100):
        e = t.generate_example()
        seen.add(e.answer)
    assert len(seen) >= 20
