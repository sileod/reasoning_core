import random

from reasoning_core.tasks.generated.wave8.consistent_distributed_cut.consistent_distributed_cut import (
    ConsistentDistributedCut,
    _count_orphans,
)


def test_gold_scores_one():
    t = ConsistentDistributedCut()
    for _ in range(20):
        x = t.generate_example()
        assert t.score_answer(x.answer, x) == 1.0


def test_garbage_does_not_score_one():
    t = ConsistentDistributedCut()
    for _ in range(20):
        x = t.generate_example()
        assert t.score_answer("", x) < 1.0
        assert t.score_answer("banana", x) < 1.0
        assert t.score_answer("-1", x) < 1.0


def test_count_matches_direct_formula():
    t = ConsistentDistributedCut()
    for _ in range(20):
        x = t.generate_example()
        m = x.metadata.messages
        c = x.metadata.cuts
        assert _count_orphans(m, c) == int(x.answer)


def test_level_scales():
    base = ConsistentDistributedCut()
    base.config.set_level(0)
    l0 = base.config.n_msgs
    low = ConsistentDistributedCut()
    low.config.set_level(5)
    l5 = low.config.n_msgs
    assert l5 >= l0


def test_answer_is_non_negative_int():
    t = ConsistentDistributedCut()
    for _ in range(20):
        x = t.generate_example()
        v = int(x.answer)
        assert v >= 0
        assert v <= len(x.metadata.messages)


def test_reproducible_under_seed():
    random.seed(12345)
    t = ConsistentDistributedCut()
    a = t.generate_example().metadata
    random.seed(12345)
    t2 = ConsistentDistributedCut()
    b = t2.generate_example().metadata
    assert a.sizes == b.sizes
    assert a.cuts == b.cuts
    assert a.messages == b.messages


def test_metadata_json_roundtrip():
    import json

    t = ConsistentDistributedCut()
    x = t.generate_example()
    d = json.loads(json.dumps(x.metadata.__dict__))
    assert d["sizes"] == x.metadata.sizes
