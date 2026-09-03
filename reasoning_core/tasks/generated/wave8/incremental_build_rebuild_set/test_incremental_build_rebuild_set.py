import random

import pytest

from reasoning_core.tasks.generated.wave8.incremental_build_rebuild_set.incremental_build_rebuild_set import (
    RebuildSet,
    RebuildSetConfig,
    _compute_closure,
)


@pytest.fixture(autouse=True)
def seeded():
    random.seed(4082635274)


def test_module_import_and_meta():
    import reasoning_core.tasks.generated.wave8.incremental_build_rebuild_set.incremental_build_rebuild_set as m

    assert m.TASK_META["hypothesis"] == "W1-079"
    assert m.TASK_META["parent_source_id"] is None


def test_gold_scores_one():
    t = RebuildSet()
    for _ in range(50):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0


def test_junk_scores_zero():
    t = RebuildSet()
    e = t.generate_example()
    assert t.score_answer("", e) == 0.0
    assert t.score_answer(None, e) == 0.0
    assert t.score_answer("not a real target", e) == 0.0


def test_answer_deterministic_under_seed():
    random.seed(12345)
    t = RebuildSet()
    a = t.generate_example()
    random.seed(12345)
    t2 = RebuildSet()
    b = t2.generate_example()
    assert a.answer == b.answer
    assert a.metadata["payload"] == b.metadata["payload"]


def test_answer_varies_across_seeds():
    t = RebuildSet()
    answers = set()
    for s in range(30):
        random.seed(s)
        answers.add(t.generate_example().answer)
    assert len(answers) > 5, "answers should vary"


def test_closure_includes_changed_and_rebuilds():
    edges = {0: [], 1: [0], 2: [1], 3: []}
    assert _compute_closure(edges, 4, [0, 3]) == [0, 1, 2, 3]


def test_non_rebuild_not_included():
    edges = {0: [], 1: [0], 2: [], 3: [2]}
    r = _compute_closure(edges, 4, [2])
    assert 1 not in r
    assert r == [2, 3]


def test_all_levels_generate():
    t = RebuildSet()
    for level in range(7):
        cfg = RebuildSetConfig()
        cfg.set_level(level)
        t.config = cfg
        e = t.generate_example()
        assert e.answer


def test_difficulty_changes():
    c0 = RebuildSetConfig()
    c0.set_level(0)
    c6 = RebuildSetConfig()
    c6.set_level(6)
    assert c6.n_nodes > c0.n_nodes


def test_extra_or_missing_target_fails():
    t = RebuildSet()
    for _ in range(30):
        e = t.generate_example()
        gold = sorted(set(e.metadata["rebuild"]))
        if len(gold) > 1:
            missing = ", ".join(gold[1:])
            assert t.score_answer(missing, e) != 1.0
            extra = ", ".join(gold + ["ZZZ"])
            assert t.score_answer(extra, e) != 1.0
            break
    else:
        raise AssertionError("no multi-target example found")
