import random

import regex

from reasoning_core.tasks.mutated.wave0.m16_defeasible_conflict_depth.m16_defeasible_conflict_depth import (
    DefeasibleConflictDepth,
)


def _task(level=0):
    t = DefeasibleConflictDepth()
    t.config.set_level(level)
    return t


def test_difficulty_increases_depth():
    d = [_task(l).config.conflict_depth for l in (0, 2, 5)]
    assert d == sorted(d)
    assert len(set(d)) == 3


def test_roundtrip_scores_gold():
    for level in (0, 2, 5):
        t = _task(level)
        ex = t.generate_example()
        assert t.score_answer(ex.answer, ex) == 1.0


def test_all_labels_generated():
    random.seed(7)
    t = _task(2)
    seen = set()
    for _ in range(60):
        seen.add(t.generate_entry().answer)
    assert {"Yes", "No", "Maybe"} <= seen


def test_wrong_answer_scores_zero():
    random.seed(3)
    t = _task(0)
    ex = t.generate_example()
    other = {"Yes", "No", "Maybe"} - {ex.answer}
    assert all(t.score_answer(x, ex) == 0.0 for x in other)


def test_conflict_depth_recorded_in_metadata():
    t = _task(5)
    ex = t.generate_example()
    assert isinstance(ex.metadata.conflict_depth, int)
    assert ex.metadata.conflict_depth >= 1
