import random
import pytest
from reasoning_core.tasks.mutated.wave0.m03_arithmetic_dags.dag_arithmetics import (
    DagArithmetics,
    DagArithmeticsConfig,
    _eval_dag,
    _count_reuse,
)


def _gen(level=2, seed=1):
    random.seed(seed)
    t = DagArithmetics()
    t.config.set_level(level)
    return t


def test_gold_scores_one():
    for level in (0, 1, 2, 3, 4, 5):
        t = _gen(level)
        for _ in range(20):
            eg = t.generate_example()
            assert t.score_answer(eg.answer, eg) == 1.0


def test_dag_evaluation_matches_answer():
    t = _gen(2)
    for _ in range(30):
        eg = t.generate_example()
        env = _eval_dag(eg.metadata.decls)
        assert str(int(env[eg.metadata.root])) == eg.answer


def test_reuse_within_bounds():
    for level in (0, 2, 5):
        t = _gen(level)
        c = t.config
        lo, hi = c.min_reuse, c.max_reuse
        seen = set()
        for _ in range(60):
            eg = t.generate_example()
            r = _count_reuse(eg.metadata.decls)
            assert lo <= r <= hi
            seen.add(int(eg.answer))
        assert len(seen) > 1


def test_depth_within_bounds():
    t = _gen(5)
    for _ in range(30):
        eg = t.generate_example()
        assert t.config.min_depth <= len(eg.metadata.decls) <= t.config.max_depth


def test_difficulty_changes_config():
    t0 = _gen(0)
    t5 = _gen(5)
    assert t5.config.max_depth > t0.config.max_depth
    assert t5.config.max_reuse >= t0.config.max_reuse


def test_rendered_prompt_contains_root():
    t = _gen(2)
    eg = t.generate_example()
    assert eg.metadata.root in t.render_prompt(eg.metadata)


def test_wrong_answer_not_one():
    t = _gen(2)
    for _ in range(20):
        eg = t.generate_example()
        wrong = str(int(eg.answer) + 1)
        assert t.score_answer(wrong, eg) < 1.0


def test_metadata_json_roundtrip():
    import json
    t = _gen(3)
    eg = t.generate_example()
    json.loads(json.dumps(eg.metadata))
