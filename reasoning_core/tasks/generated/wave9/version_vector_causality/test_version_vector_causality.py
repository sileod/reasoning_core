import random

from reasoning_core.tasks.generated.wave9.version_vector_causality.version_vector_causality import (
    VersionVectorCausality, VectorCausalityConfig)


def test_gold_scores_one_at_levels():
    for level in range(7):
        cfg = VectorCausalityConfig()
        cfg.set_level(level)
        t = VersionVectorCausality(config=cfg)
        e = t.generate_entry()
        assert t.score_answer(e.answer, e) == 1.0


def test_answer_is_clock_vector():
    cfg = VectorCausalityConfig()
    t = VersionVectorCausality(config=cfg)
    for _ in range(30):
        e = t.generate_entry()
        parts = [int(x) for x in e.answer.split(",")]
        assert len(parts) == cfg.n_procs
        assert all(x >= 0 for x in parts)


def test_bracket_and_space_accept():
    cfg = VectorCausalityConfig()
    t = VersionVectorCausality(config=cfg)
    e = t.generate_entry()
    assert t.score_answer(f"[{e.answer}]", e) == 1.0
    assert t.score_answer(e.answer.replace(",", ", "), e) == 1.0


def test_garbage_scores_zero():
    cfg = VectorCausalityConfig()
    t = VersionVectorCausality(config=cfg)
    e = t.generate_entry()
    assert t.score_answer("", e) < 1.0
    assert t.score_answer("bogus", e) < 1.0
    assert t.score_answer(None, e) < 1.0


def test_reproducible_under_seed():
    random.seed(12345)
    cfg = VectorCausalityConfig()
    t = VersionVectorCausality(config=cfg)
    a = [t.generate_entry().answer for _ in range(5)]
    random.seed(12345)
    cfg2 = VectorCausalityConfig()
    t2 = VersionVectorCausality(config=cfg2)
    b = [t2.generate_entry().answer for _ in range(5)]
    assert a == b


def test_difficulty_changes():
    cfg = VectorCausalityConfig()
    cfg.set_level(0)
    l0 = cfg.n_events
    cfg2 = VectorCausalityConfig()
    cfg2.set_level(5)
    l5 = cfg2.n_events
    assert l5 > l0
