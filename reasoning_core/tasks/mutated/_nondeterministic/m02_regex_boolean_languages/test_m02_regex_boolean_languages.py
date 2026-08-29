import random
from reasoning_core.tasks.mutated.wave0.m02_regex_boolean_languages.m02_regex_boolean_languages import (
    RegexBooleanLanguages,
    _shortest_witness,
)
from reasoning_core.tasks.regex import ALPHA
from greenery import parse as gparse


def _make(level=2, seed=1):
    random.seed(seed)
    t = RegexBooleanLanguages()
    t.config.set_level(level)
    return t


def test_gold_scores_one_across_levels():
    for level in (0, 2, 5):
        t = _make(level)
        for _ in range(10):
            e = t.generate_entry()
            assert t.score_answer(e.answer, e) == 1.0


def test_no_fixed_constant_wins():
    t = _make(3, seed=42)
    seen = {}
    for _ in range(200):
        e = t.generate_entry()
        seen[e.answer] = seen.get(e.answer, 0) + 1
    top, cnt = max(seen.items(), key=lambda kv: kv[1])
    assert cnt <= 200 * 0.4
    assert len(seen) > 20


def test_min_length_and_membership_sd():
    t = _make(2)
    for _ in range(20):
        e = t.generate_entry()
        if e.metadata.qtype != "sd":
            continue
        fa = gparse(e.metadata.regex_a).to_fsm()
        fb = gparse(e.metadata.regex_b).to_fsm()
        assert fa.accepts(e.answer) != fb.accepts(e.answer)
        assert e.answer == _shortest_witness(fa.symmetric_difference(fb), ALPHA[: t.config.n_alpha])


def test_min_length_and_membership_sub():
    t = _make(2)
    for _ in range(20):
        e = t.generate_entry()
        if e.metadata.qtype != "sub":
            continue
        fa = gparse(e.metadata.regex_a).to_fsm()
        fb = gparse(e.metadata.regex_b).to_fsm()
        fc = gparse(e.metadata.regex_c).to_fsm()
        assert fa.accepts(e.answer)
        assert not fb.accepts(e.answer)
        assert not fc.accepts(e.answer)
        diff = fa.difference(fb).difference(fc)
        assert e.answer == _shortest_witness(diff, ALPHA[: t.config.n_alpha])


def test_wrong_answers_do_not_score_one():
    t = _make(2)
    for _ in range(30):
        e = t.generate_entry()
        assert t.score_answer("zzz-not-a-witness-zzz", e) != 1.0


def test_levels_change_config():
    t = _make(0)
    base = t.config.n_alpha
    t2 = _make(5)
    assert t2.config.n_alpha >= base
    assert t2.config.max_depth > t.config.max_depth
