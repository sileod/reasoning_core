import random

from reasoning_core.tasks.generated.wave9.grammar_first_follow.grammar_first_follow import (
    GrammarFirstFollow,
    _compute,
    EPS,
    EOF,
)


def test_round_trip_all_levels():
    random.seed(1)
    t = GrammarFirstFollow()
    for L in range(7):
        t.config.set_level(L)
        for _ in range(5):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0


def test_junk_and_empty_score_zero():
    random.seed(2)
    t = GrammarFirstFollow()
    for L in range(3):
        t.config.set_level(L)
        e = t.generate_example()
        assert t.score_answer("zzz not-a-symbol", e) == 0.0
        assert t.score_answer("%%%", e) == 0.0
        assert t.score_answer("empty", e) < 1.0


def test_answers_vary():
    random.seed(3)
    t = GrammarFirstFollow()
    seen = set()
    for _ in range(40):
        e = t.generate_example()
        seen.add(e.answer)
    assert len(seen) > 5


def test_difficulty_scales():
    t = GrammarFirstFollow()
    t.config.set_level(0)
    base = (t.config.n_nonterm, t.config.n_term)
    t.config.set_level(6)
    hi = (t.config.n_nonterm, t.config.n_term)
    assert hi != base


def test_domains():
    random.seed(4)
    t = GrammarFirstFollow()
    for L in range(7):
        t.config.set_level(L)
        e = t.generate_example()
        for tok in e.answer.split():
            assert tok in ("empty", EPS, EOF) or (len(tok) == 1 and tok.isalpha())
        if e.metadata.query.startswith("FIRST"):
            assert all(tok != EOF for tok in e.answer.split())
