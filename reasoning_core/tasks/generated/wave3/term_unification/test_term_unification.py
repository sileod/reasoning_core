import pytest

from reasoning_core.tasks.generated.wave3.s27_term_unification.term_unification import (
    TermUnification,
    TermUnificationConfig,
    unify,
)


def test_generate_and_score_gold():
    task = TermUnification()
    for _ in range(50):
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_junk_scores_zero():
    task = TermUnification()
    for _ in range(50):
        entry = task.generate_example()
        assert task.score_answer("", entry) < 1.0
        assert task.score_answer("zzz", entry) < 1.0


def test_occur_check():
    assert unify(("var", "x"), ("func", "f", [("var", "x")])) is None


def test_basic_unify():
    s = unify(("var", "x"), ("const", "a"))
    assert s is not None
    assert s["x"] == ("const", "a")


def test_symbol_clash():
    assert unify(("func", "f", [("var", "x")]), ("func", "g", [("var", "x")])) is None


def test_arity_clash():
    assert unify(("func", "f", [("var", "x")]), ("func", "f", [("var", "x"), ("var", "y")])) is None


def test_none_and_identity_answers_present():
    task = TermUnification()
    task.config.set_level(2)
    saw_none = False
    saw_term = False
    for _ in range(60):
        e = task.generate_example()
        if e.answer == "none":
            saw_none = True
        else:
            saw_term = True
    assert saw_none
    assert saw_term


def test_difficulty_changes():
    cfg = TermUnificationConfig()
    base = int(cfg.max_depth)
    cfg.set_level(5)
    assert int(cfg.max_depth) >= base
