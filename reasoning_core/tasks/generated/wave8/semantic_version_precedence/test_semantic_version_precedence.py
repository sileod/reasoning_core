"""Tests for semantic_version_precedence task."""
import random

from reasoning_core.tasks.generated.wave8.semantic_version_precedence.semantic_version_precedence import (
    SemanticVersionPrecedence,
    SemVerPrecedenceConfig,
    higher_precedence,
    _version_key,
)


def _cmp(a, b):
    ka, kb = _version_key(a), _version_key(b)
    if ka == kb:
        return 0
    if ka < kb:
        return -1
    return 1


def test_example_roundtrip():
    t = SemanticVersionPrecedence()
    e = t.generate_example()
    assert t.score_answer(e.answer, e) == 1.0
    assert t.render_prompt(e.metadata)


def test_known_semver_rules():
    cases = [
        ("1.0.0", "0.9.9", "1.0.0"),
        ("1.0.0", "1.0.0", "equal"),
        ("1.0.0", "1.0.0-alpha", "1.0.0"),
        ("1.0.0-alpha", "1.0.0-beta", "1.0.0-beta"),
        ("1.0.0-2", "1.0.0-10", "1.0.0-10"),     # numeric
        ("1.0.0-10", "1.0.0-2", "1.0.0-10"),
        ("1.0.0-1", "1.0.0-alpha", "1.0.0-alpha"),  # numeric < non-numeric
        ("1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.1"),  # longer higher
        ("1.0.0-alpha", "1.0.0-rc", "1.0.0-rc"),
        ("1.2.3", "1.10.3", "1.10.3"),
        ("1.0.0-gamma.2", "1.0.0-gamma.1", "1.0.0-gamma.2"),
        ("1.0.0-beta.11", "1.0.0-beta.2", "1.0.0-beta.11"),
    ]
    for a, b, expect in cases:
        got = higher_precedence(a, b)
        assert got == expect, (a, b, got, expect)
        if expect == "equal":
            assert _cmp(a, b) == 0


def test_gold_scores_one_many():
    random.seed(12345)
    t = SemanticVersionPrecedence()
    seen = set()
    for _ in range(200):
        e = t.generate_example()
        assert t.score_answer(e.answer, e) == 1.0
        seen.add(e.answer)
    assert len(seen) > 50
    assert "equal" in seen


def test_junk_not_one():
    t = SemanticVersionPrecedence()
    e = t.generate_example()
    assert t.score_answer("", e) < 1.0
    assert t.score_answer("garbage", e) < 1.0
    wrong = "1.0.0-beta" if e.answer != "1.0.0-beta" else "1.0.0-alpha"
    assert t.score_answer(wrong, e) < 1.0


def test_answer_in_prompt_or_equal():
    t = SemanticVersionPrecedence()
    for _ in range(100):
        e = t.generate_example()
        assert e.answer in (e.metadata.a, e.metadata.b, "equal")


def test_difficulty_changes_config():
    c0 = SemVerPrecedenceConfig()
    c0.set_level(0)
    c5 = SemVerPrecedenceConfig()
    c5.set_level(5)
    assert c5.max_pre_ids >= c0.max_pre_ids
    assert c5.pre_prob >= c0.pre_prob
