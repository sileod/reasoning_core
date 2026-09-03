import random

from reasoning_core.tasks.generated.wave8.lexical_scope_resolution.lexical_scope_resolution import (
    LexicalScopeResolution,
    LexicalScopeResolutionV1Config,
    build_instance,
)


def test_generate_and_score():
    task = LexicalScopeResolution()
    task.config.set_level(2)
    for _ in range(10):
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_wrong_answers_fail():
    task = LexicalScopeResolution()
    task.config.set_level(1)
    entry = task.generate_example()
    assert task.score_answer("", entry) == 0.0
    assert task.score_answer("x = 99999", entry) == 0.0
    assert task.score_answer(123, entry) == 0.0


def test_difficulty_changes():
    c0 = LexicalScopeResolutionV1Config()
    c0.set_level(0)
    c6 = LexicalScopeResolutionV1Config()
    c6.set_level(6)
    assert c6.n_scopes > c0.n_scopes
    assert c6.n_decls > c0.n_decls


def test_all_levels_generate():
    task = LexicalScopeResolution()
    for level in range(0, 7):
        task.config.set_level(level)
        for _ in range(5):
            entry = task.generate_example()
            assert task.score_answer(entry.answer, entry) == 1.0


def test_build_instance_deterministic():
    rng = random.Random(42)
    a = build_instance(5, 6, rng)
    rng2 = random.Random(42)
    b = build_instance(5, 6, rng2)
    assert a == b
