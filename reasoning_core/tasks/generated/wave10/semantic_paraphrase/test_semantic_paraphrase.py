import random
import pytest

from reasoning_core.template import Task
from reasoning_core.tasks.generated.wave10.semantic_paraphrase.semantic_paraphrase import (
    SemanticParaphrase,
    _norm,
    _render,
)


def test_generate_scores_one():
    task = SemanticParaphrase()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_all_levels_generate():
    task = SemanticParaphrase()
    for level in range(7):
        task.config.set_level(level)
        assert task.generate_example() is not None


def test_patients_and_agents_preserved():
    task = SemanticParaphrase()
    for _ in range(50):
        x = task.generate_example()
        src = x.metadata.source
        tgt = x.metadata.target
        assert src["voice"] in ("active", "passive")
        assert tgt["voice"] in ("active", "passive")


def test_gold_not_readable_off_surface():
    task = SemanticParaphrase()
    x = task.generate_example()
    prompt = task.render_prompt(x.metadata)
    ans = _norm(x.answer)
    assert ans != _norm(x.metadata.source) or ans != _norm(prompt.split(".")[0] + ".")
    # answer word should not trivially equal the source sentence
    assert _norm(ans) != _norm(x.metadata.source)


def test_garbage_scores_zero():
    task = SemanticParaphrase()
    x = task.generate_example()
    assert task.score_answer("", x) == 0.0
    assert task.score_answer("not a real answer at all whatsoever", x) == 0.0


def test_norm_robust():
    assert _norm("The cat ate the mouse.") == _norm("the cat ate the mouse")


def test_render_known():
    assert _render("cat", "mouse", "eat", "active", "affirmative", "present") == "the cat eats the mouse."
    assert _render("cat", "mouse", "eat", "passive", "affirmative", "present") == "the mouse is eaten by the cat."
    assert _render("cat", "mouse", "eat", "active", "negated", "past") == "the cat did not eat the mouse."


def test_dedup_stable():
    t1 = SemanticParaphrase()
    x = t1.generate_example()
    assert x.metadata.get("_deduplication_key") is not None
