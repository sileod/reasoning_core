import random

from reasoning_core.tasks.generated.wave9.diff_patch_application.diff_patch_application import (
    DiffPatchApplication,
    DiffPatchConfig,
    _apply_hunks,
    _runify,
)


def test_gold_scores_one():
    task = DiffPatchApplication()
    for _ in range(20):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0
        assert e.answer == "[" + _runify(_apply_hunks(e.metadata.initial.split(","), [
            (h["op"], int(h["start"]), int(h["count"]), list(h["insert"]))
            for h in e.metadata.hunks
        ])) + "]"


def test_junk_scores_zero():
    task = DiffPatchApplication()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("garbage", e) == 0.0
    assert task.score_answer("[nonexistent]", e) == 0.0


def test_difficulty_changes():
    cfg = DiffPatchConfig()
    l0 = DiffPatchConfig()
    l0.apply_difficulty(0)
    l5 = DiffPatchConfig()
    l5.apply_difficulty(5)
    assert l5.n_hunks > l0.n_hunks
    assert l5.n_tokens > l0.n_tokens


def test_all_levels_generate():
    task = DiffPatchApplication()
    for lvl in range(7):
        task.config.set_level(lvl)
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_answer_variety():
    task = DiffPatchApplication()
    answers = set()
    for _ in range(30):
        e = task.generate_example()
        answers.add(e.answer)
    assert len(answers) > 10
