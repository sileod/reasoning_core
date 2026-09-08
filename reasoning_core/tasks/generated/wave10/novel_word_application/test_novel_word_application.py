import pytest

from reasoning_core.tasks.generated.wave10.novel_word_application.novel_word_application import NovelWordApplication, NovelWordApplicationConfig


def _normalize(s):
    return "".join(c for c in s.lower() if c.isalpha())


def test_generates_and_scores_gold():
    task = NovelWordApplication()
    for _ in range(60):
        e = task.generate_example()
        assert e.answer
        assert task.score_answer(e.answer, e) == 1.0


def test_gold_never_equals_any_defined_stem():
    task = NovelWordApplication()
    for _ in range(40):
        e = task.generate_example()
        info = e.metadata.info
        defined = set(info["stems"]) if "stems" in info else {info["stem"]}
        assert e.answer not in defined


def test_answer_alpha_only_and_lowercase():
    task = NovelWordApplication()
    for _ in range(40):
        e = task.generate_example()
        assert e.answer.isalpha()
        assert e.answer == e.answer.lower()


def test_junk_and_empty_score_zero():
    task = NovelWordApplication()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("garbage", e) == 0.0
    assert task.score_answer(None, e) == 0.0


def test_varied_answers():
    task = NovelWordApplication()
    seen = set()
    for _ in range(80):
        seen.add(task.generate_example().answer)
    assert len(seen) >= 50


def test_all_levels_generate():
    for level in (0, 1, 2, 3, 4, 5, 6):
        cfg = NovelWordApplicationConfig()
        cfg.apply_difficulty(level)
        task = NovelWordApplication(config_cls=NovelWordApplicationConfig)
        task.config = cfg
        assert task.generate_example().answer


def test_difficulty_changes_config():
    c0 = NovelWordApplicationConfig(); c0.apply_difficulty(0)
    c6 = NovelWordApplicationConfig(); c6.apply_difficulty(6)
    assert c6.level == 6 and c0.level == 0
