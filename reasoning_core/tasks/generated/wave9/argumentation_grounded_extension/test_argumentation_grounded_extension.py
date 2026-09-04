import random

from reasoning_core.template import Entry
from reasoning_core.tasks.generated.wave9.argumentation_grounded_extension.argumentation_grounded_extension import (
    ArgumentationGroundedExtension,
    ArgumentationGroundedExtensionConfig,
    _grounded,
    _grounded_labeling,
    _parse_answer,
)


def test_config_difficulty_changes():
    cfg = ArgumentationGroundedExtensionConfig()
    l0 = cfg.n_args
    cfg.set_level(3)
    assert cfg.n_args > l0


def test_generate_and_score_gold():
    random.seed(12345)
    task = ArgumentationGroundedExtension()
    for _ in range(20):
        ex = task.generate_example()
        assert isinstance(ex, Entry)
        assert ex.prompt
        assert task.score_answer(ex.answer, ex) == 1.0


def test_score_wrong_and_junk():
    random.seed(999)
    task = ArgumentationGroundedExtension()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("zzz", ex) == 0.0
    assert task.score_answer("reajrjrje9595!", ex) == 0.0
    if " " in ex.answer:
        one = ex.answer.split()[0]
        assert task.score_answer(one, ex) == 0.0


def test_grounded_implementations_agree():
    names = ["a", "b", "c", "d", "e"]
    cases = [
        set(),
        {("a", "b")},
        {("a", "b"), ("b", "a")},
        {("a", "b"), ("b", "c")},
        {("a", "b"), ("b", "c"), ("c", "a")},
        {("a", "b"), ("b", "c"), ("b", "d"), ("d", "b")},
    ]
    for att in cases:
        assert _grounded(names, att) == _grounded_labeling(names, att)


def test_parse_answer_variants():
    assert _parse_answer("a b") == frozenset({"a", "b"})
    assert _parse_answer("b a") == frozenset({"a", "b"})
    assert _parse_answer("none") == frozenset()
    assert _parse_answer("") == frozenset()
    assert _parse_answer(None) == frozenset()


def test_answer_nonempty_and_not_full():
    random.seed(7)
    task = ArgumentationGroundedExtension()
    seen = set()
    for _ in range(30):
        ex = task.generate_example()
        seen.add(ex.answer)
    assert len(seen) > 1
