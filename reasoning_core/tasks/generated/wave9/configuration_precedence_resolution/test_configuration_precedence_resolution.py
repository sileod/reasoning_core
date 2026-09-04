import random

from reasoning_core.tasks.generated.wave9.configuration_precedence_resolution.configuration_precedence_resolution import (
    ConfigurationPrecedenceResolution,
    _apply,
    _render_actions,
    _parse,
)


def test_gold_scores_one():
    task = ConfigurationPrecedenceResolution()
    entry = task.generate_example()
    assert task.score_answer(entry.answer, entry) == 1.0


def test_wrong_answer_scores_zero():
    task = ConfigurationPrecedenceResolution()
    entry = task.generate_example()
    wrong = "zzzz = WRONG"
    if _parse(wrong) == _parse(entry.answer):
        wrong = "zzzz = WRONG2"
    assert task.score_answer(wrong, entry) == 0.0


def test_empty_and_junk():
    task = ConfigurationPrecedenceResolution()
    entry = task.generate_example()
    assert task.score_answer("", entry) < 1.0
    assert task.score_answer("garbage!!!", entry) < 1.0


def test_layer_order_matters():
    defaults = [{"a": "d0", "b": "d1"}]
    stronger = [{"b": None}]
    eff = _apply(defaults + stronger)
    assert "b" not in eff
    assert eff["a"] == "d0"


def test_override_and_remove():
    layers = [
        {"a": "v1", "b": "v2"},
        {"a": "v3", "b": None},
    ]
    eff = _apply(layers)
    assert eff == {"a": "v3"}


def test_readd_after_delete():
    layers = [
        {"a": "v1"},
        {"a": None},
        {"a": "v2"},
    ]
    assert _apply(layers) == {"a": "v2"}


def test_parse_render_roundtrip():
    actions = {"a": "X", "b": None, "c": "Y"}
    rendered = _render_actions(actions)
    assert _parse(rendered) == actions


def test_difficulty_changes_config():
    task = ConfigurationPrecedenceResolution()
    task.config.set_level(0)
    n0 = task.config.n_keys
    task.config.set_level(5)
    assert task.config.n_keys > n0


def test_reproducible_under_seed():
    random.seed(12345)
    task = ConfigurationPrecedenceResolution()
    e1 = task.generate_example()
    random.seed(12345)
    task2 = ConfigurationPrecedenceResolution()
    e2 = task2.generate_example()
    assert e1.answer == e2.answer
