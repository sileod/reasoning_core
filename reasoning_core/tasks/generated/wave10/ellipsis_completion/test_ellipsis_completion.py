import random

from reasoning_core.tasks.generated.wave10.ellipsis_completion.ellipsis_completion import (
    EllipsisCompletion, EllipsisConfig,
)


def test_gold_scoring_all_levels():
    random.seed(123)
    for level in range(7):
        t = EllipsisCompletion()
        t.config = EllipsisConfig()
        t.config.apply_difficulty(level)
        e = t.generate_entry()
        assert t.score_answer(e.answer, e) == 1.0


def test_garbage_scores_zero():
    random.seed(7)
    t = EllipsisCompletion()
    t.config = EllipsisConfig()
    e = t.generate_entry()
    assert t.score_answer("", e) == 0.0
    assert t.score_answer("abc", e) == 0.0
    assert t.score_answer("12.5", e) == 0.0


def test_answer_is_expected_sum():
    random.seed(99)
    t = EllipsisCompletion()
    t.config = EllipsisConfig()
    e = t.generate_entry()
    idxs = e.metadata.target_indices
    expected = sum(e.metadata.amounts[i] for i in idxs)
    assert int(e.answer) == expected


def test_difficulty_changes_config():
    c = EllipsisConfig()
    base = c.n_turns
    c.apply_difficulty(6)
    assert c.n_turns > base


def test_metadata_json_round_trip():
    import json
    random.seed(5)
    t = EllipsisCompletion()
    t.config = EllipsisConfig()
    e = t.generate_entry()
    json.dumps(dict(e.metadata.payload))


def test_prompt_contains_dialogue():
    random.seed(11)
    t = EllipsisCompletion()
    t.config = EllipsisConfig()
    e = t.generate_entry()
    p = t.render_prompt(e.metadata)
    assert e.metadata.payload["dialogue"] in p
    assert "answer" in p
