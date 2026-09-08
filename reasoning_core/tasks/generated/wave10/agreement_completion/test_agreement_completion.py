"""Tests for the agreement_completion task."""

import random

import pytest

from reasoning_core.tasks.generated.wave10.agreement_completion.agreement_completion import (
    VERBS,
    AgreementCompletion,
)


@pytest.fixture
def task():
    return AgreementCompletion()


def test_gold_scores_one(task):
    random.seed(7)
    for _ in range(64):
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_agreees_with_singular_head(task):
    config = task.config
    config.level = 0
    random.seed(1)
    for _ in range(40):
        entry = task.generate_example()
        sing = entry.metadata.payload["agreement"] == "singular"
        assert (entry.answer == entry.metadata.payload["base_verb"]) != sing


def test_every_level_generates(task):
    for level in range(7):
        task.config.set_level(level)
        for _ in range(8):
            entry = task.generate_example()
            assert task.score_answer(entry.answer, entry) == 1.0


def test_wrong_forms_do_not_score(task):
    random.seed(3)
    for _ in range(64):
        entry = task.generate_example()
        base = entry.metadata.payload["base_verb"]
        other = [s for (b, s) in VERBS if b == base][0]
        wrong = other if entry.answer == base else base
        assert task.score_answer(wrong, entry) == 0.0
        assert task.score_answer("", entry) == 0.0
        assert task.score_answer("banana", entry) == 0.0


def test_prompt_names_the_verb(task):
    random.seed(11)
    entry = task.generate_example()
    prompt = task.render_prompt(entry.metadata)
    assert entry.metadata.payload["base_verb"] in prompt
    assert "____" in prompt


def test_metadata_json_serializable(task):
    import json

    random.seed(13)
    for _ in range(16):
        entry = task.generate_example()
        json.dumps(dict(entry.metadata))
