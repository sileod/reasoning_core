import random

from reasoning_core.template import Entry
from reasoning_core.tasks.generated.wave9.expected_utility_choice.expected_utility_choice import (
    ExpectedUtilityChoice,
    ExpectedUtilityConfig,
)


def test_generate_and_score():
    random.seed(12345)
    task = ExpectedUtilityChoice()
    cfg = ExpectedUtilityConfig()
    task.config = cfg
    entry = task.generate_example()
    assert isinstance(entry, Entry)
    assert entry.answer
    assert task.score_answer(entry.answer, entry) == 1.0


def test_difficulty_changes_config():
    cfg = ExpectedUtilityConfig()
    cfg.set_level(5)
    assert cfg.n_actions > ExpectedUtilityConfig().n_actions


def test_wrong_answer():
    random.seed(7)
    task = ExpectedUtilityChoice()
    task.config = ExpectedUtilityConfig()
    entry = task.generate_example()
    assert task.score_answer("99999.0", entry) == 0.0


def test_junk_answer():
    entry = Entry(metadata={"answer_float": 1.0}, answer="1")
    task = ExpectedUtilityChoice()
    assert task.score_answer("not a number", entry) == 0.0
