"""Tests for the clarification_question task."""

import random

from reasoning_core.tasks.generated.wave10.clarification_question.clarification_question import (
    ClarificationQuestion,
    _apply,
)


def _sample(task, seed):
    random.seed(seed)
    return task.generate_example()


def test_gold_scores_one():
    task = ClarificationQuestion()
    for seed in range(20):
        ex = _sample(task, seed)
        assert task.score_answer(ex.answer, ex) == 1.0


def test_answers_vary():
    task = ClarificationQuestion()
    answers = set()
    for seed in range(30):
        ex = _sample(task, seed)
        answers.add(ex.answer)
    assert len(answers) > 5


def test_answer_distinct_entries():
    task = ClarificationQuestion()
    seen = set()
    for seed in range(30):
        ex = _sample(task, seed)
        a, b = (int(x) for x in ex.answer.split())
        assert a != b
        seen.add((a, b))
    assert len(seen) > 5


def test_junk_and_empty_score_zero():
    task = ClarificationQuestion()
    ex = _sample(task, 7)
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("banana", ex) == 0.0
    assert task.score_answer("one two three", ex) == 0.0


def test_apply_matches():
    assert _apply("the sum", 3, 4) == 7
    assert _apply("the product", 3, 4) == 12
    assert _apply("the larger", 3, 4) == 4
    assert _apply("the absolute difference", 3, 4) == 1
    assert _apply("the smaller", 3, 4) == 3


def test_difficulty_scales():
    task = ClarificationQuestion()
    cfg = task.config
    base = cfg.max_value
    cfg.set_level(6)
    assert cfg.max_value > base


def test_reproducible():
    task = ClarificationQuestion()
    ex1 = _sample(task, 12345)
    ex2 = _sample(task, 12345)
    assert ex1.answer == ex2.answer
    assert ex1.metadata.body == ex2.metadata.body
