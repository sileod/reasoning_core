import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from reasoning_core.tasks.generated.wave10.answer_relevant_change.answer_relevant_change import (
    AnswerRelevantChange,
)


def test_gold_scores_one():
    random.seed(123)
    task = AnswerRelevantChange()
    for _ in range(200):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0, e.answer


def test_wrong_scores_zero():
    random.seed(7)
    task = AnswerRelevantChange()
    for _ in range(200):
        e = task.generate_example()
        expected = set(int(x) for x in e.answer.split(","))
        # a wrong variant: add or drop a label
        alt = set(expected)
        alt.add(999) if not alt else alt.pop()
        wrong = ",".join(str(x) for x in sorted(alt))
        assert task.score_answer(wrong, e) == 0.0


def test_junk_scores_zero():
    task = AnswerRelevantChange()
    e = task.generate_example()
    assert task.score_answer("garbage", e) == 0.0
    assert task.score_answer("", e) == 0.0


def test_answers_vary():
    random.seed(42)
    task = AnswerRelevantChange()
    answers = set()
    for _ in range(300):
        answers.add(task.generate_example().answer)
    assert len(answers) > 5


def test_difficulty_changes():
    task = AnswerRelevantChange()
    task.config.set_level(0)
    l0 = task.config.n_items
    task.config.set_level(6)
    l6 = task.config.n_items
    assert l6 >= l0
