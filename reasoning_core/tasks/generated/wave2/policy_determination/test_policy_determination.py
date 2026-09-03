import random
import re

from reasoning_core.tasks.generated.wave2.s21_policy_determination.policy_determination import (
    PolicyDetermination,
    PolicyConfig,
    total_amount,
)


def _new_task(level=0):
    random.seed(1234)
    task = PolicyDetermination()
    task.config = PolicyConfig()
    task.config.set_level(level)
    return task


def test_generates_and_scores_gold():
    task = _new_task()
    entry = task.generate_example()
    assert task.score_answer(entry.answer, entry) == 1.0


def test_random_answer_not_all_one():
    task = _new_task()
    entries = [task.generate_example() for _ in range(20)]
    scores = [task.score_answer("0", e) for e in entries]
    assert sum(s == 1.0 for s in scores) < 20


def test_difficulty_changes():
    task = _new_task(0)
    before = task.config.n_rules
    task.config.set_level(5)
    assert task.config.n_rules > before


def test_answer_not_figure_in_prompt():
    task = _new_task()
    entry = task.generate_example()
    prompt = task.render_prompt(entry.metadata)
    answer = int(entry.answer)
    for m in re.findall(r"[0-9][0-9,]*", prompt):
        num = int(m.replace(",", ""))
        if num == answer:
            raise AssertionError("answer leaked as a figure in the prompt")


def test_answer_is_strictly_positive():
    task = _new_task()
    for _ in range(30):
        entry = task.generate_example()
        assert int(entry.answer) > 0


def test_answer_spread():
    task = _new_task()
    seen = set()
    for _ in range(30):
        entry = task.generate_example()
        seen.add(entry.answer)
    assert len(seen) > 10


def test_metadata_json_roundtrip():
    import json

    task = _new_task()
    entry = task.generate_example()
    json.dumps({k: v for k, v in entry.metadata.items() if k != "payload"})
    json.dumps(entry.metadata.payload)


def test_all_levels_generate():
    task = PolicyDetermination()
    for level in range(0, 7):
        task.config = PolicyConfig()
        task.config.set_level(level)
        entry = task.generate_example()
        assert total_amount(entry.metadata.policy) == int(entry.answer)


def test_junk_scores_zero():
    task = _new_task()
    entry = task.generate_example()
    assert task.score_answer("", entry) < 1.0
    assert task.score_answer("banana", entry) < 1.0
