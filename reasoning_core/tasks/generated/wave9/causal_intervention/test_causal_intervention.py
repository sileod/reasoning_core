import random

from reasoning_core.tasks.generated.wave9.causal_intervention.causal_intervention import (
    CausalIntervention,
)


def _fixed_example(task, level):
    task.config.set_level(level)
    return task.generate_example()


def test_gold_score_value():
    task = CausalIntervention()
    task.config.set_level(2)
    entry = task.generate_example()
    assert task.score_answer(entry.answer, entry) == 1.0


def test_metadata_json_serializable():
    import json

    task = CausalIntervention()
    entry = task.generate_example()
    json.dumps(dict(entry.metadata))


def test_wrong_answer_scores_zero():
    task = CausalIntervention()
    entry = task.generate_example()
    assert task.score_answer("zzz", entry) == 0.0
    assert task.score_answer("", entry) == 0.0


def test_answers_are_floats():
    task = CausalIntervention()
    for level in range(0, 7):
        task.config.set_level(level)
        for _ in range(30):
            entry = task.generate_example()
            for gold in entry.metadata.gold_answers:
                assert isinstance(gold, float)


def test_probability_in_domain():
    task = CausalIntervention()
    for level in range(0, 7):
        task.config.set_level(level)
        for _ in range(30):
            entry = task.generate_example()
            for i, q in enumerate(entry.metadata.payload["queries"]):
                if q["type"] == "prob":
                    g = entry.metadata.gold_answers[i]
                    assert 0.0 <= g <= 1.0


def test_multi_query_format():
    task = CausalIntervention()
    task.config.set_level(5)
    for _ in range(30):
        entry = task.generate_example()
        golds = entry.metadata.gold_answers
        if len(golds) > 1:
            assert len(entry.answer.split()) == len(golds)
            assert task.score_answer(entry.answer, entry) == 1.0


def test_difficulty_changes_config():
    task = CausalIntervention()
    base = type(task.config)()
    task.config.set_level(3)
    assert task.config.n_nodes == base.n_nodes + 3


def test_distinct_answers():
    task = CausalIntervention()
    seen = set()
    for level in range(0, 7):
        task.config.set_level(level)
        for _ in range(40):
            seen.add(task.generate_example().answer)
    assert len(seen) >= 100
