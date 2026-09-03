import random

from reasoning_core.tasks.generated.wave4.s47_dimensional_analysis.s47_dimensional_analysis import (
    DimensionalAnalysisTask,
)

random.seed(1833805255)


def test_generate_and_score():
    task = DimensionalAnalysisTask()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(20):
            e = task.generate_entry()
            assert task.score_answer(e.answer, e) == 1.0


def test_wrong_answers_do_not_score():
    task = DimensionalAnalysisTask()
    task.config.set_level(2)
    correct = 0
    total = 0
    for _ in range(60):
        e = task.generate_entry()
        total += 1
        if task.score_answer(e.answer, e) == 1.0:
            correct += 1
        assert task.score_answer("", e) == 0.0
        assert task.score_answer("garbage", e) == 0.0
    assert correct == total


def test_answers_vary():
    task = DimensionalAnalysisTask()
    task.config.set_level(3)
    seen = set()
    for _ in range(50):
        e = task.generate_entry()
        seen.add(e.answer)
    assert len(seen) > 10


def test_metadata_serializable():
    import json

    task = DimensionalAnalysisTask()
    for level in range(7):
        task.config.set_level(level)
        e = task.generate_entry()
        json.dumps(dict(e.metadata.payload))
