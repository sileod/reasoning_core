import random

from reasoning_core.tasks.generated.wave9.pivot_unpivot_transform.pivot_unpivot_transform import (
    PivotUnpivotTransform,
)


TASK = PivotUnpivotTransform()


def test_generate_and_score():
    random.seed(1)
    for _ in range(5):
        entry = TASK.generate_entry()
        assert TASK.score_answer(entry.answer, entry) == 1.0


def test_junk_and_empty():
    random.seed(2)
    for _ in range(5):
        entry = TASK.generate_entry()
        assert TASK.score_answer("", entry) == 0.0
        assert TASK.score_answer("garbage input", entry) == 0.0


def test_levels():
    for level in [0, 1, 2, 3, 4, 5, 6]:
        task = PivotUnpivotTransform()
        task.config.set_level(level)
        random.seed(3)
        for _ in range(3):
            entry = task.generate_entry()
            assert TASK.score_answer(entry.answer, entry) == 1.0


def test_whitespace_insensitive():
    random.seed(4)
    for _ in range(5):
        entry = TASK.generate_entry()
        a = "   ".join(entry.answer.split())
        assert TASK.score_answer(a, entry) == 1.0


def test_pivot_both_operations():
    random.seed(5)
    ops = set()
    for _ in range(20):
        entry = TASK.generate_entry()
        ops.add(entry.metadata.operation)
    assert ops == {"pivot", "unpivot"}


def test_metadata_json_serializable():
    import json
    random.seed(6)
    for _ in range(3):
        entry = TASK.generate_entry()
        json.dumps(dict(entry.metadata))
