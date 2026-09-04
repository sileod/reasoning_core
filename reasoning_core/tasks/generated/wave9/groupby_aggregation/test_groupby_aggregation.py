import random
import tempfile
import os


def _fresh_task(level=0):
    from reasoning_core.tasks.generated.wave9.groupby_aggregation.groupby_aggregation import (
        GroupbyAggregation, GroupbyAggregationConfig,
    )
    random.seed(12345)
    task = GroupbyAggregation()
    cfg = GroupbyAggregationConfig()
    cfg.apply_difficulty(level)
    task.config = cfg
    return task


def test_generate_roundtrip():
    task = _fresh_task(0)
    for _ in range(20):
        e = task.generate_entry()
        assert task.score_answer(e.answer, e) == 1.0


def test_levels():
    from reasoning_core.tasks.generated.wave9.groupby_aggregation.groupby_aggregation import (
        GroupbyAggregation, GroupbyAggregationConfig,
    )
    for level in range(7):
        random.seed(level * 7 + 1)
        task = GroupbyAggregation()
        cfg = GroupbyAggregationConfig()
        cfg.apply_difficulty(level)
        task.config = cfg
        for _ in range(10):
            e = task.generate_entry()
            assert task.score_answer(e.answer, e) == 1.0


def test_garbage_and_empty():
    task = _fresh_task(0)
    e = task.generate_entry()
    assert task.score_answer("", e) < 1.0
    assert task.score_answer("G999", e) < 1.0
    assert task.score_answer("G0 G1", e) < 1.0
    assert task.score_answer(None, e) == 0.0


def test_answer_not_surface():
    task = _fresh_task(3)
    answers = set()
    for _ in range(40):
        answers.add(task.generate_entry().answer)
    assert len(answers) > 1


def test_metadata_json_serializable():
    import json
    task = _fresh_task(2)
    e = task.generate_entry()
    json.dumps(e.metadata.payload)
    json.dumps({"r": e.metadata.rows, "o": e.metadata.op})
