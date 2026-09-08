import random

from reasoning_core.tasks.generated.wave10.pragmatic_reference_generation.pragmatic_reference_generation import (
    RefGenTask, RefConfig, _identifies,
)


def test_generate_and_score():
    random.seed(123)
    task = RefGenTask()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_answer_identifies():
    random.seed(7)
    task = RefGenTask()
    for _ in range(40):
        x = task.generate_example()
        objs = x.metadata["objects"]
        target_name = x.metadata["target_name"]
        target = next(o for o in objs if o["name"] == target_name)
        attrs = x.metadata["chosen_attrs"]
        assert _identifies(attrs, target, objs)


def test_junk_scored_zero():
    random.seed(5)
    task = RefGenTask()
    x = task.generate_example()
    assert task.score_answer("", x) < 1.0
    assert task.score_answer("garbage nonsense", x) < 1.0


def test_level_changes_config():
    cfg = RefConfig()
    cfg.set_level(0)
    base = cfg.n_obj
    cfg2 = RefConfig()
    cfg2.set_level(6)
    assert cfg2.n_obj > base
