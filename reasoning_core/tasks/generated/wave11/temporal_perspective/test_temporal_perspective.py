import random

from reasoning_core.tasks.generated.wave11.temporal_perspective.temporal_perspective import (
    TemporalPerspective,
)


def test_gold_scores_one():
    random.seed(7)
    task = TemporalPerspective()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_difficulty_changes():
    cfg = TemporalPerspective().config
    before = cfg.n_events
    cfg.set_level(3)
    assert cfg.n_events != before


def test_levels_generate():
    task = TemporalPerspective()
    for level in range(7):
        task.config.set_level(level)
        x = task.generate_example()
        assert x.answer
        assert task.score_answer(x.answer, x) == 1.0


def test_junk_scores_zero():
    random.seed(3)
    task = TemporalPerspective()
    x = task.generate_example()
    assert task.score_answer("", x) < 1.0
    assert task.score_answer("garbage", x) < 1.0


def test_metadata_json_serializable():
    import json

    random.seed(11)
    task = TemporalPerspective()
    x = task.generate_example()
    json.dumps(dict(x.metadata))


def test_prompt_determines_answer():
    random.seed(5)
    task = TemporalPerspective()
    x1 = task.generate_example()
    x2 = task.generate_example()
    assert task.render_prompt(x1.metadata) != task.render_prompt(x2.metadata)
