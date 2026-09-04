import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import random

from reasoning_core.tasks.generated.wave9.three_dimensional_orientation.three_dimensional_orientation import (
    ThreeDimensionalOrientation,
    ThreeDimensionalOrientationConfig,
)


def test_gold_scores_one():
    random.seed(1)
    task = ThreeDimensionalOrientation()
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_wrong_scores_less_than_one():
    random.seed(2)
    task = ThreeDimensionalOrientation()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("garbage", ex) == 0.0


def test_difficulty_changes_config():
    cfg = ThreeDimensionalOrientationConfig()
    cfg.set_level(0)
    base = (cfg.n_rotations, cfg.n_questions)
    cfg.set_level(5)
    assert (cfg.n_rotations, cfg.n_questions) != base


def test_answers_are_valid_directions():
    random.seed(3)
    task = ThreeDimensionalOrientation()
    valid = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
    for _ in range(20):
        ex = task.generate_example()
        for part in ex.answer.split(" ; "):
            assert part.strip() in valid


def test_metadata_json_serializable():
    import json
    random.seed(4)
    task = ThreeDimensionalOrientation()
    for _ in range(5):
        ex = task.generate_example()
        json.dumps(ex.metadata)


def test_questions_vary():
    random.seed(5)
    task = ThreeDimensionalOrientation()
    answers = set()
    for _ in range(30):
        ex = task.generate_example()
        answers.add(ex.answer)
    assert len(answers) > 5
