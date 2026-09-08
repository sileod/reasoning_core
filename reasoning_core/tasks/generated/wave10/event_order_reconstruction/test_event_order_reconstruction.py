import random

from reasoning_core.tasks.generated.wave10.event_order_reconstruction.event_order_reconstruction import (
    EventOrderReconstruction,
    EventOrderConfig,
)


def test_gold_scores_one():
    random.seed(1)
    task = EventOrderReconstruction()
    for _ in range(50):
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0


def test_junk_scores_zero():
    random.seed(2)
    task = EventOrderReconstruction()
    for _ in range(50):
        x = task.generate_example()
        assert task.score_answer("", x) == 0.0
        assert task.score_answer("garbage", x) == 0.0


def test_answer_is_permutation_of_labels():
    random.seed(3)
    task = EventOrderReconstruction()
    for _ in range(50):
        x = task.generate_example()
        labels = set(x.metadata["labels"])
        assert set(x.answer.split(",")) == labels


def test_difficulty_scales():
    cfg = EventOrderConfig()
    assert cfg.n_events == 4
    cfg.set_level(6)
    assert cfg.n_events > 4


def test_chronological_is_valid():
    random.seed(4)
    task = EventOrderReconstruction()
    for _ in range(50):
        x = task.generate_example()
        seq = x.metadata["chronological_seq"]
        assert sorted(seq) == list(range(len(seq)))
