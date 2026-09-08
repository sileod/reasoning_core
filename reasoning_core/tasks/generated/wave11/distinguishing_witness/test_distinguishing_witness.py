import random

from reasoning_core.tasks.generated.wave11.distinguishing_witness.distinguishing_witness import (
    DistinguishingWitness,
    DistinguishingWitnessConfig,
    smallest_witness,
    parse_indices,
)


def test_generate_and_score():
    random.seed(12345)
    task = DistinguishingWitness()
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_config_diff():
    c = DistinguishingWitnessConfig()
    c.set_level(0)
    base = c.n_items
    c.set_level(6)
    assert c.n_items > base


def test_wrong_answers():
    random.seed(99)
    task = DistinguishingWitness()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("gibberish", ex) == 0.0
    assert task.score_answer("item 1, item 2, item 3, item 4, item 5, item 6", ex) in (
        0.0,
        1.0,
    )


def test_every_level_generates():
    random.seed(7)
    task = DistinguishingWitness()
    for lvl in range(7):
        task.config.set_level(lvl)
        task.generate_example()


def test_lowest_witness_property():
    random.seed(5)
    task = DistinguishingWitness()
    for _ in range(30):
        ex = task.generate_example()
        w = ex.metadata.weights
        C = ex.metadata.capacity
        D = ex.metadata.demand
        gold, card = smallest_witness(list(w), C, D)
        assert list(gold) == ex.metadata.gold_indices
