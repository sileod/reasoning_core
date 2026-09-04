import random
from math import gcd

from reasoning_core.tasks.generated.wave9.markov_chain_distribution.markov_chain_distribution import (
    MarkovChainDistribution,
    _parse_prob,
)


def _reduce(num, den):
    g = gcd(num, den)
    return num // g, den // g


def test_generate_example_levels():
    task = MarkovChainDistribution()
    for level in (0, 2, 5, 6):
        task.config.set_level(level)
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_version():
    assert MarkovChainDistribution.task_version == 2


def test_difficulty_changes():
    task = MarkovChainDistribution()
    task.config.set_level(0)
    base_states = task.config.n_states
    task.config.set_level(3)
    hi_states = task.config.n_states
    assert hi_states >= base_states


def test_contract_gold():
    task = MarkovChainDistribution()
    for _ in range(40):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0
        assert task.score_answer("", ex) < 1.0
        assert task.score_answer("garbage", ex) < 1.0


def test_domain():
    random.seed(42)
    task = MarkovChainDistribution()
    task.config.set_level(3)
    ex = task.generate_example()
    p = _parse_prob(ex.answer)
    assert 0.0 <= p <= 1.0


def test_matrix_rows_sum_to_one():
    random.seed(7)
    task = MarkovChainDistribution()
    task.config.set_level(5)
    ex = task.generate_example()
    for r in ex.metadata["matrix"]:
        assert abs(sum(r) - 1.0) < 1e-9
