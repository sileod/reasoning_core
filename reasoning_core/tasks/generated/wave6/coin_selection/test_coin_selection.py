import random

from reasoning_core.tasks.generated.wave6.s64_coin_selection.coin_selection import (
    CoinSelection,
    _unbounded_min_count,
    _greedy_count,
    _lex_largest_min_multiset,
    _greedy_fails,
)


def _denoms_usable(denoms):
    return sorted(set(list(denoms) + [1]), reverse=True)


def test_default_gold_scores_1():
    random.seed(1234)
    task = CoinSelection()
    ex = task.generate_example()
    assert task.score_answer(ex.answer, ex) == 1.0


def test_all_levels_generate_and_score():
    for level in range(0, 7):
        random.seed(100 + level)
        task = CoinSelection()
        task.config.set_level(level)
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_greedy_strictly_worse():
    random.seed(7)
    for _ in range(50):
        ex = CoinSelection().generate_example()
        denoms = _denoms_usable(ex.metadata.denominations)
        amount = ex.metadata.amount
        opt = len([int(x) for x in ex.answer.split(",")])
        greedy = _greedy_count(denoms, amount)
        assert greedy is not None and greedy > opt


def test_answer_is_valid_multiset():
    random.seed(11)
    for _ in range(50):
        ex = CoinSelection().generate_example()
        denoms = set(int(x) for x in ex.metadata.denominations)
        coins = [int(x) for x in ex.answer.split(",")]
        assert sum(coins) == ex.metadata.amount
        assert all(c in denoms for c in coins)
        assert coins == sorted(coins, reverse=True)


def test_score_rejects_junk_and_empty():
    random.seed(3)
    ex = CoinSelection().generate_example()
    assert CoinSelection().score_answer("", ex) == 0.0
    assert CoinSelection().score_answer("banana", ex) == 0.0
    assert CoinSelection().score_answer(None, ex) == 0.0
    assert CoinSelection().score_answer("  7, 0, 1  ", ex) == 0.0


def test_lex_largest_helper():
    denoms = _denoms_usable([25, 10, 4, 1])
    amount = 50
    mult = _lex_largest_min_multiset(denoms, amount)
    assert sum(mult) == amount
    assert len(mult) == _unbounded_min_count(denoms, amount)
