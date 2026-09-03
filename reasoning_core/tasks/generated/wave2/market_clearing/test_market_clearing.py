import random

from reasoning_core.tasks.generated.wave2.s23_market_clearing.market_clearing import (
    MarketClearing, MarketClearingConfig, _buyer_surplus, _seller_surplus,
    _qty_at, _clearing_prices,
)


def test_generate_scores_gold():
    random.seed(1)
    task = MarketClearing()
    for _ in range(30):
        x = task.generate_example()
        assert x.answer is not None
        assert task.score_answer(x.answer, x) == 1.0


def test_garbage_does_not_score():
    random.seed(2)
    task = MarketClearing()
    x = task.generate_example()
    assert task.score_answer("", x) < 1.0
    assert task.score_answer("abc", x) < 1.0
    assert task.score_answer("12.5.3", x) < 1.0


def test_difficulty_changes():
    c = MarketClearingConfig().set_level(0)
    n0 = c.n_buyers
    c2 = MarketClearingConfig().set_level(5)
    assert c2.n_buyers > n0


def test_answer_variance():
    random.seed(3)
    task = MarketClearing()
    answers = set()
    for _ in range(40):
        answers.add(task.generate_example().answer)
    assert len(answers) > 5


def test_all_levels_generate():
    random.seed(4)
    for level in range(7):
        task = MarketClearing(config=MarketClearingConfig(), level=level)
        task.generate_example()


def test_surplus_nonnegative():
    random.seed(5)
    task = MarketClearing()
    for _ in range(20):
        x = task.generate_example()
        if x.metadata.task == "surplus_buyer":
            assert int(x.answer) >= 0
        if x.metadata.task == "surplus_seller":
            assert int(x.answer) >= 0
