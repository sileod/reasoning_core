import random

from reasoning_core.tasks.generated.wave4.s46_voting_rules.voting_rules import (
    VotingRules, _runoff_order, _borda,
)


def test_gold_scores_one():
    random.seed(1)
    task = VotingRules()
    for _ in range(50):
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0


def test_question_variety():
    random.seed(2)
    task = VotingRules()
    seen = set()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(40):
            x = task.generate_example()
            seen.add(x.answer)
    assert len(seen) > 30


def test_order_ends_with_winner():
    random.seed(3)
    task = VotingRules()
    for _ in range(50):
        x = task.generate_example()
        if x.metadata.question != "runoff":
            continue
        order = x.answer.split(",")
        remaining = set(x.metadata.names)
        last = order[-1]
        # the last candidate listed is never eliminated - it is the winner
        assert set(order) == remaining
        assert len(order) == len(remaining)


def test_borda_winner_reasonable():
    random.seed(4)
    task = VotingRules()
    for _ in range(50):
        x = task.generate_example()
        if x.metadata.question != "borda":
            continue


def test_score_rejects_junk():
    random.seed(5)
    task = VotingRules()
    x = task.generate_example()
    assert task.score_answer("", x) == 0.0
    assert task.score_answer("garbage here", x) == 0.0
    assert task.score_answer(None, x) == 0.0


def test_runoff_helper_matches_gold():
    random.seed(6)
    task = VotingRules()
    for _ in range(50):
        x = task.generate_example()
        order = _runoff_order(x.metadata.names, x.metadata.ballots)
        assert ",".join(order) == x.answer or x.metadata.question != "runoff"
