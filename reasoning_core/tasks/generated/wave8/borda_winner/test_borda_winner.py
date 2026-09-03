import random

from reasoning_core.tasks.generated.wave8.borda_winner.borda_winner import (
    BordaWinner,
    BordaWinnerConfig,
    _borda_scores,
)


def test_gold_scores_one():
    t = BordaWinner()
    x = t.generate_example()
    assert t.score_answer(x.answer, x) == 1.0


def test_difficulty_changes():
    c = BordaWinnerConfig()
    base = (c.n_candidates, c.n_voters)
    c.set_level(5)
    assert (c.n_candidates, c.n_voters) != base


def test_lex_tiebreak():
    labels = ["A", "B", "C"]
    ballots = [["A", "B", "C"], ["B", "A", "C"]]
    scores = _borda_scores(labels, ballots)
    best = max(scores.values())
    assert [c for c in labels if scores[c] == best] == ["A", "B"]
    assert min(c for c in labels if scores[c] == best) == "A"


def test_answer_is_a_candidate():
    t = BordaWinner()
    for _ in range(20):
        x = t.generate_example()
        assert x.answer in x.metadata.labels


def test_wrong_answer_not_one():
    t = BordaWinner()
    for _ in range(20):
        x = t.generate_example()
        others = [c for c in x.metadata.labels if c != x.answer]
        if others:
            assert t.score_answer(others[0], x) == 0.0
        assert t.score_answer("", x) == 0.0
        assert t.score_answer("ZZZ", x) == 0.0
