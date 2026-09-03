import random

from reasoning_core.tasks.generated.wave8.approval_voting_winner.approval_voting_winner import (
    ApprovalVotingWinner,
    ApprovalVotingWinnerConfig,
    _compute_winner,
    _parse_answer,
)


def test_score_gold():
    random.seed(0)
    task = ApprovalVotingWinner()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_score_junk():
    random.seed(1)
    task = ApprovalVotingWinner()
    x = task.generate_example()
    assert task.score_answer("", x) == 0.0
    assert task.score_answer("zzz", x) == 0.0
    assert task.score_answer("12", x) == 0.0
    assert task.score_answer("import fakemodule", x) == 0.0


def test_score_wrong_letter():
    random.seed(2)
    task = ApprovalVotingWinner()
    x = task.generate_example()
    letter, count = _parse_answer(x.answer)
    wrong = chr(ord('A') + (ord(letter) - ord('A') + 1) % 8)
    assert task.score_answer(f"{wrong} {count}", x) == 0.0


def test_score_wrong_count():
    random.seed(3)
    task = ApprovalVotingWinner()
    x = task.generate_example()
    letter, count = _parse_answer(x.answer)
    assert task.score_answer(f"{letter} {count + 1}", x) == 0.0


def test_score_whitespace_lenient():
    random.seed(4)
    task = ApprovalVotingWinner()
    x = task.generate_example()
    assert task.score_answer("  " + x.answer.strip() + "  ", x) == 1.0


def test_compute_winner_tiebreak():
    candidates = ["A", "B", "C"]
    ballots = [["A", "B"], ["A", "B"], ["C"], ["C"]]
    winner, max_count, counts = _compute_winner(candidates, ballots)
    assert counts == {"A": 2, "B": 2, "C": 2}
    assert max_count == 2
    assert winner == "A"


def test_compute_winner_alphabetical_tie():
    candidates = ["A", "B", "C"]
    ballots = [["B"], ["C"], ["A"]]
    winner, _, _ = _compute_winner(candidates, ballots)
    assert winner == "A"


def test_difficulty_monotonic():
    cfg = ApprovalVotingWinnerConfig()
    vals = []
    for level in range(7):
        c = ApprovalVotingWinnerConfig()
        c.set_level(level)
        vals.append((int(c.n_voters), int(c.n_candidates)))
    voters = [v[0] for v in vals]
    cands = [v[1] for v in vals]
    assert all(b < a for a, b in zip(voters, voters[1:])) or voters == sorted(set(voters))
    assert voters[-1] >= voters[0]
    assert cands[-1] >= cands[0]


def test_winner_nonempty_positive():
    random.seed(7)
    task = ApprovalVotingWinner()
    for _ in range(50):
        x = task.generate_example()
        letter, count = _parse_answer(x.answer)
        assert letter is not None
        assert count >= 1


def test_validate():
    random.seed(8)
    task = ApprovalVotingWinner()
    task.validate()
