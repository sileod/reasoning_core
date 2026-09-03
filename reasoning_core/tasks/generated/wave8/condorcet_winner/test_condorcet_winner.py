import random

from reasoning_core.tasks.generated.wave8.condorcet_winner.condorcet_winner import (
    CondorcetWinner,
)


def _brute_force(ballots, n):
    half = len(ballots) / 2.0
    wins = 0
    for a in range(n):
        ok = True
        for b in range(n):
            if a == b:
                continue
            ahead = sum(
                1 for o in ballots if o.index(a) < o.index(b)
            )
            if not (ahead > half):
                ok = False
                break
        if ok:
            wins += 1
            winner = a
    if wins == 1:
        return winner
    return None


def test_gold_scores_one():
    random.seed(259485130)
    task = CondorcetWinner()
    for level in (0, 1, 2, 3, 6):
        task.config.set_level(level)
        for _ in range(20):
            e = task.generate_example()
            assert task.score_answer(e.answer, e) == 1.0


def test_matches_bruteforce():
    random.seed(12345)
    task = CondorcetWinner()
    for level in (0, 2, 5):
        task.config.set_level(level)
        for _ in range(20):
            e = task.generate_example()
            n = e.metadata.n_candidates
            bf = _brute_force(e.metadata.ballots, n)
            gold = None if e.answer == "None" else int(e.answer)
            assert bf == gold


def test_distractor_scores_low():
    random.seed(777)
    task = CondorcetWinner()
    for level in (0, 3, 6):
        task.config.set_level(level)
        e = task.generate_example()
        assert task.score_answer("", e) == 0.0
        assert task.score_answer("garbage", e) == 0.0
