import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from reasoning_core.tasks.generated.wave8.instant_runoff_winner.instant_runoff_winner import (
    InstantRunoffWinner,
    _irv_winner,
    _irv_winner2,
)


def test_generate_and_score():
    random.seed(1501836147)
    task = InstantRunoffWinner()
    for level in (0, 1, 2, 3, 4, 5, 6):
        task.config.set_level(level)
        ex = task.generate_example()
        assert ex.prompt
        assert task.score_answer(ex.answer, ex) == 1.0
        assert task.score_answer("", ex) < 1.0
        assert task.score_answer("!@#$", ex) < 1.0


def test_consistent_across_levels():
    random.seed(12345)
    task = InstantRunoffWinner()
    for level in (0, 2, 5, 6):
        task.config.set_level(level)
        seen = set()
        for _ in range(20):
            ex = task.generate_example()
            seen.add(ex.answer)
        assert len(seen) > 1


def test_reference_matches():
    random.seed(99)
    task = InstantRunoffWinner()
    task.config.set_level(0)
    for _ in range(50):
        ex = task.generate_example()
        ballots = [
            [ord(ch) - ord("A") for ch in row]
            for row in ex.metadata.ballots
        ]
        n_c = ex.metadata.n_candidates
        gold = ord(ex.answer) - ord("A")
        assert _irv_winner(ballots, n_c) == gold
        assert _irv_winner2(ballots, n_c) == gold


def test_within_irv_domain():
    random.seed(7)
    task = InstantRunoffWinner()
    for level in range(7):
        task.config.set_level(level)
        ex = task.generate_example()
        winner = ord(ex.answer) - ord("A")
        assert 0 <= winner < int(ex.metadata.n_candidates)
