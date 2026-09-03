import random

from reasoning_core.tasks.generated.wave8.two_phase_lock_blocker.two_phase_lock_blocker import (
    TwoPhaseLockBlocker, compute_answer, generate_sequence,
)


def test_gold_scores_one():
    task = TwoPhaseLockBlocker()
    for _ in range(50):
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0


def test_answer_matches_compute():
    for level in (0, 2, 5):
        task = TwoPhaseLockBlocker()
        task.config.set_level(level)
        for _ in range(30):
            entry = task.generate_example()
            seq = entry.metadata.sequence
            txns = entry.metadata.txns
            ans = compute_answer(seq, txns, None)
            expected = "None" if ans is None else ans
            assert entry.answer == expected


def test_mix_of_answers():
    task = TwoPhaseLockBlocker()
    answers = set()
    for level in (0, 2, 5):
        task.config.set_level(level)
        for _ in range(40):
            entry = task.generate_example()
            answers.add(entry.answer)
    assert len(answers) >= 3


def test_none_answer():
    # force a sequence where final request is unblocked
    task = TwoPhaseLockBlocker()
    kind = set()
    for _ in range(200):
        entry = task.generate_example()
        kind.add(entry.answer == "None")
    assert True in kind and False in kind


def test_junk_scores_zero():
    task = TwoPhaseLockBlocker()
    entry = task.generate_example()
    assert task.score_answer("", entry) == 0.0
    assert task.score_answer("garbage", entry) == 0.0
