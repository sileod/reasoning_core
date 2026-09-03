import random

from reasoning_core.tasks.generated.wave8.raft_vote_eligibility.raft_vote_eligibility import (
    RaftVoteEligibility,
    _decide,
)


def test_gold_scores_one():
    task = RaftVoteEligibility()
    for _ in range(200):
        x = task.generate_example()
        assert task.score_answer(x.answer, x) == 1.0


def test_junk_scores_zero():
    task = RaftVoteEligibility()
    x = task.generate_example()
    assert task.score_answer("", x) == 0.0
    assert task.score_answer("banana", x) == 0.0
    assert task.score_answer("grant", x) == 0.0
    assert task.score_answer(None, x) == 0.0


def test_decide_invariants():
    import itertools
    for Tc, Lc, Ic, Vv, Lv, Iv in itertools.product([1, 2, 6], [1, 2, 6], [0, 5], [0, 3, 6], [1, 6], [0, 7]):
        mode, wit = _decide(Tc, Lc, Ic, Vv, Lv, Iv)
        assert isinstance(wit, int) and wit >= 0
        if mode == "grant":
            assert Tc > Vv
            assert Lc > Lv or (Lc == Lv and Ic >= Iv)
        elif mode == "stale":
            assert Tc <= Vv
        else:
            assert Tc > Vv
            assert Lc < Lv or (Lc == Lv and Ic < Iv)


def test_answer_matches_construction():
    task = RaftVoteEligibility()
    for _ in range(200):
        x = task.generate_example()
        m = x.metadata
        dmode, dwit = _decide(
            m["candidate_term"], m["leader_log_term"], m["leader_log_index"],
            m["voter_last_voted"], m["voter_log_term"], m["voter_log_index"],
        )
        assert x.answer == f"{dmode}={dwit}"


def test_mode_balance():
    random.seed(12345)
    task = RaftVoteEligibility()
    counts = {"grant": 0, "stale": 0, "log": 0}
    for _ in range(300):
        x = task.generate_example()
        counts[x.metadata["mode"]] += 1
    total = sum(counts.values())
    for k in counts:
        assert counts[k] / total > 0.15, (counts, k)
