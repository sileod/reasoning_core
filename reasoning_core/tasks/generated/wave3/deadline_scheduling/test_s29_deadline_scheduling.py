import random

from reasoning_core.tasks.generated.wave3.s29_deadline_scheduling.s29_deadline_scheduling import (
    DeadlineScheduling,
    _moore_hodgson,
)


def _brute(times, deadlines):
    n = len(times)
    best = 0
    for mask in range(1 << n):
        chosen = [i for i in range(n) if mask >> i & 1]
        if not chosen:
            continue
        order = sorted(chosen, key=lambda i: deadlines[i])
        t = 0
        ok = True
        for i in order:
            t += times[i]
            if t > deadlines[i]:
                ok = False
                break
        if ok:
            best = max(best, len(chosen))
    return best


def test_solver_matches_bruteforce():
    random.seed(7)
    for _ in range(200):
        n = random.randint(1, 7)
        times = [random.randint(1, 8) for _ in range(n)]
        deadlines = [random.randint(1, 20) for _ in range(n)]
        jobs = sorted((t, d) for t, d in zip(times, deadlines))
        assert _moore_hodgson(jobs) == _brute(times, deadlines)


def test_gold_scores_one():
    task = DeadlineScheduling()
    e = task.generate_example()
    assert task.score_answer(e.answer, e) == 1.0


def test_answer_domain():
    task = DeadlineScheduling()
    for _ in range(30):
        e = task.generate_example()
        n = e.metadata.n_jobs
        ans = int(e.answer)
        assert 0 <= ans <= n


def test_junk_scores_zero():
    task = DeadlineScheduling()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("abc", e) == 0.0
    assert task.score_answer(str(int(e.answer) + 1), e) == 0.0
