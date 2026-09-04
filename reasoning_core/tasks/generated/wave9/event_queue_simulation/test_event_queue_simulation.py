import random
from reasoning_core.tasks.generated.wave9.event_queue_simulation.event_queue_simulation import EventQueueSimulation


def _solve(arrivals, n_procs):
    jobs = {}
    for idx, (t, p, w, pr) in enumerate(arrivals):
        jobs.setdefault(p, []).append((t, w, pr, idx))
    total = 0
    for p, plist in jobs.items():
        order = sorted(plist, key=lambda j: (j[2], j[3]))
        free = 0
        for (t, w, pr, i) in order:
            free = max(free, t) + w
        total = max(total, free)
    return total


def test_gold_scores_one():
    random.seed(0)
    task = EventQueueSimulation()
    for _ in range(50):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_answer_matches_independent_solver():
    random.seed(1)
    task = EventQueueSimulation()
    for _ in range(50):
        e = task.generate_example()
        arrivals = e.metadata.arrivals
        expect = _solve(arrivals, e.metadata.n_procs)
        assert int(e.answer) == expect


def test_garbage_rejected():
    random.seed(2)
    task = EventQueueSimulation()
    e = task.generate_example()
    for junk in ["", "abc", "-5", "1.5", "None"]:
        assert task.score_answer(junk, e) < 1.0


def test_domains():
    random.seed(3)
    task = EventQueueSimulation()
    for _ in range(50):
        e = task.generate_example()
        val = int(e.answer)
        assert val >= 0
        for (t, p, w, pr) in e.metadata.arrivals:
            assert 0 <= p < e.metadata.n_procs
            assert w >= 1
            assert pr >= 1
