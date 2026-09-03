import random

from reasoning_core.tasks.generated.wave8.candidate_key_minimality.candidate_key_minimality import (
    CandidateKeyMinimality, CandidateKeyMinimalityConfig, compute_closure, LABELS,
)


def test_gold_scores_one():
    task = CandidateKeyMinimality()
    for _ in range(30):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_each_label_representable_at_every_level():
    for level in range(7):
        task = CandidateKeyMinimality()
        task.config.set_level(level)
        seen = set()
        for _ in range(200):
            e = task.generate_example()
            seen.add(e.answer)
        assert seen == set(LABELS), (level, seen)


def test_garbage_scores_zero():
    task = CandidateKeyMinimality()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("42", e) == 0.0
    assert task.score_answer(None, e) == 0.0


def test_closure_reference():
    all_attrs = {"A0", "A1", "A2", "A3"}
    fds = [(frozenset(["A0"]), frozenset(["A1"])),
           (frozenset(["A0", "A1"]), frozenset(["A2"]))]
    assert compute_closure(["A0"], fds, all_attrs) == {"A0", "A1", "A2"}
    assert compute_closure(["A3"], fds, all_attrs) == {"A3"}
    assert compute_closure(["A0", "A3"], fds, all_attrs) == {"A0", "A1", "A2", "A3"}


def test_candidate_key_minimality_construction():
    task = CandidateKeyMinimality()
    task.config.set_level(3)
    n = int(task.config.n_attrs)
    attrs = ["A%d" % i for i in range(n)]
    r = random.randint(2, (n - 1) // 2)
    d = n - r
    key_attrs = attrs[:r]
    dep_attrs = attrs[r:]
    fds = [(frozenset([key_attrs[j % r]]), frozenset([dep_attrs[j]])) for j in range(d)]
    all_set = set(attrs)
    assert compute_closure(key_attrs, fds, all_set) == all_set
    for t in key_attrs:
        assert compute_closure([a for a in key_attrs if a != t], fds, all_set) != all_set


def test_difficulty_changes_config():
    task = CandidateKeyMinimality()
    base = int(task.config.n_attrs)
    task.config.set_level(5)
    assert int(task.config.n_attrs) > base
