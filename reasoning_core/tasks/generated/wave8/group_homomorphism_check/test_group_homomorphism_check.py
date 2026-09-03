import random

from reasoning_core.tasks.generated.wave8.group_homomorphism_check.group_homomorphism_check import (
    GroupHomomorphismCheck,
    _parse_int,
    _violation_count,
)


def test_gold_scores_one():
    random.seed(12345)
    task = GroupHomomorphismCheck()
    for _ in range(50):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_count_in_range_and_semantics():
    random.seed(999)
    task = GroupHomomorphismCheck()
    for _ in range(30):
        ex = task.generate_example()
        c = _parse_int(ex.answer)
        n = len(ex.metadata.payload["domain_table"])
        assert 0 <= c <= n * n
        brute = _violation_count(
            ex.metadata.payload["mapping"],
            ex.metadata.payload["domain_table"],
            ex.metadata.payload["codomain_table"],
        )
        assert c == brute


def test_junk_and_empty_not_correct():
    random.seed(5)
    task = GroupHomomorphismCheck()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("not a number", ex) == 0.0
    assert task.score_answer("1000000", ex) == 0.0
