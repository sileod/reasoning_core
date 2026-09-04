import math

from reasoning_core.tasks.generated.wave9.modular_constraint_solver.modular_constraint_solver import (
    ModularConstraint, _solve_system, _reduce_by_gcd,
)


def _check_solution(x, m, reduced):
    for r, mod in reduced:
        if x % mod != r % mod:
            return False
    return True


def test_gold_scores_one():
    task = ModularConstraint()
    for _ in range(20):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0


def test_consistent_answers_are_solutions():
    task = ModularConstraint()
    count = 0
    for _ in range(50):
        e = task.generate_example()
        if e.metadata.inconsistent:
            assert e.answer == "inconsistent"
        else:
            count += 1
            parts = e.answer.split(" mod ")
            assert len(parts) == 2
            x = int(parts[0])
            m = int(parts[1])
            assert _check_solution(x, m, e.metadata.constraints)
    assert count > 0


def test_inconsistent_answers_really_inconsistent():
    task = ModularConstraint()
    count = 0
    for _ in range(50):
        e = task.generate_example()
        if e.metadata.inconsistent:
            assert e.answer == "inconsistent"
            assert _solve_system(e.metadata.constraints) is None
            count += 1
    assert count > 0


def test_junk_and_empty_score_zero():
    task = ModularConstraint()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("foo", e) == 0.0


def test_answer_domain_valid():
    task = ModularConstraint()
    for _ in range(30):
        e = task.generate_example()
        if not e.metadata.inconsistent:
            parts = e.answer.split(" mod ")
            x = int(parts[0])
            m = int(parts[1])
            assert m >= 1
            assert 0 <= x < m
