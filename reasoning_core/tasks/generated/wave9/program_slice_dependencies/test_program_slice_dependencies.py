import random

from reasoning_core.tasks.generated.wave9.program_slice_dependencies.program_slice_dependencies import (
    ProgramSliceDependencies,
    _build_program,
    _compute_slice,
)


def test_validate_all_levels():
    t = ProgramSliceDependencies()
    for level in (0, 2, 5):
        t.config.set_level(level)
        t.validate()


def test_gold_scores_one():
    t = ProgramSliceDependencies()
    for level in (0, 1, 3, 5):
        t.config.set_level(level)
        for _ in range(30):
            ex = t.generate_example()
            assert t.score_answer(ex.answer, ex) == 1.0


def test_slice_is_minimal_and_reproducible():
    t = ProgramSliceDependencies()
    for level in (0, 5):
        t.config.set_level(level)
        n = int(t.config.n_stmts)
        gp = float(t.config.guard_prob)
        for _ in range(30):
            operands, guards, output, slice_set = _build_program(n, gp)
            recomputed = _compute_slice(n, operands, guards, output)
            assert recomputed == slice_set
            assert 1 <= len(slice_set) <= n - 1
            assert output in slice_set


def test_wrong_answers_score_zero():
    t = ProgramSliceDependencies()
    for level in (0, 5):
        t.config.set_level(level)
        for _ in range(30):
            ex = t.generate_example()
            slice_list = ex.metadata.slice_list
            assert t.score_answer(None, ex) == 0.0
            assert t.score_answer("", ex) == 0.0
            assert t.score_answer("abc", ex) == 0.0
            perm = slice_list[-1:] + slice_list[:-1]
            assert t.score_answer(" ".join(str(x) for x in perm), ex) == 1.0
            superset = sorted(set(slice_list) | {int(ex.metadata.n_stmts) - 1})
            if superset != slice_list:
                assert t.score_answer(" ".join(str(x) for x in superset), ex) == 0.0


def test_answers_vary_across_examples():
    t = ProgramSliceDependencies()
    t.config.set_level(5)
    seen = set()
    for _ in range(50):
        ex = t.generate_example()
        seen.add(ex.answer)
    assert len(seen) >= 10
