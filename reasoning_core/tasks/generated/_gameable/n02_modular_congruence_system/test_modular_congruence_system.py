import random

from reasoning_core.tasks.generated.wave0.n02_modular_congruence_system.modular_congruence_system import (
    ModularCongruenceSystem,
    _consistent_system,
    _inconsistent_system,
)


def test_validate_all_levels():
    t = ModularCongruenceSystem()
    for level in (0, 2, 5):
        t.config.set_level(level)
        t.validate()


def test_gold_scores_one():
    t = ModularCongruenceSystem()
    for level in (0, 1, 3, 5):
        t.config.set_level(level)
        for _ in range(20):
            ex = t.generate_example()
            assert t.score_answer(ex.answer, ex) == 1.0


def test_wrong_answers_score_zero():
    t = ModularCongruenceSystem()
    for level in (0, 5):
        t.config.set_level(level)
        for _ in range(20):
            ex = t.generate_example()
            assert t.score_answer("NONE", ex) == (0.0 if ex.metadata.has_solution else 1.0)
            assert t.score_answer(None, ex) == 0.0

            bad = "NONE"
            if ex.metadata.has_solution:
                bad = str((int(ex.metadata.canonical) + 1) % max(2, int(ex.metadata.lcm)))
            elif int(ex.metadata.lcm) > 1:
                bad = "0"
            s = t.score_answer(bad, ex)
            if ex.metadata.has_solution:
                assert s == 0.0


def test_no_solution_is_really_inconsistent():
    t = ModularCongruenceSystem()
    t.config.set_level(5)
    n = t.config.n_cong
    mm = t.config.max_mod
    for _ in range(30):
        mods, residues = _inconsistent_system(n, mm)
        from sympy.ntheory.modular import solve_congruence
        assert solve_congruence(*zip(residues, mods)) is None


def test_consistent_solutions_are_valid(void=False):
    from sympy.ntheory.modular import solve_congruence
    t = ModularCongruenceSystem()
    for level in (0, 5):
        t.config.set_level(level)
        n = t.config.n_cong
        mm = t.config.max_mod
        for _ in range(30):
            mods, residues, target = _consistent_system(n, mm)
            sol = solve_congruence(*zip(residues, mods))
            assert sol is not None
            for r, m in zip(residues, mods):
                assert int(target % m) == r
