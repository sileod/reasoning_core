import sympy as sp

from reasoning_core.tasks.equation_system import EquationSystem, EquationSystemCfg, _verify_system


def test_shifted_combination_is_inconsistent_without_duplicate_left_side():
    task = EquationSystem(EquationSystemCfg(num_vars=3))
    equations, variables, _ = task._generate_base_system()
    inconsistent = equations + [task._combined_equation(equations, shift=1)]

    assert _verify_system(inconsistent, variables)["kind"] == "inconsistent"
    assert len({sp.expand(eq.lhs) for eq in inconsistent}) == len(inconsistent)


def test_cot_is_suppressed_for_underdetermined_system():
    task = EquationSystem()
    x, y = sp.symbols("x y")

    assert task.get_cot([sp.Eq(x + y, 1)], [x, y]) is None


def test_cot_assignment_is_independently_validated():
    task = EquationSystem()
    x, y = sp.symbols("x y")
    equations = [sp.Eq(x + y, 3), sp.Eq(x - y, 1)]

    cot = task.get_cot(equations, [x, y])

    assert cot is not None
    assert "x = 2" in cot
    assert "y = 1" in cot


def test_generated_non_unique_system_omits_cot():
    task = EquationSystem(EquationSystemCfg(p_inconsistent=0.0, p_underdetermined=1.0))
    entry = task.generate_entry()

    assert entry.metadata.case != "unique"
    assert len(entry.metadata.equations) == task.config.num_vars + 1
    assert "cot" not in entry.metadata
