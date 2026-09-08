import random

import pytest

from reasoning_core.tasks._csp_utils import (
    AllDifferent, AssignmentRenderer, Eq, EqVar, Exactly, Implies, Linear, Lt,
    Mod, Ne, NeVar, Or, Var, Xor, CSPSolver, Query, analyze, minimize,
    UniqueValue, minimize_for_objective, minimize_system, possibility_metrics,
    consistency_metrics, enumeration_metrics, relation_metrics,
    semantic_consistency_pair, split_key, holds,
)


def test_ir_canonicalizes_commutative_and_linear_forms():
    x, y = Var("x", range(3)), Var("y", range(3))
    assert EqVar(x, y).canonical() == EqVar(y, x).canonical()
    assert Or((Eq(x, 1), Ne(y, 0))).canonical() == Or((Ne(y, 0), Eq(x, 1))).canonical()
    assert Linear((-2, -4), (x, y), "<=", -6).canonical() == Linear((1, 2), (x, y), ">=", 3).canonical()
    assert Mod(Linear((1,), (x,), "==", 0), 3, 1).canonical() == Mod(Linear((1,), (x,), "!=", 2), 3, 1).canonical()


def test_compositional_formulas_compile_and_solve():
    x, y = Var("x", range(3)), Var("y", range(3))
    formulas = [
        Or((Eq(x, 1), Eq(y, 1))), Xor((Eq(x, 1), Eq(y, 1))),
        Implies(Eq(x, 1), Ne(y, 1)), Exactly(1, (Eq(x, 1), Eq(y, 1))), Ne(y, 1),
    ]
    solver = CSPSolver((x, y), clues=formulas)
    assert solver.unique_value(x) == 1
    assert solver.unique_value(y) != 1


def test_exact_one_xor_preserves_duplicates_and_nesting():
    x, y, z = (Var(name, range(2), "bool") for name in "xyz")
    a, b, c = Eq(x, 1), Eq(y, 1), Eq(z, 1)
    assert not CSPSolver((x,)).is_sat([Xor((a, a))])
    assert holds(Xor((Xor((a, b)), c)), {x: 1, y: 1, z: 1})
    assert not holds(Xor((a, b, c)), {x: 1, y: 1, z: 1})


def test_alldifferent_rejects_duplicate_variables():
    x = Var("x", range(2))
    with pytest.raises(ValueError, match="must be distinct"):
        AllDifferent((x, x))


def test_assignment_renderer_recurses_with_named_owners():
    pet, drink = Var("pet_cat", range(3)), Var("drink_tea", range(3))
    renderer = AssignmentRenderer(
        {"pet_cat": "cat (pet)", "drink_tea": "tea (drink)"},
        ["Alice", "Bruno", "Clara"],
    )
    text = renderer.render(Or((Eq(pet, 0), Ne(drink, 2))))
    assert "Alice" in text and "Clara" in text
    assert " = 0" not in text and "!= 2" not in text


def test_query_aware_minimization_and_metrics_are_semantic():
    x, y, z = (Var(name, range(3)) for name in "xyz")
    base = [AllDifferent((x, y, z))]
    pool = [Ne(x, 0), Ne(x, 2), Eq(y, 0), Eq(z, 2), Ne(y, 2)]
    query = Query("scalar", x, 1, "What is x?")
    selected = minimize(CSPSolver((x, y, z), base), pool, query, random.Random(4), 4)
    assert selected is not None
    assert CSPSolver((x, y, z), base).unique_value(x, selected.clues) == 1
    assert selected.metrics["query_domain_after"] == [1]
    assert selected.metrics["wrong_answer_core_sizes"]


def test_lexicographic_solution_is_not_solver_model_order():
    x, y = Var("x", range(3)), Var("y", range(3))
    solver = CSPSolver((x, y), clues=[Ne(x, y.domain[-1]), Ne(y, 0)])
    assert solver.lex_solution() == (0, 1)


def test_structural_split_key_ignores_formula_argument_order():
    x, y = Var("x", range(2)), Var("y", range(2))
    a = split_key("graph", [AllDifferent((x, y))], [EqVar(x, y)], "scalar")
    b = split_key("graph", [AllDifferent((y, x))], [EqVar(y, x)], "scalar")
    assert a == b


def test_structural_split_key_preserves_directed_roles_and_constants():
    x, y = Var("x", range(3)), Var("y", range(3))
    assert split_key("numeric", [], [Eq(x, 0), Lt(x, y)], "scalar") != split_key(
        "numeric", [], [Eq(x, 0), Lt(y, x)], "scalar"
    )
    assert split_key("numeric", [], [Linear((1, 2), (x, y), "==", 2)], "scalar") != split_key(
        "numeric", [], [Linear((1, 3), (x, y), "==", 2)], "scalar"
    )


def test_structural_split_key_is_invariant_to_nonlexical_renaming():
    x0, x1 = Var("x0", range(3)), Var("x1", range(3))
    z1, z0 = Var("z1", range(3)), Var("z0", range(3))
    assert split_key("numeric", [], [Eq(x0, 0), Lt(x0, x1)], "scalar") == split_key(
        "numeric", [], [Eq(z1, 0), Lt(z1, z0)], "scalar"
    )


def test_system_minimizer_uses_every_requested_order():
    x, y = Var("x", range(2)), Var("y", range(2))
    systems = minimize_system(CSPSolver((x, y)), [Eq(x, 0), Eq(y, 1)], random.Random(2), 5)
    assert len(systems) == 5


def test_unique_value_objective_allows_multiple_full_solutions():
    x, y, z, free = (Var(name, range(2)) for name in ("x", "y", "z", "free"))
    solver = CSPSolver((x, y, z, free))
    systems = minimize_for_objective(
        solver, [EqVar(x, y), EqVar(y, z), Eq(z, 0)],
        UniqueValue(x, 0), random.Random(3), 2,
    )
    assert systems and solver.unique_value(x, systems[0]) == 0
    assert not solver.full_unique(systems[0])
    metrics = possibility_metrics(solver, systems[0], free, 1)
    assert metrics["possibility_expected"] and metrics["possible_value_count"] == 2


def test_final_objective_metric_schemas_have_objective_specific_essentiality():
    x, y = Var("x", range(2)), Var("y", range(2))
    solver = CSPSolver((x, y))
    sat = consistency_metrics(solver, [Eq(x, 0)])
    assert sat["schema"] == "consistency"
    assert sat["displayed_clue_essentiality"] == 0
    assert "query_domain_after" not in sat

    possibility = possibility_metrics(solver, [Eq(x, 0)], x, 1)
    assert possibility["schema"] == "possibility"
    assert possibility["essential_for_objective"] == [True]
    assert "wrong_answer_cores" not in possibility

    relation = relation_metrics(solver, [Eq(x, 0), Eq(y, 0)], EqVar(x, y), True)
    assert relation["schema"] == "relation"
    assert relation["essential_for_objective"] == [True, True]

    enumeration = enumeration_metrics(solver, [Eq(x, 0)], "lexicographic_solution")
    assert enumeration["schema"] == "enumeration"
    assert enumeration["enumeration_mode"] == "lexicographic_solution"


def test_consistency_pair_matches_surface_and_is_balanced():
    x, y, z, free = (Var(name, range(2)) for name in ("x", "y", "z", "free"))
    solver = CSPSolver((x, y, z, free))
    clues = [EqVar(x, y), EqVar(y, z), Eq(z, 0)]
    sat_clues, unsat_clues = semantic_consistency_pair(
        solver, clues, [*clues, Ne(x, 1), NeVar(x, free)], [NeVar(x, z)], random.Random(1),
    )
    assert sat_clues and unsat_clues
    assert len(sat_clues) == len(unsat_clues)
    assert solver.is_sat(sat_clues) and not solver.is_sat(unsat_clues)
