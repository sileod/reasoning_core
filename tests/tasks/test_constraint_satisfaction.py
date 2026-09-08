import random

import pytest

from reasoning_core.tasks.constraint_satisfaction import ConstraintSatisfaction, ConstraintSatisfactionConfig
from reasoning_core.tasks._csp_utils import CSPSolver, Eq, Ne, Var


def test_constraint_satisfaction_modes_generate_and_score():
    random.seed(1)
    for mode in ("attribute", "grid", "linear"):
        task = ConstraintSatisfaction(ConstraintSatisfactionConfig(model_mode=mode, n_vars=3, n_constraints=4))
        problem = task.generate_example(max_tokens=0)
        assert problem.metadata.model_mode == mode
        assert task.score_answer(problem.answer, problem) == 1


def test_possible_values_uses_exact_non_mutating_checks():
    x = Var("x", range(4))
    solver = CSPSolver((x,))

    assert solver.possible_values(x, (Ne(x, 1), Ne(x, 3))) == [0, 2]
    assert solver.unique_value(x, (Eq(x, 2),)) == 2
    assert solver.is_sat((Eq(x, 0),))


def test_scheduling_depth_counts_relations_and_global_invariant():
    random.seed(0)
    config = ConstraintSatisfactionConfig(
        model_mode="scheduling", counterfactual_prob=0,
        possibility_prob=0, consistency_prob=0,
    )
    config.set_level(2)
    problem = ConstraintSatisfaction(config).generate_entry()
    metrics = problem.metadata.metrics

    assert metrics.sampled_min_wrong_answer_core_size >= 2
    assert metrics.global_invariant_essential
    assert not metrics.single_clue_forces_query
    assert metrics.displayed_clue_essentiality >= 0.8


def test_constraint_satisfaction_finite_prompts_are_short():
    random.seed(2)
    for mode in ("attribute", "grid"):
        task = ConstraintSatisfaction(ConstraintSatisfactionConfig(
            model_mode=mode, n_vars=3, n_constraints=4,
            possibility_prob=0, relation_prob=0, consistency_prob=0,
        ))
        problem = task.generate_example(max_tokens=0)
        assert 3 <= len(problem.metadata.clues)
        assert "Answer with one" in problem.prompt


def test_grid_relational_clues_are_required_and_query_line_has_no_facts():
    random.seed(7)
    problem = ConstraintSatisfaction(ConstraintSatisfactionConfig(model_mode="grid", n_vars=3)).generate_entry()
    qr, qc = problem.metadata.query
    assert problem.metadata.relational_clues_required
    assert any(" < " in clue or " > " in clue for clue in problem.metadata.clues)
    assert not problem.metadata.metrics.single_clue_forces_query
    assert problem.metadata.metrics.maximum_single_clue_reduction <= 1


def test_linear_unsat_has_a_nontrivial_semantic_core():
    for seed in range(5):
        config = ConstraintSatisfactionConfig(
            seed=seed,
            model_mode="linear",
            n_vars=3,
            n_constraints=6,
            unsat_prob=1.0,
            max_tries=256,
        )
        random.seed(seed)
        problem = ConstraintSatisfaction(config).generate_entry()

        assert problem.answer == "UNSAT"
        assert problem.metadata.metrics.consistency_core_size >= 3
        operators = problem.metadata.metrics.operator_histogram
        assert not ("eq" in operators and "ne" in operators)


def test_consistency_questions_include_sat_with_multiple_models():
    random.seed(21)
    problem = ConstraintSatisfaction(ConstraintSatisfactionConfig(
        model_mode="linear", consistency_prob=1, unsat_given_consistency=0,
        unsat_prob=0, max_tries=128,
    )).generate_entry()
    assert problem.metadata.query_type == "consistency"
    assert problem.answer == "SAT"
    assert problem.metadata.metrics.multiple_full_solutions
    assert problem.metadata.metrics.schema == "consistency"
    assert problem.metadata.metrics.displayed_clue_essentiality == 0
    assert "query_domain_after" not in problem.metadata.metrics


def test_prompts_have_block_separators_and_consistent_payload():
    random.seed(12)
    problem = ConstraintSatisfaction(ConstraintSatisfactionConfig(
        model_mode="attribute", unsat_prob=0, possibility_prob=0,
    )).generate_entry()
    assert "\n\nConstraints:\n1. " in problem.metadata.prompt
    assert "\n\nQuestion: " in problem.metadata.prompt
    assert problem.metadata.payload["instance"] == problem.metadata.prompt.rsplit("\n\nQuestion:", 1)[0]


def test_lex_all_is_normalized_and_scored_as_enumeration():
    random.seed(3)
    task = ConstraintSatisfaction(ConstraintSatisfactionConfig(
        model_mode="linear", solve_mode="lex_all", unsat_prob=0, max_solutions=256,
    ))
    problem = task.generate_entry()
    assert problem.metadata.solve_mode == "all"
    assert problem.metadata.metrics.schema == "enumeration"
    assert problem.metadata.metrics.enumeration_mode == "all_solutions"
    assert "wrong_answer_cores" not in problem.metadata.metrics
    assert task.score_answer(problem.answer, problem) == 1


def test_all_overflow_regenerates_instead_of_changing_objective():
    random.seed(4)
    task = ConstraintSatisfaction(ConstraintSatisfactionConfig(
        model_mode="linear", solve_mode="all", max_solutions=1, max_tries=256,
    ))
    problem = task.generate_entry()
    assert problem.metadata.solve_mode == "all"


@pytest.mark.parametrize("field,value", [
    ("solve_mode", "other"), ("model_mode", "other"), ("coef_bound", 0),
    ("n_constraints", 0), ("max_solutions", 0), ("grid_width", -1), ("unsat_prob", 2),
])
def test_invalid_config_is_rejected(field, value):
    config = ConstraintSatisfactionConfig(**{field: value})
    with pytest.raises(ValueError):
        ConstraintSatisfaction(config)
