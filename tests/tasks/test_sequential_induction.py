import json

from reasoning_core.tasks.sequential_induction import (
    Sequence,
    SequenceConfig,
    SequentialInduction,
    candidate_bank,
    candidate_index,
    identify_online,
    parse_formula,
    poly_to_string,
    rollout_prefix,
)


def test_sequence_parses_and_evaluates_string_recurrence():
    sequence = Sequence("U[n - 1] + 1", initial_elem=[3])

    assert sequence.n_first_elem(5) == [3, 4, 5, 6, 7]


def test_candidate_bank_uses_ast_shortlex_canonical_form():
    polynomial = parse_formula("n + 1", recurrence_depth=0)
    bank = candidate_bank(0, max_cost=3)

    assert bank[candidate_index(0, max_cost=3)[polynomial]].syntax == "(1 + n)"


def test_polynomial_answer_is_compact_and_canonical():
    polynomial = parse_formula("((8 + 9) * n) + -9", recurrence_depth=0)

    assert poly_to_string(polynomial) == "17 * n - 9"


def test_identification_accepts_unique_minimum_cost_polynomial():
    polynomial = parse_formula("n", recurrence_depth=0)

    identification, reason = identify_online(
        polynomial,
        initial_terms=(),
        recurrence_depth=0,
        min_visible=2,
        max_visible=4,
        max_cost=3,
    )

    assert reason == "accepted"
    assert identification.candidate.syntax == "n"
    assert identification.terms == (0, 1)


def test_rollout_retains_prefix_before_explosion():
    polynomial = parse_formula("U[n - 1] * U[n - 1]", recurrence_depth=1)

    assert rollout_prefix(polynomial, (10,), 1, 5, max_digits=3) == (10, 100)


def test_generated_entry_is_json_serializable_and_self_scoring(tmp_path):
    config = SequenceConfig(
        recurrence_depth=0,
        canonical_max_cost=3,
        uniqueness_cache_dir=str(tmp_path),
        max_generation_attempts=100,
    )
    task = SequentialInduction(config)

    entry = task.generate_entry()

    json.dumps(dict(entry.metadata))
    assert task.score_answer(entry.answer, entry) == 1.0
    assert "Initial terms:" not in task.render_prompt(entry.metadata)


def test_scoring_accepts_equivalent_rhs_and_optional_equation_prefix():
    task = SequentialInduction(SequenceConfig(recurrence_depth=0, canonical_max_cost=7))
    entry = type("Entry", (), {
        "answer": "-16 * n + 28",
        "metadata": {"degree of recursion": 0, "canonical max cost": 7},
    })()

    assert task.score_answer("28 - 16*n", entry) == 1.0
    assert task.score_answer("U[n] = 28 - 16*n", entry) == 1.0
    assert task.score_answer("28 - 15*n", entry) == 0.0

    coefficient_entry = type("Entry", (), {
        "answer": "-486 * n",
        "metadata": {"degree of recursion": 0, "canonical max cost": 7},
    })()
    assert task.score_answer("-486n", coefficient_entry) == 1.0
