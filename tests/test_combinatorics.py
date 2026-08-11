import pytest

from reasoning_core.tasks.combinatorics import (
    Arrange,
    ChoiceRule,
    CombinatoricsConfig,
    CombinatoricsFormulaSelection,
    Distribute,
    ExactSymbolStrings,
    ExclusiveCommittee,
    LatticePath,
    ManagerCommittee,
    ProjectChairCommittee,
    RoleThenCommittee,
    Select,
    ThroughPointPath,
    TwoCheckpointPath,
    UnionCount,
    _compile,
    _evaluate_expression,
    _valid,
)


def test_recommended_stars_and_bars_example():
    compiled = _compile(Distribute(objects=10, boxes=4, require_nonempty=False))
    correct = next(option for option in compiled.options if option.correct)

    assert correct.expression == "C(13,3)"
    assert {option.expression for option in compiled.options} == {
        "C(13,3)",
        "C(9,3)",
        "C(14,3)",
        "4^10",
    }
    assert _valid(compiled)


def test_atomic_program_families_have_unique_options():
    programs = (
        Select(8, 3, ordered=False, replacement=False),
        Select(8, 3, ordered=True, replacement=True),
        Distribute(10, 4, require_nonempty=True),
        Arrange(7, adjacent_pair=True),
        Arrange(7, circular=True),
        UnionCount(18, 15, 7),
        ChoiceRule("sum", 5, 6),
        ChoiceRule("product", 5, 6),
        ChoiceRule("complement", 5, 3),
        LatticePath(3, 4),
    )

    assert all(_valid(_compile(program)) for program in programs)


def test_composed_programs_have_depth_two_and_unique_options():
    programs = (
        RoleThenCommittee(10, 3),
        ExactSymbolStrings(8, 5, 3),
        ManagerCommittee(12, 3, 4),
        ThroughPointPath(7, 5, 3, 2),
        ExclusiveCommittee(12, 4),
    )

    compiled = [_compile(program) for program in programs]
    assert all(problem.depth == 2 for problem in compiled)
    assert all(_valid(problem) for problem in compiled)
    assert [next(o.expression for o in problem.options if o.correct) for problem in compiled] == [
        "10*C(9,3)",
        "C(8,3)*4^5",
        "C(12,4)-C(9,4)",
        "C(5,3)*C(7,4)",
        "C(10,3)+C(10,3)",
    ]


def test_depth_three_programs_have_semantic_unique_options():
    programs = (
        ProjectChairCommittee(3, 12, 4),
        TwoCheckpointPath(9, 8, 2, 2, 6, 5),
    )
    compiled = [_compile(program) for program in programs]

    assert all(problem.depth == 3 for problem in compiled)
    assert all(_valid(problem) for problem in compiled)


def test_unknown_choice_rule_fails_closed():
    with pytest.raises(ValueError, match="unknown choice rule"):
        _compile(ChoiceRule("typo", 5, 3))


def test_difficulty_increases_composition_and_reduces_explicitness():
    config = CombinatoricsConfig()
    config.set_level(5)

    assert config.depth_2_rate > CombinatoricsConfig().depth_2_rate
    assert config.depth_3_rate > CombinatoricsConfig().depth_3_rate
    assert config.explicit_rate < CombinatoricsConfig().explicit_rate


def test_generated_metadata_and_answer():
    task = CombinatoricsFormulaSelection()

    for _ in range(50):
        example = task.generate_example()
        assert example.answer == example.metadata.correct_expression
        assert task.score_answer(example.answer, example) == 1
        assert task.score_answer("not a formula", example) == 0
        assert example.metadata.structural_depth in (1, 2, 3)
        assert example.metadata.correct_option_label == "ABCD"[example.metadata.correct_option_index]
        assert example.metadata.options[example.metadata.correct_option_index]["expression"] == example.answer
        assert example.metadata.correct_features.top_operator
        assert all(option["semantics"] for option in example.metadata.options)
        assert example.metadata.n_slots >= 2
        assert example.metadata.n_candidates <= 32
        assert example.metadata.candidate_expressions.count(example.answer) == 1
        assert example.metadata.candidate_values.count(
            str(_evaluate_expression(example.answer))
        ) == 1
        assert "Options:" not in example.prompt
        assert "The answer must have the form:" in example.prompt
