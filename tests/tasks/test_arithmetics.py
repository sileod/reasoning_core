from decimal import Decimal
from fractions import Fraction

from reasoning_core.template import Problem, edict
from reasoning_core.tasks.arithmetics import (
    Arithmetics,
    ArithmeticsConfig,
    SAFE_FUNCS,
    _canonical_decimal,
    _display_expr,
    _evaluate,
    _semantic_cue_required,
    _semantic_decoy_eligible,
    fill_num,
    relational_proof_core_size,
    MathWordProblem,
    WordProblemMathConfig,
)


def test_arithmetics_number_theory_ops():
    task = Arithmetics()
    expr, value = fill_num(
        "gcd(INT, INT) + lcm(POS, POS) + bit_count(NAT) + "
        "is_prime(NAT) + prime_count(NAT) + num_divisors(POS)"
    )
    problem = Problem({"expr": expr, "cot": task.get_cot(expr)}, str(value))

    assert task.score_answer(problem.answer, problem) == 1
    assert all(
        op in problem.metadata.cot
        for op in ("gcd", "lcm", "bit_count", "is_prime", "prime_count", "num_divisors")
    )


def test_floor_division_cot_formats_integer_result():
    assert Arithmetics().get_cot("5 // 2") == "5 // 2 = 2"


def test_semantics_cue_only_when_answer_can_change():
    task = Arithmetics()

    plain = task.render_prompt(edict(expr="3 + 3", digit_mode="normal", semantic_cue=True))
    divergent = task.render_prompt(edict(expr="0.3 // 0.1", digit_mode="normal"))
    decoy = task.render_prompt(edict(expr="0.5 // 0.25", digit_mode="normal", semantic_cue=True))

    assert "exact arithmetic" not in plain
    assert "Use exact arithmetic." in divergent
    assert "Use exact arithmetic." in decoy
    assert _semantic_cue_required("0.3 // 0.1", 3)
    assert not _semantic_cue_required("0.5 // 0.25", 3)
    assert _semantic_decoy_eligible("0.5 // 0.25")
    assert not _semantic_decoy_eligible("3 + 3")


def test_digit_perturbations_are_limited_to_clean_integers():
    cfg = ArithmeticsConfig(spaced_digits_prob=1, reversed_spaced_digits_prob=0)

    assert _display_expr("12 + 3", "15", cfg) == ("1 2 + 3", "spaced")
    for expr, answer in (("-12 + 3", "-9"), ("1.2 + 3", "4.2"), ("3 + 3", "6")):
        assert _display_expr(expr, answer, cfg) == (expr, "normal")


def test_exact_and_python_modes_share_the_selected_oracle():
    assert _evaluate("0.3 // 0.1", "exact") == 3
    assert _evaluate("0.3 // 0.1", "python") == 2.0

    task = Arithmetics(ArithmeticsConfig(semantics="python"))
    prompt = task.render_prompt(edict(expr="0.3 // 0.1", digit_mode="normal"))
    assert "Use Python floating-point semantics." in prompt
    assert task.get_cot("0.3 // 0.1").endswith("= 2.0")


def test_zero_answers_are_canonical():
    assert _canonical_decimal(Decimal("0.000"), 3) == "0"
    assert _canonical_decimal(Decimal("-0.000"), 3) == "0"


def test_round_is_exact_and_halfway_note_is_contextual():
    task = Arithmetics()

    assert SAFE_FUNCS["round"](Fraction(9_007_199_254_740_997, 2)) == 4_503_599_627_370_498
    tie_prompt = task.render_prompt(edict(expr="round(5 / 2)", digit_mode="normal"))
    non_tie_prompt = task.render_prompt(edict(expr="round(8 / 3)", digit_mode="normal"))
    assert "Round ties" in tie_prompt
    assert "nearest even integer" in tie_prompt
    assert "Round ties" not in non_tie_prompt


def test_prime_functions_use_standard_negative_semantics():
    assert SAFE_FUNCS["is_prime"](-7) == 0
    assert SAFE_FUNCS["prime_count"](-7) == 0


def test_relational_proof_core_is_query_specific():
    names = ["A", "B", "C"]
    relations = [
        ("more", "B", "A", 2, None),
        ("times", "C", "B", 3, None),
    ]

    assert relational_proof_core_size(names, relations, "A", "C", 4) == 2
    assert relational_proof_core_size(names, relations, "B", "C", 6) == 1


def test_relational_generator_can_require_multistep_queries():
    task = MathWordProblem(WordProblemMathConfig(
        relational_p=1, deep_query_p=1,
    ))

    examples = [task.generate_example(max_tokens=0) for _ in range(10)]

    assert all(example.metadata.proof_core_size >= 2 for example in examples)
